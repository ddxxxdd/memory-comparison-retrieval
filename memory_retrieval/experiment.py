"""Run anchored graph retrieval, legacy OCSP-QR, and frozen baselines."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
from statistics import mean, median
import time

import numpy as np

from .api import APIClient, usage_summary
from .baselines import BM25, rank, reciprocal_rank_fusion
from .bipolar_graph import (ANCHORED_METHODS, ANCHORED_PROMPT_SHA256, audit_anchored_probes,
                            build_knn_transition, compute_bipolar_diffusion_scores, extract_anchored_probes)
from .config import Settings
from .data import fingerprint, sha256_file, write_json, write_jsonl
from .encoders import CPU_THREADS, MODEL_LOCK, FrozenEncoder, encode_groups
from .evaluation import grouped_ranking_metrics, ranking_metrics, target_ids
from .ocsp import (ADAPTIVE_MAX_K, ADAPTIVE_MIN_ORTHO_PROJ, ADAPTIVE_MIN_VIOLATION_SIM,
                   ONLINE_ZERO_SHOT_BIPOLAR_PROMPT, TOP_KS, compute_subspace_ocsp_scores,
                   extract_online_bipolar_probes, score_fixed_k, score_variable_candidates)
from .sparsecl import (INITIAL_RETRIEVAL_DEPTH, SPARSECL_SCORE_FORMULA, SPARSECL_UPSTREAM_COMMIT,
                       fused_order, score_candidates)


BASELINES = ("bm25", "dense", "hybrid", "sparsecl_arguana", "sparsecl_msmarco", "sparsecl_hotpotqa")
MAIN_METHOD = "anchored_bipolar_diffusion"
LEGACY_METHOD = "ocsp_qr"
ABLATION = "probes_only_no_ortho"


def prepare_embeddings(dataset, args, methods, output):
    with_probes = bool({LEGACY_METHOD, ABLATION}.intersection(methods))
    texts = {"memories": [m.content.strip() for m in dataset.memories],
             "queries": [q["content"].strip() for q in dataset.queries]}
    info = {"models": {}, "timing_seconds": {}, "usage": usage_summary([])}
    if set(methods) == {"bm25"}:
        return {}, [], info
    probes = []
    if with_probes or args.embedding_backend == "api":
        settings = Settings.from_env(args.env)
        settings.require(llm=with_probes, embedding=args.embedding_backend == "api")
        client = APIClient(replace(settings, cache_path=args.cache or output / "probe_cache.sqlite3"))
    if with_probes:
        start = time.perf_counter()
        probes = extract_online_bipolar_probes(client, dataset.queries)
        info["timing_seconds"]["online_probe_llm_total"] = time.perf_counter() - start
        info["models"]["llm"] = settings.llm_identity()
        for key in ("compatible_paraphrases", "violation_probes"):
            texts[key] = [text for pair in probes for text in pair[key]]
    if args.embedding_backend == "api":
        # Match the batch boundaries of the original API embedding runs.
        start = time.perf_counter()
        all_vecs = client.embed([text for group in texts.values() for text in group], "ocsp_qr_embeddings")
        all_vecs = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
        vectors, offset = {}, 0
        for name, group in texts.items():
            vectors[name] = all_vecs[offset:offset + len(group)]
            offset += len(group)
        info["models"]["embedding"] = settings.embedding_identity()
        info["timing_seconds"]["embedding_total"] = time.perf_counter() - start
    else:
        vectors, identity, elapsed = encode_groups("bge", texts, args.model_root)
        info["models"]["embedding"] = identity
        info["timing_seconds"]["bge"] = elapsed
    if with_probes or args.embedding_backend == "api":
        info["usage"] = usage_summary(client.events)
    for key in ("compatible_paraphrases", "violation_probes"):
        if probes:
            boundaries = np.cumsum([len(pair[key]) for pair in probes[:-1]])
            vectors[key] = np.split(vectors[key], boundaries)
    return vectors, probes, info


def ranked_row(query, docs, name, scores):
    order = rank(scores, docs)
    return {"query_id": query["query_id"], "scenario_id": query.get("scenario_id"),
            "method": name, "library_size": len(docs),
            "ranked_candidates": [{"memory_id": docs[i].memory_id, "score": float(scores[i])} for i in order]}


def score_original_text(dataset, methods, vectors, probes, gamma):
    rows = {name: {} for name in methods if not name.startswith("sparsecl_")}
    details = {}
    for memory_idx, query_idx in dataset.groups().values():
        docs = [dataset.memories[i] for i in memory_idx]
        bm25 = BM25([m.content.strip() for m in docs]) if {"bm25", "hybrid"}.intersection(rows) else None
        bank = vectors["memories"][memory_idx] if vectors else None
        for i in query_idx:
            query = dataset.queries[i]
            qid = query["query_id"]
            scores = {}
            if bm25 is not None:
                scores["bm25"] = bm25.score(query["content"].strip())
            if {"dense", "hybrid"}.intersection(rows) or probes:
                scores["dense"] = bank @ vectors["queries"][i]
            if "hybrid" in rows:
                scores["hybrid"] = reciprocal_rank_fusion(scores["bm25"], scores["dense"], docs)
            if probes:
                ocsp = compute_subspace_ocsp_scores(vectors["queries"][i], vectors["compatible_paraphrases"][i],
                                                    vectors["violation_probes"][i], bank, gamma)
                scores.update(ocsp)
                targets = set(target_ids(dataset.gold[i]))
                dense_order = rank(scores["dense"], docs)[:10]
                non_gold = [j for j in dense_order if docs[j].memory_id not in targets]
                ordered = rank(scores[LEGACY_METHOD], docs)
                adaptive = [docs[j].memory_id for j in ordered
                            if ocsp["best_s_minus"][j] >= ADAPTIVE_MIN_VIOLATION_SIM
                            and ocsp["best_s_perp"][j] >= ADAPTIVE_MIN_ORTHO_PROJ][:ADAPTIVE_MAX_K]
                details[qid] = {"query_id": qid, "scenario_id": query.get("scenario_id"),
                                "query_content": query["content"], **probes[i], "gold_ids": sorted(targets),
                                "gold_s_perp": {m.memory_id: float(ocsp["best_s_perp"][j])
                                                for j, m in enumerate(docs) if m.memory_id in targets},
                                "top_non_gold_s_perp_mean": float(np.mean(ocsp["best_s_perp"][non_gold])) if non_gold else 0.0,
                                "adaptive_candidates": adaptive, "top10_ocsp_qr": [docs[j].memory_id for j in ordered[:10]],
                                "top10_dense": [docs[j].memory_id for j in dense_order]}
            for name in rows:
                rows[name][qid] = ranked_row(query, docs, name, scores[name])
    return rows, details


def sparsecl_pools(dataset, vectors):
    import faiss

    faiss.omp_set_num_threads(CPU_THREADS)
    pools = {}
    for memory_idx, query_idx in dataset.groups().values():
        bank = np.asarray(memory_idx)
        index = faiss.IndexHNSWFlat(vectors["memories"].shape[1], 64)
        index.add(vectors["memories"][bank])
        index.hnsw.efSearch = INITIAL_RETRIEVAL_DEPTH
        k = min(len(bank), INITIAL_RETRIEVAL_DEPTH)
        _, positions = index.search(vectors["queries"][query_idx], k)
        for i, local in zip(query_idx, positions):
            if len(set(local)) != k or np.any(local < 0) or np.any(local >= len(bank)):
                raise ValueError("SparseCL returned an invalid candidate pool")
            pools[i] = (len(bank), bank[local])
    return pools


def score_sparsecl(dataset, methods, vectors, args, info):
    arms = [method for method in methods if method.startswith("sparsecl_")]
    if not arms:
        return {}
    pools = sparsecl_pools(dataset, vectors)
    groups = {"memories": [m.content.strip() for m in dataset.memories],
              "queries": [q["content"].strip() for q in dataset.queries]}
    rows = {}
    for arm in arms:
        name = arm.removeprefix("sparsecl_")
        sparse, identity, seconds = encode_groups(name, groups, args.model_root)
        alpha = MODEL_LOCK[name]["alpha"]
        info["models"][arm] = {**identity, "alpha": alpha}
        info["timing_seconds"][arm] = seconds
        rows[arm] = {}
        for i, query in enumerate(dataset.queries):
            size, ids = pools[i]
            cos, hoyer = score_candidates(vectors["queries"][i], vectors["memories"][ids],
                                         sparse["queries"][i], sparse["memories"][ids])
            order, scores = fused_order(cos, hoyer, alpha)
            candidates = [{"memory_id": dataset.memories[ids[j]].memory_id, "score": float(scores[j]),
                           "cosine": float(cos[j]), "hoyer": float(hoyer[j])} for j in order]
            qid = query["query_id"]
            rows[arm][qid] = {"query_id": qid, "scenario_id": query.get("scenario_id"), "method": arm,
                              "library_size": size, "alpha": alpha, "coarse_k": len(ids), "ranked_candidates": candidates}
    return rows


def score_anchored_text(dataset, methods, vectors, *, k=5, top_seeds=5, alpha=0.2, beta=0.3):
    rows = {name: {} for name in methods if name in ANCHORED_METHODS}
    for memory_idx, query_idx in dataset.groups().values():
        docs = [dataset.memories[i] for i in memory_idx]
        bank = vectors["memories"][memory_idx]
        transition = build_knn_transition(bank, docs, k=k)
        for i in query_idx:
            scores = compute_bipolar_diffusion_scores(
                vectors["queries"][i], vectors["anchored_violation_probes"][i], bank, transition, docs,
                top_seeds=top_seeds, alpha=alpha, beta=beta)
            query = dataset.queries[i]
            for name in rows:
                rows[name][query["query_id"]] = ranked_row(query, docs, name, scores[name])
    return rows


def prepare_anchored_experiment(dataset, args, methods, output):
    """Overlap new LLM requests with BGE and SparseCL, keeping BGE loaded once."""
    if args.embedding_backend != "bge":
        raise ValueError("Anchored graph methods require the frozen BGE encoder")
    settings = Settings.from_env(args.env)
    cache = args.cache or output / "probe_cache.sqlite3"
    anchored_settings = replace(settings, cache_path=cache, llm_model="deepseek-v4.1-flash",
                                llm_temperature=0.0, llm_json_mode=True, llm_reasoning_effort="none")
    anchored_settings.require(llm=True)
    client = APIClient(anchored_settings)
    info = {"models": {"llm": anchored_settings.llm_identity(), "anchored_llm": anchored_settings.llm_identity()},
            "timing_seconds": {}}
    texts = {"memories": [m.content.strip() for m in dataset.memories],
             "queries": [q["content"].strip() for q in dataset.queries]}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_anchored_probes, client, dataset.queries, getattr(args, "probe_workers", 4))
        started = time.perf_counter()
        encoder = FrozenEncoder("bge", args.model_root)
        info["timing_seconds"]["bge_load_verify"] = time.perf_counter() - started
        vectors, identity, seconds = encoder.encode_groups(texts)
        info["models"]["embedding"] = identity
        info["timing_seconds"]["bge"] = seconds
        baseline_methods = [name for name in methods if name in BASELINES]
        rows, _ = score_original_text(dataset, baseline_methods, vectors, [], args.gamma)
        rows.update(score_sparsecl(dataset, baseline_methods, vectors, args, info))
        records, audit = future.result()
        write_json(output / "anchored_probes.json", records)
        encoded, _, seconds = encoder.encode_groups({
            "anchored_violation_probes": [text for row in records for text in row["violation_probes"]]})
        info["timing_seconds"]["bge"].update(seconds)
        vectors["anchored_violation_probes"] = np.split(encoded["anchored_violation_probes"], len(records))
    audit.update(audit_anchored_probes(dataset.queries, records, vectors["queries"], vectors["anchored_violation_probes"]))
    audit["token_audit"] = identity["token_audit"]["anchored_violation_probes"]
    info["anchored_probe_audit"] = audit
    info["timing_seconds"]["anchored_probe_llm_total"] = audit["generation"]["wall_seconds"]
    write_json(output / "anchored_probe_audit.json", audit)
    probes, details, legacy_events = [], {}, []
    legacy_methods = [name for name in methods if name in (LEGACY_METHOD, ABLATION)]
    if legacy_methods:
        legacy_client = APIClient(replace(settings, cache_path=cache))
        started = time.perf_counter()
        probes = extract_online_bipolar_probes(legacy_client, dataset.queries)
        info["timing_seconds"]["online_probe_llm_total"] = time.perf_counter() - started
        info["models"]["llm"] = settings.llm_identity()
        groups = {key: [text for row in probes for text in row[key]]
                  for key in ("compatible_paraphrases", "violation_probes")}
        encoded, _, seconds = encoder.encode_groups(groups)
        info["timing_seconds"]["bge"].update(seconds)
        for key in groups:
            vectors[key] = np.split(encoded[key], np.cumsum([len(row[key]) for row in probes[:-1]]))
        old_rows, details = score_original_text(dataset, legacy_methods, vectors, probes, args.gamma)
        rows.update(old_rows)
        legacy_events = legacy_client.events
    started = time.perf_counter()
    rows.update(score_anchored_text(dataset, methods, vectors, k=getattr(args, "graph_k", 5),
                                   top_seeds=getattr(args, "graph_seeds", 5),
                                   alpha=getattr(args, "graph_alpha", 0.2), beta=getattr(args, "graph_beta", 0.3)))
    info["timing_seconds"]["anchored_graph_and_scoring"] = time.perf_counter() - started
    info["usage"] = usage_summary(client.events + legacy_events)
    return vectors, probes, info, rows, details


def save_probe_report(output, dataset, info, rankings, details, metrics, gamma):
    gold = {g["query_id"]: set(target_ids(g)) for g in dataset.gold}
    qids = list(gold)
    adaptive = {q: details[q]["adaptive_candidates"] for q in qids}
    adaptive_metrics = {"ocsp_adaptive": score_variable_candidates(adaptive, gold)}
    rank_stats = {}
    for name, ranked in rankings.items():
        positions = metrics[name]["gold_ranks"]
        all_ranks = [position for q in qids for position in positions[q].values()]
        complete = bool(all_ranks) and None not in all_ranks
        rank_stats[name] = {"mean_gold_rank": mean(all_ranks) if complete else None,
                            "median_gold_rank": median(all_ranks) if complete else None, "all_gold_ranks": all_ranks}
        if name in ("bm25", "dense", "hybrid"):
            truncated = {q: ranked[q][:len(adaptive[q])] for q in qids}
            adaptive_metrics[name] = score_variable_candidates(truncated, gold)
    for q in qids:
        details[q]["gold_ranks"] = {m: {name: metrics[name]["gold_ranks"][q][m] for name in rankings}
                                    for m in sorted(gold[q])}
    summary = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "input_paths": {k: str(p) for k, p in dataset.paths.items()},
               "counts": {"queries": len(qids), "memories": len(dataset.memories), "gold_pairs": sum(map(len, gold.values()))},
               **info, "gamma": gamma, "rank_stats": rank_stats, "adaptive_metrics": adaptive_metrics,
               "fixed_k_metrics": {name: {str(k): score_fixed_k(ranked, gold, k) for k in TOP_KS}
                                   for name, ranked in rankings.items()}}
    write_json(output / "per_query_details.json", [details[q] for q in qids])
    write_json(output / "rankings.json", rankings)
    write_json(output / "summary.json", summary)


def run(dataset, args, output, methods):
    started = time.perf_counter()
    qids = [q["query_id"] for q in dataset.queries]
    sizes = {scene: len(indices) for scene, (indices, _) in dataset.groups().items()}
    cfg = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset.resolve()),
           "dataset_sha256": {p.name: sha256_file(p) for p in dataset.paths.values()}, "methods_run": list(methods),
           "query_ids": qids, "scenario_ids": [s for s in sizes if s is not None],
           "library_sizes": [{"scenario_id": s, "memories": n} for s, n in sizes.items()],
           "input": "Original content only, ranked within the query's memory bank", "seed_sampling": args.seed,
           "embedding_backend": args.embedding_backend, "gamma": args.gamma,
           "bm25": {"k1": 1.2, "b": 0.75}, "hybrid": {"rrf_constant": 60, "equal_weights": True},
           "candidate_pool": {"sparsecl": "IndexHNSWFlat", "M": 64, "efSearch": INITIAL_RETRIEVAL_DEPTH},
           "code_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    with_anchored = bool(set(ANCHORED_METHODS).intersection(methods))
    if {LEGACY_METHOD, ABLATION}.intersection(methods):
        cfg["probe_prompt_sha256"] = fingerprint(ONLINE_ZERO_SHOT_BIPOLAR_PROMPT)
    if with_anchored:
        cfg["anchored_prompt_sha256"] = ANCHORED_PROMPT_SHA256
        cfg["graph"] = {"k": getattr(args, "graph_k", 5), "top_seeds": getattr(args, "graph_seeds", 5),
                        "alpha": getattr(args, "graph_alpha", 0.2), "beta": getattr(args, "graph_beta", 0.3)}
        cfg["probe_workers"] = getattr(args, "probe_workers", 4)
        cfg["anchor_encoded"] = False
    if set(methods) != {"bm25"} and args.embedding_backend == "bge":
        cfg["model_lock_sha256"] = sha256_file(Path(__file__).with_name("model_lock.json"))
    if any(name.startswith("sparsecl_") for name in methods):
        cfg["sparsecl"] = {"reference_commit": SPARSECL_UPSTREAM_COMMIT, "score": SPARSECL_SCORE_FORMULA,
                          "model_code_sha256": sha256_file(args.model_root / "sparsecl/models.py")}
    if args.source:
        cfg["source_dataset"] = str(args.source.resolve())
        cfg["source_dataset_sha256"] = {f: sha256_file(args.source / f) for f in cfg["dataset_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_config.json", cfg)
    try:
        if with_anchored:
            vectors, probes, info, rows, details = prepare_anchored_experiment(dataset, args, methods, output)
        else:
            vectors, probes, info = prepare_embeddings(dataset, args, methods, output)
            rows, details = score_original_text(dataset, methods, vectors, probes, args.gamma)
            rows.update(score_sparsecl(dataset, methods, vectors, args, info))
        rankings, metrics, depths = {}, {}, {}
        for name in methods:
            method_rows = [rows[name][q] for q in qids]
            write_jsonl(output / "arms" / f"{name}.jsonl", method_rows)
            rankings[name] = {q: [c["memory_id"] for c in rows[name][q]["ranked_candidates"]] for q in qids}
            lengths = [len(ids) for ids in rankings[name].values()]
            depths[name] = lengths[0] if min(lengths) == max(lengths) else dict(zip(qids, lengths))
            metrics[name] = ranking_metrics(rankings[name], dataset.gold, (*TOP_KS, 100, max(lengths)))
        if probes:
            save_probe_report(output, dataset, info, rankings, details, metrics, args.gamma)
        if with_anchored:
            write_json(output / "rankings.json", rankings)
        if any(sha256_file(p) != cfg["dataset_sha256"][p.name] for p in dataset.paths.values()):
            raise ValueError("Dataset inputs changed during the experiment")
        groups = grouped_ranking_metrics(rankings, dataset.queries, dataset.gold)
        report = {"dataset": cfg["dataset"], "query_count": len(qids), "memory_count": len(dataset.memories),
                  "gold_pairs": sum(len(target_ids(row)) for row in dataset.gold), "arms": metrics,
                  "groups": groups,
                  "models": info["models"], "usage": info["usage"], "timing": info["timing_seconds"],
                  "total_wall_seconds": time.perf_counter() - started}
        if with_anchored:
            report["anchored_probe_audit"] = info["anchored_probe_audit"]
        write_json(output / "report.json", report)
        table = ["# 记忆检索实验", "", f"{len(qids)} 条查询，{len(dataset.memories)} 条旧记忆，{report['gold_pairs']} 对已标注目标。",
                 "", "Top-K 为已标注目标总命中数；NDCG@10 按查询平均。"]
        for group, summary in groups.items():
            table.extend(["", f"## {'全集' if group == 'all' else group}（{summary['queries']} 查询，{summary['gold_pairs']} 目标）",
                          "", "| 方法 | Top-3 | Top-5 | Top-10 | NDCG@10 | 原生 M_old Top-10 |", "|---|---:|---:|---:|---:|---:|"])
            for name, metric in summary["methods"].items():
                hits = [str(metric["at_k"][str(k)]["hits"]) for k in (3, 5, 10)]
                native = metric["at_k"]["10"]["recall_by_source"].get("stale_native")
                native_text = f"{native['hits']}/{native['gold_pairs']}" if native else "—"
                table.append(f"| {name} | {' | '.join(hits)} | {metric['ndcg_at_10']:.4f} | {native_text} |")
        (output / "report.zh.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        packages = {"numpy": np.__version__}
        if vectors and args.embedding_backend == "bge":
            packages.update({p: importlib.metadata.version(p) for p in ("torch", "transformers")})
        if any(name.startswith("sparsecl_") for name in methods):
            packages["faiss-cpu"] = importlib.metadata.version("faiss-cpu")
        cfg.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), models=info["models"], packages=packages)
        write_json(output / "run_config.json", cfg)
        artifacts = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*")
                     if p.is_file() and p.suffix in {".json", ".jsonl", ".md"}}
        write_json(output / "completion.json", {"status": "completed", "queries_per_arm": len(qids),
                   "candidates_per_query_by_arm": depths, "artifacts_sha256": artifacts})
        print("\n".join(table), flush=True)
        print(output, flush=True)
    except Exception:
        cfg["status"] = "failed"
        write_json(output / "run_config.json", cfg)
        raise
    return output

"""Run OCSP-QR and its frozen baselines on the same per-query memory banks."""

from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
from statistics import mean, median
import time

import numpy as np

from .api import APIClient, usage_summary
from .baselines import BM25, rank, reciprocal_rank_fusion
from .config import Settings
from .data import fingerprint, sha256_file, write_json, write_jsonl
from .encoders import CPU_THREADS, MODEL_LOCK, encode_groups
from .evaluation import ranking_metrics, target_ids
from .ocsp import (ADAPTIVE_MAX_K, ADAPTIVE_MIN_ORTHO_PROJ, ADAPTIVE_MIN_VIOLATION_SIM,
                   ONLINE_ZERO_SHOT_BIPOLAR_PROMPT, TOP_KS, compute_subspace_ocsp_scores,
                   extract_online_bipolar_probes, score_fixed_k, score_variable_candidates)
from .sparsecl import (INITIAL_RETRIEVAL_DEPTH, SPARSECL_SCORE_FORMULA, SPARSECL_UPSTREAM_COMMIT,
                       fused_order, score_candidates)


BASELINES = ("bm25", "dense", "hybrid", "sparsecl_arguana", "sparsecl_msmarco", "sparsecl_hotpotqa")
MAIN_METHOD = "ocsp_qr"
ABLATION = "probes_only_no_ortho"


def prepare_embeddings(dataset, args, methods, output):
    with_probes = bool({MAIN_METHOD, ABLATION}.intersection(methods))
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
                ordered = rank(scores[MAIN_METHOD], docs)
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
    if {MAIN_METHOD, ABLATION}.intersection(methods):
        cfg["probe_prompt_sha256"] = fingerprint(ONLINE_ZERO_SHOT_BIPOLAR_PROMPT)
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
        if any(sha256_file(p) != cfg["dataset_sha256"][p.name] for p in dataset.paths.values()):
            raise ValueError("Dataset inputs changed during the experiment")
        report = {"dataset": cfg["dataset"], "query_count": len(qids), "memory_count": len(dataset.memories),
                  "gold_pairs": sum(len(target_ids(row)) for row in dataset.gold), "arms": metrics,
                  "models": info["models"], "usage": info["usage"], "timing": info["timing_seconds"],
                  "total_wall_seconds": time.perf_counter() - started}
        write_json(output / "report.json", report)
        table = ["# 记忆检索实验", "", f"{len(qids)} 条查询，{len(dataset.memories)} 条旧记忆，{report['gold_pairs']} 对已标注目标。",
                 "", "| 方法 | Top-10 命中 | NDCG@10 |", "|---|---:|---:|"]
        for name, metric in metrics.items():
            at10 = metric["at_k"]["10"]
            table.append(f"| {name} | {at10['hits']}/{at10['gold_pairs']} | {metric['ndcg_at_10']:.4f} |")
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

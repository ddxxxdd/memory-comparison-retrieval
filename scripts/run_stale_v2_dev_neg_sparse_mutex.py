"""Offline STALE v2 Dev diagnostic of sparse, mutually exclusive negative seeds."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import mean
import sys
from tempfile import TemporaryDirectory
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    sys.dont_write_bytecode = True
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"

import numpy as np
import pytrec_eval

from memory_retrieval.baselines import rank
from memory_retrieval.bipolar_graph import (build_knn_transition, compute_bipolar_diffusion_scores,
                                            validate_anchored_probe_structure)
from memory_retrieval.data import read_jsonl, sha256_file, write_json
from memory_retrieval.dataset import load_dataset
from memory_retrieval.encoders import FrozenEncoder, MODEL_ROOT
from memory_retrieval.sparsecl import ndcg_at_10

EXPERIMENTS = ROOT / "runs/experiments"
DATA = ROOT / "data/stale_scenarios_v2/augmented"
LOCKED_TEST = "stale-v2-60scene-bipolar-seed20261122"
OUTPUT = EXPERIMENTS / "stale-v2-dev-neg-sparse-mutex"
PREFERRED = ("stale-unseen-10scene-query-only-bge-seed20261121", "stale-anchored-dev-test-bge-20261120/dev")
CACHE_NAMES = {"anchored_probes.json", "probes.json", "new_probes.json", "probe_cache.sqlite3"}
GRAPH_K, TOP_SEEDS, NEGATIVE_SEEDS, ALPHA, DENSE_BETA = 5, 5, 5, 0.2, 0.3
BETAS = {f"sparse_mutex_beta_{beta:g}": beta for beta in (0.3, 0.5, 0.8)}
POSITIVE, BIPOLAR = "anchored_pos_diffusion", "anchored_bipolar_diffusion"
ARMS = (POSITIVE, BIPOLAR, *BETAS)
GROUPS = ("Overall", "T1", "T2")
CUTOFFS = (3, 5, 10)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def forbid_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "socket.sendmsg"}:
        raise RuntimeError("Network access prohibited in this offline diagnostic")


def locked_partition(directory):
    protocol = read(directory / "protocol.json")
    for name in ("selection", "exclusions"):
        if sha256_file(directory / f"{name}.json") != protocol[f"{name}_sha256"]:
            raise ValueError(f"Locked {name}.json hash mismatch")
    selection = read(directory / "selection.json")
    test_ids = {row["scenario_id"] for row in selection["scenarios"]}
    test_native_hashes = {row["normalized_native_text_sha256"] for row in selection["scenarios"]}
    excluded = {row["scenario_id"]: row for row in read(directory / "exclusions.json")["records"]}
    if (len(test_ids) != 60 or len(test_native_hashes) != 60
            or Counter(row["type"] for row in selection["scenarios"]) != {"T1": 30, "T2": 30}
            or set(excluded) != set(selection["excluded_scenario_ids"]) or test_ids & excluded.keys()):
        raise ValueError("Invalid locked Test/historical exclusion partition")
    return excluded, test_ids, test_native_hashes


def probe_sources(experiments):
    folders = [*(experiments / name for name in PREFERRED),
               *(path for path in sorted(experiments.glob("stale-*")) if path.is_dir()
                 and path.name != OUTPUT.name
                 and not path.name.startswith(("stale-unseen-60scene-", "stale-v2-60scene-")))]
    return list(dict.fromkeys(path for folder in folders for path in sorted(
        (item for item in folder.rglob("*") if item.name in CACHE_NAMES and item.is_file()),
        key=lambda item: (item.suffix == ".sqlite3", item))))


def cached_records(path):
    if path.suffix != ".sqlite3":
        yield from read(path)
        return
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        for raw_request, raw_response in database.execute("SELECT request, response FROM responses ORDER BY key"):
            try:
                messages = json.loads(raw_request)["body"]["messages"]
                if len(messages) != 2 or [message["role"] for message in messages] != ["system", "user"]:
                    continue
                payload = json.loads(messages[1]["content"])
                if (not isinstance(payload, dict) or set(payload) != {"query"}
                        or not isinstance(payload["query"], str)):
                    continue
                choice = json.loads(raw_response)["choices"][0]
                response = json.loads(choice["message"]["content"])
                validate_anchored_probe_structure({"violation_probes": response.get("violation_probes")})
                yield {**response, "query_content": payload["query"], "finish_reason": choice.get("finish_reason"),
                       "prompt_sha256": hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()}
            except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                continue


def probe_key(record):
    anchor = json.dumps(record["new_state_anchor"], sort_keys=True) if "new_state_anchor" in record else None
    return record["query_content"], tuple(record["violation_probes"]), anchor


def prompt_provenance(record, path, experiments, indexes):
    if path.suffix == ".sqlite3":
        return {"prompt_sha256": record["prompt_sha256"], "prompt_status": "verified_request",
                "prompt_source": path.relative_to(experiments).as_posix()}

    folders = (path.parent, path.parent.parent) if path.parent.name == "results" else (path.parent,)
    matches = {}
    status = "missing_request_cache"
    signature = probe_key(record)
    for folder in folders:
        cache = folder / "probe_cache.sqlite3"
        if not cache.is_file():
            continue
        if cache not in indexes:
            index = defaultdict(set)
            for cached in cached_records(cache):
                if cached["finish_reason"] != "length":
                    index[probe_key(cached)].add(cached["prompt_sha256"])
            indexes[cache] = index
        for digest in indexes[cache].get(signature, ()):
            matches.setdefault(digest, cache)
        status = "no_matching_request"
    if len(matches) == 1:
        digest, cache = next(iter(matches.items()))
        return {"prompt_sha256": digest, "prompt_status": "verified_request",
                "prompt_source": cache.relative_to(experiments).as_posix()}
    return {"prompt_sha256": None, "prompt_source": None,
            "prompt_status": "ambiguous_request" if matches else status}


def select_cached_probes(experiments, queries, allowed_ids):
    by_id = {query["query_id"]: query for query in queries}
    text_counts = Counter(query["content"] for query in queries)
    unique_texts = {query["content"]: query for query in queries if text_counts[query["content"]] == 1}
    selected, indexes = {}, {}
    for path in probe_sources(experiments):
        for record in cached_records(path):
            if (not isinstance(record, dict) or record.get("finish_reason") == "length"
                    or record.get("status") not in (None, "ok", "completed")):
                continue
            text = record.get("query_content")
            if not isinstance(text, str):
                continue
            query = by_id.get(record["query_id"]) if "query_id" in record else unique_texts.get(text)
            if not query or query["content"] != text:
                continue
            sid = query["scenario_id"]
            if sid not in allowed_ids or sid in selected or record.get("scenario_id", sid) != sid:
                continue
            probes = record.get("violation_probes")
            try:
                validate_anchored_probe_structure({"violation_probes": probes})
            except ValueError:
                continue
            selected[sid] = {"query": query, "violation_probes": probes,
                             "probe_source": path.relative_to(experiments).as_posix(),
                             "legacy_anchor_ignored": "new_state_anchor" in record,
                             **prompt_provenance(record, path, experiments, indexes)}
    return selected


def native_hash(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    return hashlib.sha256(" ".join(normalized.split()).encode("utf-8")).hexdigest()


def load_dev(data, experiments):
    excluded, test_ids, test_native_hashes = locked_partition(experiments / LOCKED_TEST)
    if read(data / "manifest.json")["extraction_version"] != "v2":
        raise ValueError("Expected STALE v2 augmented corpus")
    queries = read_jsonl(data / "queries.jsonl")
    if len({query["query_id"] for query in queries}) != len(queries):
        raise ValueError("STALE v2 query IDs must be unique")
    selected = select_cached_probes(experiments, queries, set(excluded) - test_ids)
    if not selected or set(selected) & test_ids or set(selected) - excluded.keys():
        raise ValueError("Dev cache selection is empty or overlaps the locked Test")
    for sid, scene in selected.items():
        kind = excluded[sid]["type"]
        scene_path = data / "scenarios" / kind / sid
        dataset = load_dataset(scene_path)
        if (dataset.queries != [scene["query"]] or scene["query"]["source_ref"]["type"] != kind
                or any(memory.scenario_id != sid for memory in dataset.memories)):
            raise ValueError(f"STALE v2 scene mapping mismatch: {sid}")
        details = dataset.gold[0]["target_details"]
        native = [mid for mid, detail in details.items() if detail["source"] == "stale_native"]
        if native != [excluded[sid]["native_memory_id"]] or any(
                detail["source"] not in {"stale_native", "synthetic_review"} for detail in details.values()):
            raise ValueError(f"Unexpected STALE v2 targets: {sid}")
        digest = native_hash(next(memory.content for memory in dataset.memories if memory.memory_id == native[0]))
        if digest != excluded[sid]["normalized_native_text_sha256"] or digest in test_native_hashes:
            raise ValueError(f"Native M_old text changed or overlaps Test: {sid}")
        scene.update(dataset=dataset, native_id=native[0], kind=kind)
    if {scene["kind"] for scene in selected.values()} != {"T1", "T2"}:
        raise ValueError("Dev must include cached scenes from both T1 and T2")
    return list(selected.values())


def negative_seeds(s_plus, r0, docs):
    raw_neg = np.maximum(0.0, s_plus - r0)
    raw_neg[rank(r0, docs)[:TOP_SEEDS]] = 0.0
    selected = [index for index in rank(raw_neg, docs) if raw_neg[index] > 0.0][:NEGATIVE_SEEDS]
    sparse = np.zeros_like(raw_neg)
    sparse[selected] = raw_neg[selected]
    return sparse


def score_variants(query_vec, probe_vecs, bank, transition, docs):
    baselines = compute_bipolar_diffusion_scores(query_vec, probe_vecs, bank, transition, docs,
                                                top_seeds=TOP_SEEDS, alpha=ALPHA, beta=DENSE_BETA)
    r0, s_plus = baselines["anchored_probes_only"], bank @ query_vec
    z_neg_sparse = negative_seeds(s_plus, r0, docs)
    penalty_base = z_neg_sparse + ALPHA * (transition @ z_neg_sparse)
    positive = baselines[POSITIVE]
    return {POSITIVE: positive, BIPOLAR: baselines[BIPOLAR],
            **{name: positive - beta * penalty_base for name, beta in BETAS.items()}}


def measure_ranking(order, gold, native_id):
    targets = set(gold["conflict_old_memory_ids"])
    ranks = {mid: position for position, mid in enumerate(order, 1) if mid in targets}
    positions = sorted(ranks.values())
    count = len(targets)
    if len(ranks) != count or native_id not in ranks:
        raise ValueError("A full ranking must cover every target including M_old")
    return {"gold_ranks": ranks, "hits": {str(k): sum(position <= k for position in positions) for k in CUTOFFS},
            "native_hit_at_10": ranks[native_id] <= 10, "ndcg_at_10": ndcg_at_10([gold], {gold["query_id"]: order}),
            "r_precision": sum(position <= count for position in positions) / count,
            "average_precision": math.fsum(index / position for index, position in enumerate(positions, 1)) / count}


def rank_changes(before, after):
    counts = dict.fromkeys(("top10_recovered", "top10_dropped", "topG_recovered", "topG_dropped", "rank_up",
                            "rank_unchanged", "rank_down", "ndcg_wins", "ndcg_ties", "ndcg_losses"), 0)
    for old, new in zip(before, after, strict=True):
        old_ranks, new_ranks = old["gold_ranks"], new["gold_ranks"]
        for mid, old_rank in old_ranks.items():
            new_rank = new_ranks[mid]
            for label, cutoff in (("top10", 10), ("topG", len(old_ranks))):
                counts[f"{label}_recovered"] += int(old_rank > cutoff >= new_rank)
                counts[f"{label}_dropped"] += int(new_rank > cutoff >= old_rank)
            counts["rank_up" if new_rank < old_rank else "rank_down" if new_rank > old_rank else "rank_unchanged"] += 1
        difference = new["ndcg_at_10"] - old["ndcg_at_10"]
        counts["ndcg_wins" if difference > 1e-12 else "ndcg_losses" if difference < -1e-12 else "ndcg_ties"] += 1
    return counts


def summarize(per_query):
    result = {}
    for group in GROUPS:
        rows = [row for row in per_query if group == "Overall" or row["type"] == group]
        targets = sum(row["gold_count"] for row in rows)
        methods = {}
        baseline = [row["methods"][POSITIVE] for row in rows]
        for name in ARMS:
            values = [row["methods"][name] for row in rows]
            hits = {str(k): sum(row["hits"][str(k)] for row in values) for k in CUTOFFS}
            methods[name] = {"hits": hits, "micro_recall": {k: value / targets for k, value in hits.items()},
                             "ndcg_at_10": mean(row["ndcg_at_10"] for row in values),
                             "macro_r_precision": mean(row["r_precision"] for row in values),
                             "map": mean(row["average_precision"] for row in values),
                             "native_hits_at_10": sum(row["native_hit_at_10"] for row in values)}
            if name != POSITIVE:
                methods[name]["against_positive"] = rank_changes(baseline, values)
        result[group] = {"scenarios": len(rows), "gold_targets": targets, "methods": methods}
    return result


def verify_ndcg(qrels, rankings, per_query):
    max_error = 0.0
    for name in ARMS:
        result = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(rankings[name])
        if set(result) != set(qrels):
            raise ValueError(f"Incomplete pytrec_eval results: {name}")
        for row in per_query:
            expected = result[row["query_id"]]["ndcg_cut_10"]
            max_error = max(max_error, abs(row["methods"][name]["ndcg_at_10"] - expected))
    if max_error > 1e-12:
        raise ValueError(f"pytrec_eval NDCG@10 difference: {max_error}")
    return {"verified_query_method_pairs": len(per_query) * len(ARMS), "max_ndcg_at_10_difference": max_error}


def diagnose(scenes, vectors):
    per_query, qrels = [], {}
    rankings = {name: {} for name in ARMS}
    start = 0
    for index, scene in enumerate(scenes):
        query = scene["query"]
        qid, sid = query["query_id"], query["scenario_id"]
        docs = scene["dataset"].memories
        gold = scene["dataset"].gold[0]
        stop = start + len(docs)
        bank = vectors["memories"][start:stop]
        start = stop
        transition = build_knn_transition(bank, docs, k=GRAPH_K)
        scores = score_variants(vectors["queries"][index], vectors["probes"][4 * index:4 * index + 4],
                                bank, transition, docs)
        qrels[qid] = dict.fromkeys(gold["conflict_old_memory_ids"], 1)
        metrics = {}
        for name in ARMS:
            order = [docs[position].memory_id for position in rank(scores[name], docs)]
            rankings[name][qid] = {mid: len(order) - position for position, mid in enumerate(order)}
            metrics[name] = measure_ranking(order, gold, scene["native_id"])
        per_query.append({"query_id": qid, "scenario_id": sid, "type": scene["kind"], "gold_count": len(qrels[qid]),
                          "library_size": len(docs), "native_id": scene["native_id"], "methods": metrics})
    verification = verify_ndcg(qrels, rankings, per_query)
    return {"groups": summarize(per_query), "per_query": per_query, "pytrec_eval": verification}


def metric_table(group):
    lines = ["| 方法 | R@3 | R@5 | R@10 | NDCG@10 | Macro R-Precision | MAP | M_old@10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in group["methods"].items():
        recall = [f"{row['micro_recall'][str(k)]:.2%} ({row['hits'][str(k)]}/{group['gold_targets']})" for k in CUTOFFS]
        values = [f"{row[key]:.4f}" for key in ("ndcg_at_10", "macro_r_precision", "map")]
        lines.append("| " + " | ".join([name, *recall, *values, f"{row['native_hits_at_10']}/{group['scenarios']}"]) + " |")
    return lines


def markdown(summary):
    scenes = summary["selection"]["scenes"]
    sources = Counter(row["probe_source"] for row in scenes)
    prompts = Counter((row["prompt_sha256"], row["prompt_status"], row["legacy_anchor_ignored"]) for row in scenes)
    missing_reasons = {"missing_request_cache": "缺少原始请求缓存", "no_matching_request": "无完全匹配的请求与响应",
                       "ambiguous_request": "匹配到不同提示词的请求"}
    lines = ["# STALE v2 Dev：Top-5 稀疏负种子与正负互斥", "",
             "仅用历史排除集中已有四探针的场景；与锁定 60 场景的 ID 和原生 M_old 正文均无交集。"
             "冻结 BGE 本地编码，0 次 API 调用。所有方法共用同一批历史探针；实验只比较负向压制公式的增量，不比较探针提示词。", "",
             "缓存来源：" + "；".join(f"`{name}` × {count}" for name, count in sources.items()) + "。", "",
             "历史提示词 SHA256（从实际请求核实）：",
             *(f"- {'旧响应含锚点（未编码）' if legacy else '旧响应仅含探针'}：{count} 场景，"
               + (f"`{digest}`" if digest else f"未核实（{missing_reasons[status]}）")
               for (digest, status, legacy), count in prompts.items()), "",
             "k=5、正/负种子各最多 5、α=0.2，现行双极 β=0.3，改进版 β=0.3/0.5/0.8。"
             "负种子按原始余弦分差选取；正种子不直接扣分，仍可能收到邻居传播的扣分。",
             "R@k 是 Micro Recall。G 为该场景的标注目标数，R-Precision 取 Top-G 命中率再按场景平均；MAP 按全库排名。"
             "NDCG@10 用 pytrec_eval 逐场景复核，最大误差 " + f"{summary['pytrec_eval']['max_ndcg_at_10_difference']:.3g}。", ""]
    for group, result in summary["groups"].items():
        lines += [f"## {group}：{result['scenarios']} 场景、{result['gold_targets']} 目标", "", *metric_table(result), "",
                  "| 相对正向扩散 | Top-10 拉回/推走 | Top-G 拉回/推走 | 全库排名升/平/降 | NDCG 胜/平/负 |",
                  "|---|---:|---:|---:|---:|"]
        for name, values in result["methods"].items():
            if name != POSITIVE:
                changes = values["against_positive"]
                columns = [("top10_recovered", "top10_dropped"), ("topG_recovered", "topG_dropped"),
                           ("rank_up", "rank_unchanged", "rank_down"), ("ndcg_wins", "ndcg_ties", "ndcg_losses")]
                lines.append("| " + " | ".join([name, *("/".join(str(changes[key]) for key in keys) for keys in columns)]) + " |")
        lines.append("")
    return "\n".join(lines)


def require_new_output(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")


def rename_directory_noreplace(source, destination):
    """Publish atomically, even if another process creates the destination."""
    if os.name == "nt":
        source.rename(destination)
        return
    if sys.platform != "linux":
        raise NotImplementedError("Atomic no-replace publication requires Linux or Windows")

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError("Linux renameat2 is required for atomic no-replace publication")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if renameat2(at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination), rename_noreplace) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), source, None, destination)


def write_outputs(output, summary):
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as directory:
        staged = Path(directory) / "results"
        write_json(staged / "summary.json", summary)
        (staged / "report.zh.md").write_text(markdown(summary), encoding="utf-8")
        try:
            rename_directory_noreplace(staged, output)
        except FileExistsError as error:
            raise FileExistsError(error.errno, "Refusing to overwrite experiment", output) from error


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    require_new_output(OUTPUT)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.addaudithook(forbid_network)

    scenes = load_dev(DATA, EXPERIMENTS)
    groups = {"memories": [memory.content.strip() for scene in scenes for memory in scene["dataset"].memories],
              "queries": [scene["query"]["content"].strip() for scene in scenes],
              "probes": [text for scene in scenes for text in scene["violation_probes"]]}
    vectors, identity, seconds = FrozenEncoder("bge", MODEL_ROOT).encode_groups(groups)
    summary = diagnose(scenes, vectors)
    summary.update(protocol={"dataset": str(DATA), "locked_test_selection": str(EXPERIMENTS / LOCKED_TEST / "selection.json"),
                             "k": GRAPH_K, "top_seeds": TOP_SEEDS, "negative_seeds": NEGATIVE_SEEDS, "alpha": ALPHA,
                             "baseline_beta": DENSE_BETA, "improved_betas": list(BETAS.values()),
                             "negative_seed_score": "max(0, s_plus - r0)", "seed_mutex": True,
                             "llm_api_calls": 0, "embedding_api_calls": 0, "dev_only": True},
                   selection={"test_scene_overlap": 0, "test_native_text_overlap": 0, "scenes": [
                       {key: value for key, value in scene.items() if key != "dataset"} for scene in scenes]},
                   encoder=identity, encoding_seconds=seconds)
    write_outputs(OUTPUT, summary)
    print("\n".join(["Overall", *metric_table(summary["groups"]["Overall"]), "", str(OUTPUT)]))


if __name__ == "__main__":
    main()

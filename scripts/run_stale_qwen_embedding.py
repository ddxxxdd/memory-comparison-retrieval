"""Frozen 60-scene Qwen embedding transfer: cached probes, six arms, no LLM calls."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dotenv import dotenv_values

from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.bipolar_graph import (ANCHORED_PROMPT_SHA256, build_knn_transition,
                                           compute_bipolar_diffusion_scores)
from memory_retrieval.config import Settings, endpoint
from memory_retrieval.data import fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from memory_retrieval.dataset import load_dataset
from memory_retrieval.evaluation import grouped_ranking_metrics, target_ids
from memory_retrieval.experiment import ranked_row, score_original_text

SOURCE = ROOT / "runs/experiments/stale-unseen-60scene-bipolar-seed20261122"
OUTPUT = ROOT / "runs/experiments/stale-unseen-60scene-qwen-emb-seed20261122"
REPORT = ROOT / "docs/stale_unseen_60scene_qwen_emb_seed20261122_experiment.md"
PROMPT_HASH = "a38b198018c62846780914f1ef90fe2461ee1554c421d42996f54525f93217bf"
GRAPH = {"k": 5, "top_seeds": 5, "alpha": .2, "beta": .3}
FULL = "anchored_bipolar_diffusion"
METHODS = ["bm25", "dense", "hybrid", "anchored_probes_only", "anchored_pos_diffusion", FULL]
NAMES = {"bm25": "BM25", "dense": "Qwen Dense", "hybrid": "Hybrid",
         "anchored_probes_only": "仅探针", "anchored_pos_diffusion": "正向扩散", FULL: "完整方法"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def atomic_npy(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as stream:
        np.save(stream, value)
    tmp.replace(path)


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        assert sha256_file(root / name) == expected, name


def configured_dimensions():
    values = {**dotenv_values(ROOT / ".env", encoding="utf-8-sig"), **os.environ}
    configured = [int(values[k]) for k in ("EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSION") if values.get(k)]
    if not configured:
        return None
    if min(configured) < 1 or len(set(configured)) != 1:
        raise ValueError("Invalid or conflicting configured embedding dimensions")
    return configured[0]


def decode_embeddings(response, count, dimensions=None):
    items = response.get("data")
    if not isinstance(items, list) or len(items) != count:
        raise ValueError("Embedding response size does not match request")
    indices = [item.get("index") for item in items]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
        raise ValueError("Embedding indices are missing or duplicated")
    matrix = np.asarray([item["embedding"] for item in sorted(items, key=lambda x: x["index"])], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must be finite and share a nonzero dimension")
    if dimensions is not None and matrix.shape[1] != dimensions:
        raise ValueError("Returned dimension differs from explicitly configured dimension")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Cannot normalize nonfinite or zero vectors")
    return np.asarray(matrix / norms, dtype=np.float32)


def text_groups(dataset, probes):
    assert len(probes) == len(dataset.queries) == 60
    for q, p in zip(dataset.queries, probes):
        assert q["query_id"] == p["query_id"] and q["content"] == p["query_content"]
        assert len(p["violation_probes"]) == 4
        assert all(isinstance(t, str) and t.strip() for t in p["violation_probes"])
    return {"memories": [m.content.strip() for m in dataset.memories],
            "queries": [q["content"].strip() for q in dataset.queries],
            "violation_probes": [t for p in probes for t in p["violation_probes"]]}


def prepare(source, output, settings, dimensions):
    assert not output.exists(), "Use a new output directory or --resume"
    assert not REPORT.exists(), "Report already exists"
    old = read(source / "protocol.json")
    assert ANCHORED_PROMPT_SHA256 == old["prompt_sha256"] == PROMPT_HASH
    assert old["graph"] == GRAPH
    assert read(source / "completion.json")["status"] == "complete"
    check_hashes(ROOT / "memory_retrieval", old["code_sha256"])
    check_hashes(source, read(source / "completion.json")["artifacts_sha256"])
    hashes = {p.relative_to(source).as_posix(): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    output.mkdir(parents=True)
    shutil.copytree(source / "inputs", output / "inputs")
    for name in ("selection.json", "exclusions.json", "anchored_probes.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(Path(__file__), output / "runner.py")
    atomic_json(output / "source_manifest.json", hashes)
    protocol = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_manifest_sha256": sha256_file(output / "source_manifest.json"),
        "input_sha256": old["input_sha256"], "probes_sha256": sha256_file(source / "anchored_probes.json"),
        "script_sha256": sha256_file(Path(__file__)), "main_package_sha256": old["code_sha256"],
        "prompt_sha256": PROMPT_HASH, "graph": GRAPH, "methods": METHODS,
        "embedding": {**settings.embedding_identity(), "requested_dimensions": dimensions,
                      "batch_size": settings.embedding_batch_size, "workers": 4,
                      "dtype_after_decode": "float32", "normalization": "L2",
                      "text_processing": "same original text as BGE; no prefixes or instructions; no anchor",
                      "configuration_source": "Settings.from_env(.env), dimensions optional from env"},
        "transport": {"max_retries": settings.max_retries, "timeout": settings.timeout,
                      "rate_limit_backoff": settings.rate_limit_backoff},
        "llm_calls": 0, "cache_policy": "new independent sqlite response cache plus normalized per-batch/group vectors",
        "statistics": {"paired_unit": "scenario", "comparisons": ["dense", "hybrid", "anchored_probes_only"],
            "wilcoxon": "two-sided, Pratt zeros, normal approximation, no continuity correction; differences rounded 12 decimals for ranks",
            "bootstrap_replicates": 20000, "bootstrap_seed": 20261122, "confidence": .95,
            "interval": "percentile; all resampled within T1/T2 strata", "p_adjustment": "none; all primary, T1/T2 exploratory"}}
    atomic_json(output / "protocol.json", protocol)
    APIClient(settings)
    with sqlite3.connect(settings.cache_path) as db:
        assert db.execute("SELECT count(*) FROM responses").fetchone()[0] == 0


def encode_batch(name, offset, texts, settings, dimensions, output):
    key = f"{name}-{offset:06d}"
    metadata_path = output / "batches" / (key + ".json")
    vector_path = output / "batches" / (key + ".npy")
    payload = {"model": settings.embedding_model, "input": texts, "encoding_format": "float"}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    identity = fingerprint({"url": endpoint(settings.embedding_base_url, "embeddings"), "body": payload})
    if metadata_path.exists():
        record = read(metadata_path)
        assert record["request_fingerprint"] == identity
        assert sha256_file(vector_path) == record["vector_sha256"]
        return record
    client = APIClient(settings)
    event_path = output / "api_events" / (key + ".json")
    events = read(event_path) if event_path.exists() else []
    started = time.perf_counter()
    try:
        response = client._post(endpoint(settings.embedding_base_url, "embeddings"), settings.embedding_api_key,
                                payload, "qwen_embedding_" + name)
    finally:
        events.extend(client.events)
        atomic_json(event_path, events)
    vectors = decode_embeddings(response, len(texts), dimensions)
    atomic_npy(vector_path, vectors)
    record = {"group": name, "offset": offset, "texts": len(texts), "dimensions": vectors.shape[1],
              "request_fingerprint": identity, "text_sha256": fingerprint(texts),
              "vector_sha256": sha256_file(vector_path), "response_model": response.get("model"),
              "usage": response.get("usage"), "wall_seconds": time.perf_counter() - started,
              "l2_max_error": float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))}
    atomic_json(metadata_path, record)
    return record


def score_all(output, dataset, vectors):
    rows, _ = score_original_text(dataset, METHODS[:3], vectors, [], .65)
    rows.update({method: {} for method in METHODS[3:]})
    violations = vectors["violation_probes"].reshape(60, 4, vectors["queries"].shape[1])
    for sid, (memory_indices, query_indices) in dataset.groups().items():
        docs = [dataset.memories[i] for i in memory_indices]
        bank = vectors["memories"][memory_indices]
        transition = build_knn_transition(bank, docs, k=5)
        atomic_npy(output / "graphs" / (sid + ".npy"), transition)
        for i in query_indices:
            scores = compute_bipolar_diffusion_scores(vectors["queries"][i], violations[i], bank, transition,
                                                      docs, top_seeds=5, alpha=.2, beta=.3)
            for method in METHODS[3:]:
                rows[method][dataset.queries[i]["query_id"]] = ranked_row(dataset.queries[i], docs, method, scores[method])
    for method, values in rows.items():
        write_jsonl(output / "arms" / (method + ".jsonl"), [values[q["query_id"]] for q in dataset.queries])
    return rows


def evaluate(dataset, rows):
    import pytrec_eval
    rankings = {method: {qid: [r["memory_id"] for r in row["ranked_candidates"]] for qid, row in values.items()}
                for method, values in rows.items()}
    groups = grouped_ranking_metrics(rankings, dataset.queries, dataset.gold)
    qrels = {g["query_id"]: {mid: 1 for mid in target_ids(g)} for g in dataset.gold}
    native = {g["query_id"]: {mid: 1 for mid, d in g["target_details"].items() if d["source"] == "stale_native"} for g in dataset.gold}
    per_query, max_ndcg_error, max_recall_error = {}, 0., 0.
    banks = {sid: {dataset.memories[i].memory_id for i in mi} for sid, (mi, _) in dataset.groups().items()}
    for method, ranking in rankings.items():
        for q in dataset.queries:
            ids = ranking[q["query_id"]]
            assert len(ids) == len(set(ids)) and set(ids) == banks[q["scenario_id"]]
        strict = {qid: {mid: float(len(ids) - i) for i, mid in enumerate(ids)} for qid, ids in ranking.items()}
        # This pytrec_eval build retains cutoff state across evaluate() calls.
        # Reconstruct before each call so native@10 cannot remove recall@3/@5
        # from the next method's evaluation.
        result = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_3", "recall_5", "recall_10"}).evaluate(strict)
        native_result = pytrec_eval.RelevanceEvaluator(native, {"recall_10"}).evaluate(strict)
        per_query[method] = {qid: r["ndcg_cut_10"] for qid, r in result.items()}
        for name, group in groups.items():
            qids = [q["query_id"] for q in dataset.queries if name == "all" or q["source_ref"]["type"] == name]
            reported = group["methods"][method]
            difference = abs(float(np.mean([result[qid]["ndcg_cut_10"] for qid in qids])) - reported["ndcg_at_10"])
            max_ndcg_error = max(max_ndcg_error, difference)
            for k in (3, 5, 10):
                hits = sum(round(result[qid][f"recall_{k}"] * len(qrels[qid])) for qid in qids)
                assert hits == reported["at_k"][str(k)]["hits"]
                max_recall_error = max(max_recall_error, abs(hits / group["gold_pairs"] - reported["at_k"][str(k)]["micro_recall"]))
            native_hits = sum(round(native_result[qid]["recall_10"] * len(native[qid])) for qid in qids)
            assert native_hits == reported["at_k"]["10"]["recall_by_source"]["stale_native"]["hits"]
    assert max_ndcg_error < 1e-12 and max_recall_error == 0.
    return groups, per_query, {"pytrec_eval_verified": True, "max_ndcg_error": max_ndcg_error,
        "max_recall_error": max_recall_error, "topk_and_native_hits_error": 0,
        "method_group_rows": 18, "all_rankings_cover_exact_scene_bank": True}


def paired_statistics(queries, per_query):
    from scipy.stats import wilcoxon
    results = {}
    for group in ("T1", "T2", "all"):
        selected = [q for q in queries if group == "all" or q["source_ref"]["type"] == group]
        qids = [q["query_id"] for q in selected]
        results[group] = {}
        for base in ("dense", "hybrid", "anchored_probes_only"):
            diff = np.array([per_query[FULL][qid] - per_query[base][qid] for qid in qids])
            rounded = np.round(diff, 12)
            test = wilcoxon(rounded, zero_method="pratt", alternative="two-sided", method="approx", correction=False) if np.any(rounded) else None
            rng = np.random.default_rng(20261122)
            if group == "all":
                indices = [np.array([i for i, q in enumerate(selected) if q["source_ref"]["type"] == t]) for t in ("T1", "T2")]
                draws = np.concatenate([rng.choice(idx, size=(20000, len(idx)), replace=True) for idx in indices], axis=1)
            else:
                draws = rng.integers(0, len(qids), size=(20000, len(qids)))
            low, high = np.quantile(diff[draws].mean(axis=1), [.025, .975])
            results[group][base] = {"scenarios": len(qids), "mean_ndcg_difference": float(diff.mean()),
                "wilcoxon_statistic": float(test.statistic) if test else 0.,
                "wilcoxon_p_two_sided": float(test.pvalue) if test else 1., "bootstrap_ci95": [float(low), float(high)],
                "wins": int((rounded > 0).sum()), "ties": int((rounded == 0).sum()), "losses": int((rounded < 0).sum()),
                "per_scenario_difference": dict(zip(qids, map(float, diff)))}
    return results


def table(group):
    lines = ["| 方法 | Top-3 命中（召回） | Top-5 命中（召回） | Top-10 命中（召回） | NDCG@10 | 原生 M_old@10 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        m = group["methods"][method]
        cells = [f"{m['at_k'][str(k)]['hits']}/{group['gold_pairs']}（{m['at_k'][str(k)]['micro_recall']:.2%}）" for k in (3, 5, 10)]
        n = m["at_k"]["10"]["recall_by_source"]["stale_native"]
        lines.append(f"| {NAMES[method]} | " + " | ".join(cells) + f" | {m['ndcg_at_10']:.4f} | {n['hits']}/{n['gold_pairs']} |")
    return lines


def write_report(output, c):
    e = c["embedding"]
    date = datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    lines = ["# STALE 60 场景：Qwen Embedding 底座泛化对照", "",
        f"执行日期：{date}（北京时间）；20261122 是原样本抽样种子。", "",
        f"实际请求模型 `{e['model']}`，响应模型标识 `{', '.join(e['response_models'])}`，返回维度 **{e['actual_dimensions']}**。配置从服务器 `.env` 读取；维度未配置时采用接口默认返回维度，不硬编码。接口 `{e['url']}`，不记录密钥。", "",
        "严格复用前次 30 T1＋30 T2、6,673 条记忆、340 个已标注目标（60 个原生 M_old）及其 240 条探针，LLM 调用 **0 次**。仅替换向量底座；无新抽样、提示词修改、探针重抽、参数搜索或 SparseCL 运行。样本原先未见，本次属于固定样本上的底座替换验证，不能视为新的跨数据源泛化测试。", "",
        "所有三类文本按同一请求规范编码，原始文本不加指令前缀，结果转 FP32 并 L2 归一化。主包图构造为正余弦有向 Top-5、零对角、行归一化；并列按 score 降序和 memory_id 升序。Top-5 正种子、α=0.2、β=0.3 与原实验完全一致；直接调用主包打分函数。BM25 k1=1.2、b=0.75；Hybrid 沿用等权 RRF=60。", "",
        f"旧提示词 SHA256 `{PROMPT_HASH}`；探针文件复制后哈希完全一致。每批至多 {e['batch_size']} 条、{e['workers']} 并发，使用 APIClient 的 429/网络退避重试。SQLite 缓存保留原响应，各批和合并向量另存 .npy，支持断点复用。", ""]
    for name in ("T1", "T2", "all"):
        g = c["groups"][name]
        lines += [f"**{'总体' if name == 'all' else name}：{g['queries']} 场景、{g['gold_pairs']} 目标**", ""] + table(g) + [""]
    lines += ["**配对显著性：Qwen 完整方法减对应 Qwen 对照**", "",
        "双侧 Wilcoxon，Pratt 零差，正态近似，无连续性修正；仅用于检验秩的差值舍入 12 位。Bootstrap 20,000 次、种子 20261122，按场景成对重采样，总体保持 T1/T2 各 30，百分位 95% CI。总体比较为主要结果，分组为探索性结果；p 未做多重比较校正。", "",
        "| 分组 | 对照 | ΔNDCG@10 | Wilcoxon p | Bootstrap 95% CI | 胜/平/负 |", "|---|---|---:|---:|---|---|"]
    for group in ("T1", "T2", "all"):
        for base, r in c["significance"][group].items():
            lo, hi = r["bootstrap_ci95"]
            lines.append(f"| {group} | {NAMES[base]} | {r['mean_ndcg_difference']:+.4f} | {r['wilcoxon_p_two_sided']:.6g} | [{lo:+.4f}, {hi:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "**与 BGE 底座并列比较（同样本、同探针）**", "",
        "| 分组 | 底座 | Dense Top-10 | Dense NDCG@10 | 完整方法 Top-10 | 完整方法 NDCG@10 | 完整方法原生@10 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("T1", "T2", "all"):
        for model in ("BGE", "Qwen"):
            r = c["backbone_comparison"][name][model]
            lines.append(f"| {name} | {model} | {r['dense_hits10']}/{r['targets']}（{r['dense_recall10']:.2%}） | {r['dense_ndcg10']:.4f} | {r['full_hits10']}/{r['targets']}（{r['full_recall10']:.2%}） | {r['full_ndcg10']:.4f} | {r['full_native_hits10']}/{r['scenes']} |")
    t, u, retry = c["timing"], c["usage"], c["api_retry_stats"]
    lines += ["", f"本次实际运行 {t['total_wall_seconds']:.2f} 秒，其中 Embedding 编码批次墙钟时间 {t['embedding_wall_seconds']:.2f} 秒，组装向量/打分/评测 {t['scoring_evaluation_seconds']:.2f} 秒。API 单请求时间累计 {t['api_attempt_seconds_sum']:.2f} 秒与并发墙钟不同。", "",
        "| 项目 | 数值 |", "|---|---:|", f"| 编码文本 | {sum(c['text_counts'].values())} |",
        f"| 成功逻辑批次 | {retry['logical_batches']} |", f"| API 实际尝试 | {u['api_attempts']} |",
        f"| 失败后追加尝试 | {retry['extra_network_attempts']} |", f"| 缓存命中 | {u['cache_hits']} |",
        f"| 输入 Token（接口返回） | {u['known_prompt_tokens']} |", f"| 总 Token（接口返回） | {u['known_total_tokens']} |",
        "| LLM 调用 | 0 |", "", "API 状态：`" + json.dumps(retry, ensure_ascii=False) + "`。",
        f"缺少 usage 的成功响应：{u['responses_without_usage']}。接口失败且未返回 usage 的尝试无法计入 Token，Embedding 不生成文本输出。",
        "向量审计：`" + json.dumps(c["vector_audit"], ensure_ascii=False) + "`。", "",
        "Top-3/5/10、召回率和原生命中均经 pytrec_eval 复核；NDCG@10 是场景平均。核验：`" + json.dumps(c["verification"], ensure_ascii=False) + "`。BM25 完整排名、分数及指标与前轮完全一致；所有方法使用同一场景候选库。", "",
        "局限：本实验只隔离 Embedding 底座更换，240 条探针固定不变，不能消除既有探针误差；API 模型未返回不可变权重版本，记录模型名、响应模型和向量哈希以支持重算。扩增金标包含模型复核目标，并非全量人工金标。", "",
        f"产物目录：`{output}`；原实验：`{c['source']}`。`protocol.json` 和 `source_manifest.json` 固定协议与原文件哈希；`embedding_cache.sqlite3` 保存原始响应，`batches/` 和 `vectors/` 保存向量，`graphs/` 保存 Qwen 图，`arms/` 保存全排名，`comparison.json` 和 `per_query_ndcg.json` 保存指标及配对数据。", ""]
    report = "\n".join(lines)
    (output / "report.zh.md").write_text(report, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    settings = replace(Settings.from_env(), cache_path=output / "embedding_cache.sqlite3",
                       llm_base_url="", llm_api_key="", llm_model="")
    settings.require(embedding=True)
    dimensions = configured_dimensions()
    assert "qwen" in settings.embedding_model.lower(), "The configured embedding model is not Qwen"
    if not args.resume:
        prepare(source, output, settings, dimensions)
    if args.prepare_only:
        print("Prepared frozen protocol and new empty embedding cache.", flush=True)
        return
    import fcntl
    lock = (output / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output / "completion.json").exists(), "Experiment already complete"
    protocol = read(output / "protocol.json")
    assert protocol["script_sha256"] == sha256_file(Path(__file__))
    assert protocol["source"] == str(source)
    assert protocol["embedding"]["model"] == settings.embedding_model
    assert protocol["embedding"]["url"] == settings.embedding_identity()["url"]
    assert protocol["embedding"]["requested_dimensions"] == dimensions
    assert protocol["embedding"]["batch_size"] == settings.embedding_batch_size
    assert protocol["source_manifest_sha256"] == sha256_file(output / "source_manifest.json")
    assert ANCHORED_PROMPT_SHA256 == protocol["prompt_sha256"] == PROMPT_HASH and protocol["graph"] == GRAPH
    check_hashes(ROOT / "memory_retrieval", protocol["main_package_sha256"])
    check_hashes(source, read(output / "source_manifest.json"))
    check_hashes(output / "inputs", protocol["input_sha256"])
    assert sha256_file(output / "anchored_probes.json") == protocol["probes_sha256"]
    dataset = load_dataset(output / "inputs")
    assert Counter(q["source_ref"]["type"] for q in dataset.queries) == {"T1": 30, "T2": 30}
    assert len(dataset.memories) == 6673 and sum(len(target_ids(g)) for g in dataset.gold) == 340
    groups = text_groups(dataset, read(output / "anchored_probes.json"))
    batches = [(name, offset, texts[offset:offset + settings.embedding_batch_size]) for name, texts in groups.items()
               for offset in range(0, len(texts), settings.embedding_batch_size)]
    started = time.perf_counter()
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    records, failures = {}, {}
    def progress(status):
        value = {"status": status, "completed_batches": len(records), "total_batches": len(batches),
                 "failed_batches": len(failures), "elapsed_seconds": time.perf_counter() - started, "pid": os.getpid()}
        atomic_json(output / "status.json", value)
        print(json.dumps(value), flush=True)
    progress("embedding")
    embedding_started = time.perf_counter()
    # Validate the configured interface on the first real batch; cache it as part
    # of the experiment, with no extra test request or duplicated encoding.
    name, offset, texts = batches[0]
    try:
        records[(name, offset)] = encode_batch(name, offset, texts, settings, dimensions, output)
    except Exception as exc:
        failures[str((name, offset))] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(output / "failures" / (attempt + ".json"), failures)
        atomic_json(output / "attempts" / (attempt + ".json"), {
            "wall_seconds": time.perf_counter() - started,
            "embedding_seconds": time.perf_counter() - embedding_started})
        progress("failed")
        raise
    progress("embedding")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(encode_batch, name, offset, texts, settings, dimensions, output): (name, offset)
                   for name, offset, texts in batches[1:]}
        for future in as_completed(futures):
            key = futures[future]
            try:
                records[key] = future.result()
            except Exception as exc:
                failures[str(key)] = {"type": type(exc).__name__, "message": str(exc)}
            progress("embedding")
    embedding_seconds = time.perf_counter() - embedding_started
    if failures:
        atomic_json(output / "failures" / (attempt + ".json"), failures)
        atomic_json(output / "attempts" / (attempt + ".json"), {"wall_seconds": time.perf_counter() - started,
                                                                "embedding_seconds": embedding_seconds})
        progress("failed")
        raise RuntimeError("Some batches failed; cached successful batches retained for --resume")
    progress("scoring_evaluation")
    begin = time.perf_counter()
    assert len({r["dimensions"] for r in records.values()}) == 1
    actual_dimension = next(iter(records.values()))["dimensions"]
    vectors, hashes = {}, {}
    for name, texts in groups.items():
        vectors[name] = np.concatenate([np.load(output / "batches" / f"{name}-{offset:06d}.npy")
                                       for offset in range(0, len(texts), settings.embedding_batch_size)], axis=0)
        assert vectors[name].shape == (len(texts), actual_dimension)
        atomic_npy(output / "vectors" / (name + ".npy"), vectors[name])
        hashes[name] = sha256_file(output / "vectors" / (name + ".npy"))
    embedding = {**protocol["embedding"], "actual_dimensions": actual_dimension,
                 "response_models": sorted({r["response_model"] for r in records.values() if r["response_model"]}),
                 "vector_sha256": hashes}
    atomic_json(output / "vectors" / "identity.json", embedding)
    rows = score_all(output, dataset, vectors)
    original_bm25 = {r["query_id"]: r for r in read_jsonl(source / "arms/bm25.jsonl")}
    assert rows["bm25"] == original_bm25, "BM25 ranks or scores changed"
    metrics, per_query, verification = evaluate(dataset, rows)
    old = read(source / "comparison.json")
    for name in ("T1", "T2", "all"):
        assert metrics[name]["methods"]["bm25"] == old["groups"][name]["methods"]["bm25"]
    statistics = paired_statistics(dataset.queries, per_query)
    backbone = {}
    for name in ("T1", "T2", "all"):
        backbone[name] = {}
        for label, source_metrics in (("BGE", old["groups"]), ("Qwen", metrics)):
            g = source_metrics[name]
            d, f = g["methods"]["dense"], g["methods"][FULL]
            backbone[name][label] = {"targets": g["gold_pairs"], "scenes": g["queries"],
                "dense_hits10": d["at_k"]["10"]["hits"], "dense_recall10": d["at_k"]["10"]["micro_recall"], "dense_ndcg10": d["ndcg_at_10"],
                "full_hits10": f["at_k"]["10"]["hits"], "full_recall10": f["at_k"]["10"]["micro_recall"], "full_ndcg10": f["ndcg_at_10"],
                "full_native_hits10": f["at_k"]["10"]["recall_by_source"]["stale_native"]["hits"]}
    events = [e for p in sorted((output / "api_events").glob("*.json")) for e in read(p)]
    assert all(e["stage"].startswith("qwen_embedding_") for e in events)
    usage = usage_summary(events)
    previous = [read(p) for p in (output / "attempts").glob("*.json")] if (output / "attempts").exists() else []
    checks = {name: {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "finite": bool(np.isfinite(matrix).all()),
                    "l2_max_error": float(np.max(np.abs(np.linalg.norm(matrix, axis=1)-1)))} for name, matrix in vectors.items()}
    assert all(v["finite"] and v["l2_max_error"] < 1e-6 for v in checks.values())
    check_hashes(source, read(output / "source_manifest.json"))
    check_hashes(ROOT / "memory_retrieval", protocol["main_package_sha256"])
    verification.update(bm25_ranks_scores_metrics_exact=True, source_directory_unchanged=True,
                        main_package_unchanged=True, probes_reused_exactly=True, llm_calls=0)
    successful = sum(e["status"] == "ok" and e["cached"] is False for e in events)
    comparison = {"source": str(source), "groups": metrics, "per_query_ndcg": per_query, "significance": statistics,
        "backbone_comparison": backbone, "embedding": embedding, "protocol": protocol, "usage": usage,
        "text_counts": {name: len(texts) for name, texts in groups.items()}, "vector_audit": checks, "llm_calls": 0,
        "api_retry_stats": {"logical_batches": len(batches), "status_counts": dict(Counter(e["status"] for e in events)),
            "http_status_counts": dict(Counter(str(e.get("http_status")) for e in events if e["status"] == "http_error")),
            "extra_network_attempts": usage["api_attempts"] - successful,
            "successful_requests_after_retry": sum(e["status"] == "ok" and e.get("attempt", 1) > 1 for e in events),
            "max_recorded_attempt": max((e.get("attempt", 1) for e in events), default=0)},
        "timing": {"embedding_wall_seconds": embedding_seconds + sum(p["embedding_seconds"] for p in previous),
            "scoring_evaluation_seconds": time.perf_counter() - begin,
            "total_wall_seconds": time.perf_counter() - started + sum(p["wall_seconds"] for p in previous),
            "api_attempt_seconds_sum": sum(e["seconds"] for e in events if e["cached"] is False)}, "verification": verification}
    for name, value in (("comparison.json", comparison), ("per_query_ndcg.json", per_query), ("verification.json", verification)):
        atomic_json(output / name, value)
    write_report(output, comparison)
    atomic_json(output / "completion.json", {"status": "complete", "comparison_sha256": sha256_file(output / "comparison.json"),
                "report_sha256": sha256_file(REPORT), "llm_calls": 0, "vectors_sha256": hashes})
    progress("complete")


if __name__ == "__main__":
    main()

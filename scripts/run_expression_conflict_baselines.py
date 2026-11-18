"""Run existing BM25, Dense and Hybrid on the fixed expression-conflict set."""
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory

DATA = ROOT.parent / "research_notes/expression_conflict_dataset_2026-11-17"
OUT = ROOT / "runs/experiments/expression-conflict-fixed-20261117"
METHODS = ("bm25", "dense", "hybrid")
KS = (1, 5, 10)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate():
    manifest = read_json(DATA / "manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if sha(DATA / name) != expected:
            raise ValueError(f"数据哈希不一致：{name}")
    queries = read_rows(DATA / "queries.jsonl")
    memories = read_rows(DATA / "memories.jsonl")
    pairs = read_rows(DATA / "pairs.jsonl")
    golds = read_rows(DATA / "qrels.jsonl")
    qmap = {r["query_id"]: r for r in queries}
    mmap = {r["memory_id"]: r for r in memories}
    pmap = {r["query_id"]: r for r in pairs}
    gmap = {r["query_id"]: r for r in golds}
    if (len(qmap) != len(queries) or len(mmap) != len(memories)
            or len(pmap) != len(pairs) or len(gmap) != len(golds)):
        raise ValueError("存在重复 ID")
    if not (len(queries) == len(pairs) == len(golds) == manifest["query_count"] == 12
            and len(memories) == manifest["shared_corpus_records"] == 1000
            and set(qmap) == set(pmap) == set(gmap)):
        raise ValueError("查询、候选或金标数量不符合数据清单")
    if (len({m["content"] for m in memories}) != len(memories)
            or {m["content"] for m in memories} & {q["content"] for q in queries}):
        raise ValueError("候选正文重复或含有查询正文")
    for qid, query in qmap.items():
        label, pair = gmap[qid], pmap[qid]
        mid = label["memory_id"]
        if (label["label"] != "contradiction" or mid not in mmap
                or pair["memory_id"] != mid or pair["new_memory"] != query["content"]
                or pair["conflicting_old_memory"] != mmap[mid]["content"]):
            raise ValueError(f"{qid} 的目标引用不一致")
    if len({r["memory_id"] for r in golds}) != len(golds):
        raise ValueError("不同查询的目标 ID 重复")
    return manifest, queries, memories, pmap, gmap


def render(summary, pairs, details):
    lines = ["# 表达不同的冲突记忆：现有方法测试", "",
             "12 条查询共用同一份 1,000 条候选库；每条查询有一个已确认冲突目标。", "",
             "模型只接收查询和候选原文，金标仅在排序完成后用于计算命中。", "",
             f"Dense 模型配置：{summary['embedding']['model']}，实际向量维度：{summary['embedding_dimension']}。", "",
             "| 方法 | 已知目标命中@1 | 已知目标命中@5 | 已知目标命中@10 | 目标平均名次 |",
             "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        data = summary["metrics"][method]
        hits = data["hit_counts"]
        lines.append(f"| {method} | {hits['1']}/12 | {hits['5']}/12 | {hits['10']}/12 | {data['mean_rank']:.2f} |")
    by_key = {(r["query_id"], r["method"]): r for r in details}
    lines += ["", "## 每条目标排名", "", "| 编号 | BM25 | Dense | Hybrid |",
              "|---|---:|---:|---:|"]
    for qid in sorted(pairs):
        lines.append(f"| {qid} | " + " | ".join(str(by_key[(qid, m)]["target_rank"]) for m in METHODS) + " |")
    lines += ["", "## 逐条记忆", ""]
    for qid in sorted(pairs):
        p = pairs[qid]
        lines += [f"### {qid}", "", f"新记忆：{p['new_memory_zh']}", "",
                  f"冲突旧记忆：{p['old_memory_zh']}", "",
                  "目标排名：" + "；".join(f"{m} 第 {by_key[(qid, m)]['target_rank']} 名" for m in METHODS), ""]
    lines += ["## 结果边界", "",
              "- 这里只统计已确认目标是否进入前 K 条。未列入金标的其余候选并未逐对审核，不能将其一律算成非冲突。",
              "- 冲突对和 100 条个人记忆由助手构造和审核，未独立人工复核；另有 888 条公开资料摘录作为背景。",
              "- 这是一份 12 组的合成探索数据；结果不代表自然记忆中的总体失败比例。", ""]
    return "\n".join(lines)


def main():
    manifest, queries, memories, pairs, golds = validate()
    if (OUT / "summary.json").exists():
        raise ValueError("结果已经存在，不覆盖已完成的实验。")
    settings = Settings.from_env()
    settings.require(embedding=True)
    OUT.mkdir(parents=True, exist_ok=True)
    settings = replace(settings, cache_path=OUT / "embedding_cache.sqlite3")
    inputs = {name: sha(DATA / name) for name in ("manifest.json", "queries.jsonl", "memories.jsonl", "pairs.jsonl", "qrels.jsonl")}
    plan = {"data_hashes": inputs, "embedding": settings.embedding_identity(),
            "methods": list(METHODS), "top_k": list(KS), "embedding_batch_size": settings.embedding_batch_size}
    if (OUT / "run_plan.json").exists():
        if read_json(OUT / "run_plan.json") != plan:
            raise ValueError("续跑配置与原实验不一致")
    else:
        write_json(OUT / "run_plan.json", plan)
    client = APIClient(settings)
    # Each exact text is encoded once. The fixed data has unique corpus and query texts.
    texts = [m["content"] for m in memories] + [q["content"] for q in queries]
    if len(set(texts)) != len(texts):
        raise ValueError("输入正文重复")
    vectors = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start:start + settings.embedding_batch_size]
        vector = client.embed(batch, "expression_conflict_embedding")
        if vectors and vector.shape[1] != vectors[0].shape[1]:
            raise ValueError("向量服务在同一次实验中返回不同维度")
        vectors.append(vector)
        write_json(OUT / "api_events.json", client.events)
        print(f"embedding {start + len(batch)}/{len(texts)} dim={vector.shape[1]}", flush=True)
    matrix = np.concatenate(vectors)
    if matrix.shape[0] != 1012 or not np.isfinite(matrix).all():
        raise ValueError("向量数量或取值异常")
    np.savez_compressed(OUT / "embeddings.npz", vectors=matrix)
    memory_vectors, query_vectors = matrix[:len(memories)], matrix[len(memories):]
    objects = [Memory(m["memory_id"], m["content"], {}) for m in memories]
    sparse_index = BM25([m["content"] for m in memories])
    details, rankings = [], []
    for qi, query in enumerate(queries):
        sparse = sparse_index.score(query["content"])
        dense = memory_vectors @ query_vectors[qi]
        fused = reciprocal_rank_fusion(sparse, dense, objects)
        for method, scores in zip(METHODS, (sparse, dense, fused)):
            if not np.isfinite(scores).all():
                raise ValueError(f"{query['query_id']} 的 {method} 分数非有限数值")
            order = rank(scores, objects)
            ranked = [{"rank": pos, "memory_id": objects[idx].memory_id, "score": float(scores[idx])}
                      for pos, idx in enumerate(order, 1)]
            if len(ranked) != len(memories) or len({r["memory_id"] for r in ranked}) != len(memories):
                raise ValueError("排序不完整")
            target = next(r for r in ranked if r["memory_id"] == golds[query["query_id"]]["memory_id"])
            details.append({"query_id": query["query_id"], "method": method,
                            "target_id": target["memory_id"], "target_rank": target["rank"],
                            "target_score": target["score"],
                            "target_tie_ranks": [r["rank"] for r in ranked if r["score"] == target["score"]],
                            "top10": ranked[:10]})
            rankings.append({"query_id": query["query_id"], "method": method, "ranked_candidates": ranked})
    metrics = {}
    for method in METHODS:
        ranks = [r["target_rank"] for r in details if r["method"] == method]
        metrics[method] = {"query_count": len(ranks),
                           "hit_counts": {str(k): sum(x <= k for x in ranks) for k in KS},
                           "hit_rates": {str(k): sum(x <= k for x in ranks)/len(ranks) for k in KS},
                           "mean_rank": statistics.mean(ranks), "median_rank": statistics.median(ranks)}
    summary = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_path": str(DATA),
               "input_hashes": inputs, "query_count": len(queries), "candidate_count": len(memories),
               "known_conflict_pairs": len(golds), "methods": list(METHODS), "top_k_values": list(KS),
               "embedding": settings.embedding_identity(), "embedding_dimension": int(matrix.shape[1]),
               "embedding_usage": usage_summary(client.events),
               "bm25": {"tokenizer": "memory_retrieval.baselines.tokenize", "k1": 1.2, "b": 0.75},
               "hybrid": {"method": "reciprocal rank fusion", "constant": 60},
               "script_sha256": sha(Path(__file__)), "metrics": metrics,
               "limitations": ["assistant-authored and reviewed synthetic contradiction pairs; not independently human reviewed",
                               "one known target per query; other query-candidate pairs unjudged",
                               "candidate corpus includes 888 reference excerpts and 100 fictional personal memories"]}
    (OUT / "per_query.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in details), encoding="utf-8")
    with gzip.open(OUT / "rankings.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rankings:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "report.zh.md").write_text(render(summary, pairs, details), encoding="utf-8")
    write_json(OUT / "summary.json", summary)
    print(json.dumps({"output": str(OUT), "embedding_dimension": summary["embedding_dimension"],
                      "embedding_usage": summary["embedding_usage"], "metrics": metrics},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

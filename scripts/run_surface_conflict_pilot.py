"""Run the existing BM25, Dense, and RRF baselines on the reviewed pilot."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATASET = WORKSPACE / "research_notes" / "dataset_fit_check_2026-11-16" / "pilot_10_surface_conflict"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory, canonical_json, fingerprint


DATA_FILES = ("queries.jsonl", "memories.jsonl", "pair_labels.assistant_reviewed.jsonl")
TOP_KS = (1, 3, 5, 10)
METHODS = ("bm25", "dense", "hybrid")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def validate_inputs(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    queries = read_jsonl(dataset / "queries.jsonl")
    memories = read_jsonl(dataset / "memories.jsonl")
    labels = read_jsonl(dataset / "pair_labels.assistant_reviewed.jsonl")
    query_map = {row["query_id"]: row for row in queries}
    memory_map = {row["memory_id"]: row for row in memories}
    label_map = {(row["query_id"], row["memory_id"]): row for row in labels}
    if len(query_map) != len(queries) or len(memory_map) != len(memories):
        raise ValueError("查询或记忆 ID 不唯一")
    if len(label_map) != len(labels):
        raise ValueError("查询-记忆标签对存在重复")
    if len(queries) != manifest["queries"] or len(memories) != manifest["corpus_records"]:
        raise ValueError("数据行数与 manifest 不一致")
    if len(labels) != len(queries) * len(memories):
        raise ValueError("标签没有覆盖全部查询-记忆组合")
    for name in DATA_FILES:
        expected = manifest["artifact_sha256"].get(name)
        if expected and sha256(dataset / name) != expected:
            raise ValueError(f"{name} 与 manifest 哈希不一致")
    for query in queries:
        positives = [row for row in labels
                     if row["query_id"] == query["query_id"] and row["label"] == "conflict"]
        if len(positives) != 1:
            raise ValueError(f"{query['query_id']} 必须恰有一个冲突目标")
        if query["query_id"] in memory_map:
            raise ValueError(f"查询 {query['query_id']} 被放进候选库")
    if any((q["query_id"], m["memory_id"]) not in label_map
           for q in queries for m in memories):
        raise ValueError("标签引用缺失")
    return manifest, queries, memories, label_map


def evaluate(rankings, queries, labels):
    report = {}
    query_map = {row["query_id"]: row for row in queries}
    for method in METHODS:
        rows = [row for row in rankings if row["method"] == method]
        per_query = []
        for row in rows:
            qid = row["query_id"]
            ordered = row["ranked_candidates"]
            positions = {candidate["memory_id"]: candidate["rank"] for candidate in ordered}
            gold_id = next(memory_id for (query_id, memory_id), label in labels.items()
                           if query_id == qid and label["label"] == "conflict")
            compatible_ids = [memory_id for (query_id, memory_id), label in labels.items()
                              if query_id == qid and label.get("negative_type") == "same_group_compatible"]
            gold_rank = positions[gold_id]
            compatible_ranks = sorted(positions[memory_id] for memory_id in compatible_ids)
            per_query.append({
                "query_id": qid,
                "source_group": query_map[qid]["source_group"],
                "conflict_target_id": gold_id,
                "conflict_rank": gold_rank,
                "compatible_ranks": compatible_ranks,
                "compatible_ahead_of_conflict": sum(value < gold_rank for value in compatible_ranks),
                "top1_id": ordered[0]["memory_id"],
            })
        report[method] = {
            "query_count": len(per_query),
            "target_recall_at_k": {
                str(k): sum(row["conflict_rank"] <= k for row in per_query) / len(per_query)
                for k in TOP_KS
            },
            "target_rank_mean": statistics.mean(row["conflict_rank"] for row in per_query),
            "target_rank_median": statistics.median(row["conflict_rank"] for row in per_query),
            "compatible_paraphrase_before_target_queries": sum(
                row["compatible_ahead_of_conflict"] > 0 for row in per_query),
            "compatible_paraphrase_before_target_count": sum(
                row["compatible_ahead_of_conflict"] for row in per_query),
            "mean_compatible_paraphrases_in_top_k": {
                str(k): statistics.mean(
                    sum(rank_value <= k for rank_value in row["compatible_ranks"])
                    for row in per_query)
                for k in TOP_KS
            },
            "per_query": per_query,
        }
    return report


def render_markdown(summary):
    lines = [
        "# 表面差异冲突检索：Baseline 结果",
        "",
        f"运行时间（UTC）：{summary['run_at_utc']}",
        "",
        "## 设置",
        "",
        f"- 数据：{summary['dataset_path']}",
        f"- 查询 / 候选 / 配对标签：{summary['counts']['queries']} / "
        f"{summary['counts']['memories']} / {summary['counts']['pair_labels']}",
        f"- Dense：{summary['embedding']['model']}，原文向量余弦排序",
        "- BM25：复用项目原生 tokenizer 和 BM25 参数；Hybrid：BM25/Dense 的 RRF（常数 60）",
        "- 三种方法均对完整 28 条候选排序；未调用 LLM 判定器，标签未进入检索输入。",
        "",
        "## 总体指标",
        "",
        "| 方法 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | 目标平均名次 | 相容改述排在目标前的查询数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = summary["metrics"][method]
        recalls = values["target_recall_at_k"]
        lines.append(
            f"| {method} | {recalls['1']:.2f} | {recalls['3']:.2f} | {recalls['5']:.2f} | "
            f"{recalls['10']:.2f} | {values['target_rank_mean']:.1f} | "
            f"{values['compatible_paraphrase_before_target_queries']}/"
            f"{values['query_count']} |"
        )
    lines += [
        "",
        "## 逐查询目标排名",
        "",
        "| 来源组 | BM25 | Dense | Hybrid |",
        "|---|---:|---:|---:|",
    ]
    per_method = {name: {row["source_group"]: row for row in summary["metrics"][name]["per_query"]}
                  for name in METHODS}
    for group in sorted(per_method["bm25"]):
        entries = []
        for method in METHODS:
            row = per_method[method][group]
            comp = ",".join(map(str, row["compatible_ranks"])) or "-"
            entries.append(f"目标 #{row['conflict_rank']}（相容 #{comp}）")
        lines.append(f"| {group} | {entries[0]} | {entries[1]} | {entries[2]} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 这是 10 条人工挑选的探索性诊断，不是随机样本；不能估计真实数据中的失败率。",
        "- top-k 中大量非目标项是候选库构成带来的预期结果；应重点看冲突目标名次及同组相容改述是否排在其前面。",
        "- 这组数据源自 GPT-4 生成的 MSMARCO 段落；部分实体仍用匿名占位符；标签未经独立人工复核。",
        "- 4204814 和 225591 的语义标注存在前序审核指出的疑点；解释对应行时应暂缓作为确定性证据。",
        "",
        f"Embedding API：{summary['embedding_usage']['api_attempts']} 次请求，缓存命中 "
        f"{summary['embedding_usage']['cache_hits']} 次。",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在：{output}")
    manifest, queries, memories, labels = validate_inputs(dataset)
    settings = Settings.from_env(args.env)
    settings.require(embedding=True)
    output.mkdir(parents=True)
    client = APIClient(replace(settings, cache_path=output / "embedding_cache.sqlite3"))

    contents = [memory["content"] for memory in memories]
    query_contents = [query["content"] for query in queries]
    texts = list(dict.fromkeys(contents + query_contents))
    vectors = client.embed(texts, "surface_conflict_pilot_embedding")
    vector_by_text = {text: vectors[index] for index, text in enumerate(texts)}
    memory_objects = [Memory(row["memory_id"], row["content"], {}) for row in memories]
    bm25 = BM25(contents)
    memory_matrix = np.stack([vector_by_text[text] for text in contents])
    rankings = []
    for query in queries:
        sparse_scores = bm25.score(query["content"])
        dense_scores = memory_matrix @ vector_by_text[query["content"]]
        hybrid_scores = reciprocal_rank_fusion(sparse_scores, dense_scores, memory_objects)
        scores_by_method = {"bm25": sparse_scores, "dense": dense_scores, "hybrid": hybrid_scores}
        for method, scores in scores_by_method.items():
            order = rank(scores, memory_objects)
            ranked_candidates = []
            for position, index in enumerate(order, 1):
                memory = memories[index]
                label = labels[(query["query_id"], memory["memory_id"])]
                ranked_candidates.append({
                    "rank": position,
                    "memory_id": memory["memory_id"],
                    "source_group": memory["source_group"],
                    "score": float(scores[index]),
                    "label": label["label"],
                    "negative_type": label.get("negative_type"),
                })
            if len(ranked_candidates) != len(memories) or any(
                    not math.isfinite(row["score"]) for row in ranked_candidates):
                raise ValueError(f"{query['query_id']}/{method} 排名不完整或分数非有限")
            rankings.append({"query_id": query["query_id"], "source_group": query["source_group"],
                             "method": method, "ranked_candidates": ranked_candidates})

    metrics = evaluate(rankings, queries, labels)
    input_hashes = {name: sha256(dataset / name) for name in DATA_FILES}
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "dataset_input_sha256": input_hashes,
        "dataset_signature": fingerprint(input_hashes),
        "counts": {"queries": len(queries), "memories": len(memories),
                   "pair_labels": len(labels)},
        "methods": list(METHODS),
        "top_k_values": list(TOP_KS),
        "embedding": settings.embedding_identity(),
        "embedding_usage": usage_summary(client.events),
        "embedding_api_events": client.events,
        "bm25": {"tokenizer": "memory_retrieval.baselines.tokenize",
                 "k1": 1.2, "b": 0.75},
        "hybrid": {"method": "reciprocal rank fusion", "constant": 60},
        "metrics": metrics,
        "limitations": [
            "purposeful exploratory subset, not a random sample",
            "assistant-reviewed labels are not independently human-reviewed",
            "source passages are GPT-4 generated from MSMARCO",
            "semantic concerns remain for source groups 4204814 and 225591",
        ],
    }
    (output / "rankings.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rankings), encoding="utf-8")
    write_json(output / "summary.json", result)
    (output / "report.zh.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": result["counts"],
                      "embedding": result["embedding"],
                      "embedding_usage": result["embedding_usage"],
                      "metrics": {method: {
                          "recall_at_k": metrics[method]["target_recall_at_k"],
                          "mean_rank": metrics[method]["target_rank_mean"],
                          "compatible_before_target_queries":
                              metrics[method]["compatible_paraphrase_before_target_queries"],
                      } for method in METHODS}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, help="项目环境文件；只需要 embedding 配置")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

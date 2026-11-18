"""Run label-blind BM25, Dense, and Hybrid rankings on the implicit pilot."""

import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_retrieval.api import APIClient, usage_summary
from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory


METHODS = ("bm25", "dense", "hybrid")
TOP_KS = (1, 3, 5, 10, 20)
DEFAULT_QUERIES = Path.home() / "Downloads/single_pair_implicit_queries (1).jsonl"
DEFAULT_MEMORIES = Path.home() / "Downloads/single_pair_memory_bank (1).jsonl"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def validate(queries_path, memories_path):
    queries = read_jsonl(queries_path)
    memories = read_jsonl(memories_path)
    query_ids = [row["query_id"] for row in queries]
    memory_ids = [row["memory_id"] for row in memories]
    if len(query_ids) != len(set(query_ids)) or len(memory_ids) != len(set(memory_ids)):
        raise ValueError("查询 ID 或记忆 ID 重复")
    memory_map = {row["memory_id"]: row for row in memories}
    query_texts = [row["query_text"] for row in queries]
    memory_texts = [row["text"] for row in memories]
    if len(set(query_texts + memory_texts)) != len(query_texts) + len(memory_texts):
        raise ValueError("查询或记忆正文重复，无法唯一映射向量")
    for query in queries:
        target_id = query["gold_memory_id"]
        if target_id not in memory_map:
            raise ValueError(f"{query['query_id']} 的 gold_memory_id 不在记忆库")
        if query["gold_memory_text"] != memory_map[target_id]["text"]:
            raise ValueError(f"{query['query_id']} 的 gold_memory_text 与目标记忆不一致")
        distractors = query["distractor_ids"]
        if len(distractors) != len(set(distractors)) or target_id in distractors:
            raise ValueError(f"{query['query_id']} 的干扰 ID 重复或包含目标")
        if set(distractors) - memory_map.keys():
            raise ValueError(f"{query['query_id']} 引用了不存在的干扰记忆")
    return queries, memories


def evaluate(per_query):
    result = {}
    for method in METHODS:
        rows = [row for row in per_query if row["method"] == method]
        ranks = [row["target_rank"] for row in rows]
        result[method] = {
            "query_count": len(ranks),
            "known_target_hit_counts": {
                str(k): sum(value <= k for value in ranks) for k in TOP_KS
            },
            "known_target_hit_rates": {
                str(k): sum(value <= k for value in ranks) / len(ranks) for k in TOP_KS
            },
            "target_rank_mean": statistics.mean(ranks),
            "target_rank_median": statistics.median(ranks),
            "known_distractors_ahead_mean": statistics.mean(
                row["known_distractors_ahead"] for row in rows
            ),
        }
    return result


def render_report(summary, per_query, queries):
    count = summary["counts"]["queries"]
    lines = [
        "# 单条隐式冲突：现有基线探索结果",
        "",
        f"运行时间（UTC）：{summary['run_at_utc']}",
        "",
        "## 设置",
        "",
        f"- 查询 / 共享旧库：{count} / {summary['counts']['memories']} 条。",
        f"- Dense：{summary['embedding']['model']}；BM25 与 Hybrid 使用项目现有实现。",
        "- 三个方法对每条查询完整排序同一份旧库；输入只有 query_text 和 memory text。",
        "- 标签只用于排序完成后的指定目标排名统计；未运行冲突判定器。",
        "",
        "## 指定目标命中",
        "",
        "| 方法 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | 目标平均名次 | 已知相容干扰排前均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = summary["metrics"][method]
        hits = metrics["known_target_hit_counts"]
        lines.append(
            f"| {method} | {hits['1']}/{count} | {hits['3']}/{count} | "
            f"{hits['5']}/{count} | {hits['10']}/{count} | {hits['20']}/{count} | "
            f"{metrics['target_rank_mean']:.2f} | "
            f"{metrics['known_distractors_ahead_mean']:.2f} |"
        )
    lookup = {(row["query_id"], row["method"]): row for row in per_query}
    lines += ["", "## 逐查询目标名次", "", "| 查询 | 类别 | BM25 | Dense | Hybrid |",
              "|---|---|---:|---:|---:|"]
    for query in queries:
        qid = query["query_id"]
        ranks = [lookup[(qid, method)]["target_rank"] for method in METHODS]
        lines.append(f"| {qid} | {query['category']} | " + " | ".join(map(str, ranks)) + " |")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 本轮只衡量每条查询文件中指定的一个目标是否进入 Top-K，是 target Hit@K，不是全库完整冲突召回率。",
        "- 全库标签并不完整：例如 q_bs_01 与 m_0014、q_bs_05 与 m_0118 还存在未列出的冲突关系。",
        "- 因此不计算候选纯度、完整召回率或误报率；`cross_bank_audited` 布尔字段不替代逐对审计记录。",
        "- 数据是 15 条构造查询的初步诊断，不用于估计真实记忆流中的失败率。",
        "",
        f"Embedding API：{summary['embedding_usage']['api_attempts']} 次请求，缓存命中 "
        f"{summary['embedding_usage']['cache_hits']} 次。",
        "",
    ]
    return "\n".join(lines)


def run(args):
    queries_path = args.queries.resolve()
    memories_path = args.memories.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"输出目录已存在，避免覆盖：{output}")
    queries, memories = validate(queries_path, memories_path)
    settings = Settings.from_env(args.env)
    settings.require(embedding=True)
    output.mkdir(parents=True)
    client = APIClient(replace(settings, cache_path=output / "embedding_cache.sqlite3"))

    # Deliberately whitelist raw text fields so annotation metadata cannot leak.
    texts = [row["text"] for row in memories] + [row["query_text"] for row in queries]
    vectors = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start:start + settings.embedding_batch_size]
        vector = client.embed(batch, "single_pair_implicit_embedding")
        if vectors and vector.shape[1] != vectors[0].shape[1]:
            raise ValueError("向量服务在同一次实验中返回不同维度")
        vectors.append(vector)
        write_json(output / "api_events.json", client.events)
        print(f"embedding {min(start + len(batch), len(texts))}/{len(texts)} "
              f"dim={vector.shape[1]}", flush=True)
    matrix = np.concatenate(vectors)
    if matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise ValueError("向量数量或取值异常")
    np.savez_compressed(output / "embeddings.npz", vectors=matrix)
    write_json(output / "embedding_text_order.json", texts)

    memory_count = len(memories)
    memory_vectors, query_vectors = matrix[:memory_count], matrix[memory_count:]
    docs = [Memory(row["memory_id"], row["text"], {}) for row in memories]
    bm25 = BM25([row["text"] for row in memories])
    per_query, rankings = [], []
    for query_index, query in enumerate(queries):
        sparse = bm25.score(query["query_text"])
        dense = memory_vectors @ query_vectors[query_index]
        hybrid = reciprocal_rank_fusion(sparse, dense, docs)
        score_sets = dict(zip(METHODS, (sparse, dense, hybrid)))
        for method, scores in score_sets.items():
            ordered_indices = rank(scores, docs)
            ordered = [{
                "rank": position,
                "memory_id": docs[index].memory_id,
                "score": float(scores[index]),
            } for position, index in enumerate(ordered_indices, 1)]
            if len(ordered) != memory_count or not np.isfinite(scores).all():
                raise ValueError(f"{query['query_id']}/{method} 排名不完整或分数非有限")
            positions = {row["memory_id"]: row["rank"] for row in ordered}
            target_rank = positions[query["gold_memory_id"]]
            distractor_ranks = [positions[mid] for mid in query["distractor_ids"]]
            per_query.append({
                "query_id": query["query_id"],
                "method": method,
                "target_id": query["gold_memory_id"],
                "target_rank": target_rank,
                "target_score": ordered[target_rank - 1]["score"],
                "known_distractors_ahead": sum(value < target_rank for value in distractor_ranks),
                "known_distractor_ranks": distractor_ranks,
                "top10": ordered[:10],
            })
            rankings.append({"query_id": query["query_id"], "method": method,
                             "ranked_candidates": ordered})

    input_hashes = {"queries": sha256(queries_path), "memories": sha256(memories_path)}
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {"queries": str(queries_path), "memories": str(memories_path)},
        "input_sha256": input_hashes,
        "counts": {"queries": len(queries), "memories": len(memories)},
        "methods": list(METHODS),
        "top_k_values": list(TOP_KS),
        "embedding": settings.embedding_identity(),
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_usage": usage_summary(client.events),
        "bm25": {"tokenizer": "memory_retrieval.baselines.tokenize", "k1": 1.2, "b": 0.75},
        "hybrid": {"method": "reciprocal rank fusion", "constant": 60},
        "metrics": evaluate(per_query),
        "limitations": [
            "exploratory constructed pilot with 15 queries",
            "only the designated target per query is scored; full-bank pair labels are incomplete",
            "no candidate purity, full conflict recall, or false-positive rate is reported",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "per_query.json", per_query)
    with gzip.open(output / "rankings.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rankings:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "report.zh.md").write_text(render_report(summary, per_query, queries),
                                         encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "embedding": summary["embedding"],
        "embedding_usage": summary["embedding_usage"],
        "metrics": summary["metrics"],
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--memories", type=Path, default=DEFAULT_MEMORIES)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiments/single-pair-implicit-baseline-20261117")
    parser.add_argument("--env", type=Path, help="项目环境文件；只需要 embedding 配置")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

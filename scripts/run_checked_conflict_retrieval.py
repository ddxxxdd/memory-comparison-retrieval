"""Run existing baselines on the 12 reviewed pairs and their shared corpus.

Dense vectors are reused from the previous run, after checking the saved
vectors against that run's original embedding responses. No network is used.
"""

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_retrieval.baselines import BM25, rank, reciprocal_rank_fusion
from memory_retrieval.data import Memory


DATA = ROOT.parent / "research_notes/surface_form_conflict_only_2026-11-17"
PRIOR = ROOT / "runs/experiments/surface-conflict-diagnostic-20261117"
OUT = ROOT / "runs/experiments/surface-conflict-checked12-20261117"
KS = (1, 3, 5, 10, 20)
METHODS = ("bm25", "dense", "hybrid")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data():
    manifest = load_json(DATA / "manifest.json")
    audit = load_json(DATA / "fit_audit.json")
    assert sha(DATA / "manifest.json") == audit["input_manifest_sha256"]
    assert sha(DATA / "memories.jsonl") == manifest["artifact_sha256"]["memories.jsonl"]
    assert sha(DATA / "memories.jsonl") == audit["corpus"]["sha256"]
    for name, expected in audit["checked_artifact_sha256"].items():
        assert sha(DATA / name) == expected, name
    queries = load_jsonl(DATA / "queries.checked.jsonl")
    memories = load_jsonl(DATA / "memories.jsonl")
    qrels = load_jsonl(DATA / "qrels.checked.jsonl")
    pairs = load_jsonl(DATA / "pairs.checked.jsonl")
    query_by_id = {q["query_id"]: q for q in queries}
    memory_by_id = {m["memory_id"]: m for m in memories}
    gold_by_query = {g["query_id"]: g["memory_id"] for g in qrels}
    pair_by_query = {p["query_id"]: p for p in pairs}
    assert len(query_by_id) == len(queries) == len(qrels) == len(pairs) == 12
    assert len(memory_by_id) == len(memories) == 1000
    assert set(query_by_id) == set(gold_by_query) == set(pair_by_query) == set(audit["recommended_query_ids"])
    assert len({q["content"] for q in queries}) == 12
    assert len({m["content"] for m in memories}) == 1000
    assert not {q["content"] for q in queries} & {m["content"] for m in memories}
    for query_id, memory_id in gold_by_query.items():
        pair = pair_by_query[query_id]
        assert memory_id in memory_by_id and pair["memory_id"] == memory_id
        assert pair["new_memory"] == query_by_id[query_id]["content"]
        assert pair["conflicting_old_memory"] == memory_by_id[memory_id]["content"]
    return queries, memories, gold_by_query, pair_by_query


def load_vectors(required):
    summary = load_json(PRIOR / "summary.json")
    plan = load_json(PRIOR / "run_plan.json")
    assert summary["embedding"] == plan["embedding"]
    assert summary["embedding"]["model"] == "qwen3-embedding-8b"
    texts = load_json(PRIOR / "embedding_text_order.json")
    vectors = np.load(PRIOR / "embeddings.npz")["vectors"]
    assert vectors.ndim == 2 and vectors.shape == (len(texts), summary["embedding_dimension"])
    assert len(texts) == len(set(texts)) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    vec_by_text = dict(zip(texts, vectors))
    assert required <= vec_by_text.keys()

    # Verify all used vectors against the recorded response bodies for their
    # exact texts, not merely the NPZ file or the model alias.
    verified = set()
    uri = (PRIOR / "embedding_cache.sqlite3").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        for request_text, response_text in db.execute("SELECT request, response FROM responses"):
            request = json.loads(request_text)
            body = request["body"]
            assert request["url"] == summary["embedding"]["url"]
            assert body["model"] == summary["embedding"]["model"]
            inputs = body["input"]
            items = sorted(json.loads(response_text)["data"], key=lambda item: item["index"])
            assert [item["index"] for item in items] == list(range(len(inputs)))
            raw = np.asarray([item["embedding"] for item in items], dtype=np.float32)
            assert raw.shape == (len(inputs), vectors.shape[1]) and np.isfinite(raw).all()
            normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            for text, vec in zip(inputs, normalized):
                if text in required:
                    assert np.allclose(vec, vec_by_text[text], atol=1e-6), "cached embedding mismatch"
                    verified.add(text)
    assert verified == required
    provenance = {
        "model": summary["embedding"]["model"],
        "service_url": summary["embedding"]["url"],
        "dimensions": int(vectors.shape[1]),
        "source_run": str(PRIOR),
        "source_run_utc": summary["run_at_utc"],
        "source_vectors_sha256": sha(PRIOR / "embeddings.npz"),
        "source_text_order_sha256": sha(PRIOR / "embedding_text_order.json"),
        "source_cache_sha256": sha(PRIOR / "embedding_cache.sqlite3"),
        "used_vectors_checked_against_original_responses": len(verified),
        "new_embedding_api_requests": 0,
    }
    return vec_by_text, provenance


def render_report(summary, details, pairs):
    count = summary["query_count"]
    lines = ["# 表达不同的冲突记忆检索：12 条复核样例", "",
             "所有查询共用同一份 1,000 条候选。每条查询有 1 个已确认目标；未列出的配对未逐对标注。", "",
             "三种方法只使用查询和候选原文。Dense 复用前一轮同一 embedding 配置已保存的向量，本次没有发起新 API 请求。", "",
             f"Dense 模型别名：{summary['embedding']['model']}；向量维度：{summary['embedding']['dimensions']}。", "",
             "| 方法 | 命中@1 | 命中@3 | 命中@5 | 命中@10 | 平均名次 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = summary["metrics"][method]
        hits = item["known_target_hit_count"]
        lines.append(f"| {method} | {hits['1']}/{count} | {hits['3']}/{count} | {hits['5']}/{count} | {hits['10']}/{count} | {item['mean_rank']:.2f} |")
    lines += ["", "## 每组已确认目标的名次", "",
              "| 编号 | BM25 | Dense | Hybrid |", "|---|---:|---:|---:|"]
    lookup = {(row["query_id"], row["method"]): row for row in details}
    for query_id in sorted(pairs):
        lines.append(f"| {query_id} | " + " | ".join(str(lookup[(query_id, m)]["target_rank"]) for m in METHODS) + " |")
    lines += ["", "## 逐例内容和 Top-10 情况", ""]
    for query_id in sorted(pairs):
        p = pairs[query_id]
        lines.extend([f"### {query_id}", "", f"新记忆：{p['new_memory_zh_reading_aid']}", "",
                      f"已知冲突旧记忆：{p['old_memory_zh_reading_aid']}", "",
                      "目标排名：" + "；".join(f"{m} 第 {lookup[(query_id,m)]['target_rank']} 名" for m in METHODS), ""])
    lines += ["## 如何解释", "",
              "- Hit@K 是每条查询的已确认目标是否在前 K 条，并非整库所有冲突记录的完整召回率。",
              "- 候选背景未逐对审核，不能把非目标候选一律称为无关或非冲突，也不能计算完整精确率。",
              "- 12 对由助手构造及复审，尚未独立人工复核；部分事件名称和日期是明显共同锚点。",
              "- 这批查询曾在另一份候选库实验过，因此本轮不是未见过的独立测试。",
              "- 高命中只说明本批本库下已知目标没有相应遗漏；不能推论所有真实记忆也都容易检索。", ""]
    return "\n".join(lines)


def main():
    if OUT.exists():
        raise ValueError(f"结果目录已存在，避免覆盖：{OUT}")
    queries, memories, gold, pairs = load_data()
    text_set = {r["content"] for r in queries + memories}
    vectors, provenance = load_vectors(text_set)
    docs = [Memory(m["memory_id"], m["content"], {}) for m in memories]
    matrix = np.stack([vectors[m["content"]] for m in memories])
    bm25 = BM25([m["content"] for m in memories])
    details, rankings = [], []
    for query in queries:
        sparse = bm25.score(query["content"])
        dense = matrix @ vectors[query["content"]]
        combined = reciprocal_rank_fusion(sparse, dense, docs)
        for method, scores in zip(METHODS, (sparse, dense, combined)):
            assert np.isfinite(scores).all()
            order = rank(scores, docs)
            assert len(order) == len(set(order)) == 1000
            ordered = [{"rank": n, "memory_id": docs[i].memory_id, "score": float(scores[i])}
                       for n, i in enumerate(order, 1)]
            target_id = gold[query["query_id"]]
            target = next(row for row in ordered if row["memory_id"] == target_id)
            same_score = [row["rank"] for row in ordered if row["score"] == target["score"]]
            details.append({"query_id": query["query_id"], "method": method,
                            "target_id": target_id, "target_rank": target["rank"],
                            "target_score": target["score"], "target_tie_ranks": same_score,
                            "top10": ordered[:10]})
            rankings.append({"query_id": query["query_id"], "method": method,
                             "ranked_candidates": ordered})
    metrics = {}
    for method in METHODS:
        ranks = [r["target_rank"] for r in details if r["method"] == method]
        assert len(ranks) == 12
        metrics[method] = {
            "known_target_hit_count": {str(k): sum(r <= k for r in ranks) for k in KS},
            "known_target_hit_rate": {str(k): sum(r <= k for r in ranks)/len(ranks) for k in KS},
            "mean_rank": statistics.mean(ranks), "median_rank": statistics.median(ranks),
        }
    input_hashes = {name: sha(DATA / name) for name in ("queries.checked.jsonl", "memories.jsonl", "qrels.checked.jsonl", "pairs.checked.jsonl", "fit_audit.json", "manifest.json")}
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_path": str(DATA),
        "query_count": len(queries), "shared_candidate_count": len(memories),
        "methods": list(METHODS), "top_k_values": list(KS),
        "bm25_parameters": {"k1": 1.2, "b": 0.75, "tokenizer": "memory_retrieval.baselines.tokenize"},
        "hybrid": "reciprocal rank fusion, constant 60", "embedding": provenance,
        "input_sha256": input_hashes, "script_sha256": sha(Path(__file__)),
        "metrics": metrics,
        "limitations": ["12 assistant-authored and assistant-reviewed synthetic queries",
                        "previously evaluated in a different candidate bank, not an independent holdout",
                        "unlisted query-candidate pairs unjudged, so known-target Hit@K only",
                        "public backgrounds mostly unrelated to the named target entities; easy corpus risk"],
    }
    OUT.mkdir(parents=True)
    save_json(OUT / "summary.json", summary)
    (OUT / "per_query.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in details), encoding="utf-8")
    with gzip.open(OUT / "rankings.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rankings:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "report.zh.md").write_text(render_report(summary, details, pairs), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "query_count": 12, "candidate_count": 1000,
                      "embedding": provenance, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

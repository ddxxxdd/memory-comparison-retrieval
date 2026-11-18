"""Independently recompute the saved 12-query rankings from frozen inputs."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "research_notes/surface_form_conflict_only_2026-11-17"
SCENARIOS = ROOT.parent / "research_notes/surface_conflict_diagnostic_2026-11-17/scenarios.json"
PREV = ROOT / "runs/experiments/surface-conflict-diagnostic-20261117"
RUN = ROOT / "runs/experiments/surface-conflict-checked12-20261117"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text):
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.lower(), flags=re.UNICODE)


def main():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    for name, digest in summary["input_sha256"].items():
        assert sha(DATA / name) == digest, name
    assert sha(PREV / "embeddings.npz") == summary["embedding"]["source_vectors_sha256"]
    assert sha(PREV / "embedding_text_order.json") == summary["embedding"]["source_text_order_sha256"]
    queries = rows(DATA / "queries.checked.jsonl")
    docs = rows(DATA / "memories.jsonl")
    qrels = {r["query_id"]: r["memory_id"] for r in rows(DATA / "qrels.checked.jsonl")}
    pairs = {r["query_id"]: r for r in rows(DATA / "pairs.checked.jsonl")}
    entities = {r["id"]: r["entity"] for r in json.loads(SCENARIOS.read_text(encoding="utf-8"))}
    texts = json.loads((PREV / "embedding_text_order.json").read_text(encoding="utf-8"))
    vectors = np.load(PREV / "embeddings.npz")["vectors"]
    vector_map = dict(zip(texts, vectors))
    matrix = np.stack([vector_map[d["content"]] for d in docs])
    counters = [Counter(tokens(d["content"])) for d in docs]
    lens = [sum(c.values()) for c in counters]
    avg = sum(lens) / len(lens)
    df = Counter(word for c in counters for word in c)
    rankings = {}
    with gzip.open(RUN / "rankings.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["query_id"], row["method"])
            assert key not in rankings
            rankings[key] = row["ranked_candidates"]
    assert len(rankings) == 36
    details = {(r["query_id"], r["method"]): r for r in rows(RUN / "per_query.jsonl")}
    assert len(details) == 36
    ids = [d["memory_id"] for d in docs]
    entity_counts, ties = {}, []
    for q in queries:
        qid = q["query_id"]
        assert pairs[qid]["new_memory"] == q["content"]
        target = qrels[qid]
        entity_counts[qid] = sum(entities[qid].casefold() in d["content"].casefold() for d in docs)
        assert entity_counts[qid] == 1
        assert entities[qid].casefold() in q["content"].casefold()
        assert entities[qid].casefold() in next(d["content"] for d in docs if d["memory_id"] == target).casefold()
        words = set(tokens(q["content"]))
        sparse = np.zeros(len(docs))
        for word in words:
            idf = math.log(1 + (len(docs) - df[word] + .5) / (df[word] + .5))
            for i, doc in enumerate(counters):
                tf = doc[word]
                sparse[i] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lens[i]/avg))
        dense = matrix @ vector_map[q["content"]]
        orders = [sorted(range(len(docs)), key=lambda i: (-float(s[i]), ids[i])) for s in (sparse, dense)]
        hybrid = np.zeros(len(docs))
        for order in orders:
            for pos, idx in enumerate(order, 1):
                hybrid[idx] += 1 / (60 + pos)
        for method, scores in zip(("bm25", "dense", "hybrid"), (sparse, dense, hybrid)):
            expected_order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), ids[i]))
            saved = rankings[(qid, method)]
            assert len(saved) == 1000
            assert [r["rank"] for r in saved] == list(range(1, 1001))
            assert [r["memory_id"] for r in saved] == [ids[i] for i in expected_order]
            assert np.allclose([r["score"] for r in saved], [scores[i] for i in expected_order], atol=1e-7)
            target_row = next(r for r in saved if r["memory_id"] == target)
            assert target_row["rank"] == details[(qid, method)]["target_rank"]
            assert saved[:10] == details[(qid, method)]["top10"]
            same = [r["rank"] for r in saved if r["score"] == target_row["score"]]
            if len(same) > 1:
                ties.append({"query_id": qid, "method": method, "ranks": same})
    for method in ("bm25", "dense", "hybrid"):
        ranks = [details[(q["query_id"], method)]["target_rank"] for q in queries]
        for k in (1, 3, 5, 10, 20):
            assert sum(r <= k for r in ranks) == summary["metrics"][method]["known_target_hit_count"][str(k)]
    result = {"passed": True,
              "verification": "independent score recomputation script; same assistant, not independent human semantic review",
              "input_hashes_match": True,
              "ranking_count_checked": len(rankings), "candidate_positions_checked": sum(len(r) for r in rankings.values()),
              "bm25_dense_rrf_scores_recomputed": True,
              "target_score_ties": ties,
              "exact_subject_phrase_candidate_count": entity_counts,
              "note": "Exact phrase counts do not rule out related memories that use different wording."}
    (RUN / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

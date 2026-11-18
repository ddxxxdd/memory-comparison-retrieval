"""Independently check saved rankings against official qrels using BEIR."""

import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/experiments/sparsecl-data-general-pilot-20261117"
DATA = ROOT / "data/sparsecl_modified_msmarco_hotpotqa"


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected global reference: {module}.{name}")


def verify(dataset):
    out = OUT / dataset
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    corpus = {}
    for split in ("train", "dev", "test"):
        name = f"{dataset}_{split}_retrieval_gpt4_final.pkl"
        path = DATA / dataset / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == summary["plan"]["input_sha256"][name]
        with path.open("rb") as stream:
            c, q, r = (DataUnpickler(stream).load() for _ in range(3))
        corpus.update(c)
        if split == "dev":
            queries, qrels = q, r
    selected = sorted(random.Random(20261117).sample(sorted(queries), 50))
    labels = {qid: qrels[qid] for qid in selected}
    rows = [json.loads(line) for line in (out / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    rowmap = {(row["query_id"], row["method"]): row for row in rows}
    saved = np.load(out / "rankings.npz", allow_pickle=False)
    ids, qids, methods, orders = (saved[key] for key in ("corpus_ids", "query_ids", "methods", "orders"))
    assert list(ids) == list(corpus)
    assert list(qids) == selected == summary["plan"]["query_ids"]
    assert orders.shape == (50, 3, len(corpus)-1)
    checks = {}
    for mi, method in enumerate(methods):
        results = {}
        for qi, qid in enumerate(qids):
            order = orders[qi, mi]
            assert np.all((order >= 0) & (order < len(ids)))
            assert len(np.unique(order)) == len(corpus)-1
            ranked_ids = ids[order]
            assert qid not in ranked_ids
            row = rowmap[(qid, method)]
            assert row["query"] == queries[qid].strip()
            assert [x["id"] for x in row["top10"]] == list(ranked_ids[:10])
            for item in row["top10"]:
                assert item["text"] == corpus[item["id"]]["text"].strip()
                assert item["is_official_gold"] == (item["id"] in labels[qid])
            for gold, position in row["gold_ranks"].items():
                assert ranked_ids[position-1] == gold
            assert set(row["gold_ranks"]) == set(labels[qid])
            # Strict scores preserve the recorded tie resolution for independent evaluation.
            results[qid] = {doc_id: float(-position) for position, doc_id in enumerate(ranked_ids[:10], 1)}
        ndcg, _, recall, _ = EvaluateRetrieval.evaluate(labels, results, [1, 5, 10])
        for k in (1, 5, 10):
            expected = summary["metrics"][method][str(k)]
            assert abs(ndcg[f"NDCG@{k}"]-expected["mean_ndcg"]) <= 0.0000051
            assert abs(recall[f"Recall@{k}"]-expected["mean_recall"]) <= 0.0000051
        checks[str(method)] = {"BEIR_NDCG": ndcg, "BEIR_Recall": recall}
    report = {"status": "passed", "dataset": dataset, "query_count": 50,
              "corpus_count": len(corpus), "official_hashes_unchanged": True,
              "sampling_verified": True, "full_rankings_verified": True,
              "self_exclusion_verified": True, "metrics_match_BEIR": checks}
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    for dataset in (sys.argv[1:] or ["msmarco", "hotpotqa"]):
        verify(dataset)

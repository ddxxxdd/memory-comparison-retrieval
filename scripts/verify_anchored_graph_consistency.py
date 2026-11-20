"""Replay the saved anchored Test vectors through the main package, without inference."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pytrec_eval

from memory_retrieval.bipolar_graph import ANCHORED_FOUR_DIM_PROMPT, ANCHORED_PROMPT_SHA256, validate_anchored_probe_structure
from memory_retrieval.data import sha256_file, write_json
from memory_retrieval.dataset import load_dataset
from memory_retrieval.evaluation import grouped_ranking_metrics, target_ids
from memory_retrieval.experiment import score_anchored_text


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(source):
    source = Path(source).resolve()
    results = source / "results"
    protocol = read_json(source.parent / "protocol.json")
    assert protocol["system_prompt"] == ANCHORED_FOUR_DIM_PROMPT
    assert protocol["prompt_sha256"] == ANCHORED_PROMPT_SHA256
    dataset = load_dataset(source / "inputs")
    qids = [q["query_id"] for q in dataset.queries]
    for name, digest in protocol["input_sha256"][source.name].items():
        assert sha256_file(source / "inputs" / name) == digest
    manifest = read_json(results / "completion.json")["artifacts_sha256"]
    assert all(sha256_file(results / name) == digest for name, digest in manifest.items())
    records = read_json(results / "probes.json")
    assert [row["query_id"] for row in records] == qids
    for query, record in zip(dataset.queries, records, strict=True):
        assert record["query_content"] == query["content"]
        validate_anchored_probe_structure({key: record[key] for key in ("new_state_anchor", "violation_probes")})
    with np.load(results / "bge_vectors.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        assert metadata["query_ids"] == qids
        vectors = {key: saved[key].copy() for key in ("memories", "queries")}
        vectors["anchored_violation_probes"] = np.split(saved["violation_probes"].copy(), len(qids))
    assert len(vectors["memories"]) == len(dataset.memories)
    assert all(len(group) == 4 for group in vectors["anchored_violation_probes"])
    methods = {"anchored_probes_only": "anchored_probes_only", "anchored_bipolar_diffusion": "anchored_bipolar"}
    rows = score_anchored_text(dataset, methods, vectors, k=5, top_seeds=5, alpha=0.2, beta=0.3)
    expected_rankings = read_json(results / "rankings.json")
    expected_scores = read_json(results / "scores.json")
    rankings, max_score_error = {}, 0.0
    for method, old_name in methods.items():
        rankings[method] = {}
        for qid in qids:
            candidates = rows[method][qid]["ranked_candidates"]
            ranked = [row["memory_id"] for row in candidates]
            assert ranked == expected_rankings[old_name][qid], (method, qid, "ranking mismatch")
            rankings[method][qid] = ranked
            score_key = "r0" if method == "anchored_probes_only" else "r_bipolar"
            expected = dict(zip(expected_scores[qid]["memory_ids"], expected_scores[qid][score_key], strict=True))
            delta = max(abs(row["score"] - expected[row["memory_id"]]) for row in candidates)
            max_score_error = max(max_score_error, delta)
    groups = grouped_ranking_metrics(rankings, dataset.queries, dataset.gold)
    expected = read_json(results / "comparison.json")["groups"]
    evaluator = pytrec_eval.RelevanceEvaluator(
        {g["query_id"]: {mid: 1 for mid in target_ids(g)} for g in dataset.gold}, {"ndcg_cut_10"})
    max_ndcg_error = 0.0
    for method, old_name in methods.items():
        checked = evaluator.evaluate({qid: {mid: float(len(ids) - i) for i, mid in enumerate(ids)}
                                      for qid, ids in rankings[method].items()})
        for kind in ("T1", "T2"):
            metric = groups[kind]["methods"][method]
            reference = expected[kind]["methods"][old_name]
            selected = metric["gold_ranks"]
            ndcg = mean(checked[qid]["ndcg_cut_10"] for qid in selected)
            delta = abs(metric["ndcg_at_10"] - reference["ndcg_at_10"])
            assert delta < 1e-12 and abs(ndcg - metric["ndcg_at_10"]) < 1e-12
            max_ndcg_error = max(max_ndcg_error, delta)
            for k in ("3", "5", "10"):
                assert metric["at_k"][k] == reference["at_k"][k]
    assert all(sha256_file(results / name) == digest for name, digest in manifest.items())
    return {"status": "passed", "source": str(source), "queries": len(qids), "methods": methods,
            "prompt_sha256": ANCHORED_PROMPT_SHA256, "all_full_rankings_identical": True,
            "max_score_abs_difference": max_score_error, "max_ndcg_abs_difference": max_ndcg_error,
            "pytrec_eval_verified": True, "llm_calls": 0, "embedding_calls": 0,
            "historical_artifacts_unchanged": True, "groups": groups}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/experiments/stale-anchored-dev-test-bge-20261120/test")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve())):
        parser.error("--output must be a new file outside the historical Test directory")
    result = verify(args.source)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

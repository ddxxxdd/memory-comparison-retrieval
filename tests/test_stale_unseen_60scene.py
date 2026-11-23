import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("stale60", Path(__file__).resolve().parents[1] / "scripts/run_stale_unseen_60scene.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_existing_probe_records_reused_without_new_generation(tmp_path, monkeypatch):
    queries = [{"query_id": "q", "content": "Current fact"}]
    record = {"query_id": "q", "query_content": "Current fact", "violation_probes": ["A", "B", "C", "D"]}
    m.atomic_json(tmp_path / "probe_records/q.json", record)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed query must not call LLM again")
    monkeypatch.setattr(m, "extract_anchored_probes", forbidden)
    records, _ = m.collect_probes(tmp_path, object(), queries)
    assert records == [record]


def test_malformed_probe_is_never_repaired_or_regenerated(tmp_path, monkeypatch):
    calls = []
    def malformed(client, queries, max_workers):
        calls.append(queries[0]["query_id"])
        raise ValueError("Expected four probes")
    monkeypatch.setattr(m, "extract_anchored_probes", malformed)
    with pytest.raises(RuntimeError):
        m.collect_probes(tmp_path, object(), [{"query_id": "q", "content": "Current fact"}])
    assert calls == ["q"]
    assert m.read(tmp_path / "probe_failures.json")["q"]["type"] == "ValueError"
    assert not (tmp_path / "anchored_probes.json").exists()


def test_paired_statistics_zero_differences_and_stratified_bootstrap_reproducible():
    queries = [{"query_id": f"q{i}", "source_ref": {"type": "T1" if i < 3 else "T2"}} for i in range(6)]
    values = {name: {q["query_id"]: .5 for q in queries} for name in ("anchored_bipolar_diffusion", "dense", "sparsecl_msmarco")}
    first = m.paired_statistics(queries, values)
    assert first == m.paired_statistics(queries, values)
    for group in first.values():
        for row in group.values():
            assert row["wilcoxon_p_two_sided"] == 1
            assert row["bootstrap_ci95"] == [0, 0]
    for q in queries:
        values["anchored_bipolar_diffusion"][q["query_id"]] = .7
    result = m.paired_statistics(queries, values)["all"]["dense"]
    assert result["wins"] == 6 and result["ties"] == 0
    assert np.allclose(result["bootstrap_ci95"], [.2, .2])


def test_package_metrics_match_pytrec_with_original_tie_order():
    from memory_retrieval.data import Memory
    from memory_retrieval.dataset import Dataset
    queries = [{"query_id": "q", "scenario_id": "s", "content": "query", "source_ref": {"type": "T1"}}]
    docs = [Memory("a", "gold", "fixture", scenario_id="s"), Memory("b", "background", "fixture", scenario_id="s")]
    gold = [{"query_id": "q", "conflict_old_memory_ids": ["a"], "target_details": {"a": {"source": "stale_native", "relation": "conflict"}}}]
    dataset = Dataset({}, docs, queries, gold)
    row = {"query_id": "q", "ranked_candidates": [{"memory_id": "a", "score": 1}, {"memory_id": "b", "score": 1}]}
    groups, per_query, verification = m.evaluate(dataset, {"fixture": {"q": row}})
    assert per_query["fixture"]["q"] == 1
    assert groups["all"]["methods"]["fixture"]["at_k"]["10"]["hits"] == 1
    assert verification["max_ndcg_difference"] == 0

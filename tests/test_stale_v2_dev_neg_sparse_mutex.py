from contextlib import closing
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from memory_retrieval.data import Memory, sha256_file, write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_stale_v2_dev_neg_sparse_mutex.py"
SPEC = importlib.util.spec_from_file_location("stale_v2_dev_neg_sparse_mutex", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def write_probe_cache(path, entries):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        for index, (text, prompt, payload) in enumerate(entries):
            request = {"body": {"messages": [{"role": "system", "content": prompt},
                                            {"role": "user", "content": json.dumps({"query": text})}]}}
            response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
            connection.execute("INSERT INTO responses VALUES (?, ?, ?)", (str(index), json.dumps(request), json.dumps(response)))


def test_raw_negative_seed_cap_mutex_and_stable_ties():
    docs = [Memory(mid, mid, "fixture") for mid in ("p-z", "p-e", "p-b", "p-c", "p-d", "p-a",
                                                    "n-f", "n-c", "n-b", "n-e", "n-a", "n-d", "background")]
    s_plus = np.array([.35] + [0.] * 4 + [1.] + [.1] * 6 + [-.2])
    r0 = np.array([.2] * 6 + [0.] * 7)
    sparse = diagnostic.negative_seeds(s_plus, r0, docs)
    selected = {doc.memory_id: delta for doc, delta in zip(docs, sparse) if delta > 0}
    assert selected == pytest.approx({"p-z": .15, "n-a": .1, "n-b": .1, "n-c": .1, "n-d": .1})
    np.testing.assert_array_equal(diagnostic.negative_seeds(r0, r0, docs), np.zeros_like(r0))


def test_locked_baselines_and_indirect_penalty_on_a_positive_seed():
    bank = np.eye(7)
    docs = [Memory(f"doc-{index}", "fixture", "fixture") for index in range(7)]
    transition = np.zeros((7, 7))
    transition[0, 5] = transition[5, 0] = 1.
    scores = diagnostic.score_variants(bank[5], np.repeat(bank[[0]], 4, axis=0), bank, transition, docs)
    assert tuple(scores) == diagnostic.ARMS
    assert tuple(diagnostic.BETAS.values()) == (0.3, 0.5, 0.8)
    expected_positive = [.8, 0., 0., 0., 0., .2, 0.]
    np.testing.assert_allclose(scores[diagnostic.POSITIVE], expected_positive)
    np.testing.assert_allclose(scores[diagnostic.BIPOLAR], [.74, 0., 0., 0., 0., -.1, 0.])
    for name, beta in diagnostic.BETAS.items():
        expected = [.8 - beta * .2, 0., 0., 0., 0., .2 - beta, 0.]
        np.testing.assert_allclose(scores[name], expected)


def test_cache_priority_exact_matching_and_read_only_sqlite(tmp_path):
    queries = [{"query_id": f"q-{sid}", "scenario_id": sid, "content": f"query {sid}"}
               for sid in ("primary", "legacy", "extra", "locked", "outside")]

    def record(query, word, **extra):
        return {"query_id": query["query_id"], "query_content": query["content"],
                "violation_probes": [word] * 4, **extra}

    first = tmp_path / diagnostic.PREFERRED[0] / "results/anchored_probes.json"
    second = tmp_path / diagnostic.PREFERRED[1] / "results/probes.json"
    write_json(first, [record(queries[0], "preferred"), record(queries[3], "test"), record(queries[4], "outside")])
    write_json(second, [record(queries[0], "duplicate"), record(queries[1], "legacy", new_state_anchor="ignored")])
    write_json(tmp_path / diagnostic.LOCKED_TEST / "anchored_probes.json", {"not": "a probe archive"})
    write_json(tmp_path / "stale-other/new_probes.json", [
        {**record(queries[2], "wrong"), "query_content": "wrong text"},
        record(queries[2], "truncated", finish_reason="length"),
        {**record(queries[2], "three"), "violation_probes": ["three"] * 3},
    ])
    cache = tmp_path / "stale-other/probe_cache.sqlite3"
    write_probe_cache(cache, [(queries[2]["content"], "system", {"violation_probes": ["cached"] * 4})])
    digest = sha256_file(cache)
    selected = diagnostic.select_cached_probes(tmp_path, queries, {"primary", "legacy", "extra"})
    assert list(selected) == ["primary", "legacy", "extra"]
    assert selected["primary"]["violation_probes"] == ["preferred"] * 4
    assert selected["primary"]["prompt_sha256"] is None
    assert selected["primary"]["prompt_status"] == "missing_request_cache"
    assert selected["legacy"]["legacy_anchor_ignored"]
    assert selected["extra"]["violation_probes"] == ["cached"] * 4
    assert selected["extra"]["probe_source"] == "stale-other/probe_cache.sqlite3"
    assert selected["extra"]["prompt_source"] == selected["extra"]["probe_source"]
    assert selected["extra"]["prompt_sha256"] == sha256(b"system").hexdigest()
    assert sha256_file(cache) == digest and len(list(cache.parent.glob("probe_cache.sqlite3*"))) == 1


def test_archive_prompt_hash_matches_exact_response_in_same_run(tmp_path):
    queries = [{"query_id": sid, "scenario_id": sid, "content": f"query {sid}"}
               for sid in ("current", "legacy", "unmatched", "ambiguous")]
    probes = ["one", "two", "three", "four"]
    query_only = {"violation_probes": probes}
    anchored = {**query_only, "new_state_anchor": "original anchor"}
    current_run, legacy_run = (tmp_path / name for name in diagnostic.PREFERRED)
    write_json(current_run / "results/anchored_probes.json", [
        {"query_id": query["query_id"], "query_content": query["content"], **query_only}
        for query in queries if query["scenario_id"] != "legacy"
    ])
    write_json(legacy_run / "results/probes.json", [{"query_id": "legacy", "query_content": "query legacy", **anchored}])
    prompt = "actual query-only prompt\r\n"
    current_cache = current_run / "probe_cache.sqlite3"
    legacy_cache = legacy_run / "results/probe_cache.sqlite3"
    write_probe_cache(current_cache, [
        ("query current", prompt, query_only),
        ("query unmatched", prompt, {"violation_probes": probes[::-1]}),
        ("query ambiguous", prompt, query_only),
        ("query ambiguous", "different query-only prompt", query_only),
    ])
    write_probe_cache(legacy_cache, [
        ("query legacy", "actual legacy prompt", anchored),
        ("query legacy", "wrong legacy prompt", {**anchored, "new_state_anchor": "different anchor"}),
        ("query unmatched", "different run's prompt", query_only),
    ])

    selected = diagnostic.select_cached_probes(tmp_path, queries, {query["scenario_id"] for query in queries})
    for key, cache, system in (("current", current_cache, prompt), ("legacy", legacy_cache, "actual legacy prompt")):
        assert selected[key]["prompt_status"] == "verified_request"
        assert selected[key]["prompt_sha256"] == sha256(system.encode("utf-8")).hexdigest()
        assert selected[key]["prompt_source"] == cache.relative_to(tmp_path).as_posix()
    for key, status in (("unmatched", "no_matching_request"), ("ambiguous", "ambiguous_request")):
        assert selected[key]["prompt_status"] == status
        assert selected[key]["prompt_sha256"] is None and selected[key]["prompt_source"] is None
    report = diagnostic.markdown({"selection": {"scenes": list(selected.values())}, "groups": {},
                                  "pytrec_eval": {"max_ndcg_at_10_difference": 0.0}})
    assert selected["legacy"]["prompt_sha256"] in report
    assert "未核实" in report


def test_locked_partition_rejects_overlap_even_when_hashes_match(tmp_path):
    test_rows = [{"scenario_id": f"s-{index}", "type": "T1" if index < 30 else "T2",
                  "normalized_native_text_sha256": f"hash-{index}"} for index in range(60)]
    write_json(tmp_path / "selection.json", {"scenarios": test_rows, "excluded_scenario_ids": ["s-0"]})
    write_json(tmp_path / "exclusions.json", {"records": [{"scenario_id": "s-0"}]})
    write_json(tmp_path / "protocol.json", {f"{name}_sha256": sha256_file(tmp_path / f"{name}.json")
                                           for name in ("selection", "exclusions")})
    with pytest.raises(ValueError, match="Invalid locked Test"):
        diagnostic.locked_partition(tmp_path)


def test_metrics_top_g_and_ndcg_agree_on_full_synthetic_rankings():
    background = [f"b-{index}" for index in range(10)]
    before = [background[0], "native", *background[1:], "synthetic"]
    after = ["synthetic", *background[:9], "native", background[-1]]
    qrels, per_query = {}, []
    rankings = {name: {} for name in diagnostic.ARMS}
    for kind in ("T1", "T2"):
        gold = {"query_id": kind, "conflict_old_memory_ids": ["native", "synthetic"]}
        qrels[kind] = {"native": 1, "synthetic": 1}
        rows = {}
        for name in diagnostic.ARMS:
            order = before if name == diagnostic.POSITIVE else after
            rankings[name][kind] = {mid: len(order) - index for index, mid in enumerate(order)}
            rows[name] = diagnostic.measure_ranking(order, gold, "native")
        per_query.append({"query_id": kind, "type": kind, "gold_count": 2, "methods": rows})
    result = diagnostic.summarize(per_query)["Overall"]["methods"]
    baseline, improved = result[diagnostic.POSITIVE], result[next(iter(diagnostic.BETAS))]
    assert baseline["micro_recall"] == improved["micro_recall"] == {"3": .5, "5": .5, "10": .5}
    assert baseline["macro_r_precision"] == improved["macro_r_precision"] == .5
    assert improved["map"] == pytest.approx((1 + 2 / 11) / 2)
    assert baseline["native_hits_at_10"] == 2 and improved["native_hits_at_10"] == 0
    assert improved["against_positive"] == {"top10_recovered": 2, "top10_dropped": 2, "topG_recovered": 2,
                                            "topG_dropped": 2, "rank_up": 2, "rank_unchanged": 0,
                                            "rank_down": 2, "ndcg_wins": 2, "ndcg_ties": 0, "ndcg_losses": 0}
    assert diagnostic.verify_ndcg(qrels, rankings, per_query)["max_ndcg_at_10_difference"] < 1e-12


def test_existing_output_is_rejected_before_loading_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        diagnostic.main([])


def test_output_failure_is_retryable_without_partial_results(tmp_path, monkeypatch):
    output = tmp_path / "result"
    summary = {"complete": True}

    def fail_report(_summary):
        assert not output.exists()
        assert len(list(tmp_path.glob(".result-*/results/summary.json"))) == 1
        raise OSError("simulated report failure")

    monkeypatch.setattr(diagnostic, "markdown", fail_report)
    with pytest.raises(OSError, match="simulated report failure"):
        diagnostic.write_outputs(output, summary)
    assert not list(tmp_path.iterdir())

    monkeypatch.setattr(diagnostic, "markdown", lambda _summary: "completed report")
    diagnostic.write_outputs(output, summary)
    assert diagnostic.read(output / "summary.json") == summary
    assert (output / "report.zh.md").read_text(encoding="utf-8") == "completed report"
    assert list(tmp_path.iterdir()) == [output]


def test_publication_refuses_competing_empty_directory_without_precheck(tmp_path, monkeypatch):
    output = tmp_path / "result"

    def create_output(_summary):
        output.mkdir()
        return "completed report"

    monkeypatch.setattr(diagnostic, "markdown", create_output)
    monkeypatch.setattr(diagnostic, "require_new_output", lambda _output: None)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        diagnostic.write_outputs(output, {})
    assert output.is_dir() and not list(output.iterdir())
    assert list(tmp_path.iterdir()) == [output]

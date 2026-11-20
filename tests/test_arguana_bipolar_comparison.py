"""Contract tests for matching the historical SparseCL/BEIR evaluation protocol."""
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_arguana_bipolar_comparison.py"
spec = importlib.util.spec_from_file_location("arguana_comparison", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_official_loader_preserves_merge_order_and_absent_gold(tmp_path):
    (tmp_path / "data").mkdir()
    splits = {
        "train": ({"shared": {"text": "old"}, "train": {"text": "train"}}, {}, {}),
        "dev": ({"shared": {"text": "new"}, "dev": {"text": "dev"}}, {}, {}),
        "test": ({"test": {"text": " test "}}, {"q": " query "}, {"q": {"absent": 1}}),
    }
    for name, items in splits.items():
        with (tmp_path / "data" / f"arguana_{name}_retrieval_final.pkl").open("wb") as stream:
            for value in items:
                pickle.dump(value, stream)
    docs, queries, qrels, inputs, missing = module.load_official_arguana(tmp_path)
    assert [doc.memory_id for doc in docs] == ["shared", "train", "dev", "test"]
    assert docs[0].content == "new"
    assert docs[-1].content == "test"
    assert queries == [{"query_id": "q", "content": " query "}]
    assert qrels == {"q": {"absent": 1}}
    assert missing == [{"query_id": "q", "memory_id": "absent"}]
    assert set(inputs) == {"train", "dev", "test"}


def test_beir_truncates_before_self_exclusion_without_backfill_and_keeps_missing_targets():
    ids = ["q_cutoff"] + [f"noise{i}" for i in range(9)] + ["gold"]
    docs = [module.Memory(mid, mid, "fixture") for mid in ids]
    scores = np.arange(len(ids), 0, -1, dtype=np.float32)
    raw = {
        "q_cutoff": module.top10_for_official_evaluator("q_cutoff", scores, docs),
        "q_missing": {"background": 1.0},
        "q_hit": {"q_hit": 2.0, "gold": 1.0},
    }
    before = deepcopy(raw)
    qrels = {"q_cutoff": {"gold": 1}, "q_missing": {"absent": 1}, "q_hit": {"gold": 1}}
    evaluation, per_query = module.official_evaluation(qrels, raw)
    assert "gold" not in raw["q_cutoff"]
    assert raw == before
    assert evaluation["queries_evaluated"] == 3
    assert evaluation["self_ids_removed_after_top10"] == 2
    assert evaluation["metrics"]["NDCG@10"] == 0.33333
    assert evaluation["metrics"]["Recall@10"] == 0.33333
    assert per_query["q_cutoff"]["ndcg_cut_10"] == 0
    assert per_query["q_missing"]["ndcg_cut_10"] == 0
    assert per_query["q_hit"]["ndcg_cut_10"] == 1


def test_adapter_uses_frozen_main_functions_and_constants():
    from memory_retrieval import bipolar_graph
    assert module.build_knn_transition is bipolar_graph.build_knn_transition
    assert module.compute_bipolar_diffusion_scores is bipolar_graph.compute_bipolar_diffusion_scores
    assert module.extract_anchored_probes is bipolar_graph.extract_anchored_probes
    assert module.ANCHORED_PROMPT_SHA256 == module.LOCKED_PROMPT_SHA256
    assert module.GRAPH == {"k": 5, "top_seeds": 5, "alpha": 0.2, "beta": 0.3}


def test_output_directory_must_be_new(tmp_path):
    with pytest.raises(FileExistsError):
        module.main(["--output", str(tmp_path)])


def test_historical_windows_bom_status_is_read_without_changing_source(tmp_path):
    history = tmp_path / "test_results/repro_arguana"
    (history / "test").mkdir(parents=True)
    (history / "dev").mkdir()
    (tmp_path / "repro").mkdir()
    lock = tmp_path / "repro/assets.lock.json"
    lock.write_text("[]\n", encoding="utf-8")
    (history / "dev/arguana_alpha.json").write_text(json.dumps({
        "split": "dev", "smoke_test": False, "alpha": 0.18,
        "settings": {"max_seq_length": 512, "pooler_type": "avg", "metric": "both_sum",
                     "assets_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}}))
    status = history / "formal_run_status.json"
    status.write_text(json.dumps({"state": "completed", "exit_code": 0}), encoding="utf-8-sig")
    before = status.read_bytes()
    (history / "test/arguana_parallel_output").write_text(
        "sparsity contradiction retrieval results\n"
        "{'NDCG@1': 0.4936, 'NDCG@10': 0.70406} {'Recall@10': 0.91038}\n")
    result = module.historical_sparsecl(tmp_path)
    assert result["metrics"]["NDCG@10"] == 0.70406
    assert status.read_bytes() == before


def test_recovery_preserves_original_cache_and_reuses_only_structurally_valid_responses(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    protocol = {"prompt_sha256": module.LOCKED_PROMPT_SHA256, "graph": module.GRAPH,
                "probe_workers": 4, "llm": {"model": "fixture"}, "dataset_inputs": {}, "source_sha256": {}}
    (old / "status.json").write_text(json.dumps({**protocol, "status": "failed", "wall_seconds": 42}))
    events = [{"status": "ok", "cached": False, "seconds": 1, "usage": {"total_tokens": 100}}]
    (old / "api_events.json").write_text(json.dumps(events))
    valid = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"violation_probes": ["Probe A.", "Probe B.", "Probe C.", "Probe D."]})}}]})
    invalid = json.dumps({"choices": [{"finish_reason": "content_filter", "message": {"content": "{"}}]})
    cache = old / "probe_cache.sqlite3"
    with sqlite3.connect(cache) as db:
        db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        db.executemany("INSERT INTO responses VALUES (?, ?, ?)", [("valid", "request1", valid), ("invalid", "request2", invalid)])
    before = cache.read_bytes()
    history, seconds = module.recover_probe_cache(old, new, protocol)
    assert history == events and seconds == 42
    assert cache.read_bytes() == before
    with sqlite3.connect(new / "probe_cache.sqlite3") as db:
        assert db.execute("SELECT key, request, response FROM responses").fetchall() == [("valid", "request1", valid)]
    audit = json.loads((new / "cache_recovery.json").read_text())
    assert audit["cached_successes_reused"] == 1
    assert audit["invalid_responses_to_retry_unchanged"][0]["finish_reason"] == "content_filter"


def test_recovery_rejects_protocol_changes_before_creating_cache(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "status.json").write_text(json.dumps({"status": "failed", "prompt_sha256": "different"}))
    with pytest.raises(ValueError, match="Recovery protocol differs"):
        module.recover_probe_cache(old, new, {"prompt_sha256": module.LOCKED_PROMPT_SHA256})
    assert not (new / "probe_cache.sqlite3").exists()


def test_resume_reuses_vectors_and_valid_responses_preserving_invalid_original(tmp_path):
    source, output = tmp_path / "original", tmp_path / "resumed"
    source.mkdir()
    output.mkdir()
    locked = {"prompt_sha256": module.LOCKED_PROMPT_SHA256, "graph": module.GRAPH,
              "llm": {"model": "fixture"}, "dataset_inputs": {}, "source_sha256": {}, "historical_bank": {}}
    previous = {**locked, "probe_workers": 4, "wall_seconds": 10, "status": "interrupted_for_resume"}
    (output / "status.json").write_text(json.dumps(previous))
    (output / "cache_recovery.json").write_text(json.dumps({"source_run": str(source), "previous_attempt_wall_seconds": 20}))
    (source / "api_events.json").write_text("[]")
    response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"violation_probes": ["A", "B", "C", "D"]})}}], "usage": {"total_tokens": 25}})
    for directory in (source, output):
        with sqlite3.connect(directory / "probe_cache.sqlite3") as db:
            db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
            db.execute("INSERT INTO responses VALUES (?, ?, ?)", ("valid", "request", response))
    with sqlite3.connect(output / "probe_cache.sqlite3") as db:
        db.execute("INSERT INTO responses VALUES (?, ?, ?)", ("bad", "request2", json.dumps({"choices": []})))
    for name in ("bank.npy", "queries.npy", "transition.npy"):
        np.save(source / name, np.eye(2, dtype=np.float32))
    np.save(output / "queries.npy", np.ones((2, 2), dtype=np.float32))
    before = (output / "queries.npy").read_bytes()
    _, seconds = module.resume_artifacts(output, previous, {**locked, "probe_workers": 16})
    assert seconds == 30
    assert (output / "queries.npy").read_bytes() == before
    assert (output / "transition.npy").read_bytes() == (source / "transition.npy").read_bytes()
    info = json.loads((output / "resume_info.json").read_text())
    assert info["valid_cached_responses"] == 1 and info["probe_workers"] == 16
    with sqlite3.connect(output / "probe_cache.sqlite3") as db:
        assert db.execute("SELECT key FROM responses").fetchall() == [("valid",)]
    with sqlite3.connect(Path(info["snapshot"]) / "probe_cache.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


def test_resume_does_not_allow_restarting_a_completed_experiment(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="stopped incomplete"):
        module.main(["--output", str(tmp_path), "--resume", "--probe-workers", "16"])


def test_invalid_response_retries_keep_successes_and_archive_original(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cache = tmp_path / "probe_cache.sqlite3"
    valid = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"violation_probes": ["A", "B", "C", "D"]})}}]})
    bad = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
        {"violation_probes": ["A"] * 8})}}]})
    with sqlite3.connect(cache) as db:
        db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        request = json.dumps({"body": {"messages": [{}, {"content": json.dumps({"query": "query"})}]}})
        db.executemany("INSERT INTO responses VALUES (?, ?, ?)", [("good", "{}", valid), ("bad", request, bad)])
    client = SimpleNamespace(settings=SimpleNamespace(cache_path=cache), events=[])
    calls = []
    def extract(client, queries, max_workers):
        calls.append(max_workers)
        if len(calls) == 1:
            raise module.APIError("invalid response")
        with sqlite3.connect(cache) as db:
            assert db.execute("SELECT key, response FROM responses").fetchall() == [("good", valid)]
        return [{"query_id": "q", "violation_probes": ["A", "B", "C", "D"]}], {"generation": {}}
    monkeypatch.setattr(module, "extract_anchored_probes", extract)
    records, audit = module.extract_remaining_probes(client, [{"query_id": "q", "content": "query"}], tmp_path, max_workers=16)
    assert records[0]["query_id"] == "q" and calls == [1, 1]
    assert audit["generation"]["invalid_response_retries"] == 1
    original = json.loads((tmp_path / "invalid_response_attempts.jsonl").read_text())
    assert original["response"] == json.loads(bad)

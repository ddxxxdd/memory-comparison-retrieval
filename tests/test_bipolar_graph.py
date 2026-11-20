import ast
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from memory_retrieval.api import APIClient, APIError
from memory_retrieval.baselines import rank
from memory_retrieval.bipolar_graph import (ANCHORED_FOUR_DIM_PROMPT, ANCHORED_METHODS, ANCHORED_PROMPT_SHA256,
    audit_anchored_probes, build_knn_transition, compute_bipolar_diffusion_scores, extract_anchored_probes)
from memory_retrieval.cli import run_cli
from memory_retrieval.config import Settings
from memory_retrieval.data import Memory, read_jsonl, write_jsonl
from memory_retrieval.ocsp import ONLINE_ZERO_SHOT_BIPOLAR_PROMPT, compute_subspace_ocsp_scores


def docs(ids):
    return [Memory(mid, mid, "test") for mid in ids]


def response(value, finish="stop"):
    return {"model": "deepseek-v4.1-flash", "choices": [{"finish_reason": finish,
            "message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def test_locked_prompt_matches_original_bytes():
    path = Path(__file__).resolve().parents[1] / "scripts/run_stale_anchored_dev_test.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(n, ast.Name) and n.id == "PROMPT" for n in node.targets))
    assert ANCHORED_FOUR_DIM_PROMPT == original
    assert ANCHORED_PROMPT_SHA256 == "0bf33d0094a328f8d4cd796b26f1f389aee5196e7f2f2fc77bd553c9c0008fcf"


def test_graph_is_directed_positive_and_resolves_ties_by_id():
    bank = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    original = bank.copy()
    documents = docs(["b", "c", "a", "isolated", "negative"])
    transition = build_knn_transition(bank, documents, k=1)
    expected = np.zeros((5, 5), dtype=np.float32)
    expected[0, 2] = expected[1, 2] = expected[2, 0] = 1
    np.testing.assert_array_equal(transition, expected)
    np.testing.assert_array_equal(bank, original)
    assert transition.dtype == np.float32
    wider = build_knn_transition(bank, documents, k=20)
    np.testing.assert_array_equal(wider.sum(axis=1), [1, 1, 1, 0, 0])
    assert np.all(np.diag(wider) == 0)
    assert wider[0, 1] == wider[0, 2] == 0.5


def test_empty_and_singleton_graphs_keep_zero_rows():
    assert build_knn_transition(np.empty((0, 2)), []).shape == (0, 0)
    np.testing.assert_array_equal(build_knn_transition(np.array([[1.0, 0]]), docs(["a"])), [[0]])


def test_diffusion_matches_hand_computed_directed_formula():
    bank = np.eye(3)
    transition = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    scores = compute_bipolar_diffusion_scores(np.array([0.6, 0.8, 0.0]), np.array([[0.8, 0.0, 0.6]]),
                                             bank, transition, docs(["a", "b", "c"]), top_seeds=1)
    assert set(scores) == set(ANCHORED_METHODS)
    np.testing.assert_allclose(scores["anchored_probes_only"], [0.8, 0, 0.6], rtol=0, atol=1e-15)
    np.testing.assert_allclose(scores["anchored_pos_diffusion"], [0.64, 0, 0.64], rtol=0, atol=1e-15)
    np.testing.assert_allclose(scores["anchored_bipolar_diffusion"], [0.592, -0.24, 0.64], rtol=0, atol=1e-15)


def test_seed_ties_and_zero_parameters_have_expected_effect():
    bank = np.eye(3)
    vector = np.ones(3) / np.sqrt(3)
    transition = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    documents = docs(["b", "a", "c"])
    scores = compute_bipolar_diffusion_scores(vector, vector[None], bank, transition, documents, top_seeds=1, beta=0)
    # The tied seed is id a (index 1); P@z transports its value to row 0.
    assert rank(scores["anchored_pos_diffusion"], documents) == [0, 1, 2]
    np.testing.assert_array_equal(scores["anchored_bipolar_diffusion"], scores["anchored_pos_diffusion"])
    zero = compute_bipolar_diffusion_scores(vector, vector[None], bank, transition, documents, alpha=0, beta=0)
    for values in zero.values():
        np.testing.assert_array_equal(values, vector)


@pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": float("nan")}, {"alpha": 1.1},
                                   {"beta": -1}, {"beta": float("inf")}, {"top_seeds": 0}])
def test_invalid_diffusion_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        compute_bipolar_diffusion_scores(np.ones(1), np.ones((1, 1)), np.ones((1, 1)),
                                         np.zeros((1, 1)), docs(["a"]), **kwargs)


def test_four_concurrent_requests_replay_sqlite_cache_without_network(tmp_path, monkeypatch):
    client = APIClient(Settings(llm_base_url="https://example.test/v1", llm_api_key="test-key",
                                llm_model="deepseek-v4.1-flash", cache_path=tmp_path / "probe_cache.sqlite3"))
    barrier = threading.Barrier(4, timeout=10)
    requests = []

    def fake_urlopen(request, **kwargs):
        body = json.loads(request.data)
        requests.append(body)
        barrier.wait()
        query = json.loads(body["messages"][1]["content"])["query"]
        return BytesIO(json.dumps(response({"new_state_anchor": f"Anchor {query}",
                         "violation_probes": [f"Old {query} {i}" for i in range(4)]})).encode())

    monkeypatch.setattr("memory_retrieval.api.urlopen", fake_urlopen)
    queries = [{"query_id": f"q{i}", "content": str(i), "ignored_gold": "secret-gold"} for i in range(4)]
    records, audit = extract_anchored_probes(client, queries)
    assert [row["query_id"] for row in records] == [q["query_id"] for q in queries]
    assert audit["generation"]["usage"]["api_attempts"] == 4
    assert audit["probe_query_cosine_gt_085"] is None
    for body in requests:
        assert body["temperature"] == 0 and body["reasoning_effort"] == "none"
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"][0]["content"] == ANCHORED_FOUR_DIM_PROMPT
        assert set(json.loads(body["messages"][1]["content"])) == {"query"}
        assert "secret-gold" not in json.dumps(body)
    monkeypatch.setattr("memory_retrieval.api.urlopen", Mock(side_effect=AssertionError("Network forbidden")))
    replay, replay_audit = extract_anchored_probes(client, queries)
    assert [r["violation_probes"] for r in records] == [r["violation_probes"] for r in replay]
    assert replay_audit["generation"]["usage"]["cache_hits"] == 4
    assert replay_audit["generation"]["usage"]["api_attempts"] == 0


@pytest.mark.parametrize("value,finish", [({"violation_probes": ["x"] * 4}, "stop"),
    ({"new_state_anchor": "x", "violation_probes": ["a"] * 3}, "stop"),
    ({"new_state_anchor": "x", "violation_probes": ["a", "b", "c", 1]}, "stop"),
    ({"new_state_anchor": "x", "violation_probes": ["a"] * 4, "extra": 1}, "stop"),
    ({"new_state_anchor": "x", "violation_probes": ["a"] * 4}, "length")])
def test_bad_structure_or_truncation_fails_without_regeneration(value, finish):
    client = Mock(settings=Settings(llm_base_url="https://example.test", llm_api_key="test", llm_model="test"), events=[])
    client._post.return_value = response(value, finish)
    with pytest.raises(APIError):
        extract_anchored_probes(client, [{"query_id": "q", "content": "new"}])
    assert client._post.call_count == 1


def test_drift_audit_flags_but_preserves_outputs():
    queries = [{"query_id": "q", "content": "Current fact"}]
    records = [{"query_id": "q", "new_state_anchor": "Core fact", "violation_probes": [
        "Current fact", "Core fact", "I never do it", "事实 " + "word " * 31]}]
    probes = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    audit = audit_anchored_probes(queries, records, np.array([[1, 0]]), [probes])
    assert audit["exact_query_or_anchor_repeats"] == 2
    assert audit["probe_query_cosine_gt_085"] == 1
    assert audit["probes_over_30_words"] == audit["probes_with_explicit_negation"] == audit["probes_with_non_latin_letters"] == 1
    assert audit["filtered_or_regenerated"] is False and len(records[0]["violation_probes"]) == 4


def test_new_and_legacy_arms_run_together_with_shared_encoder_and_overlap(tmp_path, monkeypatch):
    from memory_retrieval import experiment

    memories = [Memory(f"{sid}-{j}", f"memory {sid} {j}", "test", sid)
                for j in range(3) for sid in ("one", "two")]
    queries = [{"query_id": f"q{i}", "content": f"query {i}", "scenario_id": sid, "source_ref": {"type": kind}}
               for i, (sid, kind) in enumerate((("one", "T1"), ("two", "T2")))]
    gold = [{"query_id": q["query_id"], "conflict_old_memory_ids": [f"{q['scenario_id']}-2"],
             "target_details": {f"{q['scenario_id']}-2": {"source": "stale_native", "relation": "conflict"}}} for q in queries]
    for name, rows in (("memories", [m.to_dict() for m in memories]), ("queries", queries), ("gold", gold)):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    settings = Settings(llm_base_url="https://example.test/v1", llm_api_key="test", llm_model="legacy-model")
    monkeypatch.setattr(experiment.Settings, "from_env", lambda path: settings)
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda name: "test")
    root = tmp_path / "models"
    (root / "sparsecl").mkdir(parents=True)
    (root / "sparsecl/models.py").write_text("# Test fixture", encoding="utf-8")
    started, baselines_done = threading.Event(), threading.Event()
    requests, encoded_texts, instances = [], [], []
    text_vectors = {m.content: np.eye(3, dtype=np.float32)[int(m.memory_id[-1])] for m in memories}
    text_vectors.update({"query 0": np.array([1, 0, 0], dtype=np.float32), "query 1": np.array([1, 0, 0], dtype=np.float32)})
    text_vectors.update({f"old {i}": np.array([0, 0, 1], dtype=np.float32) for i in range(4)})
    text_vectors.update({"compatible 0": np.array([0, 1, 0], dtype=np.float32), "compatible 1": np.array([1, 0, 0], dtype=np.float32)})

    def fake_urlopen(request, **kwargs):
        body = json.loads(request.data)
        requests.append(body)
        if body["messages"][0]["content"] == ANCHORED_FOUR_DIM_PROMPT:
            started.set()
            assert baselines_done.wait(10), "Probe generation must overlap local baseline work"
            value = {"new_state_anchor": "This anchor must never be encoded", "violation_probes": [f"old {i}" for i in range(4)]}
        else:
            assert body["messages"][0]["content"] == ONLINE_ZERO_SHOT_BIPOLAR_PROMPT
            value = {"compatible_paraphrases": ["compatible 0", "compatible 1"], "violation_probes": [f"old {i}" for i in range(3)]}
        return BytesIO(json.dumps(response(value)).encode())

    monkeypatch.setattr("memory_retrieval.api.urlopen", fake_urlopen)

    class FakeEncoder:
        def __init__(self, name, base):
            assert started.wait(10)
            assert name == "bge"
            instances.append(self)
            self.identity = {"model": "test-bge", "token_audit": {}}

        def encode_groups(self, groups):
            vectors = {}
            for key, texts in groups.items():
                encoded_texts.extend(texts)
                vectors[key] = np.stack([text_vectors[t] for t in texts])
                self.identity["token_audit"][key] = {"unknown_tokens": 0, "would_truncate": 0}
            return vectors, self.identity, {key: 0 for key in groups}

    monkeypatch.setattr(experiment, "FrozenEncoder", FakeEncoder)

    def fake_sparsecl(dataset, methods, vectors, args, info):
        assert started.is_set()
        baselines_done.set()
        rows = {"sparsecl_arguana": {}}
        for mi, qi in dataset.groups().values():
            documents = [dataset.memories[i] for i in mi]
            for i in qi:
                q = dataset.queries[i]
                rows["sparsecl_arguana"][q["query_id"]] = experiment.ranked_row(
                    q, documents, "sparsecl_arguana", vectors["memories"][mi] @ vectors["queries"][i])
        return rows

    monkeypatch.setattr(experiment, "score_sparsecl", fake_sparsecl)
    arms = [*ANCHORED_METHODS, "ocsp_qr", "probes_only_no_ortho", "dense", "bm25", "hybrid", "sparsecl_arguana"]
    with redirect_stdout(StringIO()):
        output = run_cli(["--dataset", str(tmp_path), "--output", str(tmp_path / "result"),
                          "--model-root", str(root), "--arms", *arms])
    assert len(instances) == 1
    assert all(encoded_texts.count(m.content) == 1 for m in memories)
    assert all(encoded_texts.count(q["content"]) == 1 for q in queries)
    assert "This anchor must never be encoded" not in encoded_texts
    assert len(requests) == 4
    assert {r["model"] for r in requests} == {"legacy-model", "deepseek-v4.1-flash"}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert set(report["arms"]) == set(arms)
    assert set(report["groups"]) == {"all", "T1", "T2"}
    for kind in ("T1", "T2"):
        for arm in ANCHORED_METHODS:
            metric = report["groups"][kind]["methods"][arm]
            assert metric["at_k"]["3"]["hits"] == 1
            assert metric["at_k"]["10"]["recall_by_source"]["stale_native"]["hits"] == 1
            assert metric["ndcg_at_10"] == 1
    expected_old = compute_subspace_ocsp_scores(text_vectors["query 0"], np.stack([text_vectors[f"compatible {i}"] for i in range(2)]),
                                               np.stack([text_vectors[f"old {i}"] for i in range(3)]), np.eye(3, dtype=np.float32))
    for arm in ("ocsp_qr", "probes_only_no_ortho"):
        for row in read_jsonl(output / "arms" / f"{arm}.jsonl"):
            sid = row["scenario_id"]
            by_id = {c["memory_id"]: c["score"] for c in row["ranked_candidates"]}
            np.testing.assert_array_equal([by_id[f"{sid}-{j}"] for j in range(3)], expected_old[arm])
    assert (output / "completion.json").is_file()


@pytest.mark.parametrize("option,value", [("--graph-k", "0"), ("--graph-seeds", "-1"), ("--probe-workers", "0"),
                                         ("--graph-alpha", "nan"), ("--graph-alpha", "1.1"), ("--graph-beta", "-1")])
def test_cli_rejects_invalid_graph_options_before_loading_data(option, value):
    with pytest.raises(SystemExit):
        run_cli(["--dataset", "unused", option, value])

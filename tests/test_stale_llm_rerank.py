"""Regression checks for the rerank fallback and metadata boundary; no API calls."""
from types import SimpleNamespace

from scripts.run_stale_llm_rerank import repair_order, request_payload


def test_duplicate_unknown_missing_ids_keep_first_then_bge_order():
    order, audit = repair_order('{"ranked_ids":["d","b","d","unknown",7]}', ["a", "b", "c", "d"])
    assert order == ["d", "b", "a", "c"]
    assert audit["duplicate_ids"] == ["d"]
    assert audit["unknown_ids"] == ["unknown"]
    assert audit["missing_ids"] == ["a", "c"]
    assert audit["non_string_items"] == 1


def test_unparseable_or_wrong_schema_falls_back_without_new_request():
    for raw in [None, "invalid", '{"ranked_ids": "a"}', "[]", '{}']:
        order, audit = repair_order(raw, ["a", "b"])
        assert order == ["a", "b"]
        assert audit["had_anomaly"]


def test_valid_order_has_no_anomaly():
    order, audit = repair_order('{"ranked_ids":["c","a","b"]}', ["a", "b", "c"])
    assert order == ["c", "a", "b"]
    assert not audit["had_anomaly"]


def test_request_contains_only_query_and_first_twenty_id_text_pairs():
    import json
    query = {"query_id": "query-secret", "scenario_id": "scene-secret", "content": "query text",
             "gold_memory_id": "gold-secret", "source_ref": {"type": "T1"}}
    dense = {"ranked_candidates": [{"memory_id": str(i), "score": 999-i} for i in range(25)]}
    memories = {str(i): SimpleNamespace(content=f"memory {i}", source_ref="gold-secret") for i in range(25)}
    result = request_payload(query, dense, memories, SimpleNamespace(llm_model="model"))
    user = json.loads(result["messages"][1]["content"])
    assert user == {"query": "query text", "candidates": [{"id": str(i), "text": f"memory {i}"} for i in range(20)]}
    assert result["reasoning_effort"] == "none"
    assert result["temperature"] == 0
    assert result["response_format"] == {"type": "json_object"}

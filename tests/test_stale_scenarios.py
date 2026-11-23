from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from memory_retrieval.api import APIError
from memory_retrieval.data import Memory, read_jsonl
from memory_retrieval.evaluation import validate_dataset
from scripts.augment_stale_scenarios import generate_candidates, review_batch
from memory_retrieval.data import write_jsonl
from scripts.convert_stale_scenarios import convert_scenario
from scripts.run_experiment import main
from memory_retrieval.dataset import Dataset


class StaleScenarioTests(unittest.TestCase):
    def test_converter_preserves_the_native_pair_and_extracts_only_prewrite_user_history(self):
        old = "I've been based in Seattle for the last few years."
        new = "I signed a long-term lease after moving to London."
        prewrite = "I bring homemade sourdough to the office every Thursday."
        memories, query, gold = convert_scenario({
            "uid": "scene", "type": "T1", "M_old": old, "M_new": new,
            "relevant_session_index": [0, 1],
            "haystack_session": [
                [{"role": "user", "content": prewrite + " Can you give me bread recipes?"},
                 {"role": "user", "content": old},
                 {"role": "assistant", "content": "I would serve bread with soup at the office."}],
                [{"role": "user", "content": new}],
                [{"role": "user", "content": "I have a European SIM card for my new phone."}],
            ],
        })
        self.assertEqual({row["content"] for row in memories}, {old, prewrite})
        self.assertEqual(query["content"], new)
        self.assertEqual(query["scenario_id"], "scene")
        native, = [row for row in memories if row["content"] == old]
        self.assertEqual(gold["conflict_old_memory_ids"], [native["memory_id"]])

    def test_cross_scene_gold_is_rejected_even_when_the_old_id_exists(self):
        memories = [Memory("a", "Statement A", "source", "one"),
                    Memory("b", "Statement B", "source", "two")]
        query = {"query_id": "q", "content": "Write", "scenario_id": "one"}
        with self.assertRaisesRegex(ValueError, "其他场景"):
            validate_dataset(memories, [query], [{"query_id": "q", "conflict_old_memory_ids": ["b"]}])

    def test_runner_scopes_bm25_before_top_k(self):
        write = "Saturn lithium optics satellite telemetry"
        memories = [Memory("a", "I track Saturn.", "source", "one"),
                    Memory("b", write, "source", "two"),
                    Memory("c", "I bake bread.", "source", "two")]
        queries = [{"query_id": f"q{i}", "scenario_id": sid, "content": write}
                   for i, sid in enumerate(("one", "two"))]
        gold = [{"query_id": q["query_id"], "conflict_old_memory_ids": [mid], "fully_reviewed": False,
                 "target_details": {mid: {"source": "stale_native", "relation": "official_stale"}}}
                for q, mid in zip(queries, ("a", "b"))]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, rows in (("memories", [m.to_dict() for m in memories]), ("queries", queries), ("gold", gold)):
                write_jsonl(root / f"{name}.jsonl", rows)
            with redirect_stdout(StringIO()):
                run = main(["--dataset", tmp, "--output-root", str(root / "runs"), "--arms", "bm25",
                            "--limit", "0"])
            results = {row["query_id"]: row for row in read_jsonl(run / "arms" / "bm25.jsonl")}
            summary = json.loads((run / "report.json").read_text(encoding="utf-8"))
        self.assertEqual([results[q]["library_size"] for q in ("q0", "q1")], [1, 2])
        self.assertEqual([results[q]["ranked_candidates"][0]["memory_id"] for q in ("q0", "q1")], ["a", "b"])
        retrieval = summary["arms"]["bm25"]["at_k"]["1"]
        self.assertEqual(retrieval["recall_by_source"]["stale_native"]["candidate_recall_micro"], 1)
        self.assertIsNone(retrieval["candidate_purity"])

    def test_synthetic_targets_drop_negation_and_copied_wording(self):
        new = "I spend mornings running retail orders at a downtown bookstore."
        survivor = "My weekday shifts take up the entire first half of the day at the airport."
        client = Mock()
        client.json_call.side_effect = lambda task, prompt, payload, validate: validate({"implicit": [
            survivor,
            "I never leave my apartment before the lunch break.",
            "I spend mornings auditing retail orders at a downtown bookstore.",
        ]})
        candidates, rejected = generate_candidates(client, {"new_memory": new}, [], {}, 0, ["implicit"])
        self.assertEqual([row["content"] for row in candidates], [survivor])
        self.assertEqual(len(rejected), 2)

    def test_split_synthetic_review_preserves_original_positions(self):
        client = Mock()

        def response(task, prompt, payload, validate):
            texts = payload["candidates"]
            if len(texts) > 2:
                raise APIError("HTTP 524")
            return validate({"candidates": [{"id": 0, "relation": "stale", "reason": "Different workday."},
                                             {"id": 1, "relation": "compatible", "reason": "Compatible routine."}]})

        client.json_call.side_effect = response
        labels = review_batch(client, {}, ["A", "B", "C", "D"])
        self.assertEqual([row["id"] for row in labels["candidates"] if row["relation"] == "stale"], [0, 2])
        self.assertEqual([row["id"] for row in labels["candidates"] if row["relation"] == "compatible"], [1, 3])

    def test_encoders_keep_the_global_vector_positions_of_each_scope(self):
        memories = [Memory("a", "A", "text", "one"), Memory("b", "B", "text", "two"),
                    Memory("c", "C", "text", "one")]
        queries = [{"query_id": "q", "scenario_id": "one"}, {"query_id": "p", "scenario_id": "two"}]
        self.assertEqual(Dataset({}, memories, queries, []).groups(), {"one": ([0, 2], [0]), "two": ([1], [1])})


if __name__ == "__main__":
    unittest.main()

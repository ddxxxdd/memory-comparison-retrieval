import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_dataset import convert_halumem, convert_stale, iter_json_array, judge_jobs


class DatasetBuilderTests(unittest.TestCase):
    def test_halumem_updates_link_to_the_same_users_old_memory_bank(self):
        source = [{
            "uuid": "user-1",
            "sessions": [{"memory_points": [
                {"index": 1, "memory_content": "I live in Seattle.", "is_update": "False"},
                {"index": 2, "memory_content": "I now live in Portland.", "is_update": "True",
                 "original_memories": ["I live in Seattle.", "I used to live in Denver."]},
            ]}],
        }, {
            "uuid": "user-2",
            "sessions": [{"memory_points": [
                {"index": 1, "memory_content": "I live in Seattle.", "is_update": "False"},
            ]}],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "halumem.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in source), encoding="utf-8")

            memories, queries, gold, jobs = convert_halumem(path)

        self.assertEqual(len(queries), 1)
        self.assertEqual(len(gold), 1)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(memories), 3)
        self.assertEqual(len({memory["memory_id"] for memory in memories}), 3)
        self.assertEqual(gold[0]["gold_compare_ids"], [job[2] for job in jobs])
        self.assertEqual({job[3] for job in jobs}, {"I live in Seattle.", "I used to live in Denver."})

    def test_stale_reader_handles_chunk_boundaries_and_creates_one_pair_per_row(self):
        rows = [{"uid": "u1", "M_old": "old state", "M_new": "new state", "type": "T1"},
                {"uid": "u2", "M_old": "before", "M_new": "after", "type": "T2"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stale.json"
            path.write_text(json.dumps(rows), encoding="utf-8")

            self.assertEqual(list(iter_json_array(path, chunk_size=13)), rows)
            memories, queries, gold, jobs = convert_stale(path)

        self.assertEqual(len(memories), 2)
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(gold), 2)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(gold[0]["gold_compare_ids"], [memories[0]["memory_id"]])

    def test_pair_judgments_preserve_uncertain_and_fail_on_invalid_labels(self):
        jobs = [("q1", "new", "m1", "old")]
        labels = judge_jobs(jobs, lambda _query, _memory: {"conflict_label": "uncertain"}, workers=1)
        self.assertEqual(labels[("q1", "m1")], "uncertain")
        with self.assertRaisesRegex(RuntimeError, "配对判定失败"):
            judge_jobs(jobs, lambda _query, _memory: {"conflict_label": "maybe"}, workers=1)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from unittest.mock import patch

from scripts.build_mixed_dataset import (
    export_rows, filter_source_groups, merge_bank, review_full_bank_parallel,
    review_parallel, source_groups, validate_review,
)


def pair(a, b, label, triple_a, triple_b):
    return {"sentence1": a, "sentence2": b, "label": label, "dtype": "numerics",
            "triple1": triple_a, "triple2": triple_b,
            **{f"annotation{i}": label for i in (1, 2, 3)}}


def checked(comparable, non_comparable=(), rejected=()):
    return {"query_usable": True, "query_reason": "direct statement",
            "comparable": [{"index": i, "label": label, "reason": "direct fact"}
                           for i, label in comparable],
            "non_comparable": list(non_comparable),
            "rejected": [{"index": i, "reason": "missing scope"} for i in rejected]}


class MixedDatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            pair("I have two cats.", "I own two cats.", "positive",
                 ["i", "have_pet", "2 cat"], ["i", "have_pet", "2 cat"]),
            pair("I have two cats.", "I have three cats.", "negative",
                 ["i", "have_pet", "2 cat"], ["i", "have_pet", "3 cat"]),
        ]
        self.group = source_groups(self.rows)[0]

    def test_same_anchor_needs_both_labels_and_unanimous_annotations(self):
        self.assertEqual(len(source_groups(self.rows)), 1)
        self.assertEqual(source_groups(self.rows[:1]), [])
        self.rows[1]["annotation3"] = "neutral"
        self.assertEqual(source_groups(self.rows), [])

    def test_contradictory_source_pairs_and_different_subjects_are_excluded(self):
        bad = pair("I have two cats.", "I have three cats.", "positive",
                   ["i", "have_pet", "2 cat"], ["i", "have_pet", "3 cat"])
        self.assertEqual(source_groups(self.rows + [bad]), [])
        self.rows[1]["triple2"][0] = "my mother"
        self.assertEqual(source_groups(self.rows), [])

    def test_full_bank_review_cannot_skip_duplicate_or_use_a_third_label(self):
        validate_review(checked([(0, "conflict")], [1]), 2)
        for value in (checked([(0, "conflict")]),
                      checked([(0, "conflict")], [0]),
                      checked([(0, "uncertain")], [1])):
            with self.assertRaises(ValueError):
                validate_review(value, 2)

    def test_source_disagreements_are_not_silently_relabelled(self):
        result = checked([(0, "no_conflict"), (1, "no_conflict")])
        self.assertEqual(filter_source_groups([self.group], {self.group["content"]: result}), [])

    def test_failed_reviews_are_reported_so_unresolved_groups_can_be_excluded(self):
        groups = [{"content": "ok"}, {"content": "fail"}]

        def review_fn(query, memories):
            if query == "fail":
                raise RuntimeError("unavailable")
            return {"query": query}

        reviews, failures = review_parallel(groups, lambda _: [], review_fn, 1, "test")
        self.assertEqual(reviews, {"ok": {"query": "ok"}})
        self.assertEqual(failures, [{"query": "fail", "error": "RuntimeError: unavailable"}])

    def test_full_bank_chunks_remap_indexes_and_cover_every_memory_once(self):
        def fake_review(_settings, _query, memories):
            comparable = [(i, "conflict" if "conflict" in item else "no_conflict")
                          for i, item in enumerate(memories) if "conflict" in item or "same" in item]
            non_comparable = [i for i, item in enumerate(memories)
                              if "conflict" not in item and "same" not in item]
            return checked(comparable, non_comparable)

        memories = ["conflict a", "unrelated", "same fact", "unrelated", "conflict b"]
        with patch("scripts.build_mixed_dataset.review", side_effect=fake_review):
            results, failures = review_full_bank_parallel(
                [{"content": "query"}], memories, None, workers=1, chunk_size=2)
        self.assertEqual(failures, [])
        result = results["query"]
        self.assertEqual({item["index"] for item in result["comparable"]}, {0, 2, 4})
        self.assertEqual(result["non_comparable"], [1, 3])
        validate_review(result, len(memories))

    def test_size_limit_preserves_whole_candidate_groups(self):
        groups, memories = merge_bank([self.group], 1, "hash")
        self.assertEqual((groups, memories), ([], []))
        groups, memories = merge_bank([self.group], 2, "hash")
        self.assertEqual(len(memories), 2)
        self.assertEqual({m["content"] for m in memories}, set(self.group["pairs"]))
        self.assertNotIn("source_label", json.dumps(memories))
        self.assertNotIn("annotations", json.dumps(memories))

    def test_full_bank_gold_adds_cross_group_matches_and_excludes_unresolved_queries(self):
        groups, memories = merge_bank([self.group], 2500, "hash")
        memories.append({"memory_id": "extra", "content": "I keep two cats.",
                         "source_ref": "another source group"})
        result = checked([(0, "no_conflict"), (1, "conflict"), (2, "no_conflict")])
        queries, gold, excluded = export_rows(groups, memories, {self.group["content"]: result}, "hash")
        self.assertEqual(len(queries), 1)
        self.assertEqual(gold[0]["conflict_labels"]["extra"], "no_conflict")
        self.assertEqual(len(gold[0]["gold_compare_ids"]), 3)
        self.assertFalse(gold[0]["fully_reviewed"])
        self.assertFalse(gold[0]["human_reviewed"])
        result = checked([(0, "no_conflict"), (1, "conflict")], rejected=[2])
        queries, gold, excluded = export_rows(groups, memories, {self.group["content"]: result}, "hash")
        self.assertEqual((queries, gold), ([], []))
        self.assertEqual(excluded, [self.group["content"]])

    def test_exact_query_memory_is_always_a_no_conflict_candidate(self):
        groups, memories = merge_bank([self.group], 2500, "hash")
        exact_id = "same-as-query"
        memories.append({"memory_id": exact_id, "content": self.group["content"],
                         "source_ref": "same text"})
        result = checked([(0, "no_conflict"), (1, "conflict")], rejected=[2])

        queries, gold, excluded = export_rows(
            groups, memories, {self.group["content"]: result}, "hash")

        self.assertEqual(len(queries), 1)
        self.assertEqual(excluded, [])
        self.assertIn(exact_id, gold[0]["gold_compare_ids"])
        self.assertEqual(gold[0]["conflict_labels"][exact_id], "no_conflict")


if __name__ == "__main__":
    unittest.main()

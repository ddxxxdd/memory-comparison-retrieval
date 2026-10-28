import unittest

from scripts.probe_description_quality import eligible


class EligiblePoolTests(unittest.TestCase):
    def test_control_pool_includes_non_gold_memories_from_other_users_and_scopes(self):
        memories = [
            {"memory_id": "target", "source_ref": {"scope": "personal", "user_id": "U1"}},
            {"memory_id": "other-user", "source_ref": {"scope": "personal", "user_id": "U2"}},
            {"memory_id": "public", "source_ref": {"scope": "world", "user_id": None}},
        ]

        self.assertEqual(eligible(memories, {"target"}), ["other-user", "public"])


if __name__ == "__main__":
    unittest.main()

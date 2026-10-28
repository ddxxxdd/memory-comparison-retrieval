import unittest

from scripts.run_experiment import select_queries


class ExperimentRunnerTests(unittest.TestCase):
    def test_query_selection_is_repeatable_and_uses_one_shared_batch(self):
        queries = [{"query_id": f"q{i}", "content": str(i)} for i in range(20)]
        first = select_queries(queries, limit=7, seed=19)
        second = select_queries(queries, limit=7, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(select_queries(queries, limit=None, seed=19), queries)

if __name__ == "__main__":
    unittest.main()

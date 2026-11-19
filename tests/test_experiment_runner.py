import unittest

import numpy as np

from memory_retrieval.ocsp import ORTHO_GAMMA, compute_subspace_ocsp_scores
from scripts.run_experiment import select_queries


class ExperimentRunnerTests(unittest.TestCase):
    def test_qr_score_preserves_the_frozen_gate_and_the_probes_only_ablation(self):
        e1, e2, e3 = np.eye(3)
        probe = np.array([[0.8, 0.0, 0.6]])
        scores = compute_subspace_ocsp_scores(e1, e2[None, :], probe, np.array([e1, e3, -e1]))
        self.assertEqual(ORTHO_GAMMA, 0.65)
        np.testing.assert_allclose(scores["ocsp_qr"], [0.64, 0.6 * (0.6 + 0.65), 0])
        np.testing.assert_allclose(scores["probes_only_no_ortho"], [0.8, 0.6, -0.8])

    def test_query_selection_is_repeatable_and_uses_one_shared_batch(self):
        queries = [{"query_id": f"q{i}", "content": str(i)} for i in range(20)]
        first = select_queries(queries, limit=7, seed=19)
        second = select_queries(queries, limit=7, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(select_queries(queries, limit=None, seed=19), queries)

    def test_scenario_selection_includes_all_queries_in_requested_banks(self):
        queries = [{"query_id": f"q{i}", "scenario_id": sid}
                   for i, sid in enumerate(("a", "b", "a", "c"))]
        selected = select_queries(queries, limit=1, seed=19, scenario_ids=["b", "a", "b"])

        self.assertEqual([row["query_id"] for row in selected], ["q1", "q0", "q2"])
        with self.assertRaisesRegex(ValueError, "missing"):
            select_queries(queries, limit=None, seed=19, scenario_ids=["a", "missing"])

if __name__ == "__main__":
    unittest.main()

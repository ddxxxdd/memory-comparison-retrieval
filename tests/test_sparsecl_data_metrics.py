import math
import unittest

from scripts.run_sparsecl_data_baselines import metrics_for_ranks


class SparseCLDataMetricsTests(unittest.TestCase):
    def test_perfect_three_targets(self):
        result = metrics_for_ranks([1, 2, 3])
        self.assertEqual(result["1"]["recall"], 1/3)
        self.assertEqual(result["1"]["ndcg"], 1)
        self.assertEqual(result["5"]["recall"], 1)
        self.assertEqual(result["5"]["ndcg"], 1)
        self.assertEqual(result["5"]["all_found"], 1)

    def test_missing_and_displaced_targets(self):
        result = metrics_for_ranks([3, 5, 11])
        expected = (1/math.log2(4) + 1/math.log2(6)) / (1 + 1/math.log2(3) + 0.5)
        self.assertEqual(result["1"]["hit"], 0)
        self.assertEqual(result["5"]["hits"], 2)
        self.assertEqual(result["10"]["recall"], 2/3)
        self.assertEqual(result["10"]["all_found"], 0)
        self.assertAlmostEqual(result["10"]["ndcg"], expected)


if __name__ == "__main__":
    unittest.main()

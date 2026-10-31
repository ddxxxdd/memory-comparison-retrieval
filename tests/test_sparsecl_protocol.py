import math
import unittest

import numpy as np

from memory_retrieval.sparsecl import fused_order, ndcg_at_10, score_candidates
from scripts.sparsecl_worker import input_rows


class SparseCLProtocolTests(unittest.TestCase):
    def test_scores_match_the_reproduced_scalar_formula(self):
        rng = np.random.default_rng(17)
        cosine_query = rng.normal(size=768).astype(np.float32)
        sparsecl_query = rng.normal(size=768).astype(np.float32)
        cosine_memories = rng.normal(size=(4, 768)).astype(np.float32)
        sparsecl_memories = rng.normal(size=(4, 768)).astype(np.float32)
        cosine_query /= np.linalg.norm(cosine_query)
        sparsecl_query /= np.linalg.norm(sparsecl_query)
        cosine_memories /= np.linalg.norm(cosine_memories, axis=1, keepdims=True)
        sparsecl_memories /= np.linalg.norm(sparsecl_memories, axis=1, keepdims=True)

        cosine, hoyer = score_candidates(cosine_query, cosine_memories,
                                         sparsecl_query, sparsecl_memories)
        for position in range(len(cosine)):
            cosine_reference = np.dot(cosine_memories[position], cosine_query) / (
                np.linalg.norm(cosine_memories[position]) * np.linalg.norm(cosine_query))
            difference = sparsecl_memories[position] - sparsecl_query
            difference_l2 = np.linalg.norm(difference, ord=2)
            if difference_l2 < 1e-3:
                hoyer_reference = -1e9
            else:
                hoyer_reference = (math.sqrt(768) - np.linalg.norm(difference, ord=1) / difference_l2) / (
                    math.sqrt(768) - 1)
            self.assertAlmostEqual(float(cosine[position]), float(cosine_reference), places=6)
            self.assertAlmostEqual(float(hoyer[position]), float(hoyer_reference), places=6)

    def test_hoyer_equal_vectors_use_official_small_difference_value(self):
        query = np.ones(768, dtype=np.float32)
        query /= np.linalg.norm(query)
        cosine, hoyer = score_candidates(query, query.reshape(1, -1),
                                         query, query.reshape(1, -1))
        self.assertAlmostEqual(float(cosine[0]), 1.0, places=6)
        self.assertEqual(float(hoyer[0]), -1e9)

    def test_fused_score_ties_keep_the_hnsw_input_order(self):
        order, scores = fused_order(np.array([0.4, 0.4, 0.5]), np.zeros(3), 0.18)
        self.assertEqual(order.tolist(), [2, 0, 1])
        self.assertEqual(scores.tolist(), [0.4, 0.4, 0.5])

    def test_binary_ndcg_at_ten_uses_conflict_ids_as_relevance(self):
        gold = [{"query_id": "q", "conflict_old_memory_ids": ["a", "b"]}]
        perfect = ndcg_at_10(gold, {"q": ["a", "b", "other"]})
        swapped_with_nonrelevant = ndcg_at_10(gold, {"q": ["a", "other", "b"]})
        self.assertEqual(perfect, 1.0)
        self.assertLess(swapped_with_nonrelevant, perfect)

    def test_equal_memory_texts_keep_their_original_ids(self):
        memories, queries = input_rows({
            "memories": [{"memory_id": "m1", "content": "same body"},
                         {"memory_id": "m2", "content": "same body"}],
            "queries": [{"query_id": "q1", "content": "query body"}],
        })
        self.assertEqual([row["memory_id"] for row in memories], ["m1", "m2"])
        self.assertEqual(len(queries), 1)


if __name__ == "__main__":
    unittest.main()

"""SparseCL scoring and ranking primitives shared by the WSL worker and tests."""

from __future__ import annotations

import math

import numpy as np


INITIAL_RETRIEVAL_DEPTH = 1000
FIXED_ARGUANA_ALPHA = 0.18
SPARSECL_UPSTREAM_COMMIT = "14fcd8bcfeef2603f37b1e805cd3626800430711"
SPARSECL_SCORE_FORMULA = "cosine + alpha * hoyer(normalized_sparsecl_query - normalized_sparsecl_memory)"


def score_candidates(cosine_query, cosine_memories, sparsecl_query, sparsecl_memories):
    """Apply the reproduced cosine and Hoyer score to a retrieved candidate set."""
    cosine_query = np.asarray(cosine_query, dtype=np.float32)
    cosine_memories = np.asarray(cosine_memories, dtype=np.float32)
    sparsecl_query = np.asarray(sparsecl_query, dtype=np.float32)
    sparsecl_memories = np.asarray(sparsecl_memories, dtype=np.float32)
    dimension_root = math.sqrt(sparsecl_query.shape[0])
    cosine_scores = []
    hoyer_scores = []
    for cosine_memory, sparsecl_memory in zip(cosine_memories, sparsecl_memories):
        cosine = np.dot(cosine_memory, cosine_query)
        cosine /= np.linalg.norm(cosine_memory) * np.linalg.norm(cosine_query)
        difference = sparsecl_memory - sparsecl_query
        difference_l1 = np.linalg.norm(difference, ord=1)
        difference_l2 = np.linalg.norm(difference, ord=2)
        if difference_l2 < 1e-3:
            hoyer = -1e9
        else:
            hoyer = (dimension_root - difference_l1 / difference_l2) / (dimension_root - 1)
        cosine_scores.append(float(cosine))
        hoyer_scores.append(float(hoyer))
    return np.asarray(cosine_scores, dtype=np.float64), np.asarray(hoyer_scores, dtype=np.float64)


def fused_order(cosine, hoyer, alpha):
    """Sort descending by fused score; stable ties retain the HNSW result order."""
    scores = np.asarray(cosine, dtype=np.float64) + float(alpha) * np.asarray(hoyer, dtype=np.float64)
    return np.argsort(-scores, kind="stable"), scores


def ndcg_at_10(gold_rows, rankings):
    """Binary NDCG@10; queries with no positive gold receive 0 and remain in the mean."""
    values = []
    gold_by_id = {row["query_id"]: set(row["conflict_old_memory_ids"]) for row in gold_rows}
    if set(rankings) != set(gold_by_id):
        raise ValueError("NDCG 排名必须恰好覆盖全部金标查询")
    for query_id, targets in gold_by_id.items():
        if not targets:
            values.append(0.0)
            continue
        ranked = rankings[query_id][:10]
        dcg = sum(1.0 / math.log2(position + 2)
                  for position, memory_id in enumerate(ranked)
                  if memory_id in targets)
        ideal_count = min(10, len(targets))
        ideal = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
        values.append(dcg / ideal)
    return float(np.mean(values)) if values else None

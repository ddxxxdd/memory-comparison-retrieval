import numpy as np
import pytest

from scripts.run_stale_qwen_embedding import decode_embeddings


def test_embedding_indices_restored_before_l2_normalization():
    response = {"data": [{"index": 1, "embedding": [0., 4.]}, {"index": 0, "embedding": [3., 4.]}]}
    vectors = decode_embeddings(response, 2)
    np.testing.assert_array_equal(vectors, np.array([[.6, .8], [0., 1.]], dtype=np.float32))
    assert vectors.dtype == np.float32


@pytest.mark.parametrize('response,count', [
    ({'data': []}, 1),
    ({'data': [{'index': 0, 'embedding': [1., 0.]}, {'index': 0, 'embedding': [0., 1.]}]}, 2),
    ({'data': [{'index': 0, 'embedding': [0., 0.]}]}, 1),
    ({'data': [{'index': 0, 'embedding': [float('nan'), 0.]}]}, 1),
])
def test_invalid_vectors_or_indices_are_rejected(response, count):
    with pytest.raises(ValueError):
        decode_embeddings(response, count)


def test_explicit_dimension_is_enforced_but_default_is_inferred():
    response = {'data': [{'index': 0, 'embedding': [1., 0., 0.]}]}
    assert decode_embeddings(response, 1).shape == (1, 3)
    with pytest.raises(ValueError):
        decode_embeddings(response, 1, 2)


def test_repeated_methods_retain_all_pytrec_cutoffs_after_native_evaluation():
    from memory_retrieval.data import Memory
    from memory_retrieval.dataset import Dataset
    from scripts.run_stale_qwen_embedding import evaluate
    memories = [Memory('a', 'first', 'text', 'scene'), Memory('b', 'second', 'text', 'scene')]
    queries = [{'query_id': 'q', 'scenario_id': 'scene', 'source_ref': {'type': 'T1'}}]
    gold = [{'query_id': 'q', 'conflict_old_memory_ids': ['a'],
             'target_details': {'a': {'source': 'stale_native', 'relation': 'conflict'}}}]
    dataset = Dataset({}, memories, queries, gold)
    rows = {name: {'q': {'ranked_candidates': [{'memory_id': 'a'}, {'memory_id': 'b'}]}}
            for name in ['first', 'second', 'third']}
    groups, per_query, verification = evaluate(dataset, rows)
    for name in rows:
        assert per_query[name]['q'] == 1.
        for k in (3, 5, 10):
            assert groups['all']['methods'][name]['at_k'][str(k)]['hits'] == 1
    assert verification['max_recall_error'] == 0.

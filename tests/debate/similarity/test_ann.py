"""Tests for aragora/debate/similarity/ann.py

Covers:
- _require_numpy: noop when HAS_NUMPY=True, raises ImportError when False
- compute_pairwise_matrix: empty array, single embedding, identical embeddings,
  orthogonal embeddings, zero-norm handling
- compute_batch_similarity_fast: n<2 → 1.0, identical → ~1.0,
  exclude_diagonal True/False, diverse embeddings lower similarity
- compute_min_similarity: n<2 → 1.0, identical, diverse
- find_convergence_threshold: n<2 → (True, 1.0), identical → converged,
  diverse → not converged, threshold edge cases, early termination n>20
- cluster_by_similarity: empty, single, two similar, two dissimilar,
  min_cluster_size filtering
- count_unique_fast: empty, single, identical duplicates, diverse
- FAISSIndex: init without faiss (is_available=False), add+search fallback,
  reset, 1d query reshape
- compute_argument_diversity_optimized: delegates to count_unique_fast
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unit(v: list[float]) -> np.ndarray:
    """Return a unit-normalised 1-D float64 vector."""
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# Standard test fixtures used across many tests
IDENTICAL_3D = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
ORTHOGONAL_3D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ===========================================================================
# _require_numpy
# ===========================================================================


class TestRequireNumpy:
    def test_noop_when_numpy_available(self):
        """_require_numpy must not raise when HAS_NUMPY is True."""
        from aragora.debate.similarity.ann import HAS_NUMPY, _require_numpy

        assert HAS_NUMPY is True  # sanity — numpy IS present in this environment
        _require_numpy("test_op")  # must not raise

    def test_raises_when_numpy_unavailable(self):
        """_require_numpy raises ImportError when HAS_NUMPY is False."""
        import aragora.debate.similarity.ann as ann_module

        with patch.object(ann_module, "HAS_NUMPY", False):
            with pytest.raises(ImportError, match="numpy is required for custom_op"):
                ann_module._require_numpy("custom_op")

    def test_error_message_contains_operation_name(self):
        import aragora.debate.similarity.ann as ann_module

        with patch.object(ann_module, "HAS_NUMPY", False):
            with pytest.raises(ImportError) as exc_info:
                ann_module._require_numpy("matrix_multiply")
            assert "matrix_multiply" in str(exc_info.value)
            assert "pip install numpy" in str(exc_info.value)


# ===========================================================================
# compute_pairwise_matrix
# ===========================================================================


class TestComputePairwiseMatrix:
    def test_empty_array_returns_empty(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        result = compute_pairwise_matrix(np.array([]))
        assert len(result) == 0

    def test_single_embedding_produces_1x1_matrix(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        emb = np.array([[1.0, 0.0, 0.0]])
        result = compute_pairwise_matrix(emb)
        assert result.shape == (1, 1)
        assert pytest.approx(result[0, 0], abs=1e-6) == 1.0

    def test_identical_embeddings_produce_all_ones(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        result = compute_pairwise_matrix(IDENTICAL_3D)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result, np.ones((3, 3)), atol=1e-6)

    def test_orthogonal_embeddings_produce_zero_off_diagonal(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        result = compute_pairwise_matrix(ORTHOGONAL_3D)
        assert result.shape == (3, 3)
        # Diagonal should be ~1.0
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0, 1.0], atol=1e-6)
        # Off-diagonal should be ~0.0
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(result[i, j]) < 1e-6, f"result[{i},{j}] = {result[i, j]}"

    def test_zero_norm_embedding_does_not_raise(self):
        """A zero-vector embedding must not trigger division-by-zero."""
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        emb = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Should run without error
        result = compute_pairwise_matrix(emb)
        assert result.shape == (2, 2)

    def test_returns_symmetric_matrix(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((5, 8))
        result = compute_pairwise_matrix(emb)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_diagonal_is_one_for_nonzero_embeddings(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        rng = np.random.default_rng(7)
        emb = rng.standard_normal((4, 6))
        result = compute_pairwise_matrix(emb)
        np.testing.assert_allclose(np.diag(result), np.ones(4), atol=1e-6)

    def test_values_in_range_minus_one_to_one(self):
        from aragora.debate.similarity.ann import compute_pairwise_matrix

        rng = np.random.default_rng(99)
        emb = rng.standard_normal((6, 10))
        result = compute_pairwise_matrix(emb)
        assert np.all(result >= -1.0 - 1e-9)
        assert np.all(result <= 1.0 + 1e-9)


# ===========================================================================
# compute_batch_similarity_fast
# ===========================================================================


class TestComputeBatchSimilarityFast:
    def test_empty_array_returns_zero(self):
        """n=0: len<2 triggers the n<2 branch → 1.0, but empty len is 0."""
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        # Empty array: n=0 < 2 → returns 1.0
        result = compute_batch_similarity_fast(np.array([]))
        assert result == 1.0

    def test_single_item_returns_one(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        emb = np.array([[1.0, 0.0, 0.0]])
        assert compute_batch_similarity_fast(emb) == 1.0

    def test_identical_embeddings_returns_near_one(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        result = compute_batch_similarity_fast(IDENTICAL_3D)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_orthogonal_embeddings_returns_near_zero(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        result = compute_batch_similarity_fast(ORTHOGONAL_3D)
        assert abs(result) < 1e-6

    def test_exclude_diagonal_true_vs_false(self):
        """Including diagonal (self-similarity = 1.0) raises the average."""
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        # Orthogonal: off-diagonal = 0, diagonal = 1
        with_diag = compute_batch_similarity_fast(ORTHOGONAL_3D, exclude_diagonal=False)
        without_diag = compute_batch_similarity_fast(ORTHOGONAL_3D, exclude_diagonal=True)
        assert with_diag > without_diag

    def test_diverse_embeddings_lower_than_similar(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        similar = np.array([[1.0, 0.01, 0.0], [1.0, 0.02, 0.0], [1.0, -0.01, 0.0]])
        diverse = ORTHOGONAL_3D
        assert compute_batch_similarity_fast(similar) > compute_batch_similarity_fast(diverse)

    def test_returns_float(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        result = compute_batch_similarity_fast(IDENTICAL_3D)
        assert isinstance(result, float)

    def test_two_identical_embeddings(self):
        from aragora.debate.similarity.ann import compute_batch_similarity_fast

        emb = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert pytest.approx(compute_batch_similarity_fast(emb), abs=1e-6) == 1.0


# ===========================================================================
# compute_min_similarity
# ===========================================================================


class TestComputeMinSimilarity:
    def test_single_item_returns_one(self):
        from aragora.debate.similarity.ann import compute_min_similarity

        emb = np.array([[1.0, 0.0, 0.0]])
        assert compute_min_similarity(emb) == 1.0

    def test_empty_returns_one(self):
        from aragora.debate.similarity.ann import compute_min_similarity

        assert compute_min_similarity(np.array([])) == 1.0

    def test_identical_embeddings_returns_one(self):
        from aragora.debate.similarity.ann import compute_min_similarity

        result = compute_min_similarity(IDENTICAL_3D)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_orthogonal_embeddings_returns_near_zero(self):
        from aragora.debate.similarity.ann import compute_min_similarity

        result = compute_min_similarity(ORTHOGONAL_3D)
        assert abs(result) < 1e-6

    def test_mixed_returns_minimum(self):
        """Min should reflect the most dissimilar pair."""
        from aragora.debate.similarity.ann import compute_min_similarity

        # Two identical + one orthogonal: min pair sim is ~0
        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = compute_min_similarity(emb)
        assert result < 0.1

    def test_returns_float(self):
        from aragora.debate.similarity.ann import compute_min_similarity

        result = compute_min_similarity(IDENTICAL_3D)
        assert isinstance(result, float)


# ===========================================================================
# find_convergence_threshold
# ===========================================================================


class TestFindConvergenceThreshold:
    def test_single_item_returns_converged(self):
        from aragora.debate.similarity.ann import find_convergence_threshold

        emb = np.array([[1.0, 0.0, 0.0]])
        converged, min_sim = find_convergence_threshold(emb)
        assert converged is True
        assert min_sim == 1.0

    def test_empty_array_returns_converged(self):
        from aragora.debate.similarity.ann import find_convergence_threshold

        converged, min_sim = find_convergence_threshold(np.array([]))
        assert converged is True
        assert min_sim == 1.0

    def test_identical_embeddings_converged(self):
        from aragora.debate.similarity.ann import find_convergence_threshold

        converged, min_sim = find_convergence_threshold(IDENTICAL_3D, threshold=0.85)
        assert converged is True
        assert min_sim >= 0.85

    def test_orthogonal_embeddings_not_converged(self):
        from aragora.debate.similarity.ann import find_convergence_threshold

        converged, min_sim = find_convergence_threshold(ORTHOGONAL_3D, threshold=0.85)
        assert converged is False
        assert min_sim < 0.85

    def test_threshold_zero_always_converged(self):
        """With threshold=0.0, any similarity should be above threshold."""
        from aragora.debate.similarity.ann import find_convergence_threshold

        converged, _ = find_convergence_threshold(ORTHOGONAL_3D, threshold=0.0)
        assert converged is True

    def test_threshold_one_only_converged_for_identical(self):
        """threshold=1.0 — only identical (all-ones matrix) passes."""
        from aragora.debate.similarity.ann import find_convergence_threshold

        converged_id, _ = find_convergence_threshold(IDENTICAL_3D, threshold=1.0)
        assert converged_id is True
        converged_orth, _ = find_convergence_threshold(ORTHOGONAL_3D, threshold=1.0)
        assert converged_orth is False

    def test_return_type_is_tuple_bool_float(self):
        from aragora.debate.similarity.ann import find_convergence_threshold

        result = find_convergence_threshold(IDENTICAL_3D)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_early_termination_path_n_gt_20(self):
        """n=25 forces the row-by-row path; orthogonal → not converged."""
        from aragora.debate.similarity.ann import find_convergence_threshold

        rng = np.random.default_rng(1)
        # Build 25 embeddings that are sufficiently spread out
        emb = rng.standard_normal((25, 16))
        converged, min_sim = find_convergence_threshold(emb, threshold=0.99)
        # Very unlikely that 25 random vectors all exceed 0.99 cosine similarity
        assert converged is False

    def test_early_termination_with_similar_large_batch(self):
        """n=25 identical embeddings → converged even on row-by-row path."""
        from aragora.debate.similarity.ann import find_convergence_threshold

        emb = np.tile(np.array([1.0, 0.0, 0.0]), (25, 1))
        converged, min_sim = find_convergence_threshold(emb, threshold=0.85)
        assert converged is True
        assert pytest.approx(min_sim, abs=1e-6) == 1.0

    def test_small_n_uses_matrix_path(self):
        """n=5 (≤20) uses full matrix path; result consistent with min_sim."""
        from aragora.debate.similarity.ann import find_convergence_threshold

        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.95, 0.05, 0.0],
            ]
        )
        converged, min_sim = find_convergence_threshold(emb, threshold=0.5)
        assert converged is True
        assert min_sim >= 0.5


# ===========================================================================
# cluster_by_similarity
# ===========================================================================


class TestClusterBySimilarity:
    def test_empty_embeddings_returns_empty(self):
        from aragora.debate.similarity.ann import cluster_by_similarity

        result = cluster_by_similarity(np.array([]), threshold=0.8)
        assert result == []

    def test_single_embedding_returns_one_element_cluster(self):
        """n=1 early-return always produces [[0]] for any min_cluster_size."""
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array([[1.0, 0.0, 0.0]])
        result = cluster_by_similarity(emb, threshold=0.8, min_cluster_size=1)
        assert result == [[0]]

    def test_single_embedding_min_cluster_size_2_still_returns_single_cluster(self):
        """The n<2 early-return always emits [[0]] for n=1, regardless of min_cluster_size.
        The min_cluster_size filter only applies when the full loop runs (n>=2)."""
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array([[1.0, 0.0, 0.0]])
        result = cluster_by_similarity(emb, threshold=0.8, min_cluster_size=2)
        assert result == [[0]]

    def test_two_similar_embeddings_same_cluster(self):
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array([[1.0, 0.01, 0.0], [1.0, 0.02, 0.0]])
        result = cluster_by_similarity(emb, threshold=0.8, min_cluster_size=2)
        assert len(result) == 1
        assert set(result[0]) == {0, 1}

    def test_two_orthogonal_embeddings_separate_clusters(self):
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = cluster_by_similarity(emb, threshold=0.8, min_cluster_size=2)
        # Both items are dissimilar → no pair cluster meets min_cluster_size=2
        assert result == []

    def test_identical_embeddings_one_big_cluster(self):
        from aragora.debate.similarity.ann import cluster_by_similarity

        result = cluster_by_similarity(IDENTICAL_3D, threshold=0.8, min_cluster_size=2)
        assert len(result) == 1
        assert set(result[0]) == {0, 1, 2}

    def test_min_cluster_size_filtering(self):
        """3 identical + 2 identical: two valid clusters with min_cluster_size=2."""
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        result = cluster_by_similarity(emb, threshold=0.8, min_cluster_size=2)
        assert len(result) == 2
        cluster_sets = [set(c) for c in result]
        assert {0, 1, 2} in cluster_sets
        assert {3, 4} in cluster_sets

    def test_threshold_one_no_clusters(self):
        """threshold=1.0 and slightly different embeddings → no cluster."""
        from aragora.debate.similarity.ann import cluster_by_similarity

        emb = np.array([[1.0, 0.0, 0.0], [0.9999, 0.0001, 0.0]])
        result = cluster_by_similarity(emb, threshold=1.0, min_cluster_size=2)
        assert result == []

    def test_returns_list_of_lists(self):
        from aragora.debate.similarity.ann import cluster_by_similarity

        result = cluster_by_similarity(IDENTICAL_3D, threshold=0.8)
        assert isinstance(result, list)
        assert all(isinstance(c, list) for c in result)


# ===========================================================================
# count_unique_fast
# ===========================================================================


class TestCountUniqueFast:
    def test_empty_array_returns_zero_zero_zero(self):
        from aragora.debate.similarity.ann import count_unique_fast

        result = count_unique_fast(np.array([]))
        assert result == (0, 0, 0.0)

    def test_single_item_returns_one_one_one(self):
        from aragora.debate.similarity.ann import count_unique_fast

        emb = np.array([[1.0, 0.0, 0.0]])
        result = count_unique_fast(emb)
        assert result == (1, 1, 1.0)

    def test_identical_duplicates_low_unique_count(self):
        """All identical → max similarity for each is 1.0 → none are unique."""
        from aragora.debate.similarity.ann import count_unique_fast

        unique, total, diversity = count_unique_fast(IDENTICAL_3D, threshold=0.7)
        assert total == 3
        assert unique == 0
        assert diversity == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_all_unique(self):
        """Orthogonal vectors → max similarity to others is 0 < threshold → all unique."""
        from aragora.debate.similarity.ann import count_unique_fast

        unique, total, diversity = count_unique_fast(ORTHOGONAL_3D, threshold=0.7)
        assert total == 3
        assert unique == 3
        assert diversity == pytest.approx(1.0, abs=1e-6)

    def test_mixed_unique_and_duplicate(self):
        """Two identical + one unique orthogonal."""
        from aragora.debate.similarity.ann import count_unique_fast

        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        unique, total, diversity = count_unique_fast(emb, threshold=0.7)
        assert total == 3
        # The two x-axis items see each other with sim=1.0 → not unique
        # The y-axis item sees others with sim=0.0 < 0.7 → unique
        assert unique == 1
        assert diversity == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_returns_tuple_of_three(self):
        from aragora.debate.similarity.ann import count_unique_fast

        result = count_unique_fast(IDENTICAL_3D)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_diversity_score_between_zero_and_one(self):
        from aragora.debate.similarity.ann import count_unique_fast

        rng = np.random.default_rng(5)
        emb = rng.standard_normal((10, 8))
        _, _, diversity = count_unique_fast(emb, threshold=0.5)
        assert 0.0 <= diversity <= 1.0

    def test_use_faiss_false_uses_numpy_path(self):
        """use_faiss=False must still return correct results via numpy path."""
        from aragora.debate.similarity.ann import count_unique_fast

        unique, total, diversity = count_unique_fast(ORTHOGONAL_3D, threshold=0.7, use_faiss=False)
        assert total == 3
        assert unique == 3


# ===========================================================================
# FAISSIndex
# ===========================================================================


class TestFAISSIndex:
    """All tests run without real FAISS installed — exercises fallback path."""

    def test_is_available_false_without_faiss(self):
        """FAISSIndex.is_available must be False when faiss cannot be imported."""
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=4)
        assert index.is_available is False

    def test_init_stores_dimension(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=8)
        assert index.dimension == 8

    def test_add_stores_normalized_data_in_fallback(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        emb = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float64)
        index.add(emb)
        assert index._fallback_data is not None
        assert index._fallback_data.shape == (2, 3)
        # Check normalization: each row should be unit length
        norms = np.linalg.norm(index._fallback_data, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_add_multiple_times_concatenates(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(np.array([[1.0, 0.0, 0.0]]))
        index.add(np.array([[0.0, 1.0, 0.0]]))
        assert index._fallback_data.shape == (2, 3)

    def test_search_1d_query_is_reshaped(self):
        """A 1-D query must be silently reshaped to (1, d)."""
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        query_1d = np.array([1.0, 0.0, 0.0])
        indices, sims = index.search(query_1d, k=1)
        assert indices.shape[0] == 1
        assert int(indices[0, 0]) == 0  # most similar is the first embedding

    def test_search_returns_correct_nearest_neighbor(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(ORTHOGONAL_3D.copy())
        # Query closest to embedding 1 (y-axis)
        query = np.array([[0.0, 1.0, 0.0]])
        indices, sims = index.search(query, k=1)
        assert int(indices[0, 0]) == 1

    def test_search_empty_index_returns_empty(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        query = np.array([[1.0, 0.0, 0.0]])
        indices, sims = index.search(query, k=2)
        assert len(indices) == 0
        assert len(sims) == 0

    def test_reset_clears_fallback_data(self):
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(ORTHOGONAL_3D.copy())
        assert index._fallback_data is not None
        index.reset()
        assert index._fallback_data is None

    def test_search_k_larger_than_n_returns_all(self):
        """When k >= number of stored items, all items are returned."""
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        query = np.array([[1.0, 0.0, 0.0]])
        indices, sims = index.search(query, k=10)
        assert indices.shape[1] == 2  # only 2 items stored

    def test_search_with_2d_query(self):
        """Multiple query rows should each get their neighbors."""
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=3)
        index.add(ORTHOGONAL_3D.copy())
        query = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        indices, sims = index.search(query, k=1)
        assert indices.shape == (2, 1)
        assert int(indices[0, 0]) == 0
        assert int(indices[1, 0]) == 2

    def test_similarities_decrease_with_index_order(self):
        """k-NN results should be returned in descending similarity order."""
        from aragora.debate.similarity.ann import FAISSIndex

        # Create embeddings at various angles from x-axis
        emb = np.array(
            [
                [1.0, 0.0, 0.0],  # cos=1.0 (identical)
                [0.9, 0.1, 0.0],  # close
                [0.5, 0.5, 0.0],  # medium
                [0.0, 1.0, 0.0],  # orthogonal
            ]
        )
        # Normalize each
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        index = FAISSIndex(dimension=3)
        index.add(emb)
        query = np.array([[1.0, 0.0, 0.0]])
        indices, sims = index.search(query, k=4)
        # Similarities should be non-increasing
        assert sims[0, 0] >= sims[0, 1] >= sims[0, 2] >= sims[0, 3]

    def test_faiss_init_with_gpu_false_no_error(self):
        """use_gpu=False must succeed even with no GPU (faiss unavailable)."""
        from aragora.debate.similarity.ann import FAISSIndex

        index = FAISSIndex(dimension=6, use_gpu=False)
        assert index.is_available is False


# ===========================================================================
# compute_argument_diversity_optimized
# ===========================================================================


class TestComputeArgumentDiversityOptimized:
    def test_delegates_to_count_unique_fast(self):
        """compute_argument_diversity_optimized must return the same result
        as count_unique_fast with the same arguments."""
        from aragora.debate.similarity.ann import (
            compute_argument_diversity_optimized,
            count_unique_fast,
        )

        emb = ORTHOGONAL_3D.copy()
        expected = count_unique_fast(emb, threshold=0.7)
        result = compute_argument_diversity_optimized(emb, threshold=0.7)
        assert result == expected

    def test_returns_tuple_of_three(self):
        from aragora.debate.similarity.ann import compute_argument_diversity_optimized

        result = compute_argument_diversity_optimized(IDENTICAL_3D)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_empty_input(self):
        from aragora.debate.similarity.ann import compute_argument_diversity_optimized

        assert compute_argument_diversity_optimized(np.array([])) == (0, 0, 0.0)

    def test_diverse_inputs_high_diversity_score(self):
        from aragora.debate.similarity.ann import compute_argument_diversity_optimized

        _, _, diversity = compute_argument_diversity_optimized(ORTHOGONAL_3D, threshold=0.5)
        assert diversity == pytest.approx(1.0, abs=1e-6)

    def test_identical_inputs_low_diversity_score(self):
        from aragora.debate.similarity.ann import compute_argument_diversity_optimized

        _, _, diversity = compute_argument_diversity_optimized(IDENTICAL_3D, threshold=0.5)
        assert diversity == pytest.approx(0.0, abs=1e-6)

"""Tests for ANN (Approximate Nearest Neighbor) similarity optimizations."""

import numpy as np
import pytest
from aragora.debate.similarity.ann import (
    compute_pairwise_matrix,
    compute_batch_similarity_fast,
    compute_min_similarity,
    find_convergence_threshold,
    cluster_by_similarity,
    FAISSIndex,
)


class TestComputePairwiseMatrix:
    """Tests for compute_pairwise_matrix function."""

    def test_empty_array(self):
        """Test with empty input."""
        result = compute_pairwise_matrix(np.array([]))
        assert len(result) == 0

    def test_single_embedding(self):
        """Test with single embedding."""
        emb = np.array([[1.0, 0.0, 0.0]])
        matrix = compute_pairwise_matrix(emb)
        assert matrix.shape == (1, 1)
        assert abs(matrix[0, 0] - 1.0) < 0.001  # Self-similarity = 1

    def test_diagonal_is_one(self):
        """Test that diagonal values are 1 (self-similarity)."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 10)
        matrix = compute_pairwise_matrix(embeddings)

        for i in range(5):
            assert abs(matrix[i, i] - 1.0) < 0.001

    def test_symmetric_matrix(self):
        """Test that matrix is symmetric."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 10)
        matrix = compute_pairwise_matrix(embeddings)

        assert np.allclose(matrix, matrix.T)

    def test_identical_embeddings(self):
        """Test with identical embeddings."""
        emb = np.array([[1.0, 2.0, 3.0]] * 3)
        matrix = compute_pairwise_matrix(emb)

        # All pairs should have similarity ~1
        assert np.all(matrix > 0.99)

    def test_orthogonal_embeddings(self):
        """Test with orthogonal embeddings."""
        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        matrix = compute_pairwise_matrix(emb)

        # Diagonal should be 1
        assert abs(matrix[0, 0] - 1.0) < 0.001
        # Off-diagonal should be 0
        assert abs(matrix[0, 1]) < 0.001
        assert abs(matrix[0, 2]) < 0.001


class TestComputeBatchSimilarityFast:
    """Tests for compute_batch_similarity_fast function."""

    def test_single_text(self):
        """Test with single embedding returns 1.0."""
        emb = np.array([[1.0, 0.0]])
        assert compute_batch_similarity_fast(emb) == 1.0

    def test_empty_array(self):
        """Test with empty array."""
        result = compute_batch_similarity_fast(np.array([]).reshape(0, 10))
        assert result == 1.0

    def test_identical_embeddings(self):
        """Test with identical embeddings."""
        emb = np.array([[1.0, 2.0, 3.0]] * 5)
        result = compute_batch_similarity_fast(emb)
        assert result > 0.99

    def test_random_embeddings(self):
        """Test with random embeddings."""
        np.random.seed(42)
        emb = np.random.randn(10, 50)
        result = compute_batch_similarity_fast(emb)

        # Random vectors should have near-zero average similarity
        assert -0.5 < result < 0.5


class TestComputeMinSimilarity:
    """Tests for compute_min_similarity function."""

    def test_single_embedding(self):
        """Test with single embedding."""
        emb = np.array([[1.0, 0.0]])
        assert compute_min_similarity(emb) == 1.0

    def test_identical_embeddings(self):
        """Test with identical embeddings."""
        emb = np.array([[1.0, 2.0, 3.0]] * 3)
        result = compute_min_similarity(emb)
        assert result > 0.99

    def test_orthogonal_embeddings(self):
        """Test with orthogonal embeddings."""
        emb = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        result = compute_min_similarity(emb)
        assert abs(result) < 0.001  # Orthogonal = 0 similarity


class TestFindConvergenceThreshold:
    """Tests for find_convergence_threshold function."""

    def test_single_embedding_converged(self):
        """Test with single embedding is always converged."""
        emb = np.array([[1.0, 0.0]])
        converged, min_sim = find_convergence_threshold(emb, threshold=0.9)
        assert converged is True
        assert min_sim == 1.0

    def test_identical_converged(self):
        """Test identical embeddings are converged at any threshold."""
        emb = np.array([[1.0, 2.0, 3.0]] * 5)
        converged, min_sim = find_convergence_threshold(emb, threshold=0.99)
        assert converged is True
        assert min_sim > 0.99

    def test_random_not_converged_high_threshold(self):
        """Test random embeddings not converged at high threshold."""
        np.random.seed(42)
        emb = np.random.randn(5, 50)
        converged, min_sim = find_convergence_threshold(emb, threshold=0.9)
        assert converged is False

    def test_low_threshold_more_likely_converged(self):
        """Test that lower threshold makes convergence more likely."""
        np.random.seed(42)
        emb = np.random.randn(3, 50)

        _, min_sim = find_convergence_threshold(emb, threshold=0.0)

        # Very low threshold should always converge (unless negative similarities)
        converged_low, _ = find_convergence_threshold(emb, threshold=-1.0)
        assert converged_low is True


class TestClusterBySimilarity:
    """Tests for cluster_by_similarity function."""

    def test_empty_array(self):
        """Test with empty array."""
        result = cluster_by_similarity(np.array([]).reshape(0, 10))
        assert result == []

    def test_single_embedding(self):
        """Test with single embedding."""
        emb = np.array([[1.0, 0.0]])
        result = cluster_by_similarity(emb, min_cluster_size=1)
        assert len(result) == 1

    def test_identical_embeddings(self):
        """Test identical embeddings form one cluster."""
        emb = np.array([[1.0, 2.0, 3.0]] * 5)
        result = cluster_by_similarity(emb, threshold=0.9, min_cluster_size=2)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_orthogonal_embeddings_no_cluster(self):
        """Test orthogonal embeddings don't cluster at high threshold."""
        emb = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = cluster_by_similarity(emb, threshold=0.9, min_cluster_size=2)
        # No clusters because orthogonal vectors have 0 similarity
        assert len(result) == 0


class TestFAISSIndex:
    """Tests for FAISSIndex class."""

    def test_initialization(self):
        """Test index can be initialized."""
        index = FAISSIndex(dimension=10)
        assert index is not None

    def test_add_and_search(self):
        """Test adding embeddings and searching."""
        index = FAISSIndex(dimension=10)

        # Add some embeddings
        np.random.seed(42)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        index.add(embeddings)

        # Search for nearest neighbors
        query = embeddings[0]  # Search for first embedding
        indices, sims = index.search(query, k=3)

        # First result should be the query itself (or very similar)
        assert len(indices) > 0
        assert 0 in indices[0]  # Query should match itself

    def test_search_empty_index(self):
        """Test searching empty index."""
        index = FAISSIndex(dimension=10)
        query = np.random.randn(10).astype(np.float32)
        indices, sims = index.search(query, k=3)

        # Should return empty arrays
        assert len(indices) == 0 or indices.size == 0

    def test_reset(self):
        """Test resetting the index."""
        index = FAISSIndex(dimension=10)

        # Add embeddings
        embeddings = np.random.randn(10, 10).astype(np.float32)
        index.add(embeddings)

        # Reset
        index.reset()

        # Search should return empty
        query = np.random.randn(10).astype(np.float32)
        indices, sims = index.search(query, k=3)
        assert len(indices) == 0 or indices.size == 0


class TestPerformanceComparison:
    """Tests comparing performance of vectorized vs loop-based approaches."""

    def test_vectorized_gives_same_result_as_loop(self):
        """Verify vectorized approach matches loop-based calculation."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 50)

        # Vectorized result
        vectorized_result = compute_batch_similarity_fast(embeddings)

        # Loop-based result
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        total = 0.0
        count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(normalized[i], normalized[j])
                total += sim
                count += 1
        loop_result = total / count if count > 0 else 0.0

        # Results should be very close
        assert abs(vectorized_result - loop_result) < 0.001

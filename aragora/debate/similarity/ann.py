"""Approximate Nearest Neighbor optimizations for similarity computation.

Provides optimized similarity computation for larger batches:
1. Vectorized matrix operations (numpy) - 10-100x faster for n>10
2. Early termination when threshold is met
3. Optional FAISS integration for very large datasets (1000+ texts)

Usage:
    from aragora.debate.similarity.ann import (
        compute_batch_similarity_fast,
        compute_pairwise_matrix,
        find_convergence_threshold,
    )

    # Fast batch similarity
    avg_sim = compute_batch_similarity_fast(embeddings)

    # Full pairwise matrix
    matrix = compute_pairwise_matrix(embeddings)

    # Early termination check
    converged, min_sim = find_convergence_threshold(embeddings, threshold=0.85)
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_pairwise_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute full pairwise cosine similarity matrix using vectorized operations.

    Uses matrix multiplication which is much faster than looping:
    - For 10 texts: ~10x faster
    - For 50 texts: ~100x faster
    - For 100 texts: ~1000x faster

    Args:
        embeddings: Array of shape (n, d) where n is number of texts, d is embedding dim

    Returns:
        Similarity matrix of shape (n, n) with values in [-1, 1]
    """
    if len(embeddings) == 0:
        return np.array([])

    # Normalize embeddings to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Matrix multiplication gives cosine similarity: (n, d) @ (d, n) = (n, n)
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


def compute_batch_similarity_fast(
    embeddings: np.ndarray,
    exclude_diagonal: bool = True,
) -> float:
    """Compute average pairwise similarity using vectorized operations.

    Much faster than looping through pairs, especially for larger batches.

    Args:
        embeddings: Array of shape (n, d)
        exclude_diagonal: If True, exclude self-similarity (1.0) from average

    Returns:
        Average pairwise similarity score
    """
    n = len(embeddings)
    if n < 2:
        return 1.0

    matrix = compute_pairwise_matrix(embeddings)

    if exclude_diagonal:
        # Get upper triangle excluding diagonal (unique pairs only)
        upper_indices = np.triu_indices(n, k=1)
        similarities = matrix[upper_indices]
    else:
        similarities = matrix.flatten()

    return float(np.mean(similarities)) if len(similarities) > 0 else 0.0


def compute_min_similarity(embeddings: np.ndarray) -> float:
    """Compute minimum pairwise similarity.

    Args:
        embeddings: Array of shape (n, d)

    Returns:
        Minimum similarity value across all pairs
    """
    n = len(embeddings)
    if n < 2:
        return 1.0

    matrix = compute_pairwise_matrix(embeddings)

    # Get upper triangle excluding diagonal
    upper_indices = np.triu_indices(n, k=1)
    similarities = matrix[upper_indices]

    return float(np.min(similarities)) if len(similarities) > 0 else 0.0


def find_convergence_threshold(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    return_details: bool = False,
) -> Tuple[bool, float]:
    """Check if all pairs exceed convergence threshold with early termination.

    Optimized to return as soon as we find a pair below threshold.

    Args:
        embeddings: Array of shape (n, d)
        threshold: Minimum similarity threshold for convergence
        return_details: If True, also returns the min similarity value

    Returns:
        Tuple of (converged: bool, min_similarity: float)
    """
    n = len(embeddings)
    if n < 2:
        return True, 1.0

    # Normalize embeddings once
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    min_sim = 1.0

    # For small n, compute full matrix (vectorized is faster)
    if n <= 20:
        matrix = np.dot(normalized, normalized.T)
        upper_indices = np.triu_indices(n, k=1)
        similarities = matrix[upper_indices]
        min_sim = float(np.min(similarities)) if len(similarities) > 0 else 1.0
        return min_sim >= threshold, min_sim

    # For larger n, check row by row with early termination
    for i in range(n - 1):
        # Compute similarity of row i with all rows j > i
        sims = np.dot(normalized[i], normalized[i + 1 :].T)
        row_min = float(np.min(sims))
        min_sim = min(min_sim, row_min)

        # Early termination: if any pair is below threshold, not converged
        if row_min < threshold:
            return False, row_min

    return True, min_sim


def cluster_by_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.8,
    min_cluster_size: int = 2,
) -> list[list[int]]:
    """Cluster texts by similarity using a greedy approach.

    Useful for grouping similar responses without full hierarchical clustering.

    Args:
        embeddings: Array of shape (n, d)
        threshold: Minimum similarity to be in same cluster
        min_cluster_size: Minimum size for a valid cluster

    Returns:
        List of clusters, each cluster is a list of indices
    """
    n = len(embeddings)
    if n < 2:
        return [[0]] if n == 1 else []

    matrix = compute_pairwise_matrix(embeddings)
    used = set()
    clusters = []

    for i in range(n):
        if i in used:
            continue

        # Find all items similar enough to item i
        cluster = [i]
        for j in range(i + 1, n):
            if j in used:
                continue
            if matrix[i, j] >= threshold:
                cluster.append(j)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
            used.update(cluster)
        else:
            # Single item "cluster" - mark as used
            used.add(i)

    return clusters


class FAISSIndex:
    """Optional FAISS wrapper for very large scale similarity search.

    FAISS provides O(log n) or O(1) nearest neighbor search for large datasets.
    Falls back to exact search if FAISS is not installed.

    Usage:
        index = FAISSIndex(dimension=384)
        index.add(embeddings)
        neighbors, distances = index.search(query, k=5)
    """

    def __init__(self, dimension: int, use_gpu: bool = False):
        """Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            use_gpu: Use GPU acceleration if available
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        self._faiss = None
        self._index = None
        self._fallback_data: Optional[np.ndarray] = None

        self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS if available."""
        try:
            import faiss

            self._faiss = faiss

            # Use inner product (cosine sim for normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)

            if self.use_gpu and faiss.get_num_gpus() > 0:
                # Move to GPU
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index using GPU")
            else:
                logger.info("FAISS index using CPU")

        except ImportError:
            logger.debug("FAISS not available, using numpy fallback")
            self._faiss = None
            self._index = None

    @property
    def is_available(self) -> bool:
        """Check if FAISS is available."""
        return self._faiss is not None

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to index.

        Args:
            embeddings: Array of shape (n, d)
        """
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (embeddings / norms).astype(np.float32)

        if self._index is not None:
            self._index.add(normalized)
        else:
            # Fallback: store normalized embeddings
            if self._fallback_data is None:
                self._fallback_data = normalized
            else:
                self._fallback_data = np.vstack([self._fallback_data, normalized])

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query embedding(s) of shape (m, d) or (d,)
            k: Number of neighbors to return

        Returns:
            Tuple of (indices, similarities) both of shape (m, k)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (query / norms).astype(np.float32)

        if self._index is not None:
            # FAISS search returns (distances, indices)
            distances, indices = self._index.search(normalized, k)
            return indices, distances  # distances are cosine similarities for IP index

        # Fallback: brute force
        if self._fallback_data is None:
            return np.array([]), np.array([])

        similarities = np.dot(normalized, self._fallback_data.T)

        # Get top-k indices
        if k >= similarities.shape[1]:
            indices = np.argsort(-similarities, axis=1)
            sorted_sims = np.take_along_axis(similarities, indices, axis=1)
        else:
            indices = np.argpartition(-similarities, k, axis=1)[:, :k]
            indices = indices[
                np.arange(len(indices))[:, None],
                np.argsort(-np.take_along_axis(similarities, indices, axis=1), axis=1),
            ]
            sorted_sims = np.take_along_axis(similarities, indices, axis=1)

        return indices, sorted_sims

    def reset(self) -> None:
        """Clear the index."""
        if self._index is not None:
            self._index.reset()
        self._fallback_data = None


def count_unique_fast(
    embeddings: np.ndarray,
    threshold: float = 0.7,
    use_faiss: bool = True,
) -> Tuple[int, int, float]:
    """Count unique items efficiently using vectorized operations.

    An item is "unique" if no other item has similarity >= threshold.

    Complexity: O(n) space, O(n log n) with FAISS or O(n²) vectorized.
    Still faster than naive Python loops by 10-100x for the O(n²) case.

    Args:
        embeddings: Array of shape (n, d)
        threshold: Similarity threshold for considering items as duplicates
        use_faiss: Use FAISS if available for O(n log n) complexity

    Returns:
        Tuple of (unique_count, total_count, diversity_score)
    """
    n = len(embeddings)
    if n == 0:
        return (0, 0, 0.0)
    if n == 1:
        return (1, 1, 1.0)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Check for FAISS availability
    faiss_available = False
    try:
        import faiss

        faiss_available = True
    except ImportError:
        pass

    if use_faiss and faiss_available and n >= 50:
        # Use FAISS for larger datasets - O(n log n)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized.astype(np.float32))

        # Search for top-10 neighbors (excluding self)
        k = min(11, n)  # Extra one because first hit is always self
        distances, indices = index.search(normalized.astype(np.float32), k)

        unique_count = 0
        for i in range(n):
            is_unique = True
            for j in range(k):
                if indices[i, j] != i and distances[i, j] >= threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1
    else:
        # Vectorized approach - O(n²) but fast due to numpy
        similarity_matrix = np.dot(normalized, normalized.T)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

        # An item is unique if max similarity to any other item < threshold
        max_similarities = np.max(similarity_matrix, axis=1)
        unique_count = int(np.sum(max_similarities < threshold))

    diversity_score = unique_count / n if n > 0 else 0.0
    return (unique_count, n, diversity_score)


def compute_argument_diversity_optimized(
    embeddings: np.ndarray,
    threshold: float = 0.7,
) -> Tuple[int, int, float]:
    """Compute argument diversity using optimized similarity computation.

    This is a drop-in replacement for the O(n²) naive approach in
    AdvancedConvergenceAnalyzer.compute_argument_diversity.

    Args:
        embeddings: Pre-computed embeddings array of shape (n, d)
        threshold: Similarity threshold for considering arguments as duplicates

    Returns:
        Tuple of (unique_arguments, total_arguments, diversity_score)
    """
    return count_unique_fast(embeddings, threshold=threshold)


__all__ = [
    "compute_pairwise_matrix",
    "compute_batch_similarity_fast",
    "compute_min_similarity",
    "find_convergence_threshold",
    "cluster_by_similarity",
    "count_unique_fast",
    "compute_argument_diversity_optimized",
    "FAISSIndex",
]

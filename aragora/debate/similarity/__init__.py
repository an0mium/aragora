"""Similarity computation backends for debate convergence detection.

Provides a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)

Also provides ANN optimizations for large-scale similarity:
- Vectorized matrix operations (10-100x faster for n>10)
- Early termination when thresholds are met
- Optional FAISS integration for 1000+ texts
"""

from aragora.debate.similarity.backends import (
    JaccardBackend,
    SentenceTransformerBackend,
    SimilarityBackend,
    TFIDFBackend,
    get_similarity_backend,
)
from aragora.debate.similarity.ann import (
    compute_pairwise_matrix,
    compute_batch_similarity_fast,
    compute_min_similarity,
    find_convergence_threshold,
    cluster_by_similarity,
    FAISSIndex,
)

__all__ = [
    # Backends
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
    # ANN optimizations
    "compute_pairwise_matrix",
    "compute_batch_similarity_fast",
    "compute_min_similarity",
    "find_convergence_threshold",
    "cluster_by_similarity",
    "FAISSIndex",
]

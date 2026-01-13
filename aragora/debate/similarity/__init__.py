"""Similarity computation backends for debate convergence detection."""

from aragora.debate.similarity.backends import (
    JaccardBackend,
    SentenceTransformerBackend,
    SimilarityBackend,
    TFIDFBackend,
    get_similarity_backend,
)

__all__ = [
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
]

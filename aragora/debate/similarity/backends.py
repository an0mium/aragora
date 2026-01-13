"""Similarity computation backends for semantic comparison.

Provides a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from aragora.debate.cache.embeddings_lru import EmbeddingCache, get_embedding_cache

logger = logging.getLogger(__name__)

# Environment variables for backend selection
_ENV_SIMILARITY_BACKEND = "ARAGORA_SIMILARITY_BACKEND"
_ENV_CONVERGENCE_BACKEND = "ARAGORA_CONVERGENCE_BACKEND"

_BACKEND_ALIASES = {
    "sentence-transformers": "sentence-transformer",
    "sentence_transformers": "sentence-transformer",
    "sentence": "sentence-transformer",
    "tf-idf": "tfidf",
    "tf_idf": "tfidf",
}
_VALID_BACKENDS = {"auto", "sentence-transformer", "tfidf", "jaccard"}


def _normalize_backend_name(value: str) -> str | None:
    """Normalize backend name from environment variable."""
    if not value:
        return None
    key = value.strip().lower()
    key = key.replace("_", "-")
    key = _BACKEND_ALIASES.get(key, key)
    return key if key in _VALID_BACKENDS else None


class SimilarityBackend(ABC):
    """Abstract base class for similarity computation backends."""

    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        raise NotImplementedError("Subclasses must implement compute_similarity")

    def compute_batch_similarity(self, texts: list[str]) -> float:
        """
        Compute average pairwise similarity across multiple texts.

        Args:
            texts: List of texts to compare

        Returns:
            Average similarity score
        """
        if len(texts) < 2:
            return 1.0

        total = 0.0
        count = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                total += self.compute_similarity(texts[i], texts[j])
                count += 1

        return total / count if count > 0 else 0.0


class JaccardBackend(SimilarityBackend):
    """
    Jaccard similarity using word overlap.

    Formula: |A ∩ B| / |A ∪ B|

    Pros:
        - Zero dependencies
        - Fast computation
        - Easy to understand

    Cons:
        - Doesn't understand semantics
        - Order-independent

    Performance optimization:
        - Individual similarity computations are cached (256 pairs)
    """

    _similarity_cache: dict[tuple[str, str], float] = {}
    _cache_max_size = 256
    _cache_lock = threading.RLock()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts (thread-safe)."""
        if not text1 or not text2:
            return 0.0

        # Normalize key order for symmetric cache hits
        cache_key = (text1, text2) if text1 <= text2 else (text2, text1)

        # Check cache first (with lock)
        with JaccardBackend._cache_lock:
            if cache_key in JaccardBackend._similarity_cache:
                return JaccardBackend._similarity_cache[cache_key]

        # Normalize: lowercase and split into words (outside lock)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        result = len(intersection) / len(union)

        # Cache result (with lock and simple size limit)
        with JaccardBackend._cache_lock:
            if len(JaccardBackend._similarity_cache) >= JaccardBackend._cache_max_size:
                keys = list(JaccardBackend._similarity_cache.keys())
                for k in keys[: len(keys) // 2]:
                    del JaccardBackend._similarity_cache[k]
            JaccardBackend._similarity_cache[cache_key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the similarity cache (thread-safe)."""
        with cls._cache_lock:
            cls._similarity_cache.clear()


class TFIDFBackend(SimilarityBackend):
    """
    TF-IDF similarity backend.

    Requires: scikit-learn

    Better than Jaccard because:
        - Weighs rare words higher (more discriminative)
        - Reduces impact of common words

    Performance optimization:
        - Individual similarity computations are cached (256 pairs)
    """

    _similarity_cache: dict[tuple[str, str], float] = {}
    _cache_max_size = 256
    _cache_lock = threading.RLock()

    def __init__(self):
        """Initialize TF-IDF backend."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self.vectorizer = TfidfVectorizer()
            self.cosine_similarity = cosine_similarity
        except ImportError as e:
            raise ImportError(
                "TFIDFBackend requires scikit-learn. " "Install with: pip install scikit-learn"
            ) from e

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF cosine similarity between two texts (thread-safe)."""
        if not text1 or not text2:
            return 0.0

        # Normalize key order for symmetric cache hits
        cache_key = (text1, text2) if text1 <= text2 else (text2, text1)

        # Check cache first (with lock)
        with TFIDFBackend._cache_lock:
            if cache_key in TFIDFBackend._similarity_cache:
                return TFIDFBackend._similarity_cache[cache_key]

        # Compute outside lock
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = self.cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        result = float(similarity)

        # Cache result (with lock and simple size limit)
        with TFIDFBackend._cache_lock:
            if len(TFIDFBackend._similarity_cache) >= TFIDFBackend._cache_max_size:
                keys = list(TFIDFBackend._similarity_cache.keys())
                for k in keys[: len(keys) // 2]:
                    del TFIDFBackend._similarity_cache[k]
            TFIDFBackend._similarity_cache[cache_key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the similarity cache (thread-safe)."""
        with cls._cache_lock:
            cls._similarity_cache.clear()


class SentenceTransformerBackend(SimilarityBackend):
    """
    Sentence transformer backend using neural embeddings.

    Requires: sentence-transformers (~500MB model download)

    Best accuracy because:
        - Understands semantics and context
        - Captures paraphrasing and synonyms

    Performance optimization:
        - Model is cached at class level (avoids reloading)
        - Embeddings are cached at text level (EmbeddingCache)
        - Similarity results are cached at pair level (256 pairs)
        - Expected speedup: 10-100x for repeated text comparisons
    """

    _model_cache: Optional[Any] = None
    _model_name_cache: Optional[str] = None
    _similarity_cache: dict[tuple[str, str], float] = {}
    _cache_max_size: int = 256
    _cache_lock = threading.RLock()

    model: Any
    cosine_similarity: Any
    embedding_cache: Optional[EmbeddingCache]

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_embedding_cache: bool = True,
        persist_embeddings: bool = False,
    ):
        """
        Initialize sentence transformer backend.

        Args:
            model_name: Sentence transformer model name
            use_embedding_cache: Enable embedding-level caching (default True)
            persist_embeddings: Persist embeddings to database (default False)
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            if (
                SentenceTransformerBackend._model_cache is not None
                and SentenceTransformerBackend._model_name_cache == model_name
            ):
                logger.debug(f"Reusing cached sentence transformer: {model_name}")
                self.model = SentenceTransformerBackend._model_cache
            else:
                logger.info(f"Loading sentence transformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                SentenceTransformerBackend._model_cache = self.model
                SentenceTransformerBackend._model_name_cache = model_name

            self.cosine_similarity = cosine_similarity

            # Initialize embedding cache
            if use_embedding_cache:
                self.embedding_cache = get_embedding_cache(
                    max_size=1024,
                    persist=persist_embeddings,
                )
            else:
                self.embedding_cache = None

        except (ImportError, RuntimeError) as e:
            # RuntimeError can occur from transformers/Keras compatibility issues
            raise ImportError(
                "SentenceTransformerBackend requires sentence-transformers. "
                f"Install with: pip install sentence-transformers. Error: {e}"
            ) from e

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        if self.embedding_cache:
            cached = self.embedding_cache.get(text)
            if cached is not None:
                return cached

        # Compute embedding
        embedding = self.model.encode([text])[0]

        # Cache it
        if self.embedding_cache:
            self.embedding_cache.put(text, embedding)

        return embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings (thread-safe).

        Uses multi-level caching:
        1. Similarity cache (pair level) - fastest
        2. Embedding cache (text level) - avoids re-encoding same text
        3. Model encode (slowest) - only for cache misses
        """
        if not text1 or not text2:
            return 0.0

        # Normalize key order for symmetric cache hits
        cache_key = (text1, text2) if text1 <= text2 else (text2, text1)

        # Check similarity cache first (with lock)
        with SentenceTransformerBackend._cache_lock:
            if cache_key in SentenceTransformerBackend._similarity_cache:
                return SentenceTransformerBackend._similarity_cache[cache_key]

        # Get embeddings (checks embedding cache internally)
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Compute cosine similarity
        similarity = self.cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        result = float(similarity)

        # Cache similarity result (with lock and simple size limit)
        with SentenceTransformerBackend._cache_lock:
            if (
                len(SentenceTransformerBackend._similarity_cache)
                >= SentenceTransformerBackend._cache_max_size
            ):
                # Clear oldest half when full
                keys = list(SentenceTransformerBackend._similarity_cache.keys())
                for k in keys[: len(keys) // 2]:
                    del SentenceTransformerBackend._similarity_cache[k]
            SentenceTransformerBackend._similarity_cache[cache_key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the similarity cache (thread-safe)."""
        with cls._cache_lock:
            cls._similarity_cache.clear()

    def compute_batch_similarity(self, texts: list[str]) -> float:
        """
        Optimized batch similarity using single encode call.

        Computes all embeddings at once, then calculates pairwise cosine similarities.
        Much faster than O(n²) individual encode calls.
        """
        if len(texts) < 2:
            return 1.0

        # Single batch encode (O(n) instead of O(n²) encode calls)
        embeddings = self.model.encode(texts)

        # Compute all pairwise cosine similarities efficiently
        total = 0.0
        count = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self.cosine_similarity(
                    embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1)
                )[0][0]
                total += float(sim)
                count += 1

        return total / count if count > 0 else 0.0

    def compute_pairwise_similarities(self, texts_a: list[str], texts_b: list[str]) -> list[float]:
        """
        Compute similarities for paired texts efficiently.

        Args:
            texts_a: First list of texts
            texts_b: Second list of texts (must be same length)

        Returns:
            List of similarities for each pair (a[i], b[i])
        """
        if not texts_a or not texts_b or len(texts_a) != len(texts_b):
            return []

        # Batch encode both lists in one call
        all_texts = texts_a + texts_b
        embeddings = self.model.encode(all_texts)

        # Split embeddings
        n = len(texts_a)
        emb_a = embeddings[:n]
        emb_b = embeddings[n:]

        # Compute pairwise similarities
        similarities = []
        for i in range(n):
            sim = self.cosine_similarity(emb_a[i].reshape(1, -1), emb_b[i].reshape(1, -1))[0][0]
            similarities.append(float(sim))

        return similarities


def get_similarity_backend(preferred: str = "auto") -> SimilarityBackend:
    """
    Get a similarity backend by name.

    Args:
        preferred: Backend preference: "auto", "sentence-transformer", "tfidf", "jaccard"

    Returns:
        Requested backend instance
    """
    if preferred == "auto":
        env_override = _normalize_backend_name(os.getenv(_ENV_SIMILARITY_BACKEND, ""))
        if env_override:
            preferred = env_override
        elif os.getenv(_ENV_SIMILARITY_BACKEND):
            logger.warning(f"{_ENV_SIMILARITY_BACKEND} value is invalid; using auto selection.")

    if preferred == "jaccard":
        return JaccardBackend()

    if preferred == "tfidf":
        return TFIDFBackend()

    if preferred == "sentence-transformer":
        return SentenceTransformerBackend()

    # Auto-select best available
    try:
        return SentenceTransformerBackend()
    except ImportError:
        logger.debug("sentence-transformers not available for auto-select")
    except RuntimeError as e:
        logger.debug(f"sentence-transformers failed: {e}")
    except OSError as e:
        logger.debug(f"sentence-transformers model error: {e}")

    try:
        return TFIDFBackend()
    except ImportError:
        logger.debug("scikit-learn not available for auto-select")

    return JaccardBackend()


__all__ = [
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
]

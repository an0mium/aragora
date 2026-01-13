"""
Semantic convergence detection for multi-agent debates.

Detects when agents' positions have converged, allowing early termination
of debates when further rounds would provide diminishing returns.

Uses a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)

Inspired by ai-counsel's convergence detection system.
"""

import hashlib
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Cache
# =============================================================================


class EmbeddingCache:
    """
    LRU cache for text embeddings with optional persistence.

    Caches embeddings at the text level, so the same text encoded in
    different pairs only requires one model.encode() call.

    Performance impact:
        - Without cache: O(n²) encode calls for n texts
        - With cache: O(n) encode calls (amortized)
        - Expected speedup: 10-100x for repeated text comparisons
    """

    def __init__(
        self,
        max_size: int = 1024,
        persist: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum entries in memory cache (default 1024)
            persist: Whether to persist to database (default False)
            db_path: Path to embeddings database (uses core.db if None)
        """
        self.max_size = max_size
        self.persist = persist
        self.db_path = db_path
        self._cache: dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._hash_text(text)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end of access order (LRU)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

        # Try persistent cache
        if self.persist:
            embedding = self._load_from_db(key)
            if embedding is not None:
                self._hits += 1
                with self._lock:
                    self._store_in_memory(key, embedding)
                return embedding

        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_text(text)

        with self._lock:
            self._store_in_memory(key, embedding)

        if self.persist:
            self._save_to_db(key, text[:1000], embedding)  # Truncate text

    def _store_in_memory(self, key: str, embedding: np.ndarray) -> None:
        """Store in memory cache with LRU eviction."""
        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = embedding
        if key not in self._access_order:
            self._access_order.append(key)

    def _load_from_db(self, key: str) -> Optional[np.ndarray]:
        """Load embedding from database."""
        if not self.db_path:
            return None

        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ?", (key,)
            )
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                return np.frombuffer(row[0], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Failed to load embedding from DB: {e}")

        return None

    def _save_to_db(self, key: str, text: str, embedding: np.ndarray) -> None:
        """Save embedding to database."""
        if not self.db_path:
            return

        try:
            import sqlite3
            import uuid

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, text_hash, text, embedding, provider, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    str(uuid.uuid4()),
                    key,
                    text,
                    embedding.astype(np.float32).tobytes(),
                    "sentence-transformer",
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to save embedding to DB: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0


# Global embedding cache instance
_embedding_cache: Optional[EmbeddingCache] = None
_embedding_cache_lock = threading.Lock()


def get_embedding_cache(
    max_size: int = 1024,
    persist: bool = False,
    db_path: Optional[str] = None,
) -> EmbeddingCache:
    """
    Get or create global embedding cache.

    Args:
        max_size: Maximum cache entries
        persist: Enable database persistence
        db_path: Database path for persistence

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache

    with _embedding_cache_lock:
        if _embedding_cache is None:
            # Default to core.db for persistence
            if persist and db_path is None:
                from aragora.persistence.db_config import (
                    get_db_path_str,
                    DatabaseType,
                )

                db_path = get_db_path_str(DatabaseType.EMBEDDINGS)

            _embedding_cache = EmbeddingCache(
                max_size=max_size,
                persist=persist,
                db_path=db_path,
            )

        return _embedding_cache

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
    if not value:
        return None
    key = value.strip().lower()
    key = key.replace("_", "-")
    key = _BACKEND_ALIASES.get(key, key)
    return key if key in _VALID_BACKENDS else None


# =============================================================================
# Similarity Backend Interface
# =============================================================================


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


# =============================================================================
# Jaccard Backend (Zero Dependencies)
# =============================================================================


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
                for k in keys[:len(keys) // 2]:
                    del JaccardBackend._similarity_cache[k]
            JaccardBackend._similarity_cache[cache_key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the similarity cache (thread-safe)."""
        with cls._cache_lock:
            cls._similarity_cache.clear()


# =============================================================================
# TF-IDF Backend (Requires scikit-learn)
# =============================================================================


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
                "TFIDFBackend requires scikit-learn. "
                "Install with: pip install scikit-learn"
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
                for k in keys[:len(keys) // 2]:
                    del TFIDFBackend._similarity_cache[k]
            TFIDFBackend._similarity_cache[cache_key] = result

        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the similarity cache (thread-safe)."""
        with cls._cache_lock:
            cls._similarity_cache.clear()


# =============================================================================
# Sentence Transformer Backend (Requires sentence-transformers)
# =============================================================================


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
        similarity = self.cosine_similarity(
            emb1.reshape(1, -1), emb2.reshape(1, -1)
        )[0][0]
        result = float(similarity)

        # Cache similarity result (with lock and simple size limit)
        with SentenceTransformerBackend._cache_lock:
            if len(SentenceTransformerBackend._similarity_cache) >= SentenceTransformerBackend._cache_max_size:
                # Clear oldest half when full
                keys = list(SentenceTransformerBackend._similarity_cache.keys())
                for k in keys[:len(keys) // 2]:
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

    def compute_pairwise_similarities(
        self, texts_a: list[str], texts_b: list[str]
    ) -> list[float]:
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
            sim = self.cosine_similarity(
                emb_a[i].reshape(1, -1), emb_b[i].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))

        return similarities


# =============================================================================
# Convergence Result
# =============================================================================


@dataclass
class ConvergenceResult:
    """Result of convergence detection check."""

    converged: bool
    status: str  # "converged", "diverging", "refining"
    min_similarity: float
    avg_similarity: float
    per_agent_similarity: dict[str, float] = field(default_factory=dict)
    consecutive_stable_rounds: int = 0


# =============================================================================
# Advanced Convergence Metrics (G3)
# =============================================================================


@dataclass
class ArgumentDiversityMetric:
    """
    Measures diversity of arguments across agents.

    High diversity = agents covering different points (good for exploration)
    Low diversity = agents focusing on same points (may indicate convergence)
    """
    unique_arguments: int
    total_arguments: int
    diversity_score: float  # 0-1, higher = more diverse

    @property
    def is_converging(self) -> bool:
        """Arguments becoming less diverse suggests convergence."""
        return self.diversity_score < 0.3


@dataclass
class EvidenceConvergenceMetric:
    """
    Measures overlap in cited evidence/sources.

    High overlap = agents citing same sources (strong agreement)
    Low overlap = agents using different evidence (disagreement or complementary)
    """
    shared_citations: int
    total_citations: int
    overlap_score: float  # 0-1, higher = more overlap

    @property
    def is_converging(self) -> bool:
        """High citation overlap suggests convergence."""
        return self.overlap_score > 0.6


@dataclass
class StanceVolatilityMetric:
    """
    Measures how often agents change their positions.

    High volatility = agents frequently changing stances (unstable)
    Low volatility = agents maintaining consistent positions (stable)
    """
    stance_changes: int
    total_responses: int
    volatility_score: float  # 0-1, higher = more volatile

    @property
    def is_stable(self) -> bool:
        """Low volatility indicates stable positions."""
        return self.volatility_score < 0.2


@dataclass
class AdvancedConvergenceMetrics:
    """
    Comprehensive convergence metrics for debate analysis.

    Combines multiple signals to provide a nuanced view of
    debate convergence beyond simple text similarity.
    """
    # Core similarity (from ConvergenceDetector)
    semantic_similarity: float

    # Advanced metrics
    argument_diversity: Optional[ArgumentDiversityMetric] = None
    evidence_convergence: Optional[EvidenceConvergenceMetric] = None
    stance_volatility: Optional[StanceVolatilityMetric] = None

    # Aggregate score
    overall_convergence: float = 0.0  # 0-1, higher = more converged

    # Domain context
    domain: str = "general"

    def compute_overall_score(self) -> float:
        """Compute weighted overall convergence score."""
        weights = {
            "semantic": 0.4,
            "diversity": 0.2,
            "evidence": 0.2,
            "stability": 0.2,
        }

        score = self.semantic_similarity * weights["semantic"]

        if self.argument_diversity:
            # Lower diversity = higher convergence
            score += (1 - self.argument_diversity.diversity_score) * weights["diversity"]

        if self.evidence_convergence:
            score += self.evidence_convergence.overlap_score * weights["evidence"]

        if self.stance_volatility:
            # Lower volatility = higher convergence
            score += (1 - self.stance_volatility.volatility_score) * weights["stability"]

        self.overall_convergence = min(1.0, max(0.0, score))
        return self.overall_convergence

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = {
            "semantic_similarity": self.semantic_similarity,
            "overall_convergence": self.overall_convergence,
            "domain": self.domain,
        }

        if self.argument_diversity:
            result["argument_diversity"] = {
                "unique_arguments": self.argument_diversity.unique_arguments,
                "total_arguments": self.argument_diversity.total_arguments,
                "diversity_score": self.argument_diversity.diversity_score,
            }

        if self.evidence_convergence:
            result["evidence_convergence"] = {
                "shared_citations": self.evidence_convergence.shared_citations,
                "total_citations": self.evidence_convergence.total_citations,
                "overlap_score": self.evidence_convergence.overlap_score,
            }

        if self.stance_volatility:
            result["stance_volatility"] = {
                "stance_changes": self.stance_volatility.stance_changes,
                "total_responses": self.stance_volatility.total_responses,
                "volatility_score": self.stance_volatility.volatility_score,
            }

        return result


class AdvancedConvergenceAnalyzer:
    """
    Analyzes debate convergence using multiple metrics.

    Provides a more nuanced view than simple text similarity by
    considering argument diversity, evidence overlap, and stance stability.
    """

    def __init__(self, similarity_backend: Optional[SimilarityBackend] = None):
        """
        Initialize analyzer.

        Args:
            similarity_backend: Backend for text similarity (auto-selects if None)
        """
        if similarity_backend is None:
            # Use best available backend
            try:
                self.backend: SimilarityBackend = SentenceTransformerBackend()
            except ImportError:
                try:
                    self.backend = TFIDFBackend()
                except ImportError:
                    self.backend = JaccardBackend()
        else:
            self.backend = similarity_backend

    def extract_arguments(self, text: str) -> list[str]:
        """
        Extract distinct arguments/claims from text.

        Simple heuristic: split by sentence and filter.
        """
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Filter to substantive sentences (> 5 words)
        arguments = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) > 5:
                arguments.append(s)

        return arguments

    def extract_citations(self, text: str) -> set[str]:
        """
        Extract citations/sources from text.

        Looks for URLs, academic-style citations, and quoted sources.
        """
        import re

        citations = set()

        # URLs
        urls = re.findall(r'https?://[^\s<>"]+', text)
        citations.update(urls)

        # Academic citations like (Author, 2024) or [1]
        academic = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text)
        citations.update(academic)

        # Numbered citations [1], [2], etc.
        numbered = re.findall(r'\[\d+\]', text)
        citations.update(numbered)

        # Quoted sources "According to X"
        quoted = re.findall(r'(?:according to|per|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text, re.I)
        citations.update(quoted)

        return citations

    def detect_stance(self, text: str) -> str:
        """
        Detect the stance/position in text.

        Returns: "support", "oppose", "neutral", or "mixed"
        """
        import re

        text_lower = text.lower()

        # Strong support indicators
        support_patterns = r'\b(agree|support|favor|endorse|recommend|should|must|definitely|certainly)\b'
        support_count = len(re.findall(support_patterns, text_lower))

        # Strong oppose indicators
        oppose_patterns = r'\b(disagree|oppose|against|reject|shouldn\'t|must not|definitely not|certainly not)\b'
        oppose_count = len(re.findall(oppose_patterns, text_lower))

        # Neutral indicators
        neutral_patterns = r'\b(depends|unclear|both|however|on the other hand|alternatively)\b'
        neutral_count = len(re.findall(neutral_patterns, text_lower))

        # Determine stance
        if support_count > oppose_count and support_count > neutral_count:
            return "support"
        elif oppose_count > support_count and oppose_count > neutral_count:
            return "oppose"
        elif neutral_count > 0 and support_count > 0 and oppose_count > 0:
            return "mixed"
        else:
            return "neutral"

    def compute_argument_diversity(
        self,
        agent_responses: dict[str, str],
    ) -> ArgumentDiversityMetric:
        """
        Compute argument diversity across agents.

        High diversity = agents making different points.
        """
        all_arguments: list[str] = []
        for text in agent_responses.values():
            all_arguments.extend(self.extract_arguments(text))

        if not all_arguments:
            return ArgumentDiversityMetric(
                unique_arguments=0,
                total_arguments=0,
                diversity_score=0.0,
            )

        # Compute pairwise similarity to find unique arguments
        # Arguments with < 0.7 similarity to all others are "unique"
        unique_count = 0
        for i, arg in enumerate(all_arguments):
            is_unique = True
            for j, other in enumerate(all_arguments):
                if i != j:
                    sim = self.backend.compute_similarity(arg, other)
                    if sim > 0.7:
                        is_unique = False
                        break
            if is_unique:
                unique_count += 1

        diversity_score = unique_count / len(all_arguments) if all_arguments else 0.0

        return ArgumentDiversityMetric(
            unique_arguments=unique_count,
            total_arguments=len(all_arguments),
            diversity_score=diversity_score,
        )

    def compute_evidence_convergence(
        self,
        agent_responses: dict[str, str],
    ) -> EvidenceConvergenceMetric:
        """
        Compute evidence/citation overlap across agents.

        High overlap = agents citing same sources.
        """
        all_citations: list[set[str]] = []
        for text in agent_responses.values():
            all_citations.append(self.extract_citations(text))

        # Flatten for total count
        all_unique = set().union(*all_citations) if all_citations else set()
        total = len(all_unique)

        if total == 0 or len(all_citations) < 2:
            return EvidenceConvergenceMetric(
                shared_citations=0,
                total_citations=0,
                overlap_score=0.0,
            )

        # Find citations shared by at least 2 agents
        shared = set()
        for citation in all_unique:
            count = sum(1 for agent_cites in all_citations if citation in agent_cites)
            if count >= 2:
                shared.add(citation)

        overlap_score = len(shared) / total if total > 0 else 0.0

        return EvidenceConvergenceMetric(
            shared_citations=len(shared),
            total_citations=total,
            overlap_score=overlap_score,
        )

    def compute_stance_volatility(
        self,
        response_history: list[dict[str, str]],
    ) -> StanceVolatilityMetric:
        """
        Compute stance volatility across rounds.

        Args:
            response_history: List of {agent: response} dicts per round

        Returns:
            StanceVolatilityMetric
        """
        if len(response_history) < 2:
            return StanceVolatilityMetric(
                stance_changes=0,
                total_responses=0,
                volatility_score=0.0,
            )

        # Track stance per agent per round
        agent_stances: dict[str, list[str]] = {}
        for round_responses in response_history:
            for agent, text in round_responses.items():
                if agent not in agent_stances:
                    agent_stances[agent] = []
                agent_stances[agent].append(self.detect_stance(text))

        # Count stance changes
        total_changes = 0
        total_responses = 0
        for agent, stances in agent_stances.items():
            total_responses += len(stances)
            for i in range(1, len(stances)):
                if stances[i] != stances[i-1]:
                    total_changes += 1

        volatility_score = total_changes / max(1, total_responses - len(agent_stances))

        return StanceVolatilityMetric(
            stance_changes=total_changes,
            total_responses=total_responses,
            volatility_score=min(1.0, volatility_score),
        )

    def analyze(
        self,
        current_responses: dict[str, str],
        previous_responses: Optional[dict[str, str]] = None,
        response_history: Optional[list[dict[str, str]]] = None,
        domain: str = "general",
    ) -> AdvancedConvergenceMetrics:
        """
        Perform comprehensive convergence analysis.

        Args:
            current_responses: {agent: response} for current round
            previous_responses: {agent: response} for previous round (optional)
            response_history: Full history of responses (optional)
            domain: Debate domain for context

        Returns:
            AdvancedConvergenceMetrics with all computed metrics
        """
        # Compute semantic similarity
        if previous_responses:
            common_agents = set(current_responses.keys()) & set(previous_responses.keys())
            if common_agents:
                similarities = []
                for agent in common_agents:
                    sim = self.backend.compute_similarity(
                        current_responses[agent],
                        previous_responses[agent],
                    )
                    similarities.append(sim)
                semantic_sim = sum(similarities) / len(similarities)
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 0.0

        # Compute advanced metrics
        arg_diversity = self.compute_argument_diversity(current_responses)
        evidence_conv = self.compute_evidence_convergence(current_responses)

        stance_vol = None
        if response_history:
            stance_vol = self.compute_stance_volatility(response_history)

        # Build result
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=semantic_sim,
            argument_diversity=arg_diversity,
            evidence_convergence=evidence_conv,
            stance_volatility=stance_vol,
            domain=domain,
        )

        # Compute overall score
        metrics.compute_overall_score()

        return metrics


# =============================================================================
# Convergence Detector
# =============================================================================


class ConvergenceDetector:
    """
    Detects when debate has converged semantically.

    Uses semantic similarity between consecutive rounds to determine
    if agents have reached consensus or are still refining positions.

    Thresholds:
        - converged: ≥85% similarity (agents agree)
        - refining: 40-85% similarity (still improving)
        - diverging: <40% similarity (positions splitting)
    """

    def __init__(
        self,
        convergence_threshold: float = 0.85,
        divergence_threshold: float = 0.40,
        min_rounds_before_check: int = 1,
        consecutive_rounds_needed: int = 1,
    ):
        """
        Initialize convergence detector.

        Args:
            convergence_threshold: Similarity threshold for convergence (default 0.85)
            divergence_threshold: Below this is diverging (default 0.40)
            min_rounds_before_check: Minimum rounds before checking (default 1)
            consecutive_rounds_needed: Stable rounds needed for convergence (default 1)
        """
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold
        self.min_rounds_before_check = min_rounds_before_check
        self.consecutive_rounds_needed = consecutive_rounds_needed
        self.consecutive_stable_count = 0
        self.backend = self._select_backend()

        logger.info(
            f"ConvergenceDetector initialized with {self.backend.__class__.__name__}"
        )

    def _select_backend(self) -> SimilarityBackend:
        """
        Select best available similarity backend.

        Tries: SentenceTransformer -> TF-IDF -> Jaccard
        """
        env_override = _normalize_backend_name(
            os.getenv(_ENV_CONVERGENCE_BACKEND, "")
        )
        if env_override:
            try:
                backend = get_similarity_backend(env_override)
                logger.info(f"Using {env_override} backend via {_ENV_CONVERGENCE_BACKEND}")
                return backend
            except Exception as e:
                logger.warning(
                    f"{_ENV_CONVERGENCE_BACKEND}={env_override} failed: {e}. Falling back to auto."
                )

        # Try sentence transformers (best)
        try:
            backend: SimilarityBackend = SentenceTransformerBackend()
            logger.info("Using SentenceTransformerBackend (best accuracy)")
            return backend
        except ImportError as e:
            logger.debug(f"sentence-transformers not available: {e}")
        except (RuntimeError, AttributeError) as e:
            # RuntimeError/AttributeError: transformers/scipy/numpy compatibility issues
            logger.debug(f"sentence-transformers failed to initialize: {e}")
        except OSError as e:
            # OSError can occur when model files are corrupted or missing
            logger.debug(f"sentence-transformers model error: {e}")

        # Try TF-IDF (good)
        try:
            backend = TFIDFBackend()
            logger.info("Using TFIDFBackend (good accuracy)")
            return backend
        except (ImportError, AttributeError, RuntimeError) as e:
            # AttributeError/RuntimeError: scipy/numpy version mismatch
            logger.debug(f"scikit-learn/scipy not available: {e}")

        # Fallback to Jaccard (always available)
        logger.info("Using JaccardBackend (fallback)")
        return JaccardBackend()

    def check_convergence(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> Optional[ConvergenceResult]:
        """
        Check if debate has converged.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number (1-indexed)

        Returns:
            ConvergenceResult or None if too early to check
        """
        # Don't check before minimum rounds
        if round_number <= self.min_rounds_before_check:
            return None

        # Match agents between rounds
        common_agents = set(current_responses.keys()) & set(previous_responses.keys())
        if not common_agents:
            logger.warning("No matching agents between rounds")
            return None

        # Compute similarity for each agent
        per_agent = {}
        agent_list = list(common_agents)

        # Use batch method if available (SentenceTransformerBackend)
        if hasattr(self.backend, 'compute_pairwise_similarities'):
            texts_current = [current_responses[a] for a in agent_list]
            texts_previous = [previous_responses[a] for a in agent_list]
            similarities = self.backend.compute_pairwise_similarities(
                texts_current, texts_previous
            )
            per_agent = dict(zip(agent_list, similarities))
        else:
            # Fallback to individual comparisons
            for agent in agent_list:
                similarity = self.backend.compute_similarity(
                    current_responses[agent], previous_responses[agent]
                )
                per_agent[agent] = similarity

        # Compute aggregate metrics
        similarities = list(per_agent.values())
        min_similarity = min(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Determine status
        if min_similarity >= self.convergence_threshold:
            self.consecutive_stable_count += 1
            if self.consecutive_stable_count >= self.consecutive_rounds_needed:
                status = "converged"
                converged = True
            else:
                status = "refining"
                converged = False
        elif min_similarity < self.divergence_threshold:
            status = "diverging"
            converged = False
            self.consecutive_stable_count = 0
        else:
            status = "refining"
            converged = False
            self.consecutive_stable_count = 0

        return ConvergenceResult(
            converged=converged,
            status=status,
            min_similarity=min_similarity,
            avg_similarity=avg_similarity,
            per_agent_similarity=per_agent,
            consecutive_stable_rounds=self.consecutive_stable_count,
        )

    def reset(self) -> None:
        """Reset the consecutive stable count."""
        self.consecutive_stable_count = 0


def get_similarity_backend(preferred: str = "auto") -> SimilarityBackend:
    """
    Get a similarity backend by name.

    Args:
        preferred: Backend preference: "auto", "sentence-transformer", "tfidf", "jaccard"

    Returns:
        Requested backend instance
    """
    if preferred == "auto":
        env_override = _normalize_backend_name(
            os.getenv(_ENV_SIMILARITY_BACKEND, "")
        )
        if env_override:
            preferred = env_override
        elif os.getenv(_ENV_SIMILARITY_BACKEND):
            logger.warning(
                f"{_ENV_SIMILARITY_BACKEND} value is invalid; using auto selection."
            )

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

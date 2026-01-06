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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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
    """

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize: lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)


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
    """

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
        """Compute TF-IDF cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = self.cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

        return float(similarity)


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
    """

    _model_cache = None
    _model_name_cache = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize sentence transformer backend."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            if (
                SentenceTransformerBackend._model_cache is not None
                and SentenceTransformerBackend._model_name_cache == model_name
            ):
                logger.info(f"Reusing cached sentence transformer: {model_name}")
                self.model = SentenceTransformerBackend._model_cache
            else:
                logger.info(f"Loading sentence transformer: {model_name}")
                self.model = SentenceTransformer(model_name)
                SentenceTransformerBackend._model_cache = self.model
                SentenceTransformerBackend._model_name_cache = model_name

            self.cosine_similarity = cosine_similarity

        except (ImportError, RuntimeError) as e:
            # RuntimeError can occur from transformers/Keras compatibility issues
            raise ImportError(
                "SentenceTransformerBackend requires sentence-transformers. "
                f"Install with: pip install sentence-transformers. Error: {e}"
            ) from e

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if not text1 or not text2:
            return 0.0

        embeddings = self.model.encode([text1, text2])
        similarity = self.cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
        )[0][0]

        return float(similarity)

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
        # Try sentence transformers (best)
        try:
            backend = SentenceTransformerBackend()
            logger.info("Using SentenceTransformerBackend (best accuracy)")
            return backend
        except (ImportError, RuntimeError, Exception) as e:
            logger.debug(f"sentence-transformers not available: {e}")

        # Try TF-IDF (good)
        try:
            backend = TFIDFBackend()
            logger.info("Using TFIDFBackend (good accuracy)")
            return backend
        except (ImportError, RuntimeError, Exception) as e:
            logger.debug(f"scikit-learn not available: {e}")

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

    def reset(self):
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
    if preferred == "jaccard":
        return JaccardBackend()

    if preferred == "tfidf":
        return TFIDFBackend()

    if preferred == "sentence-transformer":
        return SentenceTransformerBackend()

    # Auto-select best available
    try:
        return SentenceTransformerBackend()
    except (ImportError, RuntimeError, Exception):
        pass

    try:
        return TFIDFBackend()
    except (ImportError, RuntimeError, Exception):
        pass

    return JaccardBackend()

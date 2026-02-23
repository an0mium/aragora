"""
Convergence detector for multi-agent debates.

Detects when agents' positions have converged semantically, allowing
early termination of debates when further rounds would provide
diminishing returns.
"""

from __future__ import annotations

import logging
import os

from aragora.debate.cache.embeddings_lru import cleanup_embedding_cache
from aragora.debate.convergence.cache import cleanup_similarity_cache
from aragora.debate.convergence.metrics import ConvergenceResult
from aragora.debate.similarity.backends import (
    JaccardBackend,
    SimilarityBackend,
    _ENV_CONVERGENCE_BACKEND,
    _normalize_backend_name,
    get_similarity_backend,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Convergence Detector
# =============================================================================


class ConvergenceDetector:
    """
    Detects when debate has converged semantically.

    Uses semantic similarity between consecutive rounds to determine
    if agents have reached consensus or are still refining positions.

    Thresholds:
        - converged: >=85% similarity (agents agree)
        - refining: 40-85% similarity (still improving)
        - diverging: <40% similarity (positions splitting)
    """

    def __init__(
        self,
        convergence_threshold: float = 0.85,
        divergence_threshold: float = 0.40,
        min_rounds_before_check: int = 1,
        consecutive_rounds_needed: int = 1,
        debate_id: str | None = None,
    ):
        """
        Initialize convergence detector.

        Args:
            convergence_threshold: Similarity threshold for convergence (default 0.85)
            divergence_threshold: Below this is diverging (default 0.40)
            min_rounds_before_check: Minimum rounds before checking (default 1)
            consecutive_rounds_needed: Stable rounds needed for convergence (default 1)
            debate_id: Debate ID for scoped caching (prevents cross-debate contamination)
        """
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold
        self.min_rounds_before_check = min_rounds_before_check
        self.consecutive_rounds_needed = consecutive_rounds_needed
        self.consecutive_stable_count = 0
        self.debate_id = debate_id
        self.backend = self._select_backend()

        logger.info("ConvergenceDetector initialized with %s", self.backend.__class__.__name__)

    def _select_backend(self) -> SimilarityBackend:
        """
        Select best available similarity backend using SimilarityFactory.

        Uses the unified SimilarityFactory for backend selection, which:
        - Respects ARAGORA_SIMILARITY_BACKEND environment variable
        - Auto-selects best available backend based on input size
        - Handles debate_id for scoped caching
        """
        from aragora.debate.similarity.factory import get_backend

        # Check for legacy env override
        env_override = _normalize_backend_name(os.getenv(_ENV_CONVERGENCE_BACKEND, ""))
        if env_override:
            try:
                backend = get_similarity_backend(env_override, debate_id=self.debate_id)
                logger.info("Using %s backend via %s", env_override, _ENV_CONVERGENCE_BACKEND)
                return backend
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(
                    "%s=%s failed: %s. Falling back to factory.", _ENV_CONVERGENCE_BACKEND, env_override, e
                )
            except (ValueError, TypeError, AttributeError) as e:
                logger.exception(
                    "%s=%s unexpected error: %s. Falling back to factory.", _ENV_CONVERGENCE_BACKEND, env_override, e
                )

        # Use SimilarityFactory for unified backend selection
        try:
            backend = get_backend(
                preferred="auto",
                input_size=10,  # Default for typical debate sizes
                debate_id=self.debate_id,
            )
            logger.info("Using %s via SimilarityFactory", backend.__class__.__name__)
            return backend
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("SimilarityFactory failed: %s. Using JaccardBackend fallback.", e)
            return JaccardBackend()

    def check_convergence(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> ConvergenceResult | None:
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
        if hasattr(self.backend, "compute_pairwise_similarities"):
            texts_current = [current_responses[a] for a in agent_list]
            texts_previous = [previous_responses[a] for a in agent_list]
            similarities = self.backend.compute_pairwise_similarities(texts_current, texts_previous)
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

    def check_within_round_convergence(
        self,
        responses: dict[str, str],
        threshold: float | None = None,
    ) -> tuple[bool, float, float]:
        """
        Check if all agents' responses within a single round have converged.

        Uses ANN-optimized vectorized operations with early termination for O(n log n)
        complexity instead of O(n^2) pairwise comparison.

        This is useful for detecting when agents agree with each other within a round,
        which can indicate premature consensus or echo chamber effects.

        Args:
            responses: Agent name -> response text for current round
            threshold: Convergence threshold (defaults to self.convergence_threshold)

        Returns:
            Tuple of (converged: bool, min_similarity: float, avg_similarity: float)
        """
        import numpy as np

        from aragora.debate.similarity.ann import (
            compute_batch_similarity_fast,
            find_convergence_threshold,
        )

        if threshold is None:
            threshold = self.convergence_threshold

        texts = list(responses.values())
        if len(texts) < 2:
            return True, 1.0, 1.0

        # Get embeddings using backend
        embeddings = None
        if hasattr(self.backend, "_get_embedding"):
            # SentenceTransformerBackend
            embeddings_list = [self.backend._get_embedding(t) for t in texts]
            embeddings = np.vstack(embeddings_list).astype(np.float32)
        elif hasattr(self.backend, "vectorizer"):
            # TFIDFBackend
            from scipy.sparse import issparse

            tfidf_matrix = self.backend.vectorizer.fit_transform(texts)
            if issparse(tfidf_matrix):
                embeddings = tfidf_matrix.toarray().astype(np.float32)
            else:
                embeddings = np.array(tfidf_matrix).astype(np.float32)

        if embeddings is not None:
            # Use optimized ANN functions with early termination
            converged, min_sim = find_convergence_threshold(embeddings, threshold=threshold)
            avg_sim = compute_batch_similarity_fast(embeddings)
            return converged, min_sim, avg_sim

        # Fallback to individual comparisons for JaccardBackend
        similarities = []
        for i, t1 in enumerate(texts):
            for t2 in texts[i + 1 :]:
                sim = self.backend.compute_similarity(t1, t2)
                similarities.append(sim)
                # Early termination
                if sim < threshold:
                    return False, sim, sum(similarities) / len(similarities)

        min_sim = min(similarities) if similarities else 1.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
        return min_sim >= threshold, min_sim, avg_sim

    def check_convergence_fast(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> ConvergenceResult | None:
        """
        Fast convergence check with ANN optimizations and early termination.

        Same interface as check_convergence but uses vectorized operations
        for better performance with many agents.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number (1-indexed)

        Returns:
            ConvergenceResult or None if too early to check
        """
        import numpy as np

        # Don't check before minimum rounds
        if round_number <= self.min_rounds_before_check:
            return None

        # Match agents between rounds
        common_agents = set(current_responses.keys()) & set(previous_responses.keys())
        if not common_agents:
            logger.warning("No matching agents between rounds")
            return None

        agent_list = list(common_agents)

        # Try optimized path with embeddings
        embeddings_curr = None
        embeddings_prev = None

        if hasattr(self.backend, "_get_embedding"):
            embeddings_curr = np.vstack(
                [self.backend._get_embedding(current_responses[a]) for a in agent_list]
            ).astype(np.float32)
            embeddings_prev = np.vstack(
                [self.backend._get_embedding(previous_responses[a]) for a in agent_list]
            ).astype(np.float32)

        if embeddings_curr is not None and embeddings_prev is not None:
            # Compute pairwise similarities between current and previous
            # Normalize embeddings
            norms_curr = np.linalg.norm(embeddings_curr, axis=1, keepdims=True)
            norms_prev = np.linalg.norm(embeddings_prev, axis=1, keepdims=True)
            norms_curr = np.where(norms_curr == 0, 1, norms_curr)
            norms_prev = np.where(norms_prev == 0, 1, norms_prev)
            norm_curr = embeddings_curr / norms_curr
            norm_prev = embeddings_prev / norms_prev

            # Diagonal of matrix product gives per-agent similarity
            per_agent_sims = np.sum(norm_curr * norm_prev, axis=1)
            per_agent = dict(zip(agent_list, per_agent_sims.tolist()))

            min_similarity = float(np.min(per_agent_sims))
            avg_similarity = float(np.mean(per_agent_sims))
        else:
            # Fallback to standard computation
            return self.check_convergence(current_responses, previous_responses, round_number)

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

    def record_convergence_metrics(
        self,
        topic: str,
        convergence_round: int,
        total_rounds: int,
        final_similarity: float,
        per_round_similarity: list[float] | None = None,
    ) -> None:
        """Record convergence metrics for a completed debate.

        Stores convergence speed data in the ConvergenceHistoryStore so
        that future debates on similar topics can estimate optimal round
        counts and benefit from historical convergence patterns.

        Args:
            topic: The debate task/topic string.
            convergence_round: Round at which convergence was detected (0 = never).
            total_rounds: Total rounds executed in the debate.
            final_similarity: Final average similarity between agents.
            per_round_similarity: Optional list of avg similarity per round.
        """
        try:
            from aragora.debate.convergence.history import get_convergence_history_store

            store = get_convergence_history_store()
            if store is None:
                return

            store.store(
                topic=topic,
                convergence_round=convergence_round,
                total_rounds=total_rounds,
                final_similarity=final_similarity,
                per_round_similarity=per_round_similarity,
                debate_id=self.debate_id or "",
            )

        except ImportError:
            pass  # History store not available
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            logger.debug("Failed to record convergence metrics: %s", e)

    def cleanup(self) -> None:
        """Cleanup resources when debate session ends.

        Should be called when the debate completes to free memory.
        Cleans up embedding caches associated with this debate.
        """
        if self.debate_id:
            cleanup_embedding_cache(self.debate_id)
            cleanup_similarity_cache(self.debate_id)
            logger.debug("ConvergenceDetector cleanup complete: debate=%s", self.debate_id)

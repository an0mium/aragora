"""
Vote weight calculation for consensus phase.

This module extracts the vote weighting logic from ConsensusPhase,
providing a clean interface for calculating agent voting weights
based on multiple factors:
- Reputation (from memory)
- Reliability (from capability probing)
- Consistency (from FlipDetector)
- Calibration (from ELO or CalibrationTracker)
- Self-vote penalty (Agent-as-a-Judge bias mitigation)
- Verbosity penalty (Agent-as-a-Judge bias mitigation)

Performance optimizations:
- Session-based weight caching to avoid redundant calculations
- Automatic cache invalidation on ELO updates
- Thread-safe caching with minimal lock contention
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.debate.bias_mitigation import (
    SelfVoteConfig,
    VerbosityBiasConfig,
    detect_self_vote,
    calculate_verbosity_factor,
)

if TYPE_CHECKING:
    from aragora.core import Agent, Vote

logger = logging.getLogger(__name__)


# =============================================================================
# Weight Cache for Session-Based Caching
# =============================================================================


@dataclass
class CachedWeight:
    """Cached weight result with metadata for invalidation."""

    weight: float
    factors: WeightFactors
    computed_at: float  # timestamp
    elo_version: int  # ELO version for invalidation


class WeightCache:
    """
    Session-scoped cache for agent vote weights.

    Provides significant performance improvement by caching weight calculations
    that are expensive due to multiple external lookups (ELO, memory, etc.).

    Features:
    - Per-debate-session isolation to prevent cross-debate contamination
    - Automatic invalidation on ELO updates via version tracking
    - TTL-based expiry for long-running debates
    - Thread-safe operations with minimal lock contention

    Performance impact:
    - Without cache: O(n) lookups per consensus check (n = external systems)
    - With cache: O(1) after first computation
    - Expected speedup: 5-20x for repeated consensus checks
    """

    def __init__(
        self,
        session_id: str,
        ttl_seconds: float = 300.0,  # 5 minutes default
        max_size: int = 256,
    ):
        """
        Initialize weight cache.

        Args:
            session_id: Unique identifier for this debate session
            ttl_seconds: Time-to-live for cached entries (default 5 minutes)
            max_size: Maximum cache entries (LRU eviction when exceeded)
        """
        self.session_id = session_id
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CachedWeight] = {}
        self._lock = threading.RLock()
        self._elo_version = 0
        self._hits = 0
        self._misses = 0

    def _make_key(self, agent_name: str, domain: str) -> str:
        """Generate cache key for agent+domain combination."""
        return f"{agent_name}:{domain}"

    def get(
        self,
        agent_name: str,
        domain: str,
    ) -> tuple[float, WeightFactors] | None:
        """
        Get cached weight for an agent.

        Args:
            agent_name: Name of the agent
            domain: Debate domain

        Returns:
            Tuple of (weight, factors) or None if not cached/expired
        """
        key = self._make_key(agent_name, domain)
        now = time.time()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL expiry
            if now - entry.computed_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Check ELO version for invalidation
            if entry.elo_version != self._elo_version:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.weight, entry.factors

    def put(
        self,
        agent_name: str,
        domain: str,
        weight: float,
        factors: WeightFactors,
    ) -> None:
        """
        Store computed weight in cache.

        Args:
            agent_name: Name of the agent
            domain: Debate domain
            weight: Computed weight value
            factors: Weight factors breakdown
        """
        key = self._make_key(agent_name, domain)
        now = time.time()

        with self._lock:
            # LRU eviction if at capacity
            while len(self._cache) >= self.max_size:
                # Find oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].computed_at,
                )
                del self._cache[oldest_key]

            self._cache[key] = CachedWeight(
                weight=weight,
                factors=factors,
                computed_at=now,
                elo_version=self._elo_version,
            )

    def invalidate_elo(self) -> None:
        """
        Invalidate all cached weights due to ELO update.

        Called when ELO ratings change (e.g., after a debate round).
        Uses version tracking for efficient lazy invalidation.
        """
        with self._lock:
            self._elo_version += 1
            logger.debug(
                "Weight cache ELO invalidation: session=%s version=%s",
                self.session_id,
                self._elo_version,
            )

    def invalidate_agent(self, agent_name: str) -> None:
        """
        Invalidate cached weights for a specific agent.

        Args:
            agent_name: Agent whose weights should be invalidated
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{agent_name}:")]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "session_id": self.session_id,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "elo_version": self._elo_version,
            }


# Global cache manager for weight caches
_weight_cache_manager: dict[str, WeightCache] = {}
_weight_cache_lock = threading.Lock()


def get_weight_cache(session_id: str, ttl_seconds: float = 300.0) -> WeightCache:
    """
    Get or create a weight cache for a debate session.

    Args:
        session_id: Unique debate session identifier
        ttl_seconds: Cache TTL in seconds

    Returns:
        WeightCache instance for this session
    """
    with _weight_cache_lock:
        if session_id not in _weight_cache_manager:
            _weight_cache_manager[session_id] = WeightCache(
                session_id=session_id,
                ttl_seconds=ttl_seconds,
            )
        return _weight_cache_manager[session_id]


def cleanup_weight_cache(session_id: str) -> None:
    """
    Cleanup weight cache for a completed debate.

    Should be called when a debate ends to free memory.

    Args:
        session_id: Debate session ID to cleanup
    """
    with _weight_cache_lock:
        if session_id in _weight_cache_manager:
            _weight_cache_manager[session_id].clear()
            del _weight_cache_manager[session_id]
            logger.debug("Cleaned up weight cache for session %s", session_id)


@dataclass
class WeightFactors:
    """Individual weight factors for an agent.

    Stores the component weights that are multiplied together
    to produce the final vote weight.
    """

    reputation: float = 1.0  # 0.5-1.5 from memory vote history
    reliability: float = 1.0  # 0.0-1.0 from capability probing
    consistency: float = 1.0  # 0.5-1.0 from FlipDetector
    calibration: float = 1.0  # 0.5-1.5 from ELO calibration score
    elo_skill: float = 1.0  # 0.5-2.0 from ELO domain skill rating
    self_vote: float = 1.0  # 0.0-1.0 penalty if voting for own proposal
    verbosity: float = 1.0  # 0.7-1.0 penalty for excessively long proposals

    @property
    def total(self) -> float:
        """Calculate combined weight from all factors."""
        return (
            self.reputation
            * self.reliability
            * self.consistency
            * self.calibration
            * self.elo_skill
            * self.self_vote
            * self.verbosity
        )


@dataclass
class WeightCalculatorConfig:
    """Configuration for vote weight calculation.

    Controls which weight factors are enabled, their parameters, and bounds.

    Weight Factors (all multiply together):
    - reputation: From memory vote history (0.5-1.5)
    - reliability: From capability probing (0.0-1.0)
    - consistency: From FlipDetector (0.5-1.0)
    - calibration: From ELO calibration score (0.5-1.5)
    - elo_skill: From agent's ELO rating (elo_min_weight to elo_max_weight)
    - self_vote: Penalty if voting for own proposal (0.0-1.0)
    - verbosity: Penalty for excessively long proposals (0.7-1.0)

    Example configurations:
    - Default: reputation * reliability * consistency * calibration * elo_skill
    - Production: Add self_vote and verbosity mitigation
    - Testing: Disable all factors (all weights = 1.0)
    """

    # Enable/disable individual weight factors
    enable_reputation: bool = True  # Use memory-based vote history
    enable_reliability: bool = True  # Use capability probing results
    enable_consistency: bool = True  # Use FlipDetector consistency scores
    enable_calibration: bool = True  # Use ELO calibration scores
    enable_elo_skill: bool = True  # Use ELO rating for skill-based weighting

    # ELO skill weighting parameters
    # Formula: weight = 1.0 + ((elo - baseline) / scale) * factor
    # Example: ELO=2000, baseline=1500, scale=500, factor=0.3
    #          weight = 1.0 + ((2000-1500)/500) * 0.3 = 1.0 + 0.3 = 1.3
    elo_baseline: float = 1500.0  # ELO that maps to weight 1.0
    elo_scale: float = 500.0  # ELO points for full factor effect
    elo_weight_factor: float = 0.3  # Max adjustment at ±1 scale (1.0 ± 0.3)
    elo_min_weight: float = 0.5  # Floor for ELO skill weight
    elo_max_weight: float = 2.0  # Ceiling for ELO skill weight

    # Bounds for final combined weight
    min_weight: float = 0.1  # Minimum final vote weight
    max_weight: float = 5.0  # Maximum final vote weight

    # Agent-as-a-Judge bias mitigation (recommended for production)
    # Self-vote detection (penalize agents voting for own proposals)
    enable_self_vote_mitigation: bool = False  # Enable in production
    self_vote_mode: str = "downweight"  # "exclude", "downweight", "log_only"
    self_vote_downweight: float = 0.5  # Weight multiplier for self-votes

    # Verbosity normalization (penalize excessively long proposals)
    enable_verbosity_normalization: bool = False  # Enable in production
    verbosity_target_length: int = 1000  # Target proposal length (chars)
    verbosity_penalty_threshold: float = 3.0  # Penalty starts at 3x target
    verbosity_max_penalty: float = 0.3  # Max penalty (weight floor = 0.7)


class WeightCalculator:
    """Calculate agent voting weights from multiple sources.

    Performance optimization: Uses session-based weight caching to avoid
    redundant calculations. Cache is automatically invalidated when ELO
    ratings change.

    Usage:
        calculator = WeightCalculator(
            memory=memory_system,
            elo_system=elo_system,
            flip_detector=flip_detector,
            agent_weights=probe_weights,
            session_id="debate-123",  # Enable caching
        )

        # Calculate weights for all agents (cached after first call)
        weights = calculator.compute_weights(agents)

        # Get individual weight with breakdown
        weight, factors = calculator.get_weight_with_factors(agent_name)

        # Invalidate cache after ELO updates
        calculator.invalidate_elo_cache()

        # Cleanup when debate ends
        calculator.cleanup()
    """

    def __init__(
        self,
        memory: Any = None,
        elo_system: Any = None,
        flip_detector: Any = None,
        agent_weights: dict[str, float] | None = None,
        calibration_tracker: Any = None,
        get_calibration_weight: Callable[[str], float] | None = None,
        config: WeightCalculatorConfig | None = None,
        domain: str = "general",
        session_id: str | None = None,
        enable_cache: bool = True,
        cache_ttl_seconds: float = 300.0,
    ):
        """Initialize the weight calculator.

        Args:
            memory: Memory system with get_vote_weight method
            elo_system: ELO system for calibration and skill scores
            flip_detector: FlipDetector for consistency scores
            agent_weights: Pre-computed reliability weights from probing
            calibration_tracker: CalibrationTracker for calibration scores
            get_calibration_weight: Fallback callback for calibration
            config: Configuration for weight calculation
            domain: Debate domain for domain-specific ELO lookup
            session_id: Unique session ID for caching (None disables caching)
            enable_cache: Whether to enable weight caching (default True)
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes)
        """
        self.memory = memory
        self.elo_system = elo_system
        self.flip_detector = flip_detector
        self.agent_weights = agent_weights or {}
        self.calibration_tracker = calibration_tracker
        self._get_calibration_weight = get_calibration_weight
        self.config = config or WeightCalculatorConfig()
        self.domain = domain

        # Cache for batch operations (ELO ratings)
        self._ratings_cache: dict[str, Any] = {}

        # Context for bias mitigation (set during compute_weights_with_context)
        self._current_votes: list[Vote] = []
        self._current_proposals: dict[str, str] = {}

        # Session-based weight caching
        self._session_id = session_id
        self._enable_cache = enable_cache and session_id is not None
        self._cache_ttl = cache_ttl_seconds
        self._weight_cache: WeightCache | None = None

        if self._enable_cache and session_id:
            self._weight_cache = get_weight_cache(session_id, cache_ttl_seconds)
            logger.debug(
                "Weight calculator caching enabled: session=%s ttl=%ss",
                session_id,
                cache_ttl_seconds,
            )

    def compute_weights(self, agents: list[Agent]) -> dict[str, float]:
        """Compute vote weights for all agents.

        Args:
            agents: List of agents to compute weights for

        Returns:
            Dict mapping agent names to their weights
        """
        # Batch fetch ELO ratings for efficiency
        self._prefetch_ratings([a.name for a in agents])

        weights = {}
        for agent in agents:
            weights[agent.name] = self.get_weight(agent.name)

        return weights

    def compute_weights_with_context(
        self,
        agents: list[Agent],
        votes: list[Vote],
        proposals: dict[str, str],
    ) -> dict[str, float]:
        """Compute vote weights with bias mitigation context.

        This method enables self-vote detection and verbosity normalization
        by providing the vote and proposal context.

        Args:
            agents: List of agents to compute weights for
            votes: List of votes cast by agents
            proposals: Dict mapping agent names to their proposals

        Returns:
            Dict mapping agent names to their weights
        """
        # Store context for bias mitigation
        self._current_votes = votes
        self._current_proposals = proposals

        # Batch fetch ELO ratings for efficiency
        self._prefetch_ratings([a.name for a in agents])

        weights = {}
        for agent in agents:
            # Find this agent's vote
            agent_vote = next((v for v in votes if getattr(v, "agent", None) == agent.name), None)
            weights[agent.name] = self.get_weight_for_vote(agent.name, agent_vote)

        # Clear context after computation
        self._current_votes = []
        self._current_proposals = {}

        return weights

    def get_weight_for_vote(
        self,
        agent_name: str,
        vote: Vote | None = None,
    ) -> float:
        """Get the combined vote weight for an agent with bias mitigation.

        Args:
            agent_name: Name of the agent
            vote: The vote cast by this agent (for self-vote detection)

        Returns:
            Combined weight (product of all factors)
        """
        factors = self._compute_factors_with_vote(agent_name, vote)
        weight = factors.total

        # Apply bounds
        weight = max(self.config.min_weight, min(self.config.max_weight, weight))

        return weight

    def get_weight(self, agent_name: str) -> float:
        """Get the combined vote weight for an agent.

        Uses caching when enabled to avoid redundant calculations.

        Args:
            agent_name: Name of the agent

        Returns:
            Combined weight (product of all factors)
        """
        # Check cache first
        if self._weight_cache:
            cached = self._weight_cache.get(agent_name, self.domain)
            if cached is not None:
                return cached[0]  # Return cached weight

        factors = self._compute_factors(agent_name)
        weight = factors.total

        # Apply bounds
        weight = max(self.config.min_weight, min(self.config.max_weight, weight))

        # Cache the result
        if self._weight_cache:
            self._weight_cache.put(agent_name, self.domain, weight, factors)

        return weight

    def get_weight_with_factors(self, agent_name: str) -> tuple[float, WeightFactors]:
        """Get weight with breakdown of individual factors.

        Useful for debugging and understanding weight contributions.
        Uses caching when enabled.

        Args:
            agent_name: Name of the agent

        Returns:
            Tuple of (final_weight, WeightFactors)
        """
        # Check cache first
        if self._weight_cache:
            cached = self._weight_cache.get(agent_name, self.domain)
            if cached is not None:
                return cached  # Return (weight, factors)

        factors = self._compute_factors(agent_name)
        weight = max(self.config.min_weight, min(self.config.max_weight, factors.total))

        # Cache the result
        if self._weight_cache:
            self._weight_cache.put(agent_name, self.domain, weight, factors)

        return weight, factors

    def _prefetch_ratings(self, agent_names: list[str]) -> None:
        """Batch fetch ELO ratings for efficiency."""
        if not self.elo_system:
            return

        try:
            self._ratings_cache = self.elo_system.get_ratings_batch(agent_names)
        except Exception as e:  # noqa: BLE001 - graceful degradation, batch prefetch is best-effort
            logger.debug("Batch ratings fetch failed: %s", e)
            self._ratings_cache = {}

    def _compute_factors(self, agent_name: str) -> WeightFactors:
        """Compute individual weight factors for an agent."""
        factors = WeightFactors()

        # Reputation weight (0.5-1.5)
        if self.config.enable_reputation:
            factors.reputation = self._get_reputation_weight(agent_name)

        # Reliability weight (0.0-1.0)
        if self.config.enable_reliability:
            factors.reliability = self._get_reliability_weight(agent_name)

        # Consistency weight (0.5-1.0)
        if self.config.enable_consistency:
            factors.consistency = self._get_consistency_weight(agent_name)

        # Calibration weight (0.5-1.5)
        if self.config.enable_calibration:
            factors.calibration = self._get_calibration_weight_for_agent(agent_name)

        # ELO skill weight (0.5-2.0)
        if self.config.enable_elo_skill:
            factors.elo_skill = self._get_elo_skill_weight(agent_name)

        return factors

    def _compute_factors_with_vote(
        self,
        agent_name: str,
        vote: Vote | None = None,
    ) -> WeightFactors:
        """Compute weight factors including bias mitigation factors.

        Args:
            agent_name: Name of the agent
            vote: The vote cast by this agent (for self-vote detection)

        Returns:
            WeightFactors with all factors including bias mitigation
        """
        # Start with base factors
        factors = self._compute_factors(agent_name)

        # Self-vote penalty (Agent-as-a-Judge bias mitigation)
        if self.config.enable_self_vote_mitigation and vote:
            factors.self_vote = self._get_self_vote_factor(agent_name, vote)

        # Verbosity penalty (Agent-as-a-Judge bias mitigation)
        if self.config.enable_verbosity_normalization and vote:
            factors.verbosity = self._get_verbosity_factor(vote)

        return factors

    def _get_self_vote_factor(
        self,
        agent_name: str,
        vote: Vote,
    ) -> float:
        """Get self-vote penalty factor.

        Detects if an agent voted for their own proposal and applies penalty.

        Args:
            agent_name: Name of the voting agent
            vote: The vote cast

        Returns:
            Weight factor (1.0 = no penalty, lower = penalized)
        """
        # Create config for self-vote detection
        SelfVoteConfig(
            enabled=True,
            mode=self.config.self_vote_mode,
            downweight_factor=self.config.self_vote_downweight,
        )

        # Check if this is a self-vote
        is_self_vote = detect_self_vote(vote, self._current_proposals)

        if is_self_vote:
            logger.info(
                "self_vote_detected agent=%s choice=%s mode=%s",
                agent_name,
                vote.choice,
                self.config.self_vote_mode,
            )

            if self.config.self_vote_mode == "exclude":
                # Zero weight effectively excludes this vote
                return 0.0
            elif self.config.self_vote_mode == "downweight":
                return self.config.self_vote_downweight
            # "log_only" mode just logs, no penalty
            return 1.0

        return 1.0

    def _get_verbosity_factor(self, vote: Vote) -> float:
        """Get verbosity penalty factor for the voted proposal.

        Penalizes excessively long proposals to mitigate verbosity bias.

        Args:
            vote: The vote cast (contains the choice/proposal voted for)

        Returns:
            Weight factor (0.7-1.0, lower = penalized for verbosity)
        """
        # Get the proposal that was voted for
        choice = getattr(vote, "choice", None)
        if not choice or not self._current_proposals:
            return 1.0

        # Find proposal text - choice might be agent name or proposal text
        proposal_text = self._current_proposals.get(choice, "")
        if not proposal_text:
            # Try finding by partial match
            for agent_name, text in self._current_proposals.items():
                if choice in agent_name or agent_name in choice:
                    proposal_text = text
                    break

        if not proposal_text:
            return 1.0

        # Create config for verbosity calculation
        verbosity_config = VerbosityBiasConfig(
            enabled=True,
            target_length=self.config.verbosity_target_length,
            penalty_threshold=self.config.verbosity_penalty_threshold,
            max_penalty=self.config.verbosity_max_penalty,
        )

        factor = calculate_verbosity_factor(len(proposal_text), verbosity_config)

        if factor < 1.0:
            logger.debug(
                f"verbosity_penalty choice={choice[:30]}... "
                f"length={len(proposal_text)} factor={factor:.2f}"
            )

        return factor

    def _get_reputation_weight(self, agent_name: str) -> float:
        """Get reputation weight from memory system."""
        if not self.memory or not hasattr(self.memory, "get_vote_weight"):
            return 1.0

        try:
            return self.memory.get_vote_weight(agent_name)
        except Exception as e:  # noqa: BLE001 - graceful degradation, return default weight on error
            logger.debug("Reputation weight error for %s: %s", agent_name, e)
            return 1.0

    def _get_reliability_weight(self, agent_name: str) -> float:
        """Get reliability weight from capability probing."""
        if not self.agent_weights:
            return 1.0

        return self.agent_weights.get(agent_name, 1.0)

    def _get_consistency_weight(self, agent_name: str) -> float:
        """Get consistency weight from FlipDetector."""
        if not self.flip_detector:
            return 1.0

        try:
            consistency = self.flip_detector.get_agent_consistency(agent_name)
            # Map 0.0-1.0 consistency score to 0.5-1.0 weight
            return 0.5 + (consistency.consistency_score * 0.5)
        except Exception as e:  # noqa: BLE001 - graceful degradation, return default weight on error
            logger.debug("Consistency weight error for %s: %s", agent_name, e)
            return 1.0

    def _get_calibration_weight_for_agent(self, agent_name: str) -> float:
        """Get calibration weight from ELO or CalibrationTracker."""
        # Try cached ELO ratings first
        if agent_name in self._ratings_cache:
            cal_score = self._ratings_cache[agent_name].calibration_score
            return 0.5 + cal_score

        # Fallback to callback
        if self._get_calibration_weight:
            try:
                return self._get_calibration_weight(agent_name)
            except Exception as e:  # noqa: BLE001 - graceful degradation, return default weight on error
                logger.debug("Calibration weight callback error for %s: %s", agent_name, e)

        return 1.0

    def _get_elo_skill_weight(self, agent_name: str) -> float:
        """Get ELO skill weight based on agent's domain-specific rating.

        Higher-rated agents (better historical performance) get more voting
        weight. The weight is bounded to prevent extreme influence.

        Formula:
            elo_factor = (elo - baseline) / scale
            weight = 1.0 + (elo_factor * weight_factor)
            weight = clamp(weight, min_weight, max_weight)

        Example with defaults (baseline=1500, scale=500, factor=0.3):
            - ELO 1000 → factor=-1.0 → weight=0.7
            - ELO 1500 → factor=0.0  → weight=1.0
            - ELO 2000 → factor=1.0  → weight=1.3

        Returns:
            Weight factor between elo_min_weight and elo_max_weight
        """
        if not self.elo_system:
            return 1.0

        try:
            # Try to get domain-specific ELO first
            elo = None

            # Check cached ratings
            if agent_name in self._ratings_cache:
                rating = self._ratings_cache[agent_name]
                # Try domain-specific ELO
                if hasattr(rating, "domain_elos") and rating.domain_elos:
                    domain_elos = rating.domain_elos
                    if isinstance(domain_elos, str):
                        import json

                        domain_elos = json.loads(domain_elos)
                    elo = domain_elos.get(self.domain)
                # Fallback to overall ELO
                if elo is None:
                    elo = getattr(rating, "elo", None)

            # If not in cache, query directly
            if elo is None and hasattr(self.elo_system, "get_rating"):
                rating = self.elo_system.get_rating(agent_name)
                if rating:
                    # Try domain-specific
                    if hasattr(rating, "domain_elos") and rating.domain_elos:
                        domain_elos = rating.domain_elos
                        if isinstance(domain_elos, str):
                            import json

                            domain_elos = json.loads(domain_elos)
                        elo = domain_elos.get(self.domain)
                    if elo is None:
                        elo = getattr(rating, "elo", self.config.elo_baseline)

            if elo is None:
                return 1.0

            # Calculate weight factor
            elo_factor = (elo - self.config.elo_baseline) / self.config.elo_scale
            weight = 1.0 + (elo_factor * self.config.elo_weight_factor)

            # Clamp to bounds
            weight = max(self.config.elo_min_weight, min(self.config.elo_max_weight, weight))

            if weight != 1.0:
                logger.debug(
                    f"elo_skill_weight agent={agent_name} domain={self.domain} "
                    f"elo={elo:.0f} weight={weight:.3f}"
                )

            return weight

        except Exception as e:  # noqa: BLE001 - graceful degradation, return default weight on error
            logger.debug("ELO skill weight error for %s: %s", agent_name, e)
            return 1.0

    def clear_cache(self) -> None:
        """Clear the ratings cache."""
        self._ratings_cache.clear()

    def invalidate_elo_cache(self) -> None:
        """Invalidate weight cache due to ELO rating updates.

        Call this after ELO ratings change (e.g., after a debate round)
        to ensure fresh weight calculations.
        """
        if self._weight_cache:
            self._weight_cache.invalidate_elo()
            logger.debug("Weight cache invalidated for ELO update: session=%s", self._session_id)

    def invalidate_agent_cache(self, agent_name: str) -> None:
        """Invalidate cached weights for a specific agent.

        Call this when an agent's rating or reputation changes.

        Args:
            agent_name: Agent whose weights should be invalidated
        """
        if self._weight_cache:
            self._weight_cache.invalidate_agent(agent_name)

    def cleanup(self) -> None:
        """Cleanup resources when debate session ends.

        Should be called when the debate completes to free memory.
        """
        self.clear_cache()
        if self._session_id:
            cleanup_weight_cache(self._session_id)
            logger.debug("Weight calculator cleanup complete: session=%s", self._session_id)

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics for monitoring.

        Returns:
            Dict with cache stats or None if caching is disabled
        """
        if self._weight_cache:
            return self._weight_cache.get_stats()
        return None

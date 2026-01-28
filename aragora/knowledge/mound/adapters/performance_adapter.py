"""
PerformanceAdapter - Unified adapter for agent performance metrics.

This adapter merges EloAdapter and RankingAdapter to provide comprehensive
agent performance and expertise tracking for the Knowledge Mound:

- ELO ratings, matches, calibration, relationships (from EloAdapter)
- Domain expertise tracking with caching (from RankingAdapter)
- Bidirectional KM integration (persistence + reverse flow)
- Pattern detection and ELO adjustments from KM knowledge

ID Prefixes:
- el_: ELO ratings, matches, calibrations, relationships
- ex_: Expertise records
- dm_: Domain mappings

Usage:
    from aragora.knowledge.mound.adapters import PerformanceAdapter

    adapter = PerformanceAdapter()

    # Store performance data
    adapter.store_rating(rating, debate_id="d-123")
    adapter.store_agent_expertise("claude", "security", 1650, delta=50)

    # Query expertise
    experts = adapter.get_domain_experts("security", limit=5)

    # Analyze KM patterns
    patterns = await adapter.analyze_km_patterns_for_agent("claude", km_items)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

if TYPE_CHECKING:
    from aragora.knowledge.unified.types import KnowledgeItem
    from aragora.ranking.elo import AgentRating, EloSystem, MatchResult
    from aragora.ranking.relationships import RelationshipMetrics

EventCallback = Callable[[str, Dict[str, Any]], None]

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses from EloAdapter
# =============================================================================


@dataclass
class KMEloPattern:
    """Pattern detected in Knowledge Mound that can influence ELO.

    These patterns are derived from analyzing debate outcomes, claim
    validation, and knowledge contribution across debates.
    """

    agent_name: str
    pattern_type: (
        str  # "success_contributor", "contradiction_source", "domain_expert", "crux_resolver"
    )
    confidence: float  # 0.0-1.0 KM confidence in this pattern
    observation_count: int = 1  # How many times observed
    domain: Optional[str] = None  # Domain if pattern is domain-specific
    debate_ids: List[str] = field(default_factory=list)  # Debates that formed this pattern
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EloAdjustmentRecommendation:
    """Recommendation for ELO adjustment based on KM patterns.

    These are recommendations that should be reviewed before applying,
    as they represent inferred quality from knowledge patterns rather
    than direct match outcomes.
    """

    agent_name: str
    adjustment: float  # Positive = boost, negative = penalty
    reason: str  # Human-readable explanation
    patterns: List[KMEloPattern] = field(default_factory=list)  # Supporting patterns
    confidence: float = 0.7  # Overall confidence in recommendation
    domain: Optional[str] = None  # If domain-specific
    applied: bool = False


@dataclass
class EloSyncResult:
    """Result of syncing KM patterns to ELO."""

    total_patterns: int = 0
    adjustments_recommended: int = 0
    adjustments_applied: int = 0
    adjustments_skipped: int = 0
    total_elo_change: float = 0.0
    agents_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class RatingSearchResult:
    """Wrapper for rating search results with adapter metadata."""

    rating: Dict[str, Any]
    relevance_score: float = 0.0


# =============================================================================
# Dataclasses from RankingAdapter
# =============================================================================


@dataclass
class AgentExpertise:
    """Represents an agent's expertise in a domain."""

    agent_name: str
    domain: str
    elo: float
    confidence: float  # Based on number of debates
    last_updated: str
    debate_count: int = 0


@dataclass
class ExpertiseSearchResult:
    """Wrapper for expertise search results with adapter metadata."""

    expertise: Dict[str, Any]
    relevance_score: float = 0.0
    matched_domains: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.matched_domains is None:
            self.matched_domains = []


# =============================================================================
# PerformanceAdapter
# =============================================================================


class PerformanceAdapter:
    """
    Unified adapter for agent performance metrics (ELO + domain expertise).

    Combines functionality from EloAdapter and RankingAdapter to provide:
    - Complete agent performance lifecycle (ratings, matches, calibration)
    - Domain-based expertise tracking with caching
    - Bidirectional KM integration (persistence + reverse flow)
    - Pattern detection and ELO adjustments from KM knowledge
    - Event callbacks for real-time updates

    Usage:
        adapter = PerformanceAdapter()

        # After a match, store the updated ratings
        adapter.store_rating(rating, debate_id="d-123")
        adapter.store_match(match_result)

        # Store expertise for team selection
        adapter.store_agent_expertise("claude", "security", 1650, delta=50)

        # Query for team selection
        experts = adapter.get_domain_experts("security", limit=5)
        history = adapter.get_agent_skill_history("claude")

        # KM reverse flow
        patterns = await adapter.analyze_km_patterns_for_agent("claude", items)
        recommendation = adapter.compute_elo_adjustment(patterns)
    """

    # ID Prefixes
    ELO_PREFIX = "el_"
    EXPERTISE_PREFIX = "ex_"
    DOMAIN_PREFIX = "dm_"

    # For backwards compatibility
    ID_PREFIX = "el_"

    # Thresholds
    MIN_DEBATES_FOR_RELATIONSHIP = 5  # Min debates to store relationship metrics
    MIN_ELO_CHANGE = 25  # Minimum ELO change to record expertise
    MIN_DEBATES_FOR_CONFIDENCE = 5  # Debates needed for high expertise confidence

    # Domain keywords for detection (order matters - first match wins)
    DOMAIN_KEYWORDS = {
        "security": [
            "security",
            "vulnerability",
            "exploit",
            "attack",
            "defense",
            "crypto",
            "auth",
            "injection",
            "xss",
            "csrf",
        ],
        "coding": ["code", "programming", "implementation", "algorithm", "function", "class"],
        "architecture": ["architecture", "design", "pattern", "system", "scalable", "microservice"],
        "testing": ["test", "qa", "quality", "bug", "regression", "coverage"],
        "data": ["data", "database", "sql", "analytics", "ml", "machine learning", "ai"],
        "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure", "cloud"],
        "legal": ["legal", "compliance", "regulation", "contract", "liability", "gdpr"],
        "ethics": ["ethics", "moral", "fair", "bias", "responsible", "privacy"],
    }

    # Cache configuration
    DEFAULT_CACHE_TTL_SECONDS = 60.0

    def __init__(
        self,
        elo_system: Optional["EloSystem"] = None,
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ):
        """
        Initialize the adapter.

        Args:
            elo_system: Optional EloSystem instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
            cache_ttl_seconds: TTL for cached queries (default: 60 seconds)
        """
        self._elo_system = elo_system
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback
        self._cache_ttl_seconds = cache_ttl_seconds

        # ELO storage (from EloAdapter)
        self._ratings: Dict[str, Dict[str, Any]] = {}
        self._matches: Dict[str, Dict[str, Any]] = {}
        self._calibrations: Dict[str, Dict[str, Any]] = {}
        self._relationships: Dict[str, Dict[str, Any]] = {}

        # ELO indices
        self._agent_ratings: Dict[str, List[str]] = {}  # agent -> [rating_ids]
        self._agent_matches: Dict[str, List[str]] = {}  # agent -> [match_ids]
        self._domain_ratings: Dict[str, List[str]] = {}  # domain -> [rating_ids]

        # Expertise storage (from RankingAdapter)
        self._expertise: Dict[str, Dict[str, Any]] = {}  # {agent_domain: expertise_data}
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}  # {agent: [records]}

        # Expertise indices
        self._domain_agents: Dict[str, List[str]] = {}  # {domain: [agent_names]}
        self._agent_domains: Dict[str, List[str]] = {}  # {agent_name: [domains]}

        # Cache (from RankingAdapter)
        self._domain_experts_cache: Dict[str, tuple] = {}  # {cache_key: (timestamp, results)}
        self._cache_hits = 0
        self._cache_misses = 0

    # =========================================================================
    # Event System (from EloAdapter)
    # =========================================================================

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    # =========================================================================
    # ELO System Property (shared)
    # =========================================================================

    @property
    def elo_system(self) -> Optional["EloSystem"]:
        """Access the underlying EloSystem."""
        return self._elo_system

    def set_elo_system(self, elo_system: "EloSystem") -> None:
        """Set the ELO system to use."""
        self._elo_system = elo_system

    # =========================================================================
    # Rating Storage Methods (from EloAdapter)
    # =========================================================================

    def store_rating(
        self,
        rating: "AgentRating",
        debate_id: Optional[str] = None,
        reason: str = "match_update",
    ) -> str:
        """
        Store an agent rating snapshot in the Knowledge Mound.

        This is called after each match to track rating progression.

        Args:
            rating: The AgentRating to store
            debate_id: Optional debate ID that triggered this update
            reason: Reason for the rating update

        Returns:
            The rating ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        rating_id = f"{self.ELO_PREFIX}{rating.agent_name}_{timestamp.replace(':', '-')}"

        rating_data = {
            "id": rating_id,
            "agent_name": rating.agent_name,
            "elo": rating.elo,
            "domain_elos": rating.domain_elos,
            "wins": rating.wins,
            "losses": rating.losses,
            "draws": rating.draws,
            "debates_count": rating.debates_count,
            "win_rate": rating.win_rate,
            "games_played": rating.games_played,
            "critiques_accepted": rating.critiques_accepted,
            "critiques_total": rating.critiques_total,
            "critique_acceptance_rate": rating.critique_acceptance_rate,
            "calibration_correct": rating.calibration_correct,
            "calibration_total": rating.calibration_total,
            "calibration_accuracy": rating.calibration_accuracy,
            "debate_id": debate_id,
            "reason": reason,
            "created_at": timestamp,
            "original_updated_at": rating.updated_at,
        }

        self._ratings[rating_id] = rating_data

        # Update indices
        if rating.agent_name not in self._agent_ratings:
            self._agent_ratings[rating.agent_name] = []
        self._agent_ratings[rating.agent_name].append(rating_id)

        # Index by domains
        for domain in rating.domain_elos:
            if domain not in self._domain_ratings:
                self._domain_ratings[domain] = []
            self._domain_ratings[domain].append(rating_id)

        logger.info(f"Stored rating: {rating_id} (elo={rating.elo:.1f})")
        return rating_id

    def store_match(self, match: "MatchResult") -> str:
        """
        Store a match result in the Knowledge Mound.

        Args:
            match: The MatchResult to store

        Returns:
            The match ID
        """
        match_id = f"{self.ELO_PREFIX}match_{match.debate_id}"

        match_data = {
            "id": match_id,
            "debate_id": match.debate_id,
            "winner": match.winner,
            "participants": match.participants,
            "domain": match.domain,
            "scores": match.scores,
            "created_at": match.created_at,
        }

        self._matches[match_id] = match_data

        # Update indices
        for participant in match.participants:
            if participant not in self._agent_matches:
                self._agent_matches[participant] = []
            self._agent_matches[participant].append(match_id)

        logger.info(f"Stored match: {match_id} (winner={match.winner})")
        return match_id

    def store_calibration(
        self,
        agent_name: str,
        debate_id: str,
        predicted_winner: Optional[str],
        predicted_confidence: float,
        actual_winner: Optional[str],
        was_correct: bool,
        brier_score: float,
    ) -> str:
        """
        Store a calibration prediction in the Knowledge Mound.

        Args:
            agent_name: Agent making the prediction
            debate_id: The debate being predicted
            predicted_winner: Predicted winner (None for draw)
            predicted_confidence: Confidence in prediction (0-1)
            actual_winner: Actual winner (None for draw)
            was_correct: Whether prediction was correct
            brier_score: Brier score for this prediction

        Returns:
            The calibration ID
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        cal_id = f"{self.ELO_PREFIX}cal_{agent_name}_{debate_id}"

        cal_data = {
            "id": cal_id,
            "agent_name": agent_name,
            "debate_id": debate_id,
            "predicted_winner": predicted_winner,
            "predicted_confidence": predicted_confidence,
            "actual_winner": actual_winner,
            "was_correct": was_correct,
            "brier_score": brier_score,
            "created_at": timestamp,
        }

        self._calibrations[cal_id] = cal_data

        logger.info(f"Stored calibration: {cal_id} (correct={was_correct})")
        return cal_id

    def store_relationship(self, metrics: "RelationshipMetrics") -> Optional[str]:
        """
        Store relationship metrics between agents.

        Args:
            metrics: The RelationshipMetrics to store

        Returns:
            The relationship ID if stored, None if below threshold
        """
        if metrics.debates_together < self.MIN_DEBATES_FOR_RELATIONSHIP:  # type: ignore[attr-defined]
            logger.debug(f"Relationship {metrics.agent_a}-{metrics.agent_b} below debate threshold")
            return None

        rel_id = f"{self.ELO_PREFIX}rel_{metrics.agent_a}_{metrics.agent_b}"

        rel_data = {
            "id": rel_id,
            "agent_a": metrics.agent_a,
            "agent_b": metrics.agent_b,
            "debates_together": metrics.debates_together,  # type: ignore[attr-defined]
            "a_wins_vs_b": metrics.a_wins_vs_b,  # type: ignore[attr-defined]
            "b_wins_vs_a": metrics.b_wins_vs_a,  # type: ignore[attr-defined]
            "draws": metrics.draws,  # type: ignore[attr-defined]
            "avg_elo_diff": metrics.avg_elo_diff if hasattr(metrics, "avg_elo_diff") else 0.0,
            "synergy_score": metrics.synergy_score if hasattr(metrics, "synergy_score") else 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._relationships[rel_id] = rel_data

        logger.info(f"Stored relationship: {rel_id}")
        return rel_id

    # =========================================================================
    # Expertise Storage Methods (from RankingAdapter)
    # =========================================================================

    def store_agent_expertise(
        self,
        agent_name: str,
        domain: str,
        elo: float,
        delta: float = 0.0,
        debate_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store agent expertise in the Knowledge Mound.

        Args:
            agent_name: Name of the agent
            domain: Domain of expertise
            elo: Current ELO rating
            delta: ELO change from last update
            debate_id: Optional debate ID that triggered update

        Returns:
            The expertise ID if stored, None if below threshold
        """
        # Skip minor changes
        if abs(delta) < self.MIN_ELO_CHANGE:
            logger.debug(f"ELO change too small for {agent_name}: {delta}")
            return None

        expertise_key = f"{agent_name}:{domain}"
        expertise_id = f"{self.EXPERTISE_PREFIX}{expertise_key.replace(':', '_')}"

        # Get or create expertise record
        existing = self._expertise.get(expertise_key)
        debate_count = 1

        if existing:
            debate_count = existing.get("debate_count", 0) + 1

        # Calculate confidence based on debate count
        confidence = min(1.0, debate_count / self.MIN_DEBATES_FOR_CONFIDENCE)

        expertise_data = {
            "id": expertise_id,
            "agent_name": agent_name,
            "domain": domain,
            "elo": elo,
            "previous_elo": existing.get("elo") if existing else elo - delta,
            "delta": delta,
            "confidence": confidence,
            "debate_count": debate_count,
            "last_debate_id": debate_id,
            "created_at": (
                existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat()
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._expertise[expertise_key] = expertise_data

        # Update indices
        if domain not in self._domain_agents:
            self._domain_agents[domain] = []
        if agent_name not in self._domain_agents[domain]:
            self._domain_agents[domain].append(agent_name)

        if agent_name not in self._agent_domains:
            self._agent_domains[agent_name] = []
        if domain not in self._agent_domains[agent_name]:
            self._agent_domains[agent_name].append(domain)

        # Store in history
        if agent_name not in self._agent_history:
            self._agent_history[agent_name] = []
        self._agent_history[agent_name].append(
            {
                "domain": domain,
                "elo": elo,
                "delta": delta,
                "debate_id": debate_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Invalidate cache for this domain
        self.invalidate_domain_cache(domain)

        logger.info(
            f"Stored expertise: {agent_name} in {domain} -> {elo} (confidence={confidence:.2f})"
        )
        return expertise_id

    # =========================================================================
    # Rating Retrieval Methods (from EloAdapter)
    # =========================================================================

    def get_rating(self, rating_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific rating snapshot by ID.

        Args:
            rating_id: The rating ID (may be prefixed with "el_")

        Returns:
            Rating dict or None
        """
        if not rating_id.startswith(self.ELO_PREFIX):
            rating_id = f"{self.ELO_PREFIX}{rating_id}"
        return self._ratings.get(rating_id)

    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific match by ID.

        Args:
            match_id: The match ID (may be prefixed with "el_match_")

        Returns:
            Match dict or None
        """
        if not match_id.startswith(self.ELO_PREFIX):
            match_id = f"{self.ELO_PREFIX}match_{match_id}"
        return self._matches.get(match_id)

    def get_relationship(self, agent_a: str, agent_b: str) -> Optional[Dict[str, Any]]:
        """
        Get relationship metrics between two agents.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Relationship dict or None
        """
        # Check both orderings
        rel_id_1 = f"{self.ELO_PREFIX}rel_{agent_a}_{agent_b}"
        rel_id_2 = f"{self.ELO_PREFIX}rel_{agent_b}_{agent_a}"

        return self._relationships.get(rel_id_1) or self._relationships.get(rel_id_2)

    def get_agent_skill_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get skill progression history for an agent (ELO ratings).

        This is the key query for team selection and task assignment.

        Args:
            agent_name: The agent name
            limit: Maximum snapshots to return

        Returns:
            List of rating snapshots ordered by time (newest first)
        """
        rating_ids = self._agent_ratings.get(agent_name, [])
        results = []

        for rating_id in rating_ids:
            rating = self._ratings.get(rating_id)
            if rating:
                results.append(rating)

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def get_domain_expertise(
        self,
        domain: str,
        min_elo: float = 1000.0,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find agents with expertise in a specific domain (from ELO ratings).

        Returns the most recent rating for each agent in the domain.

        Args:
            domain: The domain to search
            min_elo: Minimum ELO in domain
            limit: Maximum results

        Returns:
            List of rating dicts for agents with domain expertise
        """
        rating_ids = self._domain_ratings.get(domain, [])

        # Get most recent rating per agent
        agent_ratings: Dict[str, Dict[str, Any]] = {}

        for rating_id in rating_ids:
            rating = self._ratings.get(rating_id)
            if not rating:
                continue

            agent = rating["agent_name"]
            domain_elo = rating.get("domain_elos", {}).get(domain, 0)

            if domain_elo < min_elo:
                continue

            # Keep most recent per agent
            if agent not in agent_ratings:
                agent_ratings[agent] = rating
            else:
                existing_time = agent_ratings[agent].get("created_at", "")
                new_time = rating.get("created_at", "")
                if new_time > existing_time:
                    agent_ratings[agent] = rating

        # Sort by domain ELO descending
        results = list(agent_ratings.values())
        results.sort(
            key=lambda x: x.get("domain_elos", {}).get(domain, 0),
            reverse=True,
        )

        return results[:limit]

    def get_agent_matches(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get match history for an agent.

        Args:
            agent_name: The agent name
            limit: Maximum matches to return

        Returns:
            List of match dicts ordered by time (newest first)
        """
        match_ids = self._agent_matches.get(agent_name, [])
        results = []

        for match_id in match_ids:
            match = self._matches.get(match_id)
            if match:
                results.append(match)

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def get_agent_calibration_history(
        self,
        agent_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get calibration prediction history for an agent.

        Args:
            agent_name: The agent name
            limit: Maximum predictions to return

        Returns:
            List of calibration dicts
        """
        results = [
            cal for cal in self._calibrations.values() if cal.get("agent_name") == agent_name
        ]

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    # =========================================================================
    # Expertise Retrieval Methods (from RankingAdapter)
    # =========================================================================

    def get_agent_expertise(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get expertise record for an agent.

        Args:
            agent_name: Agent name
            domain: Optional specific domain (returns all if not specified)

        Returns:
            Expertise dict or None
        """
        if domain:
            expertise_key = f"{agent_name}:{domain}"
            return self._expertise.get(expertise_key)

        # Return all domains for this agent
        domains = self._agent_domains.get(agent_name, [])
        result = {}
        for d in domains:
            expertise_key = f"{agent_name}:{d}"
            if expertise_key in self._expertise:
                result[d] = self._expertise[expertise_key]
        return result if result else None

    def get_domain_experts(
        self,
        domain: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        use_cache: bool = True,
    ) -> List[AgentExpertise]:
        """
        Get top experts for a domain.

        Args:
            domain: Domain to query
            limit: Maximum experts to return
            min_confidence: Minimum confidence threshold
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of AgentExpertise sorted by ELO descending
        """
        # Check cache first
        cache_key = f"{domain}:{limit}:{min_confidence}"
        if use_cache and cache_key in self._domain_experts_cache:
            timestamp, cached_results = self._domain_experts_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                self._cache_hits += 1
                logger.debug(f"Cache hit for domain experts: {domain}")
                return cast(List[AgentExpertise], cached_results)
            else:
                # Cache expired, remove it
                del self._domain_experts_cache[cache_key]

        self._cache_misses += 1

        # Query fresh data
        agents = self._domain_agents.get(domain, [])
        results = []

        for agent_name in agents:
            expertise_key = f"{agent_name}:{domain}"
            expertise = self._expertise.get(expertise_key)

            if expertise and expertise.get("confidence", 0) >= min_confidence:
                results.append(
                    AgentExpertise(
                        agent_name=agent_name,
                        domain=domain,
                        elo=expertise.get("elo", 1500),
                        confidence=expertise.get("confidence", 0.0),
                        last_updated=expertise.get("updated_at", ""),
                        debate_count=expertise.get("debate_count", 0),
                    )
                )

        # Sort by ELO descending
        results.sort(key=lambda x: x.elo, reverse=True)
        results = results[:limit]

        # Cache results
        if use_cache:
            self._domain_experts_cache[cache_key] = (time.time(), results)
            logger.debug(f"Cached domain experts for {domain}: {len(results)} results")

        return results

    def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical performance records for an agent.

        Args:
            agent_name: Agent name
            limit: Maximum records to return
            domain: Optional domain filter

        Returns:
            List of history records (newest first)
        """
        history = self._agent_history.get(agent_name, [])

        if domain:
            history = [h for h in history if h.get("domain") == domain]

        # Sort by timestamp descending
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return history[:limit]

    # =========================================================================
    # Domain Detection (from RankingAdapter)
    # =========================================================================

    def detect_domain(self, question: str) -> str:
        """
        Detect domain from debate question.

        Args:
            question: The debate question

        Returns:
            Detected domain name (defaults to "general")
        """
        question_lower = question.lower()

        # Check each domain's keywords
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question_lower:
                    logger.debug(f"Detected domain '{domain}' from keyword '{keyword}'")
                    return domain

        return "general"

    def get_all_domains(self) -> List[str]:
        """Get all domains with stored expertise."""
        return list(self._domain_agents.keys())

    # =========================================================================
    # Cache Management (from RankingAdapter)
    # =========================================================================

    def invalidate_domain_cache(self, domain: Optional[str] = None) -> int:
        """
        Invalidate the domain experts cache.

        Args:
            domain: If specified, only invalidate cache for this domain.
                   If None, invalidate all cached entries.

        Returns:
            Number of cache entries invalidated
        """
        if domain is None:
            count = len(self._domain_experts_cache)
            self._domain_experts_cache.clear()
            return count

        # Invalidate entries for specific domain
        keys_to_remove = [k for k in self._domain_experts_cache if k.startswith(f"{domain}:")]
        for key in keys_to_remove:
            del self._domain_experts_cache[key]
        return len(keys_to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache_hits, cache_misses, cache_size, hit_rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._domain_experts_cache),
            "hit_rate": hit_rate,
            "ttl_seconds": self._cache_ttl_seconds,
        }

    # =========================================================================
    # Knowledge Item Conversion (from EloAdapter)
    # =========================================================================

    def to_knowledge_item(self, rating: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert a rating dict to a KnowledgeItem.

        Args:
            rating: The rating dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Confidence based on games played (more games = more confident rating)
        games = rating.get("games_played", 0)
        if games >= 50:
            confidence = ConfidenceLevel.VERIFIED
        elif games >= 20:
            confidence = ConfidenceLevel.HIGH
        elif games >= 10:
            confidence = ConfidenceLevel.MEDIUM
        elif games >= 3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = rating.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        elo = rating.get("elo", 1000)
        # Normalize ELO to 0-1 importance (1000 = 0.5, 1500 = 0.75, 2000 = 1.0)
        importance = min(1.0, max(0.0, (elo - 500) / 1500))

        content = (
            f"Agent {rating.get('agent_name', 'unknown')}: "
            f"ELO {elo:.0f}, "
            f"{rating.get('wins', 0)}W/{rating.get('losses', 0)}L/{rating.get('draws', 0)}D"
        )

        return KnowledgeItem(
            id=rating["id"],
            content=content,
            source=KnowledgeSource.ELO,
            source_id=rating.get("agent_name", rating["id"]),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "agent_name": rating.get("agent_name", ""),
                "elo": elo,
                "win_rate": rating.get("win_rate", 0.0),
                "games_played": games,
                "domain_elos": rating.get("domain_elos", {}),
                "calibration_accuracy": rating.get("calibration_accuracy", 0.0),
                "reason": rating.get("reason", ""),
            },
            importance=importance,
        )

    # =========================================================================
    # Statistics (combined from both)
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get unified statistics about stored performance data."""
        self.__init_reverse_flow_state()
        agents_per_domain = {d: len(a) for d, a in self._domain_agents.items()}
        return {
            # ELO stats
            "total_ratings": len(self._ratings),
            "total_matches": len(self._matches),
            "total_calibrations": len(self._calibrations),
            "total_relationships": len(self._relationships),
            "agents_tracked": len(self._agent_ratings),
            "domains_tracked": len(self._domain_ratings),
            # Expertise stats
            "total_expertise_records": len(self._expertise),
            "total_agents_with_expertise": len(self._agent_domains),
            "total_expertise_domains": len(self._domain_agents),
            "total_history_records": sum(len(h) for h in self._agent_history.values()),
            "agents_per_domain": agents_per_domain,
            # Backward compatibility aliases (from RankingAdapter)
            "total_agents": len(self._agent_domains),
            "total_domains": len(self._domain_agents),
            # KM adjustment stats
            "km_adjustments_applied": self._km_adjustments_applied,
            "km_adjustments_pending": len(self._pending_km_adjustments),
        }

    # =========================================================================
    # Reverse Flow Methods (KM → ELO) - from EloAdapter
    # =========================================================================

    def __init_reverse_flow_state(self) -> None:
        """Initialize state for reverse flow tracking."""
        if not hasattr(self, "_km_patterns"):
            self._km_patterns: Dict[str, List[KMEloPattern]] = {}  # agent -> patterns
            self._pending_km_adjustments: List[EloAdjustmentRecommendation] = []
            self._applied_km_adjustments: List[EloAdjustmentRecommendation] = []
            self._km_adjustments_applied: int = 0

    async def analyze_km_patterns_for_agent(
        self,
        agent_name: str,
        km_items: List[Dict[str, Any]],
        min_confidence: float = 0.7,
    ) -> List[KMEloPattern]:
        """
        Analyze Knowledge Mound items to detect patterns for an agent.

        Examines KM data to identify:
        - success_contributor: Agent's claims frequently lead to successful outcomes
        - contradiction_source: Agent's claims frequently contradicted
        - domain_expert: Agent shows consistent expertise in a domain
        - crux_resolver: Agent identifies and resolves debate cruxes

        Args:
            agent_name: The agent to analyze
            km_items: KM items mentioning this agent
            min_confidence: Minimum confidence threshold for patterns

        Returns:
            List of detected patterns
        """
        self.__init_reverse_flow_state()

        patterns: List[KMEloPattern] = []

        # Counters for pattern detection
        success_count = 0
        contradiction_count = 0
        domain_mentions: Dict[str, int] = {}
        crux_resolutions = 0
        total_items = len(km_items)

        if total_items == 0:
            return patterns

        debate_ids: List[str] = []

        for item in km_items:
            metadata = item.get("metadata", {})
            debate_id = metadata.get("debate_id")
            if debate_id:
                debate_ids.append(debate_id)

            # Check for success contribution
            if metadata.get("outcome_success") or metadata.get("claim_validated"):
                success_count += 1

            # Check for contradictions
            if metadata.get("was_contradicted") or metadata.get("claim_invalidated"):
                contradiction_count += 1

            # Track domain mentions
            domain = metadata.get("domain") or item.get("domain")
            if domain:
                domain_mentions[domain] = domain_mentions.get(domain, 0) + 1

            # Check for crux resolution
            if metadata.get("crux_resolved") or metadata.get("key_insight"):
                crux_resolutions += 1

        # Detect success_contributor pattern
        success_rate = success_count / total_items
        if success_rate >= 0.6 and success_count >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="success_contributor",
                    confidence=min(0.95, success_rate + 0.1),
                    observation_count=success_count,
                    debate_ids=debate_ids[:10],  # Cap at 10
                    metadata={"success_rate": success_rate, "total_items": total_items},
                )
            )

        # Detect contradiction_source pattern (negative)
        contradiction_rate = contradiction_count / total_items
        if contradiction_rate >= 0.3 and contradiction_count >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="contradiction_source",
                    confidence=min(0.95, contradiction_rate + 0.2),
                    observation_count=contradiction_count,
                    debate_ids=debate_ids[:10],
                    metadata={"contradiction_rate": contradiction_rate},
                )
            )

        # Detect domain_expert patterns
        for domain, count in domain_mentions.items():
            if count >= 5:  # Need sufficient domain presence
                patterns.append(
                    KMEloPattern(
                        agent_name=agent_name,
                        pattern_type="domain_expert",
                        confidence=min(0.9, count / 20 + 0.5),
                        observation_count=count,
                        domain=domain,
                        debate_ids=debate_ids[:5],
                        metadata={"domain_item_count": count},
                    )
                )

        # Detect crux_resolver pattern
        if crux_resolutions >= 3:
            patterns.append(
                KMEloPattern(
                    agent_name=agent_name,
                    pattern_type="crux_resolver",
                    confidence=min(0.9, crux_resolutions / 10 + 0.5),
                    observation_count=crux_resolutions,
                    debate_ids=debate_ids[:10],
                    metadata={"crux_resolutions": crux_resolutions},
                )
            )

        # Filter by confidence threshold
        patterns = [p for p in patterns if p.confidence >= min_confidence]

        # Store patterns
        self._km_patterns[agent_name] = patterns

        logger.info(
            f"Analyzed KM patterns for {agent_name}: "
            f"found {len(patterns)} patterns from {total_items} items"
        )

        return patterns

    def compute_elo_adjustment(
        self,
        patterns: List[KMEloPattern],
        max_adjustment: float = 50.0,
    ) -> Optional[EloAdjustmentRecommendation]:
        """
        Compute ELO adjustment recommendation from KM patterns.

        Args:
            patterns: List of KM patterns for an agent
            max_adjustment: Maximum absolute ELO change allowed

        Returns:
            EloAdjustmentRecommendation or None if no adjustment warranted
        """
        self.__init_reverse_flow_state()

        if not patterns:
            return None

        agent_name = patterns[0].agent_name
        total_adjustment = 0.0
        reasons: List[str] = []
        overall_confidence = 0.0
        domain_adjustments: Dict[str, float] = {}

        for pattern in patterns:
            confidence_weight = pattern.confidence * (1 + min(0.5, pattern.observation_count / 20))

            if pattern.pattern_type == "success_contributor":
                # Boost for contributing to successful outcomes
                adj = 15.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"+{adj:.1f} success contributor ({pattern.observation_count} obs)")

            elif pattern.pattern_type == "contradiction_source":
                # Penalty for frequently contradicted claims
                adj = -10.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"{adj:.1f} contradictions ({pattern.observation_count} obs)")

            elif pattern.pattern_type == "domain_expert":
                # Domain-specific boost
                domain = pattern.domain or "general"
                adj = 20.0 * confidence_weight
                domain_adjustments[domain] = domain_adjustments.get(domain, 0) + adj
                reasons.append(f"+{adj:.1f} domain expert: {domain}")

            elif pattern.pattern_type == "crux_resolver":
                # Boost for resolving key debate cruxes
                adj = 12.0 * confidence_weight
                total_adjustment += adj
                reasons.append(f"+{adj:.1f} crux resolver ({pattern.observation_count} obs)")

            overall_confidence = max(overall_confidence, pattern.confidence)

        # Apply domain adjustments (take highest)
        if domain_adjustments:
            best_domain = max(domain_adjustments, key=domain_adjustments.get)  # type: ignore
            total_adjustment += domain_adjustments[best_domain]

        # Clamp to max adjustment
        total_adjustment = max(-max_adjustment, min(max_adjustment, total_adjustment))

        # Skip tiny adjustments
        if abs(total_adjustment) < 2.0:
            return None

        recommendation = EloAdjustmentRecommendation(
            agent_name=agent_name,
            adjustment=total_adjustment,
            reason="; ".join(reasons),
            patterns=patterns,
            confidence=overall_confidence,
            domain=list(domain_adjustments.keys())[0] if len(domain_adjustments) == 1 else None,
        )

        self._pending_km_adjustments.append(recommendation)

        logger.info(
            f"KM ELO adjustment recommended for {agent_name}: "
            f"{total_adjustment:+.1f} (confidence={overall_confidence:.2f})"
        )

        return recommendation

    async def apply_km_elo_adjustment(
        self,
        recommendation: EloAdjustmentRecommendation,
        force: bool = False,
    ) -> bool:
        """
        Apply a KM-based ELO adjustment to the underlying ELO system.

        Args:
            recommendation: The adjustment to apply
            force: If True, apply even if confidence is below threshold

        Returns:
            True if applied, False if skipped
        """
        self.__init_reverse_flow_state()

        if not self._elo_system:
            logger.warning("Cannot apply KM adjustment: no ELO system configured")
            return False

        if recommendation.applied:
            logger.debug(f"Adjustment for {recommendation.agent_name} already applied")
            return False

        if not force and recommendation.confidence < 0.7:
            logger.debug(
                f"Skipping low-confidence adjustment for {recommendation.agent_name}: "
                f"confidence={recommendation.confidence:.2f}"
            )
            return False

        agent_name = recommendation.agent_name

        try:
            # Get current rating
            current_rating = self._elo_system.get_rating(agent_name)
            if not current_rating:
                logger.warning(f"Agent {agent_name} not found in ELO system")
                return False

            # Apply adjustment
            new_elo = current_rating.elo + recommendation.adjustment

            # Use ELO system's update mechanism if available
            if hasattr(self._elo_system, "adjust_rating"):
                self._elo_system.adjust_rating(
                    agent_name,
                    adjustment=recommendation.adjustment,
                    reason=f"KM pattern: {recommendation.reason}",
                )
            else:
                # Fallback: direct rating modification
                current_rating.elo = new_elo

            # Mark as applied
            recommendation.applied = True
            self._applied_km_adjustments.append(recommendation)
            self._km_adjustments_applied += 1

            # Remove from pending
            if recommendation in self._pending_km_adjustments:
                self._pending_km_adjustments.remove(recommendation)

            logger.info(
                f"Applied KM ELO adjustment for {agent_name}: "
                f"{current_rating.elo - recommendation.adjustment:.0f} -> {new_elo:.0f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error applying KM adjustment for {agent_name}: {e}")
            return False

    async def sync_km_to_elo(
        self,
        agent_patterns: Dict[str, List[KMEloPattern]],
        max_adjustment: float = 50.0,
        min_confidence: float = 0.7,
        auto_apply: bool = False,
    ) -> EloSyncResult:
        """
        Batch sync KM patterns to ELO adjustments.

        Args:
            agent_patterns: Dict mapping agent names to their KM patterns
            max_adjustment: Maximum ELO adjustment per agent
            min_confidence: Minimum confidence to apply adjustment
            auto_apply: If True, automatically apply adjustments

        Returns:
            EloSyncResult with sync statistics
        """
        start_time = time.time()
        result = EloSyncResult()

        for agent_name, patterns in agent_patterns.items():
            result.total_patterns += len(patterns)

            # Compute adjustment recommendation
            recommendation = self.compute_elo_adjustment(patterns, max_adjustment)

            if recommendation:
                result.adjustments_recommended += 1

                if auto_apply and recommendation.confidence >= min_confidence:
                    applied = await self.apply_km_elo_adjustment(recommendation)
                    if applied:
                        result.adjustments_applied += 1
                        result.total_elo_change += recommendation.adjustment
                        if agent_name not in result.agents_affected:
                            result.agents_affected.append(agent_name)
                    else:
                        result.adjustments_skipped += 1
                else:
                    result.adjustments_skipped += 1

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"KM → ELO sync complete: "
            f"patterns={result.total_patterns}, "
            f"recommended={result.adjustments_recommended}, "
            f"applied={result.adjustments_applied}, "
            f"total_change={result.total_elo_change:+.1f}"
        )

        return result

    def get_pending_adjustments(self) -> List[EloAdjustmentRecommendation]:
        """Get list of pending KM-based ELO adjustments."""
        self.__init_reverse_flow_state()
        return list(self._pending_km_adjustments)

    def get_applied_adjustments(
        self,
        limit: int = 50,
    ) -> List[EloAdjustmentRecommendation]:
        """Get list of applied KM-based ELO adjustments."""
        self.__init_reverse_flow_state()
        return self._applied_km_adjustments[-limit:]

    def get_agent_km_patterns(
        self,
        agent_name: str,
    ) -> List[KMEloPattern]:
        """Get stored KM patterns for an agent."""
        self.__init_reverse_flow_state()
        return self._km_patterns.get(agent_name, [])

    def clear_pending_adjustments(self) -> int:
        """Clear all pending adjustments. Returns count cleared."""
        self.__init_reverse_flow_state()
        count = len(self._pending_km_adjustments)
        self._pending_km_adjustments = []
        return count

    def get_reverse_flow_stats(self) -> Dict[str, Any]:
        """Get statistics about KM → ELO reverse flow."""
        self.__init_reverse_flow_state()

        total_patterns = sum(len(p) for p in self._km_patterns.values())
        avg_confidence = 0.0
        if total_patterns > 0:
            all_confidences = [
                p.confidence for patterns in self._km_patterns.values() for p in patterns
            ]
            avg_confidence = sum(all_confidences) / len(all_confidences)

        return {
            "agents_with_patterns": len(self._km_patterns),
            "total_patterns": total_patterns,
            "pending_adjustments": len(self._pending_km_adjustments),
            "applied_adjustments": self._km_adjustments_applied,
            "avg_pattern_confidence": round(avg_confidence, 3),
            "pattern_types": self._count_pattern_types(),
        }

    def _count_pattern_types(self) -> Dict[str, int]:
        """Count patterns by type."""
        self.__init_reverse_flow_state()
        counts: Dict[str, int] = {}
        for patterns in self._km_patterns.values():
            for p in patterns:
                counts[p.pattern_type] = counts.get(p.pattern_type, 0) + 1
        return counts

    # =========================================================================
    # KM Persistence Methods (from RankingAdapter)
    # =========================================================================

    async def sync_to_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Persist all expertise data to the Knowledge Mound.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID for storage

        Returns:
            Dict with sync statistics
        """
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        result: Dict[str, Any] = {
            "expertise_synced": 0,
            "history_synced": 0,
            "errors": [],
        }

        # Sync expertise records
        for expertise_key, expertise_data in self._expertise.items():
            try:
                content = (
                    f"Agent: {expertise_data['agent_name']}\n"
                    f"Domain: {expertise_data['domain']}\n"
                    f"ELO: {expertise_data['elo']}\n"
                    f"Confidence: {expertise_data['confidence']:.2f}\n"
                    f"Debates: {expertise_data['debate_count']}"
                )

                request = IngestionRequest(
                    content=content,
                    source_type=SourceType.RANKING,
                    workspace_id=workspace_id,
                    confidence=expertise_data["confidence"],
                    tier="slow",  # Slow tier for expertise data
                    metadata={
                        "type": "agent_expertise",
                        "expertise_id": expertise_data["id"],
                        "agent_name": expertise_data["agent_name"],
                        "domain": expertise_data["domain"],
                        "elo": expertise_data["elo"],
                        "debate_count": expertise_data["debate_count"],
                    },
                )

                await mound.ingest(request)
                result["expertise_synced"] += 1

            except Exception as e:
                result["errors"].append(f"Expertise {expertise_key}: {e}")

        logger.info(
            f"Performance sync to KM: expertise={result['expertise_synced']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    async def load_from_mound(
        self,
        mound: Any,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Load expertise data from the Knowledge Mound.

        This restores adapter state from KM persistence.

        Args:
            mound: KnowledgeMound instance
            workspace_id: Workspace ID to load from

        Returns:
            Dict with load statistics
        """
        result: Dict[str, Any] = {
            "expertise_loaded": 0,
            "errors": [],
        }

        try:
            # Query KM for expertise records
            nodes = await mound.query_nodes(
                workspace_id=workspace_id,
                source_type="ranking",
                limit=1000,
            )

            for node in nodes:
                metadata = node.metadata or {}
                if metadata.get("type") != "agent_expertise":
                    continue

                agent_name = metadata.get("agent_name")
                domain = metadata.get("domain")
                if not agent_name or not domain:
                    continue

                expertise_key = f"{agent_name}:{domain}"
                expertise_id = f"{self.EXPERTISE_PREFIX}{expertise_key.replace(':', '_')}"

                self._expertise[expertise_key] = {
                    "id": expertise_id,
                    "agent_name": agent_name,
                    "domain": domain,
                    "elo": metadata.get("elo", 1500),
                    "confidence": (
                        metadata.get("confidence", node.confidence)
                        if hasattr(node, "confidence")
                        else 0.5
                    ),
                    "debate_count": metadata.get("debate_count", 0),
                    "created_at": (
                        node.created_at.isoformat()
                        if node.created_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "updated_at": (
                        node.updated_at.isoformat()
                        if node.updated_at
                        else datetime.now(timezone.utc).isoformat()
                    ),
                }

                # Update indices
                if domain not in self._domain_agents:
                    self._domain_agents[domain] = []
                if agent_name not in self._domain_agents[domain]:
                    self._domain_agents[domain].append(agent_name)

                if agent_name not in self._agent_domains:
                    self._agent_domains[agent_name] = []
                if domain not in self._agent_domains[agent_name]:
                    self._agent_domains[agent_name].append(domain)

                result["expertise_loaded"] += 1

        except Exception as e:
            result["errors"].append(f"Load failed: {e}")
            logger.error(f"Failed to load expertise from KM: {e}")

        logger.info(
            f"Performance load from KM: loaded={result['expertise_loaded']}, "
            f"errors={len(result['errors'])}"
        )
        return result


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# These allow existing code to continue using EloAdapter and RankingAdapter names
EloAdapter = PerformanceAdapter
RankingAdapter = PerformanceAdapter


__all__ = [
    # Main adapter
    "PerformanceAdapter",
    # Backwards compatibility
    "EloAdapter",
    "RankingAdapter",
    # Dataclasses from EloAdapter
    "KMEloPattern",
    "EloAdjustmentRecommendation",
    "EloSyncResult",
    "RatingSearchResult",
    # Dataclasses from RankingAdapter
    "AgentExpertise",
    "ExpertiseSearchResult",
]

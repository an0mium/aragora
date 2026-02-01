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
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin

from aragora.knowledge.mound.adapters.performance.elo_storage import EloStorageMixin
from aragora.knowledge.mound.adapters.performance.expertise import ExpertiseMixin
from aragora.knowledge.mound.adapters.performance.km_persistence import KMPersistenceMixin
from aragora.knowledge.mound.adapters.performance.reverse_flow import ReverseFlowMixin
from aragora.knowledge.mound.adapters.performance.search import (
    FusionImplementationMixin,
    SearchMixin,
    SemanticSearchImplementationMixin,
)

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem

EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


class PerformanceAdapter(
    EloStorageMixin,
    ExpertiseMixin,
    ReverseFlowMixin,
    KMPersistenceMixin,
    FusionImplementationMixin,
    SemanticSearchImplementationMixin,
    SearchMixin,
    FusionMixin,
    SemanticSearchMixin,
    KnowledgeMoundAdapter,
):
    """
    Unified adapter for agent performance metrics (ELO + domain expertise).

    Combines functionality from EloAdapter and RankingAdapter to provide:
    - Complete agent performance lifecycle (ratings, matches, calibration)
    - Domain-based expertise tracking with caching
    - Bidirectional KM integration (persistence + reverse flow)
    - Pattern detection and ELO adjustments from KM knowledge
    - Event callbacks for real-time updates
    - Semantic vector search (via SemanticSearchMixin)

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

    # FusionMixin configuration
    adapter_name = "performance"
    source_type = "elo"

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
        event_callback: EventCallback | None = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        enable_resilience: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            elo_system: Optional EloSystem instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
            cache_ttl_seconds: TTL for cached queries (default: 60 seconds)
            enable_resilience: If True, enables circuit breaker and bulkhead protection
        """
        # Initialize base adapter (handles dual_write, event_callback, resilience, metrics, tracing)
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )

        self._elo_system = elo_system
        self._cache_ttl_seconds = cache_ttl_seconds

        # ELO storage (from EloAdapter)
        self._ratings: dict[str, dict[str, Any]] = {}
        self._matches: dict[str, dict[str, Any]] = {}
        self._calibrations: dict[str, dict[str, Any]] = {}
        self._relationships: dict[str, dict[str, Any]] = {}

        # ELO indices
        self._agent_ratings: dict[str, list[str]] = {}  # agent -> [rating_ids]
        self._agent_matches: dict[str, list[str]] = {}  # agent -> [match_ids]
        self._domain_ratings: dict[str, list[str]] = {}  # domain -> [rating_ids]

        # Expertise storage (from RankingAdapter)
        self._expertise: dict[str, dict[str, Any]] = {}  # {agent_domain: expertise_data}
        self._agent_history: dict[str, list[dict[str, Any]]] = {}  # {agent: [records]}

        # Expertise indices
        self._domain_agents: dict[str, list[str]] = {}  # {domain: [agent_names]}
        self._agent_domains: dict[str, list[str]] = {}  # {agent_name: [domains]}

        # Cache (from RankingAdapter)
        self._domain_experts_cache: dict[str, tuple] = {}  # {cache_key: (timestamp, results)}
        self._cache_hits = 0
        self._cache_misses = 0

    # set_event_callback, _emit_event inherited from KnowledgeMoundAdapter

    # =========================================================================
    # ELO System Property
    # =========================================================================

    @property
    def elo_system(self) -> Optional["EloSystem"]:
        """Access the underlying EloSystem."""
        return self._elo_system

    def set_elo_system(self, elo_system: "EloSystem") -> None:
        """Set the ELO system to use."""
        self._elo_system = elo_system

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get unified statistics about stored performance data."""
        self._ReverseFlowMixin__init_reverse_flow_state()  # name-mangled from ReverseFlowMixin
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


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# These allow existing code to continue using EloAdapter and RankingAdapter names
EloAdapter = PerformanceAdapter
RankingAdapter = PerformanceAdapter

__all__ = [
    "PerformanceAdapter",
    "EloAdapter",
    "RankingAdapter",
    "EventCallback",
]

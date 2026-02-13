"""
Data models for the PerformanceAdapter.

Contains all dataclasses used across the performance adapter subsystem:
- KMEloPattern: Patterns detected in Knowledge Mound influencing ELO
- EloAdjustmentRecommendation: Recommendations for ELO adjustments from KM patterns
- EloSyncResult: Result of syncing KM patterns to ELO
- RatingSearchResult: Wrapper for rating search results
- AgentExpertise: Agent expertise in a domain
- ExpertiseSearchResult: Wrapper for expertise search results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    domain: str | None = None  # Domain if pattern is domain-specific
    debate_ids: list[str] = field(default_factory=list)  # Debates that formed this pattern
    metadata: dict[str, Any] = field(default_factory=dict)


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
    patterns: list[KMEloPattern] = field(default_factory=list)  # Supporting patterns
    confidence: float = 0.7  # Overall confidence in recommendation
    domain: str | None = None  # If domain-specific
    applied: bool = False


@dataclass
class EloSyncResult:
    """Result of syncing KM patterns to ELO."""

    total_patterns: int = 0
    adjustments_recommended: int = 0
    adjustments_applied: int = 0
    adjustments_skipped: int = 0
    total_elo_change: float = 0.0
    agents_affected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class RatingSearchResult:
    """Wrapper for rating search results with adapter metadata."""

    rating: dict[str, Any]
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

    expertise: dict[str, Any]
    relevance_score: float = 0.0
    matched_domains: list[str] | None = None

    def __post_init__(self) -> None:
        if self.matched_domains is None:
            self.matched_domains = []


__all__ = [
    "KMEloPattern",
    "EloAdjustmentRecommendation",
    "EloSyncResult",
    "RatingSearchResult",
    "AgentExpertise",
    "ExpertiseSearchResult",
]

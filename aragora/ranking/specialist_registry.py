"""Specialist Agent Registry — promotes high-ELO agents as domain experts.

When agents consistently outperform in specific domains (via ELO tracking),
they get promoted to the specialist registry. The debate TeamSelector then
prefers specialists for domain-relevant debates.

Flow:
    1. ELO system records domain-specific ratings
    2. After each match, check_and_promote() evaluates promotion criteria
    3. Specialists are stored with domain, ELO rating, and optional genome_id
    4. TeamSelector.score() queries specialists for domain-aware scoring

Usage:
    registry = SpecialistRegistry()
    registry.check_and_promote("claude", "security", elo_rating=1250)
    specialists = registry.get_specialists("security", limit=3)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default: agent must be 150 ELO above baseline (1000) to be a specialist
DEFAULT_PROMOTION_THRESHOLD = 150
DEFAULT_BASELINE_ELO = 1000
DEFAULT_DEMOTION_THRESHOLD = 50  # Below baseline + this = demotion
DEFAULT_MIN_MATCHES = 5  # Minimum domain matches before promotion eligible


@dataclass
class SpecialistEntry:
    """A registered domain specialist."""

    agent_name: str
    domain: str
    elo_rating: float
    promoted_at: float = field(default_factory=time.time)
    genome_id: str | None = None
    match_count: int = 0
    win_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "domain": self.domain,
            "elo_rating": self.elo_rating,
            "promoted_at": self.promoted_at,
            "genome_id": self.genome_id,
            "match_count": self.match_count,
            "win_rate": self.win_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecialistEntry:
        return cls(
            agent_name=data["agent_name"],
            domain=data["domain"],
            elo_rating=data.get("elo_rating", DEFAULT_BASELINE_ELO),
            promoted_at=data.get("promoted_at", 0.0),
            genome_id=data.get("genome_id"),
            match_count=data.get("match_count", 0),
            win_rate=data.get("win_rate", 0.0),
            metadata=data.get("metadata", {}),
        )


class SpecialistRegistry:
    """Registry of domain-specialist agents promoted from ELO performance.

    Provides:
    - check_and_promote(): Auto-promote agents exceeding domain ELO threshold
    - get_specialists(): Retrieve top specialists for a domain
    - get_agent_specialties(): List domains where an agent is a specialist
    - score_bonus(): Compute team selection bonus for domain specialists
    """

    def __init__(
        self,
        promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
        demotion_threshold: float = DEFAULT_DEMOTION_THRESHOLD,
        baseline_elo: float = DEFAULT_BASELINE_ELO,
        min_matches: int = DEFAULT_MIN_MATCHES,
    ):
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.baseline_elo = baseline_elo
        self.min_matches = min_matches
        # In-memory registry: (agent_name, domain) → SpecialistEntry
        self._registry: dict[tuple[str, str], SpecialistEntry] = {}

    def check_and_promote(
        self,
        agent_name: str,
        domain: str,
        elo_rating: float,
        match_count: int = 0,
        win_rate: float = 0.0,
        genome_id: str | None = None,
    ) -> bool:
        """Check if agent qualifies for specialist status and promote/demote.

        Returns True if the agent is now a specialist (newly promoted or existing).
        """
        key = (agent_name, domain)

        # Check if already a specialist
        if key in self._registry:
            entry = self._registry[key]
            entry.elo_rating = elo_rating
            entry.match_count = match_count
            entry.win_rate = win_rate
            # Demote if fallen below threshold
            if elo_rating < self.baseline_elo + self.demotion_threshold:
                del self._registry[key]
                logger.info(
                    "specialist_demoted agent=%s domain=%s elo=%.0f",
                    agent_name, domain, elo_rating,
                )
                return False
            return True

        # Check promotion criteria
        if match_count < self.min_matches:
            return False
        if elo_rating < self.baseline_elo + self.promotion_threshold:
            return False

        # Promote
        self._registry[key] = SpecialistEntry(
            agent_name=agent_name,
            domain=domain,
            elo_rating=elo_rating,
            genome_id=genome_id,
            match_count=match_count,
            win_rate=win_rate,
        )
        logger.info(
            "specialist_promoted agent=%s domain=%s elo=%.0f matches=%d",
            agent_name, domain, elo_rating, match_count,
        )
        return True

    def get_specialists(
        self, domain: str, limit: int = 5
    ) -> list[SpecialistEntry]:
        """Get top specialists for a domain, sorted by ELO rating."""
        entries = [
            e for e in self._registry.values()
            if e.domain == domain
        ]
        entries.sort(key=lambda e: e.elo_rating, reverse=True)
        return entries[:limit]

    def get_agent_specialties(self, agent_name: str) -> list[SpecialistEntry]:
        """Get all domains where an agent is a registered specialist."""
        return [
            e for e in self._registry.values()
            if e.agent_name == agent_name
        ]

    def is_specialist(self, agent_name: str, domain: str) -> bool:
        """Check if an agent is a specialist in a given domain."""
        return (agent_name, domain) in self._registry

    def score_bonus(
        self, agent_name: str, domain: str, weight: float = 0.25
    ) -> float:
        """Compute selection score bonus for a specialist.

        Returns a bonus in [0, weight] proportional to how far above
        the promotion threshold the agent's domain ELO is.
        """
        key = (agent_name, domain)
        entry = self._registry.get(key)
        if entry is None:
            return 0.0
        # Bonus scales from 0 (at promotion threshold) to weight (at 2x threshold)
        excess = entry.elo_rating - (self.baseline_elo + self.promotion_threshold)
        scale = min(1.0, max(0.0, excess / self.promotion_threshold))
        return weight * scale

    def all_specialists(self) -> list[SpecialistEntry]:
        """Get all registered specialists across all domains."""
        return list(self._registry.values())

    def domains(self) -> list[str]:
        """Get all domains that have at least one specialist."""
        return list({e.domain for e in self._registry.values()})

    def clear(self) -> None:
        """Clear the registry (useful for testing)."""
        self._registry.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry for persistence."""
        return {
            "specialists": [e.to_dict() for e in self._registry.values()],
            "promotion_threshold": self.promotion_threshold,
            "baseline_elo": self.baseline_elo,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecialistRegistry:
        """Deserialize a registry from persisted data."""
        registry = cls(
            promotion_threshold=data.get("promotion_threshold", DEFAULT_PROMOTION_THRESHOLD),
            baseline_elo=data.get("baseline_elo", DEFAULT_BASELINE_ELO),
        )
        for entry_data in data.get("specialists", []):
            entry = SpecialistEntry.from_dict(entry_data)
            registry._registry[(entry.agent_name, entry.domain)] = entry
        return registry


# Module-level singleton
_specialist_registry: SpecialistRegistry | None = None


def get_specialist_registry() -> SpecialistRegistry:
    """Get the global specialist registry instance."""
    global _specialist_registry
    if _specialist_registry is None:
        _specialist_registry = SpecialistRegistry()
    return _specialist_registry


def reset_specialist_registry() -> None:
    """Reset the global specialist registry (for testing)."""
    global _specialist_registry
    _specialist_registry = None

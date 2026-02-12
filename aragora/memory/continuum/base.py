"""
Continuum Memory System - Base Types and Configuration.

This module contains the foundational types, constants, and data structures
used throughout the Continuum Memory System.

Key exports:
- ContinuumMemoryEntry: The core data structure for memory entries
- ContinuumHyperparams: Type definition for hyperparameters
- MaxEntriesPerTier: Type definition for tier limits
- AwaitableList: List wrapper for async compatibility
- Constants: CONTINUUM_SCHEMA_VERSION, DEFAULT_RETENTION_MULTIPLIER, TIER_CONFIGS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Generator, TypedDict

from aragora.memory.tier_manager import DEFAULT_TIER_CONFIGS, MemoryTier, TierConfig
from aragora.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# Schema version for ContinuumMemory
CONTINUUM_SCHEMA_VERSION = 3

# Default retention multiplier (entries older than multiplier * half_life are eligible for cleanup)
DEFAULT_RETENTION_MULTIPLIER = 2.0

# Re-export for backwards compatibility (use DEFAULT_TIER_CONFIGS from tier_manager)
TIER_CONFIGS = DEFAULT_TIER_CONFIGS

# Cache for KM similarity queries (5 min TTL, 1000 entries)
_km_similarity_cache: TTLCache[list] = TTLCache(maxsize=1000, ttl_seconds=300)


class MaxEntriesPerTier(TypedDict):
    """Type definition for max entries per tier configuration."""

    fast: int
    medium: int
    slow: int
    glacial: int


class ContinuumHyperparams(TypedDict):
    """
    Type definition for ContinuumMemory hyperparameters.

    These parameters control memory consolidation, tier transitions,
    and retention policies. They can be modified by MetaLearner.
    """

    surprise_weight_success: float  # Weight for success rate surprise
    surprise_weight_semantic: float  # Weight for semantic novelty
    surprise_weight_temporal: float  # Weight for timing surprise
    surprise_weight_agent: float  # Weight for agent prediction error
    consolidation_threshold: float  # Updates to reach full consolidation
    promotion_cooldown_hours: float  # Minimum time between promotions
    max_entries_per_tier: MaxEntriesPerTier  # Max entries per tier
    retention_multiplier: float  # multiplier * half_life for cleanup


def get_default_hyperparams() -> ContinuumHyperparams:
    """Get default hyperparameters for ContinuumMemory."""
    # TypedDict constructor returns dict; mypy cannot verify nested TypedDict compatibility
    return ContinuumHyperparams(  # type: ignore[return-value]
        surprise_weight_success=0.3,  # Weight for success rate surprise
        surprise_weight_semantic=0.3,  # Weight for semantic novelty
        surprise_weight_temporal=0.2,  # Weight for timing surprise
        surprise_weight_agent=0.2,  # Weight for agent prediction error
        consolidation_threshold=100.0,  # Updates to reach full consolidation
        promotion_cooldown_hours=24.0,  # Minimum time between promotions
        # Retention policy settings
        max_entries_per_tier=MaxEntriesPerTier(
            fast=1000,
            medium=5000,
            slow=10000,
            glacial=50000,
        ),
        retention_multiplier=DEFAULT_RETENTION_MULTIPLIER,  # multiplier * half_life for cleanup
    )


@dataclass
class ContinuumMemoryEntry:
    """A single entry in the continuum memory system."""

    id: str
    tier: MemoryTier
    content: str
    importance: float
    surprise_score: float
    consolidation_score: float  # 0-1, how consolidated/stable the memory is
    update_count: int
    success_count: int
    failure_count: int
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    red_line: bool = False  # If True, entry cannot be deleted/forgotten
    red_line_reason: str = ""  # Why this entry is protected

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def stability_score(self) -> float:
        """Inverse of surprise - how predictable this pattern is."""
        return 1.0 - self.surprise_score

    def should_promote(self) -> bool:
        """Check if this entry should be promoted to a faster tier."""
        if self.tier == MemoryTier.FAST:
            return False  # Already at fastest
        config = TIER_CONFIGS[self.tier]
        return self.surprise_score > config.promotion_threshold

    def should_demote(self) -> bool:
        """Check if this entry should be demoted to a slower tier."""
        if self.tier == MemoryTier.GLACIAL:
            return False  # Already at slowest
        config = TIER_CONFIGS[self.tier]
        return self.stability_score > config.demotion_threshold and self.update_count > 10

    # Cross-reference support for Knowledge Mound integration

    @property
    def cross_references(self) -> list[str]:
        """Get list of cross-reference IDs linked to this entry."""
        return self.metadata.get("cross_references", [])

    @cross_references.setter
    def cross_references(self, refs: list[str]) -> None:
        """Set cross-reference IDs for this entry."""
        self.metadata["cross_references"] = refs

    def add_cross_reference(self, ref_id: str) -> None:
        """Add a cross-reference to another knowledge item."""
        refs = self.cross_references
        if ref_id not in refs:
            refs.append(ref_id)
            self.cross_references = refs

    def remove_cross_reference(self, ref_id: str) -> None:
        """Remove a cross-reference from this entry.

        Uses try/except (EAFP) instead of check-then-remove to avoid
        O(n) in check followed by O(n) remove. Now just O(n) total.
        """
        refs = self.cross_references
        try:
            refs.remove(ref_id)
            self.cross_references = refs
        except ValueError:
            pass  # ref_id was not in refs, which is fine

    @property
    def knowledge_mound_id(self) -> str:
        """Get the Knowledge Mound ID for this entry."""
        return f"cm_{self.id}"

    @property
    def tags(self) -> list[str]:
        """Get tags associated with this entry."""
        return self.metadata.get("tags", [])

    @tags.setter
    def tags(self, tag_list: list[str]) -> None:
        """Set tags for this entry."""
        self.metadata["tags"] = tag_list

    @property
    def last_updated(self) -> str:
        """Alias for updated_at for Knowledge Mound compatibility."""
        return self.updated_at


class AwaitableList(list[ContinuumMemoryEntry]):
    """List wrapper that can be awaited for async compatibility."""

    def __await__(self) -> Generator[Any, None, AwaitableList]:
        async def _wrap() -> AwaitableList:
            return self

        return _wrap().__await__()


# Re-export TierConfig for backwards compatibility
__all__ = [
    "CONTINUUM_SCHEMA_VERSION",
    "DEFAULT_RETENTION_MULTIPLIER",
    "TIER_CONFIGS",
    "MaxEntriesPerTier",
    "ContinuumHyperparams",
    "ContinuumMemoryEntry",
    "AwaitableList",
    "TierConfig",
    "MemoryTier",
    "get_default_hyperparams",
    "_km_similarity_cache",
]

"""
Type definitions for Continuum Memory System.

Contains TypedDict definitions for hyperparameters and configuration.
"""

from __future__ import annotations

from typing import TypedDict


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


# Default retention multiplier (entries older than multiplier * half_life are eligible for cleanup)
DEFAULT_RETENTION_MULTIPLIER = 2.0


__all__ = [
    "ContinuumHyperparams",
    "MaxEntriesPerTier",
    "DEFAULT_RETENTION_MULTIPLIER",
]

"""
Continuum Memory System (CMS) for Nested Learning.

This package implements Google Research's Nested Learning paradigm with
multi-timescale memory updates. Memory is treated as a spectrum where
different modules update at different frequencies, enabling continual
learning without catastrophic forgetting.

Based on: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

Key concepts:
- Fast tier: Updates on every event (immediate patterns, 1h half-life)
- Medium tier: Updates per debate round (tactical learning, 24h half-life)
- Slow tier: Updates per nomic cycle (strategic learning, 7d half-life)
- Glacial tier: Updates monthly (foundational knowledge, 30d half-life)

Package Structure:
- core.py: Main ContinuumMemory class (with mixin composition)
- entry.py: ContinuumMemoryEntry dataclass and AwaitableList
- types.py: Type definitions, constants, schema
- singleton.py: Global singleton functions
- crud.py: CRUD operations mixin
- retrieval.py: Search and retrieval mixin
- outcome.py: Surprise scoring and outcome tracking mixin
- tier_ops.py: Tier promotion/demotion mixin
- km_integration.py: Knowledge Mound integration mixin
- fast_tier.py: Fast tier operations mixin
- medium_tier.py: Medium tier operations mixin
- slow_tier.py: Slow tier operations mixin
- glacial_tier.py: Glacial tier operations mixin
- base.py: Base types (re-exports from types.py and entry.py)
- coordinator.py: Alternative ContinuumMemory implementation

Usage:
    from aragora.memory.continuum import ContinuumMemory, MemoryTier

    cms = ContinuumMemory()
    cms.add("pattern_id", "content", tier=MemoryTier.FAST, importance=0.8)
    results = cms.retrieve(query="pattern", limit=10)

For singleton access across subsystems:
    from aragora.memory.continuum import get_continuum_memory, reset_continuum_memory

    memory = get_continuum_memory()
"""

from aragora.memory.tier_manager import MemoryTier, TierConfig

# Core exports
from .core import ContinuumMemory
from .entry import AwaitableList, ContinuumMemoryEntry
from .singleton import get_continuum_memory, reset_continuum_memory
from .types import (
    CONTINUUM_SCHEMA_VERSION,
    DEFAULT_RETENTION_MULTIPLIER,
    TIER_CONFIGS,
    ContinuumHyperparams,
    MaxEntriesPerTier,
    INITIAL_SCHEMA,
    SCHEMA_MIGRATIONS,
)

# Tier-specific mixins and constants
from .fast_tier import (
    FastTierMixin,
    FAST_TIER_HALF_LIFE_HOURS,
    FAST_TIER_TTL_MINUTES,
    FAST_TIER_DECAY_RATE,
)
from .medium_tier import (
    MediumTierMixin,
    MEDIUM_TIER_HALF_LIFE_HOURS,
    MEDIUM_TIER_TTL_HOURS,
    MEDIUM_TIER_DECAY_RATE,
)
from .slow_tier import (
    SlowTierMixin,
    SLOW_TIER_HALF_LIFE_HOURS,
    SLOW_TIER_TTL_DAYS,
    SLOW_TIER_DECAY_RATE,
)
from .glacial_tier import (
    GlacialTierMixin,
    GLACIAL_TIER_HALF_LIFE_HOURS,
    GLACIAL_TIER_HALF_LIFE_DAYS,
    GLACIAL_TIER_TTL_WEEKS,
    GLACIAL_TIER_DECAY_RATE,
)

__all__ = [
    # Main class
    "ContinuumMemory",
    # Entry dataclass
    "ContinuumMemoryEntry",
    "AwaitableList",
    # Enums and types
    "MemoryTier",
    "TierConfig",
    # Singleton functions
    "get_continuum_memory",
    "reset_continuum_memory",
    # Type definitions
    "ContinuumHyperparams",
    "MaxEntriesPerTier",
    # Constants
    "CONTINUUM_SCHEMA_VERSION",
    "DEFAULT_RETENTION_MULTIPLIER",
    "TIER_CONFIGS",
    "INITIAL_SCHEMA",
    "SCHEMA_MIGRATIONS",
    # Fast tier
    "FastTierMixin",
    "FAST_TIER_HALF_LIFE_HOURS",
    "FAST_TIER_TTL_MINUTES",
    "FAST_TIER_DECAY_RATE",
    # Medium tier
    "MediumTierMixin",
    "MEDIUM_TIER_HALF_LIFE_HOURS",
    "MEDIUM_TIER_TTL_HOURS",
    "MEDIUM_TIER_DECAY_RATE",
    # Slow tier
    "SlowTierMixin",
    "SLOW_TIER_HALF_LIFE_HOURS",
    "SLOW_TIER_TTL_DAYS",
    "SLOW_TIER_DECAY_RATE",
    # Glacial tier
    "GlacialTierMixin",
    "GLACIAL_TIER_HALF_LIFE_HOURS",
    "GLACIAL_TIER_HALF_LIFE_DAYS",
    "GLACIAL_TIER_TTL_WEEKS",
    "GLACIAL_TIER_DECAY_RATE",
]

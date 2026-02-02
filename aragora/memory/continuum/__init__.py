"""
Continuum Memory System (CMS) for Nested Learning.

This package implements Google Research's Nested Learning paradigm with
multi-timescale memory updates. Memory is treated as a spectrum where
different modules update at different frequencies, enabling continual
learning without catastrophic forgetting.

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

from .core import ContinuumMemory
from .entry import AwaitableList, ContinuumMemoryEntry
from .singleton import get_continuum_memory, reset_continuum_memory
from .types import (
    CONTINUUM_SCHEMA_VERSION,
    DEFAULT_RETENTION_MULTIPLIER,
    TIER_CONFIGS,
    ContinuumHyperparams,
    MaxEntriesPerTier,
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
]

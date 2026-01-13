"""
Aragora Learning Module.

Implements continual learning with Nested Learning paradigm:
- ContinuumMemory: Multi-timescale memory system
- MetaLearner: Self-tuning hyperparameter optimization
"""

from aragora.learning.meta import MetaLearner
from aragora.memory.continuum import (
    ContinuumMemory,
    ContinuumMemoryEntry,
    MemoryTier,
    TIER_CONFIGS,
)
from aragora.memory.tier_manager import TierConfig

__all__ = [
    "ContinuumMemory",
    "ContinuumMemoryEntry",
    "MemoryTier",
    "TierConfig",
    "TIER_CONFIGS",
    "MetaLearner",
]

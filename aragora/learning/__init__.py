"""
Aragora Learning Module.

Implements continual learning with Nested Learning paradigm:
- ContinuumMemory: Multi-timescale memory system
- MetaLearner: Self-tuning hyperparameter optimization
"""

from aragora.learning.meta import MetaLearner
from aragora.memory.continuum import (
    TIER_CONFIGS,
    ContinuumMemory,
    ContinuumMemoryEntry,
    MemoryTier,
    TierConfig,
)

__all__ = [
    "ContinuumMemory",
    "ContinuumMemoryEntry",
    "MemoryTier",
    "TierConfig",
    "TIER_CONFIGS",
    "MetaLearner",
]

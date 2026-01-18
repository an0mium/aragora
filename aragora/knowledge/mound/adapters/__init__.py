"""
Knowledge Mound Adapters - Connect existing memory systems to the unified Knowledge Mound.

This module provides adapter classes that bridge Aragora's existing memory systems
(ContinuumMemory, ConsensusMemory, CritiqueStore) to the Knowledge Mound's unified API.

The adapter pattern enables:
- Gradual migration with dual-write period
- Unified queries across all memory systems
- Consistent provenance and metadata tracking
- Backward compatibility with existing code

Usage:
    from aragora.knowledge.mound.adapters import (
        ContinuumAdapter,
        ConsensusAdapter,
        CritiqueAdapter,
    )
    from aragora.memory.continuum import ContinuumMemory
    from aragora.knowledge.mound import KnowledgeMound

    # Create adapters
    continuum = ContinuumMemory()
    adapter = ContinuumAdapter(continuum)

    # Connect to mound
    mound = KnowledgeMound()
    await mound.initialize()
    await mound.sync_from_continuum(continuum)
"""

from .continuum_adapter import ContinuumAdapter, ContinuumSearchResult
from .consensus_adapter import ConsensusAdapter, ConsensusSearchResult
from .critique_adapter import CritiqueAdapter, CritiqueSearchResult

__all__ = [
    "ContinuumAdapter",
    "ContinuumSearchResult",
    "ConsensusAdapter",
    "ConsensusSearchResult",
    "CritiqueAdapter",
    "CritiqueSearchResult",
]

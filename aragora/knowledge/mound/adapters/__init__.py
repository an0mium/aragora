"""
Knowledge Mound Adapters - Connect existing memory systems to the unified Knowledge Mound.

This module provides adapter classes that bridge Aragora's existing memory systems
and subsystems to the Knowledge Mound's unified API.

Core adapters (memory systems):
- ContinuumAdapter: Multi-tier memory (fast/medium/slow/glacial)
- ConsensusAdapter: Debate outcomes and agreements
- CritiqueAdapter: Critique patterns and feedback

Bidirectional adapters (subsystem integration):
- EvidenceAdapter: Evidence snippets with quality scores
- BeliefAdapter: Belief network nodes and cruxes
- InsightsAdapter: Debate insights and Trickster flips
- EloAdapter: Agent rankings and calibration
- PulseAdapter: Trending topics and scheduled debates
- CostAdapter: Budget alerts and cost patterns (opt-in)

The adapter pattern enables:
- Gradual migration with dual-write period
- Unified queries across all memory systems
- Consistent provenance and metadata tracking
- Backward compatibility with existing code
- Bidirectional data flow (IN/OUT/reverse)

Usage:
    from aragora.knowledge.mound.adapters import (
        ContinuumAdapter,
        ConsensusAdapter,
        CritiqueAdapter,
        EvidenceAdapter,
        BeliefAdapter,
        InsightsAdapter,
        EloAdapter,
        PulseAdapter,
        CostAdapter,
    )
"""

# Core memory adapters
from .continuum_adapter import ContinuumAdapter, ContinuumSearchResult
from .consensus_adapter import ConsensusAdapter, ConsensusSearchResult
from .critique_adapter import CritiqueAdapter, CritiqueSearchResult

# Bidirectional integration adapters
from .evidence_adapter import (
    EvidenceAdapter,
    EvidenceSearchResult,
    EvidenceAdapterError,
    EvidenceStoreUnavailableError,
    EvidenceNotFoundError,
)
from .belief_adapter import BeliefAdapter, BeliefSearchResult, CruxSearchResult
from .insights_adapter import InsightsAdapter, InsightSearchResult, FlipSearchResult
from .elo_adapter import EloAdapter, RatingSearchResult
from .pulse_adapter import PulseAdapter, TopicSearchResult
from .cost_adapter import CostAdapter, CostAnomaly, AlertSearchResult
from .ranking_adapter import RankingAdapter, AgentExpertise, ExpertiseSearchResult
from .rlm_adapter import RlmAdapter, CompressionPattern, ContentPriority
from .culture_adapter import CultureAdapter, StoredCulturePattern, CultureSearchResult
from .control_plane_adapter import (
    ControlPlaneAdapter,
    TaskOutcome,
    AgentCapabilityRecord,
    CrossWorkspaceInsight,
)

# Factory for auto-creating adapters from Arena subsystems
from .factory import AdapterFactory, AdapterSpec, CreatedAdapter, ADAPTER_SPECS

__all__ = [
    # Core memory adapters
    "ContinuumAdapter",
    "ContinuumSearchResult",
    "ConsensusAdapter",
    "ConsensusSearchResult",
    "CritiqueAdapter",
    "CritiqueSearchResult",
    # Bidirectional integration adapters
    "EvidenceAdapter",
    "EvidenceSearchResult",
    "EvidenceAdapterError",
    "EvidenceStoreUnavailableError",
    "EvidenceNotFoundError",
    "BeliefAdapter",
    "BeliefSearchResult",
    "CruxSearchResult",
    "InsightsAdapter",
    "InsightSearchResult",
    "FlipSearchResult",
    "EloAdapter",
    "RatingSearchResult",
    "PulseAdapter",
    "TopicSearchResult",
    "CostAdapter",
    "CostAnomaly",
    "AlertSearchResult",
    "RankingAdapter",
    "AgentExpertise",
    "ExpertiseSearchResult",
    "RlmAdapter",
    "CompressionPattern",
    "ContentPriority",
    "CultureAdapter",
    "StoredCulturePattern",
    "CultureSearchResult",
    # Control Plane adapter
    "ControlPlaneAdapter",
    "TaskOutcome",
    "AgentCapabilityRecord",
    "CrossWorkspaceInsight",
    # Factory for auto-creating adapters
    "AdapterFactory",
    "AdapterSpec",
    "CreatedAdapter",
    "ADAPTER_SPECS",
]

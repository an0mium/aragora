"""
Reasoning primitives for structured debates.

Provides typed claims, evidence tracking, logical inference,
and cryptographic provenance for evidence.
"""

from aragora.reasoning.belief import (
    BeliefDistribution,
    BeliefNetwork,
    BeliefNode,
    BeliefStatus,
    Factor,
    PropagationResult,
)
from aragora.reasoning.crux_detector import BeliefPropagationAnalyzer

# Scholarly Citation Grounding (Heavy3-inspired evidence-backed verdicts)
from aragora.reasoning.citations import (
    CitationExtractor,
    CitationQuality,
    CitationStore,
    CitationType,
    CitedClaim,
    GroundedVerdict,
    ScholarlyEvidence,
    create_citation_from_url,
)

# ClaimCheck - lazy loaded to avoid circular imports with evidence.collector
# Use: from aragora.reasoning.claim_check import ClaimCheck, ClaimCheckConfig
from aragora.reasoning.claims import (
    ArgumentChain,
    ClaimRelation,
    ClaimsKernel,
    ClaimType,
    EvidenceType,
    RelationType,
    SourceReference,
    TypedClaim,
    TypedEvidence,
)
from aragora.reasoning.provenance import (
    Citation,
    CitationGraph,
    MerkleTree,
    ProvenanceChain,
    ProvenanceManager,
    ProvenanceRecord,
    ProvenanceVerifier,
    SourceType,
    TransformationType,
)
from aragora.reasoning.provenance_enhanced import (
    EnhancedProvenanceManager,
    GitProvenanceTracker,
    GitSourceInfo,
    ProvenanceValidator,
    RevalidationTrigger,
    StalenessCheck,
    StalenessStatus,
    WebProvenanceTracker,
    WebSourceInfo,
)

# Power Sampling for inference-time reasoning
from aragora.reasoning.sampling import (
    PowerSampler,
    PowerSamplingConfig,
    SamplingResult,
    sample_with_power_law,
)

from aragora.reasoning.reliability import (
    ClaimReliability,
    EvidenceReliability,
    ReliabilityLevel,
    ReliabilityScorer,
    compute_claim_reliability,
)

# Position Tracking (stance evolution across debate rounds)
from aragora.reasoning.position_tracker import (
    PositionTracker,
    PositionEvolution,
    PositionRecord,
    PositionPivot,
    PositionStance,
    get_position_tracker,
)

# Evidence-Provenance Bridge - lazy loaded to avoid circular imports
# Use: from aragora.reasoning.evidence_bridge import EvidenceProvenanceBridge

__all__ = [
    # Claims
    "ClaimsKernel",
    "TypedClaim",
    "TypedEvidence",
    "ClaimType",
    "RelationType",
    "EvidenceType",
    "ClaimRelation",
    "ArgumentChain",
    "SourceReference",
    # Provenance
    "ProvenanceManager",
    "ProvenanceChain",
    "ProvenanceRecord",
    "ProvenanceVerifier",
    "CitationGraph",
    "Citation",
    "MerkleTree",
    "SourceType",
    "TransformationType",
    # Belief Propagation
    "BeliefNetwork",
    "BeliefNode",
    "BeliefDistribution",
    "BeliefStatus",
    "Factor",
    "PropagationResult",
    "BeliefPropagationAnalyzer",
    # Enhanced Provenance
    "EnhancedProvenanceManager",
    "GitProvenanceTracker",
    "WebProvenanceTracker",
    "GitSourceInfo",
    "WebSourceInfo",
    "StalenessCheck",
    "StalenessStatus",
    "RevalidationTrigger",
    "ProvenanceValidator",
    # Scholarly Citations
    "ScholarlyEvidence",
    "CitationType",
    "CitationQuality",
    "CitedClaim",
    "GroundedVerdict",
    "CitationExtractor",
    "CitationStore",
    "create_citation_from_url",
    "ClaimCheck",
    "ClaimCheckConfig",
    # Reliability Scoring
    "ReliabilityScorer",
    "ReliabilityLevel",
    "ClaimReliability",
    "EvidenceReliability",
    "compute_claim_reliability",
    # Power Sampling
    "PowerSampler",
    "PowerSamplingConfig",
    "SamplingResult",
    "sample_with_power_law",
    # Position Tracking
    "PositionTracker",
    "PositionEvolution",
    "PositionRecord",
    "PositionPivot",
    "PositionStance",
    "get_position_tracker",
]


def __getattr__(name: str):
    """Lazy load ClaimCheck to avoid circular imports."""
    if name in ("ClaimCheck", "ClaimCheckConfig", "EvidenceMatch"):
        from aragora.reasoning import claim_check

        return getattr(claim_check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

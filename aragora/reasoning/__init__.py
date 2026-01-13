"""
Reasoning primitives for structured debates.

Provides typed claims, evidence tracking, logical inference,
and cryptographic provenance for evidence.
"""

from aragora.reasoning.belief import (
    BeliefDistribution,
    BeliefNetwork,
    BeliefNode,
    BeliefPropagationAnalyzer,
    BeliefStatus,
    Factor,
    PropagationResult,
)

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
from aragora.reasoning.reliability import (
    ClaimReliability,
    EvidenceReliability,
    ReliabilityLevel,
    ReliabilityScorer,
    compute_claim_reliability,
)

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
    # Reliability Scoring
    "ReliabilityScorer",
    "ReliabilityLevel",
    "ClaimReliability",
    "EvidenceReliability",
    "compute_claim_reliability",
]

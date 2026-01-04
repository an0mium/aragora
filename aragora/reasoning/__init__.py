"""
Reasoning primitives for structured debates.

Provides typed claims, evidence tracking, logical inference,
and cryptographic provenance for evidence.
"""

from aragora.reasoning.claims import (
    ClaimsKernel,
    TypedClaim,
    TypedEvidence,
    ClaimType,
    RelationType,
    EvidenceType,
    ClaimRelation,
    ArgumentChain,
    SourceReference,
)
from aragora.reasoning.provenance import (
    ProvenanceManager,
    ProvenanceChain,
    ProvenanceRecord,
    ProvenanceVerifier,
    CitationGraph,
    Citation,
    MerkleTree,
    SourceType,
    TransformationType,
)
from aragora.reasoning.belief import (
    BeliefNetwork,
    BeliefNode,
    BeliefDistribution,
    BeliefStatus,
    Factor,
    PropagationResult,
    BeliefPropagationAnalyzer,
)
from aragora.reasoning.provenance_enhanced import (
    EnhancedProvenanceManager,
    GitProvenanceTracker,
    WebProvenanceTracker,
    GitSourceInfo,
    WebSourceInfo,
    StalenessCheck,
    StalenessStatus,
    RevalidationTrigger,
    ProvenanceValidator,
)

# Scholarly Citation Grounding (Heavy3-inspired evidence-backed verdicts)
from aragora.reasoning.citations import (
    ScholarlyEvidence,
    CitationType,
    CitationQuality,
    CitedClaim,
    GroundedVerdict,
    CitationExtractor,
    CitationStore,
    create_citation_from_url,
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
]

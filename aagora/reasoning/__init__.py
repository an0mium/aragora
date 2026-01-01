"""
Reasoning primitives for structured debates.

Provides typed claims, evidence tracking, and logical inference.
"""

from aagora.reasoning.claims import (
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

__all__ = [
    "ClaimsKernel",
    "TypedClaim",
    "TypedEvidence",
    "ClaimType",
    "RelationType",
    "EvidenceType",
    "ClaimRelation",
    "ArgumentChain",
    "SourceReference",
]

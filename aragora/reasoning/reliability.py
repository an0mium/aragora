"""
Reliability Scoring - Compute confidence scores for claims and evidence.

Analyzes:
- Evidence source reliability (authority, freshness, confidence)
- Citation coverage (how many sources support a claim)
- Contradiction detection (conflicting evidence)
- Verification status (formally proven vs unverified)

Outputs reliability metrics that integrate with the provenance system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from aragora.reasoning.provenance import (
    ProvenanceManager,
    ProvenanceChain,
    CitationGraph,
    SourceType,
)


class ReliabilityLevel(Enum):
    """Qualitative reliability levels."""

    VERY_HIGH = "very_high"  # >= 0.9
    HIGH = "high"  # >= 0.7
    MEDIUM = "medium"  # >= 0.5
    LOW = "low"  # >= 0.3
    VERY_LOW = "very_low"  # < 0.3
    SPECULATIVE = "speculative"  # No evidence


@dataclass
class ClaimReliability:
    """Reliability assessment for a single claim."""

    claim_id: str
    claim_text: str

    # Core scores (0-1)
    reliability_score: float = 0.0
    confidence: float = 0.0

    # Component scores
    evidence_coverage: float = 0.0  # How much evidence supports this
    source_quality: float = 0.0  # Average quality of sources
    consistency: float = 1.0  # 1.0 = no contradictions
    verification_status: float = 0.0  # Formal verification score

    # Counts
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    total_citations: int = 0

    # Status
    level: ReliabilityLevel = ReliabilityLevel.SPECULATIVE
    warnings: list[str] = field(default_factory=list)
    verified_by: list[str] = field(default_factory=list)  # Verification methods

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "reliability_score": self.reliability_score,
            "confidence": self.confidence,
            "evidence_coverage": self.evidence_coverage,
            "source_quality": self.source_quality,
            "consistency": self.consistency,
            "verification_status": self.verification_status,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "total_citations": self.total_citations,
            "level": self.level.value,
            "warnings": self.warnings,
            "verified_by": self.verified_by,
        }


@dataclass
class EvidenceReliability:
    """Reliability assessment for evidence."""

    evidence_id: str
    source_type: SourceType

    # Scores
    reliability_score: float = 0.5
    freshness: float = 0.5
    authority: float = 0.5
    confidence: float = 0.5

    # Verification
    chain_verified: bool = False
    content_verified: bool = False

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type.value,
            "reliability_score": self.reliability_score,
            "freshness": self.freshness,
            "authority": self.authority,
            "confidence": self.confidence,
            "chain_verified": self.chain_verified,
            "content_verified": self.content_verified,
        }


class ReliabilityScorer:
    """
    Computes reliability scores for claims and evidence.

    Uses the provenance chain and citation graph to assess
    how well-supported claims are.
    """

    # Weights for combining scores
    WEIGHTS = {
        "evidence_coverage": 0.25,
        "source_quality": 0.25,
        "consistency": 0.25,
        "verification": 0.25,
    }

    # Source type authority defaults
    SOURCE_AUTHORITY = {
        SourceType.AGENT_GENERATED: 0.4,
        SourceType.USER_PROVIDED: 0.6,
        SourceType.EXTERNAL_API: 0.7,
        SourceType.WEB_SEARCH: 0.5,
        SourceType.DOCUMENT: 0.7,
        SourceType.CODE_ANALYSIS: 0.8,
        SourceType.DATABASE: 0.8,
        SourceType.COMPUTATION: 0.9,
        SourceType.SYNTHESIS: 0.5,
        SourceType.UNKNOWN: 0.3,
    }

    def __init__(
        self,
        provenance: ProvenanceManager,
        verification_results: Optional[dict] = None,
    ):
        self.provenance = provenance
        self.chain = provenance.chain
        self.graph = provenance.graph
        self.verification_results = verification_results or {}

    def score_claim(self, claim_id: str, claim_text: str = "") -> ClaimReliability:
        """
        Compute reliability score for a claim.

        Analyzes all evidence citing this claim and computes
        a combined reliability score.
        """
        result = ClaimReliability(
            claim_id=claim_id,
            claim_text=claim_text,
        )

        # Get all citations for this claim
        citations = self.graph.get_claim_evidence(claim_id)
        result.total_citations = len(citations)

        if not citations:
            result.level = ReliabilityLevel.SPECULATIVE
            result.warnings.append("No evidence supports this claim")
            return result

        # Analyze each piece of evidence
        supporting_scores = []
        contradicting_count = 0

        for citation in citations:
            record = self.chain.get_record(citation.evidence_id)
            if not record:
                continue

            # Get evidence reliability
            ev_reliability = self.score_evidence(citation.evidence_id)

            if citation.support_type == "supports":
                result.supporting_evidence += 1
                supporting_scores.append(ev_reliability.reliability_score * citation.relevance)
            elif citation.support_type == "contradicts":
                result.contradicting_evidence += 1
                contradicting_count += 1

        # Compute component scores
        if supporting_scores:
            result.source_quality = sum(supporting_scores) / len(supporting_scores)
            result.evidence_coverage = min(
                1.0, len(supporting_scores) / 3
            )  # 3+ sources = full coverage
        else:
            result.source_quality = 0.0
            result.evidence_coverage = 0.0

        # Consistency (penalize contradictions)
        if result.total_citations > 0:
            contradiction_ratio = contradicting_count / result.total_citations
            result.consistency = 1.0 - contradiction_ratio

        # Check formal verification
        if claim_id in self.verification_results:
            v_result = self.verification_results[claim_id]
            if v_result.get("status") == "verified":
                result.verification_status = 1.0
                result.verified_by.append(v_result.get("method", "unknown"))
            elif v_result.get("status") == "refuted":
                result.verification_status = 0.0
                result.warnings.append("Claim was formally refuted")

        # Compute final reliability score
        result.reliability_score = (
            self.WEIGHTS["evidence_coverage"] * result.evidence_coverage
            + self.WEIGHTS["source_quality"] * result.source_quality
            + self.WEIGHTS["consistency"] * result.consistency
            + self.WEIGHTS["verification"] * result.verification_status
        )

        # Set confidence (how sure we are about the score)
        result.confidence = min(1.0, result.total_citations / 5) * result.consistency

        # Determine level
        result.level = self._score_to_level(result.reliability_score)

        # Add warnings
        if result.contradicting_evidence > 0:
            result.warnings.append(
                f"{result.contradicting_evidence} source(s) contradict this claim"
            )
        if result.evidence_coverage < 0.5:
            result.warnings.append("Limited evidence coverage")

        return result

    def score_evidence(self, evidence_id: str) -> EvidenceReliability:
        """
        Compute reliability score for a piece of evidence.

        Uses source type, freshness, and chain verification.
        """
        record = self.chain.get_record(evidence_id)

        if not record:
            return EvidenceReliability(
                evidence_id=evidence_id,
                source_type=SourceType.UNKNOWN,
                reliability_score=0.0,
            )

        # Get base authority from source type
        authority = self.SOURCE_AUTHORITY.get(record.source_type, 0.5)

        # Adjust for verification status
        if record.verified:
            authority = min(1.0, authority + 0.2)

        # Calculate freshness from timestamp
        freshness = self._calculate_freshness(record.timestamp)

        # Check chain integrity
        valid, _ = self.provenance.verify_chain_integrity()
        chain_verified = valid

        # Combine scores
        reliability = (
            0.4 * authority
            + 0.3 * record.confidence
            + 0.2 * freshness
            + 0.1 * (1.0 if chain_verified else 0.5)
        )

        return EvidenceReliability(
            evidence_id=evidence_id,
            source_type=record.source_type,
            reliability_score=reliability,
            freshness=freshness,
            authority=authority,
            confidence=record.confidence,
            chain_verified=chain_verified,
            content_verified=record.verified,
        )

    def score_all_claims(self, claims: dict[str, str]) -> dict[str, ClaimReliability]:
        """
        Score multiple claims.

        Args:
            claims: Dict of claim_id -> claim_text

        Returns:
            Dict of claim_id -> ClaimReliability
        """
        return {
            claim_id: self.score_claim(claim_id, claim_text)
            for claim_id, claim_text in claims.items()
        }

    def get_speculative_claims(
        self,
        claims: dict[str, str],
    ) -> list[str]:
        """Get claims that have no evidence support."""
        speculative = []
        for claim_id, claim_text in claims.items():
            reliability = self.score_claim(claim_id, claim_text)
            if reliability.level == ReliabilityLevel.SPECULATIVE:
                speculative.append(claim_id)
        return speculative

    def get_low_reliability_claims(
        self,
        claims: dict[str, str],
        threshold: float = 0.5,
    ) -> list[tuple[str, ClaimReliability]]:
        """Get claims below reliability threshold."""
        low_reliability = []
        for claim_id, claim_text in claims.items():
            reliability = self.score_claim(claim_id, claim_text)
            if reliability.reliability_score < threshold:
                low_reliability.append((claim_id, reliability))
        return sorted(low_reliability, key=lambda x: x[1].reliability_score)

    def _score_to_level(self, score: float) -> ReliabilityLevel:
        """Convert numeric score to qualitative level."""
        if score >= 0.9:
            return ReliabilityLevel.VERY_HIGH
        elif score >= 0.7:
            return ReliabilityLevel.HIGH
        elif score >= 0.5:
            return ReliabilityLevel.MEDIUM
        elif score >= 0.3:
            return ReliabilityLevel.LOW
        else:
            return ReliabilityLevel.VERY_LOW

    def _calculate_freshness(self, timestamp: datetime) -> float:
        """Calculate freshness score from timestamp."""
        try:
            age_days = (datetime.now() - timestamp).days

            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.9
            elif age_days < 90:
                return 0.7
            elif age_days < 365:
                return 0.5
            else:
                return 0.3
        except (ValueError, TypeError):
            return 0.5  # Default score on date parsing error

    def generate_reliability_report(
        self,
        claims: dict[str, str],
    ) -> dict:
        """
        Generate a comprehensive reliability report.

        Returns summary statistics and per-claim details.
        """
        results = self.score_all_claims(claims)

        # Compute summary
        scores = [r.reliability_score for r in results.values()]
        levels = [r.level for r in results.values()]

        speculative_count = sum(1 for l in levels if l == ReliabilityLevel.SPECULATIVE)
        low_count = sum(1 for s in scores if s < 0.5)
        high_count = sum(1 for s in scores if s >= 0.7)

        # Check provenance chain
        chain_valid, chain_errors = self.provenance.verify_chain_integrity()

        return {
            "summary": {
                "total_claims": len(claims),
                "avg_reliability": sum(scores) / len(scores) if scores else 0,
                "speculative_claims": speculative_count,
                "low_reliability_claims": low_count,
                "high_reliability_claims": high_count,
                "chain_integrity": chain_valid,
                "chain_errors": len(chain_errors),
            },
            "claims": {claim_id: result.to_dict() for claim_id, result in results.items()},
            "warnings": [w for r in results.values() for w in r.warnings],
        }


def compute_claim_reliability(
    claim_id: str,
    claim_text: str,
    provenance: ProvenanceManager,
    verification_results: Optional[dict] = None,
) -> ClaimReliability:
    """Convenience function to compute single claim reliability."""
    scorer = ReliabilityScorer(provenance, verification_results)
    return scorer.score_claim(claim_id, claim_text)

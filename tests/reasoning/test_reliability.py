"""
Tests for Reliability Scoring.

Tests the reliability module including:
- ReliabilityLevel enum
- ClaimReliability dataclass
- EvidenceReliability dataclass
- ReliabilityScorer class
- Claim and evidence scoring
- Reliability report generation
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aragora.reasoning.provenance import (
    ProvenanceManager,
    SourceType,
    TransformationType,
)
from aragora.reasoning.reliability import (
    ClaimReliability,
    EvidenceReliability,
    ReliabilityLevel,
    ReliabilityScorer,
    compute_claim_reliability,
)


# =============================================================================
# ReliabilityLevel Tests
# =============================================================================


class TestReliabilityLevel:
    """Test ReliabilityLevel enum."""

    def test_level_values(self):
        """Test reliability level values."""
        assert ReliabilityLevel.VERY_HIGH.value == "very_high"
        assert ReliabilityLevel.HIGH.value == "high"
        assert ReliabilityLevel.MEDIUM.value == "medium"
        assert ReliabilityLevel.LOW.value == "low"
        assert ReliabilityLevel.VERY_LOW.value == "very_low"
        assert ReliabilityLevel.SPECULATIVE.value == "speculative"

    def test_all_levels_present(self):
        """Test all expected levels are defined."""
        levels = [level.value for level in ReliabilityLevel]
        assert len(levels) == 6
        assert "very_high" in levels
        assert "speculative" in levels


# =============================================================================
# ClaimReliability Tests
# =============================================================================


class TestClaimReliability:
    """Test ClaimReliability dataclass."""

    def test_basic_creation(self):
        """Test basic ClaimReliability creation."""
        reliability = ClaimReliability(
            claim_id="claim-123",
            claim_text="Test claim about something.",
        )
        assert reliability.claim_id == "claim-123"
        assert reliability.claim_text == "Test claim about something."
        assert reliability.reliability_score == 0.0
        assert reliability.level == ReliabilityLevel.SPECULATIVE

    def test_default_values(self):
        """Test default values are set correctly."""
        reliability = ClaimReliability(
            claim_id="test",
            claim_text="test",
        )
        assert reliability.confidence == 0.0
        assert reliability.evidence_coverage == 0.0
        assert reliability.source_quality == 0.0
        assert reliability.consistency == 1.0  # Default to no contradictions
        assert reliability.verification_status == 0.0
        assert reliability.supporting_evidence == 0
        assert reliability.contradicting_evidence == 0
        assert reliability.total_citations == 0
        assert reliability.warnings == []
        assert reliability.verified_by == []

    def test_custom_values(self):
        """Test custom values are preserved."""
        reliability = ClaimReliability(
            claim_id="claim-456",
            claim_text="Well-supported claim.",
            reliability_score=0.85,
            confidence=0.9,
            evidence_coverage=0.8,
            source_quality=0.9,
            consistency=0.95,
            verification_status=1.0,
            supporting_evidence=5,
            contradicting_evidence=1,
            total_citations=6,
            level=ReliabilityLevel.HIGH,
            warnings=["Minor inconsistency"],
            verified_by=["z3"],
        )
        assert reliability.reliability_score == 0.85
        assert reliability.confidence == 0.9
        assert reliability.level == ReliabilityLevel.HIGH
        assert len(reliability.warnings) == 1
        assert "z3" in reliability.verified_by

    def test_to_dict(self):
        """Test serialization to dictionary."""
        reliability = ClaimReliability(
            claim_id="claim-789",
            claim_text="Serializable claim.",
            reliability_score=0.7,
            level=ReliabilityLevel.HIGH,
        )
        data = reliability.to_dict()

        assert data["claim_id"] == "claim-789"
        assert data["claim_text"] == "Serializable claim."
        assert data["reliability_score"] == 0.7
        assert data["level"] == "high"
        assert "warnings" in data
        assert "verified_by" in data


# =============================================================================
# EvidenceReliability Tests
# =============================================================================


class TestEvidenceReliability:
    """Test EvidenceReliability dataclass."""

    def test_basic_creation(self):
        """Test basic EvidenceReliability creation."""
        reliability = EvidenceReliability(
            evidence_id="ev-123",
            source_type=SourceType.DOCUMENT,
        )
        assert reliability.evidence_id == "ev-123"
        assert reliability.source_type == SourceType.DOCUMENT

    def test_default_values(self):
        """Test default values are set correctly."""
        reliability = EvidenceReliability(
            evidence_id="test",
            source_type=SourceType.UNKNOWN,
        )
        assert reliability.reliability_score == 0.5
        assert reliability.freshness == 0.5
        assert reliability.authority == 0.5
        assert reliability.confidence == 0.5
        assert reliability.chain_verified is False
        assert reliability.content_verified is False

    def test_custom_values(self):
        """Test custom values are preserved."""
        reliability = EvidenceReliability(
            evidence_id="ev-456",
            source_type=SourceType.COMPUTATION,
            reliability_score=0.95,
            freshness=1.0,
            authority=0.9,
            confidence=0.98,
            chain_verified=True,
            content_verified=True,
        )
        assert reliability.reliability_score == 0.95
        assert reliability.freshness == 1.0
        assert reliability.chain_verified is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        reliability = EvidenceReliability(
            evidence_id="ev-789",
            source_type=SourceType.DATABASE,
            reliability_score=0.85,
        )
        data = reliability.to_dict()

        assert data["evidence_id"] == "ev-789"
        assert data["source_type"] == "database"
        assert data["reliability_score"] == 0.85
        assert "chain_verified" in data
        assert "content_verified" in data


# =============================================================================
# ReliabilityScorer Tests
# =============================================================================


class TestReliabilityScorerInitialization:
    """Test ReliabilityScorer initialization."""

    def test_init_with_provenance(self):
        """Test initialization with ProvenanceManager."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer.provenance == provenance
        assert scorer.chain == provenance.chain
        assert scorer.graph == provenance.graph
        assert scorer.verification_results == {}

    def test_init_with_verification_results(self):
        """Test initialization with verification results."""
        provenance = ProvenanceManager()
        results = {"claim-1": {"status": "verified", "method": "z3"}}
        scorer = ReliabilityScorer(provenance, results)

        assert scorer.verification_results == results


class TestReliabilityScorerClaimScoring:
    """Test claim scoring functionality."""

    @pytest.fixture
    def provenance_with_evidence(self):
        """Create provenance with evidence and citations."""
        provenance = ProvenanceManager()

        # Add evidence
        ev1 = provenance.record_evidence(
            content="Supporting evidence 1",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        ev2 = provenance.record_evidence(
            content="Supporting evidence 2",
            source_type=SourceType.COMPUTATION,
            source_id="comp-1",
        )

        # Cite evidence for claim
        provenance.cite_evidence("claim-1", ev1.id, relevance=0.9, support_type="supports")
        provenance.cite_evidence("claim-1", ev2.id, relevance=0.8, support_type="supports")

        return provenance

    def test_score_claim_no_evidence(self):
        """Test scoring claim with no evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        result = scorer.score_claim("no-evidence-claim", "Unsupported claim")

        assert result.level == ReliabilityLevel.SPECULATIVE
        assert result.total_citations == 0
        assert "No evidence supports this claim" in result.warnings

    def test_score_claim_with_supporting_evidence(self, provenance_with_evidence):
        """Test scoring claim with supporting evidence."""
        scorer = ReliabilityScorer(provenance_with_evidence)

        result = scorer.score_claim("claim-1", "Well-supported claim")

        assert result.total_citations == 2
        assert result.supporting_evidence == 2
        assert result.contradicting_evidence == 0
        assert result.consistency == 1.0
        assert result.evidence_coverage > 0
        assert result.source_quality > 0

    def test_score_claim_with_contradicting_evidence(self):
        """Test scoring claim with contradicting evidence."""
        provenance = ProvenanceManager()

        ev1 = provenance.record_evidence(
            content="Supporting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        ev2 = provenance.record_evidence(
            content="Contradicting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-2",
        )

        provenance.cite_evidence("claim-1", ev1.id, support_type="supports")
        provenance.cite_evidence("claim-1", ev2.id, support_type="contradicts")

        scorer = ReliabilityScorer(provenance)
        result = scorer.score_claim("claim-1", "Contested claim")

        assert result.supporting_evidence == 1
        assert result.contradicting_evidence == 1
        assert result.consistency < 1.0
        assert any("contradict" in w for w in result.warnings)

    def test_score_claim_with_verification_verified(self):
        """Test scoring claim with formal verification (verified)."""
        provenance = ProvenanceManager()

        # Must have evidence for verification to be applied
        ev = provenance.record_evidence(
            content="Supporting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        provenance.cite_evidence("claim-1", ev.id, support_type="supports")

        verification_results = {"claim-1": {"status": "verified", "method": "z3"}}
        scorer = ReliabilityScorer(provenance, verification_results)

        result = scorer.score_claim("claim-1", "Verified claim")

        assert result.verification_status == 1.0
        assert "z3" in result.verified_by

    def test_score_claim_with_verification_refuted(self):
        """Test scoring claim with formal verification (refuted)."""
        provenance = ProvenanceManager()

        # Must have evidence for verification to be applied
        ev = provenance.record_evidence(
            content="Supporting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        provenance.cite_evidence("claim-1", ev.id, support_type="supports")

        verification_results = {"claim-1": {"status": "refuted", "method": "lean"}}
        scorer = ReliabilityScorer(provenance, verification_results)

        result = scorer.score_claim("claim-1", "Refuted claim")

        assert result.verification_status == 0.0
        assert any("refuted" in w for w in result.warnings)


class TestReliabilityScorerEvidenceScoring:
    """Test evidence scoring functionality."""

    def test_score_evidence_not_found(self):
        """Test scoring non-existent evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        result = scorer.score_evidence("nonexistent-id")

        assert result.reliability_score == 0.0
        assert result.source_type == SourceType.UNKNOWN

    def test_score_evidence_by_source_type(self):
        """Test scoring varies by source type."""
        provenance = ProvenanceManager()

        ev_computation = provenance.record_evidence(
            content="Computed result",
            source_type=SourceType.COMPUTATION,
            source_id="calc-1",
        )
        ev_agent = provenance.record_evidence(
            content="Agent generated",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
        )

        scorer = ReliabilityScorer(provenance)

        comp_result = scorer.score_evidence(ev_computation.id)
        agent_result = scorer.score_evidence(ev_agent.id)

        # Computation should have higher authority than agent-generated
        assert comp_result.authority > agent_result.authority

    def test_score_evidence_verified_boost(self):
        """Test verified evidence gets score boost."""
        provenance = ProvenanceManager()

        ev = provenance.record_evidence(
            content="Verified evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
            verified=True,
        )

        scorer = ReliabilityScorer(provenance)
        result = scorer.score_evidence(ev.id)

        # Verified evidence should have content_verified flag
        assert result.content_verified is True


class TestReliabilityScorerLevelConversion:
    """Test score to level conversion."""

    def test_score_to_level_very_high(self):
        """Test very high reliability level."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer._score_to_level(0.95) == ReliabilityLevel.VERY_HIGH
        assert scorer._score_to_level(0.9) == ReliabilityLevel.VERY_HIGH

    def test_score_to_level_high(self):
        """Test high reliability level."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer._score_to_level(0.85) == ReliabilityLevel.HIGH
        assert scorer._score_to_level(0.7) == ReliabilityLevel.HIGH

    def test_score_to_level_medium(self):
        """Test medium reliability level."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer._score_to_level(0.6) == ReliabilityLevel.MEDIUM
        assert scorer._score_to_level(0.5) == ReliabilityLevel.MEDIUM

    def test_score_to_level_low(self):
        """Test low reliability level."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer._score_to_level(0.4) == ReliabilityLevel.LOW
        assert scorer._score_to_level(0.3) == ReliabilityLevel.LOW

    def test_score_to_level_very_low(self):
        """Test very low reliability level."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        assert scorer._score_to_level(0.2) == ReliabilityLevel.VERY_LOW
        assert scorer._score_to_level(0.0) == ReliabilityLevel.VERY_LOW


class TestReliabilityScorerFreshness:
    """Test freshness calculation."""

    def test_freshness_recent(self):
        """Test freshness for recent evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        # Recent timestamp (within 7 days)
        recent = datetime.now() - timedelta(days=1)
        assert scorer._calculate_freshness(recent) == 1.0

    def test_freshness_one_week(self):
        """Test freshness for week-old evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        # Within 30 days
        one_week = datetime.now() - timedelta(days=15)
        assert scorer._calculate_freshness(one_week) == 0.9

    def test_freshness_one_month(self):
        """Test freshness for month-old evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        # Within 90 days
        one_month = datetime.now() - timedelta(days=60)
        assert scorer._calculate_freshness(one_month) == 0.7

    def test_freshness_three_months(self):
        """Test freshness for quarter-old evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        # Within 365 days
        three_months = datetime.now() - timedelta(days=180)
        assert scorer._calculate_freshness(three_months) == 0.5

    def test_freshness_old(self):
        """Test freshness for old evidence."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        # Over 1 year
        old = datetime.now() - timedelta(days=400)
        assert scorer._calculate_freshness(old) == 0.3


class TestReliabilityScorerBatchOperations:
    """Test batch operations."""

    @pytest.fixture
    def provenance_with_claims(self):
        """Create provenance with multiple claims and evidence."""
        provenance = ProvenanceManager()

        # Create evidence
        ev = provenance.record_evidence(
            content="Evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )

        # Cite for some claims
        provenance.cite_evidence("supported-claim", ev.id, support_type="supports")

        return provenance

    def test_score_all_claims(self, provenance_with_claims):
        """Test scoring multiple claims at once."""
        scorer = ReliabilityScorer(provenance_with_claims)

        claims = {
            "supported-claim": "Claim with evidence",
            "unsupported-claim": "Claim without evidence",
        }
        results = scorer.score_all_claims(claims)

        assert "supported-claim" in results
        assert "unsupported-claim" in results
        assert results["supported-claim"].total_citations > 0
        assert results["unsupported-claim"].total_citations == 0

    def test_get_speculative_claims(self, provenance_with_claims):
        """Test getting speculative claims."""
        scorer = ReliabilityScorer(provenance_with_claims)

        claims = {
            "supported-claim": "Has evidence",
            "speculative-1": "No evidence",
            "speculative-2": "Also no evidence",
        }
        speculative = scorer.get_speculative_claims(claims)

        assert "speculative-1" in speculative
        assert "speculative-2" in speculative
        assert "supported-claim" not in speculative

    def test_get_low_reliability_claims(self, provenance_with_claims):
        """Test getting low reliability claims."""
        scorer = ReliabilityScorer(provenance_with_claims)

        claims = {
            "claim-1": "Some claim",
            "claim-2": "Another claim",
        }
        low_reliability = scorer.get_low_reliability_claims(claims, threshold=0.9)

        # All claims likely below 0.9 threshold
        assert len(low_reliability) > 0
        # Should be sorted by score
        if len(low_reliability) > 1:
            assert (
                low_reliability[0][1].reliability_score <= low_reliability[1][1].reliability_score
            )


class TestReliabilityScorerReport:
    """Test reliability report generation."""

    def test_generate_report_empty(self):
        """Test generating report with no claims."""
        provenance = ProvenanceManager()
        scorer = ReliabilityScorer(provenance)

        report = scorer.generate_reliability_report({})

        assert report["summary"]["total_claims"] == 0
        assert report["summary"]["avg_reliability"] == 0
        assert report["claims"] == {}
        assert report["warnings"] == []

    def test_generate_report_with_claims(self):
        """Test generating report with claims."""
        provenance = ProvenanceManager()
        ev = provenance.record_evidence(
            content="Evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        provenance.cite_evidence("claim-1", ev.id, support_type="supports")

        scorer = ReliabilityScorer(provenance)
        claims = {
            "claim-1": "Supported claim",
            "claim-2": "Unsupported claim",
        }
        report = scorer.generate_reliability_report(claims)

        assert report["summary"]["total_claims"] == 2
        assert "chain_integrity" in report["summary"]
        assert "chain_errors" in report["summary"]
        assert "claim-1" in report["claims"]
        assert "claim-2" in report["claims"]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestComputeClaimReliability:
    """Test compute_claim_reliability convenience function."""

    def test_compute_claim_reliability(self):
        """Test convenience function."""
        provenance = ProvenanceManager()

        result = compute_claim_reliability(
            claim_id="test-claim",
            claim_text="Test claim text",
            provenance=provenance,
        )

        assert isinstance(result, ClaimReliability)
        assert result.claim_id == "test-claim"
        assert result.claim_text == "Test claim text"

    def test_compute_claim_reliability_with_verification(self):
        """Test convenience function with verification results."""
        provenance = ProvenanceManager()

        # Must have evidence for verification to be applied
        ev = provenance.record_evidence(
            content="Supporting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        provenance.cite_evidence("test-claim", ev.id, support_type="supports")

        verification = {"test-claim": {"status": "verified", "method": "z3"}}

        result = compute_claim_reliability(
            claim_id="test-claim",
            claim_text="Verified claim",
            provenance=provenance,
            verification_results=verification,
        )

        assert result.verification_status == 1.0
        assert "z3" in result.verified_by

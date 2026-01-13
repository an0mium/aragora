"""Tests for the reliability module - claim and evidence scoring."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from aragora.reasoning.reliability import (
    ReliabilityLevel,
    ClaimReliability,
    EvidenceReliability,
    ReliabilityScorer,
    compute_claim_reliability,
)
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provenance_manager():
    """Mock ProvenanceManager with chain and graph."""
    manager = MagicMock()
    manager.chain = MagicMock()
    manager.graph = MagicMock()
    manager.verify_chain_integrity.return_value = (True, [])
    return manager


@pytest.fixture
def reliability_scorer(mock_provenance_manager):
    """ReliabilityScorer with mocked dependencies."""
    return ReliabilityScorer(mock_provenance_manager)


def create_mock_record(
    evidence_id: str = "ev-1",
    source_type: SourceType = SourceType.DOCUMENT,
    confidence: float = 0.8,
    verified: bool = False,
    timestamp: datetime = None,
):
    """Create a mock provenance record."""
    record = MagicMock()
    record.evidence_id = evidence_id
    record.source_type = source_type
    record.confidence = confidence
    record.verified = verified
    record.timestamp = timestamp or datetime.now()
    return record


def create_mock_citation(
    evidence_id: str = "ev-1",
    support_type: str = "supports",
    relevance: float = 0.9,
):
    """Create a mock citation."""
    citation = MagicMock()
    citation.evidence_id = evidence_id
    citation.support_type = support_type
    citation.relevance = relevance
    return citation


# =============================================================================
# ReliabilityLevel Enum Tests
# =============================================================================


class TestReliabilityLevel:
    """Tests for ReliabilityLevel enum."""

    def test_all_6_levels_exist(self):
        """Should have all 6 reliability levels."""
        expected = ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW", "SPECULATIVE"]
        actual = [l.name for l in ReliabilityLevel]

        for e in expected:
            assert e in actual

    def test_enum_values_are_strings(self):
        """Enum values should be lowercase strings."""
        assert ReliabilityLevel.VERY_HIGH.value == "very_high"
        assert ReliabilityLevel.SPECULATIVE.value == "speculative"


# =============================================================================
# ClaimReliability Dataclass Tests
# =============================================================================


class TestClaimReliability:
    """Tests for ClaimReliability dataclass."""

    def test_default_values(self):
        """Default values should be correctly initialized."""
        claim = ClaimReliability(claim_id="test", claim_text="Test claim")

        assert claim.reliability_score == 0.0
        assert claim.confidence == 0.0
        assert claim.consistency == 1.0  # Default to consistent
        assert claim.level == ReliabilityLevel.SPECULATIVE

    def test_to_dict_converts_enum(self):
        """to_dict should convert level enum to value."""
        claim = ClaimReliability(
            claim_id="test",
            claim_text="Test",
            level=ReliabilityLevel.HIGH,
        )
        d = claim.to_dict()

        assert d["level"] == "high"

    def test_warnings_list_preserved(self):
        """Warnings list should be preserved in to_dict."""
        claim = ClaimReliability(
            claim_id="test",
            claim_text="Test",
            warnings=["Warning 1", "Warning 2"],
        )
        d = claim.to_dict()

        assert d["warnings"] == ["Warning 1", "Warning 2"]

    def test_verified_by_list_preserved(self):
        """verified_by list should be preserved in to_dict."""
        claim = ClaimReliability(
            claim_id="test",
            claim_text="Test",
            verified_by=["z3", "lean4"],
        )
        d = claim.to_dict()

        assert d["verified_by"] == ["z3", "lean4"]


# =============================================================================
# EvidenceReliability Dataclass Tests
# =============================================================================


class TestEvidenceReliability:
    """Tests for EvidenceReliability dataclass."""

    def test_default_values(self):
        """Default values should be 0.5 for scores."""
        evidence = EvidenceReliability(
            evidence_id="ev-1",
            source_type=SourceType.DOCUMENT,
        )

        assert evidence.reliability_score == 0.5
        assert evidence.freshness == 0.5
        assert evidence.authority == 0.5
        assert evidence.confidence == 0.5

    def test_to_dict_converts_source_type(self):
        """to_dict should convert SourceType enum."""
        evidence = EvidenceReliability(
            evidence_id="ev-1",
            source_type=SourceType.DATABASE,
        )
        d = evidence.to_dict()

        assert d["source_type"] == "database"

    def test_verification_flags(self):
        """Verification flags should be preserved."""
        evidence = EvidenceReliability(
            evidence_id="ev-1",
            source_type=SourceType.DOCUMENT,
            chain_verified=True,
            content_verified=True,
        )

        assert evidence.chain_verified is True
        assert evidence.content_verified is True


# =============================================================================
# Evidence Scoring Tests
# =============================================================================


class TestEvidenceScoring:
    """Tests for evidence reliability scoring."""

    def test_source_authority_computation(self, mock_provenance_manager):
        """Authority should match SOURCE_AUTHORITY for each type."""
        record = create_mock_record(source_type=SourceType.COMPUTATION)
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        # COMPUTATION has base authority 0.9
        assert result.authority == 0.9

    def test_source_authority_agent_generated(self, mock_provenance_manager):
        """AGENT_GENERATED should have authority 0.4."""
        record = create_mock_record(source_type=SourceType.AGENT_GENERATED)
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.authority == 0.4

    def test_verification_boost(self, mock_provenance_manager):
        """Verified evidence should get +0.2 authority."""
        record = create_mock_record(
            source_type=SourceType.AGENT_GENERATED,  # Base 0.4
            verified=True,
        )
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert abs(result.authority - 0.6) < 0.001  # 0.4 + 0.2 (float tolerance)

    def test_authority_capped_at_1(self, mock_provenance_manager):
        """Authority should not exceed 1.0."""
        record = create_mock_record(
            source_type=SourceType.COMPUTATION,  # Base 0.9
            verified=True,  # +0.2 = 1.1, but should cap
        )
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.authority == 1.0

    def test_freshness_very_recent(self, mock_provenance_manager):
        """Evidence < 7 days should have freshness 1.0."""
        record = create_mock_record(timestamp=datetime.now() - timedelta(days=3))
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.freshness == 1.0

    def test_freshness_7_to_30_days(self, mock_provenance_manager):
        """Evidence 7-30 days should have freshness 0.9."""
        record = create_mock_record(timestamp=datetime.now() - timedelta(days=15))
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.freshness == 0.9

    def test_freshness_30_to_90_days(self, mock_provenance_manager):
        """Evidence 30-90 days should have freshness 0.7."""
        record = create_mock_record(timestamp=datetime.now() - timedelta(days=60))
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.freshness == 0.7

    def test_freshness_90_to_365_days(self, mock_provenance_manager):
        """Evidence 90-365 days should have freshness 0.5."""
        record = create_mock_record(timestamp=datetime.now() - timedelta(days=200))
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.freshness == 0.5

    def test_freshness_over_365_days(self, mock_provenance_manager):
        """Evidence > 365 days should have freshness 0.3."""
        record = create_mock_record(timestamp=datetime.now() - timedelta(days=400))
        mock_provenance_manager.chain.get_record.return_value = record

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("ev-1")

        assert result.freshness == 0.3

    def test_missing_evidence_returns_zero_score(self, mock_provenance_manager):
        """Missing evidence should return 0.0 reliability."""
        mock_provenance_manager.chain.get_record.return_value = None

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_evidence("nonexistent")

        assert result.reliability_score == 0.0
        assert result.source_type == SourceType.UNKNOWN


# =============================================================================
# Claim Scoring Tests
# =============================================================================


class TestClaimScoring:
    """Tests for claim reliability scoring."""

    def test_no_evidence_speculative(self, mock_provenance_manager):
        """Claim with no evidence should be SPECULATIVE."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1", "Test claim")

        assert result.level == ReliabilityLevel.SPECULATIVE
        assert "No evidence supports this claim" in result.warnings

    def test_evidence_coverage_zero_citations(self, mock_provenance_manager):
        """Zero supporting evidence -> coverage 0.0."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        assert result.evidence_coverage == 0.0

    def test_evidence_coverage_saturates_at_3(self, mock_provenance_manager):
        """3+ supporting sources -> coverage 1.0."""
        citations = [create_mock_citation(f"ev-{i}") for i in range(5)]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations

        # Setup mock records
        def get_record_side_effect(ev_id):
            return create_mock_record(evidence_id=ev_id)

        mock_provenance_manager.chain.get_record.side_effect = get_record_side_effect

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        assert result.evidence_coverage == 1.0

    def test_consistency_no_contradictions(self, mock_provenance_manager):
        """No contradictions -> consistency 1.0."""
        citations = [
            create_mock_citation("ev-1", support_type="supports"),
            create_mock_citation("ev-2", support_type="supports"),
        ]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        assert result.consistency == 1.0

    def test_consistency_all_contradictions(self, mock_provenance_manager):
        """All contradictions -> consistency 0.0."""
        citations = [
            create_mock_citation("ev-1", support_type="contradicts"),
            create_mock_citation("ev-2", support_type="contradicts"),
        ]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        assert result.consistency == 0.0

    def test_verification_status_verified(self, mock_provenance_manager):
        """Verified claim should have verification_status 1.0."""
        # Need citations to reach verification check (early return otherwise)
        citations = [create_mock_citation("ev-1")]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        verification_results = {"claim-1": {"status": "verified", "method": "z3"}}
        scorer = ReliabilityScorer(mock_provenance_manager, verification_results)
        result = scorer.score_claim("claim-1")

        assert result.verification_status == 1.0
        assert "z3" in result.verified_by

    def test_verification_status_refuted(self, mock_provenance_manager):
        """Refuted claim should have verification_status 0.0 and warning."""
        # Need citations to reach verification check (early return otherwise)
        citations = [create_mock_citation("ev-1")]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        verification_results = {"claim-1": {"status": "refuted"}}
        scorer = ReliabilityScorer(mock_provenance_manager, verification_results)
        result = scorer.score_claim("claim-1")

        assert result.verification_status == 0.0
        # Check that refuted warning is in warnings list
        refuted_warning = [w for w in result.warnings if "refuted" in w.lower()]
        assert len(refuted_warning) > 0

    def test_final_score_weighted_sum(self, mock_provenance_manager):
        """Final score should be weighted sum of components."""
        # Setup for known component values
        citations = [create_mock_citation("ev-1")]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        # Verify weights are 0.25 each
        expected = (
            0.25 * result.evidence_coverage
            + 0.25 * result.source_quality
            + 0.25 * result.consistency
            + 0.25 * result.verification_status
        )
        assert abs(result.reliability_score - expected) < 0.001

    def test_confidence_saturates_at_5_citations(self, mock_provenance_manager):
        """Confidence should saturate when >= 5 citations."""
        citations = [create_mock_citation(f"ev-{i}") for i in range(10)]
        mock_provenance_manager.graph.get_claim_evidence.return_value = citations
        mock_provenance_manager.chain.get_record.return_value = create_mock_record()

        scorer = ReliabilityScorer(mock_provenance_manager)
        result = scorer.score_claim("claim-1")

        # 10/5 = 2.0, min(1.0, 2.0) * consistency = 1.0 * consistency
        assert result.confidence <= 1.0


# =============================================================================
# Level Classification Tests
# =============================================================================


class TestLevelClassification:
    """Tests for score to level conversion."""

    def test_very_high_threshold(self, reliability_scorer):
        """Score >= 0.9 should be VERY_HIGH."""
        level = reliability_scorer._score_to_level(0.9)
        assert level == ReliabilityLevel.VERY_HIGH

        level = reliability_scorer._score_to_level(0.95)
        assert level == ReliabilityLevel.VERY_HIGH

    def test_high_threshold(self, reliability_scorer):
        """Score >= 0.7 and < 0.9 should be HIGH."""
        level = reliability_scorer._score_to_level(0.7)
        assert level == ReliabilityLevel.HIGH

        level = reliability_scorer._score_to_level(0.85)
        assert level == ReliabilityLevel.HIGH

    def test_medium_threshold(self, reliability_scorer):
        """Score >= 0.5 and < 0.7 should be MEDIUM."""
        level = reliability_scorer._score_to_level(0.5)
        assert level == ReliabilityLevel.MEDIUM

        level = reliability_scorer._score_to_level(0.65)
        assert level == ReliabilityLevel.MEDIUM

    def test_low_threshold(self, reliability_scorer):
        """Score >= 0.3 and < 0.5 should be LOW."""
        level = reliability_scorer._score_to_level(0.3)
        assert level == ReliabilityLevel.LOW

        level = reliability_scorer._score_to_level(0.45)
        assert level == ReliabilityLevel.LOW

    def test_very_low_threshold(self, reliability_scorer):
        """Score < 0.3 should be VERY_LOW."""
        level = reliability_scorer._score_to_level(0.1)
        assert level == ReliabilityLevel.VERY_LOW

        level = reliability_scorer._score_to_level(0.29)
        assert level == ReliabilityLevel.VERY_LOW


# =============================================================================
# Filtering and Report Tests
# =============================================================================


class TestFilteringAndReports:
    """Tests for filtering methods and report generation."""

    def test_get_speculative_claims(self, mock_provenance_manager):
        """Should return only speculative claims."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        claims = {"claim-1": "Test 1", "claim-2": "Test 2"}
        speculative = scorer.get_speculative_claims(claims)

        assert len(speculative) == 2
        assert "claim-1" in speculative
        assert "claim-2" in speculative

    def test_get_low_reliability_claims_threshold(self, mock_provenance_manager):
        """Should filter by threshold and sort by score."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        claims = {"claim-1": "Test"}
        low = scorer.get_low_reliability_claims(claims, threshold=0.5)

        assert len(low) == 1
        assert low[0][1].reliability_score < 0.5

    def test_generate_reliability_report_summary(self, mock_provenance_manager):
        """Report should include summary statistics."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        claims = {"claim-1": "Test 1", "claim-2": "Test 2"}
        report = scorer.generate_reliability_report(claims)

        assert "summary" in report
        assert report["summary"]["total_claims"] == 2
        assert "speculative_claims" in report["summary"]
        assert "chain_integrity" in report["summary"]

    def test_score_all_claims_batch(self, mock_provenance_manager):
        """score_all_claims should process all claims."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        scorer = ReliabilityScorer(mock_provenance_manager)
        claims = {"claim-1": "Test 1", "claim-2": "Test 2", "claim-3": "Test 3"}
        results = scorer.score_all_claims(claims)

        assert len(results) == 3
        assert "claim-1" in results
        assert "claim-2" in results
        assert "claim-3" in results


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for compute_claim_reliability function."""

    def test_compute_claim_reliability(self, mock_provenance_manager):
        """Should return ClaimReliability instance."""
        mock_provenance_manager.graph.get_claim_evidence.return_value = []

        result = compute_claim_reliability(
            claim_id="test-claim",
            claim_text="Test text",
            provenance=mock_provenance_manager,
        )

        assert isinstance(result, ClaimReliability)
        assert result.claim_id == "test-claim"
        assert result.claim_text == "Test text"

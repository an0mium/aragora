"""
Tests for the Gauntlet adversarial stress-testing module.

Tests cover:
- GauntletConfig creation and validation
- GauntletResult properties and aggregation
- GauntletOrchestrator orchestration flow
- Finding conversion and categorization
- Decision Receipt generation
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

from aragora.modes.gauntlet import (
    GauntletOrchestrator,
    GauntletConfig,
    GauntletResult,
    Finding,
    VerifiedClaim,
    InputType,
    Verdict,
    run_gauntlet,
    QUICK_GAUNTLET,
    THOROUGH_GAUNTLET,
    CODE_REVIEW_GAUNTLET,
    POLICY_GAUNTLET,
)
from aragora.export.decision_receipt import (
    DecisionReceipt,
    DecisionReceiptGenerator,
    ReceiptFinding,
    ReceiptDissent,
    ReceiptVerification,
    generate_decision_receipt,
)
from aragora.debate.risk_assessor import RiskLevel, RiskAssessment
from aragora.debate.consensus import DissentRecord, UnresolvedTension


class TestGauntletConfig:
    """Tests for GauntletConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GauntletConfig()
        assert config.input_type == InputType.SPEC
        assert config.input_content == ""
        assert config.severity_threshold == 0.5
        assert config.risk_threshold == 0.7
        assert config.max_duration_seconds == 600
        assert config.enable_redteam is True
        assert config.enable_verification is True

    def test_config_with_content(self):
        """Test configuration with input content."""
        config = GauntletConfig(
            input_type=InputType.CODE,
            input_content="def hello(): pass",
            severity_threshold=0.3,
        )
        assert config.input_type == InputType.CODE
        assert config.input_content == "def hello(): pass"
        assert config.severity_threshold == 0.3

    def test_config_profiles(self):
        """Test pre-configured profiles."""
        assert QUICK_GAUNTLET.deep_audit_rounds == 2
        assert QUICK_GAUNTLET.enable_verification is False
        assert QUICK_GAUNTLET.max_duration_seconds == 120

        assert THOROUGH_GAUNTLET.deep_audit_rounds == 6
        assert THOROUGH_GAUNTLET.enable_verification is True

        assert CODE_REVIEW_GAUNTLET.input_type == InputType.CODE

        assert POLICY_GAUNTLET.input_type == InputType.POLICY
        assert POLICY_GAUNTLET.severity_threshold == 0.3  # More sensitive


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_severity_level(self):
        """Test severity level classification."""
        critical = Finding(
            finding_id="f1",
            category="attack",
            severity=0.95,
            title="Critical Issue",
            description="Very bad",
        )
        assert critical.severity_level == "CRITICAL"

        high = Finding(
            finding_id="f2",
            category="attack",
            severity=0.75,
            title="High Issue",
            description="Bad",
        )
        assert high.severity_level == "HIGH"

        medium = Finding(
            finding_id="f3",
            category="attack",
            severity=0.5,
            title="Medium Issue",
            description="Concerning",
        )
        assert medium.severity_level == "MEDIUM"

        low = Finding(
            finding_id="f4",
            category="attack",
            severity=0.2,
            title="Low Issue",
            description="Minor",
        )
        assert low.severity_level == "LOW"

    def test_finding_with_mitigation(self):
        """Test finding with mitigation."""
        finding = Finding(
            finding_id="f1",
            category="risk",
            severity=0.7,
            title="Security Risk",
            description="Missing input validation",
            mitigation="Add input sanitization",
            source="RedTeam",
            verified=True,
        )
        assert finding.mitigation == "Add input sanitization"
        assert finding.verified is True


class TestGauntletResult:
    """Tests for GauntletResult dataclass."""

    def test_result_all_findings(self):
        """Test all_findings aggregation."""
        result = GauntletResult(
            gauntlet_id="test-123",
            input_type=InputType.SPEC,
            input_summary="Test spec",
            verdict=Verdict.NEEDS_REVIEW,
            confidence=0.7,
            risk_score=0.5,
            robustness_score=0.8,
            coverage_score=0.6,
            critical_findings=[Finding("c1", "a", 0.95, "Crit", "d")],
            high_findings=[Finding("h1", "a", 0.75, "High", "d"), Finding("h2", "a", 0.8, "High2", "d")],
            medium_findings=[Finding("m1", "a", 0.5, "Med", "d")],
            low_findings=[Finding("l1", "a", 0.2, "Low", "d")],
        )

        assert len(result.all_findings) == 5
        assert result.total_findings == 5
        assert result.all_findings[0].severity_level == "CRITICAL"

    def test_result_checksum(self):
        """Test checksum generation."""
        result = GauntletResult(
            gauntlet_id="test-123",
            input_type=InputType.SPEC,
            input_summary="Test",
            verdict=Verdict.APPROVED,
            confidence=0.9,
            risk_score=0.1,
            robustness_score=0.95,
            coverage_score=0.8,
        )
        checksum = result.checksum
        assert len(checksum) == 16
        # Checksum should be deterministic
        assert result.checksum == checksum

    def test_result_summary(self):
        """Test summary generation."""
        result = GauntletResult(
            gauntlet_id="test-123",
            input_type=InputType.SPEC,
            input_summary="Test specification",
            verdict=Verdict.APPROVED,
            confidence=0.85,
            risk_score=0.2,
            robustness_score=0.9,
            coverage_score=0.75,
            agents_involved=["claude", "gpt4"],
            duration_seconds=45.5,
        )
        summary = result.summary()
        assert "GAUNTLET STRESS-TEST RESULT" in summary
        assert "test-123" in summary
        assert "APPROVED" in summary
        assert "85%" in summary
        assert "claude" in summary


class TestGauntletOrchestrator:
    """Tests for GauntletOrchestrator."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.name = "mock_agent"
        agent.generate = AsyncMock(return_value="Test response")
        return agent

    @pytest.fixture
    def orchestrator(self, mock_agent):
        """Create orchestrator with mock agent."""
        return GauntletOrchestrator([mock_agent])

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.agents) == 1
        assert orchestrator.redteam_mode is not None
        assert orchestrator.prober is not None
        assert orchestrator.risk_assessor is not None

    def test_next_finding_id(self, orchestrator):
        """Test finding ID generation."""
        id1 = orchestrator._next_finding_id()
        id2 = orchestrator._next_finding_id()
        assert id1 == "finding-0001"
        assert id2 == "finding-0002"

    def test_risk_level_to_severity(self, orchestrator):
        """Test risk level conversion."""
        assert orchestrator._risk_level_to_severity(RiskLevel.LOW) == 0.25
        assert orchestrator._risk_level_to_severity(RiskLevel.MEDIUM) == 0.5
        assert orchestrator._risk_level_to_severity(RiskLevel.HIGH) == 0.75
        assert orchestrator._risk_level_to_severity(RiskLevel.CRITICAL) == 0.95

    def test_extract_verifiable_claims(self, orchestrator):
        """Test claim extraction."""
        content = """
        If the user is authenticated then they can access the resource.
        For all users, the password must be at least 8 characters.
        The system must always respond within 100ms.
        """
        claims = orchestrator._extract_verifiable_claims(content)
        assert len(claims) > 0
        assert any("if" in c.lower() or "then" in c.lower() for c in claims)

    def test_determine_verdict_rejected(self, orchestrator):
        """Test verdict determination - rejection."""
        critical = [Finding(f"c{i}", "a", 0.95, f"Crit{i}", "d") for i in range(2)]
        verdict, confidence = orchestrator._determine_verdict(
            critical=critical,
            high=[],
            medium=[],
            risk_score=0.5,
            robustness_score=0.8,
            dissents=[],
        )
        assert verdict == Verdict.REJECTED
        assert confidence >= 0.8

    def test_determine_verdict_needs_review(self, orchestrator):
        """Test verdict determination - needs review."""
        verdict, confidence = orchestrator._determine_verdict(
            critical=[Finding("c1", "a", 0.95, "Crit", "d")],
            high=[],
            medium=[],
            risk_score=0.5,
            robustness_score=0.8,
            dissents=[],
        )
        assert verdict == Verdict.NEEDS_REVIEW

    def test_determine_verdict_approved_with_conditions(self, orchestrator):
        """Test verdict determination - approved with conditions."""
        high = [Finding("h1", "a", 0.75, "High", "d")]
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=high,
            medium=[],
            risk_score=0.3,
            robustness_score=0.9,
            dissents=[],
        )
        assert verdict == Verdict.APPROVED_WITH_CONDITIONS

    def test_determine_verdict_approved(self, orchestrator):
        """Test verdict determination - clean approval."""
        verdict, confidence = orchestrator._determine_verdict(
            critical=[],
            high=[],
            medium=[],
            risk_score=0.1,
            robustness_score=0.95,
            dissents=[],
        )
        assert verdict == Verdict.APPROVED
        assert confidence > 0.8


class TestDecisionReceipt:
    """Tests for DecisionReceipt."""

    @pytest.fixture
    def sample_receipt(self):
        """Create a sample receipt."""
        return DecisionReceipt(
            receipt_id="receipt-test-123",
            gauntlet_id="test-123",
            input_summary="Test specification document",
            input_type="spec",
            verdict="APPROVED_WITH_CONDITIONS",
            confidence=0.75,
            risk_level="MEDIUM",
            risk_score=0.4,
            robustness_score=0.85,
            coverage_score=0.7,
            verification_coverage=0.5,
            findings=[
                ReceiptFinding(
                    id="f1",
                    severity="HIGH",
                    category="attack",
                    title="Input Validation Missing",
                    description="User input not validated",
                    mitigation="Add input sanitization",
                    source="RedTeam",
                ),
                ReceiptFinding(
                    id="f2",
                    severity="MEDIUM",
                    category="risk",
                    title="Rate Limiting",
                    description="No rate limiting on API",
                    source="DeepAudit",
                ),
            ],
            critical_count=0,
            high_count=1,
            medium_count=1,
            low_count=0,
            mitigations=["Add input sanitization", "Implement rate limiting"],
            agents_involved=["claude", "gpt4"],
            duration_seconds=120.5,
        )

    def test_receipt_checksum(self, sample_receipt):
        """Test checksum computation."""
        checksum = sample_receipt.checksum
        assert len(checksum) == 16
        assert sample_receipt.verify_integrity()

    def test_receipt_to_dict(self, sample_receipt):
        """Test dict serialization."""
        data = sample_receipt.to_dict()
        assert data["receipt_id"] == "receipt-test-123"
        assert data["verdict"] == "APPROVED_WITH_CONDITIONS"
        assert len(data["findings"]) == 2
        assert data["checksum"] == sample_receipt.checksum

    def test_receipt_to_json(self, sample_receipt):
        """Test JSON serialization."""
        json_str = sample_receipt.to_json()
        data = json.loads(json_str)
        assert data["receipt_id"] == "receipt-test-123"
        assert data["confidence"] == 0.75

    def test_receipt_to_markdown(self, sample_receipt):
        """Test Markdown export."""
        md = sample_receipt.to_markdown()
        assert "# Decision Receipt" in md
        assert "APPROVED_WITH_CONDITIONS" in md
        assert "Input Validation Missing" in md
        assert "Robustness" in md
        assert "checksum" in md.lower()

    def test_receipt_to_html(self, sample_receipt):
        """Test HTML export."""
        html = sample_receipt.to_html()
        assert "<!DOCTYPE html>" in html
        assert "Decision Receipt" in html
        assert "APPROVED_WITH_CONDITIONS" in html
        assert sample_receipt.checksum in html

    def test_receipt_from_json(self, sample_receipt):
        """Test loading from JSON."""
        json_str = sample_receipt.to_json()
        loaded = DecisionReceipt.from_json(json_str)
        assert loaded.receipt_id == sample_receipt.receipt_id
        assert loaded.verdict == sample_receipt.verdict
        assert len(loaded.findings) == len(sample_receipt.findings)

    def test_receipt_integrity_tamper_detection(self, sample_receipt):
        """Test integrity verification after tampering."""
        assert sample_receipt.verify_integrity()
        # Tamper with the verdict
        sample_receipt.verdict = "APPROVED"
        # Checksum should not match after tampering
        assert not sample_receipt.verify_integrity()


class TestDecisionReceiptGenerator:
    """Tests for DecisionReceiptGenerator."""

    def test_generate_from_gauntlet_result(self):
        """Test generating receipt from GauntletResult."""
        result = GauntletResult(
            gauntlet_id="test-456",
            input_type=InputType.ARCHITECTURE,
            input_summary="System architecture document",
            verdict=Verdict.NEEDS_REVIEW,
            confidence=0.65,
            risk_score=0.55,
            robustness_score=0.7,
            coverage_score=0.6,
            verification_coverage=0.3,
            critical_findings=[
                Finding("c1", "audit", 0.95, "Critical Bug", "Severe issue"),
            ],
            high_findings=[
                Finding("h1", "attack", 0.75, "Security Gap", "Missing auth", mitigation="Add auth"),
            ],
            medium_findings=[],
            low_findings=[],
            dissenting_views=[
                DissentRecord(
                    agent="gpt4",
                    claim_id="c1",
                    dissent_type="partial",
                    reasons=["Not as severe as stated"],
                    severity=0.5,
                ),
            ],
            unresolved_tensions=[
                UnresolvedTension(
                    tension_id="t1",
                    description="Performance vs Security tradeoff",
                    agents_involved=["claude", "gpt4"],
                    options=["Optimize for speed", "Maximize security"],
                    impact="Architecture choice",
                ),
            ],
            verified_claims=[
                VerifiedClaim(
                    claim="Auth required for all endpoints",
                    verified=True,
                    verification_method="z3",
                    proof_hash="abc123",
                ),
            ],
            unverified_claims=["Response time < 100ms"],
            agents_involved=["claude", "gpt4"],
            duration_seconds=180.0,
        )

        receipt = generate_decision_receipt(result)

        assert receipt.gauntlet_id == "test-456"
        assert receipt.verdict == "NEEDS_REVIEW"
        assert receipt.risk_level == "MEDIUM"  # 0.55 risk score
        assert receipt.critical_count == 1
        assert receipt.high_count == 1
        assert len(receipt.findings) == 2
        assert len(receipt.dissenting_views) == 1
        assert len(receipt.unresolved_tensions) == 1
        assert len(receipt.verified_claims) == 1
        assert len(receipt.mitigations) == 1  # "Add auth"


class TestGauntletIntegration:
    """Integration tests for the Gauntlet workflow."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for integration tests."""
        agent = MagicMock()
        agent.name = "integration_test_agent"
        agent.generate = AsyncMock(return_value="Integration test response")
        return agent

    @pytest.mark.asyncio
    async def test_run_gauntlet_minimal(self, mock_agent):
        """Test minimal gauntlet run."""
        with patch("aragora.modes.gauntlet.GauntletOrchestrator._run_redteam") as mock_redteam, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_probing") as mock_probe, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_deep_audit") as mock_audit, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_verification") as mock_verify:

            # Mock returns
            mock_redteam.return_value = None
            mock_probe.return_value = None
            mock_audit.return_value = None
            mock_verify.return_value = ([], [])

            result = await run_gauntlet(
                input_content="Simple test specification",
                agents=[mock_agent],
                input_type=InputType.SPEC,
            )

            assert result is not None
            assert result.verdict in [Verdict.APPROVED, Verdict.APPROVED_WITH_CONDITIONS, Verdict.NEEDS_REVIEW, Verdict.REJECTED]
            assert result.gauntlet_id.startswith("gauntlet-")
            assert result.input_type == InputType.SPEC

    @pytest.mark.asyncio
    async def test_full_gauntlet_workflow(self, mock_agent, tmp_path):
        """Test complete gauntlet workflow with receipt generation."""
        with patch("aragora.modes.gauntlet.GauntletOrchestrator._run_redteam") as mock_redteam, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_probing") as mock_probe, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_deep_audit") as mock_audit, \
             patch("aragora.modes.gauntlet.GauntletOrchestrator._run_verification") as mock_verify:

            # Mock returns
            mock_redteam.return_value = None
            mock_probe.return_value = None
            mock_audit.return_value = None
            mock_verify.return_value = ([], ["Unverified claim"])

            # Run gauntlet
            result = await run_gauntlet(
                input_content="Test architecture document for stress testing",
                agents=[mock_agent],
                input_type=InputType.ARCHITECTURE,
            )

            # Generate receipt
            receipt = generate_decision_receipt(result)

            # Save to all formats
            for fmt in ["json", "md", "html"]:
                output_path = tmp_path / f"receipt.{fmt}"
                saved_path = receipt.save(output_path, format=fmt)
                assert saved_path.exists()
                content = saved_path.read_text()
                assert len(content) > 0

            # Verify we can load JSON receipt
            json_path = tmp_path / "receipt.json"
            loaded = DecisionReceipt.load(json_path)
            assert loaded.gauntlet_id == result.gauntlet_id
            assert loaded.verify_integrity()

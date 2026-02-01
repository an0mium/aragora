"""
Tests for Audit-Consensus Adapter.

Tests the FindingVerifier which connects the document audit system
to the debate consensus verification system for multi-agent
verification of audit findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.audit.consensus_adapter import (
    FindingVerifier,
    VerificationConfig,
    VerificationResult,
    verify_finding,
)
from aragora.debate.consensus import (
    ConsensusVote,
    DissentRecord,
    VoteType,
)


# ===========================================================================
# Mock Data
# ===========================================================================


class MockSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MockAuditType(str, Enum):
    SECURITY = "security"
    CODE_QUALITY = "code_quality"
    COMPLIANCE = "compliance"


@dataclass
class MockAuditFinding:
    """Mock audit finding for testing."""

    id: str = "finding_001"
    title: str = "SQL Injection Risk"
    description: str = "User input not sanitized in database query"
    severity: MockSeverity = MockSeverity.HIGH
    category: str = "security"
    audit_type: MockAuditType = MockAuditType.SECURITY
    confidence: float = 0.80
    document_id: str = "doc_001"
    evidence_text: str = "cursor.execute(f'SELECT * FROM users WHERE name={user_input}')"
    evidence_location: str = "Line 42"
    found_by: str = "security_scanner"
    recommendation: str = "Use parameterized queries"


# ===========================================================================
# Tests: VerificationConfig
# ===========================================================================


class TestVerificationConfig:
    """Tests for VerificationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VerificationConfig()

        assert config.agents == ["anthropic-api", "openai-api"]
        assert config.consensus_threshold == 0.8
        assert config.confidence_threshold == 0.7
        assert config.max_rounds == 2
        assert config.verify_severity is True
        assert config.verify_validity is True
        assert config.verify_recommendation is False
        assert config.include_evidence is True
        assert config.include_document_context is True
        assert config.max_context_tokens == 4000

    def test_custom_config(self):
        """Test custom configuration."""
        config = VerificationConfig(
            agents=["agent_a", "agent_b", "agent_c"],
            consensus_threshold=0.9,
            confidence_threshold=0.8,
            max_rounds=3,
        )

        assert config.agents == ["agent_a", "agent_b", "agent_c"]
        assert config.consensus_threshold == 0.9
        assert config.confidence_threshold == 0.8
        assert config.max_rounds == 3


# ===========================================================================
# Tests: VerificationResult
# ===========================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_result(self):
        """Test creating a verification result."""
        proof = MagicMock()
        result = VerificationResult(
            finding_id="f_001",
            verified=True,
            consensus_proof=proof,
            original_severity="high",
        )

        assert result.finding_id == "f_001"
        assert result.verified is True
        assert result.original_severity == "high"
        assert result.verified_severity is None
        assert result.severity_changed is False
        assert result.verification_notes == []
        assert result.duration_ms == 0

    def test_result_with_severity_change(self):
        """Test result with severity change."""
        proof = MagicMock()
        result = VerificationResult(
            finding_id="f_001",
            verified=True,
            consensus_proof=proof,
            original_severity="high",
            verified_severity="medium",
            severity_changed=True,
            verification_notes=["Agent suggests lower severity"],
        )

        assert result.severity_changed is True
        assert result.verified_severity == "medium"
        assert len(result.verification_notes) == 1


# ===========================================================================
# Tests: FindingVerifier Initialization
# ===========================================================================


class TestFindingVerifierInit:
    """Tests for FindingVerifier initialization."""

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = VerificationConfig(
            agents=["agent_a"],
            consensus_threshold=0.9,
        )
        verifier = FindingVerifier(config=config)

        assert verifier.config.agents == ["agent_a"]
        assert verifier.config.consensus_threshold == 0.9

    def test_init_with_agents_shorthand(self):
        """Test initialization with agents shorthand."""
        verifier = FindingVerifier(agents=["claude", "gpt"])

        assert verifier.config.agents == ["claude", "gpt"]

    def test_init_default(self):
        """Test default initialization."""
        verifier = FindingVerifier()

        assert verifier.config.agents == ["anthropic-api", "openai-api"]

    def test_config_takes_precedence_over_agents(self):
        """Test that config parameter takes precedence."""
        config = VerificationConfig(agents=["from_config"])
        verifier = FindingVerifier(config=config, agents=["from_shorthand"])

        assert verifier.config.agents == ["from_config"]


# ===========================================================================
# Tests: FindingVerifier._finding_to_claim
# ===========================================================================


class TestFindingToClaim:
    """Tests for converting findings to claims."""

    def test_finding_to_claim(self):
        """Test converting a finding to a claim."""
        verifier = FindingVerifier()
        finding = MockAuditFinding()

        claim = verifier._finding_to_claim(finding, "claim_001")

        assert claim.claim_id == "claim_001"
        assert "HIGH" in claim.statement
        assert "SQL Injection" in claim.statement
        assert claim.author == "audit_system"
        assert claim.confidence == 0.80

    def test_finding_to_claim_with_evidence(self):
        """Test that finding evidence is converted to claim evidence."""
        verifier = FindingVerifier()
        finding = MockAuditFinding(
            evidence_text="Some SQL injection evidence",
            evidence_location="File.py:42",
        )

        claim = verifier._finding_to_claim(finding, "claim_001")

        assert len(claim.supporting_evidence) == 1
        assert claim.supporting_evidence[0].content == "Some SQL injection evidence"
        assert claim.supporting_evidence[0].strength == 0.80

    def test_finding_to_claim_no_evidence(self):
        """Test claim creation when finding has no evidence text."""
        verifier = FindingVerifier()
        finding = MockAuditFinding(evidence_text="")

        claim = verifier._finding_to_claim(finding, "claim_001")

        assert len(claim.supporting_evidence) == 0


# ===========================================================================
# Tests: FindingVerifier._build_verification_prompt
# ===========================================================================


class TestBuildVerificationPrompt:
    """Tests for building verification prompts."""

    def test_prompt_includes_finding_details(self):
        """Test prompt includes finding information."""
        verifier = FindingVerifier()
        finding = MockAuditFinding()

        prompt = verifier._build_verification_prompt(finding)

        assert "SQL Injection Risk" in prompt
        assert "high" in prompt
        assert "security" in prompt
        assert "80%" in prompt

    def test_prompt_includes_evidence(self):
        """Test prompt includes evidence when available."""
        verifier = FindingVerifier()
        finding = MockAuditFinding(evidence_text="cursor.execute(f'...')")

        prompt = verifier._build_verification_prompt(finding)

        assert "cursor.execute" in prompt

    def test_prompt_includes_recommendation(self):
        """Test prompt includes recommendation."""
        verifier = FindingVerifier()
        finding = MockAuditFinding(recommendation="Use parameterized queries")

        prompt = verifier._build_verification_prompt(finding)

        assert "parameterized queries" in prompt

    def test_prompt_includes_document_context(self):
        """Test prompt includes document context when provided."""
        verifier = FindingVerifier()
        finding = MockAuditFinding()

        prompt = verifier._build_verification_prompt(
            finding,
            document_context="Some relevant document context here.",
        )

        assert "Some relevant document context here" in prompt

    def test_prompt_truncates_context(self):
        """Test prompt truncates document context to max tokens."""
        config = VerificationConfig(max_context_tokens=20)
        verifier = FindingVerifier(config=config)
        finding = MockAuditFinding()

        long_context = "A" * 1000
        prompt = verifier._build_verification_prompt(finding, long_context)

        # The context in the prompt should be truncated
        assert "A" * 1000 not in prompt


# ===========================================================================
# Tests: FindingVerifier._parse_verification_response
# ===========================================================================


class TestParseVerificationResponse:
    """Tests for parsing agent verification responses."""

    @pytest.fixture
    def verifier(self):
        return FindingVerifier()

    def test_parse_agree_response(self, verifier):
        """Test parsing an agreement response."""
        response = "VALID. I agree with this finding. Confidence: 85%. The SQL injection is real."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert vote == VoteType.AGREE
        assert confidence == 0.85
        assert len(reasoning) > 0

    def test_parse_disagree_response(self, verifier):
        """Test parsing a disagreement response."""
        response = "INVALID. This is a false positive. Confidence: 90%."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert vote == VoteType.DISAGREE

    def test_parse_disagree_keyword(self, verifier):
        """Test parsing with explicit disagree keyword."""
        response = "I disagree with the severity. Confidence: 70%."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert vote == VoteType.DISAGREE

    def test_parse_conditional_response(self, verifier):
        """Test parsing a conditional response."""
        response = "The finding is partially valid. Conditional on additional review. 60%."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert vote == VoteType.CONDITIONAL
        assert confidence == 0.60

    def test_parse_default_confidence(self, verifier):
        """Test default confidence when no percentage found."""
        response = "VALID. I agree with this finding."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert confidence == 0.7  # Default

    def test_parse_false_positive(self, verifier):
        """Test parsing a false positive detection."""
        response = "This appears to be a false positive. The code is actually safe."
        vote, confidence, reasoning = verifier._parse_verification_response(response, "agent_1")

        assert vote == VoteType.DISAGREE


# ===========================================================================
# Tests: FindingVerifier._extract_alternative
# ===========================================================================


class TestExtractAlternative:
    """Tests for extracting alternative severity suggestions."""

    @pytest.fixture
    def verifier(self):
        return FindingVerifier()

    def test_extract_should_be(self, verifier):
        """Test extraction with 'should be' keyword."""
        response = "The severity should be low instead of the current rating."
        result = verifier._extract_alternative(response)

        assert result is not None
        assert "low" in result.lower()

    def test_extract_suggest(self, verifier):
        """Test extraction with 'suggest' keyword."""
        response = "I suggest the severity level is low."
        result = verifier._extract_alternative(response)

        assert result is not None
        assert "low" in result.lower()

    def test_extract_no_alternative(self, verifier):
        """Test when no alternative is suggested."""
        response = "The finding is valid and the severity is correct."
        result = verifier._extract_alternative(response)

        assert result is None


# ===========================================================================
# Tests: FindingVerifier._mock_verification
# ===========================================================================


class TestMockVerification:
    """Tests for mock verification fallback."""

    @pytest.fixture
    def verifier(self):
        return FindingVerifier()

    def test_mock_high_confidence(self, verifier):
        """Test mock verification with high confidence finding."""
        finding = MockAuditFinding(confidence=0.85)
        votes, dissents = verifier._mock_verification(finding)

        assert len(votes) == 2
        assert all(v.vote == VoteType.AGREE for v in votes)
        assert len(dissents) == 0

    def test_mock_low_confidence(self, verifier):
        """Test mock verification with low confidence finding."""
        finding = MockAuditFinding(confidence=0.40)
        votes, dissents = verifier._mock_verification(finding)

        assert len(votes) == 2
        assert len(dissents) >= 1  # Should generate a dissent for low confidence

    def test_mock_medium_confidence(self, verifier):
        """Test mock verification with medium confidence."""
        finding = MockAuditFinding(confidence=0.55)
        votes, dissents = verifier._mock_verification(finding)

        assert len(votes) == 2
        # Second agent should AGREE (confidence > 0.5)
        # First agent may be CONDITIONAL (confidence <= 0.6)


# ===========================================================================
# Tests: FindingVerifier._build_reasoning_summary
# ===========================================================================


class TestBuildReasoningSummary:
    """Tests for building reasoning summaries."""

    @pytest.fixture
    def verifier(self):
        return FindingVerifier()

    def test_summary_with_votes(self, verifier):
        """Test summary generation with votes."""
        votes = [
            ConsensusVote(
                agent="agent_1",
                vote=VoteType.AGREE,
                confidence=0.9,
                reasoning="Finding is valid.",
            ),
            ConsensusVote(
                agent="agent_2",
                vote=VoteType.AGREE,
                confidence=0.85,
                reasoning="Confirmed the issue.",
            ),
        ]

        summary = verifier._build_reasoning_summary(votes, [])

        assert "2 agents agreed" in summary
        assert "0 disagreed" in summary
        assert "agent_1" in summary
        assert "agent_2" in summary

    def test_summary_with_dissents(self, verifier):
        """Test summary with dissenting views."""
        votes = [
            ConsensusVote(
                agent="agent_1",
                vote=VoteType.AGREE,
                confidence=0.9,
                reasoning="Valid finding.",
            ),
            ConsensusVote(
                agent="agent_2",
                vote=VoteType.DISAGREE,
                confidence=0.8,
                reasoning="False positive.",
            ),
        ]

        dissents = [
            DissentRecord(
                agent="agent_2",
                claim_id="claim_001",
                dissent_type="full",
                reasons=["The code is actually safe"],
                severity=0.8,
            ),
        ]

        summary = verifier._build_reasoning_summary(votes, dissents)

        assert "1 agents agreed" in summary
        assert "1 disagreed" in summary
        assert "Dissenting views:" in summary


# ===========================================================================
# Tests: FindingVerifier.verify_finding (integration)
# ===========================================================================


class TestVerifyFinding:
    """Integration tests for verify_finding method."""

    @pytest.mark.asyncio
    async def test_verify_finding_uses_mock_when_agents_unavailable(self):
        """Test verification falls back to mock when agents are not available."""
        verifier = FindingVerifier()
        finding = MockAuditFinding()

        with patch.object(
            verifier,
            "_run_verification_debate",
            return_value=verifier._mock_verification(finding),
        ):
            result = await verifier.verify_finding(finding)

        assert isinstance(result, VerificationResult)
        assert result.finding_id == "finding_001"
        assert result.original_severity == "high"
        assert result.duration_ms >= 0
        assert result.consensus_proof is not None

    @pytest.mark.asyncio
    async def test_verify_finding_high_confidence(self):
        """Test verification of a high confidence finding."""
        verifier = FindingVerifier()
        finding = MockAuditFinding(confidence=0.90)

        mock_votes = [
            ConsensusVote(agent="agent_1", vote=VoteType.AGREE, confidence=0.9, reasoning="Valid"),
            ConsensusVote(
                agent="agent_2", vote=VoteType.AGREE, confidence=0.85, reasoning="Confirmed"
            ),
        ]

        with patch.object(
            verifier,
            "_run_verification_debate",
            return_value=(mock_votes, []),
        ):
            result = await verifier.verify_finding(finding)

        assert result.verified is True
        assert result.severity_changed is False

    @pytest.mark.asyncio
    async def test_verify_finding_with_severity_dissent(self):
        """Test verification with severity disagreement."""
        verifier = FindingVerifier()
        finding = MockAuditFinding()

        mock_votes = [
            ConsensusVote(agent="agent_1", vote=VoteType.AGREE, confidence=0.9, reasoning="Valid"),
            ConsensusVote(
                agent="agent_2", vote=VoteType.DISAGREE, confidence=0.8, reasoning="Wrong severity"
            ),
        ]

        mock_dissents = [
            DissentRecord(
                agent="agent_2",
                claim_id="claim_001",
                dissent_type="partial",
                reasons=["severity should be medium"],
                alternative_view="The severity should be medium",
                severity=0.8,
            ),
        ]

        with patch.object(
            verifier,
            "_run_verification_debate",
            return_value=(mock_votes, mock_dissents),
        ):
            result = await verifier.verify_finding(finding)

        assert result.severity_changed is True
        assert result.verified_severity == "medium"
        assert len(result.verification_notes) > 0


# ===========================================================================
# Tests: Convenience Function
# ===========================================================================


class TestConvenienceFunction:
    """Tests for module-level verify_finding convenience function."""

    @pytest.mark.asyncio
    async def test_verify_finding_function(self):
        """Test the convenience function."""
        finding = MockAuditFinding()

        with patch.object(
            FindingVerifier,
            "_run_verification_debate",
        ) as mock_debate:
            mock_debate.return_value = (
                [
                    ConsensusVote(
                        agent="mock", vote=VoteType.AGREE, confidence=0.9, reasoning="OK"
                    ),
                ],
                [],
            )

            result = await verify_finding(finding, agents=["mock"])

        assert isinstance(result, VerificationResult)

"""
Tests for aragora.skills.builtin.fact_check module.

Covers:
- FactCheckSkill manifest and initialization
- Claim type analysis
- Verification status handling
- Evidence aggregation
- Support/refute determination
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.skills.base import SkillCapability, SkillContext
from aragora.skills.builtin.fact_check import (
    FactCheckSkill,
    VerificationEvidence,
    VerificationResult,
    VerificationStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> FactCheckSkill:
    """Create a fact check skill for testing."""
    return FactCheckSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123")


# =============================================================================
# VerificationStatus Tests
# =============================================================================


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_verified_status(self):
        """Test verified status value."""
        assert VerificationStatus.VERIFIED.value == "verified"

    def test_refuted_status(self):
        """Test refuted status value."""
        assert VerificationStatus.REFUTED.value == "refuted"

    def test_partially_true_status(self):
        """Test partially true status value."""
        assert VerificationStatus.PARTIALLY_TRUE.value == "partially_true"

    def test_unverifiable_status(self):
        """Test unverifiable status value."""
        assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"

    def test_opinion_status(self):
        """Test opinion status value."""
        assert VerificationStatus.OPINION.value == "opinion"


# =============================================================================
# VerificationEvidence Tests
# =============================================================================


class TestVerificationEvidence:
    """Tests for VerificationEvidence dataclass."""

    def test_create_evidence(self):
        """Test creating verification evidence."""
        evidence = VerificationEvidence(
            source="test_source",
            content="Test content",
            relevance=0.8,
            supports_claim=True,
        )

        assert evidence.source == "test_source"
        assert evidence.content == "Test content"
        assert evidence.relevance == 0.8
        assert evidence.supports_claim is True

    def test_evidence_defaults(self):
        """Test evidence default values."""
        evidence = VerificationEvidence(
            source="test",
            content="content",
            relevance=0.5,
            supports_claim=None,
        )

        assert evidence.url is None
        assert evidence.metadata == {}

    def test_to_dict(self):
        """Test converting evidence to dict."""
        evidence = VerificationEvidence(
            source="web",
            content="Test claim is true",
            relevance=0.9,
            supports_claim=True,
            url="https://example.com",
            metadata={"title": "Test Article"},
        )

        data = evidence.to_dict()

        assert data["source"] == "web"
        assert data["supports_claim"] is True
        assert data["url"] == "https://example.com"


# =============================================================================
# VerificationResult Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_result(self):
        """Test creating verification result."""
        result = VerificationResult(
            claim="Test claim",
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            explanation="Claim is supported by evidence",
        )

        assert result.claim == "Test claim"
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 0.85

    def test_result_defaults(self):
        """Test result default values."""
        result = VerificationResult(
            claim="Test",
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.5,
            explanation="Test explanation",
        )

        assert result.evidence == []
        assert result.related_claims == []

    def test_to_dict(self):
        """Test converting result to dict."""
        result = VerificationResult(
            claim="Test claim",
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            explanation="Verified",
            evidence=[
                VerificationEvidence(
                    source="test",
                    content="content",
                    relevance=0.8,
                    supports_claim=True,
                )
            ],
        )

        data = result.to_dict()

        assert data["claim"] == "Test claim"
        assert data["status"] == "verified"
        assert data["confidence"] == 0.9
        assert len(data["evidence"]) == 1


# =============================================================================
# FactCheckSkill Manifest Tests
# =============================================================================


class TestFactCheckSkillManifest:
    """Tests for FactCheckSkill manifest."""

    def test_manifest_name(self, skill: FactCheckSkill):
        """Test manifest name."""
        assert skill.manifest.name == "fact_check"

    def test_manifest_version(self, skill: FactCheckSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_capabilities(self, skill: FactCheckSkill):
        """Test manifest capabilities."""
        caps = skill.manifest.capabilities
        assert SkillCapability.KNOWLEDGE_QUERY in caps
        assert SkillCapability.EXTERNAL_API in caps

    def test_manifest_input_schema(self, skill: FactCheckSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "claim" in schema
        assert schema["claim"]["type"] == "string"
        assert schema["claim"]["required"] is True

        assert "context" in schema
        assert "sources" in schema
        assert "detailed" in schema

    def test_manifest_debate_compatible(self, skill: FactCheckSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True


# =============================================================================
# FactCheckSkill Initialization Tests
# =============================================================================


class TestFactCheckSkillInit:
    """Tests for FactCheckSkill initialization."""

    def test_default_confidence_threshold(self):
        """Test default confidence threshold."""
        skill = FactCheckSkill()
        assert skill._min_confidence == 0.5

    def test_custom_confidence_threshold(self):
        """Test custom confidence threshold."""
        skill = FactCheckSkill(min_confidence_threshold=0.7)
        assert skill._min_confidence == 0.7

    def test_default_max_evidence(self):
        """Test default max evidence items."""
        skill = FactCheckSkill()
        assert skill._max_evidence == 5

    def test_custom_max_evidence(self):
        """Test custom max evidence items."""
        skill = FactCheckSkill(max_evidence_items=10)
        assert skill._max_evidence == 10


# =============================================================================
# Claim Type Analysis Tests
# =============================================================================


class TestClaimTypeAnalysis:
    """Tests for claim type analysis."""

    def test_detect_opinion_i_think(self, skill: FactCheckSkill):
        """Test detecting opinion with 'I think'."""
        claim = "I think Python is the best language"
        assert skill._analyze_claim_type(claim) == "opinion"

    def test_detect_opinion_should(self, skill: FactCheckSkill):
        """Test detecting opinion with 'should'."""
        claim = "Everyone should learn to code"
        assert skill._analyze_claim_type(claim) == "opinion"

    def test_detect_factual_statistics(self, skill: FactCheckSkill):
        """Test detecting factual claim with statistics."""
        claim = "The unemployment rate is 4.2 percent"
        assert skill._analyze_claim_type(claim) == "factual"

    def test_detect_factual_research(self, skill: FactCheckSkill):
        """Test detecting factual claim with research reference."""
        claim = "Studies show that exercise improves mental health"
        assert skill._analyze_claim_type(claim) == "factual"

    def test_detect_general_claim(self, skill: FactCheckSkill):
        """Test detecting general claim."""
        claim = "The sky appears blue during daytime"
        # Without specific indicators, should be general
        assert skill._analyze_claim_type(claim) in ("general", "factual")


# =============================================================================
# Support Determination Tests
# =============================================================================


class TestSupportDetermination:
    """Tests for support/refute determination."""

    def test_determine_support_positive(self, skill: FactCheckSkill):
        """Test determining support for positive evidence."""
        claim = "Climate change is real"
        evidence = "Confirmed by scientific consensus"

        result = skill._determine_support(claim, evidence)
        assert result is True

    def test_determine_support_negative(self, skill: FactCheckSkill):
        """Test determining support for negative evidence."""
        claim = "The earth is flat"
        evidence = "This claim has been debunked by scientists"

        result = skill._determine_support(claim, evidence)
        assert result is False

    def test_determine_support_neutral(self, skill: FactCheckSkill):
        """Test determining support for neutral evidence."""
        claim = "There are many stars in the universe"
        evidence = "The universe contains billions of galaxies"

        result = skill._determine_support(claim, evidence)
        # Could be True (high overlap) or None (neutral)
        assert result in (True, None)

    def test_determine_support_empty_evidence(self, skill: FactCheckSkill):
        """Test determining support for empty evidence."""
        claim = "Test claim"
        evidence = ""

        result = skill._determine_support(claim, evidence)
        assert result is None


# =============================================================================
# Evidence Analysis Tests
# =============================================================================


class TestEvidenceAnalysis:
    """Tests for evidence analysis."""

    def test_analyze_no_evidence(self, skill: FactCheckSkill):
        """Test analysis with no evidence."""
        result = skill._analyze_evidence("Test claim", [])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.confidence < 0.5

    def test_analyze_supporting_evidence(self, skill: FactCheckSkill):
        """Test analysis with supporting evidence."""
        evidence = [
            VerificationEvidence(
                source="source1",
                content="Supports claim",
                relevance=0.9,
                supports_claim=True,
            ),
            VerificationEvidence(
                source="source2",
                content="Also supports",
                relevance=0.8,
                supports_claim=True,
            ),
        ]

        result = skill._analyze_evidence("Test claim", evidence)

        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence > 0.5

    def test_analyze_refuting_evidence(self, skill: FactCheckSkill):
        """Test analysis with refuting evidence."""
        evidence = [
            VerificationEvidence(
                source="source1",
                content="Refutes claim",
                relevance=0.9,
                supports_claim=False,
            ),
            VerificationEvidence(
                source="source2",
                content="Also refutes",
                relevance=0.8,
                supports_claim=False,
            ),
        ]

        result = skill._analyze_evidence("Test claim", evidence)

        assert result.status == VerificationStatus.REFUTED
        assert result.confidence > 0.5

    def test_analyze_mixed_evidence(self, skill: FactCheckSkill):
        """Test analysis with mixed evidence."""
        evidence = [
            VerificationEvidence(
                source="source1",
                content="Supports claim",
                relevance=0.8,
                supports_claim=True,
            ),
            VerificationEvidence(
                source="source2",
                content="Refutes claim",
                relevance=0.7,
                supports_claim=False,
            ),
        ]

        result = skill._analyze_evidence("Test claim", evidence)

        assert result.status == VerificationStatus.PARTIALLY_TRUE


# =============================================================================
# Full Execution Tests
# =============================================================================


class TestFactCheckExecution:
    """Tests for full skill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_claim(self, skill: FactCheckSkill, context: SkillContext):
        """Test execution fails without claim."""
        result = await skill.execute({}, context)

        assert result.success is False
        assert "claim" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_claim(self, skill: FactCheckSkill, context: SkillContext):
        """Test execution fails with empty claim."""
        result = await skill.execute({"claim": ""}, context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_opinion_claim(self, skill: FactCheckSkill, context: SkillContext):
        """Test execution with opinion claim."""
        result = await skill.execute(
            {"claim": "I think Python is the best programming language"},
            context,
        )

        assert result.success is True
        assert result.data["status"] == "opinion"

    @pytest.mark.asyncio
    async def test_execute_with_mocked_sources(self, skill: FactCheckSkill, context: SkillContext):
        """Test execution with mocked knowledge sources."""
        with patch.object(skill, "_check_knowledge_mound", new_callable=AsyncMock) as mock_km:
            mock_km.return_value = [
                VerificationEvidence(
                    source="km",
                    content="Supporting evidence",
                    relevance=0.8,
                    supports_claim=True,
                )
            ]

            with patch.object(skill, "_check_web_sources", new_callable=AsyncMock) as mock_web:
                mock_web.return_value = []

                with patch.object(
                    skill, "_check_debate_history", new_callable=AsyncMock
                ) as mock_debate:
                    mock_debate.return_value = []

                    result = await skill.execute(
                        {
                            "claim": "The earth orbits the sun",
                            "sources": ["knowledge_mound"],
                        },
                        context,
                    )

        assert result.success is True
        assert "status" in result.data
        assert "confidence" in result.data

    @pytest.mark.asyncio
    async def test_execute_unverifiable_claim(self, skill: FactCheckSkill, context: SkillContext):
        """Test execution with unverifiable claim."""
        with patch.object(skill, "_check_knowledge_mound", new_callable=AsyncMock) as mock_km:
            mock_km.return_value = []

            with patch.object(skill, "_check_web_sources", new_callable=AsyncMock) as mock_web:
                mock_web.return_value = []

                with patch.object(
                    skill, "_check_debate_history", new_callable=AsyncMock
                ) as mock_debate:
                    mock_debate.return_value = []

                    result = await skill.execute(
                        {"claim": "Something obscure happened in 1523"},
                        context,
                    )

        assert result.success is True
        assert result.data["status"] == "unverifiable"


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import fact_check

        assert hasattr(fact_check, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains FactCheckSkill."""
        from aragora.skills.builtin.fact_check import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], FactCheckSkill)

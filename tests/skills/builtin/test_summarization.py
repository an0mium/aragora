"""
Tests for aragora.skills.builtin.summarization module.

Covers:
- SummarizationSkill manifest and initialization
- Extractive summarization
- Bullet point summarization
- TLDR summarization
- Text length validation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.skills.base import SkillCapability, SkillContext, SkillStatus
from aragora.skills.builtin.summarization import SummarizationSkill


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> SummarizationSkill:
    """Create a summarization skill for testing."""
    return SummarizationSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123")


@pytest.fixture
def sample_text() -> str:
    """Sample text for summarization tests."""
    return """
    Artificial intelligence (AI) is transforming industries across the globe.
    Machine learning algorithms are being used to analyze vast amounts of data
    and make predictions that were previously impossible. In healthcare, AI is
    helping doctors diagnose diseases earlier and more accurately. In finance,
    AI-powered systems are detecting fraud and optimizing investment strategies.
    The transportation industry is seeing self-driving vehicles become a reality.
    Manufacturing is being revolutionized by AI-powered robots and predictive
    maintenance systems. However, there are concerns about job displacement and
    the ethical implications of AI decision-making. Researchers and policymakers
    are working to address these challenges while maximizing the benefits of
    this transformative technology. The future of AI promises even more
    remarkable advances in areas like natural language processing, computer
    vision, and robotics.
    """


# =============================================================================
# SummarizationSkill Manifest Tests
# =============================================================================


class TestSummarizationSkillManifest:
    """Tests for SummarizationSkill manifest."""

    def test_manifest_name(self, skill: SummarizationSkill):
        """Test manifest name."""
        assert skill.manifest.name == "summarization"

    def test_manifest_version(self, skill: SummarizationSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_capabilities(self, skill: SummarizationSkill):
        """Test manifest capabilities."""
        caps = skill.manifest.capabilities
        assert SkillCapability.LLM_INFERENCE in caps

    def test_manifest_input_schema(self, skill: SummarizationSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "text" in schema
        assert schema["text"]["type"] == "string"
        assert schema["text"]["required"] is True

        assert "max_length" in schema
        assert "style" in schema
        assert "focus" in schema

    def test_manifest_debate_compatible(self, skill: SummarizationSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True

    def test_manifest_rate_limit(self, skill: SummarizationSkill):
        """Test manifest has rate limit."""
        assert skill.manifest.rate_limit_per_minute == 30


# =============================================================================
# SummarizationSkill Initialization Tests
# =============================================================================


class TestSummarizationSkillInit:
    """Tests for SummarizationSkill initialization."""

    def test_default_max_length(self):
        """Test default max length."""
        skill = SummarizationSkill()
        assert skill._default_max_length == 150

    def test_custom_max_length(self):
        """Test custom max length."""
        skill = SummarizationSkill(default_max_length=200)
        assert skill._default_max_length == 200

    def test_default_style(self):
        """Test default style."""
        skill = SummarizationSkill()
        assert skill._default_style == "abstractive"

    def test_custom_style(self):
        """Test custom style."""
        skill = SummarizationSkill(default_style="bullets")
        assert skill._default_style == "bullets"


# =============================================================================
# SummarizationSkill Execution Tests
# =============================================================================


class TestSummarizationSkillExecution:
    """Tests for SummarizationSkill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_text(self, skill: SummarizationSkill, context: SkillContext):
        """Test execution fails without text."""
        result = await skill.execute({}, context)

        assert result.success is False
        assert "text" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_text(self, skill: SummarizationSkill, context: SkillContext):
        """Test execution fails with empty text."""
        result = await skill.execute({"text": ""}, context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_text_too_short(self, skill: SummarizationSkill, context: SkillContext):
        """Test execution fails with text too short."""
        result = await skill.execute({"text": "Too short"}, context)

        assert result.success is False
        assert "too short" in result.error_message.lower()


# =============================================================================
# Extractive Summarization Tests
# =============================================================================


class TestExtractiveSummarization:
    """Tests for extractive summarization."""

    @pytest.mark.asyncio
    async def test_extractive_summary_returns_list(
        self, skill: SummarizationSkill, sample_text: str
    ):
        """Test extractive summarization returns result."""
        result = await skill._extractive_summarize(sample_text, 50)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_extractive_summary_respects_max_length(
        self, skill: SummarizationSkill, sample_text: str
    ):
        """Test extractive summarization respects max length."""
        result = await skill._extractive_summarize(sample_text, 30)

        words = result.split()
        # Allow some flexibility due to sentence boundaries
        assert len(words) <= 50

    @pytest.mark.asyncio
    async def test_extractive_execute_success(
        self, skill: SummarizationSkill, context: SkillContext, sample_text: str
    ):
        """Test extractive summarization through execute."""
        result = await skill.execute(
            {"text": sample_text, "style": "extractive", "max_length": 50}, context
        )

        assert result.success is True
        assert "summary" in result.data
        assert "compression_ratio" in result.data
        assert result.data["style"] == "extractive"


# =============================================================================
# Bullet Summarization Tests
# =============================================================================


class TestBulletSummarization:
    """Tests for bullet point summarization."""

    @pytest.mark.asyncio
    async def test_bullet_summary_fallback(self, skill: SummarizationSkill, sample_text: str):
        """Test bullet summarization fallback when no LLM."""
        # Mock _get_llm to return None
        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None

            result = await skill._bullet_summarize(sample_text, 100, None)

        assert isinstance(result, str)
        # Should have bullet points
        assert "-" in result or "•" in result

    @pytest.mark.asyncio
    async def test_bullet_execute_success(
        self, skill: SummarizationSkill, context: SkillContext, sample_text: str
    ):
        """Test bullet summarization through execute with mock LLM."""
        mock_llm = MagicMock()

        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_llm
            with patch.object(skill, "_call_llm", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = "- Point 1\n- Point 2\n- Point 3"

                result = await skill.execute({"text": sample_text, "style": "bullets"}, context)

        assert result.success is True
        assert "summary" in result.data


# =============================================================================
# TLDR Summarization Tests
# =============================================================================


class TestTLDRSummarization:
    """Tests for TLDR summarization."""

    @pytest.mark.asyncio
    async def test_tldr_fallback(self, skill: SummarizationSkill, sample_text: str):
        """Test TLDR fallback when no LLM."""
        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None

            result = await skill._tldr_summarize(sample_text)

        assert isinstance(result, str)
        # Should return first sentence or limited text
        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_tldr_execute_success(
        self, skill: SummarizationSkill, context: SkillContext, sample_text: str
    ):
        """Test TLDR summarization through execute."""
        mock_llm = MagicMock()

        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_llm
            with patch.object(skill, "_call_llm", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = (
                    "AI is transforming industries with benefits and challenges."
                )

                result = await skill.execute({"text": sample_text, "style": "tldr"}, context)

        assert result.success is True
        assert result.data["style"] == "tldr"


# =============================================================================
# Result Metrics Tests
# =============================================================================


class TestResultMetrics:
    """Tests for result metrics."""

    @pytest.mark.asyncio
    async def test_compression_ratio(
        self, skill: SummarizationSkill, context: SkillContext, sample_text: str
    ):
        """Test compression ratio calculation."""
        result = await skill.execute(
            {"text": sample_text, "style": "extractive", "max_length": 30}, context
        )

        assert result.success is True
        ratio = result.data["compression_ratio"]
        assert 0 <= ratio <= 1

    @pytest.mark.asyncio
    async def test_word_counts(
        self, skill: SummarizationSkill, context: SkillContext, sample_text: str
    ):
        """Test word count metrics."""
        result = await skill.execute({"text": sample_text, "style": "extractive"}, context)

        assert result.success is True
        assert "original_words" in result.data
        assert "summary_words" in result.data
        assert result.data["summary_words"] <= result.data["original_words"]


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import summarization

        assert hasattr(summarization, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains SummarizationSkill."""
        from aragora.skills.builtin.summarization import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], SummarizationSkill)

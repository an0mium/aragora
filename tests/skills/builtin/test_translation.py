"""
Tests for aragora.skills.builtin.translation module.

Covers:
- TranslationSkill manifest and initialization
- Language normalization
- Language detection (heuristic)
- Translation execution
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.skills.base import SkillCapability, SkillContext
from aragora.skills.builtin.translation import (
    TranslationSkill,
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAME_TO_CODE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> TranslationSkill:
    """Create a translation skill for testing."""
    return TranslationSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123")


# =============================================================================
# SUPPORTED_LANGUAGES Tests
# =============================================================================


class TestSupportedLanguages:
    """Tests for supported languages configuration."""

    def test_english_supported(self):
        """Test English is supported."""
        assert "en" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"] == "English"

    def test_spanish_supported(self):
        """Test Spanish is supported."""
        assert "es" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["es"] == "Spanish"

    def test_chinese_supported(self):
        """Test Chinese is supported."""
        assert "zh" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["zh"] == "Chinese"

    def test_japanese_supported(self):
        """Test Japanese is supported."""
        assert "ja" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["ja"] == "Japanese"

    def test_name_to_code_mapping(self):
        """Test name to code mapping."""
        assert LANGUAGE_NAME_TO_CODE["english"] == "en"
        assert LANGUAGE_NAME_TO_CODE["spanish"] == "es"
        assert LANGUAGE_NAME_TO_CODE["french"] == "fr"


# =============================================================================
# TranslationSkill Manifest Tests
# =============================================================================


class TestTranslationSkillManifest:
    """Tests for TranslationSkill manifest."""

    def test_manifest_name(self, skill: TranslationSkill):
        """Test manifest name."""
        assert skill.manifest.name == "translation"

    def test_manifest_version(self, skill: TranslationSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_capabilities(self, skill: TranslationSkill):
        """Test manifest capabilities."""
        caps = skill.manifest.capabilities
        assert SkillCapability.LLM_INFERENCE in caps

    def test_manifest_input_schema(self, skill: TranslationSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "text" in schema
        assert schema["text"]["type"] == "string"
        assert schema["text"]["required"] is True

        assert "target_language" in schema
        assert "source_language" in schema
        assert "formality" in schema

    def test_manifest_debate_compatible(self, skill: TranslationSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True


# =============================================================================
# TranslationSkill Initialization Tests
# =============================================================================


class TestTranslationSkillInit:
    """Tests for TranslationSkill initialization."""

    def test_default_target_language(self):
        """Test default target language."""
        skill = TranslationSkill()
        assert skill._default_target == "en"

    def test_custom_target_language(self):
        """Test custom target language."""
        skill = TranslationSkill(default_target_language="es")
        assert skill._default_target == "es"


# =============================================================================
# Language Normalization Tests
# =============================================================================


class TestLanguageNormalization:
    """Tests for language normalization."""

    def test_normalize_code(self, skill: TranslationSkill):
        """Test normalizing language code."""
        assert skill._normalize_language("en") == "en"
        assert skill._normalize_language("ES") == "es"
        assert skill._normalize_language("FR") == "fr"

    def test_normalize_full_name(self, skill: TranslationSkill):
        """Test normalizing full language name."""
        assert skill._normalize_language("English") == "en"
        assert skill._normalize_language("Spanish") == "es"
        assert skill._normalize_language("FRENCH") == "fr"

    def test_normalize_partial_match(self, skill: TranslationSkill):
        """Test normalizing partial language name."""
        # Should find partial matches
        result = skill._normalize_language("span")
        assert result == "es"  # Spanish

    def test_normalize_unknown(self, skill: TranslationSkill):
        """Test normalizing unknown language."""
        # Should return as-is
        result = skill._normalize_language("unknown_lang")
        assert result == "unknown_lang"


# =============================================================================
# Heuristic Language Detection Tests
# =============================================================================


class TestHeuristicLanguageDetection:
    """Tests for heuristic language detection."""

    def test_detect_chinese(self, skill: TranslationSkill):
        """Test detecting Chinese text."""
        text = "你好世界"
        result = skill._heuristic_detect(text)
        assert result == "zh"

    def test_detect_japanese(self, skill: TranslationSkill):
        """Test detecting Japanese text."""
        text = "こんにちは"
        result = skill._heuristic_detect(text)
        assert result == "ja"

    def test_detect_korean(self, skill: TranslationSkill):
        """Test detecting Korean text."""
        text = "안녕하세요"
        result = skill._heuristic_detect(text)
        assert result == "ko"

    def test_detect_arabic(self, skill: TranslationSkill):
        """Test detecting Arabic text."""
        text = "مرحبا"
        result = skill._heuristic_detect(text)
        assert result == "ar"

    def test_detect_russian(self, skill: TranslationSkill):
        """Test detecting Russian text."""
        text = "Привет мир"
        result = skill._heuristic_detect(text)
        assert result == "ru"

    def test_detect_spanish(self, skill: TranslationSkill):
        """Test detecting Spanish text."""
        # Use more Spanish indicator words: el, la, de, en, que, es
        text = "El gato que está en la casa es de mi amigo"
        result = skill._heuristic_detect(text)
        assert result == "es"

    def test_detect_french(self, skill: TranslationSkill):
        """Test detecting French text."""
        text = "Bonjour, le chat est dans la maison"
        result = skill._heuristic_detect(text)
        assert result == "fr"

    def test_detect_german(self, skill: TranslationSkill):
        """Test detecting German text."""
        text = "Guten Tag, der Hund ist nicht hier"
        result = skill._heuristic_detect(text)
        assert result == "de"

    def test_detect_english_default(self, skill: TranslationSkill):
        """Test detecting English (default) text."""
        text = "Hello world"
        result = skill._heuristic_detect(text)
        # Should default to English for Latin script without specific patterns
        assert result == "en"


# =============================================================================
# Full Execution Tests
# =============================================================================


class TestTranslationExecution:
    """Tests for full skill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_text(self, skill: TranslationSkill, context: SkillContext):
        """Test execution fails without text."""
        result = await skill.execute({}, context)

        assert result.success is False
        assert "text" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_text(self, skill: TranslationSkill, context: SkillContext):
        """Test execution fails with empty text."""
        result = await skill.execute({"text": ""}, context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_unsupported_language(
        self, skill: TranslationSkill, context: SkillContext
    ):
        """Test execution fails with unsupported target language."""
        result = await skill.execute(
            {"text": "Hello", "target_language": "xyz"},
            context,
        )

        assert result.success is False
        assert "unsupported" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_same_language(self, skill: TranslationSkill, context: SkillContext):
        """Test execution with same source and target language."""
        with patch.object(skill, "_detect_language", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = "en"

            result = await skill.execute(
                {"text": "Hello world", "target_language": "en"},
                context,
            )

        assert result.success is True
        assert "same" in result.data.get("note", "").lower()
        assert result.data["translation"] == "Hello world"

    @pytest.mark.asyncio
    async def test_execute_with_mocked_llm(self, skill: TranslationSkill, context: SkillContext):
        """Test execution with mocked LLM."""
        mock_llm = MagicMock()

        with patch.object(skill, "_detect_language", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = "en"

            with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_llm

                with patch.object(skill, "_call_llm", new_callable=AsyncMock) as mock_call:
                    mock_call.return_value = "Hola mundo"

                    result = await skill.execute(
                        {"text": "Hello world", "target_language": "es"},
                        context,
                    )

        assert result.success is True
        assert result.data["translation"] == "Hola mundo"
        assert result.data["source_language"] == "en"
        assert result.data["target_language"] == "es"

    @pytest.mark.asyncio
    async def test_execute_with_formality(self, skill: TranslationSkill, context: SkillContext):
        """Test execution with formality setting."""
        mock_llm = MagicMock()

        with patch.object(skill, "_detect_language", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = "en"

            with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_llm

                with patch.object(skill, "_call_llm", new_callable=AsyncMock) as mock_call:
                    mock_call.return_value = "Guten Tag"

                    result = await skill.execute(
                        {
                            "text": "Hello",
                            "target_language": "de",
                            "formality": "formal",
                        },
                        context,
                    )

        assert result.success is True
        assert result.data["formality"] == "formal"

    @pytest.mark.asyncio
    async def test_execute_no_llm_fallback(self, skill: TranslationSkill, context: SkillContext):
        """Test execution fallback when no LLM available."""
        with patch.object(skill, "_detect_language", new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = "en"

            with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = None

                result = await skill.execute(
                    {"text": "Hello world", "target_language": "es"},
                    context,
                )

        assert result.success is True
        # Should have a note about translation being unavailable
        assert "unavailable" in result.data["translation"].lower()


# =============================================================================
# Language Detection with LLM Tests
# =============================================================================


class TestLanguageDetectionWithLLM:
    """Tests for language detection with LLM."""

    @pytest.mark.asyncio
    async def test_detect_with_llm(self, skill: TranslationSkill):
        """Test language detection with LLM."""
        mock_llm = MagicMock()

        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_llm

            with patch.object(skill, "_call_llm", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = "es"

                result = await skill._detect_language("Hola mundo")

        assert result == "es"

    @pytest.mark.asyncio
    async def test_detect_llm_fallback(self, skill: TranslationSkill):
        """Test language detection falls back to heuristic."""
        with patch.object(skill, "_get_llm", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            result = await skill._detect_language("Привет мир")

        # Should use heuristic detection
        assert result == "ru"


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import translation

        assert hasattr(translation, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains TranslationSkill."""
        from aragora.skills.builtin.translation import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], TranslationSkill)

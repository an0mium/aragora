"""
Tests for orchestrator output formatting module.

Tests cover:
- format_conclusion function
- translate_conclusions async function
- Delegation to ResultFormatter
- Translation service integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.orchestrator_output import (
    format_conclusion,
    translate_conclusions,
)


# =============================================================================
# Mock DebateResult for testing
# =============================================================================


@dataclass
class MockVote:
    """Mock Vote class for testing."""

    voter: str
    choice: str
    reasoning: str = ""
    confidence: float = 0.8


@dataclass
class MockDebateResult:
    """Mock DebateResult for testing orchestrator_output functions."""

    debate_id: str = "test_debate_123"
    task: str = "Test task"
    final_answer: str = "This is the final answer."
    confidence: float = 0.85
    consensus_reached: bool = True
    rounds_used: int = 3
    status: str = "consensus_reached"
    participants: list[str] = field(default_factory=lambda: ["claude", "gpt4"])
    winner: str | None = None
    votes: list[MockVote] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    belief_cruxes: list[dict[str, Any]] = field(default_factory=list)
    consensus_strength: str = ""
    translations: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Test format_conclusion
# =============================================================================


class TestFormatConclusion:
    """Tests for format_conclusion function."""

    def test_format_conclusion_basic(self):
        """Test basic conclusion formatting."""
        result = MockDebateResult()
        output = format_conclusion(result)

        assert "DEBATE CONCLUSION" in output
        assert "VERDICT" in output
        assert "FINAL ANSWER" in output
        assert "This is the final answer." in output

    def test_format_conclusion_with_consensus(self):
        """Test formatting when consensus is reached."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.90,
        )
        output = format_conclusion(result)

        assert "Consensus: YES" in output
        assert "90%" in output

    def test_format_conclusion_without_consensus(self):
        """Test formatting when no consensus is reached."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.45,
        )
        output = format_conclusion(result)

        assert "Consensus: NO" in output
        assert "45%" in output

    def test_format_conclusion_with_winner(self):
        """Test formatting with a winner specified."""
        result = MockDebateResult(winner="claude")
        output = format_conclusion(result)

        assert "Winner: claude" in output

    def test_format_conclusion_with_consensus_strength(self):
        """Test formatting with consensus strength."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.95,
            consensus_strength="strong",
        )
        output = format_conclusion(result)

        assert "STRONG" in output

    def test_format_conclusion_with_votes(self):
        """Test formatting with vote breakdown."""
        result = MockDebateResult(
            votes=[
                MockVote(voter="claude", choice="proposal_a"),
                MockVote(voter="gpt4", choice="proposal_a"),
            ]
        )
        output = format_conclusion(result)

        assert "VOTE BREAKDOWN" in output
        assert "claude: proposal_a" in output
        assert "gpt4: proposal_a" in output

    def test_format_conclusion_with_dissenting_views(self):
        """Test formatting with dissenting views."""
        result = MockDebateResult(
            dissenting_views=[
                "An alternative approach would be using Redis.",
                "We should consider the performance implications.",
            ]
        )
        output = format_conclusion(result)

        assert "DISSENTING VIEWS" in output
        assert "alternative approach" in output

    def test_format_conclusion_with_cruxes(self):
        """Test formatting with belief cruxes."""
        result = MockDebateResult(
            belief_cruxes=[
                {"claim": "Redis provides better performance", "uncertainty": 0.3},
                {"claim": "Cost savings are significant", "uncertainty": 0.5},
            ]
        )
        output = format_conclusion(result)

        assert "KEY CRUXES" in output
        assert "Redis provides better performance" in output

    def test_format_conclusion_with_translations(self):
        """Test formatting with translations."""
        result = MockDebateResult(
            final_answer="This is the final answer.",
            translations={
                "es": "Esta es la respuesta final.",
                "fr": "C'est la reponse finale.",
            },
        )
        output = format_conclusion(result)

        assert "TRANSLATIONS" in output

    def test_format_conclusion_truncates_long_answer(self):
        """Test that very long answers are truncated."""
        long_answer = "A" * 2000
        result = MockDebateResult(final_answer=long_answer)
        output = format_conclusion(result)

        # Should be truncated with ...
        assert "..." in output
        # Should not contain the full 2000 chars
        assert len(output) < 2000 + 500  # Some buffer for formatting

    def test_format_conclusion_no_final_answer(self):
        """Test formatting when no final answer is provided."""
        result = MockDebateResult(final_answer="")
        output = format_conclusion(result)

        assert "No final answer determined." in output

    def test_format_conclusion_returns_string(self):
        """Test that format_conclusion returns a string."""
        result = MockDebateResult()
        output = format_conclusion(result)

        assert isinstance(output, str)
        assert len(output) > 0


# =============================================================================
# Test translate_conclusions
# =============================================================================


class TestTranslateConclusions:
    """Tests for translate_conclusions async function."""

    @pytest.mark.asyncio
    async def test_translate_no_final_answer(self):
        """Test translation skipped when no final answer."""
        result = MockDebateResult(final_answer="")
        protocol = MagicMock()
        protocol.target_languages = ["es", "fr"]

        await translate_conclusions(result, protocol)

        # No translations should be added
        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_no_target_languages(self):
        """Test translation skipped when no target languages."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = []

        await translate_conclusions(result, protocol)

        # No translations should be added
        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_skips_same_language(self):
        """Test translation skips when target equals source."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["en"]
        protocol.default_language = "en"

        with patch.dict(
            "sys.modules",
            {"aragora.debate.translation": MagicMock()}
        ):
            await translate_conclusions(result, protocol)

        # Should not translate same language - translations dict stays empty or doesn't add "en"
        # The function may still succeed but not add a translation
        assert "en" not in result.translations or result.translations.get("en", "") == ""

    @pytest.mark.asyncio
    async def test_translate_handles_import_error(self):
        """Test graceful handling of missing translation module."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["es"]

        # The import happens inside the function, so we need to mock sys.modules
        # to make the import raise ImportError
        import sys
        original_module = sys.modules.get("aragora.debate.translation")
        try:
            sys.modules["aragora.debate.translation"] = None  # Force import to fail
            # Reload the module to pick up the change - but actually the import
            # happens each time so we just need to make it fail
            # Actually, let's just test that it doesn't crash
            # The function catches ImportError internally
            await translate_conclusions(result, protocol)
        finally:
            if original_module is not None:
                sys.modules["aragora.debate.translation"] = original_module

        # May or may not have translations depending on import behavior
        # The key is it didn't raise

    @pytest.mark.asyncio
    async def test_translate_handles_attribute_error(self):
        """Test graceful handling of attribute errors."""
        result = MockDebateResult(final_answer="Test answer")
        # Protocol without target_languages attribute
        protocol = object()

        # Should not raise, just skip
        await translate_conclusions(result, protocol)
        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_handles_connection_error(self):
        """Test graceful handling of connection errors during translation."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["es"]
        protocol.default_language = "en"

        # Create mock translation module
        mock_translation_module = MagicMock()
        mock_service = AsyncMock()
        mock_service.translate.side_effect = ConnectionError("Network error")
        mock_translation_module.get_translation_service.return_value = mock_service

        mock_en = MagicMock()
        mock_es = MagicMock()
        mock_en.name_english = "English"
        mock_es.name_english = "Spanish"

        def from_code(code):
            if code == "en":
                return mock_en
            if code == "es":
                return mock_es
            return None

        mock_translation_module.Language.from_code = from_code
        mock_translation_module.Language.ENGLISH = mock_en

        with patch.dict(
            "sys.modules",
            {"aragora.debate.translation": mock_translation_module}
        ):
            # Should not raise
            await translate_conclusions(result, protocol)

        # No translations due to error
        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_success_path(self):
        """Test successful translation flow."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["es"]
        protocol.default_language = "en"

        # Create mock translation module
        mock_translation_module = MagicMock()

        mock_result = MagicMock()
        mock_result.confidence = 0.95
        mock_result.translated_text = "Respuesta de prueba"

        mock_service = AsyncMock()
        mock_service.translate.return_value = mock_result
        mock_translation_module.get_translation_service.return_value = mock_service

        mock_en = MagicMock()
        mock_es = MagicMock()
        mock_en.name_english = "English"
        mock_es.name_english = "Spanish"

        def from_code(code):
            if code == "en":
                return mock_en
            if code == "es":
                return mock_es
            return None

        mock_translation_module.Language.from_code = from_code
        mock_translation_module.Language.ENGLISH = mock_en

        with patch.dict(
            "sys.modules",
            {"aragora.debate.translation": mock_translation_module}
        ):
            await translate_conclusions(result, protocol)

        # Translation should be added
        assert "es" in result.translations
        assert result.translations["es"] == "Respuesta de prueba"

    @pytest.mark.asyncio
    async def test_translate_low_confidence_skipped(self):
        """Test that low-confidence translations are skipped."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["es"]
        protocol.default_language = "en"

        # Create mock translation module
        mock_translation_module = MagicMock()

        mock_result = MagicMock()
        mock_result.confidence = 0.3  # Below 0.5 threshold
        mock_result.translated_text = "Respuesta de prueba"

        mock_service = AsyncMock()
        mock_service.translate.return_value = mock_result
        mock_translation_module.get_translation_service.return_value = mock_service

        mock_en = MagicMock()
        mock_es = MagicMock()
        mock_en.name_english = "English"
        mock_es.name_english = "Spanish"

        def from_code(code):
            if code == "en":
                return mock_en
            if code == "es":
                return mock_es
            return None

        mock_translation_module.Language.from_code = from_code
        mock_translation_module.Language.ENGLISH = mock_en

        with patch.dict(
            "sys.modules",
            {"aragora.debate.translation": mock_translation_module}
        ):
            await translate_conclusions(result, protocol)

        # Translation should NOT be added due to low confidence
        assert result.translations == {}

    @pytest.mark.asyncio
    async def test_translate_multiple_languages(self):
        """Test translating to multiple languages."""
        result = MockDebateResult(final_answer="Test answer")
        protocol = MagicMock()
        protocol.target_languages = ["es", "fr", "de"]
        protocol.default_language = "en"

        translations = {
            "es": "Respuesta de prueba",
            "fr": "Reponse de test",
            "de": "Testantwort",
        }

        # Create mock translation module
        mock_translation_module = MagicMock()

        async def mock_translate(text, target_lang, source_lang):
            mock_result = MagicMock()
            mock_result.confidence = 0.9
            # Get language code from mock
            lang_code = getattr(target_lang, "value", "unknown")
            mock_result.translated_text = translations.get(lang_code, text)
            return mock_result

        mock_service = MagicMock()
        mock_service.translate = mock_translate
        mock_translation_module.get_translation_service.return_value = mock_service

        mock_en = MagicMock()
        mock_en.value = "en"
        mock_es = MagicMock()
        mock_es.value = "es"
        mock_fr = MagicMock()
        mock_fr.value = "fr"
        mock_de = MagicMock()
        mock_de.value = "de"

        lang_map = {"en": mock_en, "es": mock_es, "fr": mock_fr, "de": mock_de}

        def from_code(code):
            return lang_map.get(code)

        mock_translation_module.Language.from_code = from_code
        mock_translation_module.Language.ENGLISH = mock_en

        with patch.dict(
            "sys.modules",
            {"aragora.debate.translation": mock_translation_module}
        ):
            await translate_conclusions(result, protocol)

        # All translations should be added
        assert "es" in result.translations
        assert "fr" in result.translations
        assert "de" in result.translations


# =============================================================================
# Test integration with ResultFormatter
# =============================================================================


class TestResultFormatterIntegration:
    """Tests for integration with ResultFormatter."""

    def test_format_conclusion_delegates_to_formatter(self):
        """Test that format_conclusion delegates to ResultFormatter."""
        with patch(
            "aragora.debate.result_formatter.ResultFormatter"
        ) as mock_formatter_cls:
            mock_formatter = MagicMock()
            mock_formatter.format_conclusion.return_value = "Formatted output"
            mock_formatter_cls.return_value = mock_formatter

            result = MockDebateResult()
            output = format_conclusion(result)

            mock_formatter.format_conclusion.assert_called_once_with(result)
            assert output == "Formatted output"


# =============================================================================
# Test module-level exports
# =============================================================================


class TestModuleExports:
    """Tests for module structure."""

    def test_format_conclusion_is_callable(self):
        """Test format_conclusion is a callable function."""
        assert callable(format_conclusion)

    def test_translate_conclusions_is_coroutine(self):
        """Test translate_conclusions is an async function."""
        import asyncio
        import inspect

        assert inspect.iscoroutinefunction(translate_conclusions)

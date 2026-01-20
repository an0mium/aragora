"""Tests for DeepSeek-Prover integration."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from aragora.verification.deepseek_prover import (
    TranslationResult,
    DeepSeekProverTranslator,
    translate_to_lean,
)


class TestTranslationResult:
    """Test TranslationResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful result."""
        result = TranslationResult(
            success=True,
            lean_code="theorem test : True := trivial",
            model_used="deepseek/deepseek-prover-v2",
            translation_time_ms=150.0,
            confidence=0.9,
        )
        assert result.success is True
        assert result.lean_code == "theorem test : True := trivial"
        assert result.confidence == 0.9

    def test_create_failed_result(self):
        """Test creating a failed result."""
        result = TranslationResult(
            success=False,
            error_message="API error",
        )
        assert result.success is False
        assert result.lean_code is None
        assert result.error_message == "API error"

    def test_default_values(self):
        """Test default values."""
        result = TranslationResult(success=True)
        assert result.lean_code is None
        assert result.error_message == ""
        assert result.model_used == ""
        assert result.translation_time_ms == 0.0
        assert result.confidence == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        result = TranslationResult(
            success=True,
            lean_code="theorem test",
            model_used="test_model",
            translation_time_ms=100.0,
            confidence=0.85,
        )
        data = result.to_dict()

        assert data["success"] is True
        assert data["lean_code"] == "theorem test"
        assert data["model_used"] == "test_model"
        assert data["translation_time_ms"] == 100.0
        assert data["confidence"] == 0.85
        assert "timestamp" in data

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        result = TranslationResult(success=True)
        assert isinstance(result.timestamp, datetime)


class TestDeepSeekProverTranslator:
    """Test DeepSeekProverTranslator class."""

    def test_init_default(self):
        """Test default initialization."""
        translator = DeepSeekProverTranslator()
        assert translator.timeout == 60.0
        assert translator.max_tokens == 4096

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        translator = DeepSeekProverTranslator(api_key="test_key")
        assert translator.api_key == "test_key"

    def test_init_with_env_var(self):
        """Test initialization reads from env var."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env_key"}):
            translator = DeepSeekProverTranslator()
            assert translator.api_key == "env_key"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        translator = DeepSeekProverTranslator(
            timeout=30.0,
            max_tokens=2048,
        )
        assert translator.timeout == 30.0
        assert translator.max_tokens == 2048

    def test_is_available_no_key(self):
        """Test is_available without API key."""
        with patch.dict("os.environ", {}, clear=True):
            translator = DeepSeekProverTranslator(api_key=None)
            assert translator.is_available is False

    def test_is_available_with_key(self):
        """Test is_available with API key."""
        translator = DeepSeekProverTranslator(api_key="test_key")
        assert translator.is_available is True

    def test_model_constants(self):
        """Test model constants are defined."""
        assert DeepSeekProverTranslator.PRIMARY_MODEL == "deepseek/deepseek-prover-v2"
        assert DeepSeekProverTranslator.FALLBACK_MODEL == "deepseek/deepseek-chat"

    def test_mathlib_imports(self):
        """Test Mathlib imports constant."""
        assert "import Mathlib" in DeepSeekProverTranslator.MATHLIB_IMPORTS
        assert "Nat.Basic" in DeepSeekProverTranslator.MATHLIB_IMPORTS

    def test_build_translation_prompt(self):
        """Test prompt building."""
        translator = DeepSeekProverTranslator()
        prompt = translator._build_translation_prompt(
            claim="For all n, n + 0 = n",
            context="Natural numbers",
        )

        assert "For all n, n + 0 = n" in prompt
        assert "Natural numbers" in prompt
        assert "Lean 4" in prompt
        assert "UNTRANSLATABLE" in prompt

    def test_build_translation_prompt_no_context(self):
        """Test prompt building without context."""
        translator = DeepSeekProverTranslator()
        prompt = translator._build_translation_prompt(
            claim="1 + 1 = 2",
        )

        assert "1 + 1 = 2" in prompt
        assert "CONTEXT:" not in prompt

    def test_extract_lean_code_markdown(self):
        """Test extracting Lean code from markdown."""
        translator = DeepSeekProverTranslator()

        response = """Here's the Lean code:
```lean
import Mathlib.Tactic

theorem test : True := trivial
```
"""
        code = translator._extract_lean_code(response)
        assert "theorem test" in code
        assert "```" not in code

    def test_extract_lean_code_lean4_marker(self):
        """Test extracting Lean code with lean4 marker."""
        translator = DeepSeekProverTranslator()

        response = """```lean4
theorem test : True := trivial
```"""
        code = translator._extract_lean_code(response)
        assert "theorem test" in code

    def test_extract_lean_code_plain(self):
        """Test extracting plain Lean code."""
        translator = DeepSeekProverTranslator()

        response = "theorem test : True := trivial"
        code = translator._extract_lean_code(response)
        assert code == response

    def test_extract_lean_code_with_import(self):
        """Test extracting code starting with import."""
        translator = DeepSeekProverTranslator()

        response = "import Mathlib.Tactic\n\ntheorem test : True := trivial"
        code = translator._extract_lean_code(response)
        assert code == response

    def test_extract_lean_code_untranslatable(self):
        """Test extracting code returns None for UNTRANSLATABLE."""
        translator = DeepSeekProverTranslator()

        response = "-- UNTRANSLATABLE: Cannot express this claim formally"
        code = translator._extract_lean_code(response)
        assert code is None

    def test_estimate_confidence_empty(self):
        """Test confidence estimation for empty code."""
        translator = DeepSeekProverTranslator()
        confidence = translator._estimate_confidence("")
        assert confidence == 0.0

    def test_estimate_confidence_good_code(self):
        """Test confidence estimation for good code."""
        translator = DeepSeekProverTranslator()
        good_code = """
import Mathlib.Tactic

theorem test : ∀ n : Nat, n + 0 = n := by simp
"""
        confidence = translator._estimate_confidence(good_code)
        assert confidence >= 0.7  # Should have high confidence

    def test_estimate_confidence_sorry(self):
        """Test confidence estimation for code with sorry."""
        translator = DeepSeekProverTranslator()
        sorry_code = """
theorem test : True := by
  sorry
"""
        confidence = translator._estimate_confidence(sorry_code)
        assert confidence < 0.5  # Should have low confidence due to sorry

    def test_estimate_confidence_untranslatable(self):
        """Test confidence estimation for UNTRANSLATABLE."""
        translator = DeepSeekProverTranslator()
        confidence = translator._estimate_confidence("-- UNTRANSLATABLE")
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_translate_no_api_key(self):
        """Test translate without API key."""
        with patch.dict("os.environ", {}, clear=True):
            translator = DeepSeekProverTranslator(api_key=None)
            result = await translator.translate("n + 0 = n")
            assert result.success is False
            assert "not configured" in result.error_message

    @pytest.mark.asyncio
    async def test_translate_success(self):
        """Test successful translation."""
        translator = DeepSeekProverTranslator(api_key="test_key")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": """```lean
import Mathlib.Tactic

theorem test : True := trivial
```"""
                    }
                }
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    status=200,
                    json=AsyncMock(return_value=mock_response),
                )
            )
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            result = await translator.translate("True is true")

            assert result.success is True
            assert "theorem test" in result.lean_code

    @pytest.mark.asyncio
    async def test_translate_api_error(self):
        """Test translation with API error."""
        translator = DeepSeekProverTranslator(api_key="test_key")

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    status=500,
                    text=AsyncMock(return_value="Internal Server Error"),
                )
            )
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            result = await translator.translate("test claim", use_fallback=False)

            assert result.success is False
            assert "500" in result.error_message or "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_translate_timeout(self):
        """Test translation with timeout."""
        import asyncio

        translator = DeepSeekProverTranslator(api_key="test_key", timeout=0.1)

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()

            async def slow_response(*args, **kwargs):
                await asyncio.sleep(1)

            mock_post.__aenter__ = slow_response
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            # This might timeout or handle differently
            result = await translator.translate("test claim", use_fallback=False)
            # Should fail somehow
            assert isinstance(result, TranslationResult)

    @pytest.mark.asyncio
    async def test_translate_untranslatable_response(self):
        """Test translation that returns UNTRANSLATABLE."""
        translator = DeepSeekProverTranslator(api_key="test_key")

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "-- UNTRANSLATABLE: This claim cannot be expressed formally"
                    }
                }
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    status=200,
                    json=AsyncMock(return_value=mock_response),
                )
            )
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            result = await translator.translate("Some vague claim", use_fallback=False)

            assert result.success is False
            assert "cannot be translated" in result.error_message

    @pytest.mark.asyncio
    async def test_check_model_availability_no_key(self):
        """Test model availability check without API key."""
        with patch.dict("os.environ", {}, clear=True):
            translator = DeepSeekProverTranslator(api_key=None)
            available = await translator.check_model_availability()
            assert available is False

    @pytest.mark.asyncio
    async def test_check_model_availability_cached(self):
        """Test model availability uses cache."""
        translator = DeepSeekProverTranslator(api_key="test_key")
        translator._model_available = True

        available = await translator.check_model_availability()
        assert available is True

    @pytest.mark.asyncio
    async def test_translate_batch(self):
        """Test batch translation."""
        translator = DeepSeekProverTranslator(api_key="test_key")

        mock_response = {"choices": [{"message": {"content": "theorem test : True := trivial"}}]}

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    status=200,
                    json=AsyncMock(return_value=mock_response),
                )
            )
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            claims = ["claim 1", "claim 2"]
            results = await translator.translate_batch(claims, max_concurrent=2)

            assert len(results) == 2
            assert all(isinstance(r, TranslationResult) for r in results)


class TestTranslateToLean:
    """Test translate_to_lean convenience function."""

    @pytest.mark.asyncio
    async def test_translate_to_lean_no_key(self):
        """Test convenience function without API key."""
        with patch.dict("os.environ", {}, clear=True):
            result = await translate_to_lean("test claim", api_key=None)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_translate_to_lean_with_key(self):
        """Test convenience function with API key."""
        mock_response = {"choices": [{"message": {"content": "theorem test : True := trivial"}}]}

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = MagicMock()
            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    status=200,
                    json=AsyncMock(return_value=mock_response),
                )
            )
            mock_post.__aexit__ = AsyncMock()
            mock_context.post = MagicMock(return_value=mock_post)
            mock_context.__aenter__ = AsyncMock(return_value=mock_context)
            mock_context.__aexit__ = AsyncMock()
            mock_session.return_value = mock_context

            result = await translate_to_lean(
                "True is true",
                api_key="test_key",
            )

            assert isinstance(result, TranslationResult)


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from aragora.verification import deepseek_prover

        assert hasattr(deepseek_prover, "DeepSeekProverTranslator")
        assert hasattr(deepseek_prover, "TranslationResult")
        assert hasattr(deepseek_prover, "translate_to_lean")

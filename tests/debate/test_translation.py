"""
Tests for debate translation module.

Covers Language enum, TranslationResult, TranslationCache,
heuristic language detection, TranslationService, and
MultilingualDebateManager.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from aragora.debate.translation import (
    Language,
    LanguageDetectionResult,
    LLMTranslationBackend,
    MultilingualDebateConfig,
    MultilingualDebateManager,
    TranslationBackend,
    TranslationCache,
    TranslationResult,
    TranslationService,
    get_multilingual_manager,
    get_translation_service,
)


# ===========================================================================
# Language enum
# ===========================================================================


class TestLanguage:
    """Tests for Language enum."""

    def test_from_code_valid(self):
        assert Language.from_code("en") == Language.ENGLISH
        assert Language.from_code("es") == Language.SPANISH
        assert Language.from_code("ja") == Language.JAPANESE

    def test_from_code_case_insensitive(self):
        assert Language.from_code("EN") == Language.ENGLISH
        assert Language.from_code("Es") == Language.SPANISH

    def test_from_code_strips_and_truncates(self):
        assert Language.from_code("  en  ") == Language.ENGLISH
        assert Language.from_code("eng") == Language.ENGLISH  # truncated to 2 chars

    def test_from_code_invalid_returns_none(self):
        assert Language.from_code("xx") is None
        assert Language.from_code("") is None

    def test_name_english(self):
        assert Language.ENGLISH.name_english == "English"
        assert Language.SPANISH.name_english == "Spanish"
        assert Language.JAPANESE.name_english == "Japanese"
        assert Language.ARABIC.name_english == "Arabic"

    def test_value_is_iso_code(self):
        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE.value == "zh"
        assert Language.KOREAN.value == "ko"


# ===========================================================================
# TranslationResult
# ===========================================================================


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_to_dict_basic(self):
        result = TranslationResult(
            original_text="Hello",
            translated_text="Hola",
            source_language=Language.ENGLISH,
            target_language=Language.SPANISH,
            confidence=0.95,
            backend="llm",
        )
        d = result.to_dict()
        assert d["original_text"] == "Hello"
        assert d["translated_text"] == "Hola"
        assert d["source_language"] == "en"
        assert d["target_language"] == "es"
        assert d["confidence"] == 0.95
        assert d["backend"] == "llm"

    def test_to_dict_truncates_long_text(self):
        long_text = "x" * 200
        result = TranslationResult(
            original_text=long_text,
            translated_text=long_text,
            source_language=Language.ENGLISH,
            target_language=Language.FRENCH,
        )
        d = result.to_dict()
        assert len(d["original_text"]) == 103  # 100 + "..."
        assert d["original_text"].endswith("...")

    def test_defaults(self):
        result = TranslationResult(
            original_text="Hi",
            translated_text="Salut",
            source_language=Language.ENGLISH,
            target_language=Language.FRENCH,
        )
        assert result.confidence == 1.0
        assert result.backend == "unknown"
        assert result.cached is False
        assert result.translation_time_ms == 0.0
        assert result.metadata == {}


# ===========================================================================
# LanguageDetectionResult
# ===========================================================================


class TestLanguageDetectionResult:
    """Tests for LanguageDetectionResult."""

    def test_to_dict(self):
        result = LanguageDetectionResult(
            text="Hola",
            detected_language=Language.SPANISH,
            confidence=0.95,
            alternatives=[(Language.PORTUGUESE, 0.3), (Language.ITALIAN, 0.1)],
        )
        d = result.to_dict()
        assert d["detected_language"] == "es"
        assert d["confidence"] == 0.95
        assert d["alternatives"] == [("pt", 0.3), ("it", 0.1)]


# ===========================================================================
# TranslationCache
# ===========================================================================


class TestTranslationCache:
    """Tests for TranslationCache."""

    def _make_result(self, text="Hello", source=Language.ENGLISH, target=Language.SPANISH):
        return TranslationResult(
            original_text=text,
            translated_text="Hola",
            source_language=source,
            target_language=target,
            confidence=0.95,
        )

    def test_put_and_get(self):
        cache = TranslationCache()
        result = self._make_result()
        cache.put(result)

        cached = cache.get("Hello", Language.ENGLISH, Language.SPANISH)
        assert cached is not None
        assert cached.translated_text == "Hola"
        assert cached.cached is True

    def test_cache_miss(self):
        cache = TranslationCache()
        assert cache.get("Hello", Language.ENGLISH, Language.SPANISH) is None

    def test_ttl_expiry(self):
        cache = TranslationCache(ttl_seconds=0.01)
        result = self._make_result()
        cache.put(result)

        time.sleep(0.02)
        assert cache.get("Hello", Language.ENGLISH, Language.SPANISH) is None

    def test_lru_eviction(self):
        cache = TranslationCache(max_entries=2)
        r1 = self._make_result("first")
        r2 = self._make_result("second")
        r3 = self._make_result("third")

        cache.put(r1)
        cache.put(r2)
        cache.put(r3)

        # First entry should be evicted
        assert cache.get("first", Language.ENGLISH, Language.SPANISH) is None
        assert cache.get("second", Language.ENGLISH, Language.SPANISH) is not None
        assert cache.get("third", Language.ENGLISH, Language.SPANISH) is not None

    def test_get_stats(self):
        cache = TranslationCache()
        result = self._make_result()
        cache.put(result)

        cache.get("Hello", Language.ENGLISH, Language.SPANISH)  # hit
        cache.get("World", Language.ENGLISH, Language.SPANISH)  # miss

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        cache = TranslationCache()
        cache.put(self._make_result())
        cache.clear()
        assert cache.get("Hello", Language.ENGLISH, Language.SPANISH) is None

    def test_stats_empty(self):
        cache = TranslationCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0


# ===========================================================================
# LLMTranslationBackend - heuristic detection
# ===========================================================================


class TestHeuristicDetection:
    """Tests for heuristic language detection in LLMTranslationBackend."""

    def setup_method(self):
        self.backend = LLMTranslationBackend()

    def test_chinese_characters(self):
        result = self.backend._heuristic_detect("这是中文")
        assert result is not None
        assert result.detected_language == Language.CHINESE
        assert result.confidence == 0.95

    def test_japanese_hiragana(self):
        result = self.backend._heuristic_detect("こんにちは")
        assert result is not None
        assert result.detected_language == Language.JAPANESE

    def test_korean_hangul(self):
        result = self.backend._heuristic_detect("안녕하세요")
        assert result is not None
        assert result.detected_language == Language.KOREAN

    def test_arabic(self):
        result = self.backend._heuristic_detect("مرحبا")
        assert result is not None
        assert result.detected_language == Language.ARABIC

    def test_hebrew(self):
        result = self.backend._heuristic_detect("שלום")
        assert result is not None
        assert result.detected_language == Language.HEBREW

    def test_cyrillic(self):
        result = self.backend._heuristic_detect("Привет")
        assert result is not None
        assert result.detected_language == Language.RUSSIAN

    def test_thai(self):
        result = self.backend._heuristic_detect("สวัสดี")
        assert result is not None
        assert result.detected_language == Language.THAI

    def test_greek(self):
        result = self.backend._heuristic_detect("Γεια σας")
        assert result is not None
        assert result.detected_language == Language.GREEK

    def test_hindi(self):
        result = self.backend._heuristic_detect("नमस्ते")
        assert result is not None
        assert result.detected_language == Language.HINDI

    def test_latin_returns_none(self):
        """Latin script returns None, falling through to LLM detection."""
        result = self.backend._heuristic_detect("Hello world")
        assert result is None


# ===========================================================================
# TranslationService
# ===========================================================================


class MockBackend(TranslationBackend):
    """Mock translation backend for testing."""

    name = "mock"

    async def translate(self, text, source, target):
        return TranslationResult(
            original_text=text,
            translated_text=f"[{target.value}]{text}",
            source_language=source,
            target_language=target,
            confidence=0.9,
            backend="mock",
        )

    async def detect_language(self, text):
        return LanguageDetectionResult(
            text=text[:100],
            detected_language=Language.ENGLISH,
            confidence=0.9,
        )


class TestTranslationService:
    """Tests for TranslationService."""

    @pytest.fixture
    def service(self):
        return TranslationService(backend=MockBackend())

    @pytest.mark.asyncio
    async def test_translate_empty_text(self, service):
        result = await service.translate("", Language.SPANISH)
        assert result.translated_text == ""
        assert result.backend == "empty"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_translate_whitespace_only(self, service):
        result = await service.translate("   ", Language.SPANISH)
        assert result.backend == "empty"

    @pytest.mark.asyncio
    async def test_translate_same_language(self, service):
        result = await service.translate("Hello", Language.ENGLISH, Language.ENGLISH)
        assert result.translated_text == "Hello"
        assert result.backend == "same-language"

    @pytest.mark.asyncio
    async def test_translate_with_source(self, service):
        result = await service.translate("Hello", Language.SPANISH, Language.ENGLISH)
        assert result.translated_text == "[es]Hello"
        assert result.backend == "mock"

    @pytest.mark.asyncio
    async def test_translate_auto_detect_source(self, service):
        result = await service.translate("Hello", Language.SPANISH)
        # MockBackend detects English, then translates EN→ES
        assert result.translated_text == "[es]Hello"

    @pytest.mark.asyncio
    async def test_translate_uses_cache(self, service):
        # First call - cache miss
        r1 = await service.translate("Hello", Language.SPANISH, Language.ENGLISH)
        assert r1.cached is False

        # Second call - cache hit
        r2 = await service.translate("Hello", Language.SPANISH, Language.ENGLISH)
        assert r2.cached is True

    @pytest.mark.asyncio
    async def test_translate_batch(self, service):
        texts = ["Hello", "World", "Test"]
        results = await service.translate_batch(texts, Language.SPANISH, Language.ENGLISH)
        assert len(results) == 3
        assert all(r.target_language == Language.SPANISH for r in results)

    def test_get_stats(self, service):
        stats = service.get_stats()
        assert stats["backend"] == "mock"
        assert "cache" in stats


# ===========================================================================
# MultilingualDebateManager
# ===========================================================================


class TestMultilingualDebateManager:
    """Tests for MultilingualDebateManager."""

    @pytest.fixture
    def manager(self):
        service = TranslationService(backend=MockBackend())
        return MultilingualDebateManager(translation_service=service)

    def test_set_and_get_participant_language(self, manager):
        manager.set_participant_language("alice", Language.FRENCH)
        assert manager.get_participant_language("alice") == Language.FRENCH

    def test_default_participant_language(self, manager):
        assert manager.get_participant_language("unknown") == Language.ENGLISH

    @pytest.mark.asyncio
    async def test_translate_message(self, manager):
        manager.set_participant_language("alice", Language.ENGLISH)
        manager.set_participant_language("bob", Language.SPANISH)

        result = await manager.translate_message("Hello", "alice", "bob")
        assert result.target_language == Language.SPANISH

    @pytest.mark.asyncio
    async def test_translate_for_all(self, manager):
        config = MultilingualDebateConfig(
            supported_languages=[Language.ENGLISH, Language.SPANISH, Language.FRENCH],
        )
        manager.config = config

        results = await manager.translate_for_all("Hello", Language.ENGLISH)
        assert Language.ENGLISH in results
        assert Language.SPANISH in results
        assert Language.FRENCH in results
        # Original should have backend="original"
        assert results[Language.ENGLISH].backend == "original"

    @pytest.mark.asyncio
    async def test_translate_debate_conclusion(self, manager):
        translations = await manager.translate_debate_conclusion(
            "The consensus is...",
            source_language=Language.ENGLISH,
            target_languages=[Language.ENGLISH, Language.SPANISH],
        )
        assert "en" in translations
        assert "es" in translations
        assert translations["en"] == "The consensus is..."


# ===========================================================================
# Singletons
# ===========================================================================


class TestSingletons:
    """Tests for module-level singleton getters."""

    def test_get_translation_service_returns_instance(self):
        svc = get_translation_service()
        assert isinstance(svc, TranslationService)

    def test_get_multilingual_manager_returns_instance(self):
        mgr = get_multilingual_manager()
        assert isinstance(mgr, MultilingualDebateManager)


# ===========================================================================
# MultilingualDebateConfig
# ===========================================================================


class TestMultilingualDebateConfig:
    """Tests for MultilingualDebateConfig defaults."""

    def test_defaults(self):
        config = MultilingualDebateConfig()
        assert config.default_language == Language.ENGLISH
        assert config.auto_translate is True
        assert config.translate_consensus is True
        assert config.translate_messages is True
        assert config.preserve_original is True
        assert len(config.supported_languages) == len(Language)

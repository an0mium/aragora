"""
Multi-Language Support for Aragora Debates.

Provides automatic translation of debates to support:
- Agents responding in multiple languages
- Cross-language debate participation
- Automatic translation of messages and conclusions
- Language detection and routing

Supported translation backends:
- OpenAI (gpt-4 based translation)
- Anthropic (Claude based translation)
- Google Cloud Translation API
- DeepL API (if configured)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages for debates."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    POLISH = "pl"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    UKRAINIAN = "uk"
    CZECH = "cs"
    SWEDISH = "sv"
    DANISH = "da"
    FINNISH = "fi"
    NORWEGIAN = "no"
    HEBREW = "he"
    GREEK = "el"
    HUNGARIAN = "hu"
    ROMANIAN = "ro"

    @classmethod
    def from_code(cls, code: str) -> Optional["Language"]:
        """Get language from ISO code."""
        code = code.lower().strip()[:2]
        for lang in cls:
            if lang.value == code:
                return lang
        return None

    @property
    def name_english(self) -> str:
        """Get English name of the language."""
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "vi": "Vietnamese",
            "th": "Thai",
            "id": "Indonesian",
            "uk": "Ukrainian",
            "cs": "Czech",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "no": "Norwegian",
            "he": "Hebrew",
            "el": "Greek",
            "hu": "Hungarian",
            "ro": "Romanian",
        }
        return names.get(self.value, self.value)


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    original_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float = 1.0
    backend: str = "unknown"
    cached: bool = False
    translation_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original_text": (
                self.original_text[:100] + "..."
                if len(self.original_text) > 100
                else self.original_text
            ),
            "translated_text": (
                self.translated_text[:100] + "..."
                if len(self.translated_text) > 100
                else self.translated_text
            ),
            "source_language": self.source_language.value,
            "target_language": self.target_language.value,
            "confidence": self.confidence,
            "backend": self.backend,
            "cached": self.cached,
            "translation_time_ms": self.translation_time_ms,
        }


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""

    text: str
    detected_language: Language
    confidence: float
    alternatives: list[tuple[Language, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "detected_language": self.detected_language.value,
            "confidence": self.confidence,
            "alternatives": [(lang.value, conf) for lang, conf in self.alternatives],
        }


class TranslationCache:
    """
    LRU cache for translations with TTL support.

    Uses content hashing to store translations efficiently.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        ttl_seconds: float = 3600.0,  # 1 hour
    ):
        self._cache: OrderedDict[str, tuple[TranslationResult, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, source: Language, target: Language) -> str:
        """Create a cache key from text and language pair."""
        content = f"{source.value}:{target.value}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, source: Language, target: Language) -> Optional[TranslationResult]:
        """Get a cached translation if available."""
        key = self._make_key(text, source, target)
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self._hits += 1
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                    # Mark as cached
                    result.cached = True
                    return result
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, result: TranslationResult) -> None:
        """Cache a translation result."""
        key = self._make_key(result.original_text, result.source_language, result.target_language)
        with self._lock:
            # Evict if full
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (result, time.time())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()


class TranslationBackend:
    """Base class for translation backends."""

    name: str = "base"

    async def translate(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> TranslationResult:
        """Translate text from source to target language."""
        raise NotImplementedError

    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language of text."""
        raise NotImplementedError


class LLMTranslationBackend(TranslationBackend):
    """Translation backend using LLM (Claude or GPT)."""

    name = "llm"

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider

    async def translate(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> TranslationResult:
        """Translate using LLM."""
        start_time = time.time()

        prompt = f"""Translate the following text from {source.name_english} to {target.name_english}.
Maintain the original meaning, tone, and technical accuracy.
If the text contains technical terms, keep them accurate.
Only output the translation, nothing else.

Text to translate:
{text}"""

        try:
            if self.provider == "anthropic":
                from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

                agent = AnthropicAPIAgent(name="translator", model="claude-3-haiku-20240307")
                response = await agent.respond(prompt, [])
                translated = response.content
            else:
                from aragora.agents.api_agents.openai import OpenAIAPIAgent

                agent = OpenAIAPIAgent(name="translator", model="gpt-4o-mini")
                response = await agent.respond(prompt, [])
                translated = response.content

            translation_time = (time.time() - start_time) * 1000

            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_language=source,
                target_language=target,
                confidence=0.95,
                backend=f"llm-{self.provider}",
                translation_time_ms=translation_time,
            )
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original on failure
                source_language=source,
                target_language=target,
                confidence=0.0,
                backend=f"llm-{self.provider}-error",
                metadata={"error": str(e)},
            )

    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language using LLM."""
        # Use simple heuristics for common cases
        detected = self._heuristic_detect(text)
        if detected:
            return detected

        prompt = f"""Detect the language of the following text.
Respond with just the ISO 639-1 two-letter language code (e.g., 'en' for English, 'es' for Spanish).

Text:
{text[:500]}"""

        try:
            if self.provider == "anthropic":
                from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

                agent = AnthropicAPIAgent(name="detector", model="claude-3-haiku-20240307")
                response = await agent.respond(prompt, [])
                code = response.content.strip().lower()[:2]
            else:
                from aragora.agents.api_agents.openai import OpenAIAPIAgent

                agent = OpenAIAPIAgent(name="detector", model="gpt-4o-mini")
                response = await agent.respond(prompt, [])
                code = response.content.strip().lower()[:2]

            language = Language.from_code(code)
            if language:
                return LanguageDetectionResult(
                    text=text[:100],
                    detected_language=language,
                    confidence=0.9,
                )
        except Exception as e:
            logger.warning(f"LLM language detection failed: {e}")

        # Default to English
        return LanguageDetectionResult(
            text=text[:100],
            detected_language=Language.ENGLISH,
            confidence=0.5,
        )

    def _heuristic_detect(self, text: str) -> Optional[LanguageDetectionResult]:
        """Simple heuristic-based language detection."""
        text = text.lower()

        # Check for CJK characters
        if re.search(r"[\u4e00-\u9fff]", text):
            return LanguageDetectionResult(text[:100], Language.CHINESE, 0.95)
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return LanguageDetectionResult(text[:100], Language.JAPANESE, 0.95)
        if re.search(r"[\uac00-\ud7af]", text):
            return LanguageDetectionResult(text[:100], Language.KOREAN, 0.95)

        # Check for Arabic
        if re.search(r"[\u0600-\u06ff]", text):
            return LanguageDetectionResult(text[:100], Language.ARABIC, 0.95)

        # Check for Hebrew
        if re.search(r"[\u0590-\u05ff]", text):
            return LanguageDetectionResult(text[:100], Language.HEBREW, 0.95)

        # Check for Cyrillic (Russian/Ukrainian)
        if re.search(r"[\u0400-\u04ff]", text):
            # Default to Russian for Cyrillic
            return LanguageDetectionResult(text[:100], Language.RUSSIAN, 0.8)

        # Check for Thai
        if re.search(r"[\u0e00-\u0e7f]", text):
            return LanguageDetectionResult(text[:100], Language.THAI, 0.95)

        # Check for Greek
        if re.search(r"[\u0370-\u03ff]", text):
            return LanguageDetectionResult(text[:100], Language.GREEK, 0.95)

        # Check for Hindi/Devanagari
        if re.search(r"[\u0900-\u097f]", text):
            return LanguageDetectionResult(text[:100], Language.HINDI, 0.95)

        return None


class TranslationService:
    """
    Main translation service for Aragora debates.

    Features:
    - Automatic language detection
    - Cached translations
    - Multiple backend support
    - Batch translation
    """

    def __init__(
        self,
        backend: Optional[TranslationBackend] = None,
        cache: Optional[TranslationCache] = None,
    ):
        self.backend = backend or LLMTranslationBackend()
        self.cache = cache or TranslationCache()

    async def translate(
        self,
        text: str,
        target: Language,
        source: Optional[Language] = None,
    ) -> TranslationResult:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target: Target language
            source: Source language (auto-detected if None)

        Returns:
            TranslationResult with translated text
        """
        if not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source or Language.ENGLISH,
                target_language=target,
                confidence=1.0,
                backend="empty",
            )

        # Detect source language if not provided
        if source is None:
            detection = await self.detect_language(text)
            source = detection.detected_language

        # Same language - no translation needed
        if source == target:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source,
                target_language=target,
                confidence=1.0,
                backend="same-language",
            )

        # Check cache
        cached = self.cache.get(text, source, target)
        if cached:
            return cached

        # Translate
        result = await self.backend.translate(text, source, target)

        # Cache result
        if result.confidence > 0.5:
            self.cache.put(result)

        return result

    async def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect the language of text."""
        return await self.backend.detect_language(text)

    async def translate_batch(
        self,
        texts: list[str],
        target: Language,
        source: Optional[Language] = None,
        max_concurrent: int = 5,
    ) -> list[TranslationResult]:
        """
        Translate multiple texts in batch.

        Args:
            texts: List of texts to translate
            target: Target language
            source: Source language (auto-detected if None)
            max_concurrent: Max concurrent translation requests

        Returns:
            List of TranslationResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def translate_with_limit(text: str) -> TranslationResult:
            async with semaphore:
                return await self.translate(text, target, source)

        tasks = [translate_with_limit(text) for text in texts]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "backend": self.backend.name,
            "cache": self.cache.get_stats(),
        }


@dataclass
class MultilingualDebateConfig:
    """Configuration for multilingual debate support."""

    default_language: Language = Language.ENGLISH
    supported_languages: list[Language] = field(default_factory=lambda: list(Language))
    auto_translate: bool = True
    translate_consensus: bool = True
    translate_messages: bool = True
    preserve_original: bool = True


class MultilingualDebateManager:
    """
    Manages multilingual aspects of debates.

    Integrates with the debate system to:
    - Detect participant languages
    - Translate messages across languages
    - Maintain original and translated versions
    - Generate multilingual conclusions
    """

    def __init__(
        self,
        translation_service: Optional[TranslationService] = None,
        config: Optional[MultilingualDebateConfig] = None,
    ):
        self.service = translation_service or TranslationService()
        self.config = config or MultilingualDebateConfig()
        self._participant_languages: dict[str, Language] = {}

    def set_participant_language(self, participant_id: str, language: Language) -> None:
        """Set the preferred language for a participant."""
        self._participant_languages[participant_id] = language

    def get_participant_language(self, participant_id: str) -> Language:
        """Get the preferred language for a participant."""
        return self._participant_languages.get(participant_id, self.config.default_language)

    async def translate_message(
        self,
        message: str,
        from_participant: str,
        to_participant: str,
    ) -> TranslationResult:
        """
        Translate a message from one participant's language to another's.

        Args:
            message: The message to translate
            from_participant: Source participant ID
            to_participant: Target participant ID

        Returns:
            TranslationResult with translated message
        """
        source_lang = self.get_participant_language(from_participant)
        target_lang = self.get_participant_language(to_participant)

        return await self.service.translate(message, target_lang, source_lang)

    async def translate_for_all(
        self,
        message: str,
        source_language: Language,
    ) -> dict[Language, TranslationResult]:
        """
        Translate a message to all supported languages.

        Args:
            message: The message to translate
            source_language: The source language

        Returns:
            Dict mapping language to TranslationResult
        """
        results: dict[Language, TranslationResult] = {}

        # Include original
        results[source_language] = TranslationResult(
            original_text=message,
            translated_text=message,
            source_language=source_language,
            target_language=source_language,
            confidence=1.0,
            backend="original",
        )

        # Translate to other languages
        other_languages = [
            lang for lang in self.config.supported_languages if lang != source_language
        ]

        # Translate to each target language
        for lang in other_languages:
            result = await self.service.translate(message, lang, source_language)
            results[lang] = result

        return results

    async def translate_debate_conclusion(
        self,
        conclusion: str,
        source_language: Language = Language.ENGLISH,
        target_languages: Optional[list[Language]] = None,
    ) -> dict[str, str]:
        """
        Translate debate conclusion to multiple languages.

        Args:
            conclusion: The conclusion text
            source_language: Language of the conclusion
            target_languages: Languages to translate to (defaults to all supported)

        Returns:
            Dict mapping language code to translated conclusion
        """
        if target_languages is None:
            target_languages = self.config.supported_languages

        translations: dict[str, str] = {source_language.value: conclusion}

        for lang in target_languages:
            if lang != source_language:
                result = await self.service.translate(conclusion, lang, source_language)
                translations[lang.value] = result.translated_text

        return translations


# Global singleton instances
_translation_service: Optional[TranslationService] = None
_multilingual_manager: Optional[MultilingualDebateManager] = None
_lock = threading.Lock()


def get_translation_service() -> TranslationService:
    """Get the global translation service instance."""
    global _translation_service
    with _lock:
        if _translation_service is None:
            _translation_service = TranslationService()
        return _translation_service


def get_multilingual_manager() -> MultilingualDebateManager:
    """Get the global multilingual debate manager instance."""
    global _multilingual_manager
    with _lock:
        if _multilingual_manager is None:
            _multilingual_manager = MultilingualDebateManager()
        return _multilingual_manager


__all__ = [
    "Language",
    "TranslationResult",
    "LanguageDetectionResult",
    "TranslationCache",
    "TranslationBackend",
    "LLMTranslationBackend",
    "TranslationService",
    "MultilingualDebateConfig",
    "MultilingualDebateManager",
    "get_translation_service",
    "get_multilingual_manager",
]

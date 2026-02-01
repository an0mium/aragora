"""
Translation Skill.

Provides text translation capabilities using LLM inference.
Supports translation between multiple languages with context awareness.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


# Supported languages with their codes
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "id": "Indonesian",
    "ms": "Malay",
}

# Reverse mapping for name to code
LANGUAGE_NAME_TO_CODE = {v.lower(): k for k, v in SUPPORTED_LANGUAGES.items()}


class TranslationSkill(Skill):
    """
    Skill for translating text between languages.

    Features:
    - Translation between 30+ languages
    - Language detection
    - Context-aware translation
    - Preserves formatting
    - Technical term handling
    """

    def __init__(
        self,
        default_target_language: str = "en",
    ):
        """
        Initialize translation skill.

        Args:
            default_target_language: Default target language code
        """
        self._default_target = default_target_language

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="translation",
            version="1.0.0",
            description="Translate text between languages",
            capabilities=[
                SkillCapability.LLM_INFERENCE,
            ],
            input_schema={
                "text": {
                    "type": "string",
                    "description": "Text to translate",
                    "required": True,
                },
                "target_language": {
                    "type": "string",
                    "description": "Target language code (e.g., 'es', 'fr') or name (e.g., 'Spanish')",
                    "default": "en",
                },
                "source_language": {
                    "type": "string",
                    "description": "Source language code or name (auto-detect if not specified)",
                },
                "context": {
                    "type": "string",
                    "description": "Context to help with accurate translation",
                },
                "preserve_formatting": {
                    "type": "boolean",
                    "description": "Preserve original text formatting",
                    "default": True,
                },
                "formality": {
                    "type": "string",
                    "description": "Formality level: formal, informal, neutral",
                    "default": "neutral",
                },
            },
            tags=["translation", "language", "nlp"],
            debate_compatible=True,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=30,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute translation."""
        text = input_data.get("text", "").strip()
        if not text:
            return SkillResult.create_failure(
                "Text is required",
                error_code="missing_text",
            )

        target_lang = input_data.get("target_language", self._default_target)
        source_lang = input_data.get("source_language")
        translation_context = input_data.get("context", "")
        preserve_formatting = input_data.get("preserve_formatting", True)
        formality = input_data.get("formality", "neutral")

        # Normalize language codes
        target_lang = self._normalize_language(target_lang)
        if source_lang:
            source_lang = self._normalize_language(source_lang)

        if target_lang not in SUPPORTED_LANGUAGES:
            return SkillResult.create_failure(
                f"Unsupported target language: {target_lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
                error_code="unsupported_language",
            )

        try:
            # Detect source language if not provided
            if not source_lang:
                source_lang = await self._detect_language(text)

            if source_lang == target_lang:
                return SkillResult.create_success(
                    {
                        "translation": text,
                        "source_language": source_lang,
                        "source_language_name": SUPPORTED_LANGUAGES.get(source_lang, source_lang),
                        "target_language": target_lang,
                        "target_language_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
                        "note": "Source and target languages are the same",
                    }
                )

            # Perform translation
            translation = await self._translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=translation_context,
                preserve_formatting=preserve_formatting,
                formality=formality,
            )

            return SkillResult.create_success(
                {
                    "translation": translation,
                    "source_language": source_lang,
                    "source_language_name": SUPPORTED_LANGUAGES.get(source_lang, source_lang),
                    "target_language": target_lang,
                    "target_language_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
                    "original_text": text,
                    "char_count": len(translation),
                    "formality": formality,
                }
            )

        except Exception as e:
            logger.exception(f"Translation failed: {e}")
            return SkillResult.create_failure(f"Translation failed: {e}")

    def _normalize_language(self, lang: str) -> str:
        """Normalize language input to language code."""
        lang = lang.lower().strip()

        # Already a code
        if lang in SUPPORTED_LANGUAGES:
            return lang

        # Full name
        if lang in LANGUAGE_NAME_TO_CODE:
            return LANGUAGE_NAME_TO_CODE[lang]

        # Try partial match
        for name, code in LANGUAGE_NAME_TO_CODE.items():
            if lang in name or name in lang:
                return code

        # Return as-is
        return lang

    async def _detect_language(self, text: str) -> str:
        """Detect the language of text."""
        # Try using LLM for detection
        llm = await self._get_llm()
        if llm:
            prompt = f"""Detect the language of the following text.
Respond with ONLY the two-letter language code (e.g., 'en', 'es', 'fr', 'de', 'zh', 'ja').

Text: {text[:500]}

Language code:"""
            try:
                response = await self._call_llm(llm, prompt)
                code = response.strip().lower()[:2]
                if code in SUPPORTED_LANGUAGES:
                    return code
            except Exception as e:
                logger.warning(f"Language detection via LLM failed: {e}")

        # Fallback: simple heuristic-based detection
        return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> str:
        """Simple heuristic-based language detection."""
        text_lower = text.lower()

        # Check for script-based detection
        # Chinese characters
        if any("\u4e00" <= c <= "\u9fff" for c in text):
            return "zh"

        # Japanese (Hiragana/Katakana)
        if any("\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in text):
            return "ja"

        # Korean (Hangul)
        if any("\uac00" <= c <= "\ud7af" for c in text):
            return "ko"

        # Arabic
        if any("\u0600" <= c <= "\u06ff" for c in text):
            return "ar"

        # Hebrew
        if any("\u0590" <= c <= "\u05ff" for c in text):
            return "he"

        # Russian/Cyrillic
        if any("\u0400" <= c <= "\u04ff" for c in text):
            return "ru"

        # Greek
        if any("\u0370" <= c <= "\u03ff" for c in text):
            return "el"

        # Thai
        if any("\u0e00" <= c <= "\u0e7f" for c in text):
            return "th"

        # Latin script - check for language-specific patterns
        word_patterns = {
            "es": ["el", "la", "los", "las", "de", "en", "que", "es", "por", "con"],
            "fr": ["le", "la", "les", "de", "et", "en", "est", "que", "pour", "dans"],
            "de": [
                "der",
                "die",
                "das",
                "und",
                "ist",
                "ein",
                "eine",
                "nicht",
                "mit",
                "auf",
            ],
            "it": [
                "il",
                "la",
                "di",
                "che",
                "e",
                "un",
                "una",
                "per",
                "con",
                "sono",
            ],
            "pt": ["o", "a", "os", "as", "de", "que", "em", "para", "com", "por"],
            "nl": [
                "de",
                "het",
                "een",
                "van",
                "en",
                "in",
                "is",
                "dat",
                "op",
                "met",
            ],
        }

        words = text_lower.split()
        word_set = set(words)

        best_match = "en"
        best_score = 0

        for lang, patterns in word_patterns.items():
            score = len(word_set & set(patterns))
            if score > best_score:
                best_score = score
                best_match = lang

        return best_match if best_score > 2 else "en"

    async def _translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str,
        preserve_formatting: bool,
        formality: str,
    ) -> str:
        """Perform the translation using LLM."""
        llm = await self._get_llm()
        if not llm:
            # Fallback: return original with note
            logger.warning("No LLM available for translation")
            return f"[Translation unavailable - LLM not configured]\n{text}"

        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        # Build translation prompt
        prompt_parts = [
            f"Translate the following text from {source_name} to {target_name}.",
        ]

        if formality == "formal":
            prompt_parts.append("Use formal/polite language.")
        elif formality == "informal":
            prompt_parts.append("Use informal/casual language.")

        if preserve_formatting:
            prompt_parts.append(
                "Preserve the original formatting, including line breaks and structure."
            )

        if context:
            prompt_parts.append(f"Context: {context}")

        prompt_parts.append("Provide ONLY the translation, without any explanations or notes.")
        prompt_parts.append(f"\nOriginal text:\n{text}\n\nTranslation:")

        prompt = " ".join(prompt_parts)

        return await self._call_llm(llm, prompt)

    async def _get_llm(self) -> Any | None:
        """Get an LLM instance for translation."""
        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            return AnthropicAPIAgent(model="claude-3-haiku-20240307")
        except ImportError:
            pass

        try:
            from aragora.agents.api_agents.openai import OpenAIAPIAgent

            return OpenAIAPIAgent(model="gpt-4o-mini")
        except ImportError:
            pass

        return None

    async def _call_llm(self, llm: Any, prompt: str) -> str:
        """Call LLM for text generation."""
        try:
            if hasattr(llm, "generate"):
                response = await llm.generate(prompt, max_tokens=2000)
                if hasattr(response, "text"):
                    return response.text.strip()
                return str(response).strip()
            elif hasattr(llm, "complete"):
                response = await llm.complete(prompt, max_tokens=2000)
                return str(response).strip()
            else:
                response = await llm(prompt)
                return str(response).strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            raise


# Skill instance for registration
SKILLS = [TranslationSkill()]

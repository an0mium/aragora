"""Translation and output formatting helpers for Arena.

Extracted from orchestrator.py to reduce its size. These functions
handle post-debate output concerns: formatting conclusions and
translating them into configured target languages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aragora.logging_config import get_logger as get_structured_logger

if TYPE_CHECKING:
    from aragora.core import DebateResult

logger = get_structured_logger(__name__)


def format_conclusion(result: "DebateResult") -> str:
    """Format debate conclusion. Delegates to ResultFormatter.

    Args:
        result: The completed debate result.

    Returns:
        A formatted conclusion string.
    """
    from aragora.debate.result_formatter import ResultFormatter

    return ResultFormatter().format_conclusion(result)


async def translate_conclusions(
    result: "DebateResult",
    protocol: Any,
) -> None:
    """Translate debate conclusions to configured target languages.

    Uses the translation module to provide multi-language support.
    Translations are stored in ``result.translations`` dict.

    Args:
        result: The completed debate result (mutated in place).
        protocol: The debate protocol (checked for ``target_languages``,
            ``default_language``).
    """
    if not result.final_answer:
        return

    target_languages = getattr(protocol, "target_languages", [])
    if not target_languages:
        return

    try:
        from aragora.debate.translation import (
            Language,
            get_translation_service,
        )

        service = get_translation_service()
        default_lang = getattr(protocol, "default_language", "en")

        # Detect or use configured source language
        source_lang = Language.from_code(default_lang) or Language.ENGLISH

        for target_code in target_languages:
            target_lang = Language.from_code(target_code)
            if not target_lang or target_lang == source_lang:
                continue

            try:
                translation_result = await service.translate(
                    result.final_answer,
                    target_lang,
                    source_lang,
                )
                if translation_result.confidence > 0.5:
                    result.translations[target_code] = translation_result.translated_text
                    logger.debug(
                        f"Translated conclusion to {target_lang.name_english} "
                        f"(confidence: {translation_result.confidence:.2f})"
                    )
            except (ConnectionError, OSError, ValueError, TypeError) as e:
                logger.warning(f"Translation to {target_code} failed: {e}")

    except ImportError as e:
        logger.debug(f"Translation module not available: {e}")
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Translation failed (non-critical): {e}")

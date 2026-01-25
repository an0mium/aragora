"""
TTS Helper for chat platform handlers.

Provides voice synthesis for chat responses when TTS is enabled.
Integrates with the existing TTS bridge and backends.

Usage:
    from aragora.server.handlers.social.tts_helper import (
        get_tts_helper,
        is_tts_enabled,
    )

    helper = get_tts_helper()
    if helper.is_available:
        audio_bytes = await helper.synthesize_response(
            "The debate concluded with consensus.",
            voice="narrator",
        )
        # Send audio to chat platform
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TTS configuration
TTS_ENABLED = os.environ.get("ARAGORA_TTS_CHAT_ENABLED", "false").lower() == "true"
TTS_DEFAULT_VOICE = os.environ.get("ARAGORA_TTS_DEFAULT_VOICE", "narrator")
TTS_MAX_TEXT_LENGTH = int(os.environ.get("ARAGORA_TTS_MAX_TEXT", "2000"))


def is_tts_enabled() -> bool:
    """Check if TTS is enabled for chat responses."""
    return TTS_ENABLED


@dataclass
class SynthesisResult:
    """Result of TTS synthesis."""

    audio_bytes: bytes
    format: str  # mp3, wav, ogg
    duration_seconds: float
    voice: str
    text_length: int


class TTSHelper:
    """
    Helper class for TTS synthesis in chat handlers.

    Provides a simple interface for converting text to audio that
    can be sent as voice messages on chat platforms.
    """

    def __init__(self):
        """Initialize TTS Helper."""
        self._bridge: Optional[Any] = None
        self._backend: Optional[Any] = None
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        if not TTS_ENABLED:
            return False

        if self._available is not None:
            return self._available

        try:
            self._get_backend()
            self._available = True
            return True
        except (ImportError, RuntimeError) as e:
            logger.debug(f"TTS backend not available: {e}")
            self._available = False
            return False
        except Exception as e:
            logger.exception(f"Unexpected error checking TTS availability: {e}")
            self._available = False
            return False

    def _get_backend(self) -> Any:
        """Get TTS backend (lazy initialization)."""
        if self._backend is not None:
            return self._backend

        try:
            from aragora.broadcast.tts_backends import get_tts_backend

            self._backend = get_tts_backend()
            logger.info(f"TTSHelper using backend: {self._backend.name}")
            return self._backend
        except ImportError as e:
            logger.warning(f"TTS backends not available: {e}")
            raise RuntimeError("TTS backends not available")
        except Exception as e:
            logger.error(f"Failed to initialize TTS backend: {e}")
            raise

    def _get_bridge(self) -> Any:
        """Get TTS bridge (lazy initialization)."""
        if self._bridge is not None:
            return self._bridge

        try:
            from aragora.connectors.chat.tts_bridge import get_tts_bridge

            self._bridge = get_tts_bridge()
            return self._bridge
        except ImportError as e:
            logger.warning(f"TTS bridge not available: {e}")
            raise RuntimeError("TTS bridge not available")

    async def synthesize_response(
        self,
        text: str,
        voice: Optional[str] = None,
        output_format: str = "mp3",
    ) -> Optional[SynthesisResult]:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice: Voice identifier (narrator, moderator, etc.)
            output_format: Audio format (mp3, wav, ogg)

        Returns:
            SynthesisResult with audio bytes, or None if synthesis fails
        """
        if not self.is_available:
            logger.debug("TTS not available for synthesis")
            return None

        # Truncate if too long
        if len(text) > TTS_MAX_TEXT_LENGTH:
            text = text[: TTS_MAX_TEXT_LENGTH - 3] + "..."
            logger.debug(f"TTS text truncated to {TTS_MAX_TEXT_LENGTH} chars")

        voice = voice or TTS_DEFAULT_VOICE

        try:
            bridge = self._get_bridge()
            audio_path = await bridge.synthesize(
                text=text,
                voice=voice,
                output_format=output_format,
            )

            if not audio_path or not Path(audio_path).exists():
                logger.warning("TTS synthesis returned no audio file")
                return None

            # Read audio bytes
            audio_bytes = Path(audio_path).read_bytes()

            # Get duration estimate (rough: ~128kbps for mp3)
            duration_estimate = len(audio_bytes) / 16000  # Very rough estimate

            result = SynthesisResult(
                audio_bytes=audio_bytes,
                format=output_format,
                duration_seconds=duration_estimate,
                voice=voice,
                text_length=len(text),
            )

            # Clean up temp file
            try:
                Path(audio_path).unlink()
            except (OSError, PermissionError) as e:
                logger.debug(f"Could not remove temp TTS file {audio_path}: {e}")

            logger.info(f"TTS synthesized: {len(text)} chars -> {len(audio_bytes)} bytes")
            return result

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    async def synthesize_debate_result(
        self,
        task: str,
        final_answer: Optional[str],
        consensus_reached: bool,
        confidence: float,
        rounds_used: int,
    ) -> Optional[SynthesisResult]:
        """
        Synthesize a debate result summary.

        Args:
            task: The debate task/question
            final_answer: The final answer if available
            consensus_reached: Whether consensus was reached
            confidence: Confidence level
            rounds_used: Number of rounds

        Returns:
            SynthesisResult or None if synthesis fails
        """
        # Build summary
        lines = [f"Debate completed on: {task[:150]}"]

        if consensus_reached:
            lines.append(
                f"Consensus was reached with {confidence:.0%} confidence "
                f"after {rounds_used} rounds."
            )
            if final_answer:
                preview = final_answer[:400] + ("..." if len(final_answer) > 400 else "")
                lines.append(f"The conclusion is: {preview}")
        else:
            lines.append(
                f"No consensus was reached after {rounds_used} rounds. "
                f"Final confidence was {confidence:.0%}."
            )

        summary = " ".join(lines)
        return await self.synthesize_response(summary, voice="narrator")

    async def synthesize_gauntlet_result(
        self,
        statement: str,
        passed: bool,
        score: float,
        vulnerability_count: int,
    ) -> Optional[SynthesisResult]:
        """
        Synthesize a gauntlet result summary.

        Args:
            statement: The statement tested
            passed: Whether it passed
            score: The score
            vulnerability_count: Number of vulnerabilities found

        Returns:
            SynthesisResult or None if synthesis fails
        """
        status = "passed" if passed else "failed"
        summary = (
            f"Gauntlet stress test {status} with a score of {score:.0%}. "
            f"The statement was: {statement[:150]}. "
        )

        if vulnerability_count > 0:
            summary += f"{vulnerability_count} vulnerabilities were found."
        else:
            summary += "No significant vulnerabilities were found."

        return await self.synthesize_response(summary, voice="moderator")


# Singleton instance
_tts_helper: Optional[TTSHelper] = None


def get_tts_helper() -> TTSHelper:
    """Get the TTS helper singleton."""
    global _tts_helper
    if _tts_helper is None:
        _tts_helper = TTSHelper()
    return _tts_helper


__all__ = [
    "TTSHelper",
    "SynthesisResult",
    "get_tts_helper",
    "is_tts_enabled",
]

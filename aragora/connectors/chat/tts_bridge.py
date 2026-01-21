"""
TTS Bridge - Text-to-speech integration for chat voice responses.

Bridges chat platform responses to TTS backends for synthesizing
voice messages that can be sent back to users.

Usage:
    from aragora.connectors.chat import get_tts_bridge

    bridge = get_tts_bridge()
    audio_path = await bridge.synthesize_response(
        "The debate concluded with consensus on option A.",
        voice="narrator"
    )

    # Send to chat platform
    await connector.send_voice_message(channel_id, audio_path)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .base import ChatPlatformConnector

if TYPE_CHECKING:
    from aragora.broadcast.tts_backends import TTSBackend

logger = logging.getLogger(__name__)

# TTS configuration from environment
TTS_DEFAULT_VOICE = os.environ.get("ARAGORA_TTS_DEFAULT_VOICE", "narrator")
TTS_MAX_TEXT_LENGTH = int(os.environ.get("ARAGORA_TTS_MAX_TEXT", "4000"))
TTS_CACHE_DIR = os.environ.get("ARAGORA_TTS_CACHE_DIR", "")


@dataclass
class TTSConfig:
    """Configuration for TTS bridge."""

    default_voice: str = TTS_DEFAULT_VOICE
    max_text_length: int = TTS_MAX_TEXT_LENGTH
    cache_enabled: bool = True
    cache_dir: Optional[str] = TTS_CACHE_DIR or None

    # Voice mappings for different contexts
    voice_map: dict[str, str] = field(default_factory=lambda: {
        "narrator": "narrator",
        "moderator": "moderator",
        "claude": "analyst",
        "gpt": "expert",
        "gemini": "researcher",
        "consensus": "narrator",
        "error": "moderator",
    })


class TTSBridge:
    """
    Bridge between text responses and TTS synthesis.

    Converts text to audio for sending voice responses
    via chat platforms.
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize TTS Bridge.

        Args:
            config: TTS configuration
            **kwargs: Additional configuration passed to TTS backend
        """
        self.config = config or TTSConfig()
        self._kwargs = kwargs
        self._backend: Optional["TTSBackend"] = None
        self._temp_dir: Optional[Path] = None

    def _get_backend(self) -> "TTSBackend":
        """Lazy-load TTS backend."""
        if self._backend is None:
            try:
                from aragora.broadcast.tts_backends import get_tts_backend

                self._backend = get_tts_backend()
                logger.info(f"TTS Bridge using backend: {self._backend.name}")
            except ImportError:
                logger.error("TTS backends not available")
                raise RuntimeError("TTS backends not available - install aragora[broadcast]")
        return self._backend

    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        try:
            self._get_backend()
            return True
        except RuntimeError:
            return False

    def _get_temp_dir(self) -> Path:
        """Get temporary directory for audio files."""
        if self._temp_dir is None or not self._temp_dir.exists():
            if self.config.cache_dir:
                self._temp_dir = Path(self.config.cache_dir)
                self._temp_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._temp_dir = Path(tempfile.mkdtemp(prefix="aragora_tts_"))
        return self._temp_dir

    def _resolve_voice(self, voice: Optional[str], context: Optional[str] = None) -> str:
        """Resolve voice identifier from context or explicit voice."""
        if voice:
            return self.config.voice_map.get(voice, voice)
        if context:
            return self.config.voice_map.get(context, self.config.default_voice)
        return self.config.default_voice

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        context: Optional[str] = None,
        output_format: str = "mp3",
    ) -> Path:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice: Voice identifier (narrator, moderator, etc.)
            context: Context hint for voice selection
            output_format: Audio format (mp3, wav, ogg)

        Returns:
            Path to generated audio file
        """
        backend = self._get_backend()

        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length - 3] + "..."
            logger.warning(f"TTS text truncated to {self.config.max_text_length} chars")

        # Resolve voice
        resolved_voice = self._resolve_voice(voice, context)

        # Generate audio
        try:
            audio_path = await backend.synthesize(
                text=text,
                voice=resolved_voice,
                output_dir=str(self._get_temp_dir()),
            )
            return Path(audio_path)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    async def synthesize_debate_summary(
        self,
        task: str,
        final_answer: Optional[str],
        consensus_reached: bool,
        confidence: float,
        rounds_used: int,
    ) -> Path:
        """
        Synthesize a debate summary as audio.

        Args:
            task: The debate task/question
            final_answer: The final answer if consensus reached
            consensus_reached: Whether consensus was reached
            confidence: Confidence level
            rounds_used: Number of rounds used

        Returns:
            Path to generated audio file
        """
        # Build summary text
        lines = [f"Debate completed on: {task[:200]}"]

        if consensus_reached:
            lines.append(
                f"Consensus was reached with {confidence:.0%} confidence "
                f"after {rounds_used} rounds."
            )
            if final_answer:
                answer_preview = final_answer[:500]
                if len(final_answer) > 500:
                    answer_preview += "..."
                lines.append(f"The final answer is: {answer_preview}")
        else:
            lines.append(
                f"No consensus was reached after {rounds_used} rounds. "
                f"Final confidence was {confidence:.0%}."
            )

        summary_text = " ".join(lines)
        return await self.synthesize(summary_text, voice="narrator")

    async def synthesize_consensus_alert(
        self,
        answer: str,
        confidence: float,
    ) -> Path:
        """
        Synthesize a consensus alert as audio.

        Args:
            answer: The consensus answer
            confidence: Confidence level

        Returns:
            Path to generated audio file
        """
        text = (
            f"Consensus has been reached with {confidence:.0%} confidence. "
            f"The agreed answer is: {answer[:400]}"
        )
        return await self.synthesize(text, voice="consensus")

    async def synthesize_error_alert(
        self,
        error_type: str,
        error_message: str,
    ) -> Path:
        """
        Synthesize an error alert as audio.

        Args:
            error_type: Type of error
            error_message: Error details

        Returns:
            Path to generated audio file
        """
        text = f"An error occurred: {error_type}. {error_message[:200]}"
        return await self.synthesize(text, voice="error")

    async def synthesize_response(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> str:
        """
        Synthesize a text response as audio.

        Convenience method for simple text-to-speech synthesis.

        Args:
            text: Text to synthesize
            voice: Voice identifier (narrator, moderator, consensus, etc.)

        Returns:
            Path to generated audio file as string
        """
        audio_path = await self.synthesize(text, voice=voice)
        return str(audio_path)

    async def send_voice_response(
        self,
        connector: ChatPlatformConnector,
        channel_id: str,
        text: str,
        voice: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Synthesize and send a voice response to a chat channel.

        Args:
            connector: Chat platform connector
            channel_id: Channel to send to
            text: Text to synthesize and send
            voice: Voice to use
            reply_to: Message ID to reply to

        Returns:
            True if sent successfully
        """
        try:
            # Synthesize audio
            audio_path = await self.synthesize(text, voice=voice)

            # Read audio content
            audio_content = audio_path.read_bytes()

            # Send via connector
            response = await connector.send_voice_message(
                channel_id=channel_id,
                audio_content=audio_content,
                filename=audio_path.name,
                reply_to=reply_to,
            )

            # Clean up temp file
            try:
                audio_path.unlink()
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to clean up temp audio file {audio_path}: {e}")

            return response.success
        except Exception as e:
            logger.error(f"Failed to send voice response: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
                self._temp_dir = None
            except Exception as e:
                logger.warning(f"Failed to cleanup TTS temp dir: {e}")


# Singleton instance
_tts_bridge: Optional[TTSBridge] = None


def get_tts_bridge(**config: Any) -> TTSBridge:
    """Get or create the TTS Bridge singleton."""
    global _tts_bridge
    if _tts_bridge is None:
        _tts_bridge = TTSBridge(**config)
    return _tts_bridge


def clear_tts_bridge() -> None:
    """Clear the TTS bridge singleton (for testing)."""
    global _tts_bridge
    if _tts_bridge:
        _tts_bridge.cleanup()
    _tts_bridge = None

"""
Voice Bridge - Transcription integration for chat voice messages.

Bridges chat platform voice messages to the WhisperConnector for
speech-to-text transcription.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import ChatPlatformConnector
from .models import VoiceMessage

logger = logging.getLogger(__name__)


class VoiceBridge:
    """
    Bridge between chat platforms and speech-to-text services.

    Downloads voice messages from chat platforms and transcribes them
    using the configured transcription backend (default: Whisper).
    """

    def __init__(
        self,
        max_file_size: int = 25 * 1024 * 1024,  # 25MB (Whisper limit)
        default_language: str = "en",
        **config: Any,
    ):
        """
        Initialize VoiceBridge.

        Args:
            max_file_size: Maximum file size to process (bytes)
            default_language: Default language hint for transcription
            **config: Additional configuration passed to Whisper
        """
        self.max_file_size = max_file_size
        self.default_language = default_language
        self.config = config
        self._whisper = None

    def _get_whisper(self):
        """Lazy-load WhisperConnector."""
        if self._whisper is None:
            try:
                from aragora.connectors.whisper import WhisperConnector

                self._whisper = WhisperConnector(**self.config)
            except ImportError:
                logger.error("WhisperConnector not available")
                raise RuntimeError("WhisperConnector not available")
        return self._whisper

    async def transcribe_voice_message(
        self,
        voice_message: VoiceMessage,
        connector: Optional[ChatPlatformConnector] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe a voice message.

        Args:
            voice_message: The voice message to transcribe
            connector: Chat connector to download the file if needed
            language: Language hint for transcription

        Returns:
            Transcribed text
        """
        # Check if we have content or need to download
        content = voice_message.file.content

        if content is None and connector is not None:
            # Download the file
            logger.info(f"Downloading voice file: {voice_message.file.id}")
            attachment = await connector.download_file(
                voice_message.file.id,
                url=voice_message.file.url,
                filename=voice_message.file.filename,
            )
            content = attachment.content

        if content is None:
            raise ValueError("No audio content available for transcription")

        # Check file size
        if len(content) > self.max_file_size:
            raise ValueError(
                f"Audio file too large: {len(content)} bytes " f"(max: {self.max_file_size})"
            )

        # Transcribe using Whisper
        whisper = self._get_whisper()
        lang = language or self.default_language

        logger.info(
            f"Transcribing {len(content)} bytes from {voice_message.platform} "
            f"(language: {lang})"
        )

        # Call Whisper transcription
        transcription = await whisper.transcribe(
            audio_data=content,
            filename=voice_message.file.filename or "audio.ogg",
            language=lang,
        )

        # Update the voice message with transcription
        voice_message.transcription = transcription

        return transcription

    async def transcribe_file(
        self,
        content: bytes,
        filename: str = "audio.ogg",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe raw audio content.

        Args:
            content: Audio file content
            filename: Filename (used for format detection)
            language: Language hint

        Returns:
            Transcribed text
        """
        if len(content) > self.max_file_size:
            raise ValueError(f"Audio file too large: {len(content)} bytes")

        whisper = self._get_whisper()
        lang = language or self.default_language

        return await whisper.transcribe(
            audio_data=content,
            filename=filename,
            language=lang,
        )

    async def process_chat_audio(
        self,
        connector: ChatPlatformConnector,
        file_id: str,
        file_url: Optional[str] = None,
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Download and transcribe audio from a chat platform.

        Convenience method that handles the download and transcription
        in one call.

        Args:
            connector: Chat platform connector
            file_id: ID of the file to download
            file_url: Optional URL for direct download
            filename: Optional filename hint
            language: Language hint for transcription

        Returns:
            Transcribed text
        """
        # Download file
        attachment = await connector.download_file(
            file_id,
            url=file_url,
            filename=filename,
        )

        if attachment.content is None:
            raise ValueError(f"Failed to download file: {file_id}")

        # Transcribe
        return await self.transcribe_file(
            content=attachment.content,
            filename=attachment.filename or filename or "audio",
            language=language,
        )


# Singleton instance
_voice_bridge: Optional[VoiceBridge] = None


def get_voice_bridge(**config: Any) -> VoiceBridge:
    """Get or create the VoiceBridge singleton."""
    global _voice_bridge
    if _voice_bridge is None:
        _voice_bridge = VoiceBridge(**config)
    return _voice_bridge

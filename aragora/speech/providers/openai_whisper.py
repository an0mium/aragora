"""
OpenAI Whisper API provider for speech-to-text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Optional

from aragora.speech.providers.base import (
    STTProvider,
    STTProviderConfig,
    TranscriptionResult,
    TranscriptionSegment,
)


class OpenAIWhisperProvider(STTProvider):
    """Speech-to-text provider using OpenAI's Whisper API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[STTProviderConfig] = None,
    ):
        super().__init__(config)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def name(self) -> str:
        return "openai_whisper"

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI Whisper provider. "
                    "Install it with: pip install openai"
                )
        return self._client

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self._api_key:
            return False
        try:
            self._get_client()
            return True
        except (ImportError, ValueError, RuntimeError) as e:
            import logging

            logging.getLogger(__name__).warning(f"OpenAI Whisper not available: {e}")
            return False

    async def transcribe(
        self,
        audio_file: Path | BinaryIO,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI Whisper API.

        Args:
            audio_file: Path to audio file or file-like object
            language: Language code (e.g., "en"). None for auto-detect.
            prompt: Optional prompt to guide transcription (useful for domain-specific terms)

        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        client = self._get_client()

        # Use configured language if not specified
        lang = language or self.config.default_language

        # Determine model from config
        model = self.config.model if self.config.model != "default" else "whisper-1"

        # Prepare file for API call
        if isinstance(audio_file, Path):
            with open(audio_file, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    language=lang,
                    response_format="verbose_json",
                    prompt=prompt,
                    timestamp_granularities=(
                        ["word", "segment"]
                        if self.config.include_word_timestamps
                        else ["segment"]
                    ),
                )
        else:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=lang,
                response_format="verbose_json",
                prompt=prompt,
                timestamp_granularities=(
                    ["word", "segment"]
                    if self.config.include_word_timestamps
                    else ["segment"]
                ),
            )

        # Parse segments
        segments = []
        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segment = TranscriptionSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    confidence=seg.get("avg_logprob"),
                )

                # Add word-level timestamps if available
                if self.config.include_word_timestamps and hasattr(response, "words"):
                    # Filter words that fall within this segment
                    segment_words = [
                        w
                        for w in (response.words or [])
                        if seg.get("start", 0) <= w.get("start", 0) < seg.get("end", 0)
                    ]
                    if segment_words:
                        segment.words = segment_words

                segments.append(segment)

        return TranscriptionResult(
            text=response.text,
            language=getattr(response, "language", lang or "en"),
            duration=getattr(response, "duration", 0.0),
            segments=segments,
            provider=self.name,
            model=model,
        )

    def supported_formats(self) -> list[str]:
        """OpenAI Whisper supported formats."""
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

    def max_file_size_mb(self) -> int:
        """OpenAI Whisper max file size."""
        return 25  # 25MB limit

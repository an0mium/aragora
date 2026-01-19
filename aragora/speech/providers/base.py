"""
Base interface for speech-to-text providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional


@dataclass
class STTProviderConfig:
    """Configuration for STT providers."""

    # Provider identification
    provider_name: str = "base"

    # Language settings
    default_language: Optional[str] = None  # None = auto-detect

    # Model settings
    model: str = "default"

    # Processing options
    include_timestamps: bool = True
    include_word_timestamps: bool = False

    # Additional provider-specific options
    extra_options: dict = field(default_factory=dict)


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing information."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: Optional[float] = None
    words: Optional[list[dict]] = None  # Word-level timestamps if available


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    language: str
    duration: float  # Audio duration in seconds
    segments: list[TranscriptionSegment] = field(default_factory=list)

    # Metadata
    provider: str = "unknown"
    model: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "confidence": s.confidence,
                    "words": s.words,
                }
                for s in self.segments
            ],
            "provider": self.provider,
            "model": self.model,
        }


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    def __init__(self, config: Optional[STTProviderConfig] = None):
        self.config = config or STTProviderConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_file: Path | BinaryIO,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_file: Path to audio file or file-like object
            language: Language code (e.g., "en", "es"). None for auto-detect.
            prompt: Optional prompt to guide transcription

        Returns:
            TranscriptionResult with text and metadata
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass

    def supported_formats(self) -> list[str]:
        """Return list of supported audio formats."""
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac"]

    def max_file_size_mb(self) -> int:
        """Return maximum file size in megabytes."""
        return 25  # Default 25MB

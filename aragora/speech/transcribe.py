"""
High-level transcription functions for speech-to-text.
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
from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider


# Re-export for convenience
__all__ = [
    "transcribe_audio",
    "transcribe_audio_file",
    "get_provider",
    "TranscriptionResult",
    "TranscriptionSegment",
]


def get_provider(
    provider_name: Optional[str] = None,
    config: Optional[STTProviderConfig] = None,
) -> STTProvider:
    """
    Get an STT provider by name.

    Args:
        provider_name: Provider name. Defaults to ARAGORA_STT_PROVIDER env var
                      or "openai_whisper".
        config: Optional provider configuration.

    Returns:
        Configured STTProvider instance.
    """
    name = provider_name or os.getenv("ARAGORA_STT_PROVIDER", "openai_whisper")

    providers = {
        "openai_whisper": OpenAIWhisperProvider,
        "whisper": OpenAIWhisperProvider,  # Alias
    }

    provider_class = providers.get(name.lower())
    if not provider_class:
        available = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown STT provider: {name}. Available providers: {available}"
        )

    return provider_class(config=config)


async def transcribe_audio(
    audio_file: Path | BinaryIO,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    provider_name: Optional[str] = None,
    config: Optional[STTProviderConfig] = None,
) -> TranscriptionResult:
    """
    Transcribe audio to text.

    This is the main entry point for transcription.

    Args:
        audio_file: Path to audio file or file-like object
        language: Language code (e.g., "en", "es"). None for auto-detect.
        prompt: Optional prompt to guide transcription
        provider_name: STT provider to use. Defaults to env var or "openai_whisper".
        config: Optional provider configuration.

    Returns:
        TranscriptionResult with text and metadata.

    Example:
        >>> result = await transcribe_audio(Path("audio.mp3"))
        >>> print(result.text)
        "Hello, this is a test."
        >>> print(result.duration)
        5.2
    """
    provider = get_provider(provider_name, config)

    if not await provider.is_available():
        raise RuntimeError(
            f"STT provider '{provider.name}' is not available. "
            "Check that required API keys are configured."
        )

    return await provider.transcribe(audio_file, language, prompt)


async def transcribe_audio_file(
    file_path: str | Path,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe an audio file by path.

    Convenience function that accepts string paths.

    Args:
        file_path: Path to the audio file
        language: Language code (e.g., "en")
        prompt: Optional prompt to guide transcription
        provider_name: STT provider to use

    Returns:
        TranscriptionResult with text and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return await transcribe_audio(
        path, language=language, prompt=prompt, provider_name=provider_name
    )

"""
Speech processing module for Aragora.

Provides speech-to-text (STT) capabilities using various providers.
"""

from aragora.speech.transcribe import (
    transcribe_audio,
    transcribe_audio_file,
    TranscriptionResult,
    TranscriptionSegment,
)
from aragora.speech.providers.base import STTProvider, STTProviderConfig
from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

__all__ = [
    "transcribe_audio",
    "transcribe_audio_file",
    "TranscriptionResult",
    "TranscriptionSegment",
    "STTProvider",
    "STTProviderConfig",
    "OpenAIWhisperProvider",
]

"""
Transcription module for speech-to-text and video transcription.

Provides multiple transcription backends with fallback support:
1. OpenAI Whisper API - Cloud-based, fast, paid
2. faster-whisper - Local CTranslate2-optimized, GPU recommended
3. whisper.cpp - Local C++ implementation, CPU fallback

Usage:
    from aragora.transcription import get_transcription_backend, transcribe_audio

    # Auto-select best available backend
    backend = get_transcription_backend()
    result = await backend.transcribe("audio.mp3")
    print(result.text)

    # Or use convenience function
    result = await transcribe_audio("audio.mp3")

    # Transcribe YouTube video
    from aragora.transcription import transcribe_youtube
    result = await transcribe_youtube("https://youtube.com/watch?v=...")
"""

from aragora.transcription.whisper_backend import (
    TranscriptionBackend,
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionSegment,
    get_transcription_backend,
    get_available_backends,
    transcribe_audio,
    transcribe_video,
)
from aragora.transcription.youtube import (
    YouTubeFetcher,
    YouTubeVideoInfo,
    transcribe_youtube,
    fetch_youtube_audio,
)

__all__ = [
    # Backend classes
    "TranscriptionBackend",
    "TranscriptionConfig",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Backend utilities
    "get_transcription_backend",
    "get_available_backends",
    # Convenience functions
    "transcribe_audio",
    "transcribe_video",
    "transcribe_youtube",
    # YouTube utilities
    "YouTubeFetcher",
    "YouTubeVideoInfo",
    "fetch_youtube_audio",
]

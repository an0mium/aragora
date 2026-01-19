"""
Speech-to-text providers for Aragora.

Supported providers:
- OpenAI Whisper API (default)
"""

from aragora.speech.providers.base import STTProvider, STTProviderConfig
from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

__all__ = [
    "STTProvider",
    "STTProviderConfig",
    "OpenAIWhisperProvider",
]

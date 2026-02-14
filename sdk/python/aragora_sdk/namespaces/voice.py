"""
Voice Namespace API

Provides methods for voice and TTS operations:
- Text-to-speech synthesis
- Voice session management
- Audio streaming
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class VoiceAPI:
    """Synchronous Voice API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client


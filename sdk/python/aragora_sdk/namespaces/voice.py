"""
Voice Namespace API.

Provides text-to-speech synthesis and voice session management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class VoiceAPI:
    """Synchronous Voice API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def synthesize(
        self, text: str, voice: str = "default", format: str = "mp3", **kwargs: Any
    ) -> Any:
        """Synthesize speech from text."""
        return self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={"text": text, "voice": voice, "format": format, **kwargs},
        )

    def list_voices(self) -> Any:
        """List available voices."""
        return self._client.request("GET", "/api/v1/voice/voices")

    def create_session(self, debate_id: str, **kwargs: Any) -> Any:
        """Create a voice streaming session."""
        return self._client.request(
            "POST",
            "/api/v1/voice/sessions",
            json={"debate_id": debate_id, **kwargs},
        )

    def end_session(self, session_id: str) -> Any:
        """End a voice session."""
        return self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )


class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def synthesize(
        self, text: str, voice: str = "default", format: str = "mp3", **kwargs: Any
    ) -> Any:
        """Synthesize speech from text."""
        return await self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={"text": text, "voice": voice, "format": format, **kwargs},
        )

    async def list_voices(self) -> Any:
        """List available voices."""
        return await self._client.request("GET", "/api/v1/voice/voices")

    async def create_session(self, debate_id: str, **kwargs: Any) -> Any:
        """Create a voice streaming session."""
        return await self._client.request(
            "POST",
            "/api/v1/voice/sessions",
            json={"debate_id": debate_id, **kwargs},
        )

    async def end_session(self, session_id: str) -> Any:
        """End a voice session."""
        return await self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )

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

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        format: str = "mp3",
        speed: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize text to speech."""
        body: dict[str, Any] = {"text": text, "voice": voice, "format": format}
        if speed is not None:
            body["speed"] = speed
        return self._client.request("POST", "/api/v1/voice/synthesize", json=body)

    def list_voices(self) -> dict[str, Any]:
        """List available voices."""
        return self._client.request("GET", "/api/v1/voice/voices")

    def create_session(self, debate_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create a voice session for a debate."""
        body: dict[str, Any] = {"debate_id": debate_id, **kwargs}
        return self._client.request("POST", "/api/v1/voice/sessions", json=body)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End a voice session."""
        return self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get a voice session by ID."""
        return self._client.request(
            "GET", f"/api/v1/voice/sessions/{session_id}"
        )

    def get_config(self) -> dict[str, Any]:
        """Get voice configuration."""
        return self._client.request("GET", "/api/v1/voice/config")


class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        format: str = "mp3",
        speed: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize text to speech."""
        body: dict[str, Any] = {"text": text, "voice": voice, "format": format}
        if speed is not None:
            body["speed"] = speed
        return await self._client.request(
            "POST", "/api/v1/voice/synthesize", json=body
        )

    async def list_voices(self) -> dict[str, Any]:
        """List available voices."""
        return await self._client.request("GET", "/api/v1/voice/voices")

    async def create_session(self, debate_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create a voice session for a debate."""
        body: dict[str, Any] = {"debate_id": debate_id, **kwargs}
        return await self._client.request(
            "POST", "/api/v1/voice/sessions", json=body
        )

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End a voice session."""
        return await self._client.request(
            "DELETE", f"/api/v1/voice/sessions/{session_id}"
        )

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get a voice session by ID."""
        return await self._client.request(
            "GET", f"/api/v1/voice/sessions/{session_id}"
        )

    async def get_config(self) -> dict[str, Any]:
        """Get voice configuration."""
        return await self._client.request("GET", "/api/v1/voice/config")

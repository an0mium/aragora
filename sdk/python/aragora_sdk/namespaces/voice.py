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

    def synthesize(self, text: str, voice: str = "default", format: str = "mp3") -> dict[str, Any]:
        """Synthesize text to speech."""
        return self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={
                "text": text,
                "voice": voice,
                "format": format,
            },
        )

    def list_voices(self) -> dict[str, Any]:
        """List available voices."""
        return self._client.request("GET", "/api/v1/voice/voices")

    def get_voice(self, voice_id: str) -> dict[str, Any]:
        """Get voice details."""
        return self._client.request("GET", f"/api/v1/voice/voices/{voice_id}")

    def create_session(
        self, debate_id: str | None = None, voice: str = "default"
    ) -> dict[str, Any]:
        """Create a voice session."""
        data: dict[str, Any] = {"voice": voice}
        if debate_id:
            data["debate_id"] = debate_id
        return self._client.request("POST", "/api/v1/voice/sessions", json=data)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get voice session."""
        return self._client.request("GET", f"/api/v1/voice/sessions/{session_id}")

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End a voice session."""
        return self._client.request("DELETE", f"/api/v1/voice/sessions/{session_id}")

    def get_stream_url(self, session_id: str) -> dict[str, Any]:
        """Get WebSocket URL for voice streaming."""
        return self._client.request("GET", f"/api/v1/voice/sessions/{session_id}/stream")


class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def synthesize(
        self, text: str, voice: str = "default", format: str = "mp3"
    ) -> dict[str, Any]:
        """Synthesize text to speech."""
        return await self._client.request(
            "POST",
            "/api/v1/voice/synthesize",
            json={
                "text": text,
                "voice": voice,
                "format": format,
            },
        )

    async def list_voices(self) -> dict[str, Any]:
        """List available voices."""
        return await self._client.request("GET", "/api/v1/voice/voices")

    async def get_voice(self, voice_id: str) -> dict[str, Any]:
        """Get voice details."""
        return await self._client.request("GET", f"/api/v1/voice/voices/{voice_id}")

    async def create_session(
        self, debate_id: str | None = None, voice: str = "default"
    ) -> dict[str, Any]:
        """Create a voice session."""
        data: dict[str, Any] = {"voice": voice}
        if debate_id:
            data["debate_id"] = debate_id
        return await self._client.request("POST", "/api/v1/voice/sessions", json=data)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get voice session."""
        return await self._client.request("GET", f"/api/v1/voice/sessions/{session_id}")

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End a voice session."""
        return await self._client.request("DELETE", f"/api/v1/voice/sessions/{session_id}")

    async def get_stream_url(self, session_id: str) -> dict[str, Any]:
        """Get WebSocket URL for voice streaming."""
        return await self._client.request("GET", f"/api/v1/voice/sessions/{session_id}/stream")

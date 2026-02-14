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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice: Voice identifier.
            format: Audio format (mp3, wav, ogg).

        Returns:
            Synthesis result with audio URL or data.
        """
        payload: dict[str, Any] = {"text": text, "voice": voice, "format": format, **kwargs}
        return self._client.request("POST", "/api/v1/voice/synthesize", json=payload)

    def list_voices(self) -> dict[str, Any]:
        """
        List available voices.

        Returns:
            Available voice configurations.
        """
        return self._client.request("GET", "/api/v1/voice/voices")

    def create_session(
        self,
        debate_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a new voice streaming session.

        Args:
            debate_id: Optional debate to stream audio for.

        Returns:
            Session details including session ID.
        """
        payload: dict[str, Any] = {**kwargs}
        if debate_id is not None:
            payload["debate_id"] = debate_id
        return self._client.request("POST", "/api/v1/voice/sessions", json=payload)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        End a voice streaming session.

        Args:
            session_id: Session identifier.

        Returns:
            Confirmation of session termination.
        """
        return self._client.request("DELETE", f"/api/v1/voice/sessions/{session_id}")


class AsyncVoiceAPI:
    """Asynchronous Voice API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        format: str = "mp3",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synthesize speech from text."""
        payload: dict[str, Any] = {"text": text, "voice": voice, "format": format, **kwargs}
        return await self._client.request("POST", "/api/v1/voice/synthesize", json=payload)

    async def list_voices(self) -> dict[str, Any]:
        """List available voices."""
        return await self._client.request("GET", "/api/v1/voice/voices")

    async def create_session(
        self,
        debate_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new voice streaming session."""
        payload: dict[str, Any] = {**kwargs}
        if debate_id is not None:
            payload["debate_id"] = debate_id
        return await self._client.request("POST", "/api/v1/voice/sessions", json=payload)

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End a voice streaming session."""
        return await self._client.request("DELETE", f"/api/v1/voice/sessions/{session_id}")


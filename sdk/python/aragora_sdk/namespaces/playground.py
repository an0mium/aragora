"""
Playground Namespace API

Provides methods for the interactive debate playground:
- Create playground debates
- Stream live debates
- Cost estimation
- Status and TTS
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PlaygroundAPI:
    """
    Synchronous Playground API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.playground.create_debate(task="Should we use Rust?")
        >>> status = client.playground.get_status()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def create_debate(self, **kwargs: Any) -> dict[str, Any]:
        """
        Create a playground debate.

        Args:
            **kwargs: Debate parameters (task, agents, rounds, etc.)

        Returns:
            Dict with debate_id and initial state.
        """
        return self._client.request("POST", "/api/playground/debate", json=kwargs)

    def create_live_debate(self, **kwargs: Any) -> dict[str, Any]:
        """
        Create a live-streaming playground debate.

        Args:
            **kwargs: Debate parameters.

        Returns:
            Dict with debate_id and stream URL.
        """
        return self._client.request("POST", "/api/playground/debate/live", json=kwargs)

    def estimate_live_cost(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get cost estimate for a live playground debate before starting it.

        Args:
            **kwargs: Debate configuration for estimation.

        Returns:
            Dict with estimated cost breakdown.
        """
        return self._client.request("POST", "/api/playground/debate/live/cost-estimate", json=kwargs)

    def get_status(self) -> dict[str, Any]:
        """
        Get playground system status.

        Returns:
            Dict with available models, capacity, and queue depth.
        """
        return self._client.request("GET", "/api/playground/status")

    def text_to_speech(self, **kwargs: Any) -> dict[str, Any]:
        """
        Convert debate text to speech audio.

        Args:
            **kwargs: TTS parameters (text, voice, speed, etc.)

        Returns:
            Dict with audio URL or base64-encoded audio data.
        """
        return self._client.request("POST", "/api/playground/tts", json=kwargs)


class AsyncPlaygroundAPI:
    """
    Asynchronous Playground API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.playground.create_debate(task="Should we use Rust?")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create_debate(self, **kwargs: Any) -> dict[str, Any]:
        """Create a playground debate."""
        return await self._client.request("POST", "/api/playground/debate", json=kwargs)

    async def create_live_debate(self, **kwargs: Any) -> dict[str, Any]:
        """Create a live-streaming playground debate."""
        return await self._client.request("POST", "/api/playground/debate/live", json=kwargs)

    async def estimate_live_cost(self, **kwargs: Any) -> dict[str, Any]:
        """Get cost estimate for a live playground debate."""
        return await self._client.request("POST", "/api/playground/debate/live/cost-estimate", json=kwargs)

    async def get_status(self) -> dict[str, Any]:
        """Get playground system status."""
        return await self._client.request("GET", "/api/playground/status")

    async def text_to_speech(self, **kwargs: Any) -> dict[str, Any]:
        """Convert debate text to speech audio."""
        return await self._client.request("POST", "/api/playground/tts", json=kwargs)

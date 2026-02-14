"""
Audio Namespace API.

Provides audio file serving and podcast feed management.

Features:
- Direct audio file URL generation
- Podcast episode listing
- RSS feed URL access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AudioAPI:
    """
    Synchronous Audio API.

    Provides methods for audio content access:
    - Get direct URLs for audio files
    - List and browse podcast episodes
    - Access RSS feed for subscription

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> audio_url = client.audio.get_audio_url("debate_123")
        >>> episodes = client.audio.list_episodes()
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def get_audio_url(self, audio_id: str) -> str:
        """
        Get the direct URL for an audio file.

        This URL can be used to stream or download the audio file directly.

        Args:
            audio_id: The audio file identifier.

        Returns:
            Direct URL to the audio file in MP3 format.
        """
        base_url = getattr(self._client, "_base_url", "https://api.aragora.ai")
        return f"{base_url}/audio/{audio_id}.mp3"

    def get_audio_info(self, audio_id: str) -> dict[str, Any]:
        """
        Get metadata for an audio file.

        Args:
            audio_id: The audio file identifier.

        Returns:
            Audio file metadata including format, duration, and size.
        """
        return self._client._request("GET", f"/api/v1/audio/{audio_id}")

    def list_episodes(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        List podcast episodes.

        Args:
            limit: Maximum number of episodes to return.
            offset: Pagination offset.

        Returns:
            Dict containing list of episodes and optional total count.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return self._client._request(
            "GET",
            "/api/v1/podcast/episodes",
            params=params if params else None,
        )

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        """
        Get a specific podcast episode.

        Args:
            episode_id: The episode identifier.

        Returns:
            Episode details including title, description, audio URL, and duration.
        """
        return self._client._request("GET", f"/api/v1/podcast/episodes/{episode_id}")

    def get_feed_url(self) -> str:
        """
        Get the podcast RSS feed URL.

        This URL can be used in podcast apps like Apple Podcasts, Spotify, etc.

        Returns:
            RSS feed URL for podcast subscription.
        """
        base_url = getattr(self._client, "_base_url", "https://api.aragora.ai")
        return f"{base_url}/api/v1/podcast/feed.xml"


    def serve_audio(self, audio_path: str) -> dict[str, Any]:
        """
        Serve audio file by path.

        GET /audio/:path

        Args:
            audio_path: Audio file path

        Returns:
            Audio file data
        """
        return self._client.request("GET", f"/audio/{audio_path}")


class AsyncAudioAPI:
    """
    Asynchronous Audio API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     episodes = await client.audio.list_episodes()
        ...     print(f"Found {len(episodes['episodes'])} episodes")
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    def get_audio_url(self, audio_id: str) -> str:
        """
        Get the direct URL for an audio file.

        Args:
            audio_id: The audio file identifier.

        Returns:
            Direct URL to the audio file in MP3 format.
        """
        base_url = getattr(self._client, "_base_url", "https://api.aragora.ai")
        return f"{base_url}/audio/{audio_id}.mp3"

    async def get_audio_info(self, audio_id: str) -> dict[str, Any]:
        """Get metadata for an audio file."""
        return await self._client._request("GET", f"/api/v1/audio/{audio_id}")

    async def list_episodes(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List podcast episodes."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return await self._client._request(
            "GET",
            "/api/v1/podcast/episodes",
            params=params if params else None,
        )

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        """Get a specific podcast episode."""
        return await self._client._request("GET", f"/api/v1/podcast/episodes/{episode_id}")

    def get_feed_url(self) -> str:
        """Get the podcast RSS feed URL."""
        base_url = getattr(self._client, "_base_url", "https://api.aragora.ai")
        return f"{base_url}/api/v1/podcast/feed.xml"

    async def serve_audio(self, audio_path: str) -> dict[str, Any]:
        """Serve audio file by path. GET /audio/:path"""
        return await self._client.request("GET", f"/audio/{audio_path}")

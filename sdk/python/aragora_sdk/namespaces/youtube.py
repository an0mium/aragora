"""
YouTube Namespace API

Provides YouTube video analysis and integration:
- Analyze video content
- Extract transcripts
- Summarize videos
- Integrate with debates

Features:
- Video metadata extraction
- Automatic transcription
- Content summarization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class YouTubeAPI:
    """
    Synchronous YouTube API.

    Provides methods for YouTube video analysis:
    - Analyze video content
    - Extract transcripts
    - Summarize videos
    - Integrate with debates

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> info = client.youtube.get_video_info("https://youtube.com/watch?v=...")
        >>> transcript = client.youtube.get_transcript(video_id)
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_auth(self) -> dict[str, Any]:
        """Get YouTube OAuth authorization URL."""
        return self._client.request("GET", "/api/v1/youtube/auth")

    def get_callback(self) -> dict[str, Any]:
        """Handle YouTube OAuth callback."""
        return self._client.request("GET", "/api/v1/youtube/callback")

    def get_status(self) -> dict[str, Any]:
        """Get YouTube integration status."""
        return self._client.request("GET", "/api/v1/youtube/status")


class AsyncYouTubeAPI:
    """
    Asynchronous YouTube API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     info = await client.youtube.get_video_info(url)
        ...     summary = await client.youtube.summarize(url)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_auth(self) -> dict[str, Any]:
        """Get YouTube OAuth authorization URL."""
        return await self._client.request("GET", "/api/v1/youtube/auth")

    async def get_callback(self) -> dict[str, Any]:
        """Handle YouTube OAuth callback."""
        return await self._client.request("GET", "/api/v1/youtube/callback")

    async def get_status(self) -> dict[str, Any]:
        """Get YouTube integration status."""
        return await self._client.request("GET", "/api/v1/youtube/status")

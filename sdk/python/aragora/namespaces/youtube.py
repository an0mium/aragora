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

    def get_video_info(self, url: str) -> dict[str, Any]:
        """
        Get video metadata.

        Args:
            url: YouTube video URL

        Returns:
            Dict with:
            - video_id: YouTube video ID
            - title: Video title
            - duration: Duration in seconds
            - channel: Channel name
            - description: Video description
            - upload_date: Upload date
            - view_count: View count
            - thumbnail_url: Thumbnail URL
        """
        return self._client.request("POST", "/api/v1/youtube/info", json={"url": url})

    def get_transcript(
        self,
        video_id: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Get video transcript.

        Args:
            video_id: YouTube video ID
            language: Preferred language (ISO-639-1)

        Returns:
            Dict with transcript text and segments
        """
        data: dict[str, Any] = {"video_id": video_id}
        if language:
            data["language"] = language
        return self._client.request("POST", "/api/v1/youtube/transcript", json=data)

    def summarize(
        self,
        url: str,
        max_length: int | None = None,
        style: str | None = None,
    ) -> dict[str, Any]:
        """
        Summarize a YouTube video.

        Args:
            url: YouTube video URL
            max_length: Maximum summary length
            style: Summary style (brief, detailed, bullet_points)

        Returns:
            Dict with summary and key points
        """
        data: dict[str, Any] = {"url": url}
        if max_length:
            data["max_length"] = max_length
        if style:
            data["style"] = style
        return self._client.request("POST", "/api/v1/youtube/summarize", json=data)

    def analyze(
        self,
        url: str,
        aspects: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze video content.

        Args:
            url: YouTube video URL
            aspects: Aspects to analyze (content, sentiment, topics)

        Returns:
            Dict with analysis results
        """
        data: dict[str, Any] = {"url": url}
        if aspects:
            data["aspects"] = aspects
        return self._client.request("POST", "/api/v1/youtube/analyze", json=data)

    def create_debate_context(
        self,
        url: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """
        Create debate context from a YouTube video.

        Args:
            url: YouTube video URL
            topic: Optional debate topic to focus on

        Returns:
            Dict with context suitable for debates
        """
        data: dict[str, Any] = {"url": url}
        if topic:
            data["topic"] = topic
        return self._client.request("POST", "/api/v1/youtube/debate-context", json=data)

    def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Search YouTube videos.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Dict with video results
        """
        params: dict[str, Any] = {"q": query}
        if limit:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/youtube/search", params=params)


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

    async def get_video_info(self, url: str) -> dict[str, Any]:
        """Get video metadata."""
        return await self._client.request("POST", "/api/v1/youtube/info", json={"url": url})

    async def get_transcript(
        self,
        video_id: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Get video transcript."""
        data: dict[str, Any] = {"video_id": video_id}
        if language:
            data["language"] = language
        return await self._client.request("POST", "/api/v1/youtube/transcript", json=data)

    async def summarize(
        self,
        url: str,
        max_length: int | None = None,
        style: str | None = None,
    ) -> dict[str, Any]:
        """Summarize a YouTube video."""
        data: dict[str, Any] = {"url": url}
        if max_length:
            data["max_length"] = max_length
        if style:
            data["style"] = style
        return await self._client.request("POST", "/api/v1/youtube/summarize", json=data)

    async def analyze(
        self,
        url: str,
        aspects: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze video content."""
        data: dict[str, Any] = {"url": url}
        if aspects:
            data["aspects"] = aspects
        return await self._client.request("POST", "/api/v1/youtube/analyze", json=data)

    async def create_debate_context(
        self,
        url: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Create debate context from a YouTube video."""
        data: dict[str, Any] = {"url": url}
        if topic:
            data["topic"] = topic
        return await self._client.request("POST", "/api/v1/youtube/debate-context", json=data)

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search YouTube videos."""
        params: dict[str, Any] = {"q": query}
        if limit:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/youtube/search", params=params)

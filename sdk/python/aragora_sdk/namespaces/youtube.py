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


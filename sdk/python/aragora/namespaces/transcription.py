"""
Transcription Namespace API

Provides audio/video transcription capabilities:
- Transcribe audio and video files
- Transcribe YouTube videos
- Manage transcription jobs
- Get timestamped segments

Features:
- Multiple transcription backends
- YouTube integration
- Async job processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


TranscriptionStatus = Literal["pending", "processing", "completed", "failed"]
TranscriptionBackend = Literal["openai", "faster-whisper", "whisper-cpp"]
WhisperModel = Literal["tiny", "base", "small", "medium", "large"]


class TranscriptionAPI:
    """
    Synchronous Transcription API.

    Provides methods for audio/video transcription:
    - Transcribe audio and video files
    - Transcribe YouTube videos
    - Get transcription status and results
    - Get timestamped segments

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> config = client.transcription.get_config()
        >>> result = client.transcription.transcribe_youtube("https://youtube.com/watch?v=...")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_config(self) -> dict[str, Any]:
        """
        Get transcription service configuration.

        Returns:
            Dict with:
            - available: Service availability
            - backends: Available backends
            - audio_formats/video_formats: Supported formats
            - max_audio_size_mb/max_video_size_mb: Size limits
            - models: Available Whisper models
            - youtube_enabled: YouTube support
        """
        return self._client.request("GET", "/api/v1/transcription/config")

    def get_formats(self) -> dict[str, Any]:
        """
        Get supported audio/video formats.

        Returns:
            Dict with audio/video format lists and limits
        """
        return self._client.request("GET", "/api/v1/transcription/formats")

    def transcribe_audio(
        self,
        audio_data: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_data: Base64-encoded audio data
            language: ISO-639-1 language code
            backend: Transcription backend to use

        Returns:
            Dict with transcription result
        """
        data: dict[str, Any] = {"audio_data": audio_data}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        return self._client.request("POST", "/api/v1/transcription/audio", json=data)

    def transcribe_video(
        self,
        video_data: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe a video file.

        Args:
            video_data: Base64-encoded video data
            language: ISO-639-1 language code
            backend: Transcription backend to use

        Returns:
            Dict with transcription result
        """
        data: dict[str, Any] = {"video_data": video_data}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        return self._client.request("POST", "/api/v1/transcription/video", json=data)

    def transcribe_youtube(
        self,
        url: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe a YouTube video.

        Args:
            url: YouTube video URL
            language: ISO-639-1 language code
            backend: Transcription backend
            use_cache: Use cached audio if available

        Returns:
            Dict with transcription result
        """
        data: dict[str, Any] = {"url": url}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        if use_cache is not None:
            data["use_cache"] = use_cache
        return self._client.request("POST", "/api/v1/transcription/youtube", json=data)

    def get_youtube_info(self, url: str) -> dict[str, Any]:
        """
        Get YouTube video metadata without transcribing.

        Args:
            url: YouTube video URL

        Returns:
            Dict with video info (title, duration, channel, etc.)
        """
        return self._client.request("POST", "/api/v1/transcription/youtube/info", json={"url": url})

    def get_status(self, job_id: str) -> dict[str, Any]:
        """
        Get transcription job status.

        Args:
            job_id: The job ID

        Returns:
            Dict with job status and result if completed
        """
        return self._client.request("GET", f"/api/v1/transcription/status/{job_id}")

    def get_job(self, job_id: str) -> dict[str, Any]:
        """
        Get transcription job details.

        Args:
            job_id: The job ID

        Returns:
            Dict with full job details
        """
        return self._client.request("GET", f"/api/v1/transcription/{job_id}")

    def get_segments(self, job_id: str) -> dict[str, Any]:
        """
        Get timestamped segments for a completed transcription.

        Args:
            job_id: The job ID

        Returns:
            Dict with segments list
        """
        return self._client.request("GET", f"/api/v1/transcription/{job_id}/segments")

    def upload(self, file_data: str, filename: str) -> dict[str, Any]:
        """
        Upload and queue audio/video for async transcription.

        Args:
            file_data: Base64-encoded file data
            filename: Original filename

        Returns:
            Dict with job_id and upload status
        """
        return self._client.request(
            "POST",
            "/api/v1/transcription/upload",
            json={"file_data": file_data, "filename": filename},
        )

    def delete_job(self, job_id: str) -> dict[str, Any]:
        """
        Delete a transcription job.

        Args:
            job_id: The job ID

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/transcription/{job_id}")


class AsyncTranscriptionAPI:
    """
    Asynchronous Transcription API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.transcription.transcribe_youtube(url)
        ...     segments = await client.transcription.get_segments(result["job_id"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_config(self) -> dict[str, Any]:
        """Get transcription service configuration."""
        return await self._client.request("GET", "/api/v1/transcription/config")

    async def get_formats(self) -> dict[str, Any]:
        """Get supported audio/video formats."""
        return await self._client.request("GET", "/api/v1/transcription/formats")

    async def transcribe_audio(
        self,
        audio_data: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file."""
        data: dict[str, Any] = {"audio_data": audio_data}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        return await self._client.request("POST", "/api/v1/transcription/audio", json=data)

    async def transcribe_video(
        self,
        video_data: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> dict[str, Any]:
        """Transcribe a video file."""
        data: dict[str, Any] = {"video_data": video_data}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        return await self._client.request("POST", "/api/v1/transcription/video", json=data)

    async def transcribe_youtube(
        self,
        url: str,
        language: str | None = None,
        backend: TranscriptionBackend | None = None,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """Transcribe a YouTube video."""
        data: dict[str, Any] = {"url": url}
        if language:
            data["language"] = language
        if backend:
            data["backend"] = backend
        if use_cache is not None:
            data["use_cache"] = use_cache
        return await self._client.request("POST", "/api/v1/transcription/youtube", json=data)

    async def get_youtube_info(self, url: str) -> dict[str, Any]:
        """Get YouTube video metadata without transcribing."""
        return await self._client.request(
            "POST", "/api/v1/transcription/youtube/info", json={"url": url}
        )

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Get transcription job status."""
        return await self._client.request("GET", f"/api/v1/transcription/status/{job_id}")

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Get transcription job details."""
        return await self._client.request("GET", f"/api/v1/transcription/{job_id}")

    async def get_segments(self, job_id: str) -> dict[str, Any]:
        """Get timestamped segments for a completed transcription."""
        return await self._client.request("GET", f"/api/v1/transcription/{job_id}/segments")

    async def upload(self, file_data: str, filename: str) -> dict[str, Any]:
        """Upload and queue audio/video for async transcription."""
        return await self._client.request(
            "POST",
            "/api/v1/transcription/upload",
            json={"file_data": file_data, "filename": filename},
        )

    async def delete_job(self, job_id: str) -> dict[str, Any]:
        """Delete a transcription job."""
        return await self._client.request("DELETE", f"/api/v1/transcription/{job_id}")

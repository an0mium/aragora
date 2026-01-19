"""
Transcription endpoint handlers for speech-to-text and media processing.

Endpoints:
- POST /api/transcription/audio - Transcribe audio file
- POST /api/transcription/video - Extract and transcribe audio from video
- POST /api/transcription/youtube - Transcribe YouTube video
- GET  /api/transcription/status/:id - Get transcription job status
- GET  /api/transcription/config - Get supported formats and limits
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiters (per minute limits)
_audio_limiter = RateLimiter(requests_per_minute=10)
_youtube_limiter = RateLimiter(requests_per_minute=5)

# Maximum file sizes
MAX_AUDIO_SIZE_MB = 100
MAX_VIDEO_SIZE_MB = 500

# Supported formats
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
VIDEO_FORMATS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# In-memory job storage (replace with database in production)
_transcription_jobs: Dict[str, Dict[str, Any]] = {}


def _check_transcription_available() -> tuple[bool, Optional[str]]:
    """Check if transcription module is available."""
    try:
        from aragora.transcription import get_available_backends

        backends = get_available_backends()
        if not backends:
            return False, (
                "No transcription backend available. "
                "Set OPENAI_API_KEY or install faster-whisper."
            )
        return True, None
    except ImportError:
        return False, "Transcription module not installed."


class TranscriptionHandler(BaseHandler):
    """Handler for audio/video transcription endpoints."""

    ROUTES = [
        "/api/transcription/audio",
        "/api/transcription/video",
        "/api/transcription/youtube",
        "/api/transcription/youtube/info",
        "/api/transcription/status/*",
        "/api/transcription/config",
        # Alias routes for frontend compatibility
        "/api/transcribe/audio",
        "/api/transcribe/video",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path in (
            "/api/transcription/audio",
            "/api/transcription/video",
            "/api/transcription/youtube",
            "/api/transcription/youtube/info",
            "/api/transcription/config",
            # Alias routes
            "/api/transcribe/audio",
            "/api/transcribe/video",
        ):
            return True
        if path.startswith("/api/transcription/status/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/transcription/config":
            return self._get_config()

        if path.startswith("/api/transcription/status/"):
            job_id = path.split("/")[-1]
            return self._get_status(job_id)

        return None

    def handle_post(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle POST requests."""
        client_ip = get_client_ip(handler)

        if path in ("/api/transcription/audio", "/api/transcribe/audio"):
            if not _audio_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return self._handle_audio_transcription(handler)

        if path in ("/api/transcription/video", "/api/transcribe/video"):
            if not _audio_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return self._handle_video_transcription(handler)

        if path == "/api/transcription/youtube":
            if not _youtube_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return self._handle_youtube_transcription(handler)

        if path == "/api/transcription/youtube/info":
            return self._handle_youtube_info(handler)

        return None

    def _get_config(self) -> HandlerResult:
        """Return transcription configuration and supported formats."""
        available, error = _check_transcription_available()

        if not available:
            return json_response(
                {
                    "available": False,
                    "error": error,
                    "audio_formats": list(AUDIO_FORMATS),
                    "video_formats": list(VIDEO_FORMATS),
                    "max_audio_size_mb": MAX_AUDIO_SIZE_MB,
                    "max_video_size_mb": MAX_VIDEO_SIZE_MB,
                }
            )

        try:
            from aragora.transcription import get_available_backends
            from aragora.transcription.whisper_backend import WHISPER_MODELS

            backends = get_available_backends()

            return json_response(
                {
                    "available": True,
                    "backends": backends,
                    "audio_formats": list(AUDIO_FORMATS),
                    "video_formats": list(VIDEO_FORMATS),
                    "max_audio_size_mb": MAX_AUDIO_SIZE_MB,
                    "max_video_size_mb": MAX_VIDEO_SIZE_MB,
                    "models": list(WHISPER_MODELS.keys()),
                    "youtube_enabled": True,
                }
            )
        except Exception as e:
            logger.error(f"Error getting transcription config: {e}")
            return error_response("Failed to get configuration", 500)

    def _get_status(self, job_id: str) -> HandlerResult:
        """Get transcription job status."""
        job = _transcription_jobs.get(job_id)
        if not job:
            return error_response("Job not found", 404)

        return json_response(
            {
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", 0),
                "result": job.get("result"),
                "error": job.get("error"),
            }
        )

    def _handle_audio_transcription(self, handler) -> HandlerResult:
        """Handle audio file transcription."""
        available, error = _check_transcription_available()
        if not available:
            return error_response(error, 503)

        # Parse multipart form data
        try:
            content_type = handler.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                return error_response("Expected multipart/form-data", 400)

            # Read the file from the request
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > MAX_AUDIO_SIZE_MB * 1024 * 1024:
                return error_response(
                    f"File too large. Max: {MAX_AUDIO_SIZE_MB}MB", 413
                )

            # Parse multipart data
            file_data, filename, params = self._parse_multipart(handler, content_type)
            if not file_data:
                return error_response("No file provided", 400)

            # Validate file extension
            suffix = Path(filename).suffix.lower()
            if suffix not in AUDIO_FORMATS:
                return error_response(
                    f"Unsupported format: {suffix}. Supported: {list(AUDIO_FORMATS)}", 400
                )

            # Save to temp file
            temp_path = Path(tempfile.mktemp(suffix=suffix))
            temp_path.write_bytes(file_data)

            try:
                # Create job
                job_id = str(uuid.uuid4())
                _transcription_jobs[job_id] = {
                    "status": "processing",
                    "progress": 0,
                    "filename": filename,
                }

                # Run transcription (synchronous for now, can be made async with job queue)
                from aragora.transcription import transcribe_audio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    language = params.get("language")
                    backend = params.get("backend")
                    result = loop.run_until_complete(
                        transcribe_audio(temp_path, language=language, backend=backend)
                    )
                finally:
                    loop.close()

                _transcription_jobs[job_id] = {
                    "status": "completed",
                    "progress": 100,
                    "result": result.to_dict(),
                }

                return json_response(
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "segments": [
                            {
                                "start": s.start,
                                "end": s.end,
                                "text": s.text,
                            }
                            for s in result.segments
                        ],
                        "backend": result.backend,
                        "processing_time": result.processing_time,
                    }
                )

            finally:
                # Cleanup temp file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return error_response(f"Transcription failed: {str(e)}", 500)

    def _handle_video_transcription(self, handler) -> HandlerResult:
        """Handle video file transcription (extract audio and transcribe)."""
        available, error = _check_transcription_available()
        if not available:
            return error_response(error, 503)

        try:
            content_type = handler.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                return error_response("Expected multipart/form-data", 400)

            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                return error_response(
                    f"File too large. Max: {MAX_VIDEO_SIZE_MB}MB", 413
                )

            file_data, filename, params = self._parse_multipart(handler, content_type)
            if not file_data:
                return error_response("No file provided", 400)

            suffix = Path(filename).suffix.lower()
            if suffix not in VIDEO_FORMATS:
                return error_response(
                    f"Unsupported format: {suffix}. Supported: {list(VIDEO_FORMATS)}", 400
                )

            temp_path = Path(tempfile.mktemp(suffix=suffix))
            temp_path.write_bytes(file_data)

            try:
                job_id = str(uuid.uuid4())
                _transcription_jobs[job_id] = {
                    "status": "processing",
                    "progress": 0,
                    "filename": filename,
                }

                from aragora.transcription import transcribe_video

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    language = params.get("language")
                    backend = params.get("backend")
                    result = loop.run_until_complete(
                        transcribe_video(temp_path, language=language, backend=backend)
                    )
                finally:
                    loop.close()

                _transcription_jobs[job_id] = {
                    "status": "completed",
                    "progress": 100,
                    "result": result.to_dict(),
                }

                return json_response(
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "segments": [
                            {"start": s.start, "end": s.end, "text": s.text}
                            for s in result.segments
                        ],
                        "backend": result.backend,
                        "processing_time": result.processing_time,
                    }
                )

            finally:
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"Video transcription failed: {e}", exc_info=True)
            return error_response(f"Transcription failed: {str(e)}", 500)

    def _handle_youtube_transcription(self, handler) -> HandlerResult:
        """Handle YouTube video transcription."""
        available, error = _check_transcription_available()
        if not available:
            return error_response(error, 503)

        try:
            # Parse JSON body
            body, err = self.read_json_body_validated(handler)
            if err:
                return err

            url = body.get("url")
            if not url:
                return error_response("Missing 'url' field", 400)

            # Validate YouTube URL
            from aragora.transcription.youtube import YouTubeFetcher

            if not YouTubeFetcher.is_youtube_url(url):
                return error_response("Invalid YouTube URL", 400)

            job_id = str(uuid.uuid4())
            _transcription_jobs[job_id] = {
                "status": "processing",
                "progress": 0,
                "url": url,
            }

            try:
                from aragora.transcription import transcribe_youtube

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    language = body.get("language")
                    backend = body.get("backend")
                    use_cache = body.get("use_cache", True)
                    result = loop.run_until_complete(
                        transcribe_youtube(
                            url,
                            language=language,
                            backend=backend,
                            use_cache=use_cache,
                        )
                    )
                finally:
                    loop.close()

                _transcription_jobs[job_id] = {
                    "status": "completed",
                    "progress": 100,
                    "result": result.to_dict(),
                }

                return json_response(
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "text": result.text,
                        "language": result.language,
                        "duration": result.duration,
                        "segments": [
                            {"start": s.start, "end": s.end, "text": s.text}
                            for s in result.segments
                        ],
                        "backend": result.backend,
                        "processing_time": result.processing_time,
                    }
                )

            except ValueError as e:
                # Video too long or other validation error
                _transcription_jobs[job_id] = {
                    "status": "failed",
                    "error": str(e),
                }
                return error_response(str(e), 400)

        except Exception as e:
            logger.error(f"YouTube transcription failed: {e}", exc_info=True)
            return error_response(f"Transcription failed: {str(e)}", 500)

    def _handle_youtube_info(self, handler) -> HandlerResult:
        """Get YouTube video info without transcribing."""
        try:
            body, err = self.read_json_body_validated(handler)
            if err:
                return err

            url = body.get("url")
            if not url:
                return error_response("Missing 'url' field", 400)

            from aragora.transcription.youtube import YouTubeFetcher

            if not YouTubeFetcher.is_youtube_url(url):
                return error_response("Invalid YouTube URL", 400)

            fetcher = YouTubeFetcher()

            # Get video info synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                info = loop.run_until_complete(fetcher.get_video_info(url))
            finally:
                loop.close()

            return json_response(
                {
                    "video_id": info.video_id,
                    "title": info.title,
                    "duration": info.duration,
                    "channel": info.channel,
                    "description": info.description[:500] if info.description else None,
                    "upload_date": info.upload_date,
                    "view_count": info.view_count,
                    "thumbnail_url": info.thumbnail_url,
                }
            )

        except RuntimeError as e:
            # yt-dlp not installed or other runtime issues
            return error_response(str(e), 503)
        except ValueError as e:
            # Invalid URL
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Failed to get YouTube info: {e}", exc_info=True)
            return error_response(f"Failed to get video info: {str(e)}", 500)

    def _parse_multipart(
        self, handler, content_type: str
    ) -> tuple[Optional[bytes], str, dict]:
        """Parse multipart form data.

        Returns:
            Tuple of (file_data, filename, params)
        """
        import cgi
        import io

        try:
            # Get boundary from content type
            _, pdict = cgi.parse_header(content_type)
            boundary = pdict.get("boundary")
            if not boundary:
                return None, "", {}

            # Read body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Parse multipart
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(len(body)),
            }

            # Use cgi.FieldStorage for parsing
            fs = cgi.FieldStorage(
                fp=io.BytesIO(body),
                environ=environ,
                keep_blank_values=True,
            )

            file_data = None
            filename = ""
            params = {}

            for key in fs.keys():
                item = fs[key]
                if item.filename:
                    # This is a file
                    file_data = item.file.read()
                    filename = item.filename
                else:
                    # This is a form field
                    params[key] = item.value

            return file_data, filename, params

        except Exception as e:
            logger.error(f"Failed to parse multipart: {e}")
            return None, "", {}

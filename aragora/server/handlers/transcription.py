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
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Lazy-loaded job store for persistence
_job_store = None


def _get_job_store():
    """Get or create the job store for transcription job persistence."""
    global _job_store
    if _job_store is None:
        try:
            from aragora.storage.job_queue_store import get_job_store

            _job_store = get_job_store()
        except Exception as e:
            logger.debug(f"Job store not available: {e}")
    return _job_store


# Rate limiters (per minute limits)
_audio_limiter = RateLimiter(requests_per_minute=10)
_youtube_limiter = RateLimiter(requests_per_minute=5)

# Maximum file sizes
MAX_AUDIO_SIZE_MB = 100
MAX_VIDEO_SIZE_MB = 500

# Supported formats
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
VIDEO_FORMATS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# In-memory job cache (backed by durable store)
_transcription_jobs: Dict[str, Dict[str, Any]] = {}


def _save_job(job_id: str, job_data: Dict[str, Any]) -> None:
    """Save a transcription job to both memory and durable store."""
    _transcription_jobs[job_id] = job_data

    # Persist to job store for durability
    store = _get_job_store()
    if store:
        try:
            from aragora.storage.job_queue_store import QueuedJob, JobStatus

            status_map = {
                "processing": JobStatus.PROCESSING,
                "completed": JobStatus.COMPLETED,
                "failed": JobStatus.FAILED,
            }

            job = QueuedJob(
                id=job_id,
                job_type="transcription",
                payload=job_data,
                status=status_map.get(job_data.get("status", "pending"), JobStatus.PENDING),
                result=job_data.get("result"),
                error=job_data.get("error"),
            )

            # Use sync wrapper since handler is sync
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule for later
                    asyncio.ensure_future(store.enqueue(job))
                else:
                    loop.run_until_complete(store.enqueue(job))
            except RuntimeError:
                # No event loop, create one
                asyncio.run(store.enqueue(job))
        except Exception as e:
            logger.debug(f"Failed to persist transcription job: {e}")


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a transcription job from memory cache or durable store."""
    # Check memory cache first
    if job_id in _transcription_jobs:
        return _transcription_jobs[job_id]

    # Try durable store
    store = _get_job_store()
    if store:
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't block, return None for now
                    return None
                job = loop.run_until_complete(store.get(job_id))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    job = loop.run_until_complete(store.get(job_id))
                finally:
                    loop.close()

            if job:
                # Cache in memory
                job_data = {
                    "status": job.status.value if hasattr(job.status, "value") else job.status,
                    "progress": 100 if job.status == "completed" else 0,
                    "result": job.result,
                    "error": job.error,
                    **job.payload,
                }
                _transcription_jobs[job_id] = job_data
                return job_data
        except Exception as e:
            logger.debug(f"Failed to load transcription job from store: {e}")

    return None


def _check_transcription_available() -> tuple[bool, Optional[str]]:
    """Check if transcription module is available."""
    try:
        from aragora.transcription import get_available_backends

        backends = get_available_backends()
        if not backends:
            return False, (
                "No transcription backend available. Set OPENAI_API_KEY or install faster-whisper."
            )
        return True, None
    except ImportError:
        return False, "Transcription module not installed."


class TranscriptionHandler(BaseHandler):
    """Handler for audio/video transcription endpoints."""

    ROUTES = [
        "/api/v1/transcription/audio",
        "/api/v1/transcription/video",
        "/api/v1/transcription/youtube",
        "/api/v1/transcription/youtube/info",
        "/api/v1/transcription/status/*",
        "/api/v1/transcription/config",
        # Alias routes for frontend compatibility
        "/api/v1/transcribe/audio",
        "/api/v1/transcribe/video",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path in (
            "/api/v1/transcription/audio",
            "/api/v1/transcription/video",
            "/api/v1/transcription/youtube",
            "/api/v1/transcription/youtube/info",
            "/api/v1/transcription/config",
            # Alias routes
            "/api/v1/transcribe/audio",
            "/api/v1/transcribe/video",
        ):
            return True
        if path.startswith("/api/v1/transcription/status/"):
            return True
        return False

    @require_permission("transcription:read")
    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/v1/transcription/config":
            return self._get_config()

        if path.startswith("/api/v1/transcription/status/"):
            job_id = path.split("/")[-1]
            return self._get_status(job_id)

        return None

    async def handle_post(
        self, path: str, query_params: dict, handler=None
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        client_ip = get_client_ip(handler)

        if path in ("/api/v1/transcription/audio", "/api/v1/transcribe/audio"):
            if not _audio_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return await self._handle_audio_transcription(handler)

        if path in ("/api/v1/transcription/video", "/api/v1/transcribe/video"):
            if not _audio_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return await self._handle_video_transcription(handler)

        if path == "/api/v1/transcription/youtube":
            if not _youtube_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Try again later.", 429)
            return await self._handle_youtube_transcription(handler)

        if path == "/api/v1/transcription/youtube/info":
            return await self._handle_youtube_info(handler)

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
        except (ImportError, AttributeError) as e:
            logger.warning(f"Transcription module not available: {e}")
            return error_response("Transcription service not fully configured", 503)
        except Exception as e:
            logger.exception(f"Unexpected error getting transcription config: {e}")
            return error_response("Failed to get configuration", 500)

    def _get_status(self, job_id: str) -> HandlerResult:
        """Get transcription job status."""
        job = _get_job(job_id)
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
                return error_response(f"File too large. Max: {MAX_AUDIO_SIZE_MB}MB", 413)

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
                _save_job(
                    job_id,
                    {
                        "status": "processing",
                        "progress": 0,
                        "filename": filename,
                    },
                )

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

                _save_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "result": result.to_dict(),
                    },
                )

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

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid transcription request data: {e}")
            return error_response(safe_error_message(e, "transcription"), 400)
        except (OSError, IOError) as e:
            logger.error(f"File I/O error during transcription: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)
        except Exception as e:
            logger.exception(f"Unexpected transcription error: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)

    async def _handle_video_transcription(self, handler) -> HandlerResult:
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
                return error_response(f"File too large. Max: {MAX_VIDEO_SIZE_MB}MB", 413)

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
                _save_job(
                    job_id,
                    {
                        "status": "processing",
                        "progress": 0,
                        "filename": filename,
                    },
                )

                from aragora.transcription import transcribe_video

                language = params.get("language")
                backend = params.get("backend")
                result = await transcribe_video(temp_path, language=language, backend=backend)

                _save_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "result": result.to_dict(),
                    },
                )

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

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid video transcription request data: {e}")
            return error_response(safe_error_message(e, "transcription"), 400)
        except (OSError, IOError) as e:
            logger.error(f"File I/O error during video transcription: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)
        except Exception as e:
            logger.exception(f"Unexpected video transcription error: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)

    async def _handle_youtube_transcription(self, handler) -> HandlerResult:
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
            _save_job(
                job_id,
                {
                    "status": "processing",
                    "progress": 0,
                    "url": url,
                },
            )

            try:
                from aragora.transcription import transcribe_youtube

                language = body.get("language")
                backend = body.get("backend")
                use_cache = body.get("use_cache", True)
                result = await transcribe_youtube(
                    url,
                    language=language,
                    backend=backend,
                    use_cache=use_cache,
                )

                _save_job(
                    job_id,
                    {
                        "status": "completed",
                        "progress": 100,
                        "result": result.to_dict(),
                    },
                )

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
                _save_job(
                    job_id,
                    {
                        "status": "failed",
                        "error": str(e),
                    },
                )
                return error_response(str(e), 400)

        except (KeyError, TypeError) as e:
            logger.warning(f"Invalid YouTube transcription request data: {e}")
            return error_response(safe_error_message(e, "transcription"), 400)
        except (OSError, IOError) as e:
            logger.error(f"File I/O error during YouTube transcription: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)
        except Exception as e:
            logger.exception(f"Unexpected YouTube transcription error: {e}")
            return error_response(safe_error_message(e, "transcription"), 500)

    async def _handle_youtube_info(self, handler) -> HandlerResult:
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
            info = await fetcher.get_video_info(url)

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
            return error_response(safe_error_message(e, "transcription service"), 503)
        except ValueError as e:
            # Invalid URL
            return error_response(str(e), 400)
        except (KeyError, TypeError) as e:
            logger.warning(f"Invalid YouTube info request data: {e}")
            return error_response(safe_error_message(e, "video info"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting YouTube info: {e}")
            return error_response(safe_error_message(e, "video info"), 500)

    def _parse_multipart(self, handler, content_type: str) -> tuple[Optional[bytes], str, dict]:
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

        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid multipart form data: {e}")
            return None, "", {}
        except (OSError, IOError) as e:
            logger.error(f"I/O error parsing multipart data: {e}")
            return None, "", {}
        except Exception as e:
            logger.exception(f"Unexpected error parsing multipart data: {e}")
            return None, "", {}

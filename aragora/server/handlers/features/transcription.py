"""
Audio/Video transcription endpoint handlers.

Endpoints:
- GET /api/transcription/formats - Get supported audio/video formats
- GET /api/transcription/{job_id} - Get transcription job status/result
- GET /api/transcription/{job_id}/segments - Get timestamped segments
- POST /api/transcription/upload - Upload and transcribe audio/video
- DELETE /api/transcription/{job_id} - Delete a transcription
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)

logger = logging.getLogger(__name__)

# File size limits (Whisper API limit is 25MB)
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_FILE_SIZE = 1  # Minimum 1 byte

# Supported extensions
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".mpga", ".mpeg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
ALL_SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# DoS protection
MAX_MULTIPART_PARTS = 10
MAX_FILENAME_LENGTH = 255


class TranscriptionErrorCode(Enum):
    """Specific error codes for transcription failures."""

    RATE_LIMITED = "rate_limited"
    FILE_TOO_LARGE = "file_too_large"
    FILE_TOO_SMALL = "file_too_small"
    INVALID_CONTENT_LENGTH = "invalid_content_length"
    NO_CONTENT = "no_content"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_FILENAME = "invalid_filename"
    FILENAME_TOO_LONG = "filename_too_long"
    CORRUPTED_UPLOAD = "corrupted_upload"
    MULTIPART_PARSE_ERROR = "multipart_parse_error"
    MISSING_BOUNDARY = "missing_boundary"
    TRANSCRIPTION_FAILED = "transcription_failed"
    JOB_NOT_FOUND = "job_not_found"
    API_NOT_CONFIGURED = "api_not_configured"
    QUOTA_EXCEEDED = "quota_exceeded"


class TranscriptionStatus(Enum):
    """Status of a transcription job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionError:
    """Structured transcription error response."""

    code: TranscriptionErrorCode
    message: str
    details: Optional[dict] = None

    def to_response(self, status: int = 400) -> HandlerResult:
        """Convert to error response."""
        payload: dict[str, Any] = {
            "error": self.message,
            "error_code": self.code.value,
        }
        if self.details:
            payload["details"] = self.details
        return json_response(payload, status=status)


@dataclass
class TranscriptionJob:
    """Tracks a transcription job."""

    id: str
    filename: str
    status: TranscriptionStatus
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    transcription_id: Optional[str] = None
    error: Optional[str] = None
    file_size_bytes: int = 0
    duration_seconds: Optional[float] = None
    text: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    segments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "transcription_id": self.transcription_id,
            "error": self.error,
            "file_size_bytes": self.file_size_bytes,
            "duration_seconds": self.duration_seconds,
            "text": self.text,
            "language": self.language,
            "word_count": self.word_count,
            "segment_count": len(self.segments),
        }


class TranscriptionHandler(BaseHandler):
    """Handler for audio/video transcription endpoints."""

    ROUTES = [
        "/api/transcription/upload",
        "/api/transcription/formats",
    ]

    # Rate limiting (stricter than documents due to API cost)
    _upload_counts: OrderedDict[str, list] = OrderedDict()
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 3
    MAX_UPLOADS_PER_HOUR = 20
    MAX_TRACKED_IPS = 5000

    # In-memory job tracking (could be moved to database for persistence)
    _jobs: dict[str, TranscriptionJob] = {}
    _jobs_lock = threading.Lock()
    MAX_JOBS = 1000  # Prevent unbounded memory growth

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/transcription/{job_id} and /api/transcription/{job_id}/segments
        if path.startswith("/api/transcription/") and path.count("/") >= 3:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/transcription/formats":
            return self._get_supported_formats()

        if path.startswith("/api/transcription/"):
            parts = path.split("/")
            if len(parts) >= 4 and parts[3] != "upload":
                job_id = parts[3]

                if len(parts) == 5 and parts[4] == "segments":
                    return self._get_job_segments(job_id)
                elif len(parts) == 4:
                    return self._get_job_status(job_id)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/transcription/upload":
            return self._upload_and_transcribe(handler)
        return None

    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE requests to appropriate methods."""
        if path.startswith("/api/transcription/"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] != "upload":
                job_id = parts[3]
                return self._delete_job(job_id)
        return None

    def _get_supported_formats(self) -> HandlerResult:
        """Get list of supported audio/video formats."""
        return json_response({
            "audio": sorted(list(AUDIO_EXTENSIONS)),
            "video": sorted(list(VIDEO_EXTENSIONS)),
            "max_size_mb": MAX_FILE_SIZE_MB,
            "model": "whisper-1",
            "note": "Video files have audio extracted before transcription",
        })

    def _get_job_status(self, job_id: str) -> HandlerResult:
        """Get transcription job status and result."""
        with TranscriptionHandler._jobs_lock:
            job = TranscriptionHandler._jobs.get(job_id)

        if not job:
            return TranscriptionError(
                TranscriptionErrorCode.JOB_NOT_FOUND,
                f"Transcription job not found: {job_id}",
            ).to_response(404)

        return json_response(job.to_dict())

    def _get_job_segments(self, job_id: str) -> HandlerResult:
        """Get timestamped segments for a transcription job."""
        with TranscriptionHandler._jobs_lock:
            job = TranscriptionHandler._jobs.get(job_id)

        if not job:
            return TranscriptionError(
                TranscriptionErrorCode.JOB_NOT_FOUND,
                f"Transcription job not found: {job_id}",
            ).to_response(404)

        if job.status != TranscriptionStatus.COMPLETED:
            return json_response({
                "job_id": job_id,
                "status": job.status.value,
                "segments": [],
                "message": f"Job is {job.status.value}, segments available when completed",
            })

        return json_response({
            "job_id": job_id,
            "status": job.status.value,
            "segments": job.segments,
            "segment_count": len(job.segments),
        })

    def _delete_job(self, job_id: str) -> HandlerResult:
        """Delete a transcription job."""
        with TranscriptionHandler._jobs_lock:
            if job_id not in TranscriptionHandler._jobs:
                return TranscriptionError(
                    TranscriptionErrorCode.JOB_NOT_FOUND,
                    f"Transcription job not found: {job_id}",
                ).to_response(404)

            del TranscriptionHandler._jobs[job_id]

        logger.info(f"Transcription job deleted: {job_id}")
        return json_response({
            "success": True,
            "message": f"Transcription job {job_id} deleted",
        })

    def _check_rate_limit(self, handler) -> Optional[HandlerResult]:
        """Check IP-based upload rate limit."""
        client_ip = self._get_client_ip(handler)

        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with TranscriptionHandler._upload_counts_lock:
            if client_ip not in TranscriptionHandler._upload_counts:
                TranscriptionHandler._upload_counts[client_ip] = []
            else:
                TranscriptionHandler._upload_counts.move_to_end(client_ip)

            # Clean up old entries
            TranscriptionHandler._upload_counts[client_ip] = [
                ts for ts in TranscriptionHandler._upload_counts[client_ip]
                if ts > one_hour_ago
            ]

            if not TranscriptionHandler._upload_counts[client_ip]:
                del TranscriptionHandler._upload_counts[client_ip]
                TranscriptionHandler._upload_counts[client_ip] = []

            timestamps = TranscriptionHandler._upload_counts[client_ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= TranscriptionHandler.MAX_UPLOADS_PER_MINUTE:
                return TranscriptionError(
                    TranscriptionErrorCode.RATE_LIMITED,
                    f"Transcription rate limit exceeded. Max {TranscriptionHandler.MAX_UPLOADS_PER_MINUTE} per minute.",
                ).to_response(429)

            # Check per-hour limit
            if len(timestamps) >= TranscriptionHandler.MAX_UPLOADS_PER_HOUR:
                return TranscriptionError(
                    TranscriptionErrorCode.RATE_LIMITED,
                    f"Transcription rate limit exceeded. Max {TranscriptionHandler.MAX_UPLOADS_PER_HOUR} per hour.",
                ).to_response(429)

            # LRU eviction
            while len(TranscriptionHandler._upload_counts) > TranscriptionHandler.MAX_TRACKED_IPS:
                TranscriptionHandler._upload_counts.popitem(last=False)

            TranscriptionHandler._upload_counts[client_ip].append(now)

        return None

    def _get_client_ip(self, handler) -> str:
        """Get client IP address."""
        return handler.client_address[0] if hasattr(handler, "client_address") else "unknown"

    def _create_job(self, filename: str, file_size: int) -> TranscriptionJob:
        """Create and register a new transcription job."""
        job_id = f"trans_{uuid.uuid4().hex[:12]}"
        job = TranscriptionJob(
            id=job_id,
            filename=filename,
            status=TranscriptionStatus.PENDING,
            file_size_bytes=file_size,
        )

        with TranscriptionHandler._jobs_lock:
            # LRU eviction if needed
            while len(TranscriptionHandler._jobs) >= TranscriptionHandler.MAX_JOBS:
                # Find oldest completed/failed job to remove
                oldest_key = None
                oldest_time = float("inf")
                for k, v in TranscriptionHandler._jobs.items():
                    if v.status in (TranscriptionStatus.COMPLETED, TranscriptionStatus.FAILED):
                        if v.created_at < oldest_time:
                            oldest_time = v.created_at
                            oldest_key = k
                if oldest_key:
                    del TranscriptionHandler._jobs[oldest_key]
                else:
                    # No completed jobs, just remove oldest
                    oldest_key = min(
                        TranscriptionHandler._jobs.keys(),
                        key=lambda k: TranscriptionHandler._jobs[k].created_at
                    )
                    del TranscriptionHandler._jobs[oldest_key]

            TranscriptionHandler._jobs[job_id] = job

        return job

    def _update_job(
        self,
        job_id: str,
        status: TranscriptionStatus,
        **kwargs,
    ) -> None:
        """Update a transcription job."""
        with TranscriptionHandler._jobs_lock:
            if job_id in TranscriptionHandler._jobs:
                job = TranscriptionHandler._jobs[job_id]
                job.status = status
                if status == TranscriptionStatus.COMPLETED:
                    job.completed_at = time.time()
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)

    @require_user_auth
    @handle_errors("transcription upload")
    def _upload_and_transcribe(self, handler, user=None) -> HandlerResult:
        """Handle audio/video upload and queue for transcription."""
        # Check rate limit
        rate_error = self._check_rate_limit(handler)
        if rate_error:
            return rate_error

        # Validate content length
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return TranscriptionError(
                TranscriptionErrorCode.INVALID_CONTENT_LENGTH,
                "Invalid Content-Length header",
            ).to_response(400)

        if content_length == 0:
            return TranscriptionError(
                TranscriptionErrorCode.NO_CONTENT,
                "No content provided. Include audio/video file in request body.",
            ).to_response(400)

        if content_length < MIN_FILE_SIZE:
            return TranscriptionError(
                TranscriptionErrorCode.FILE_TOO_SMALL,
                f"File too small. Minimum size: {MIN_FILE_SIZE} bytes",
            ).to_response(400)

        if content_length > MAX_FILE_SIZE_BYTES:
            return TranscriptionError(
                TranscriptionErrorCode.FILE_TOO_LARGE,
                f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB (Whisper API limit)",
                {"received_bytes": content_length, "max_bytes": MAX_FILE_SIZE_BYTES},
            ).to_response(413)

        # Parse file from request
        content_type = handler.headers.get("Content-Type", "")
        file_content, filename, parse_error = self._parse_upload(
            handler, content_type, content_length
        )

        if parse_error:
            return parse_error.to_response(400)

        if not file_content or not filename:
            return TranscriptionError(
                TranscriptionErrorCode.CORRUPTED_UPLOAD,
                "Could not extract file from upload",
            ).to_response(400)

        # Validate filename
        if len(filename) > MAX_FILENAME_LENGTH:
            return TranscriptionError(
                TranscriptionErrorCode.FILENAME_TOO_LONG,
                f"Filename too long. Maximum: {MAX_FILENAME_LENGTH} characters",
            ).to_response(400)

        # Validate extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in ALL_SUPPORTED_EXTENSIONS:
            return TranscriptionError(
                TranscriptionErrorCode.UNSUPPORTED_FORMAT,
                f"Unsupported file type: {ext}",
                {"extension": ext, "supported": sorted(list(ALL_SUPPORTED_EXTENSIONS))},
            ).to_response(400)

        # Verify content matches declared length
        if len(file_content) != content_length:
            return TranscriptionError(
                TranscriptionErrorCode.CORRUPTED_UPLOAD,
                "Upload appears truncated or corrupted",
                {"expected_bytes": content_length, "received_bytes": len(file_content)},
            ).to_response(400)

        # Create job
        job = self._create_job(filename, len(file_content))

        # Queue for async processing
        asyncio.create_task(self._process_transcription(job.id, file_content, filename))

        logger.info(f"Transcription job created: {job.id} for {filename} ({len(file_content)} bytes)")

        return json_response({
            "success": True,
            "job_id": job.id,
            "filename": filename,
            "file_size_bytes": len(file_content),
            "status": "pending",
            "message": f"Transcription queued. Poll /api/transcription/{job.id} for status.",
        }, status=202)

    async def _process_transcription(
        self,
        job_id: str,
        content: bytes,
        filename: str,
    ) -> None:
        """Background task to process transcription."""
        self._update_job(job_id, TranscriptionStatus.PROCESSING)

        try:
            from aragora.connectors.whisper import WhisperConnector

            connector = WhisperConnector()

            if not connector.is_available:
                self._update_job(
                    job_id,
                    TranscriptionStatus.FAILED,
                    error="OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
                )
                return

            result = await connector.transcribe(content, filename)

            # Update job with result
            self._update_job(
                job_id,
                TranscriptionStatus.COMPLETED,
                transcription_id=result.id,
                text=result.text,
                language=result.language,
                duration_seconds=result.duration_seconds,
                word_count=result.word_count,
                segments=[s.to_dict() for s in result.segments],
            )

            logger.info(
                f"Transcription completed: {job_id} - "
                f"{result.word_count} words, {result.duration_seconds:.1f}s duration"
            )

        except Exception as e:
            logger.error(f"Transcription failed for {job_id}: {e}")
            self._update_job(
                job_id,
                TranscriptionStatus.FAILED,
                error=str(e)[:500],
            )

    def _parse_upload(
        self,
        handler,
        content_type: str,
        content_length: int,
    ) -> tuple[Optional[bytes], Optional[str], Optional[TranscriptionError]]:
        """Parse file content and filename from upload request."""
        if "multipart/form-data" in content_type:
            return self._parse_multipart(handler, content_type, content_length)
        else:
            return self._parse_raw_upload(handler, content_length)

    def _parse_multipart(
        self,
        handler,
        content_type: str,
        content_length: int,
    ) -> tuple[Optional[bytes], Optional[str], Optional[TranscriptionError]]:
        """Parse multipart form data upload."""
        # Extract boundary
        boundary = None
        for part in content_type.split(";"):
            if "boundary=" in part:
                parts = part.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    boundary = parts[1].strip().strip('"')
                break

        if not boundary:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.MISSING_BOUNDARY,
                "Missing boundary in multipart/form-data Content-Type header",
            ))

        try:
            body: bytes = handler.rfile.read(content_length)
        except Exception as e:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.CORRUPTED_UPLOAD,
                f"Failed to read upload body: {str(e)[:100]}",
            ))

        boundary_bytes = f"--{boundary}".encode()
        body_parts: list[bytes] = body.split(boundary_bytes)

        if len(body_parts) > MAX_MULTIPART_PARTS:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.MULTIPART_PARSE_ERROR,
                f"Too many parts in multipart upload. Maximum: {MAX_MULTIPART_PARTS}",
            ))

        for part in body_parts:
            if b"Content-Disposition" not in part:
                continue

            try:
                header_end = part.index(b"\r\n\r\n")
                headers_raw = part[:header_end].decode("utf-8", errors="ignore")
                file_data = part[header_end + 4:]

                # Remove trailing boundary markers
                if file_data.endswith(b"--\r\n"):
                    file_data = file_data[:-4]
                elif file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]

                # Extract filename
                if 'filename="' in headers_raw:
                    start = headers_raw.index('filename="') + 10
                    end = headers_raw.index('"', start)
                    raw_filename = headers_raw[start:end]
                    filename = os.path.basename(raw_filename)

                    if not filename:
                        return (None, None, TranscriptionError(
                            TranscriptionErrorCode.INVALID_FILENAME,
                            "Empty filename in upload",
                        ))
                    if "\x00" in filename or ".." in filename:
                        return (None, None, TranscriptionError(
                            TranscriptionErrorCode.INVALID_FILENAME,
                            "Invalid filename (potential path traversal)",
                        ))

                    return (file_data, filename, None)

            except (ValueError, IndexError):
                continue

        return (None, None, TranscriptionError(
            TranscriptionErrorCode.MULTIPART_PARSE_ERROR,
            "No file found in multipart upload",
        ))

    def _parse_raw_upload(
        self,
        handler,
        content_length: int,
    ) -> tuple[Optional[bytes], Optional[str], Optional[TranscriptionError]]:
        """Parse raw file upload with X-Filename header."""
        filename = handler.headers.get("X-Filename")

        if not filename:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.INVALID_FILENAME,
                "Missing filename. Use multipart/form-data or set X-Filename header.",
            ))

        filename = os.path.basename(filename)
        if not filename or "\x00" in filename or ".." in filename:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.INVALID_FILENAME,
                "Invalid filename",
            ))

        try:
            content = handler.rfile.read(content_length)
            return (content, filename, None)
        except Exception as e:
            return (None, None, TranscriptionError(
                TranscriptionErrorCode.CORRUPTED_UPLOAD,
                f"Failed to read upload: {str(e)[:100]}",
            ))

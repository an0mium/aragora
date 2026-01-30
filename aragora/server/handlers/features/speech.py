"""
Speech-to-text API handlers using the aragora.speech module.

Provides simplified endpoints for speech transcription.

Endpoints:
- POST /api/speech/transcribe - Transcribe uploaded audio
- POST /api/speech/transcribe-url - Transcribe from URL
- GET /api/speech/providers - List available STT providers
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)
from ..utils.rate_limit import RateLimiter, get_client_ip
from ..utils.responses import error_dict
from aragora.rbac.decorators import require_permission
from aragora.utils.async_utils import run_async

logger = logging.getLogger(__name__)

# Rate limiter for speech endpoints (10 requests per minute - resource intensive)
_speech_limiter = RateLimiter(requests_per_minute=10)

# File size limits
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported extensions
SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".mpga", ".mpeg", ".ogg", ".flac"}


class SpeechHandler(BaseHandler):
    """Handler for speech-to-text endpoints."""

    ROUTES = [
        "/api/v1/speech/transcribe",
        "/api/v1/speech/transcribe-url",
        "/api/v1/speech/providers",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> HandlerResult | None:
        """Route GET requests."""
        if path == "/api/v1/speech/providers":
            return self._get_providers()
        return None

    @require_permission("speech:create")
    def handle_post(self, path: str, query_params: dict, handler) -> HandlerResult | None:
        """Route POST requests."""
        # Rate limit check for resource-intensive speech operations
        client_ip = get_client_ip(handler)
        if not _speech_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for speech endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/v1/speech/transcribe":
            return self._transcribe_upload(handler, query_params)
        elif path == "/api/v1/speech/transcribe-url":
            return self._transcribe_from_url(handler, query_params)
        return None

    def _get_providers(self) -> HandlerResult:
        """Get list of available STT providers."""
        providers = [
            {
                "name": "openai_whisper",
                "display_name": "OpenAI Whisper",
                "model": "whisper-1",
                "available": bool(os.getenv("OPENAI_API_KEY")),
                "formats": list(SUPPORTED_EXTENSIONS),
                "max_size_mb": MAX_FILE_SIZE_MB,
                "features": ["timestamps", "language_detection", "word_timestamps"],
            },
        ]

        # Mark default provider
        default_provider = os.getenv("ARAGORA_STT_PROVIDER", "openai_whisper")
        for p in providers:
            p["is_default"] = p["name"] == default_provider

        return json_response(
            {
                "providers": providers,
                "default": default_provider,
            }
        )

    @require_user_auth
    @handle_errors("speech transcription")
    def _transcribe_upload(self, handler, query_params: dict, user=None) -> HandlerResult:
        """Transcribe uploaded audio file."""
        # Validate content length
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return error_response("Invalid Content-Length header", 400)

        if content_length == 0:
            return error_response("No audio file provided", 400)

        if content_length > MAX_FILE_SIZE_BYTES:
            return json_response(
                {
                    "error": f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
                    "max_bytes": MAX_FILE_SIZE_BYTES,
                },
                status=413,
            )

        # Parse options from query params
        language = query_params.get("language", [None])[0]
        prompt = query_params.get("prompt", [None])[0]
        provider = query_params.get("provider", [None])[0]
        include_timestamps = query_params.get("timestamps", ["true"])[0].lower() == "true"

        # Parse file from request
        content_type = handler.headers.get("Content-Type", "")
        file_content, filename = self._parse_upload(handler, content_type, content_length)

        if not file_content:
            return error_response("Could not extract file from upload", 400)

        # Validate extension
        ext = ("." + filename.split(".")[-1].lower()) if filename and "." in filename else ""
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return json_response(
                {
                    "error": f"Unsupported audio format: {ext}",
                    "supported": sorted(list(SUPPORTED_EXTENSIONS)),
                },
                status=400,
            )

        # Run transcription asynchronously
        try:
            result = run_async(
                self._do_transcription(file_content, filename, language, prompt, provider)
            )
        except RuntimeError:
            # No event loop running, create new one
            result = asyncio.run(
                self._do_transcription(file_content, filename, language, prompt, provider)
            )

        if "error" in result:
            return json_response(result, status=500)

        # Optionally strip segments if timestamps not requested
        if not include_timestamps and "segments" in result:
            del result["segments"]

        return json_response(result)

    @require_user_auth
    @handle_errors("speech transcription from URL")
    def _transcribe_from_url(self, handler, query_params: dict, user=None) -> HandlerResult:
        """Transcribe audio from URL."""
        import json

        # Parse JSON body
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return error_response("Invalid JSON body", 400)

        url = data.get("url")
        if not url:
            return error_response("Missing 'url' in request body", 400)

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return error_response("Invalid URL. Must start with http:// or https://", 400)

        language = data.get("language")
        prompt = data.get("prompt")
        provider = data.get("provider")

        # Fetch audio from URL
        try:
            import aiohttp

            async def fetch_audio():
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            return None, f"Failed to fetch URL: HTTP {resp.status}"
                        if resp.content_length and resp.content_length > MAX_FILE_SIZE_BYTES:
                            return None, f"Audio file too large. Maximum: {MAX_FILE_SIZE_MB}MB"
                        return await resp.read(), None

            try:
                content, error = run_async(fetch_audio())
            except RuntimeError:
                content, error = asyncio.run(fetch_audio())

            if error:
                return error_response(error, 400)

        except ImportError:
            return error_response("aiohttp package required for URL transcription", 500)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching audio from URL: {url}")
            return error_response("Timeout fetching audio from URL", 400)
        except OSError as e:
            logger.warning(f"Network error fetching audio: {e}")
            return error_response(safe_error_message(e, "Failed to fetch audio"), 400)

        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0] or "audio.mp3"

        # Run transcription
        try:
            result = run_async(
                self._do_transcription(content, filename, language, prompt, provider)
            )
        except RuntimeError:
            result = asyncio.run(
                self._do_transcription(content, filename, language, prompt, provider)
            )

        if "error" in result:
            return json_response(result, status=500)

        return json_response(result)

    async def _do_transcription(
        self,
        content: bytes,
        filename: str,
        language: str | None,
        prompt: str | None,
        provider_name: str | None,
    ) -> dict:
        """Perform the actual transcription."""
        try:
            from aragora.speech import transcribe_audio, STTProviderConfig

            # Create temp file
            suffix = Path(filename).suffix if filename else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                # Configure provider
                config = STTProviderConfig(include_timestamps=True)

                # Transcribe
                result = await transcribe_audio(
                    tmp_path,
                    language=language,
                    prompt=prompt,
                    provider_name=provider_name,
                    config=config,
                )

                return result.to_dict()

            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except OSError:
                    # Best effort cleanup - file may already be deleted
                    pass

        except ImportError as e:
            logger.error(f"Speech module import error: {e}")
            return error_dict("Speech transcription not available. Check server configuration.", code="SERVICE_UNAVAILABLE")
        except RuntimeError as e:
            logger.error(f"STT provider error: {e}")
            return error_dict(str(e), code="INTERNAL_ERROR", status=500)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid transcription parameters: {e}")
            return error_dict(f"Invalid parameters: {str(e)}", code="VALIDATION_ERROR")
        except OSError as e:
            logger.exception(f"IO error during transcription: {e}")
            return error_dict(f"Transcription failed: {str(e)}", code="INTERNAL_ERROR", status=500)

    def _parse_upload(
        self,
        handler,
        content_type: str,
        content_length: int,
    ) -> tuple[bytes | None, str | None]:
        """Parse file content and filename from upload request."""
        if "multipart/form-data" in content_type:
            return self._parse_multipart(handler, content_type, content_length)
        else:
            return self._parse_raw(handler, content_length)

    def _parse_multipart(
        self,
        handler,
        content_type: str,
        content_length: int,
    ) -> tuple[bytes | None, str | None]:
        """Parse multipart form data."""
        # Extract boundary
        boundary = None
        for part in content_type.split(";"):
            if "boundary=" in part:
                parts = part.split("=", 1)
                if len(parts) == 2:
                    boundary = parts[1].strip().strip('"')
                break

        if not boundary:
            return None, None

        try:
            body = handler.rfile.read(content_length)
        except OSError as e:
            logger.warning(f"Failed to read multipart body: {e}")
            return None, None

        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            try:
                header_end = part.index(b"\r\n\r\n")
                headers_raw = part[:header_end].decode("utf-8", errors="ignore")
                file_data = part[header_end + 4 :]

                # Remove trailing boundary markers
                if file_data.endswith(b"--\r\n"):
                    file_data = file_data[:-4]
                elif file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]

                # Extract filename
                if 'filename="' in headers_raw:
                    start = headers_raw.index('filename="') + 10
                    end = headers_raw.index('"', start)
                    filename = os.path.basename(headers_raw[start:end])
                    return file_data, filename

            except (ValueError, IndexError):
                continue

        return None, None

    def _parse_raw(
        self,
        handler,
        content_length: int,
    ) -> tuple[bytes | None, str | None]:
        """Parse raw file upload with X-Filename header."""
        filename = handler.headers.get("X-Filename", "audio.mp3")
        filename = os.path.basename(filename)

        try:
            content = handler.rfile.read(content_length)
            return content, filename
        except OSError as e:
            logger.warning(f"Failed to read raw upload body: {e}")
            return None, None

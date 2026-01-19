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
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    handle_errors,
    json_response,
    require_user_auth,
)

logger = logging.getLogger(__name__)

# File size limits
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Supported extensions
SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".webm", ".mpga", ".mpeg", ".ogg", ".flac"}


class SpeechHandler(BaseHandler):
    """Handler for speech-to-text endpoints."""

    ROUTES = [
        "/api/speech/transcribe",
        "/api/speech/transcribe-url",
        "/api/speech/providers",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests."""
        if path == "/api/speech/providers":
            return self._get_providers()
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests."""
        if path == "/api/speech/transcribe":
            return self._transcribe_upload(handler, query_params)
        elif path == "/api/speech/transcribe-url":
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

        return json_response({
            "providers": providers,
            "default": default_provider,
        })

    @require_user_auth
    @handle_errors("speech transcription")
    def _transcribe_upload(self, handler, query_params: dict, user=None) -> HandlerResult:
        """Transcribe uploaded audio file."""
        # Validate content length
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return json_response({"error": "Invalid Content-Length header"}, status=400)

        if content_length == 0:
            return json_response({"error": "No audio file provided"}, status=400)

        if content_length > MAX_FILE_SIZE_BYTES:
            return json_response({
                "error": f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
                "max_bytes": MAX_FILE_SIZE_BYTES,
            }, status=413)

        # Parse options from query params
        language = query_params.get("language", [None])[0]
        prompt = query_params.get("prompt", [None])[0]
        provider = query_params.get("provider", [None])[0]
        include_timestamps = query_params.get("timestamps", ["true"])[0].lower() == "true"

        # Parse file from request
        content_type = handler.headers.get("Content-Type", "")
        file_content, filename = self._parse_upload(handler, content_type, content_length)

        if not file_content:
            return json_response({"error": "Could not extract file from upload"}, status=400)

        # Validate extension
        ext = ("." + filename.split(".")[-1].lower()) if filename and "." in filename else ""
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return json_response({
                "error": f"Unsupported audio format: {ext}",
                "supported": sorted(list(SUPPORTED_EXTENSIONS)),
            }, status=400)

        # Run transcription asynchronously
        try:
            result = asyncio.get_event_loop().run_until_complete(
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
            return json_response({"error": "Invalid JSON body"}, status=400)

        url = data.get("url")
        if not url:
            return json_response({"error": "Missing 'url' in request body"}, status=400)

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return json_response({"error": "Invalid URL. Must start with http:// or https://"}, status=400)

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
                content, error = asyncio.get_event_loop().run_until_complete(fetch_audio())
            except RuntimeError:
                content, error = asyncio.run(fetch_audio())

            if error:
                return json_response({"error": error}, status=400)

        except ImportError:
            return json_response({
                "error": "aiohttp package required for URL transcription"
            }, status=500)
        except Exception as e:
            return json_response({"error": f"Failed to fetch audio: {str(e)}"}, status=400)

        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0] or "audio.mp3"

        # Run transcription
        try:
            result = asyncio.get_event_loop().run_until_complete(
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
        language: Optional[str],
        prompt: Optional[str],
        provider_name: Optional[str],
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
                except Exception:  # noqa: BLE001 - Best effort cleanup
                    pass

        except ImportError as e:
            logger.error(f"Speech module import error: {e}")
            return {"error": "Speech transcription not available. Check server configuration."}
        except RuntimeError as e:
            logger.error(f"STT provider error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"error": f"Transcription failed: {str(e)}"}

    def _parse_upload(
        self,
        handler,
        content_type: str,
        content_length: int,
    ) -> tuple[Optional[bytes], Optional[str]]:
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
    ) -> tuple[Optional[bytes], Optional[str]]:
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
        except Exception:  # noqa: BLE001 - IO error handling
            return None, None

        boundary_bytes = f"--{boundary}".encode()
        parts = body.split(boundary_bytes)

        for part in parts:
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
                    filename = os.path.basename(headers_raw[start:end])
                    return file_data, filename

            except (ValueError, IndexError):
                continue

        return None, None

    def _parse_raw(
        self,
        handler,
        content_length: int,
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Parse raw file upload with X-Filename header."""
        filename = handler.headers.get("X-Filename", "audio.mp3")
        filename = os.path.basename(filename)

        try:
            content = handler.rfile.read(content_length)
            return content, filename
        except Exception:  # noqa: BLE001 - IO error handling
            return None, None

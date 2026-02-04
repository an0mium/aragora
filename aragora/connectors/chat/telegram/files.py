"""
Telegram Bot Connector - File Operations.

Contains file upload and download functionality.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..models import (
    FileAttachment,
    SendMessageResponse,
    VoiceMessage,
)

logger = logging.getLogger(__name__)

# Distributed tracing support
try:
    from aragora.observability.tracing import build_trace_headers
except ImportError:

    def build_trace_headers() -> dict[str, str]:
        return {}


class TelegramFilesMixin:
    """Mixin providing file operations for TelegramConnector."""

    # These are provided by TelegramConnectorBase
    bot_token: str
    _request_timeout: float

    async def _telegram_api_request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        operation: str = "api_call",
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any] | None, str | None]: ...

    async def _http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        return_raw: bool = False,
        operation: str = "http_request",
        **kwargs: Any,
    ) -> tuple[bool, bytes | dict[str, Any] | None, str | None]: ...

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload a file as a document.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target channel
            content: File content as bytes
            filename: Name of the file
            content_type: MIME type
            title: Optional display title (used as caption)
            thread_id: Optional thread for the file (reply_to_message_id)
            **kwargs: Additional options (comment alias for title)
        """
        actual_filename = filename
        file_content = content
        # Support legacy 'comment' kwarg as alias for title
        comment = kwargs.pop("comment", None) or title

        files = {"document": (actual_filename, file_content, content_type)}
        payload: dict[str, Any] = {"chat_id": channel_id}
        if comment:
            payload["caption"] = comment
        if thread_id:
            payload["reply_to_message_id"] = thread_id

        success, data, error = await self._telegram_api_request(
            "sendDocument",
            payload=payload,
            files=files,
            operation="upload_file",
            timeout=self._request_timeout * 2,
        )

        if success and data:
            doc = data.get("result", {}).get("document", {})
            return FileAttachment(
                id=doc.get("file_id", ""),
                filename=doc.get("file_name", actual_filename),
                size=doc.get("file_size", 0),
                content_type=doc.get("mime_type", "application/octet-stream"),
            )

        raise RuntimeError(error or "Upload failed")

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download a file by file_id.

        Uses _telegram_api_request and _http_request for circuit breaker,
        retry, and timeout handling.

        Args:
            file_id: Telegram file_id to download
            **kwargs: Additional options (url, filename for hints)

        Returns:
            FileAttachment with content populated
        """
        # Step 1: Get file path via API
        success, data, error = await self._telegram_api_request(
            "getFile",
            payload={"file_id": file_id},
            method="GET",
            operation="download_file_info",
            timeout=self._request_timeout * 2,
        )

        if not success or not data:
            raise RuntimeError(error or "Failed to get file info")

        file_info = data.get("result", {})
        file_path = file_info.get("file_path")
        if not file_path:
            raise RuntimeError("No file path returned")

        # Step 2: Download binary content
        download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        dl_success, content, dl_error = await self._http_request(
            method="GET",
            url=download_url,
            headers=build_trace_headers(),
            timeout=self._request_timeout * 2,
            return_raw=True,
            operation="download_file_content",
        )

        if not dl_success or not isinstance(content, bytes):
            raise RuntimeError(dl_error or "Failed to download file")

        # Extract filename from path or use kwargs hint
        filename = kwargs.get("filename") or file_path.split("/")[-1]

        # Guess content type from extension
        content_type = "application/octet-stream"
        if filename.endswith(".ogg") or filename.endswith(".oga"):
            content_type = "audio/ogg"
        elif filename.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        elif filename.endswith(".m4a"):
            content_type = "audio/mp4"

        return FileAttachment(
            id=file_id,
            filename=filename,
            content_type=content_type,
            size=file_info.get("file_size", len(content)),
            url=download_url,
            content=content,
            metadata={"telegram_file_path": file_path},
        )

    async def send_voice_message(
        self,
        channel_id: str,
        audio_content: bytes,
        filename: str = "voice_response.mp3",
        content_type: str = "audio/mpeg",
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target channel
            audio_content: Audio file content as bytes
            filename: Audio filename (default: voice_response.mp3)
            content_type: MIME type (audio/mpeg, audio/ogg, etc.)
            reply_to: Optional message ID to reply to
            **kwargs: Additional options (duration for Telegram-specific duration)
        """
        # Telegram prefers .ogg for voice, but we accept what's given
        # Map common content types to Telegram-friendly format
        actual_content_type = content_type
        actual_filename = filename
        if content_type == "audio/mpeg" and not filename.endswith(".ogg"):
            # Keep as-is, Telegram accepts various formats
            pass

        duration = kwargs.pop("duration", None)
        files = {"voice": (actual_filename, audio_content, actual_content_type)}
        payload: dict[str, Any] = {"chat_id": channel_id}
        if duration:
            payload["duration"] = str(duration)
        if reply_to:
            payload["reply_to_message_id"] = reply_to

        success, data, error = await self._telegram_api_request(
            "sendVoice",
            payload=payload,
            files=files,
            operation="send_voice_message",
            timeout=self._request_timeout * 2,
        )

        if success and data:
            msg = data.get("result", {})
            return SendMessageResponse(
                success=True,
                message_id=str(msg.get("message_id")),
                channel_id=channel_id,
                timestamp=datetime.fromtimestamp(msg.get("date", 0)).isoformat(),
            )

        return SendMessageResponse(success=False, error=error)

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message."""
        attachment = await self.download_file(voice_message.file.id)
        return attachment.content or b""

"""
File Operations Mixin for Chat Platform Connectors.

Contains methods for file upload, download, and voice message handling.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import FileAttachment, SendMessageResponse

logger = logging.getLogger(__name__)


class FileOperationsMixin:
    """
    Mixin providing file operations for chat connectors.

    Includes:
    - File upload and download
    - Voice/audio message sending
    """

    # These attributes/methods are expected from the base class
    webhook_url: str | None
    _http_request: Any
    upload_file: Any

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier."""
        raise NotImplementedError

    # ==========================================================================
    # File Operations
    # ==========================================================================

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> "FileAttachment":
        """
        Upload a file to a channel.

        Default implementation posts the file to ``webhook_url`` as a
        multipart upload if configured.  Subclasses should override for
        platform-specific file upload APIs.

        Args:
            channel_id: Target channel
            content: File content as bytes
            filename: Name of the file
            content_type: MIME type
            title: Optional display title
            thread_id: Optional thread for the file
            **kwargs: Platform-specific options

        Returns:
            FileAttachment with file ID and URL

        Raises:
            NotImplementedError: If no webhook_url is configured and
                the subclass does not override this method
        """
        from ..models import FileAttachment

        if self.webhook_url:
            success, data, error = await self._http_request(
                method="POST",
                url=self.webhook_url,
                files={"file": (filename, content, content_type)},
                data={
                    "channel_id": channel_id,
                    **({"title": title} if title else {}),
                    **({"thread_id": thread_id} if thread_id else {}),
                },
                operation="upload_file",
            )

            if success and isinstance(data, dict):
                return FileAttachment(
                    id=data.get("file_id") or data.get("id", ""),
                    filename=data.get("filename", filename),
                    content_type=content_type,
                    size=len(content),
                    url=data.get("url"),
                )

            raise RuntimeError(error or "File upload failed")

        raise NotImplementedError(
            f"{self.platform_name} upload_file requires a platform-specific override "
            f"or a configured webhook_url"
        )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> "FileAttachment":
        """
        Download a file by ID.

        Default implementation downloads from a ``url`` keyword argument
        if provided.  Subclasses should override for platform-specific
        file download APIs that require resolving ``file_id`` to a URL.

        Args:
            file_id: ID of the file to download
            **kwargs: Platform-specific options.  Pass ``url`` to specify
                the direct download URL, ``filename`` for a filename hint.

        Returns:
            FileAttachment with content populated

        Raises:
            NotImplementedError: If no ``url`` is provided and the
                subclass does not override this method
        """
        from ..models import FileAttachment

        url = kwargs.get("url")
        if url:
            success, data, error = await self._http_request(
                method="GET",
                url=url,
                return_raw=True,
                operation="download_file",
            )

            if success and isinstance(data, bytes):
                filename = kwargs.get("filename") or url.split("/")[-1] or "file"
                return FileAttachment(
                    id=file_id,
                    filename=filename,
                    content_type=kwargs.get("content_type", "application/octet-stream"),
                    size=len(data),
                    url=url,
                    content=data,
                )

            raise RuntimeError(error or "File download failed")

        raise NotImplementedError(
            f"{self.platform_name} download_file requires a platform-specific override "
            f"or a 'url' keyword argument"
        )

    async def send_voice_message(
        self,
        channel_id: str,
        audio_content: bytes,
        filename: str = "voice_response.mp3",
        content_type: str = "audio/mpeg",
        reply_to: str | None = None,
        **kwargs: Any,
    ) -> "SendMessageResponse":
        """
        Send a voice/audio message to a channel.

        Default implementation uploads as a file attachment.
        Platforms may override for native voice message support.

        Args:
            channel_id: Target channel
            audio_content: Audio file content as bytes
            filename: Audio filename
            content_type: MIME type (audio/mpeg, audio/ogg, etc.)
            reply_to: Optional message ID to reply to
            **kwargs: Platform-specific options

        Returns:
            SendMessageResponse with status
        """
        from ..models import SendMessageResponse

        try:
            # Default implementation: upload as file
            attachment = await self.upload_file(
                channel_id=channel_id,
                content=audio_content,
                filename=filename,
                content_type=content_type,
                thread_id=reply_to,
                **kwargs,
            )
            return SendMessageResponse(
                success=True,
                message_id=attachment.id,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Failed to send voice message on {self.platform_name}: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )


__all__ = ["FileOperationsMixin"]

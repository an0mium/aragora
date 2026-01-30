# mypy: ignore-errors
"""
Microsoft Teams file operations mixin.

Provides upload and download file functionality via Microsoft Graph API
for the TeamsConnector.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aragora.connectors.chat.models import FileAttachment

import aragora.connectors.chat.teams._constants as _tc

try:
    import httpx
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TeamsFilesMixin:
    """Mixin providing file upload/download operations for TeamsConnector."""

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: str | None = None,
        thread_id: str | None = None,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """
        Upload file to Teams channel via Microsoft Graph API.

        Files are stored in the channel's SharePoint document library.
        Requires Files.ReadWrite.All permission.

        Args:
            channel_id: Teams channel ID
            content: File content as bytes
            filename: Name for the uploaded file
            content_type: MIME type of the file
            title: Optional display title (uses filename if not provided)
            thread_id: Optional thread ID (not used for Teams files)
            team_id: Optional team ID (extracted from kwargs if not provided)
            **kwargs: Additional options (may include service_url, team_id)

        Returns:
            FileAttachment with file ID and URL
        """
        if not _tc.HTTPX_AVAILABLE:
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
            )

        # Extract team_id from various sources
        actual_team_id = team_id or kwargs.get("team_id")
        if not actual_team_id:
            # Try to extract from channel_id format (some Teams IDs include team info)
            logger.warning("Team ID not provided for file upload. Attempting extraction.")
            # If channel_id is a full conversation ID, we may not have team_id
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
                metadata={"error": "team_id required for file upload"},
            )

        try:
            # Step 1: Get the channel's files folder
            folder_endpoint = f"/teams/{actual_team_id}/channels/{channel_id}/filesFolder"
            success, folder_data, error = await self._graph_api_request(
                endpoint=folder_endpoint,
                method="GET",
                operation="get_files_folder",
            )

            if not success or not folder_data:
                logger.error(f"Failed to get channel files folder: {error}")
                return FileAttachment(
                    id="",
                    filename=filename,
                    content_type=content_type,
                    size=len(content),
                    content=content,
                    metadata={"error": error or "Failed to get files folder"},
                )

            drive_id = folder_data.get("parentReference", {}).get("driveId")
            folder_id = folder_data.get("id")

            if not drive_id or not folder_id:
                logger.error("Could not extract drive/folder IDs from response")
                return FileAttachment(
                    id="",
                    filename=filename,
                    content_type=content_type,
                    size=len(content),
                    content=content,
                    metadata={"error": "Missing drive/folder IDs"},
                )

            # Step 2: Upload the file
            # For small files (<4MB), use direct upload
            # For large files, use upload session
            file_size = len(content)
            upload_data = None

            if file_size < 4 * 1024 * 1024:  # 4MB threshold
                # Direct upload for small files
                upload_endpoint = f"/drives/{drive_id}/items/{folder_id}:/{filename}:/content"
                success, upload_data, error = await self._graph_api_request(
                    endpoint=upload_endpoint,
                    method="PUT",
                    data=content,
                    content_type=content_type,
                    operation="upload_file",
                )
            else:
                # For large files, create upload session
                session_endpoint = (
                    f"/drives/{drive_id}/items/{folder_id}:/{filename}:/createUploadSession"
                )
                success, session_data, error = await self._graph_api_request(
                    endpoint=session_endpoint,
                    method="POST",
                    json_data={"item": {"@microsoft.graph.conflictBehavior": "rename"}},
                    operation="create_upload_session",
                )

                if not success or not session_data:
                    logger.error(f"Failed to create upload session: {error}")
                    return FileAttachment(
                        id="",
                        filename=filename,
                        content_type=content_type,
                        size=file_size,
                        content=content,
                        metadata={"error": error or "Failed to create upload session"},
                    )

                # Upload content to the session URL using _http_request for retry/circuit breaker
                upload_url = session_data.get("uploadUrl")
                if upload_url:
                    success, upload_data, error = await self._http_request(
                        method="PUT",
                        url=upload_url,
                        headers={
                            "Content-Length": str(file_size),
                            "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                        },
                        content=content,
                        timeout=self._upload_timeout,
                        operation="large_file_upload",
                    )

                    if not success:
                        return FileAttachment(
                            id="",
                            filename=filename,
                            content_type=content_type,
                            size=file_size,
                            content=content,
                            metadata={"error": error or "Upload failed"},
                        )

            if success and upload_data:
                file_id = upload_data.get("id", "")
                web_url = upload_data.get("webUrl")

                logger.info(f"Teams file uploaded: {filename} ({file_size} bytes)")
                return FileAttachment(
                    id=file_id,
                    filename=filename,
                    content_type=content_type,
                    size=file_size,
                    url=web_url,
                    metadata={
                        "drive_id": drive_id,
                        "item_id": file_id,
                        "web_url": web_url,
                    },
                )
            else:
                return FileAttachment(
                    id="",
                    filename=filename,
                    content_type=content_type,
                    size=file_size,
                    content=content,
                    metadata={"error": error or "Upload failed"},
                )

        except httpx.TimeoutException as e:
            classified = _tc._classify_teams_error(f"Timeout: {e}")
            logger.error(f"Teams file upload timeout: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
                metadata={"error": str(e)},
            )
        except httpx.ConnectError as e:
            classified = _tc._classify_teams_error(f"Connection error: {e}")
            logger.error(f"Teams file upload connection error: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
                metadata={"error": str(e)},
            )
        except (
            httpx.HTTPError,
            RuntimeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            classified = _tc._classify_teams_error(str(e))
            logger.error(f"Teams file upload error: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
                metadata={"error": str(e)},
            )

    async def download_file(
        self,
        file_id: str,
        drive_id: str | None = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """
        Download file from Teams via Microsoft Graph API.

        Args:
            file_id: The file item ID (or full drive item path)
            drive_id: Optional drive ID. If not provided, file_id should be
                     a full path like "drives/{drive-id}/items/{item-id}"
            **kwargs: Additional options

        Returns:
            FileAttachment with content populated
        """
        if not _tc.HTTPX_AVAILABLE:
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        try:
            # Get file metadata first
            if drive_id:
                meta_endpoint = f"/drives/{drive_id}/items/{file_id}"
            else:
                # Assume file_id is a full item ID
                meta_endpoint = f"/drives/items/{file_id}"

            success, meta_data, error = await self._graph_api_request(
                endpoint=meta_endpoint,
                method="GET",
                operation="get_file_metadata",
            )

            if not success or not meta_data:
                logger.error(f"Failed to get file metadata: {error}")
                return FileAttachment(
                    id=file_id,
                    filename="",
                    content_type="application/octet-stream",
                    size=0,
                    metadata={"error": error or "Failed to get metadata"},
                )

            filename = meta_data.get("name", "")
            file_size = meta_data.get("size", 0)
            mime_type = meta_data.get("file", {}).get("mimeType", "application/octet-stream")
            download_url = meta_data.get("@microsoft.graph.downloadUrl")

            # Download the content using _http_request for retry/circuit breaker
            if download_url:
                success, content_data, error = await self._http_request(
                    method="GET",
                    url=download_url,
                    timeout=self._upload_timeout,
                    return_raw=True,
                    operation="file_download",
                )

                if success and content_data:
                    file_content = content_data if isinstance(content_data, bytes) else b""
                    logger.info(f"Teams file downloaded: {filename} ({len(file_content)} bytes)")
                    return FileAttachment(
                        id=file_id,
                        filename=filename,
                        content_type=mime_type,
                        size=len(file_content),
                        content=file_content,
                        url=meta_data.get("webUrl"),
                        metadata={
                            "drive_id": drive_id,
                            "item_id": file_id,
                        },
                    )
                else:
                    return FileAttachment(
                        id=file_id,
                        filename=filename,
                        content_type=mime_type,
                        size=file_size,
                        metadata={"error": error or "Download failed"},
                    )
            else:
                logger.error("No download URL in file metadata")
                return FileAttachment(
                    id=file_id,
                    filename=filename,
                    content_type=mime_type,
                    size=file_size,
                    metadata={"error": "No download URL available"},
                )

        except httpx.TimeoutException as e:
            classified = _tc._classify_teams_error(f"Timeout: {e}")
            logger.error(f"Teams file download timeout: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
                metadata={"error": str(e)},
            )
        except httpx.ConnectError as e:
            classified = _tc._classify_teams_error(f"Connection error: {e}")
            logger.error(f"Teams file download connection error: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
                metadata={"error": str(e)},
            )
        except (
            httpx.HTTPError,
            RuntimeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            classified = _tc._classify_teams_error(str(e))
            logger.error(f"Teams file download error: {e}")
            self._record_failure(classified)
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
                metadata={"error": str(e)},
            )

# mypy: ignore-errors
"""
Microsoft Teams Chat Connector.

Implements ChatPlatformConnector for Microsoft Teams using
Bot Framework and Adaptive Cards.

Includes circuit breaker protection for fault tolerance.

Environment Variables:
- TEAMS_APP_ID: Bot application ID
- TEAMS_APP_PASSWORD: Bot application password
- TEAMS_TENANT_ID: Optional tenant ID for single-tenant apps
- TEAMS_REQUEST_TIMEOUT: HTTP request timeout in seconds (default: 30)
- TEAMS_UPLOAD_TIMEOUT: File upload/download timeout in seconds (default: 120)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .thread_manager import ThreadInfo, ThreadStats

__all__ = ["TeamsConnector", "TeamsThreadManager"]

logger = logging.getLogger(__name__)

# Try to import httpx for API calls
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - Teams connector will have limited functionality")

# Distributed tracing support
try:
    from aragora.observability.tracing import build_trace_headers

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def build_trace_headers() -> dict[str, str]:
        return {}


from .base import ChatPlatformConnector
from .models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    SendMessageResponse,
    UserInteraction,
    WebhookEvent,
)

# Environment configuration
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID", "")

# Timeout configuration (in seconds)
TEAMS_REQUEST_TIMEOUT = float(os.environ.get("TEAMS_REQUEST_TIMEOUT", "30"))
TEAMS_UPLOAD_TIMEOUT = float(os.environ.get("TEAMS_UPLOAD_TIMEOUT", "120"))

# Bot Framework API endpoints
BOT_FRAMEWORK_AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
BOT_FRAMEWORK_API_BASE = "https://smba.trafficmanager.net"

# Microsoft Graph API for file operations and channel history
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_AUTH_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

# Graph API scopes for different operations
GRAPH_SCOPE_FILES = "https://graph.microsoft.com/.default"


class TeamsConnector(ChatPlatformConnector):
    """
    Microsoft Teams connector using Bot Framework.

    Supports:
    - Sending messages with Adaptive Cards
    - Responding to commands and interactions
    - File uploads via OneDrive integration
    - Threaded conversations

    Includes circuit breaker protection for fault tolerance against
    Bot Framework API failures and rate limiting.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_password: Optional[str] = None,
        tenant_id: Optional[str] = None,
        request_timeout: Optional[float] = None,
        upload_timeout: Optional[float] = None,
        **config: Any,
    ):
        """
        Initialize Teams connector.

        Args:
            app_id: Bot application ID (defaults to TEAMS_APP_ID env var)
            app_password: Bot application password (defaults to TEAMS_APP_PASSWORD)
            tenant_id: Optional tenant ID for single-tenant apps
            request_timeout: HTTP request timeout in seconds (default from TEAMS_REQUEST_TIMEOUT env var or 30s)
            upload_timeout: File upload/download timeout in seconds (default from TEAMS_UPLOAD_TIMEOUT env var or 120s)
            **config: Additional configuration
        """
        super().__init__(
            bot_token=app_password or TEAMS_APP_PASSWORD,
            signing_secret=None,  # Teams uses JWT validation
            request_timeout=request_timeout or TEAMS_REQUEST_TIMEOUT,
            **config,
        )
        self.app_id = app_id or TEAMS_APP_ID
        self.app_password = app_password or TEAMS_APP_PASSWORD
        self.tenant_id = tenant_id or TEAMS_TENANT_ID
        self._upload_timeout = upload_timeout or TEAMS_UPLOAD_TIMEOUT
        self._access_token: Optional[str] = None
        self._token_expires: float = 0
        # Separate token cache for Microsoft Graph API
        self._graph_token: Optional[str] = None
        self._graph_token_expires: float = 0

    @property
    def platform_name(self) -> str:
        return "teams"

    @property
    def platform_display_name(self) -> str:
        return "Microsoft Teams"

    async def _get_access_token(self) -> str:
        """
        Get or refresh Bot Framework access token.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        import time

        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for Teams API calls")

        # Use _http_request which handles circuit breaker, retries, and backoff
        success, data, error = await self._http_request(
            method="POST",
            url=BOT_FRAMEWORK_AUTH_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "scope": "https://api.botframework.com/.default",
            },
            operation="get_access_token",
        )

        if not success or not data:
            raise RuntimeError(f"Failed to get Bot Framework token: {error}")

        self._access_token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600)

        return self._access_token

    async def _get_graph_token(self) -> str:
        """
        Get or refresh Microsoft Graph API access token.

        Graph API uses a separate OAuth flow from Bot Framework.
        Requires ChannelMessage.Read.All and Files.ReadWrite.All permissions.
        Uses _http_request for retry logic and circuit breaker protection.
        """
        import time

        if self._graph_token and time.time() < self._graph_token_expires - 60:
            return self._graph_token

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for Graph API calls")

        if not self.tenant_id:
            raise RuntimeError("Tenant ID required for Graph API. Set TEAMS_TENANT_ID env var.")

        auth_url = GRAPH_AUTH_URL.format(tenant=self.tenant_id)

        # Use _http_request which handles circuit breaker, retries, and backoff
        success, data, error = await self._http_request(
            method="POST",
            url=auth_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "scope": GRAPH_SCOPE_FILES,
            },
            operation="get_graph_token",
        )

        if not success or not data:
            raise RuntimeError(f"Failed to get Graph API token: {error}")

        self._graph_token = data["access_token"]
        self._graph_token_expires = time.time() + data.get("expires_in", 3600)

        return self._graph_token

    async def _graph_api_request(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[dict] = None,
        data: Optional[bytes] = None,
        content_type: Optional[str] = None,
        operation: str = "graph_api",
    ) -> tuple[bool, Optional[dict], Optional[str]]:
        """
        Make a Microsoft Graph API request with auth and circuit breaker.

        Args:
            endpoint: API endpoint (will be appended to GRAPH_API_BASE)
            method: HTTP method
            json_data: Optional JSON body
            data: Optional raw bytes body (for file uploads)
            content_type: Content-Type header for raw data
            operation: Operation name for logging

        Returns:
            Tuple of (success, response_json, error_message)
        """
        try:
            token = await self._get_graph_token()
        except Exception as e:
            return False, None, f"Failed to get Graph token: {e}"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            **build_trace_headers(),  # Distributed tracing
        }

        if content_type:
            headers["Content-Type"] = content_type

        # Build the full URL
        url = f"{GRAPH_API_BASE}{endpoint}"

        return await self._http_request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            data=data,
            operation=operation,
        )

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        service_url: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send message to Teams channel.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(
                success=False,
                error="httpx not available",
            )

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE
            conv_id = conversation_id or channel_id

            # Build activity payload
            activity = {
                "type": "message",
                "text": text,
            }

            # Add Adaptive Card if blocks provided
            if blocks:
                activity["attachments"] = [  # type: ignore[assignment]
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Handle threaded reply
            if thread_id:
                activity["replyToId"] = thread_id

            # Use shared HTTP helper with retry and circuit breaker
            success, data, error = await self._http_request(
                method="POST",
                url=f"{base_url}/v3/conversations/{conv_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="send_message",
            )

            if success and data:
                return SendMessageResponse(
                    success=True,
                    message_id=data.get("id"),
                    channel_id=conv_id,
                )
            else:
                return SendMessageResponse(
                    success=False,
                    error=error or "Unknown error",
                )

        except Exception as e:
            logger.error(f"Teams send_message error: {e}")
            self._record_failure(e)
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update an existing Teams message.

        Includes circuit breaker protection for fault tolerance.
        """
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE

            activity = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [  # type: ignore[assignment]
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Use shared HTTP helper with retry and circuit breaker
            success, _, error = await self._http_request(
                method="PUT",
                url=f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="update_message",
            )

            if success:
                return SendMessageResponse(
                    success=True,
                    message_id=message_id,
                    channel_id=channel_id,
                )
            else:
                return SendMessageResponse(
                    success=False,
                    error=error or "Unknown error",
                )

        except Exception as e:
            logger.error(f"Teams update_message error: {e}")
            self._record_failure(e)
            return SendMessageResponse(success=False, error=str(e))

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Teams message.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not HTTPX_AVAILABLE:
            return False

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="DELETE",
                url=f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                headers={"Authorization": f"Bearer {token}"},
                operation="delete_message",
            )

            if not success:
                logger.warning(f"Teams delete_message failed: {error}")

            return success

        except Exception as e:
            logger.error(f"Teams delete_message error: {e}")
            return False

    async def send_typing_indicator(
        self,
        channel_id: str,
        service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Send typing indicator to a Teams conversation.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not HTTPX_AVAILABLE:
            return False

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="POST",
                url=f"{base_url}/v3/conversations/{channel_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"type": "typing"},
                operation="send_typing_indicator",
            )

            if not success:
                logger.debug(f"Teams typing indicator failed: {error}")

            return success

        except Exception as e:
            logger.debug(f"Teams typing indicator error: {e}")
            return False

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams command (mention or direct message)."""
        if command.response_url:
            # Use response URL for async response
            return await self._send_to_response_url(
                command.response_url,
                text,
                blocks,
            )

        if command.channel:
            return await self.send_message(
                command.channel.id,
                text,
                blocks,
                service_url=command.metadata.get("service_url"),
                **kwargs,
            )

        return SendMessageResponse(
            success=False,
            error="No channel or response URL available",
        )

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams Adaptive Card action."""
        if interaction.response_url:
            return await self._send_to_response_url(
                interaction.response_url,
                text,
                blocks,
            )

        if interaction.channel and interaction.message_id and replace_original:
            return await self.update_message(
                interaction.channel.id,
                interaction.message_id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        if interaction.channel:
            return await self.send_message(
                interaction.channel.id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        return SendMessageResponse(success=False, error="No response target available")

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: Optional[list[dict]] = None,
    ) -> SendMessageResponse:
        """
        Send response to a Bot Framework response URL.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            token = await self._get_access_token()

            activity: dict[str, Any] = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Use _http_request which handles circuit breaker, retries, and backoff
            success, _, error = await self._http_request(
                method="POST",
                url=response_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="send_to_response_url",
            )

            if success:
                return SendMessageResponse(success=True)
            else:
                return SendMessageResponse(success=False, error=error or "Unknown error")

        except Exception as e:
            logger.error(f"Teams response URL error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        team_id: Optional[str] = None,
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
        if not HTTPX_AVAILABLE:
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

                # Upload content to the session URL with retry logic
                upload_url = session_data.get("uploadUrl")
                if upload_url:
                    import asyncio
                    import random

                    max_retries = 3
                    last_error: Optional[str] = None

                    for attempt in range(max_retries):
                        try:
                            async with httpx.AsyncClient(timeout=self._upload_timeout) as client:
                                response = await client.put(
                                    upload_url,
                                    content=content,
                                    headers={
                                        "Content-Length": str(file_size),
                                        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                                    },
                                )

                                # Check for retryable status codes
                                if response.status_code in (429, 500, 502, 503, 504):
                                    last_error = f"HTTP {response.status_code}"
                                    if attempt < max_retries - 1:
                                        delay = min(1.0 * (2**attempt), 30.0)
                                        jitter = random.uniform(0, delay * 0.1)
                                        logger.warning(
                                            f"Large file upload got {response.status_code} "
                                            f"(attempt {attempt + 1}/{max_retries}). "
                                            f"Retrying in {delay + jitter:.1f}s"
                                        )
                                        await asyncio.sleep(delay + jitter)
                                        continue

                                response.raise_for_status()
                                upload_data = response.json()
                                success = True
                                break

                        except (httpx.TimeoutException, httpx.ConnectError) as e:
                            last_error = str(e)
                            if attempt < max_retries - 1:
                                delay = min(1.0 * (2**attempt), 30.0)
                                logger.warning(
                                    f"Large file upload failed (attempt {attempt + 1}/{max_retries}): {e}. "
                                    f"Retrying in {delay:.1f}s"
                                )
                                await asyncio.sleep(delay)
                            else:
                                logger.error(
                                    f"Large file upload failed after {max_retries} attempts: {e}"
                                )

                        except Exception as e:
                            last_error = str(e)
                            logger.error(f"Large file upload failed: {e}")
                            break

                    if not success:
                        return FileAttachment(
                            id="",
                            filename=filename,
                            content_type=content_type,
                            size=file_size,
                            content=content,
                            metadata={"error": last_error or "Upload failed"},
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

        except Exception as e:
            logger.error(f"Teams file upload error: {e}")
            self._record_failure(e)
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
        drive_id: Optional[str] = None,
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
        if not HTTPX_AVAILABLE:
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

            # Download the content with retry logic
            if download_url:
                import asyncio
                import random

                max_retries = 3
                last_error: Optional[str] = None
                content: Optional[bytes] = None

                for attempt in range(max_retries):
                    try:
                        async with httpx.AsyncClient(timeout=self._upload_timeout) as client:
                            response = await client.get(download_url)

                            # Check for retryable status codes
                            if response.status_code in (429, 500, 502, 503, 504):
                                last_error = f"HTTP {response.status_code}"
                                if attempt < max_retries - 1:
                                    delay = min(1.0 * (2**attempt), 30.0)
                                    jitter = random.uniform(0, delay * 0.1)
                                    logger.warning(
                                        f"File download got {response.status_code} "
                                        f"(attempt {attempt + 1}/{max_retries}). "
                                        f"Retrying in {delay + jitter:.1f}s"
                                    )
                                    await asyncio.sleep(delay + jitter)
                                    continue

                            response.raise_for_status()
                            content = response.content
                            break

                    except (httpx.TimeoutException, httpx.ConnectError) as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            delay = min(1.0 * (2**attempt), 30.0)
                            logger.warning(
                                f"File download failed (attempt {attempt + 1}/{max_retries}): {e}. "
                                f"Retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"File download failed after {max_retries} attempts: {e}")

                    except Exception as e:
                        last_error = str(e)
                        logger.error(f"File content download failed: {e}")
                        break

                if content is not None:
                    logger.info(f"Teams file downloaded: {filename} ({len(content)} bytes)")
                    return FileAttachment(
                        id=file_id,
                        filename=filename,
                        content_type=mime_type,
                        size=len(content),
                        content=content,
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
                        metadata={"error": last_error or "Download failed"},
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

        except Exception as e:
            logger.error(f"Teams file download error: {e}")
            self._record_failure(e)
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
                metadata={"error": str(e)},
            )

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[MessageButton]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Format content as Adaptive Card elements."""
        elements: list[dict] = []

        if title:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": title,
                    "size": "Large",
                    "weight": "Bolder",
                }
            )

        if body:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": body,
                    "wrap": True,
                }
            )

        if fields:
            fact_set = {
                "type": "FactSet",
                "facts": [{"title": label, "value": value} for label, value in fields],
            }
            elements.append(fact_set)

        if actions:
            action_set = {
                "type": "ActionSet",
                "actions": [
                    self.format_button(btn.text, btn.action_id, btn.value, btn.style)
                    for btn in actions
                ],
            }
            elements.append(action_set)

        return elements

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict:
        """Format an Adaptive Card action button."""
        if url:
            return {
                "type": "Action.OpenUrl",
                "title": text,
                "url": url,
            }

        action = {
            "type": "Action.Submit",
            "title": text,
            "data": {
                "action": action_id,
                "value": value or action_id,
            },
        }

        if style == "danger":
            action["style"] = "destructive"
        elif style == "primary":
            action["style"] = "positive"

        return action

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """
        Verify Bot Framework JWT token.

        Uses PyJWT to validate the token against Microsoft's public keys.
        SECURITY: Fails closed in production if PyJWT is not available.
        Uses centralized webhook_security module for production-safe bypass handling.
        """
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header")
            return False

        # Use JWT verification if available
        try:
            from .jwt_verify import verify_teams_webhook, HAS_JWT

            if HAS_JWT:
                return verify_teams_webhook(auth_header, self.app_id)
            else:
                # SECURITY: Use centralized bypass check (ignores flag in production)
                if should_allow_unverified("teams"):
                    logger.warning(
                        "Teams webhook verification skipped - PyJWT not available (dev mode). "
                        "Install PyJWT for secure webhook validation: pip install PyJWT"
                    )
                    return True
                logger.error("Teams webhook rejected - PyJWT not available")
                return False
        except ImportError:
            if should_allow_unverified("teams"):
                logger.warning("Teams JWT verification module not available (dev mode)")
                return True
            logger.error("Teams webhook rejected - JWT verification module not available")
            return False

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse Teams Bot Framework activity into WebhookEvent."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
            )

        activity_type = payload.get("type", "")
        service_url = payload.get("serviceUrl", "")

        # Parse user
        from_data = payload.get("from", {})
        user = ChatUser(
            id=from_data.get("id", ""),
            platform=self.platform_name,
            display_name=from_data.get("name"),
            metadata={"aadObjectId": from_data.get("aadObjectId")},
        )

        # Parse channel
        conversation = payload.get("conversation", {})
        channel = ChatChannel(
            id=conversation.get("id", ""),
            platform=self.platform_name,
            name=conversation.get("name"),
            is_private=conversation.get("isGroup") is False,
            team_id=conversation.get("tenantId"),
            metadata={"conversationType": conversation.get("conversationType")},
        )

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=activity_type,
            raw_payload=payload,
            metadata={"service_url": service_url},
        )

        if activity_type == "message":
            # Regular message
            text = payload.get("text", "")

            # Check for command (bot mention)
            entities = payload.get("entities", [])
            is_mention = any(e.get("type") == "mention" for e in entities)

            if is_mention and text.strip().startswith("<at>"):
                # Extract command after mention
                import re

                clean_text = re.sub(r"<at>.*?</at>\s*", "", text).strip()
                parts = clean_text.split(maxsplit=1)

                event.command = BotCommand(
                    name=parts[0] if parts else "",
                    text=clean_text,
                    args=parts[1].split() if len(parts) > 1 else [],
                    user=user,
                    channel=channel,
                    platform=self.platform_name,
                    metadata={"service_url": service_url},
                )
            else:
                event.message = ChatMessage(
                    id=payload.get("id", ""),
                    platform=self.platform_name,
                    channel=channel,
                    author=user,
                    content=text,
                    thread_id=payload.get("replyToId"),
                    metadata={"service_url": service_url},
                )

        elif activity_type == "invoke":
            # Adaptive Card action
            action_data = payload.get("value", {})

            event.interaction = UserInteraction(
                id=payload.get("id", ""),
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id=action_data.get("action", ""),
                value=action_data.get("value"),
                user=user,
                channel=channel,
                message_id=payload.get("replyToId"),
                platform=self.platform_name,
                metadata={"service_url": service_url},
            )

        return event

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        team_id: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get message history from a Teams channel via Microsoft Graph API.

        Uses the channelMessages API to retrieve messages.
        Requires ChannelMessage.Read.All permission.

        Args:
            channel_id: Teams channel ID
            limit: Maximum number of messages (max 50 per request)
            oldest: ISO timestamp - messages after this time
            latest: ISO timestamp - messages before this time
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            List of ChatMessage objects
        """
        from datetime import datetime

        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for Graph API")
            return []

        actual_team_id = team_id or kwargs.get("team_id")
        if not actual_team_id:
            logger.error("Team ID required for get_channel_history")
            return []

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.warning(f"Circuit breaker open: {cb_error}")
            return []

        try:
            messages: list[ChatMessage] = []
            next_link: Optional[str] = None

            # Build initial endpoint with filters
            endpoint = f"/teams/{actual_team_id}/channels/{channel_id}/messages"
            params = [f"$top={min(limit, 50)}"]  # Graph API max is 50 per page

            if oldest:
                params.append(f"$filter=createdDateTime gt {oldest}")

            if params:
                endpoint = f"{endpoint}?{'&'.join(params)}"

            while True:
                if next_link:
                    # Use the full nextLink URL directly
                    success, data, error = await self._graph_api_request(
                        endpoint=next_link.replace(GRAPH_API_BASE, ""),
                        method="GET",
                        operation="get_channel_messages",
                    )
                else:
                    success, data, error = await self._graph_api_request(
                        endpoint=endpoint,
                        method="GET",
                        operation="get_channel_messages",
                    )

                if not success or not data:
                    logger.error(f"Failed to get channel messages: {error}")
                    break

                # Parse messages from response
                for msg_data in data.get("value", []):
                    msg_id = msg_data.get("id", "")
                    created_at_str = msg_data.get("createdDateTime", "")

                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.utcnow()

                    # Filter by latest timestamp if provided
                    if latest:
                        try:
                            latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                            if timestamp > latest_dt:
                                continue
                        except (ValueError, AttributeError):
                            pass

                    # Parse author
                    from_data = msg_data.get("from", {}) or {}
                    user_data = from_data.get("user", {}) or {}
                    author = ChatUser(
                        id=user_data.get("id", ""),
                        platform=self.platform_name,
                        display_name=user_data.get("displayName"),
                        metadata={"aadObjectId": user_data.get("aadObjectId")},
                    )

                    # Parse channel info
                    channel = ChatChannel(
                        id=channel_id,
                        platform=self.platform_name,
                        team_id=actual_team_id,
                    )

                    # Extract message content
                    body = msg_data.get("body", {}) or {}
                    content = body.get("content", "")

                    # Strip HTML if content type is html
                    if body.get("contentType") == "html":
                        import re

                        content = re.sub(r"<[^>]+>", "", content)

                    messages.append(
                        ChatMessage(
                            id=msg_id,
                            platform=self.platform_name,
                            channel=channel,
                            author=author,
                            content=content,
                            timestamp=timestamp,
                            thread_id=msg_data.get("replyToId"),
                            metadata={
                                "importance": msg_data.get("importance"),
                                "web_url": msg_data.get("webUrl"),
                            },
                        )
                    )

                    if len(messages) >= limit:
                        break

                # Check for more pages
                next_link = data.get("@odata.nextLink")
                if not next_link or len(messages) >= limit:
                    break

            logger.debug(f"Retrieved {len(messages)} messages from Teams channel {channel_id}")
            return messages[:limit]

        except Exception as e:
            logger.error(f"Teams get_channel_history error: {e}")
            self._record_failure(e)
            return []

    async def collect_evidence(
        self,
        channel_id: str,
        query: Optional[str] = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        team_id: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a Teams channel, filters by relevance,
        and converts to ChatEvidence format with provenance tracking.

        Args:
            channel_id: Teams channel ID
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include reply messages
            min_relevance: Minimum relevance score for inclusion (0-1)
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            List of ChatEvidence objects with relevance scoring
        """
        # Get channel history
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=limit,
            team_id=team_id or kwargs.get("team_id"),
            **kwargs,
        )

        if not messages:
            return []

        # Convert to evidence with relevance scoring
        evidence_list: list[ChatEvidence] = []

        for msg in messages:
            # Skip replies if not including threads
            if not include_threads and msg.thread_id:
                continue

            # Calculate relevance using base class helper
            relevance = self._compute_message_relevance(msg, query)

            # Apply minimum relevance filter
            if relevance < min_relevance:
                continue

            # Convert to ChatEvidence
            evidence = ChatEvidence.from_message(
                message=msg,
                query=query or "",
                relevance_score=relevance,
            )

            evidence_list.append(evidence)

        # Sort by relevance score (highest first)
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        logger.debug(
            f"Collected {len(evidence_list)} evidence items from Teams channel {channel_id}"
        )
        return evidence_list

    async def get_channel_info(
        self,
        channel_id: str,
        team_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[ChatChannel]:
        """
        Get information about a Teams channel via Microsoft Graph API.

        Args:
            channel_id: Channel ID
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            ChatChannel info or None
        """
        actual_team_id = team_id or kwargs.get("team_id")
        if not actual_team_id:
            logger.debug("Team ID required for get_channel_info")
            return None

        try:
            endpoint = f"/teams/{actual_team_id}/channels/{channel_id}"
            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_channel_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get channel info: {error}")
                return None

            return ChatChannel(
                id=channel_id,
                platform=self.platform_name,
                name=data.get("displayName"),
                is_private=data.get("membershipType") == "private",
                team_id=actual_team_id,
                metadata={
                    "description": data.get("description"),
                    "web_url": data.get("webUrl"),
                    "membership_type": data.get("membershipType"),
                },
            )

        except Exception as e:
            logger.debug(f"Teams get_channel_info error: {e}")
            return None

    async def list_channels(
        self,
        team_id: str,
        include_private: bool = False,
        **kwargs: Any,
    ) -> list[ChatChannel]:
        """
        List all channels in a Microsoft Teams team.

        Uses Microsoft Graph API to enumerate channels.

        Args:
            team_id: Team ID to list channels for
            include_private: Whether to include private channels (default: False)
            **kwargs: Additional options

        Returns:
            List of ChatChannel objects
        """
        channels: list[ChatChannel] = []

        try:
            endpoint = f"/teams/{team_id}/channels"
            if not include_private:
                endpoint += "?$filter=membershipType eq 'standard'"

            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="list_channels",
            )

            if not success or not data:
                logger.warning(f"Failed to list channels for team {team_id}: {error}")
                return channels

            channel_list = data.get("value", [])
            for channel_data in channel_list:
                channel = ChatChannel(
                    id=channel_data.get("id", ""),
                    platform=self.platform_name,
                    name=channel_data.get("displayName"),
                    is_private=channel_data.get("membershipType") == "private",
                    team_id=team_id,
                    metadata={
                        "description": channel_data.get("description"),
                        "web_url": channel_data.get("webUrl"),
                        "membership_type": channel_data.get("membershipType"),
                    },
                )
                channels.append(channel)

            logger.debug(f"Listed {len(channels)} channels for team {team_id}")
            return channels

        except Exception as e:
            logger.error(f"Teams list_channels error: {e}")
            return channels

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> Optional[ChatUser]:
        """
        Get information about a user via Microsoft Graph API.

        Args:
            user_id: User ID (AAD Object ID)
            **kwargs: Additional options

        Returns:
            ChatUser info or None
        """
        try:
            endpoint = f"/users/{user_id}"
            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_user_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get user info: {error}")
                return None

            return ChatUser(
                id=user_id,
                platform=self.platform_name,
                username=data.get("userPrincipalName"),
                display_name=data.get("displayName"),
                email=data.get("mail"),
                metadata={
                    "job_title": data.get("jobTitle"),
                    "office_location": data.get("officeLocation"),
                    "department": data.get("department"),
                },
            )

        except Exception as e:
            logger.debug(f"Teams get_user_info error: {e}")
            return None


class TeamsThreadManager:
    """
    Microsoft Teams thread management using Graph API.

    Teams threads are message-based - replies reference a parent message ID.
    Requires team_id for all Graph API operations on channels.
    """

    def __init__(self, connector: TeamsConnector, team_id: str):
        """
        Initialize Teams thread manager.

        Args:
            connector: TeamsConnector instance for API calls
            team_id: Team ID (required for Graph API channel operations)
        """
        self.connector = connector
        self.team_id = team_id

    @property
    def platform_name(self) -> str:
        """Return platform identifier."""
        return "teams"

    async def get_thread(
        self,
        thread_id: str,
        channel_id: str,
    ) -> "ThreadInfo":
        """
        Get thread metadata for a Teams message thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            ThreadInfo with thread metadata

        Raises:
            ThreadNotFoundError: If thread doesn't exist
        """
        from .thread_manager import ThreadInfo, ThreadNotFoundError

        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}"
        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="GET",
            operation="get_thread",
        )

        if not success or not data:
            raise ThreadNotFoundError(
                thread_id=thread_id,
                channel_id=channel_id,
                platform="teams",
            )

        # Get reply count
        replies_endpoint = f"{endpoint}/replies"
        _, replies_data, _ = await self.connector._graph_api_request(
            endpoint=replies_endpoint,
            method="GET",
            operation="get_thread_replies",
        )

        reply_count = len(replies_data.get("value", [])) if replies_data else 0
        participants = set()

        if replies_data and replies_data.get("value"):
            for reply in replies_data["value"]:
                if reply.get("from", {}).get("user", {}).get("id"):
                    participants.add(reply["from"]["user"]["id"])

        # Add original author
        if data.get("from", {}).get("user", {}).get("id"):
            participants.add(data["from"]["user"]["id"])

        created_at = data.get("createdDateTime", "")
        last_modified = data.get("lastModifiedDateTime", created_at)

        return ThreadInfo(
            id=thread_id,
            channel_id=channel_id,
            platform="teams",
            created_by=data.get("from", {}).get("user", {}).get("id", "unknown"),
            created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_at
            else datetime.now(),
            updated_at=datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            if last_modified
            else datetime.now(),
            message_count=reply_count + 1,  # Include root message
            participant_count=len(participants),
            title=data.get("subject"),
            metadata={
                "team_id": self.team_id,
                "importance": data.get("importance"),
                "message_type": data.get("messageType"),
            },
        )

    async def get_thread_messages(
        self,
        thread_id: str,
        channel_id: str,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> tuple[list[ChatMessage], Optional[str]]:
        """
        Get messages in a thread with pagination.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID
            limit: Maximum messages to retrieve
            cursor: Pagination cursor (nextLink URL)

        Returns:
            Tuple of (messages list, next cursor or None)
        """
        if cursor:
            # Use provided nextLink for pagination
            success, data, error = await self.connector._graph_api_request(
                endpoint=cursor,
                method="GET",
                operation="get_thread_messages_page",
                use_full_url=True,
            )
        else:
            endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}/replies"
            params = {"$top": str(limit)}
            success, data, error = await self.connector._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_thread_messages",
                params=params,
            )

        if not success or not data:
            return [], None

        messages = []
        for msg in data.get("value", []):
            user_data = msg.get("from", {}).get("user", {})
            messages.append(
                ChatMessage(
                    id=msg.get("id", ""),
                    platform="teams",
                    channel=ChatChannel(
                        id=channel_id,
                        platform="teams",
                        name=channel_id,
                    ),
                    author=ChatUser(
                        id=user_data.get("id", "unknown"),
                        platform="teams",
                        display_name=user_data.get("displayName"),
                    ),
                    content=msg.get("body", {}).get("content", ""),
                    thread_id=thread_id,
                    timestamp=datetime.fromisoformat(
                        msg.get("createdDateTime", "").replace("Z", "+00:00")
                    )
                    if msg.get("createdDateTime")
                    else datetime.now(),
                    metadata={
                        "importance": msg.get("importance"),
                        "content_type": msg.get("body", {}).get("contentType"),
                    },
                )
            )

        next_cursor = data.get("@odata.nextLink")
        return messages, next_cursor

    async def list_threads(
        self,
        channel_id: str,
        limit: int = 20,
    ) -> list["ThreadInfo"]:
        """
        List recent threads (root messages) in a channel.

        Args:
            channel_id: Channel ID
            limit: Maximum threads to retrieve

        Returns:
            List of ThreadInfo objects for threads with replies
        """
        from .thread_manager import ThreadInfo

        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages"
        params = {"$top": str(limit)}

        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="GET",
            operation="list_threads",
            params=params,
        )

        if not success or not data:
            return []

        threads = []
        for msg in data.get("value", []):
            # Only include messages that have replies (are threads)
            # Teams doesn't directly expose reply count in list, so we include all root messages
            created_at = msg.get("createdDateTime", "")
            last_modified = msg.get("lastModifiedDateTime", created_at)

            threads.append(
                ThreadInfo(
                    id=msg.get("id", ""),
                    channel_id=channel_id,
                    platform="teams",
                    created_by=msg.get("from", {}).get("user", {}).get("id", "unknown"),
                    created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_at
                    else datetime.now(),
                    updated_at=datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                    if last_modified
                    else datetime.now(),
                    message_count=1,  # Would need separate call for accurate count
                    participant_count=1,
                    title=msg.get("subject"),
                    metadata={
                        "team_id": self.team_id,
                        "importance": msg.get("importance"),
                    },
                )
            )

        return threads

    async def reply_to_thread(
        self,
        thread_id: str,
        channel_id: str,
        message: str,
    ) -> ChatMessage:
        """
        Reply to an existing thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID
            message: Reply message content

        Returns:
            ChatMessage representing the sent reply
        """
        endpoint = f"/teams/{self.team_id}/channels/{channel_id}/messages/{thread_id}/replies"

        success, data, error = await self.connector._graph_api_request(
            endpoint=endpoint,
            method="POST",
            operation="reply_to_thread",
            json_data={
                "body": {
                    "contentType": "text",
                    "content": message,
                }
            },
        )

        if not success or not data:
            raise RuntimeError(f"Failed to reply to thread: {error}")

        user_data = data.get("from", {}).get("user", {})
        return ChatMessage(
            id=data.get("id", ""),
            platform="teams",
            channel=ChatChannel(
                id=channel_id,
                platform="teams",
                name=channel_id,
            ),
            author=ChatUser(
                id=user_data.get("id", "bot"),
                platform="teams",
                display_name=user_data.get("displayName", "Bot"),
            ),
            content=message,
            thread_id=thread_id,
            timestamp=datetime.now(),
        )

    async def get_thread_stats(
        self,
        thread_id: str,
        channel_id: str,
    ) -> "ThreadStats":
        """
        Get statistics for a thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            ThreadStats with thread metrics
        """
        from .thread_manager import ThreadStats

        # Get thread info which includes counts
        thread_info = await self.get_thread(thread_id, channel_id)

        return ThreadStats(
            thread_id=thread_id,
            message_count=thread_info.message_count,
            participant_count=thread_info.participant_count,
            last_activity=thread_info.updated_at,
        )

    async def get_thread_participants(
        self,
        thread_id: str,
        channel_id: str,
    ) -> list[str]:
        """
        Get list of user IDs who participated in a thread.

        Args:
            thread_id: Message ID of the thread root
            channel_id: Channel ID

        Returns:
            List of user IDs
        """
        messages, _ = await self.get_thread_messages(thread_id, channel_id, limit=100)
        participants = set()

        for msg in messages:
            if msg.user_id:
                participants.add(msg.user_id)

        return list(participants)

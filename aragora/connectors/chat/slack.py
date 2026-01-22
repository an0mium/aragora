"""
Slack Chat Connector.

Implements ChatPlatformConnector for Slack using
Slack's Web API and Block Kit.

Environment Variables:
- SLACK_BOT_TOKEN: Bot OAuth token (xoxb-...)
- SLACK_SIGNING_SECRET: For webhook verification
- SLACK_WEBHOOK_URL: For incoming webhooks

Resilience Features:
- Circuit breaker protection against Slack API failures
- Exponential backoff retry logic for transient errors
- Configurable timeouts on all API calls
- Rate limit handling (429 responses)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from aragora.resilience import get_circuit_breaker

try:
    from aragora.observability.tracing import build_trace_headers
except ImportError:

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
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Slack API
SLACK_API_BASE = "https://slack.com/api"

# Resilience configuration
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN = 60.0  # seconds


def _is_retryable_error(status_code: int, error: Optional[str] = None) -> bool:
    """Check if an error is retryable (transient)."""
    # Rate limited
    if status_code == 429:
        return True
    # Server errors
    if 500 <= status_code < 600:
        return True
    # Slack-specific retryable errors
    retryable_errors = {"service_unavailable", "timeout", "internal_error", "fatal_error"}
    if error and error.lower() in retryable_errors:
        return True
    return False


async def _exponential_backoff(attempt: int, base: float = 1.0, max_delay: float = 30.0) -> None:
    """Sleep with exponential backoff and jitter."""
    delay = min(base * (2**attempt) + random.uniform(0, 1), max_delay)
    await asyncio.sleep(delay)


class SlackConnector(ChatPlatformConnector):
    """
    Slack connector using Slack Web API.

    Supports:
    - Sending messages with Block Kit
    - Slash commands
    - Interactive components (buttons, menus)
    - File uploads
    - Threaded conversations
    - Ephemeral messages

    Resilience Features:
    - Circuit breaker protection against API failures
    - Exponential backoff retry for transient errors
    - Configurable timeouts on all API calls
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        use_circuit_breaker: bool = True,
        **config: Any,
    ):
        """
        Initialize Slack connector.

        Args:
            bot_token: Bot OAuth token (defaults to SLACK_BOT_TOKEN)
            signing_secret: Webhook signing secret
            webhook_url: Incoming webhook URL
            timeout: Request timeout in seconds (default 30)
            max_retries: Maximum retry attempts for transient errors (default 3)
            use_circuit_breaker: Whether to use circuit breaker (default True)
            **config: Additional configuration
        """
        super().__init__(
            bot_token=bot_token or SLACK_BOT_TOKEN,
            signing_secret=signing_secret or SLACK_SIGNING_SECRET,
            webhook_url=webhook_url or SLACK_WEBHOOK_URL,
            **config,
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._use_circuit_breaker = use_circuit_breaker

        # Initialize circuit breaker
        if use_circuit_breaker:
            self._circuit_breaker = get_circuit_breaker(
                "slack_api",
                failure_threshold=CIRCUIT_BREAKER_THRESHOLD,
                cooldown_seconds=CIRCUIT_BREAKER_COOLDOWN,
            )
        else:
            self._circuit_breaker = None

    @property
    def platform_name(self) -> str:
        return "slack"

    @property
    def platform_display_name(self) -> str:
        return "Slack"

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers with trace context for distributed tracing."""
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        # Add trace context headers for distributed tracing
        headers.update(build_trace_headers())
        return headers

    async def _slack_api_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        operation: str = "api_call",
    ) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        """
        Make a Slack API request with circuit breaker, retry, and timeout.

        Centralizes the resilience pattern for all Slack API calls.

        Args:
            endpoint: API endpoint (e.g., "chat.postMessage")
            payload: JSON payload to send
            operation: Operation name for logging

        Returns:
            Tuple of (success, response_data, error_message)
        """
        if not HTTPX_AVAILABLE:
            return False, None, "httpx not available"

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            remaining = self._circuit_breaker.cooldown_remaining()
            return False, None, f"Circuit breaker open (retry in {remaining:.0f}s)"

        last_error: Optional[str] = None
        url = f"{SLACK_API_BASE}/{endpoint}"

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        url,
                        headers=self._get_headers(),
                        json=payload,
                    )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return True, data, None
                    else:
                        error = data.get("error", "Unknown error")
                        last_error = error

                        # Check if retryable
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                logger.warning(
                                    f"Slack {operation} retryable error: {error} "
                                    f"(attempt {attempt + 1}/{self._max_retries})"
                                )
                                await _exponential_backoff(attempt)
                                continue

                        # Non-retryable error
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False, data, error

            except httpx.TimeoutException:
                last_error = f"Request timeout after {self._timeout}s"
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Slack {operation} timeout (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await _exponential_backoff(attempt)
                    continue

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Slack {operation} connection error (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"Slack {operation} error: {e}")
                # Don't retry on unexpected errors
                break

        # All retries exhausted
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False, None, last_error or "Unknown error"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send message to Slack channel with retry and circuit breaker."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return SendMessageResponse(
                success=False,
                error=f"Circuit breaker open - Slack API unavailable (retry in {self._circuit_breaker.cooldown_remaining():.0f}s)",
            )

        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": text,
        }

        if blocks:
            payload["blocks"] = blocks

        if thread_id:
            payload["thread_ts"] = thread_id

        # Optional: unfurl links/media
        if "unfurl_links" in kwargs:
            payload["unfurl_links"] = kwargs["unfurl_links"]
        if "unfurl_media" in kwargs:
            payload["unfurl_media"] = kwargs["unfurl_media"]

        last_error: Optional[str] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/chat.postMessage",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return SendMessageResponse(
                            success=True,
                            message_id=data.get("ts"),
                            channel_id=data.get("channel"),
                            timestamp=data.get("ts"),
                        )
                    else:
                        error = data.get("error", "Unknown error")
                        last_error = error

                        # Check if retryable
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                logger.warning(
                                    f"Slack send_message retryable error: {error} (attempt {attempt + 1}/{self._max_retries})"
                                )
                                await _exponential_backoff(attempt)
                                continue

                        # Non-retryable error
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return SendMessageResponse(success=False, error=error)

            except httpx.TimeoutException:
                last_error = "Request timeout"
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Slack send_message timeout (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"Slack send_message error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        # All retries exhausted
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return SendMessageResponse(success=False, error=last_error or "Unknown error")

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Update a Slack message with retry and circuit breaker."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return SendMessageResponse(
                success=False,
                error="Circuit breaker open - Slack API unavailable",
            )

        payload: dict[str, Any] = {
            "channel": channel_id,
            "ts": message_id,
            "text": text,
        }

        if blocks:
            payload["blocks"] = blocks

        last_error: Optional[str] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/chat.update",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return SendMessageResponse(
                            success=True,
                            message_id=data.get("ts"),
                            channel_id=data.get("channel"),
                        )
                    else:
                        error = data.get("error", "Unknown error")
                        last_error = error

                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                logger.warning(f"Slack update_message retryable error: {error}")
                                await _exponential_backoff(attempt)
                                continue

                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return SendMessageResponse(success=False, error=error)

            except httpx.TimeoutException:
                last_error = "Request timeout"
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"Slack update_message error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return SendMessageResponse(success=False, error=last_error or "Unknown error")

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a Slack message with retry and circuit breaker."""
        if not HTTPX_AVAILABLE:
            return False

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open - cannot delete message")
            return False

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/chat.delete",
                        headers=self._get_headers(),
                        json={
                            "channel": channel_id,
                            "ts": message_id,
                        },
                    )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return True

                    error = data.get("error", "")
                    if _is_retryable_error(response.status_code, error):
                        if attempt < self._max_retries - 1:
                            await _exponential_backoff(attempt)
                            continue

                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    return False

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack delete_message error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send ephemeral message visible only to one user with retry."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return SendMessageResponse(success=False, error="Circuit breaker open")

        payload: dict[str, Any] = {
            "channel": channel_id,
            "user": user_id,
            "text": text,
        }

        if blocks:
            payload["blocks"] = blocks

        last_error: Optional[str] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/chat.postEphemeral",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return SendMessageResponse(success=True)

                    error = data.get("error", "Unknown error")
                    last_error = error

                    if _is_retryable_error(response.status_code, error):
                        if attempt < self._max_retries - 1:
                            await _exponential_backoff(attempt)
                            continue

                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    return SendMessageResponse(success=False, error=error)

            except httpx.TimeoutException:
                last_error = "Request timeout"
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                last_error = str(e)
                logger.error(f"Slack send_ephemeral error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return SendMessageResponse(success=False, error=last_error)

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Slack slash command."""
        # Use response_url for async response
        if command.response_url:
            return await self._send_to_response_url(
                command.response_url,
                text,
                blocks,
                response_type="ephemeral" if ephemeral else "in_channel",
            )

        # Fallback to regular message
        if command.channel and command.user:
            if ephemeral:
                return await self.send_ephemeral(
                    command.channel.id,
                    command.user.id,
                    text,
                    blocks,
                )
            else:
                return await self.send_message(
                    command.channel.id,
                    text,
                    blocks,
                )

        return SendMessageResponse(success=False, error="No response target")

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Slack interaction."""
        if interaction.response_url:
            return await self._send_to_response_url(
                interaction.response_url,
                text,
                blocks,
                replace_original=replace_original,
            )

        if interaction.channel and interaction.message_id and replace_original:
            return await self.update_message(
                interaction.channel.id,
                interaction.message_id,
                text,
                blocks,
            )

        if interaction.channel:
            return await self.send_message(
                interaction.channel.id,
                text,
                blocks,
            )

        return SendMessageResponse(success=False, error="No response target")

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        response_type: str = "ephemeral",
        replace_original: bool = False,
    ) -> SendMessageResponse:
        """Send response to Slack response_url with timeout and retries."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        payload: dict[str, Any] = {
            "text": text,
            "response_type": response_type,
        }

        if blocks:
            payload["blocks"] = blocks

        if replace_original:
            payload["replace_original"] = True

        # Response URLs have shorter validity, use fewer retries
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(response_url, json=payload)
                    if response.status_code == 200:
                        return SendMessageResponse(success=True)

                    # Retry on 5xx errors
                    if 500 <= response.status_code < 600 and attempt == 0:
                        await _exponential_backoff(0, base=0.5)
                        continue

                    return SendMessageResponse(
                        success=False,
                        error=f"HTTP {response.status_code}",
                    )

            except httpx.TimeoutException:
                if attempt == 0:
                    await _exponential_backoff(0, base=0.5)
                    continue
                return SendMessageResponse(success=False, error="Request timeout")

            except Exception as e:
                logger.error(f"Slack response_url error: {e}")
                return SendMessageResponse(success=False, error=str(e))

        return SendMessageResponse(success=False, error="Request failed")

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload file to Slack with timeout and retry."""
        if not HTTPX_AVAILABLE:
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

        files = {"file": (filename, content, content_type)}
        data: dict[str, Any] = {
            "channels": channel_id,
            "filename": filename,
        }

        if title:
            data["title"] = title

        if thread_id:
            data["thread_ts"] = thread_id

        for attempt in range(self._max_retries):
            try:
                # Use longer timeout for file uploads
                async with httpx.AsyncClient(timeout=self._timeout * 2) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/files.upload",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        data=data,
                        files=files,
                    )
                    result = response.json()

                    if result.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        file_data = result.get("file", {})
                        return FileAttachment(
                            id=file_data.get("id", ""),
                            filename=file_data.get("name", filename),
                            content_type=file_data.get("mimetype", content_type),
                            size=file_data.get("size", len(content)),
                            url=file_data.get("url_private"),
                        )

                    error = result.get("error", "")
                    if _is_retryable_error(response.status_code, error):
                        if attempt < self._max_retries - 1:
                            await _exponential_backoff(attempt)
                            continue

                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    return FileAttachment(
                        id="",
                        filename=filename,
                        content_type=content_type,
                        size=len(content),
                    )

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack upload_file error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return FileAttachment(
            id="",
            filename=filename,
            content_type=content_type,
            size=len(content),
        )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download file from Slack with timeout and retry."""
        if not HTTPX_AVAILABLE:
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        for attempt in range(self._max_retries):
            try:
                # Use longer timeout for file downloads
                async with httpx.AsyncClient(timeout=self._timeout * 2) as client:
                    # Get file info first
                    info_response = await client.get(
                        f"{SLACK_API_BASE}/files.info",
                        headers=self._get_headers(),
                        params={"file": file_id},
                    )
                    info = info_response.json()

                    if not info.get("ok"):
                        error = info.get("error", "")
                        if _is_retryable_error(info_response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return FileAttachment(
                            id=file_id,
                            filename="",
                            content_type="application/octet-stream",
                            size=0,
                        )

                    file_data = info.get("file", {})
                    url = file_data.get("url_private_download") or file_data.get("url_private")

                    if not url:
                        return FileAttachment(
                            id=file_id,
                            filename=file_data.get("name", ""),
                            content_type=file_data.get("mimetype", "application/octet-stream"),
                            size=file_data.get("size", 0),
                        )

                    # Download the file
                    download_response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                    )

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    return FileAttachment(
                        id=file_id,
                        filename=file_data.get("name", ""),
                        content_type=file_data.get("mimetype", "application/octet-stream"),
                        size=len(download_response.content),
                        url=url,
                        content=download_response.content,
                    )

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack download_file error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return FileAttachment(
            id=file_id,
            filename="",
            content_type="application/octet-stream",
            size=0,
        )

    # ==========================================================================
    # Channel and User Info (implements abstract methods)
    # ==========================================================================

    async def get_channel_info(self, channel_id: str, **kwargs: Any) -> Optional[ChatChannel]:
        """Get channel information from Slack."""
        if not HTTPX_AVAILABLE:
            return None

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return None

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{SLACK_API_BASE}/conversations.info",
                    headers=self._get_headers(),
                    params={"channel": channel_id},
                )
                data = response.json()

                if data.get("ok"):
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    channel_data = data.get("channel", {})
                    return ChatChannel(
                        id=channel_id,
                        platform=self.platform_name,
                        name=channel_data.get("name"),
                        team_id=channel_data.get("context_team_id"),
                        is_private=channel_data.get("is_private", False),
                        metadata={
                            "topic": channel_data.get("topic", {}).get("value", ""),
                            "purpose": channel_data.get("purpose", {}).get("value", ""),
                            "num_members": channel_data.get("num_members", 0),
                        },
                    )

                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                return None

        except Exception as e:
            logger.error(f"Slack get_channel_info error: {e}")
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            return None

    async def get_user_info(self, user_id: str, **kwargs: Any) -> Optional[ChatUser]:
        """Get user information from Slack."""
        if not HTTPX_AVAILABLE:
            return None

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return None

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    f"{SLACK_API_BASE}/users.info",
                    headers=self._get_headers(),
                    params={"user": user_id},
                )
                data = response.json()

                if data.get("ok"):
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    user_data = data.get("user", {})
                    profile = user_data.get("profile", {})
                    return ChatUser(
                        id=user_id,
                        platform=self.platform_name,
                        username=user_data.get("name"),
                        display_name=profile.get("display_name") or profile.get("real_name"),
                        email=profile.get("email"),
                        is_bot=user_data.get("is_bot", False),
                        metadata={
                            "title": profile.get("title", ""),
                            "team_id": user_data.get("team_id"),
                            "tz": user_data.get("tz"),
                        },
                    )

                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                return None

        except Exception as e:
            logger.error(f"Slack get_user_info error: {e}")
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            return None

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[MessageButton]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Format content as Slack Block Kit blocks."""
        blocks: list[dict] = []

        if title:
            blocks.append(
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": title,
                        "emoji": True,
                    },
                }
            )

        if body:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": body,
                    },
                }
            )

        if fields:
            field_elements = []
            for label, value in fields:
                field_elements.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{label}*\n{value}",
                    }
                )
            blocks.append(
                {
                    "type": "section",
                    "fields": field_elements,
                }
            )

        if actions:
            action_elements = [
                self.format_button(btn.text, btn.action_id, btn.value, btn.style, btn.url)
                for btn in actions
            ]
            blocks.append(
                {
                    "type": "actions",
                    "elements": action_elements,
                }
            )

        return blocks

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict:
        """Format a Slack button element."""
        if url:
            return {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": text,
                    "emoji": True,
                },
                "url": url,
            }

        button: dict[str, Any] = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": True,
            },
            "action_id": action_id,
            "value": value or action_id,
        }

        if style == "primary":
            button["style"] = "primary"
        elif style == "danger":
            button["style"] = "danger"

        return button

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Slack webhook signature.

        SECURITY: Fails closed in production if signing_secret is not configured.
        Uses centralized webhook_security module for production-safe bypass handling.
        """
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        if not self.signing_secret:
            # SECURITY: Use centralized bypass check (ignores flag in production)
            if should_allow_unverified("slack"):
                logger.warning("Slack webhook verification skipped (dev mode)")
                return True
            logger.error("Slack webhook rejected - signing_secret not configured")
            return False

        timestamp = headers.get("X-Slack-Request-Timestamp", "")
        signature = headers.get("X-Slack-Signature", "")

        if not timestamp or not signature:
            return False

        # Check timestamp to prevent replay attacks
        try:
            request_time = int(timestamp)
            if abs(time.time() - request_time) > 300:
                return False
        except ValueError:
            return False

        # Compute expected signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = (
            "v0="
            + hmac.new(
                self.signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(expected, signature)

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse Slack webhook payload into WebhookEvent."""
        content_type = headers.get("Content-Type", "")

        # Handle URL-encoded form data (slash commands, interactions)
        if "application/x-www-form-urlencoded" in content_type:
            from urllib.parse import parse_qs

            parsed = parse_qs(body.decode("utf-8"))

            # Check for payload field (interactions)
            if "payload" in parsed:
                payload = json.loads(parsed["payload"][0])
                return self._parse_interaction_payload(payload)

            # Slash command
            return self._parse_slash_command(parsed)

        # Handle JSON (events API)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
            )

        # URL verification challenge
        if payload.get("type") == "url_verification":
            return WebhookEvent(
                platform=self.platform_name,
                event_type="url_verification",
                raw_payload=payload,
                challenge=payload.get("challenge"),
            )

        # Event callback
        if payload.get("type") == "event_callback":
            return self._parse_event_callback(payload)

        return WebhookEvent(
            platform=self.platform_name,
            event_type=payload.get("type", "unknown"),
            raw_payload=payload,
        )

    def _parse_slash_command(self, parsed: dict) -> WebhookEvent:
        """Parse slash command from form data."""

        def get_first(key: str, default: str = "") -> str:
            values = parsed.get(key, [default])
            return values[0] if values else default

        user = ChatUser(
            id=get_first("user_id"),
            platform=self.platform_name,
            username=get_first("user_name"),
        )

        channel = ChatChannel(
            id=get_first("channel_id"),
            platform=self.platform_name,
            name=get_first("channel_name"),
            team_id=get_first("team_id"),
        )

        command_name = get_first("command").lstrip("/")
        command_text = get_first("text")

        return WebhookEvent(
            platform=self.platform_name,
            event_type="slash_command",
            raw_payload=parsed,
            command=BotCommand(
                name=command_name,
                text=f"/{command_name} {command_text}".strip(),
                args=command_text.split() if command_text else [],
                user=user,
                channel=channel,
                platform=self.platform_name,
                response_url=get_first("response_url"),
                metadata={"trigger_id": get_first("trigger_id")},
            ),
        )

    def _parse_interaction_payload(self, payload: dict) -> WebhookEvent:
        """Parse interactive component payload."""
        interaction_type = payload.get("type", "")

        user_data = payload.get("user", {})
        user = ChatUser(
            id=user_data.get("id", ""),
            platform=self.platform_name,
            username=user_data.get("username"),
            display_name=user_data.get("name"),
        )

        channel_data = payload.get("channel", {})
        channel = ChatChannel(
            id=channel_data.get("id", ""),
            platform=self.platform_name,
            name=channel_data.get("name"),
            team_id=payload.get("team", {}).get("id"),
        )

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=interaction_type,
            raw_payload=payload,
        )

        if interaction_type == "block_actions":
            actions = payload.get("actions", [])
            if actions:
                action = actions[0]
                event.interaction = UserInteraction(
                    id=payload.get("trigger_id", ""),
                    interaction_type=(
                        InteractionType.BUTTON_CLICK
                        if action.get("type") == "button"
                        else InteractionType.SELECT_MENU
                    ),
                    action_id=action.get("action_id", ""),
                    value=action.get("value"),
                    values=action.get("selected_options", []),
                    user=user,
                    channel=channel,
                    message_id=payload.get("message", {}).get("ts"),
                    platform=self.platform_name,
                    response_url=payload.get("response_url"),
                )

        elif interaction_type == "view_submission":
            event.interaction = UserInteraction(
                id=payload.get("trigger_id", ""),
                interaction_type=InteractionType.MODAL_SUBMIT,
                action_id=payload.get("view", {}).get("callback_id", ""),
                user=user,
                channel=channel,
                platform=self.platform_name,
                response_url=payload.get("response_url"),
                metadata={"view": payload.get("view", {})},
            )

        return event

    def _parse_event_callback(self, payload: dict) -> WebhookEvent:
        """Parse Events API callback."""
        event_data = payload.get("event", {})
        event_type = event_data.get("type", "")

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=event_type,
            raw_payload=payload,
        )

        if event_type == "message" and not event_data.get("bot_id"):
            user = ChatUser(
                id=event_data.get("user", ""),
                platform=self.platform_name,
            )

            channel = ChatChannel(
                id=event_data.get("channel", ""),
                platform=self.platform_name,
                team_id=payload.get("team_id"),
            )

            event.message = ChatMessage(
                id=event_data.get("ts", ""),
                platform=self.platform_name,
                channel=channel,
                author=user,
                content=event_data.get("text", ""),
                thread_id=event_data.get("thread_ts"),
            )

        return event

    # ==========================================================================
    # Evidence Collection
    # ==========================================================================

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get message history from a Slack channel with timeout.

        Uses conversations.history API to retrieve messages.

        Args:
            channel_id: Channel ID to get history from
            limit: Maximum number of messages (max 1000)
            oldest: Oldest timestamp to include
            latest: Latest timestamp to include
            **kwargs: Additional API parameters

        Returns:
            List of ChatMessage objects
        """
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for Slack API")
            return []

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open - cannot get channel history")
            return []

        params: dict[str, Any] = {
            "channel": channel_id,
            "limit": min(limit, 1000),  # Slack API max
        }

        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest

        # Include thread replies if requested
        if kwargs.get("include_all_metadata", True):
            params["include_all_metadata"] = True

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/conversations.history",
                        headers=self._get_headers(),
                        params=params,
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack API error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return []

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    messages: list[ChatMessage] = []
                    channel_info = await self.get_channel_info(channel_id)
                    channel = channel_info or ChatChannel(
                        id=channel_id,
                        platform=self.platform_name,
                    )

                    for msg in data.get("messages", []):
                        # Skip bot messages if configured
                        if kwargs.get("skip_bots", True) and msg.get("bot_id"):
                            continue

                        user = ChatUser(
                            id=msg.get("user", msg.get("bot_id", "")),
                            platform=self.platform_name,
                            is_bot=bool(msg.get("bot_id")),
                        )

                        chat_msg = ChatMessage(
                            id=msg.get("ts", ""),
                            platform=self.platform_name,
                            channel=channel,
                            author=user,
                            content=msg.get("text", ""),
                            thread_id=msg.get("thread_ts"),
                            timestamp=datetime.fromtimestamp(
                                float(msg.get("ts", "0").split(".")[0])
                            ),
                            metadata={
                                "reply_count": msg.get("reply_count", 0),
                                "reactions": msg.get("reactions", []),
                            },
                        )
                        messages.append(chat_msg)

                    return messages

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Error getting Slack channel history: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return []

    async def collect_evidence(
        self,
        channel_id: str,
        query: Optional[str] = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a Slack channel, filters by relevance,
        and converts to ChatEvidence format with provenance tracking.

        Args:
            channel_id: Slack channel ID
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include threaded replies
            min_relevance: Minimum relevance score for inclusion (0-1)
            **kwargs: Additional options

        Returns:
            List of ChatEvidence objects with relevance scoring
        """
        # Get channel history
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=limit,
            **kwargs,
        )

        if not messages:
            return []

        # Convert to evidence with relevance scoring
        evidence_list: list[ChatEvidence] = []

        for msg in messages:
            # Calculate relevance
            relevance = self._compute_message_relevance(msg, query)

            # Skip low-relevance messages
            if relevance < min_relevance:
                continue

            # Create evidence
            evidence = ChatEvidence.from_message(
                message=msg,
                query=query,
                relevance_score=relevance,
            )

            evidence_list.append(evidence)

        # Sort by relevance (highest first)
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        # Optionally fetch thread replies for high-relevance messages
        if include_threads:
            await self._enrich_with_threads(evidence_list, limit=5, **kwargs)

        logger.info(
            f"Collected {len(evidence_list)} evidence items from Slack channel {channel_id}"
        )

        return evidence_list

    async def _enrich_with_threads(
        self,
        evidence_list: list[ChatEvidence],
        limit: int = 5,
        **kwargs: Any,
    ) -> None:
        """Enrich evidence with thread reply information."""
        if not HTTPX_AVAILABLE:
            return

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            return

        for evidence in evidence_list[:limit]:
            # Only enrich if this is a thread root with replies
            reply_count = evidence.metadata.get("reply_count", 0)
            if not evidence.is_thread_root or reply_count == 0:
                continue

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/conversations.replies",
                        headers=self._get_headers(),
                        params={
                            "channel": evidence.channel_id,
                            "ts": evidence.source_id,
                            "limit": 10,
                        },
                    )
                    data = response.json()

                    if data.get("ok"):
                        replies = data.get("messages", [])[1:]  # Skip root
                        evidence.reply_count = len(replies)
                        evidence.metadata["thread_replies"] = [
                            {
                                "text": r.get("text", "")[:200],
                                "user": r.get("user", ""),
                                "ts": r.get("ts", ""),
                            }
                            for r in replies
                        ]

            except Exception as e:
                logger.debug(f"Error enriching thread: {e}")

    async def search_messages(
        self,
        query: str,
        channel_id: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Search for messages across Slack workspace with timeout.

        Uses Slack's search.messages API (requires search:read scope).

        Args:
            query: Search query
            channel_id: Optional channel to restrict search
            limit: Maximum results to return
            **kwargs: Additional search parameters

        Returns:
            List of ChatEvidence from matching messages
        """
        if not HTTPX_AVAILABLE:
            return []

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open - cannot search messages")
            return []

        search_query = query
        if channel_id:
            search_query = f"in:<#{channel_id}> {query}"

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/search.messages",
                        headers=self._get_headers(),
                        params={
                            "query": search_query,
                            "count": limit,
                            "sort": kwargs.get("sort", "score"),
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack search error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return []

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    matches = data.get("messages", {}).get("matches", [])
                    evidence_list: list[ChatEvidence] = []

                    for match in matches:
                        channel = ChatChannel(
                            id=match.get("channel", {}).get("id", ""),
                            platform=self.platform_name,
                            name=match.get("channel", {}).get("name"),
                        )

                        user = ChatUser(
                            id=match.get("user", ""),
                            platform=self.platform_name,
                            username=match.get("username"),
                        )

                        msg = ChatMessage(
                            id=match.get("ts", ""),
                            platform=self.platform_name,
                            channel=channel,
                            author=user,
                            content=match.get("text", ""),
                            timestamp=datetime.fromtimestamp(
                                float(match.get("ts", "0").split(".")[0])
                            ),
                            metadata={
                                "permalink": match.get("permalink"),
                                "score": match.get("score"),
                            },
                        )

                        evidence = ChatEvidence.from_message(
                            message=msg,
                            query=query,
                            relevance_score=match.get("score", 1.0) / 100,  # Normalize
                        )
                        evidence.metadata["permalink"] = match.get("permalink")

                        evidence_list.append(evidence)

                    return evidence_list

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack search error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return []

    # =========================================================================
    # Reactions / Emoji Support
    # =========================================================================

    async def add_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
        **kwargs: Any,
    ) -> bool:
        """Add an emoji reaction to a message.

        Args:
            channel_id: Channel containing the message
            message_id: Timestamp of the message (Slack uses ts as message ID)
            emoji: Emoji name without colons (e.g., "thumbsup" not ":thumbsup:")
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack reactions")
            return False

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/reactions.add",
                        headers=self._get_headers(),
                        json={
                            "channel": channel_id,
                            "timestamp": message_id,
                            "name": emoji.strip(":"),
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        # "already_reacted" is not really a failure
                        if error == "already_reacted":
                            return True
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack add_reaction error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return True

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack add_reaction error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    async def remove_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
        **kwargs: Any,
    ) -> bool:
        """Remove an emoji reaction from a message.

        Args:
            channel_id: Channel containing the message
            message_id: Timestamp of the message
            emoji: Emoji name without colons
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack reactions")
            return False

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/reactions.remove",
                        headers=self._get_headers(),
                        json={
                            "channel": channel_id,
                            "timestamp": message_id,
                            "name": emoji.strip(":"),
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        # "no_reaction" is not really a failure
                        if error == "no_reaction":
                            return True
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack remove_reaction error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return True

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack remove_reaction error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    # =========================================================================
    # Channel & User Discovery
    # =========================================================================

    async def list_channels(
        self,
        exclude_archived: bool = True,
        types: str = "public_channel,private_channel",
        limit: int = 100,
        **kwargs: Any,
    ) -> list[ChatChannel]:
        """List channels in the workspace.

        Args:
            exclude_archived: Whether to exclude archived channels
            types: Comma-separated channel types (public_channel, private_channel)
            limit: Maximum number of channels to return
            **kwargs: Additional parameters

        Returns:
            List of ChatChannel objects
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack list_channels")
            return []

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/conversations.list",
                        headers=self._get_headers(),
                        params={
                            "exclude_archived": str(exclude_archived).lower(),
                            "types": types,
                            "limit": limit,
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack list_channels error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return []

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    channels = []
                    for ch in data.get("channels", []):
                        channels.append(
                            ChatChannel(
                                id=ch.get("id", ""),
                                platform=self.platform_name,
                                name=ch.get("name"),
                                metadata={
                                    "is_private": ch.get("is_private", False),
                                    "is_archived": ch.get("is_archived", False),
                                    "is_member": ch.get("is_member", False),
                                    "num_members": ch.get("num_members", 0),
                                    "topic": ch.get("topic", {}).get("value"),
                                    "purpose": ch.get("purpose", {}).get("value"),
                                },
                            )
                        )
                    return channels

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack list_channels error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return []

    async def list_users(
        self,
        limit: int = 100,
        include_bots: bool = False,
        **kwargs: Any,
    ) -> list[ChatUser]:
        """List users in the workspace.

        Args:
            limit: Maximum number of users to return
            include_bots: Whether to include bot users
            **kwargs: Additional parameters

        Returns:
            List of ChatUser objects
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack list_users")
            return []

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/users.list",
                        headers=self._get_headers(),
                        params={"limit": limit},
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack list_users error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return []

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    users = []
                    for member in data.get("members", []):
                        # Skip bots if not requested
                        if not include_bots and member.get("is_bot", False):
                            continue
                        # Skip deleted users
                        if member.get("deleted", False):
                            continue

                        profile = member.get("profile", {})
                        users.append(
                            ChatUser(
                                id=member.get("id", ""),
                                platform=self.platform_name,
                                username=member.get("name"),
                                display_name=profile.get("display_name")
                                or profile.get("real_name"),
                                avatar_url=profile.get("image_72"),
                                metadata={
                                    "email": profile.get("email"),
                                    "title": profile.get("title"),
                                    "is_admin": member.get("is_admin", False),
                                    "is_owner": member.get("is_owner", False),
                                    "tz": member.get("tz"),
                                },
                            )
                        )
                    return users

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack list_users error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return []

    # =========================================================================
    # User Mention Helpers
    # =========================================================================

    @staticmethod
    def format_user_mention(user_id: str) -> str:
        """Format a user mention for Slack messages.

        Args:
            user_id: The Slack user ID (e.g., "U123ABC")

        Returns:
            Formatted mention string (e.g., "<@U123ABC>")
        """
        return f"<@{user_id}>"

    @staticmethod
    def format_channel_mention(channel_id: str) -> str:
        """Format a channel mention for Slack messages.

        Args:
            channel_id: The Slack channel ID (e.g., "C123ABC")

        Returns:
            Formatted mention string (e.g., "<#C123ABC>")
        """
        return f"<#{channel_id}>"

    # =========================================================================
    # Modal / View Support
    # =========================================================================

    async def open_modal(
        self,
        trigger_id: str,
        view: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[str]:
        """Open a modal view.

        Args:
            trigger_id: Trigger ID from interaction payload (valid for 3 seconds)
            view: Modal view payload following Slack Block Kit spec
            **kwargs: Additional parameters

        Returns:
            View ID if successful, None otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack open_modal")
            return None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/views.open",
                        headers=self._get_headers(),
                        json={
                            "trigger_id": trigger_id,
                            "view": view,
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack open_modal error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return None

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return data.get("view", {}).get("id")

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack open_modal error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return None

    async def update_modal(
        self,
        view_id: str,
        view: dict[str, Any],
        view_hash: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Update an existing modal view.

        Args:
            view_id: ID of the view to update
            view: Updated view payload
            view_hash: Optional view hash for optimistic locking
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack update_modal")
            return False

        for attempt in range(self._max_retries):
            try:
                payload: dict[str, Any] = {
                    "view_id": view_id,
                    "view": view,
                }
                if view_hash:
                    payload["hash"] = view_hash

                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/views.update",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack update_modal error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return True

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack update_modal error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    # =========================================================================
    # Pinned Messages
    # =========================================================================

    async def pin_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Pin a message to a channel.

        Args:
            channel_id: Channel ID
            message_id: Message timestamp
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack pin_message")
            return False

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/pins.add",
                        headers=self._get_headers(),
                        json={
                            "channel": channel_id,
                            "timestamp": message_id,
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        # "already_pinned" is not really a failure
                        if error == "already_pinned":
                            return True
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack pin_message error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return True

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack pin_message error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    async def unpin_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Unpin a message from a channel.

        Args:
            channel_id: Channel ID
            message_id: Message timestamp
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack unpin_message")
            return False

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{SLACK_API_BASE}/pins.remove",
                        headers=self._get_headers(),
                        json={
                            "channel": channel_id,
                            "timestamp": message_id,
                        },
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        # "no_pin" is not really a failure
                        if error == "no_pin":
                            return True
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack unpin_message error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()
                    return True

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack unpin_message error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return False

    async def get_pinned_messages(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Get pinned messages in a channel.

        Args:
            channel_id: Channel ID
            **kwargs: Additional parameters

        Returns:
            List of pinned ChatMessage objects
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Slack connector")

        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("Circuit breaker open for Slack get_pinned_messages")
            return []

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/pins.list",
                        headers=self._get_headers(),
                        params={"channel": channel_id},
                    )
                    data = response.json()

                    if not data.get("ok"):
                        error = data.get("error", "")
                        if _is_retryable_error(response.status_code, error):
                            if attempt < self._max_retries - 1:
                                await _exponential_backoff(attempt)
                                continue
                        logger.error(f"Slack get_pinned_messages error: {error}")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return []

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    messages = []
                    for item in data.get("items", []):
                        if item.get("type") != "message":
                            continue
                        msg_data = item.get("message", {})
                        channel = ChatChannel(
                            id=channel_id,
                            platform=self.platform_name,
                        )
                        user = ChatUser(
                            id=msg_data.get("user", ""),
                            platform=self.platform_name,
                        )
                        messages.append(
                            ChatMessage(
                                id=msg_data.get("ts", ""),
                                platform=self.platform_name,
                                channel=channel,
                                author=user,
                                content=msg_data.get("text", ""),
                                timestamp=datetime.fromtimestamp(
                                    float(msg_data.get("ts", "0").split(".")[0])
                                ),
                                metadata={"pinned": True},
                            )
                        )
                    return messages

            except httpx.TimeoutException:
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

            except Exception as e:
                logger.error(f"Slack get_pinned_messages error: {e}")
                if attempt < self._max_retries - 1:
                    await _exponential_backoff(attempt)
                    continue

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        return []


__all__ = ["SlackConnector"]

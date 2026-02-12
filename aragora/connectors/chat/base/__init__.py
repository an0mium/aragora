"""
Chat Platform Connector - Abstract base class for chat integrations.

All chat platform connectors inherit from ChatPlatformConnector and implement
standardized methods for:
- Sending and updating messages
- Handling bot commands
- Processing user interactions
- File operations (upload/download)
- Voice message handling

Includes circuit breaker support for fault tolerance.

This module composes the ChatPlatformConnector from multiple mixins for
better code organization while maintaining the same public API.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.connectors.base import ConnectorCapabilities

from ..http_resilience import HTTPResilienceMixin
from ._channel_user import ChannelUserMixin
from ._evidence import EvidenceMixin
from ._file_ops import FileOperationsMixin
from ._messaging import MessagingMixin
from ._rich_context import RichContextMixin
from ._session import SessionMixin
from ._webhook import WebhookMixin

logger = logging.getLogger(__name__)


class ChatPlatformConnector(
    HTTPResilienceMixin,
    MessagingMixin,
    FileOperationsMixin,
    ChannelUserMixin,
    WebhookMixin,
    EvidenceMixin,
    RichContextMixin,
    SessionMixin,
    ABC,
):
    """
    Abstract base class for chat platform integrations.

    Provides a unified interface for interacting with chat platforms
    like Slack, Microsoft Teams, Discord, and Google Chat.

    Subclasses must implement the abstract methods to handle
    platform-specific APIs and message formats.

    Includes circuit breaker support for fault tolerance in HTTP operations.

    This class is composed of multiple mixins:
    - HTTPResilienceMixin: Circuit breaker, retry, HTTP request handling
    - MessagingMixin: Message send/update/delete, command/interaction handling
    - FileOperationsMixin: File upload/download, voice messages
    - ChannelUserMixin: Channel/user info, DMs, reactions, pinning, threading
    - WebhookMixin: Webhook verification and event parsing
    - EvidenceMixin: Evidence collection from channels
    - RichContextMixin: Rich context fetching for deliberation
    - SessionMixin: Session management integration
    """

    def __init__(
        self,
        bot_token: str | None = None,
        signing_secret: str | None = None,
        webhook_url: str | None = None,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown: float = 60.0,
        request_timeout: float = 30.0,
        **config: Any,
    ):
        """
        Initialize the connector.

        Args:
            bot_token: API token for the bot
            signing_secret: Secret for webhook verification
            webhook_url: Default webhook URL for sending messages
            enable_circuit_breaker: Enable circuit breaker for fault tolerance
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_cooldown: Seconds before attempting recovery
            request_timeout: HTTP request timeout in seconds
            **config: Additional platform-specific configuration
        """
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.webhook_url = webhook_url
        self.config = config
        self._initialized = False

        # Circuit breaker settings
        self._enable_circuit_breaker = enable_circuit_breaker
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_cooldown = circuit_breaker_cooldown
        self._request_timeout = request_timeout
        self._circuit_breaker: Any | None = None
        self._circuit_breaker_initialized = False

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier (e.g., 'slack', 'teams', 'discord')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def platform_display_name(self) -> str:
        """Return human-readable platform name (e.g., 'Microsoft Teams')."""
        raise NotImplementedError

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    async def test_connection(self) -> dict[str, Any]:
        """
        Test the connection to the platform.

        Returns:
            Dict with success status and details
        """
        return {
            "platform": self.platform_name,
            "success": bool(self.bot_token or self.webhook_url),
            "bot_token_configured": bool(self.bot_token),
            "webhook_configured": bool(self.webhook_url),
        }

    async def get_health(self) -> dict[str, Any]:
        """
        Get detailed health status for the channel connector.

        Returns comprehensive health information including circuit breaker
        state, configuration status, and connectivity details.

        Returns:
            Dict with health status and metrics
        """
        details: dict[str, Any] = {}
        circuit_breaker: dict[str, Any] | None = None
        health: dict[str, Any] = {
            "platform": self.platform_name,
            "display_name": self.platform_display_name,
            "status": "unknown",
            "configured": self.is_configured,
            "timestamp": time.time(),
            "circuit_breaker": circuit_breaker,
            "details": details,
        }

        # Check configuration
        if not self.is_configured:
            health["status"] = "unconfigured"
            details["error"] = "Missing required configuration (bot_token or webhook_url)"
            return health

        # Get circuit breaker status
        cb = self._get_circuit_breaker()
        if cb:
            cb_status = cb.get_status()
            cb_info: dict[str, Any] = {
                "state": cb_status,
                "enabled": True,
            }
            if cb_status == "open":
                cb_info["cooldown_remaining"] = cb.cooldown_remaining()
            health["circuit_breaker"] = cb_info

            # Determine health based on circuit breaker
            if cb_status == "open":
                health["status"] = "unhealthy"
                details["reason"] = "Circuit breaker is open due to repeated failures"
            elif cb_status == "half_open":
                health["status"] = "degraded"
                details["reason"] = "Circuit breaker in recovery mode"
            else:
                health["status"] = "healthy"
        else:
            health["circuit_breaker"] = {"enabled": False}
            health["status"] = "healthy"

        # Add configuration details
        details["bot_token_configured"] = bool(self.bot_token)
        details["webhook_configured"] = bool(self.webhook_url)
        details["request_timeout"] = self._request_timeout

        return health

    @property
    def is_configured(self) -> bool:
        """Check if the connector has minimum required configuration."""
        return bool(self.bot_token or self.webhook_url)

    @property
    def is_connected(self) -> bool:
        """Whether the connector has an active connection."""
        return self.is_configured and self._initialized

    async def connect(self) -> bool:
        """
        Establish connection to the chat platform.

        For webhook-based connectors, this validates configuration.
        For socket-based connectors, subclasses should establish connection.

        Returns:
            True if connection successful
        """
        if not self.is_configured:
            return False
        self._initialized = True
        return True

    async def disconnect(self) -> None:
        """Close connection to the chat platform."""
        self._initialized = False

    async def send(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Send a message to the chat platform.

        This is a generic wrapper around send_message for ConnectorProtocol.
        For full functionality, use platform-specific methods.

        Args:
            data: Dict with 'channel_id' and 'text' keys

        Returns:
            Response from send_message
        """
        channel_id = data.get("channel_id", data.get("channel"))
        text = data.get("text", data.get("message", ""))

        if not channel_id:
            return {"success": False, "error": "channel_id required"}

        result = await self.send_message(
            channel_id=channel_id,
            text=text,
            thread_id=data.get("thread_id"),
        )
        return {"success": result is not None, "response": result}

    async def receive(self) -> dict[str, Any] | None:
        """
        Receive messages from the platform.

        Chat connectors use webhooks for receiving, so this returns None.
        Messages are received via parse_webhook_event() instead.

        Returns:
            None - use webhooks for receiving messages
        """
        return None

    def capabilities(self) -> ConnectorCapabilities:
        """
        Report the capabilities of this chat platform connector.

        Returns:
            ConnectorCapabilities describing chat platform features
        """
        from aragora.connectors.base import ConnectorCapabilities

        return ConnectorCapabilities(
            can_send=True,
            can_receive=True,  # Via webhooks
            can_search=False,  # Chat connectors don't search
            can_sync=False,
            can_stream=False,  # Subclasses may override
            can_batch=False,
            is_stateful=False,  # Webhook-based
            requires_auth=True,
            supports_oauth=False,  # Subclasses may override
            supports_webhooks=True,
            supports_files=True,  # Via upload_file
            supports_rich_text=True,  # Most platforms support markdown
            supports_reactions=True,  # Via add_reaction
            supports_threads=True,  # Via thread_id
            supports_voice=True,  # Via send_voice_message
            supports_delivery_receipts=False,
            supports_retry=True,
            has_circuit_breaker=self._enable_circuit_breaker,
            platform_features=[
                "commands",  # Bot commands
                "interactions",  # Button interactions
                "mentions",  # User mentions
            ],
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(platform={self.platform_name})"


__all__ = [
    "ChatPlatformConnector",
    # Mixins (for advanced usage/testing)
    "MessagingMixin",
    "FileOperationsMixin",
    "ChannelUserMixin",
    "WebhookMixin",
    "EvidenceMixin",
    "RichContextMixin",
    "SessionMixin",
]

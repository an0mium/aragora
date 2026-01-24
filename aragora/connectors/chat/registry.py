"""
Chat Platform Registry - Factory and management for chat connectors.

Provides a unified interface for creating and managing chat platform
connectors across Slack, Teams, Discord, and Google Chat.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Type

from .base import ChatPlatformConnector

logger = logging.getLogger(__name__)

# Registry of connector classes
_CONNECTOR_CLASSES: Dict[str, Type[ChatPlatformConnector]] = {}

# Singleton instances
_CONNECTOR_INSTANCES: Dict[str, ChatPlatformConnector] = {}


def register_connector(platform: str, connector_class: Type[ChatPlatformConnector]) -> None:
    """
    Register a connector class for a platform.

    Args:
        platform: Platform identifier (e.g., 'slack', 'teams')
        connector_class: ChatPlatformConnector subclass
    """
    _CONNECTOR_CLASSES[platform.lower()] = connector_class
    logger.debug(f"Registered chat connector: {platform}")


def get_connector(
    platform: str,
    **config: Any,
) -> Optional[ChatPlatformConnector]:
    """
    Get or create a connector instance for a platform.

    Args:
        platform: Platform identifier
        **config: Configuration passed to connector constructor

    Returns:
        ChatPlatformConnector instance or None if not available
    """
    platform = platform.lower()

    # Check for existing instance
    if platform in _CONNECTOR_INSTANCES and not config:
        return _CONNECTOR_INSTANCES[platform]

    # Get connector class
    connector_class = _CONNECTOR_CLASSES.get(platform)
    if connector_class is None:
        # Try lazy loading
        connector_class = _lazy_load_connector(platform)

    if connector_class is None:
        logger.warning(f"No connector available for platform: {platform}")
        return None

    # Create instance
    try:
        connector = connector_class(**config)

        # Cache if no custom config
        if not config:
            _CONNECTOR_INSTANCES[platform] = connector

        return connector
    except Exception as e:
        logger.error(f"Failed to create {platform} connector: {e}")
        return None


def _lazy_load_connector(platform: str) -> Optional[Type[ChatPlatformConnector]]:
    """Lazy-load connector class on first access."""
    try:
        if platform == "slack":
            # Slack uses existing handler, but we can create a connector wrapper
            from aragora.connectors.chat.slack import SlackConnector

            _CONNECTOR_CLASSES["slack"] = SlackConnector
            return SlackConnector

        elif platform == "teams":
            from aragora.connectors.chat.teams import TeamsConnector

            _CONNECTOR_CLASSES["teams"] = TeamsConnector
            return TeamsConnector

        elif platform == "discord":
            from aragora.connectors.chat.discord import DiscordConnector

            _CONNECTOR_CLASSES["discord"] = DiscordConnector
            return DiscordConnector

        elif platform in ("google_chat", "googlechat", "gchat"):
            from aragora.connectors.chat.google_chat import GoogleChatConnector

            _CONNECTOR_CLASSES["google_chat"] = GoogleChatConnector
            _CONNECTOR_CLASSES["googlechat"] = GoogleChatConnector
            _CONNECTOR_CLASSES["gchat"] = GoogleChatConnector
            return GoogleChatConnector

        elif platform == "telegram":
            from aragora.connectors.chat.telegram import TelegramConnector

            _CONNECTOR_CLASSES["telegram"] = TelegramConnector
            return TelegramConnector

        elif platform == "whatsapp":
            from aragora.connectors.chat.whatsapp import WhatsAppConnector

            _CONNECTOR_CLASSES["whatsapp"] = WhatsAppConnector
            return WhatsAppConnector

    except ImportError as e:
        logger.debug(f"Could not load {platform} connector: {e}")

    return None


def list_available_platforms() -> list[str]:
    """List all registered platform identifiers."""
    # Ensure all connectors are loaded
    for platform in ["slack", "teams", "discord", "google_chat", "telegram", "whatsapp"]:
        _lazy_load_connector(platform)

    return list(_CONNECTOR_CLASSES.keys())


def get_configured_platforms() -> list[str]:
    """List platforms that have required environment variables set."""
    configured = []

    # Check Slack
    if os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_WEBHOOK_URL"):
        configured.append("slack")

    # Check Teams
    if os.environ.get("TEAMS_APP_ID") and os.environ.get("TEAMS_APP_PASSWORD"):
        configured.append("teams")

    # Check Discord
    if os.environ.get("DISCORD_BOT_TOKEN"):
        configured.append("discord")

    # Check Google Chat
    if os.environ.get("GOOGLE_CHAT_CREDENTIALS"):
        configured.append("google_chat")

    # Check Telegram
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        configured.append("telegram")

    # Check WhatsApp
    if os.environ.get("WHATSAPP_ACCESS_TOKEN") and os.environ.get("WHATSAPP_PHONE_NUMBER_ID"):
        configured.append("whatsapp")

    return configured


def get_all_connectors(**config: Any) -> Dict[str, ChatPlatformConnector]:
    """
    Get connectors for all configured platforms.

    Args:
        **config: Additional configuration for all connectors

    Returns:
        Dict mapping platform name to connector instance
    """
    connectors = {}

    for platform in get_configured_platforms():
        connector = get_connector(platform, **config)
        if connector and connector.is_configured:
            connectors[platform] = connector

    return connectors


def clear_instances() -> None:
    """Clear all cached connector instances (for testing)."""
    _CONNECTOR_INSTANCES.clear()


class ChatPlatformRegistry:
    """
    Registry class for chat platform connectors.

    Provides methods for managing connectors and routing messages
    to the appropriate platform.
    """

    def __init__(self):
        """Initialize the registry."""
        self._connectors: Dict[str, ChatPlatformConnector] = {}

    def register(self, connector: ChatPlatformConnector) -> None:
        """Register a connector instance."""
        self._connectors[connector.platform_name] = connector

    def get(self, platform: str) -> Optional[ChatPlatformConnector]:
        """Get a registered connector by platform name."""
        return self._connectors.get(platform.lower())

    def all(self) -> Dict[str, ChatPlatformConnector]:
        """Get all registered connectors."""
        return dict(self._connectors)

    def platforms(self) -> list[str]:
        """List registered platform names."""
        return list(self._connectors.keys())

    async def broadcast(
        self,
        text: str,
        channels: Dict[str, str],  # platform -> channel_id
        blocks: Optional[Dict[str, list[dict]]] = None,  # platform -> blocks
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple platforms.

        Args:
            text: Message text
            channels: Dict mapping platform to channel ID
            blocks: Optional dict mapping platform to platform-specific blocks

        Returns:
            Dict mapping platform to success status
        """
        results = {}

        for platform, channel_id in channels.items():
            connector = self.get(platform)
            if connector is None:
                results[platform] = False
                continue

            platform_blocks = blocks.get(platform) if blocks else None

            try:
                response = await connector.send_message(
                    channel_id,
                    text,
                    blocks=platform_blocks,
                )
                results[platform] = response.success
            except Exception as e:
                logger.error(f"Broadcast to {platform} failed: {e}")
                results[platform] = False

        return results

    def load_configured(self, **config: Any) -> None:
        """Load all configured connectors."""
        for platform, connector in get_all_connectors(**config).items():
            self.register(connector)


# Global registry instance
_registry: Optional[ChatPlatformRegistry] = None


def get_registry() -> ChatPlatformRegistry:
    """Get or create the global chat platform registry."""
    global _registry
    if _registry is None:
        _registry = ChatPlatformRegistry()
        _registry.load_configured()
    return _registry

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

BACKWARD COMPATIBILITY NOTE:
This module now re-exports from the decomposed `base/` subpackage.
The API remains identical - all imports from `aragora.connectors.chat.base`
will continue to work as before.
"""

from __future__ import annotations

# Re-export the main class and all mixins from the base package
from .base import (
    ChannelUserMixin,
    ChatPlatformConnector,
    EvidenceMixin,
    FileOperationsMixin,
    MessagingMixin,
    RichContextMixin,
    SessionMixin,
    WebhookMixin,
)

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

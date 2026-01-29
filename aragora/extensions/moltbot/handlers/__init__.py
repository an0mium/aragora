"""
Moltbot HTTP Handlers - REST API Layer for Moltbot Extension.

Provides HTTP endpoints for:
- Device gateway management (devices, commands)
- Inbox message and channel management
- Voice session management (STT/TTS)
- Canvas collaboration
- Onboarding flows and sessions
- Device capability queries

All handlers follow Aragora's BaseHandler pattern with RBAC integration.
"""

from __future__ import annotations

from .canvas import MoltbotCanvasHandler
from .capabilities import MoltbotCapabilitiesHandler
from .gateway import MoltbotGatewayHandler
from .inbox import MoltbotInboxHandler
from .onboarding import MoltbotOnboardingHandler
from .types import serialize_datetime, serialize_enum
from .voice import MoltbotVoiceHandler

__all__ = [
    # Handlers
    "MoltbotGatewayHandler",
    "MoltbotInboxHandler",
    "MoltbotVoiceHandler",
    "MoltbotCanvasHandler",
    "MoltbotOnboardingHandler",
    "MoltbotCapabilitiesHandler",
    "get_all_handlers",
    # Helpers
    "serialize_datetime",
    "serialize_enum",
]


def get_all_handlers():
    """Get all Moltbot handler classes."""
    return [
        MoltbotGatewayHandler,
        MoltbotInboxHandler,
        MoltbotVoiceHandler,
        MoltbotCanvasHandler,
        MoltbotOnboardingHandler,
        MoltbotCapabilitiesHandler,
    ]

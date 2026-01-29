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

from .gateway import MoltbotGatewayHandler
from .inbox import MoltbotInboxHandler
from .voice import MoltbotVoiceHandler
from .canvas import MoltbotCanvasHandler
from .onboarding import MoltbotOnboardingHandler
from .capabilities import MoltbotCapabilitiesHandler

__all__ = [
    "MoltbotGatewayHandler",
    "MoltbotInboxHandler",
    "MoltbotVoiceHandler",
    "MoltbotCanvasHandler",
    "MoltbotOnboardingHandler",
    "MoltbotCapabilitiesHandler",
]

# Collect all routes for unified registration
MOLTBOT_ROUTES = []


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

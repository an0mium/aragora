"""
Chat Platform Webhook Handlers.

Unified handlers for processing webhooks from all chat platforms.

Routes:
- POST /api/chat/webhook - Auto-detect platform
- POST /api/chat/{platform}/webhook - Platform-specific
- GET  /api/chat/status - Integration status
"""

from .router import (
    ChatWebhookRouter,
    get_webhook_router,
)

__all__ = [
    "ChatWebhookRouter",
    "get_webhook_router",
]

# Export handler class if available
try:
    from .router import ChatHandler  # noqa: F401

    __all__.append("ChatHandler")
except ImportError:
    pass

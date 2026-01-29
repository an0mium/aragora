"""
Debate Origin Tracking for Bidirectional Chat.

Tracks which chat channel/platform originated each debate so results
can be routed back to the user on the same channel.

Usage:
    from aragora.server.debate_origin import (
        register_debate_origin,
        get_debate_origin,
        route_debate_result,
    )

    # When starting a debate from Telegram
    register_debate_origin(
        debate_id="abc123",
        platform="telegram",
        channel_id="12345678",
        user_id="87654321",
        metadata={"username": "john_doe"},
    )

    # When debate completes, route result back
    await route_debate_result(debate_id, result)
"""

from __future__ import annotations

# Models
from .models import DebateOrigin  # noqa: F401

# Stores and constants
from .stores import (  # noqa: F401
    ORIGIN_TTL_SECONDS,
    PostgresOriginStore,
    SQLiteOriginStore,
    _get_postgres_store,
    _get_postgres_store_sync,
    _get_sqlite_store,
)

# Formatting utilities
from .formatting import (  # noqa: F401
    _format_result_message,
    _format_receipt_summary,
    format_error_for_chat,
)

# Voice synthesis
from .voice import _synthesize_voice  # noqa: F401

# Session management
from .sessions import (  # noqa: F401
    _create_and_link_session,
    get_sessions_for_debate,
)

# Registry (origin store, registration, lookup, cleanup)
from .registry import (  # noqa: F401
    _origin_store,
    _store_origin_redis,
    _load_origin_redis,
    register_debate_origin,
    get_debate_origin,
    get_debate_origin_async,
    mark_result_sent,
    cleanup_expired_origins,
)

# Router (result/receipt/error dispatch)
from .router import (  # noqa: F401
    USE_DOCK_ROUTING,
    route_debate_result,
    post_receipt_to_channel,
    send_error_to_channel,
    route_result_to_all_sessions,
)

# Platform senders (re-exported for backward compatibility)
from .senders import (  # noqa: F401
    _send_slack_result,
    _send_slack_receipt,
    _send_slack_error,
    _send_teams_result,
    _send_teams_receipt,
    _send_teams_error,
    _send_telegram_result,
    _send_telegram_receipt,
    _send_telegram_error,
    _send_telegram_voice,
    _send_discord_result,
    _send_discord_receipt,
    _send_discord_error,
    _send_discord_voice,
    _send_whatsapp_result,
    _send_whatsapp_voice,
    _send_email_result,
    _send_google_chat_result,
    _send_google_chat_receipt,
)

__all__ = [
    "DebateOrigin",
    "register_debate_origin",
    "get_debate_origin",
    "get_debate_origin_async",
    "mark_result_sent",
    "route_debate_result",
    "route_result_to_all_sessions",
    "get_sessions_for_debate",
    "post_receipt_to_channel",
    "format_error_for_chat",
    "send_error_to_channel",
    "cleanup_expired_origins",
]

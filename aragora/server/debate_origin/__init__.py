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

# Re-export models
from .models import DebateOrigin

# Re-export stores (for test patching and advanced usage)
from .stores import (
    ORIGIN_TTL_SECONDS,
    SQLiteOriginStore,
    PostgresOriginStore,
    _get_sqlite_store,
    _get_postgres_store,
    _get_postgres_store_sync,
)

# Re-export formatting functions
from .formatting import (
    _format_result_message,
    _format_receipt_summary,
    format_error_for_chat,
    format_consensus_event,
    format_compliance_event,
    format_knowledge_event,
    format_graph_debate_event,
    format_workflow_event,
)

# Re-export voice synthesis
from .voice import _synthesize_voice

# Re-export session management
from .sessions import (
    _create_and_link_session,
    get_sessions_for_debate,
)

# Re-export registry (origin registration, lookup, lifecycle)
from .registry import (
    _origin_store,
    _store_origin_redis,
    _load_origin_redis,
    register_debate_origin,
    get_debate_origin,
    get_debate_origin_async,
    mark_result_sent,
    cleanup_expired_origins,
)

# Re-export routing
from .router import (
    USE_DOCK_ROUTING,
    route_debate_result,
    route_capability_event,
    route_plan_result,
    post_receipt_to_channel,
    send_error_to_channel,
    route_result_to_all_sessions,
)

# Re-export senders for direct platform access
from .senders import (
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
    # Models
    "DebateOrigin",
    # Core API
    "register_debate_origin",
    "get_debate_origin",
    "get_debate_origin_async",
    "mark_result_sent",
    "route_debate_result",
    "route_capability_event",
    "route_plan_result",
    "route_result_to_all_sessions",
    "get_sessions_for_debate",
    "post_receipt_to_channel",
    "format_error_for_chat",
    "send_error_to_channel",
    "cleanup_expired_origins",
    # Stores (for advanced usage/testing)
    "ORIGIN_TTL_SECONDS",
    "SQLiteOriginStore",
    "PostgresOriginStore",
    "_get_sqlite_store",
    "_get_postgres_store",
    "_get_postgres_store_sync",
    "_origin_store",
    "_store_origin_redis",
    "_load_origin_redis",
    # Routing flag
    "USE_DOCK_ROUTING",
    # Formatting (internal but exported for testing)
    "_format_result_message",
    "_format_receipt_summary",
    "format_consensus_event",
    "format_compliance_event",
    "format_knowledge_event",
    "format_graph_debate_event",
    "format_workflow_event",
    # Voice
    "_synthesize_voice",
    # Sessions
    "_create_and_link_session",
    # Senders (internal but exported for direct access)
    "_send_slack_result",
    "_send_slack_receipt",
    "_send_slack_error",
    "_send_teams_result",
    "_send_teams_receipt",
    "_send_teams_error",
    "_send_telegram_result",
    "_send_telegram_receipt",
    "_send_telegram_error",
    "_send_telegram_voice",
    "_send_discord_result",
    "_send_discord_receipt",
    "_send_discord_error",
    "_send_discord_voice",
    "_send_whatsapp_result",
    "_send_whatsapp_voice",
    "_send_email_result",
    "_send_google_chat_result",
    "_send_google_chat_receipt",
]

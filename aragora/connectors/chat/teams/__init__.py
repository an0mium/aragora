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

This package is split into submodules for maintainability:
- _constants: Environment config, API endpoints, error classification
- _messaging: Send, update, delete messages and responses
- _files: File upload/download via Graph API
- _events: Webhook verification, event parsing, Adaptive Card formatting
- _channels: Channel history, evidence collection, channel/user info
- connector: TeamsConnector class (combines all mixins)
- thread_mgr: TeamsThreadManager class
"""

from __future__ import annotations

# Re-export the main classes
from aragora.connectors.chat.teams.connector import TeamsConnector
from aragora.connectors.chat.teams.thread_mgr import TeamsThreadManager

# Re-export constants and helpers for backwards compatibility
from aragora.connectors.chat.teams._constants import (
    BOT_FRAMEWORK_API_BASE,
    BOT_FRAMEWORK_AUTH_URL,
    GRAPH_API_BASE,
    GRAPH_AUTH_URL,
    GRAPH_SCOPE_FILES,
    HTTPX_AVAILABLE,
    TEAMS_APP_ID,
    TEAMS_APP_PASSWORD,
    TEAMS_REQUEST_TIMEOUT,
    TEAMS_TENANT_ID,
    TEAMS_UPLOAD_TIMEOUT,
    TRACING_AVAILABLE,
    _classify_teams_error,
    build_trace_headers,
)

__all__ = [
    "TeamsConnector",
    "TeamsThreadManager",
    "BOT_FRAMEWORK_API_BASE",
    "BOT_FRAMEWORK_AUTH_URL",
    "GRAPH_API_BASE",
    "GRAPH_AUTH_URL",
    "GRAPH_SCOPE_FILES",
    "HTTPX_AVAILABLE",
    "TEAMS_APP_ID",
    "TEAMS_APP_PASSWORD",
    "TEAMS_REQUEST_TIMEOUT",
    "TEAMS_TENANT_ID",
    "TEAMS_UPLOAD_TIMEOUT",
    "TRACING_AVAILABLE",
    "_classify_teams_error",
    "build_trace_headers",
]

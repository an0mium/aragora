"""
Microsoft Teams connector constants and error classification.

Contains environment configuration, API endpoints, and error
classification helpers shared across Teams submodules.
"""

from __future__ import annotations

import logging
import os

from aragora.connectors.exceptions import (
    ConnectorError,
    classify_connector_error,
)

logger = logging.getLogger(__name__)

# Try to import httpx for API calls
try:
    import httpx  # noqa: F401

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - Teams connector will have limited functionality")

# Distributed tracing support
try:
    from aragora.observability.tracing import build_trace_headers

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def build_trace_headers() -> dict[str, str]:
        return {}


# Environment configuration
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID", "")

# Timeout configuration (in seconds)
TEAMS_REQUEST_TIMEOUT = float(os.environ.get("TEAMS_REQUEST_TIMEOUT", "30"))
TEAMS_UPLOAD_TIMEOUT = float(os.environ.get("TEAMS_UPLOAD_TIMEOUT", "120"))

# Bot Framework API endpoints
BOT_FRAMEWORK_AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
BOT_FRAMEWORK_API_BASE = "https://smba.trafficmanager.net"

# Microsoft Graph API for file operations and channel history
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_AUTH_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

# Graph API scopes for different operations
GRAPH_SCOPE_FILES = "https://graph.microsoft.com/.default"


def _classify_teams_error(
    error_str: str,
    status_code: int = 0,
    retry_after: float | None = None,
) -> ConnectorError:
    """Classify a Teams/Graph API error into a specific ConnectorError type.

    This is a thin wrapper around the shared classify_connector_error function
    for backward compatibility.
    """
    return classify_connector_error(
        error_str,
        connector_name="teams",
        status_code=status_code if status_code else None,
        retry_after=retry_after,
    )

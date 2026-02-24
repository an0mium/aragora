"""
Session Health Monitoring Handlers.

Provides admin endpoints for monitoring and managing session health:
- GET /api/v1/auth/sessions/health   - Session health metrics (admin only)
- POST /api/v1/auth/sessions/sweep   - Trigger expired session cleanup (admin only)
- GET /api/v1/auth/sessions/active   - List active sessions for current user
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import auth_rate_limit

logger = logging.getLogger(__name__)


def _get_monitor():
    """Lazy import the session monitor to avoid circular imports."""
    from aragora.auth.session_monitor import get_session_monitor

    return get_session_monitor()


# =============================================================================
# Session Health Metrics (admin only)
# =============================================================================


@require_permission("admin:read")
@auth_rate_limit(
    requests_per_minute=30,
    limiter_name="auth_session_health",
    endpoint_name="session health metrics",
)
@handle_errors("session health metrics")
async def handle_session_health(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get session health metrics.

    GET /api/v1/auth/sessions/health

    Returns aggregate session metrics including active count,
    average duration, auth failure rate, and hijacking detections.
    Requires admin:read permission.
    """
    monitor = _get_monitor()
    metrics = monitor.get_metrics()

    # Also include suspicious sessions summary
    suspicious = monitor.get_suspicious_sessions()

    return json_response(
        {
            "metrics": metrics.to_dict(),
            "suspicious_sessions": len(suspicious),
            "should_sweep": monitor.should_sweep(),
        }
    )


# =============================================================================
# Sweep Expired Sessions (admin only)
# =============================================================================


@require_permission("admin:write")
@auth_rate_limit(
    requests_per_minute=5,
    limiter_name="auth_session_sweep",
    endpoint_name="session sweep",
)
@handle_errors("session sweep")
async def handle_session_sweep(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Trigger expired session cleanup.

    POST /api/v1/auth/sessions/sweep

    Removes expired and revoked sessions from tracking.
    Requires admin:write permission.
    """
    monitor = _get_monitor()
    removed = monitor.sweep_expired()

    logger.info(
        "Session sweep triggered by user=%s, removed=%d",
        user_id,
        removed,
    )

    return json_response(
        {
            "swept": True,
            "sessions_removed": removed,
            "message": f"Removed {removed} expired/revoked sessions",
        }
    )


# =============================================================================
# Active Sessions for Current User
# =============================================================================


@auth_rate_limit(
    requests_per_minute=30,
    limiter_name="auth_session_active",
    endpoint_name="active sessions",
)
@handle_errors("list active sessions")
async def handle_active_sessions(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    List active sessions for the current user.

    GET /api/v1/auth/sessions/active

    Returns all tracked sessions for the authenticated user.
    """
    if not user_id or user_id == "default":
        return error_response("Authentication required", 401)

    monitor = _get_monitor()
    sessions = monitor.get_sessions_for_user(user_id)

    return json_response(
        {
            "sessions": sessions,
            "total": len(sessions),
            "user_id": user_id,
        }
    )


# =============================================================================
# Handler Registration
# =============================================================================


def get_session_health_handlers() -> dict[str, Any]:
    """Get all session health handlers for registration."""
    return {
        "session_health": handle_session_health,
        "session_sweep": handle_session_sweep,
        "session_active": handle_active_sessions,
    }


__all__ = [
    "handle_session_health",
    "handle_session_sweep",
    "handle_active_sessions",
    "get_session_health_handlers",
]

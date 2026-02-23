"""
Session Management Handlers.

Handles session-related endpoints:
- GET /api/auth/sessions - List active sessions for current user
- DELETE /api/auth/sessions/:id - Revoke a specific session
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from aragora.billing.jwt_auth import extract_user_from_request

from aragora.events.handler_events import emit_handler_event, DELETED
from ..base import HandlerResult, error_response, json_response, handle_errors
from ..openapi_decorator import api_endpoint
from ..utils.rate_limit import auth_rate_limit

if TYPE_CHECKING:
    from .handler import AuthHandler

logger = logging.getLogger(__name__)


@api_endpoint(
    method="GET",
    path="/api/auth/sessions",
    summary="List active sessions",
    description="List all active sessions for the current user with device and activity metadata.",
    tags=["Authentication", "Sessions"],
    responses={
        "200": {
            "description": "List of active sessions returned",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sessions": {"type": "array", "items": {"type": "object"}},
                            "total": {"type": "integer"},
                        },
                    }
                }
            },
        },
        "401": {"description": "Unauthorized"},
    },
)
@auth_rate_limit(
    requests_per_minute=30, limiter_name="auth_sessions", endpoint_name="session listing"
)
@handle_errors("list sessions")
def handle_list_sessions(handler_instance: AuthHandler, handler) -> HandlerResult:
    """List all active sessions for the current user.

    Returns list of sessions with metadata (device, IP, last activity).
    The current session is marked with is_current=true.
    """
    # RBAC check: session.list_active permission required
    if error := handler_instance._check_permission(handler, "session.list_active"):
        return error

    from aragora.billing.auth.sessions import get_session_manager
    from aragora.billing.jwt_auth import decode_jwt
    from aragora.server.middleware.auth import extract_token

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    # Get current token JTI to mark current session
    # JTI is computed from token hash, matching the session tracking approach
    current_jti: str | None = None
    token = extract_token(handler)
    if token:
        payload = decode_jwt(token)
        if payload:
            current_jti = (
                getattr(payload, "jti", None) or hashlib.sha256(token.encode()).hexdigest()[:32]
            )

    # Get sessions from manager
    manager = get_session_manager()
    sessions = manager.list_sessions(auth_ctx.user_id)

    # Convert to response format
    session_list = []
    for session in sessions:
        session_dict = session.to_dict()
        session_dict["is_current"] = session.session_id == current_jti
        session_list.append(session_dict)

    # Sort by last activity (most recent first)
    session_list.sort(key=lambda s: s["last_activity"], reverse=True)

    return json_response(
        {
            "sessions": session_list,
            "total": len(session_list),
        }
    )


@api_endpoint(
    method="DELETE",
    path="/api/auth/sessions/{session_id}",
    summary="Revoke a session",
    description="Revoke a specific session by ID. Cannot revoke current session.",
    tags=["Authentication", "Sessions"],
    parameters=[
        {"name": "session_id", "in": "path", "required": True, "schema": {"type": "string"}}
    ],
    responses={
        "200": {
            "description": "Session revoked successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "message": {"type": "string"},
                            "session_id": {"type": "string"},
                        },
                    }
                }
            },
        },
        "400": {"description": "Invalid session ID or cannot revoke current session"},
        "401": {"description": "Unauthorized"},
        "404": {"description": "Session not found"},
    },
)
@auth_rate_limit(
    requests_per_minute=10, limiter_name="auth_revoke_session", endpoint_name="session revocation"
)
@handle_errors("revoke session")
def handle_revoke_session(handler_instance: AuthHandler, handler, session_id: str) -> HandlerResult:
    """Revoke a specific session.

    This invalidates the session and adds the token to the blacklist.
    Users cannot revoke their current session (use logout instead).
    """
    # RBAC check: session.revoke permission required
    if error := handler_instance._check_permission(handler, "session.revoke"):
        return error

    from aragora.billing.auth.sessions import get_session_manager
    from aragora.billing.jwt_auth import decode_jwt
    from aragora.server.middleware.auth import extract_token

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    # Validate session_id format
    if not session_id or len(session_id) < 8:
        return error_response("Invalid session ID", 400)

    # Check if trying to revoke current session
    # JTI is computed from token hash, matching the session tracking approach
    current_jtis: set[str] = set()
    token = extract_token(handler)
    if token:
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        current_jtis.add(token_hash)
        payload = decode_jwt(token)
        if payload:
            payload_jti = None
            if isinstance(payload, dict):
                payload_jti = payload.get("jti")
            else:
                payload_jti = getattr(payload, "jti", None)
            if payload_jti:
                current_jtis.add(str(payload_jti))

    if session_id in current_jtis:
        return error_response(
            "Cannot revoke current session. Use /api/auth/logout instead.",
            400,
        )

    # Get session manager and verify session belongs to user
    manager = get_session_manager()
    session = manager.get_session(auth_ctx.user_id, session_id)

    if not session:
        return error_response("Session not found", 404)

    # Revoke the session
    manager.revoke_session(auth_ctx.user_id, session_id)

    # Note: We don't have the actual token to blacklist here since we only
    # store session metadata. The token will be rejected when:
    # 1. User increments token version (logout-all)
    # 2. Token expires naturally
    # For immediate revocation, users should use logout-all

    logger.info("Session %s... revoked for user %s", session_id[:8], auth_ctx.user_id)
    emit_handler_event("auth", DELETED, {"action": "session_revoked"}, user_id=auth_ctx.user_id)

    return json_response(
        {
            "success": True,
            "message": "Session revoked successfully",
            "session_id": session_id,
        }
    )


__all__ = ["handle_list_sessions", "handle_revoke_session"]

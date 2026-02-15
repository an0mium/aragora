"""
Session orchestration for OpenClaw Gateway.

Stability: STABLE

Contains:
- Session management handler methods (mixin class)
- Action execution and cancellation handlers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openclaw._base import OpenClawMixinBase, _has_permission
from aragora.server.handlers.openclaw.models import (
    ActionStatus,
    SessionStatus,
)
from aragora.server.handlers.openclaw.store import _get_store
from aragora.server.handlers.openclaw.validation import (
    MAX_SESSION_METADATA_SIZE,
    sanitize_action_parameters,
    validate_action_input,
    validate_action_type,
    validate_metadata,
    validate_session_config,
)
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import (
    auth_rate_limit,
    rate_limit,
)
from aragora.server.validation.query_params import safe_query_int

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Session Orchestration Mixin
# =============================================================================


class SessionOrchestrationMixin(OpenClawMixinBase):
    """Mixin class providing session and action orchestration handler methods.

    This mixin is intended to be used with OpenClawGatewayHandler.
    It requires the following methods from the parent class:
    - _get_user_id(handler) -> str
    - _get_tenant_id(handler) -> str | None
    - get_current_user(handler) -> User | None
    """

    # =========================================================================
    # Session Handlers
    # =========================================================================

    @require_permission("gateway:sessions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_list_sessions")
    def _handle_list_sessions(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List sessions with optional filtering."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            # Parse query parameters
            status_str = query_params.get("status")
            status = SessionStatus(status_str) if status_str else None
            limit = safe_query_int(query_params, "limit", default=50, max_val=500)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            # List sessions (scoped to user/tenant for non-admin)
            sessions, total = store.list_sessions(
                user_id=user_id,
                tenant_id=tenant_id,
                status=status,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "sessions": [s.to_dict() for s in sessions],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except ValueError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Invalid parameter", 400)
        except (KeyError, TypeError, AttributeError, OSError) as e:
            logger.error("Error listing sessions: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_get_session")
    def _handle_get_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Get session by ID."""
        try:
            store = _get_store()
            session = store.get_session(session_id)

            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Check access (user can only see their own sessions unless admin)
            user_id = self._get_user_id(handler)
            user = self.get_current_user(handler)
            is_admin = user and _has_permission(
                user.role if hasattr(user, "role") else None, "gateway:admin"
            )

            if not is_admin and session.user_id != user_id:
                return error_response("Access denied", 403)

            return json_response(session.to_dict())
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error getting session %s: %s", session_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.create")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_create_session")
    def _handle_create_session(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Create a new session."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            config = body.get("config", {})
            metadata = body.get("metadata", {})

            # Validate config
            is_valid, error = validate_session_config(config)
            if not is_valid:
                return error_response(error, 400)

            # Validate metadata
            is_valid, error = validate_metadata(metadata, MAX_SESSION_METADATA_SIZE)
            if not is_valid:
                return error_response(error, 400)

            session = store.create_session(
                user_id=user_id,
                tenant_id=tenant_id,
                config=config,
                metadata=metadata,
            )

            # Audit
            store.add_audit_entry(
                action="session.create",
                actor_id=user_id,
                resource_type="session",
                resource_id=session.id,
                result="success",
            )

            logger.info("Created session %s for user %s", session.id, user_id)
            return json_response(session.to_dict(), status=201)

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error creating session: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.delete")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_close_session")
    def _handle_close_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Close a session."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Verify ownership
            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and _has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Close the session
            store.update_session_status(session_id, SessionStatus.CLOSED)

            # Audit
            store.add_audit_entry(
                action="session.close",
                actor_id=user_id,
                resource_type="session",
                resource_id=session_id,
                result="success",
            )

            logger.info("Closed session %s", session_id)
            return json_response({"closed": True, "session_id": session_id})

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error closing session %s: %s", session_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.delete")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_end_session")
    def _handle_end_session(self, session_id: str, handler: Any) -> HandlerResult:
        """End a session via POST (SDK-compatible endpoint)."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Verify ownership
            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and _has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Close the session
            store.update_session_status(session_id, SessionStatus.CLOSED)

            # Audit
            store.add_audit_entry(
                action="session.end",
                actor_id=user_id,
                resource_type="session",
                resource_id=session_id,
                result="success",
            )

            logger.info("Ended session %s", session_id)
            return json_response({"success": True, "session_id": session_id})

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error ending session %s: %s", session_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # Action Handlers
    # =========================================================================

    @require_permission("gateway:actions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_get_action")
    def _handle_get_action(self, action_id: str, handler: Any) -> HandlerResult:
        """Get action status by ID."""
        try:
            store = _get_store()
            action = store.get_action(action_id)

            if not action:
                return error_response(f"Action not found: {action_id}", 404)

            # Check access via session ownership
            session = store.get_session(action.session_id)
            if session:
                user_id = self._get_user_id(handler)
                user = self.get_current_user(handler)
                is_admin = user and _has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )

                if not is_admin and session.user_id != user_id:
                    return error_response("Access denied", 403)

            return json_response(action.to_dict())
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error getting action %s: %s", action_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:actions.execute")
    @auth_rate_limit(
        requests_per_minute=60,
        limiter_name="openclaw_gateway_execute_action",
        endpoint_name="OpenClaw execute action",
    )
    def _handle_execute_action(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Execute an action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            # Validate required fields
            session_id = body.get("session_id")
            if not session_id:
                return error_response("session_id is required", 400)

            action_type = body.get("action_type")

            # Validate action_type
            is_valid, error = validate_action_type(action_type)
            if not is_valid:
                return error_response(error, 400)

            # Verify session exists and is owned by user
            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and _has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            if session.status != SessionStatus.ACTIVE:
                return error_response(
                    f"Session is not active (status: {session.status.value})", 400
                )

            input_data = body.get("input", {})
            metadata = body.get("metadata", {})

            # Validate input data
            is_valid, error = validate_action_input(input_data)
            if not is_valid:
                return error_response(error, 400)

            # Validate metadata
            is_valid, error = validate_metadata(metadata)
            if not is_valid:
                return error_response(error, 400)

            # Sanitize input data to prevent command injection
            sanitized_input = sanitize_action_parameters(input_data)

            # Create action with sanitized input
            action = store.create_action(
                session_id=session_id,
                action_type=action_type,
                input_data=sanitized_input,
                metadata=metadata,
            )

            # Update session activity
            session.last_activity_at = datetime.now(timezone.utc)

            # Audit
            store.add_audit_entry(
                action="action.execute",
                actor_id=user_id,
                resource_type="action",
                resource_id=action.id,
                result="pending",
                details={"action_type": action_type, "session_id": session_id},
            )

            # In a real implementation, this would dispatch to the OpenClaw runtime
            # For now, we just mark it as running
            store.update_action(action.id, status=ActionStatus.RUNNING)

            logger.info(
                "Created action %s (type: %s) in session %s", action.id, action_type, session_id
            )
            return json_response(action.to_dict(), status=202)

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error executing action: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:actions.cancel")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_cancel_action")
    def _handle_cancel_action(self, action_id: str, handler: Any) -> HandlerResult:
        """Cancel a running action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            action = store.get_action(action_id)
            if not action:
                return error_response(f"Action not found: {action_id}", 404)

            # Verify access
            session = store.get_session(action.session_id)
            if session and session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and _has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Check if cancellable
            if action.status not in (ActionStatus.PENDING, ActionStatus.RUNNING):
                return error_response(
                    f"Action cannot be cancelled (status: {action.status.value})", 400
                )

            # Cancel the action
            store.update_action(action.id, status=ActionStatus.CANCELLED)

            # Audit
            store.add_audit_entry(
                action="action.cancel",
                actor_id=user_id,
                resource_type="action",
                resource_id=action_id,
                result="success",
            )

            logger.info("Cancelled action %s", action_id)
            return json_response({"cancelled": True, "action_id": action_id})

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error cancelling action %s: %s", action_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)


__all__ = [
    "SessionOrchestrationMixin",
]

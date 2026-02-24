"""
Emergency break-glass access endpoint handlers.

Endpoints:
- POST /api/v1/admin/emergency/activate - Activate break-glass access
- POST /api/v1/admin/emergency/deactivate - Deactivate break-glass access
- GET  /api/v1/admin/emergency/status - Check active emergency sessions

All endpoints require admin-level permissions.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    require_permission,
)
from ..secure import SecureHandler
from aragora.server.middleware.mfa import enforce_admin_mfa_policy

logger = logging.getLogger(__name__)


class EmergencyAccessHandler(SecureHandler):
    """Handler for emergency break-glass access management."""

    RESOURCE_TYPE = "emergency_access"

    ROUTES = [
        "/api/v1/admin/emergency/activate",
        "/api/v1/admin/emergency/deactivate",
        "/api/v1/admin/emergency/status",
        # Non-versioned routes (backwards compatibility)
        "/api/admin/emergency/activate",
        "/api/admin/emergency/deactivate",
        "/api/admin/emergency/status",
    ]

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route emergency access requests."""
        from aragora.rbac.decorators import PermissionDeniedError

        try:
            normalized = path.replace("/api/v1/", "/api/").replace("/api/", "/api/v1/")
            if normalized.endswith("/activate") or path.endswith("/activate"):
                return self._activate(handler)
            elif normalized.endswith("/deactivate") or path.endswith("/deactivate"):
                return self._deactivate(handler)
            elif normalized.endswith("/status") or path.endswith("/status"):
                return self._status(handler)
            return None
        except PermissionDeniedError as exc:
            return self.handle_security_error(exc, handler)

    @require_permission("admin:security")
    def _activate(self, handler: Any, user: Any = None) -> HandlerResult:
        """Activate break-glass emergency access.

        POST /api/v1/admin/emergency/activate

        Body:
            user_id: str - User to grant emergency access (required)
            reason: str - Reason for emergency access (required, min 10 chars)
            duration_minutes: int - Duration in minutes (optional, default 60, max 1440)
        """
        # Enforce MFA for admin users (SOC 2 CC5-01)
        if user is not None:
            user_store = self.ctx.get("user_store") if hasattr(self, "ctx") else None
            if user_store:
                mfa_result = enforce_admin_mfa_policy(user, user_store)
                if mfa_result and mfa_result.get("enforced"):
                    return error_response(
                        "Administrative access requires MFA. Please enable MFA at /api/auth/mfa/setup",
                        403,
                    )

        from aragora.rbac.emergency import get_break_glass_access
        from aragora.server.http_utils import run_async

        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid JSON body", 400)

        target_user_id = data.get("user_id", "").strip()
        if not target_user_id:
            return error_response("Missing required field: user_id", 400)

        reason = data.get("reason", "").strip()
        if not reason or len(reason) < 10:
            return error_response("Reason must be at least 10 characters", 400)

        try:
            duration_minutes = int(data.get("duration_minutes", 60))
        except (ValueError, TypeError):
            duration_minutes = 60

        ip_address = None
        if hasattr(handler, "client_address"):
            ip_address = handler.client_address[0] if handler.client_address else None

        user_agent = None
        if hasattr(handler, "headers"):
            user_agent = handler.headers.get("User-Agent")

        emergency = get_break_glass_access()

        try:
            access_id = run_async(
                emergency.activate(
                    user_id=target_user_id,
                    reason=reason,
                    duration_minutes=duration_minutes,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={
                        "activated_by": getattr(user, "user_id", "unknown"),
                    },
                )
            )
        except ValueError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Invalid request", 400)

        record = emergency._all_records.get(access_id)
        logger.warning(
            "Emergency access activated via API: access_id=%s user=%s by=%s",
            access_id,
            target_user_id,
            getattr(user, "user_id", "unknown"),
        )

        return json_response(
            {
                "access_id": access_id,
                "user_id": target_user_id,
                "status": "active",
                "duration_minutes": duration_minutes,
                "expires_at": record.expires_at.isoformat() if record else None,
                "message": "Emergency break-glass access activated",
            }
        )

    @require_permission("admin:security")
    def _deactivate(self, handler: Any, user: Any = None) -> HandlerResult:
        """Deactivate break-glass emergency access.

        POST /api/v1/admin/emergency/deactivate

        Body:
            access_id: str - Emergency access ID to deactivate (required)
        """
        from aragora.rbac.emergency import get_break_glass_access
        from aragora.server.http_utils import run_async

        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid JSON body", 400)

        access_id = data.get("access_id", "").strip()
        if not access_id:
            return error_response("Missing required field: access_id", 400)

        emergency = get_break_glass_access()
        deactivated_by = getattr(user, "user_id", "unknown")

        try:
            record = run_async(
                emergency.deactivate(access_id=access_id, deactivated_by=deactivated_by)
            )
        except ValueError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Resource not found", 404)

        logger.info(
            "Emergency access deactivated via API: access_id=%s by=%s",
            access_id,
            deactivated_by,
        )

        return json_response(
            {
                "access_id": access_id,
                "status": record.status.value,
                "deactivated_by": deactivated_by,
                "actions_taken": len(record.actions_taken),
                "message": "Emergency break-glass access deactivated",
            }
        )

    @require_permission("admin:security")
    def _status(self, handler: Any, user: Any = None) -> HandlerResult:
        """Get emergency access status.

        GET /api/v1/admin/emergency/status

        Returns all active emergency sessions and recent history.
        """
        from aragora.rbac.emergency import get_break_glass_access
        from aragora.server.http_utils import run_async

        emergency = get_break_glass_access()

        # Expire any old records first
        run_async(emergency.expire_old_access())

        active_sessions = [
            record.to_dict() for record in emergency._active_records.values() if record.is_active
        ]

        history = run_async(emergency.get_history(limit=20))
        recent_history = [record.to_dict() for record in history]

        return json_response(
            {
                "active_count": len(active_sessions),
                "active_sessions": active_sessions,
                "recent_history": recent_history,
                "persistence_enabled": emergency._persistence_enabled,
            }
        )

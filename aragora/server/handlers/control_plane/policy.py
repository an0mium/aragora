"""
Policy violation handlers for Control Plane.

Provides REST API endpoints for:
- Listing policy violations
- Getting violation details
- Updating violation status
- Violation statistics
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.decorators import has_permission as _has_permission
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


def _get_has_permission():
    control_plane = sys.modules.get("aragora.server.handlers.control_plane")
    if control_plane is not None:
        candidate = getattr(control_plane, "has_permission", None)
        if callable(candidate):
            return candidate
    return _has_permission


class PolicyHandlerMixin:
    """
    Mixin class providing policy violation handlers.

    This mixin provides methods for:
    - Listing policy violations
    - Getting specific violation details
    - Updating violation status
    - Getting violation statistics
    """

    # These methods are expected from the base class
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]:
        """Require authentication and return user or error."""
        # Cast super() to Any - mixin expects base class to provide this method
        return cast(Any, super()).require_auth_or_error(handler)

    # Attribute declaration - provided by BaseHandler
    ctx: dict[str, Any]

    # =========================================================================
    # Policy Store Helper
    # =========================================================================

    def _get_policy_store(self) -> Any:
        """Get the control plane policy store."""
        try:
            from aragora.control_plane.policy_store import get_control_plane_policy_store

            return get_control_plane_policy_store()
        except ImportError:
            logger.warning("Control plane policy store module not available")
            return None

    # =========================================================================
    # Policy Violation Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/control-plane/policies/violations",
        summary="List policy violations",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:violations.read")
    def _handle_list_policy_violations(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """List control plane policy violations."""
        # Require authentication for violation access
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for policy management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:policies"
        ):
            return error_response("Permission denied", 403)

        store = self._get_policy_store()
        if not store:
            return error_response("Policy store not available", 503)

        try:
            # Parse query parameters
            policy_id = (
                query_params.get("policy_id", [None])[0]
                if isinstance(query_params.get("policy_id"), list)
                else query_params.get("policy_id")
            )
            violation_type = (
                query_params.get("violation_type", [None])[0]
                if isinstance(query_params.get("violation_type"), list)
                else query_params.get("violation_type")
            )
            status = (
                query_params.get("status", [None])[0]
                if isinstance(query_params.get("status"), list)
                else query_params.get("status")
            )
            workspace_id = (
                query_params.get("workspace_id", [None])[0]
                if isinstance(query_params.get("workspace_id"), list)
                else query_params.get("workspace_id")
            )
            limit = int(
                query_params.get("limit", ["100"])[0]
                if isinstance(query_params.get("limit"), list)
                else query_params.get("limit", 100)
            )
            offset = int(
                query_params.get("offset", ["0"])[0]
                if isinstance(query_params.get("offset"), list)
                else query_params.get("offset", 0)
            )

            violations = store.list_violations(
                policy_id=policy_id,
                violation_type=violation_type,
                status=status,
                workspace_id=workspace_id,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "violations": violations,  # Already dicts from _row_to_violation_dict
                    "total": len(violations),
                    "limit": limit,
                    "offset": offset,
                }
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:
            logger.error("Error listing policy violations: %s", e)
            return error_response(safe_error_message(e, "policy"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/policies/violations/{violation_id}",
        summary="Get policy violation details",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:violations.read")
    def _handle_get_policy_violation(self, violation_id: str, handler: Any) -> HandlerResult:
        """Get a specific policy violation."""
        # Require authentication for violation access
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for policy management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:policies"
        ):
            return error_response("Permission denied", 403)

        store = self._get_policy_store()
        if not store:
            return error_response("Policy store not available", 503)

        try:
            # Query for specific violation
            violations = store.list_violations(limit=1000)
            violation = next((v for v in violations if v.get("id") == violation_id), None)

            if not violation:
                return error_response(f"Violation not found: {violation_id}", 404)

            return json_response({"violation": violation})
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:
            logger.error("Error getting policy violation %s: %s", violation_id, e)
            return error_response(safe_error_message(e, "policy"), 500)

    @api_endpoint(
        method="GET",
        path="/api/control-plane/policies/violations/stats",
        summary="Get policy violation statistics",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:violations.read")
    def _handle_get_policy_violation_stats(self, handler: Any) -> HandlerResult:
        """Get policy violation statistics."""
        # Require authentication for stats access
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for policy management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:policies"
        ):
            return error_response("Permission denied", 403)

        store = self._get_policy_store()
        if not store:
            return error_response("Policy store not available", 503)

        try:
            # Get counts by status
            open_counts = store.count_violations(status="open")
            total_counts = store.count_violations()

            # Sum up the counts
            total_open = sum(open_counts.values())
            total_all = sum(total_counts.values())

            return json_response(
                {
                    "total": total_all,
                    "open": total_open,
                    "resolved": total_all - total_open,
                    "by_type": total_counts,
                    "open_by_type": open_counts,
                }
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:
            logger.error("Error getting policy violation stats: %s", e)
            return error_response(safe_error_message(e, "policy"), 500)

    @api_endpoint(
        method="PATCH",
        path="/api/control-plane/policies/violations/{violation_id}",
        summary="Update policy violation status",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:violations.update")
    def _handle_update_policy_violation(
        self, violation_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Update a policy violation status."""
        # Require authentication for violation updates
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for policy management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:policies"
        ):
            return error_response("Permission denied", 403)

        store = self._get_policy_store()
        if not store:
            return error_response("Policy store not available", 503)

        status = body.get("status")
        if not status:
            return error_response("Missing required field: status", 400)

        valid_statuses = ["open", "investigating", "resolved", "false_positive"]
        if status not in valid_statuses:
            return error_response(f"Invalid status: {status}. Valid values: {valid_statuses}", 400)

        try:
            resolved_by = None
            if hasattr(user, "user_id"):
                resolved_by = user.user_id
            elif hasattr(user, "id"):
                resolved_by = user.id

            resolution_notes = body.get("resolution_notes")

            success = store.update_violation_status(
                violation_id=violation_id,
                status=status,
                resolved_by=resolved_by,
                resolution_notes=resolution_notes,
            )

            if not success:
                return error_response(f"Violation not found: {violation_id}", 404)

            return json_response(
                {
                    "updated": True,
                    "violation_id": violation_id,
                    "status": status,
                    "message": f"Violation status updated to {status}",
                }
            )
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as e:
            logger.error("Error updating policy violation %s: %s", violation_id, e)
            return error_response(safe_error_message(e, "policy"), 500)

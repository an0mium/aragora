"""Approval flow HTTP handlers.

All endpoints require authentication. Approve/reject operations
require the 'approvals:manage' permission.
"""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import ApprovalFlow
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.rbac.checker import get_permission_checker

logger = logging.getLogger(__name__)

# RBAC permission key for approval management
# Maps to ResourceType.APPROVAL + Action.GRANT from aragora.rbac.models
APPROVAL_MANAGE_PERMISSION = "approval.grant"

# Global approval flow instance (can be replaced with dependency injection)
_approval_flow: Optional[ApprovalFlow] = None


def get_approval_flow() -> ApprovalFlow:
    """Get or create the global approval flow instance."""
    global _approval_flow
    if _approval_flow is None:
        _approval_flow = ApprovalFlow(auto_approve_low_risk=True)
    return _approval_flow


def set_approval_flow(flow: ApprovalFlow) -> None:
    """Set the global approval flow instance."""
    global _approval_flow
    _approval_flow = flow


class ApprovalHandler:
    """HTTP handlers for approval flow operations."""

    @staticmethod
    async def list_pending(request: web.Request) -> web.Response:
        """
        List all pending approval requests.

        GET /api/autonomous/approvals/pending

        Requires authentication.

        Returns:
            List of pending approval requests
        """
        try:
            # Require authentication
            auth_ctx = await get_auth_context(request, require_auth=True)
            logger.debug(f"list_pending called by user {auth_ctx.user_id}")

            flow = get_approval_flow()
            pending = flow.list_pending()

            return web.json_response(
                {
                    "success": True,
                    "pending": [
                        {
                            "id": req.id,
                            "title": req.title,
                            "description": req.description,
                            "changes": req.changes,
                            "risk_level": req.risk_level,
                            "requested_at": req.requested_at.isoformat(),
                            "requested_by": req.requested_by,
                            "timeout_seconds": req.timeout_seconds,
                            "metadata": req.metadata,
                        }
                        for req in pending
                    ],
                    "count": len(pending),
                }
            )

        except UnauthorizedError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=401,
            )
        except Exception as e:
            logger.error(f"Error listing pending approvals: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_request(request: web.Request) -> web.Response:
        """
        Get a specific approval request.

        GET /api/autonomous/approvals/{request_id}

        Requires authentication.

        Returns:
            Approval request details
        """
        request_id = request.match_info.get("request_id")

        try:
            # Require authentication
            auth_ctx = await get_auth_context(request, require_auth=True)
            logger.debug(f"get_request {request_id} called by user {auth_ctx.user_id}")

            flow = get_approval_flow()
            req = flow._load_request(request_id)

            if not req:
                return web.json_response(
                    {"success": False, "error": "Request not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "request": {
                        "id": req.id,
                        "title": req.title,
                        "description": req.description,
                        "changes": req.changes,
                        "risk_level": req.risk_level,
                        "requested_at": req.requested_at.isoformat(),
                        "requested_by": req.requested_by,
                        "timeout_seconds": req.timeout_seconds,
                        "status": req.status.value,
                        "approved_by": req.approved_by,
                        "approved_at": req.approved_at.isoformat() if req.approved_at else None,
                        "rejection_reason": req.rejection_reason,
                        "metadata": req.metadata,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=401,
            )
        except Exception as e:
            logger.error(f"Error getting approval request: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def approve(request: web.Request) -> web.Response:
        """
        Approve a pending request.

        POST /api/autonomous/approvals/{request_id}/approve

        Requires authentication and 'approvals.manage' permission.

        Body:
            approved_by: str - Who is approving (optional, defaults to auth user)

        Returns:
            Updated approval request
        """
        request_id = request.match_info.get("request_id")

        try:
            # Require authentication
            auth_ctx = await get_auth_context(request, require_auth=True)

            # Check RBAC permission
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, APPROVAL_MANAGE_PERMISSION)
            if not decision.allowed:
                logger.warning(
                    f"User {auth_ctx.user_id} denied approval permission: {decision.reason}"
                )
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data = await request.json()
            # Use authenticated user as approver, or override if specified
            approved_by = data.get("approved_by") or auth_ctx.user_id

            logger.info(f"User {auth_ctx.user_id} approving request {request_id}")

            flow = get_approval_flow()
            req = flow.approve(request_id, approved_by)

            return web.json_response(
                {
                    "success": True,
                    "request": {
                        "id": req.id,
                        "status": req.status.value,
                        "approved_by": req.approved_by,
                        "approved_at": req.approved_at.isoformat() if req.approved_at else None,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=401,
            )
        except ForbiddenError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=403,
            )
        except ValueError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=404,
            )
        except Exception as e:
            logger.error(f"Error approving request: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def reject(request: web.Request) -> web.Response:
        """
        Reject a pending request.

        POST /api/autonomous/approvals/{request_id}/reject

        Requires authentication and 'approvals.manage' permission.

        Body:
            rejected_by: str - Who is rejecting (optional, defaults to auth user)
            reason: str - Reason for rejection

        Returns:
            Updated approval request
        """
        request_id = request.match_info.get("request_id")

        try:
            # Require authentication
            auth_ctx = await get_auth_context(request, require_auth=True)

            # Check RBAC permission
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, APPROVAL_MANAGE_PERMISSION)
            if not decision.allowed:
                logger.warning(
                    f"User {auth_ctx.user_id} denied rejection permission: {decision.reason}"
                )
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data = await request.json()
            # Use authenticated user as rejecter, or override if specified
            rejected_by = data.get("rejected_by") or auth_ctx.user_id
            reason = data.get("reason", "No reason provided")

            logger.info(f"User {auth_ctx.user_id} rejecting request {request_id}: {reason}")

            flow = get_approval_flow()
            req = flow.reject(request_id, rejected_by, reason)

            return web.json_response(
                {
                    "success": True,
                    "request": {
                        "id": req.id,
                        "status": req.status.value,
                        "approved_by": req.approved_by,
                        "rejection_reason": req.rejection_reason,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=401,
            )
        except ForbiddenError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=403,
            )
        except ValueError as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=404,
            )
        except Exception as e:
            logger.error(f"Error rejecting request: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/autonomous") -> None:
        """Register approval routes with the application."""
        app.router.add_get(
            f"{prefix}/approvals/pending",
            ApprovalHandler.list_pending,
        )
        app.router.add_get(
            f"{prefix}/approvals/{{request_id}}",
            ApprovalHandler.get_request,
        )
        app.router.add_post(
            f"{prefix}/approvals/{{request_id}}/approve",
            ApprovalHandler.approve,
        )
        app.router.add_post(
            f"{prefix}/approvals/{{request_id}}/reject",
            ApprovalHandler.reject,
        )

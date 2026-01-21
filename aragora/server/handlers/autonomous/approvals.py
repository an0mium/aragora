"""Approval flow HTTP handlers."""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import ApprovalFlow

logger = logging.getLogger(__name__)

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

        Returns:
            List of pending approval requests
        """
        try:
            flow = get_approval_flow()
            pending = flow.list_pending()

            return web.json_response({
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
            })

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

        Returns:
            Approval request details
        """
        request_id = request.match_info.get("request_id")

        try:
            flow = get_approval_flow()
            req = flow._load_request(request_id)

            if not req:
                return web.json_response(
                    {"success": False, "error": "Request not found"},
                    status=404,
                )

            return web.json_response({
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
            })

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

        Body:
            approved_by: str - Who is approving

        Returns:
            Updated approval request
        """
        request_id = request.match_info.get("request_id")

        try:
            data = await request.json()
            approved_by = data.get("approved_by", "api_user")

            flow = get_approval_flow()
            req = flow.approve(request_id, approved_by)

            return web.json_response({
                "success": True,
                "request": {
                    "id": req.id,
                    "status": req.status.value,
                    "approved_by": req.approved_by,
                    "approved_at": req.approved_at.isoformat() if req.approved_at else None,
                },
            })

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

        Body:
            rejected_by: str - Who is rejecting
            reason: str - Reason for rejection

        Returns:
            Updated approval request
        """
        request_id = request.match_info.get("request_id")

        try:
            data = await request.json()
            rejected_by = data.get("rejected_by", "api_user")
            reason = data.get("reason", "No reason provided")

            flow = get_approval_flow()
            req = flow.reject(request_id, rejected_by, reason)

            return web.json_response({
                "success": True,
                "request": {
                    "id": req.id,
                    "status": req.status.value,
                    "approved_by": req.approved_by,
                    "rejection_reason": req.rejection_reason,
                },
            })

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
    def register_routes(app: web.Application, prefix: str = "/api/autonomous") -> None:
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

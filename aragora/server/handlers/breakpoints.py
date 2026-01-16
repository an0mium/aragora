"""
Breakpoints endpoint handlers for human-in-the-loop intervention.

Endpoints:
- GET /api/breakpoints/pending - List pending breakpoints awaiting resolution
- POST /api/breakpoints/{id}/resolve - Resolve a pending breakpoint
- GET /api/breakpoints/{id}/status - Get status of a specific breakpoint
"""

from __future__ import annotations

__all__ = [
    "BreakpointsHandler",
]

import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

from .base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
    validate_path_segment,
)
from .utils.rate_limit import RateLimiter, get_client_ip

# Rate limiter for breakpoints endpoints (60 requests per minute - debug feature)
_breakpoints_limiter = RateLimiter(requests_per_minute=60)


class BreakpointsHandler(BaseHandler):
    """Handler for breakpoint management endpoints."""

    ROUTES = [
        "/api/breakpoints/pending",
    ]

    # Pattern for breakpoint-specific routes
    BREAKPOINT_PATTERN = re.compile(r"^/api/breakpoints/([a-zA-Z0-9_-]+)/(resolve|status)$")

    def __init__(self, storage=None):
        """Initialize with optional storage backend."""
        super().__init__(storage)
        self._breakpoint_manager = None

    @property
    def breakpoint_manager(self) -> Any:
        """Lazy-load breakpoint manager."""
        if self._breakpoint_manager is None:
            try:
                from aragora.debate.breakpoints import BreakpointManager

                self._breakpoint_manager = BreakpointManager()
            except ImportError:
                pass
        return self._breakpoint_manager

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        return bool(self.BREAKPOINT_PATTERN.match(path))

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route breakpoint requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _breakpoints_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for breakpoints endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/breakpoints/pending":
            return self._get_pending_breakpoints()

        match = self.BREAKPOINT_PATTERN.match(path)
        if match:
            breakpoint_id, action = match.groups()

            # Validate breakpoint ID
            is_valid, err = validate_path_segment(breakpoint_id, "breakpoint_id", SAFE_ID_PATTERN)
            if not is_valid:
                return error_response(err, 400)

            if action == "status":
                return self._get_breakpoint_status(breakpoint_id)
            elif action == "resolve":
                # POST only
                return error_response(
                    "Use POST to resolve breakpoints",
                    405,
                    headers={"Allow": "POST"},
                )

        return None

    def handle_post(self, path: str, body: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests for breakpoint resolution."""
        match = self.BREAKPOINT_PATTERN.match(path)
        if not match:
            return None

        breakpoint_id, action = match.groups()

        # Validate breakpoint ID
        is_valid, err = validate_path_segment(breakpoint_id, "breakpoint_id", SAFE_ID_PATTERN)
        if not is_valid:
            return error_response(err, 400)

        if action == "resolve":
            return self._resolve_breakpoint(breakpoint_id, body)

        return None

    def _get_pending_breakpoints(self) -> HandlerResult:
        """Get all pending breakpoints awaiting human resolution.

        Returns:
            List of pending breakpoints with their triggers and snapshots
        """
        if not self.breakpoint_manager:
            return error_response("Breakpoints module not available", 503)

        try:
            pending = self.breakpoint_manager.get_pending_breakpoints()

            return json_response(
                {
                    "breakpoints": [
                        {
                            "breakpoint_id": bp.breakpoint_id,
                            "trigger": bp.trigger.value,
                            "message": bp.message,
                            "created_at": bp.created_at,
                            "timeout_minutes": bp.timeout_minutes,
                            "snapshot": (
                                {
                                    "debate_id": bp.snapshot.debate_id,
                                    "round_num": bp.snapshot.round_num,
                                    "task": bp.snapshot.task,
                                    "confidence": bp.snapshot.current_confidence,
                                    "agents": bp.snapshot.agent_names,
                                }
                                if bp.snapshot
                                else None
                            ),
                        }
                        for bp in pending
                    ],
                    "count": len(pending),
                }
            )

        except Exception as e:
            logger.exception("Failed to get pending breakpoints: %s", e)
            return error_response(safe_error_message(e, "get pending breakpoints"), 500)

    def _get_breakpoint_status(self, breakpoint_id: str) -> HandlerResult:
        """Get status of a specific breakpoint.

        Args:
            breakpoint_id: ID of the breakpoint to check

        Returns:
            Breakpoint status and details
        """
        if not self.breakpoint_manager:
            return error_response("Breakpoints module not available", 503)

        try:
            bp = self.breakpoint_manager.get_breakpoint(breakpoint_id)

            if not bp:
                return json_response(
                    {"error": "Breakpoint not found", "breakpoint_id": breakpoint_id},
                    status=404,
                )

            return json_response(
                {
                    "breakpoint_id": bp.breakpoint_id,
                    "trigger": bp.trigger.value,
                    "message": bp.message,
                    "status": bp.status if hasattr(bp, "status") else "pending",
                    "created_at": bp.created_at,
                    "resolved_at": bp.resolved_at if hasattr(bp, "resolved_at") else None,
                    "snapshot": (
                        {
                            "debate_id": bp.snapshot.debate_id,
                            "round_num": bp.snapshot.round_num,
                            "task": bp.snapshot.task,
                            "confidence": bp.snapshot.current_confidence,
                        }
                        if bp.snapshot
                        else None
                    ),
                }
            )

        except Exception as e:
            logger.exception("Failed to get breakpoint status: %s", e)
            return error_response(safe_error_message(e, "get breakpoint status"), 500)

    def _resolve_breakpoint(self, breakpoint_id: str, body: dict[str, Any]) -> HandlerResult:
        """Resolve a pending breakpoint with human guidance.

        Args:
            breakpoint_id: ID of the breakpoint to resolve
            body: Request body with resolution details:
                - action: "continue" | "abort" | "redirect" | "inject"
                - message: Human guidance message
                - redirect_task: New task if redirecting (optional)

        Returns:
            Resolution confirmation
        """
        if not self.breakpoint_manager:
            return error_response("Breakpoints module not available", 503)

        # Validate required fields
        action = body.get("action")
        if not action:
            return error_response("Missing required field: action", 400)

        valid_actions = ["continue", "abort", "redirect", "inject"]
        if action not in valid_actions:
            return error_response(f"Invalid action: {action}. Must be one of: {valid_actions}", 400)

        message = body.get("message", "")
        redirect_task = body.get("redirect_task")

        try:
            import uuid

            from aragora.debate.breakpoints import HumanGuidance

            guidance = HumanGuidance(
                guidance_id=str(uuid.uuid4()),
                debate_id=breakpoint_id.split("_")[0] if "_" in breakpoint_id else "",
                human_id=body.get("reviewer_id", "api_user"),
                action=action,
                reasoning=message,
                preferred_direction=redirect_task,
            )

            # Resolve the breakpoint
            success = self.breakpoint_manager.resolve_breakpoint(breakpoint_id, guidance)

            if not success:
                return json_response(
                    {
                        "error": "Failed to resolve breakpoint",
                        "breakpoint_id": breakpoint_id,
                        "message": "Breakpoint may not exist or already resolved",
                    },
                    status=404,
                )

            return json_response(
                {
                    "breakpoint_id": breakpoint_id,
                    "status": "resolved",
                    "action": action,
                    "message": message,
                }
            )

        except ImportError:
            return error_response("Breakpoints module not available", 503)
        except Exception as e:
            logger.exception("Failed to resolve breakpoint: %s", e)
            return error_response(safe_error_message(e, "resolve breakpoint"), 500)

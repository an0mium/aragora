"""
Workspace Activity Feed - Recent workspace events for the admin dashboard.

Provides a handler mixin for fetching recent workspace activity events
(member joins, debates created, settings changes, etc.) with pagination.

Stability: STABLE
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import handle_errors
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.protocols import HTTPRequestHandler
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)


class WorkspaceActivityMixin:
    """Mixin providing workspace activity feed handler methods.

    Expects the host class to provide:
    - _get_audit_log()
    - _run_async(coro)
    - read_json_body(handler)
    """

    @api_endpoint(
        method="GET",
        path="/api/v1/workspaces/{workspace_id}/activity",
        summary="Get recent workspace activity",
        tags=["Workspace"],
        response_model={
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "actor": {"type": "string"},
                            "description": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
            },
        },
    )
    @handle_errors
    @require_permission("workspaces:read")
    @rate_limit(requests_per_minute=60)
    def handle_workspace_activity(
        self: Any, handler: HTTPRequestHandler
    ) -> HandlerResult:
        """Return recent activity events for a workspace."""
        from aragora.server.handlers.utils.path_utils import extract_path_param

        path = handler.path
        workspace_id = extract_path_param(path, "workspaces")

        # Parse query params
        query = _parse_query(path)
        limit = min(int(query.get("limit", "50")), 200)
        offset = int(query.get("offset", "0"))

        try:
            audit_log = self._get_audit_log()
        except (AttributeError, ImportError):
            audit_log = None

        events: list[dict[str, Any]] = []

        if audit_log is not None:
            try:
                raw_events = audit_log.query(
                    workspace_id=workspace_id,
                    limit=limit,
                    offset=offset,
                )
                for evt in raw_events:
                    events.append({
                        "id": getattr(evt, "id", str(hash(str(evt)))),
                        "type": getattr(evt, "event_type", getattr(evt, "type", "unknown")),
                        "actor": getattr(evt, "actor", getattr(evt, "user_id", "system")),
                        "description": _describe_event(evt),
                        "timestamp": str(getattr(evt, "timestamp", getattr(evt, "created_at", ""))),
                        "metadata": getattr(evt, "metadata", {}),
                    })
            except (TypeError, ValueError, AttributeError, OSError) as exc:
                logger.debug("Activity feed query failed: %s", exc)

        return {
            "status": 200,
            "body": {"events": events},
        }


def _describe_event(evt: Any) -> str:
    """Generate a human-readable description from an audit event."""
    event_type = getattr(evt, "event_type", getattr(evt, "type", ""))
    actor = getattr(evt, "actor", getattr(evt, "user_id", "Someone"))
    target = getattr(evt, "target", getattr(evt, "resource", ""))

    descriptions = {
        "member_joined": f"{actor} joined the workspace",
        "member_removed": f"{actor} was removed from the workspace",
        "member_invited": f"{actor} invited {target} to the workspace",
        "role_changed": f"{actor}'s role was changed",
        "debate_created": f"{actor} started a new debate",
        "debate_completed": f"Debate completed",
        "settings_updated": f"{actor} updated workspace settings",
        "invite_sent": f"Invite sent to {target}",
        "invite_revoked": f"Invite to {target} was revoked",
        "policy_updated": f"{actor} updated retention policies",
    }

    return descriptions.get(event_type, getattr(evt, "description", f"{actor} performed {event_type}"))


def _parse_query(path: str) -> dict[str, str]:
    """Extract query parameters from URL path."""
    if "?" not in path:
        return {}
    query_string = path.split("?", 1)[1]
    params: dict[str, str] = {}
    for pair in query_string.split("&"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            params[key] = value
    return params

"""Federation status and management endpoints.

Endpoints:
- GET  /api/v1/federation/status       - Get federation overview status
- GET  /api/v1/federation/workspaces   - List connected workspaces with health
- GET  /api/v1/federation/activity     - Get recent sync activity feed
- GET  /api/v1/federation/config       - Get federation configuration

These endpoints expose the CrossWorkspaceCoordinator through a REST API,
enabling the admin UI to display federation status, connected workspaces,
sync activity, and knowledge sharing configuration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.utils.decorators import (
    require_permission,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)


def _safe_import_coordinator() -> Any | None:
    """Safely import the cross-workspace coordinator."""
    try:
        from aragora.coordination.cross_workspace import get_coordinator

        return get_coordinator()
    except (ImportError, RuntimeError) as e:
        logger.debug("Coordination module not available: %s", e)
        return None


class FederationStatusHandler(BaseHandler):
    """Handler for federation status and management.

    Provides read-only endpoints for monitoring cross-workspace
    federation state, connected workspaces, sync activity,
    and knowledge sharing configuration.

    RBAC Permissions:
    - federation:read - View federation status and configuration
    """

    ROUTES = [
        "/api/federation/status",
        "/api/federation/workspaces",
        "/api/federation/activity",
        "/api/federation/config",
    ]

    def handle_request(
        self, method: str, path: str, handler: Any, query_params: dict[str, Any]
    ) -> HandlerResult | None:
        """Route federation requests to the appropriate handler method."""
        normalized = strip_version_prefix(path)

        if method == "GET":
            if normalized == "/api/federation/status":
                return self._handle_federation_status(query_params)
            if normalized == "/api/federation/workspaces":
                return self._handle_list_workspaces(query_params)
            if normalized == "/api/federation/activity":
                return self._handle_sync_activity(query_params)
            if normalized == "/api/federation/config":
                return self._handle_federation_config(query_params)

        return None

    # =========================================================================
    # GET /api/v1/federation/status
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/federation/status",
        summary="Get federation overview status",
        tags=["Federation"],
    )
    @require_permission("federation:read")
    def _handle_federation_status(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return a high-level federation status overview.

        Response includes connected workspace count, shared knowledge
        totals, overall health, and active policy summary.
        """
        coordinator = _safe_import_coordinator()
        if not coordinator:
            return json_response(
                {
                    "status": "unavailable",
                    "connected_workspaces": 0,
                    "shared_knowledge_count": 0,
                    "sync_health": "offline",
                    "federation_mode": "isolated",
                    "message": "Federation service not initialized",
                }
            )

        try:
            stats = coordinator.get_stats()
            workspaces = coordinator.list_workspaces()
            consents = coordinator.list_consents()

            online_count = sum(1 for w in workspaces if w.is_online)
            total_count = len(workspaces)

            # Determine overall health
            if total_count == 0:
                sync_health = "idle"
            elif online_count == total_count:
                sync_health = "healthy"
            elif online_count > 0:
                sync_health = "degraded"
            else:
                sync_health = "offline"

            # Compute shared knowledge count from consent usage
            shared_knowledge_count = sum(c.times_used for c in consents if c.is_valid())

            # Get the default federation mode
            default_policy = getattr(coordinator, "_default_policy", None)
            federation_mode = default_policy.mode.value if default_policy else "isolated"

            return json_response(
                {
                    "status": "active" if total_count > 0 else "idle",
                    "connected_workspaces": total_count,
                    "online_workspaces": online_count,
                    "shared_knowledge_count": shared_knowledge_count,
                    "valid_consents": stats.get("valid_consents", 0),
                    "pending_requests": stats.get("pending_requests", 0),
                    "sync_health": sync_health,
                    "federation_mode": federation_mode,
                    "registered_handlers": stats.get("registered_handlers", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error fetching federation status: %s", e)
            return error_response(safe_error_message(e, "federation"), 500)

    # =========================================================================
    # GET /api/v1/federation/workspaces
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/federation/workspaces",
        summary="List connected workspaces with health info",
        tags=["Federation"],
    )
    @require_permission("federation:read")
    def _handle_list_workspaces(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return all federated workspaces with status and sync metadata."""
        coordinator = _safe_import_coordinator()
        if not coordinator:
            return json_response({"workspaces": [], "total": 0})

        try:
            workspaces = coordinator.list_workspaces()
            consents = coordinator.list_consents()

            result = []
            for ws in workspaces:
                # Count shared items via consents involving this workspace
                ws_consents = [
                    c
                    for c in consents
                    if (c.source_workspace_id == ws.id or c.target_workspace_id == ws.id)
                    and c.is_valid()
                ]
                shared_items = sum(c.times_used for c in ws_consents)

                # Determine status color indicator
                if ws.is_online:
                    status = "connected"
                elif ws.last_heartbeat:
                    status = "stale"
                else:
                    status = "disconnected"

                result.append(
                    {
                        "id": ws.id,
                        "name": ws.name or ws.id,
                        "org_id": ws.org_id,
                        "status": status,
                        "is_online": ws.is_online,
                        "federation_mode": ws.federation_mode.value,
                        "last_heartbeat": (
                            ws.last_heartbeat.isoformat() if ws.last_heartbeat else None
                        ),
                        "latency_ms": ws.latency_ms,
                        "shared_items": shared_items,
                        "active_consents": len(ws_consents),
                        "capabilities": {
                            "agent_execution": ws.supports_agent_execution,
                            "workflow_execution": ws.supports_workflow_execution,
                            "knowledge_query": ws.supports_knowledge_query,
                        },
                    }
                )

            return json_response(
                {
                    "workspaces": result,
                    "total": len(result),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error listing federation workspaces: %s", e)
            return error_response(safe_error_message(e, "federation"), 500)

    # =========================================================================
    # GET /api/v1/federation/activity
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/federation/activity",
        summary="Get recent sync activity feed",
        tags=["Federation"],
    )
    @require_permission("federation:read")
    def _handle_sync_activity(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return recent cross-workspace sync events.

        Aggregates consent usage and pending requests into a
        chronological activity feed for the admin dashboard.
        """
        coordinator = _safe_import_coordinator()
        if not coordinator:
            return json_response({"activity": [], "total": 0})

        try:
            consents = coordinator.list_consents()
            pending = coordinator.list_pending_requests()
            activity: list[dict[str, Any]] = []

            # Add consent-based activity (recent grants, revocations)
            for consent in consents:
                event: dict[str, Any] = {
                    "id": consent.id,
                    "type": "consent_revoked" if consent.revoked else "consent_active",
                    "source_workspace": consent.source_workspace_id,
                    "target_workspace": consent.target_workspace_id,
                    "scope": consent.scope.value,
                    "data_types": list(consent.data_types),
                    "times_used": consent.times_used,
                    "timestamp": (
                        consent.revoked_at.isoformat()
                        if consent.revoked and consent.revoked_at
                        else consent.granted_at.isoformat()
                    ),
                }
                if consent.last_used:
                    event["last_sync"] = consent.last_used.isoformat()
                activity.append(event)

            # Add pending request activity
            for req in pending:
                activity.append(
                    {
                        "id": req.id,
                        "type": "pending_approval",
                        "operation": req.operation.value,
                        "source_workspace": req.source_workspace_id,
                        "target_workspace": req.target_workspace_id,
                        "requester": req.requester_id,
                        "timestamp": req.created_at.isoformat(),
                    }
                )

            # Sort by timestamp descending
            activity.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

            return json_response(
                {
                    "activity": activity,
                    "total": len(activity),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error fetching sync activity: %s", e)
            return error_response(safe_error_message(e, "federation"), 500)

    # =========================================================================
    # GET /api/v1/federation/config
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/federation/config",
        summary="Get federation configuration",
        tags=["Federation"],
    )
    @require_permission("federation:read")
    def _handle_federation_config(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return the current federation configuration.

        Includes default policy, knowledge sharing settings,
        and approval requirements.
        """
        coordinator = _safe_import_coordinator()
        if not coordinator:
            return json_response(
                {
                    "default_policy": None,
                    "knowledge_sharing": {
                        "types": [],
                        "approval_required": True,
                        "scope": "none",
                    },
                }
            )

        try:
            default_policy = getattr(coordinator, "_default_policy", None)
            workspace_policies = getattr(coordinator, "_workspace_policies", {})

            policy_dict = default_policy.to_dict() if default_policy else None

            # Summarize knowledge sharing configuration
            sharing_types = set()
            for consent in coordinator.list_consents():
                if consent.is_valid():
                    sharing_types.update(consent.data_types)

            knowledge_sharing = {
                "types": sorted(sharing_types),
                "approval_required": (default_policy.require_approval if default_policy else True),
                "scope": (default_policy.sharing_scope.value if default_policy else "none"),
                "audit_enabled": (default_policy.audit_all_requests if default_policy else True),
            }

            return json_response(
                {
                    "default_policy": policy_dict,
                    "workspace_policy_count": len(workspace_policies),
                    "knowledge_sharing": knowledge_sharing,
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error fetching federation config: %s", e)
            return error_response(safe_error_message(e, "federation"), 500)


__all__ = ["FederationStatusHandler"]

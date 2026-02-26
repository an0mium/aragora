"""
Coordination handlers for Cross-Workspace Federation.

Provides REST API endpoints for:
- Workspace registration and management
- Federation policy configuration
- Cross-workspace execution
- Data sharing consent management
- Coordination health and statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aragora.coordination.fleet import create_fleet_coordinator
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.decorators import (
    handle_errors,
    require_permission,
)

logger = logging.getLogger(__name__)


def _parse_tail(query_params: dict[str, Any], default: int = 200) -> int:
    raw = query_params.get("tail", default)
    try:
        tail = int(raw)
    except (TypeError, ValueError):
        tail = default
    if tail < 1:
        return 1
    return min(tail, 5000)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class CoordinationHandlerMixin:
    """
    Mixin class providing cross-workspace coordination handlers.

    Provides methods for:
    - Workspace registration, listing, and unregistration
    - Federation policy creation and listing
    - Cross-workspace execution and approval
    - Data sharing consent management
    - Coordination stats and health
    """

    # Attribute declarations - provided by BaseHandler / ControlPlaneHandler
    ctx: dict[str, Any]

    def _get_coordination_coordinator(self) -> Any | None:
        """Get the cross-workspace coordinator."""
        return self.ctx.get("coordination_coordinator")

    def _require_coordination_coordinator(self) -> tuple[Any | None, HandlerResult | None]:
        """Return coordinator and None, or None and error response if not available."""
        coord = self._get_coordination_coordinator()
        if not coord:
            return None, error_response("Coordination service not initialized", 503)
        return coord, None

    # =========================================================================
    # Workspace Management
    # =========================================================================

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/workspaces",
        summary="Register a workspace for federation",
        tags=["Coordination"],
    )
    @handle_errors("coordination workspace registration")
    @require_permission("coordination:workspaces.write")
    def _handle_register_workspace(self, body: dict[str, Any]) -> HandlerResult:
        """Register a workspace for federation."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        workspace_id = body.get("id")
        if not workspace_id:
            return error_response("Workspace id is required", 400)

        name = body.get("name", "")
        org_id = body.get("org_id", "")
        federation_mode = body.get("federation_mode", "readonly")
        endpoint_url = body.get("endpoint_url")

        try:
            from aragora.coordination.cross_workspace import (
                FederatedWorkspace,
                FederationMode,
            )

            mode = FederationMode(federation_mode)
            workspace = FederatedWorkspace(
                id=workspace_id,
                name=name,
                org_id=org_id,
                federation_mode=mode,
                endpoint_url=endpoint_url,
                supports_agent_execution=body.get("supports_agent_execution", True),
                supports_workflow_execution=body.get("supports_workflow_execution", True),
                supports_knowledge_query=body.get("supports_knowledge_query", True),
            )
            coordinator.register_workspace(workspace)
            return json_response(workspace.to_dict(), status=201)
        except (ValueError, KeyError) as e:
            logger.warning("Invalid workspace registration data: %s", e)
            return error_response(safe_error_message(e, "coordination"), 400)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/workspaces",
        summary="List registered workspaces",
        tags=["Coordination"],
    )
    @require_permission("coordination:workspaces.read")
    def _handle_list_workspaces(self, query_params: dict[str, Any]) -> HandlerResult:
        """List all registered workspaces."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            workspaces = coordinator.list_workspaces()
            return json_response(
                {
                    "workspaces": [w.to_dict() for w in workspaces],
                    "total": len(workspaces),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error listing workspaces: %s", e)
            return error_response(safe_error_message(e, "coordination"), 500)

    @api_endpoint(
        method="DELETE",
        path="/api/v1/coordination/workspaces/{workspace_id}",
        summary="Unregister a workspace",
        tags=["Coordination"],
    )
    @handle_errors("coordination workspace unregistration")
    @require_permission("coordination:workspaces.write")
    def _handle_unregister_workspace(self, workspace_id: str) -> HandlerResult:
        """Unregister a workspace from federation."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            coordinator.unregister_workspace(workspace_id)
            return json_response({"unregistered": True})
        except (ValueError, KeyError) as e:
            logger.warning("Error unregistering workspace %s: %s", workspace_id, e)
            return error_response(safe_error_message(e, "coordination"), 400)

    # =========================================================================
    # Federation Policies
    # =========================================================================

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/federation",
        summary="Create a federation policy",
        tags=["Coordination"],
    )
    @handle_errors("coordination federation policy creation")
    @require_permission("coordination:federation.write")
    def _handle_create_federation_policy(self, body: dict[str, Any]) -> HandlerResult:
        """Create a federation policy."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        name = body.get("name", "")
        if not name:
            return error_response("Policy name is required", 400)

        try:
            from aragora.coordination.cross_workspace import (
                FederationMode,
                FederationPolicy,
                OperationType,
                SharingScope,
            )

            mode = FederationMode(body.get("mode", "isolated"))
            sharing_scope = SharingScope(body.get("sharing_scope", "none"))

            allowed_ops = set()
            for op_str in body.get("allowed_operations", []):
                allowed_ops.add(OperationType(op_str))

            policy = FederationPolicy(
                name=name,
                description=body.get("description", ""),
                mode=mode,
                sharing_scope=sharing_scope,
                allowed_operations=allowed_ops,
                max_requests_per_hour=body.get("max_requests_per_hour", 100),
                require_approval=body.get("require_approval", False),
                audit_all_requests=body.get("audit_all_requests", True),
            )

            workspace_id = body.get("workspace_id")
            source_workspace_id = body.get("source_workspace_id")
            target_workspace_id = body.get("target_workspace_id")

            coordinator.set_policy(
                policy,
                workspace_id=workspace_id,
                source_workspace_id=source_workspace_id,
                target_workspace_id=target_workspace_id,
            )

            return json_response(policy.to_dict(), status=201)
        except (ValueError, KeyError) as e:
            logger.warning("Invalid federation policy data: %s", e)
            return error_response(safe_error_message(e, "coordination"), 400)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/federation",
        summary="List federation policies",
        tags=["Coordination"],
    )
    @require_permission("coordination:federation.read")
    def _handle_list_federation_policies(self, query_params: dict[str, Any]) -> HandlerResult:
        """List federation policies."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            # Collect all policies (default + workspace-specific + pair-specific)
            policies: list[dict[str, Any]] = []

            # Default policy
            if hasattr(coordinator, "_default_policy"):
                default = coordinator._default_policy
                entry = default.to_dict()
                entry["scope"] = "default"
                policies.append(entry)

            # Workspace policies
            if hasattr(coordinator, "_workspace_policies"):
                for ws_id, policy in coordinator._workspace_policies.items():
                    entry = policy.to_dict()
                    entry["scope"] = "workspace"
                    entry["workspace_id"] = ws_id
                    policies.append(entry)

            # Pair policies
            if hasattr(coordinator, "_pair_policies"):
                for (src, tgt), policy in coordinator._pair_policies.items():
                    entry = policy.to_dict()
                    entry["scope"] = "pair"
                    entry["source_workspace_id"] = src
                    entry["target_workspace_id"] = tgt
                    policies.append(entry)

            return json_response(
                {
                    "policies": policies,
                    "total": len(policies),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error listing federation policies: %s", e)
            return error_response(safe_error_message(e, "coordination"), 500)

    # =========================================================================
    # Cross-Workspace Execution
    # =========================================================================

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/execute",
        summary="Execute a cross-workspace operation",
        tags=["Coordination"],
    )
    @handle_errors("coordination cross-workspace execution")
    @require_permission("coordination:execute.write")
    def _handle_execute(self, body: dict[str, Any]) -> HandlerResult:
        """Execute a cross-workspace operation."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        operation = body.get("operation")
        source_workspace_id = body.get("source_workspace_id")
        target_workspace_id = body.get("target_workspace_id")

        if not operation or not source_workspace_id or not target_workspace_id:
            return error_response(
                "operation, source_workspace_id, and target_workspace_id are required", 400
            )

        try:
            from aragora.coordination.cross_workspace import (
                CrossWorkspaceRequest,
                OperationType,
            )
            from aragora.server.http_utils import run_async

            op = OperationType(operation)
            request = CrossWorkspaceRequest(
                operation=op,
                source_workspace_id=source_workspace_id,
                target_workspace_id=target_workspace_id,
                payload=body.get("payload", {}),
                timeout_seconds=body.get("timeout_seconds", 30.0),
                requester_id=body.get("requester_id", ""),
                consent_id=body.get("consent_id"),
            )

            result = run_async(coordinator.execute(request))
            return json_response(result.to_dict(), status=200 if result.success else 422)
        except (ValueError, KeyError) as e:
            logger.warning("Invalid execution request: %s", e)
            return error_response(safe_error_message(e, "coordination"), 400)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/executions",
        summary="List pending executions",
        tags=["Coordination"],
    )
    @require_permission("coordination:execute.read")
    def _handle_list_executions(self, query_params: dict[str, Any]) -> HandlerResult:
        """List pending cross-workspace execution requests."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            workspace_id = query_params.get("workspace_id")
            requests = coordinator.list_pending_requests(workspace_id=workspace_id)
            return json_response(
                {
                    "executions": [r.to_dict() for r in requests],
                    "total": len(requests),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error listing executions: %s", e)
            return error_response(safe_error_message(e, "coordination"), 500)

    # =========================================================================
    # Consent Management
    # =========================================================================

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/consent",
        summary="Grant data sharing consent",
        tags=["Coordination"],
    )
    @handle_errors("coordination consent grant")
    @require_permission("coordination:consent.write")
    def _handle_grant_consent(self, body: dict[str, Any]) -> HandlerResult:
        """Grant data sharing consent between workspaces."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        source_workspace_id = body.get("source_workspace_id")
        target_workspace_id = body.get("target_workspace_id")
        if not source_workspace_id or not target_workspace_id:
            return error_response("source_workspace_id and target_workspace_id are required", 400)

        try:
            from aragora.coordination.cross_workspace import (
                OperationType,
                SharingScope,
            )

            scope = SharingScope(body.get("scope", "metadata"))
            data_types = set(body.get("data_types", []))
            operations = {OperationType(op) for op in body.get("operations", [])}
            granted_by = body.get("granted_by", "")
            expires_in_days = body.get("expires_in_days")

            consent = coordinator.grant_consent(
                source_workspace_id=source_workspace_id,
                target_workspace_id=target_workspace_id,
                scope=scope,
                data_types=data_types,
                operations=operations,
                granted_by=granted_by,
                expires_in_days=expires_in_days,
            )

            return json_response(consent.to_dict(), status=201)
        except (ValueError, KeyError) as e:
            logger.warning("Invalid consent data: %s", e)
            return error_response(safe_error_message(e, "coordination"), 400)

    @api_endpoint(
        method="DELETE",
        path="/api/v1/coordination/consent/{consent_id}",
        summary="Revoke data sharing consent",
        tags=["Coordination"],
    )
    @handle_errors("coordination consent revocation")
    @require_permission("coordination:consent.write")
    def _handle_revoke_consent(self, consent_id: str, body: dict[str, Any]) -> HandlerResult:
        """Revoke a data sharing consent."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        revoked_by = body.get("revoked_by", "api")

        success = coordinator.revoke_consent(consent_id, revoked_by)
        if not success:
            return error_response(f"Consent not found: {consent_id}", 404)

        return json_response({"revoked": True})

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/consent",
        summary="List data sharing consents",
        tags=["Coordination"],
    )
    @require_permission("coordination:consent.read")
    def _handle_list_consents(self, query_params: dict[str, Any]) -> HandlerResult:
        """List data sharing consents."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            workspace_id = query_params.get("workspace_id")
            consents = coordinator.list_consents(workspace_id=workspace_id)
            return json_response(
                {
                    "consents": [c.to_dict() for c in consents],
                    "total": len(consents),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error listing consents: %s", e)
            return error_response(safe_error_message(e, "coordination"), 500)

    # =========================================================================
    # Approval
    # =========================================================================

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/approve/{request_id}",
        summary="Approve a pending execution request",
        tags=["Coordination"],
    )
    @handle_errors("coordination request approval")
    @require_permission("coordination:execute.write")
    def _handle_approve_request(self, request_id: str, body: dict[str, Any]) -> HandlerResult:
        """Approve a pending cross-workspace execution request."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        approved_by = body.get("approved_by", "api")

        success = coordinator.approve_request(request_id, approved_by)
        if not success:
            return error_response(f"Request not found or not pending: {request_id}", 404)

        return json_response({"approved": True})

    # =========================================================================
    # Fleet Monitor
    # =========================================================================

    def _fleet(self):
        repo_root = self.ctx.get("repo_root")
        if isinstance(repo_root, Path):
            return create_fleet_coordinator(repo_root=repo_root)
        return create_fleet_coordinator(repo_root=Path.cwd())

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/fleet/status",
        summary="Get fleet status for all active sessions",
        tags=["Coordination"],
    )
    @require_permission("coordination:stats.read")
    def _handle_fleet_status(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get fleet monitor status."""
        fleet = self._fleet()
        tail = _parse_tail(query_params, default=200)
        status = fleet.fleet_status(tail_lines=tail)

        if _parse_bool(query_params.get("write_report")):
            report_dir = query_params.get("report_dir")
            output_dir = None
            if isinstance(report_dir, str) and report_dir.strip():
                output_dir = Path(report_dir)
                if not output_dir.is_absolute():
                    output_dir = (fleet.repo_root / output_dir).resolve()
            report = fleet.write_report(status=status, output_dir=output_dir)
            status["report_path"] = report["report_path"]

        return json_response(status)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/fleet/logs",
        summary="Get tailed logs for fleet sessions",
        tags=["Coordination"],
    )
    @require_permission("coordination:stats.read")
    def _handle_fleet_logs(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get tailed logs across active sessions."""
        tail = _parse_tail(query_params, default=200)
        session_id = query_params.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        payload = self._fleet().fleet_logs(tail_lines=tail, session_id=session_id)
        return json_response(payload)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/fleet/claims",
        summary="Get path claims and conflicts",
        tags=["Coordination"],
    )
    @require_permission("coordination:stats.read")
    def _handle_fleet_claims(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get current path claims and conflicts."""
        _ = query_params
        return json_response(self._fleet().get_claims())

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/fleet/claims",
        summary="Claim file paths for a session/owner",
        tags=["Coordination"],
    )
    @handle_errors("fleet path claim")
    @require_permission("coordination:execute.write")
    def _handle_fleet_claim_paths(self, body: dict[str, Any]) -> HandlerResult:
        """Claim paths for an owner with optional conflict override."""
        owner = str(body.get("owner", "")).strip()
        paths = body.get("paths", [])
        if not isinstance(paths, list):
            return error_response("'paths' must be a list", 400)
        normalized_paths = [str(path) for path in paths if isinstance(path, str)]
        override = bool(body.get("override", False))
        result = self._fleet().claim_paths(owner, normalized_paths, override=override)
        status = 200 if result.get("ok") else 409
        if not result.get("ok") and result.get("error"):
            status = 400
        return json_response(result, status=status)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/fleet/merge-queue",
        summary="Get merge queue state with blockers",
        tags=["Coordination"],
    )
    @require_permission("coordination:stats.read")
    def _handle_fleet_merge_queue(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get merge queue state and live blockers."""
        _ = query_params
        return json_response(self._fleet().get_merge_queue())

    @api_endpoint(
        method="POST",
        path="/api/v1/coordination/fleet/merge-queue",
        summary="Update merge queue (enqueue/advance/remove/clear)",
        tags=["Coordination"],
    )
    @handle_errors("fleet merge queue operation")
    @require_permission("coordination:execute.write")
    def _handle_fleet_merge_queue_action(self, body: dict[str, Any]) -> HandlerResult:
        """Perform merge queue actions."""
        action = str(body.get("action", "enqueue")).strip().lower()
        fleet = self._fleet()

        if action == "enqueue":
            owner = str(body.get("owner", "")).strip()
            branch = str(body.get("branch", "")).strip()
            session_id = body.get("session_id")
            if session_id is not None:
                session_id = str(session_id)
            pr_number = body.get("pr_number")
            if pr_number is not None:
                try:
                    pr_number = int(pr_number)
                except (TypeError, ValueError):
                    return error_response("'pr_number' must be an integer", 400)
            override = bool(body.get("override_claim_conflicts", False))
            result = fleet.enqueue_merge(
                owner=owner,
                branch=branch,
                session_id=session_id,
                pr_number=pr_number,
                override_claim_conflicts=override,
            )
            if not result.get("ok"):
                return error_response(result.get("error", "invalid merge queue request"), 400)
            return json_response(result, status=201)

        if action == "advance":
            return json_response(fleet.advance_merge_queue())

        if action == "remove":
            item_id = str(body.get("id", "")).strip()
            if not item_id:
                return error_response("'id' is required for remove", 400)
            return json_response(fleet.remove_merge_item(item_id))

        if action == "clear":
            return json_response(fleet.clear_merge_queue())

        return error_response("Unsupported action. Use enqueue|advance|remove|clear", 400)

    # =========================================================================
    # Stats and Health
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/stats",
        summary="Get coordination statistics",
        tags=["Coordination"],
    )
    @require_permission("coordination:stats.read")
    def _handle_coordination_stats(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get coordination statistics."""
        coordinator, err = self._require_coordination_coordinator()
        if err:
            return err

        try:
            stats = coordinator.get_stats()
            return json_response(stats)
        except (ValueError, RuntimeError) as e:
            logger.error("Error getting coordination stats: %s", e)
            return error_response(safe_error_message(e, "coordination"), 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/coordination/health",
        summary="Coordination health check",
        tags=["Coordination"],
    )
    @require_permission("coordination:health.read")
    def _handle_coordination_health(self, query_params: dict[str, Any]) -> HandlerResult:
        """Check coordination service health."""
        coordinator = self._get_coordination_coordinator()
        if not coordinator:
            return json_response(
                {
                    "status": "unavailable",
                    "message": "Coordination service not initialized",
                }
            )

        try:
            stats = coordinator.get_stats()
            return json_response(
                {
                    "status": "healthy",
                    "total_workspaces": stats.get("total_workspaces", 0),
                    "pending_requests": stats.get("pending_requests", 0),
                    "valid_consents": stats.get("valid_consents", 0),
                }
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Coordination health check failed: %s", e)
            return json_response(
                {
                    "status": "degraded",
                    "error": safe_error_message(e, "coordination"),
                }
            )


__all__ = ["CoordinationHandlerMixin"]

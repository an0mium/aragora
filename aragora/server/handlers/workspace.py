"""
Workspace Handler - Enterprise Privacy and Data Isolation APIs.

Provides API endpoints for:
- Workspace creation and management
- Data isolation and access control
- Retention policy management
- Sensitivity classification
- Privacy audit logging

Endpoints:
- POST /api/workspaces - Create a new workspace
- GET /api/workspaces - List workspaces
- GET /api/workspaces/{id} - Get workspace details
- DELETE /api/workspaces/{id} - Delete workspace
- POST /api/workspaces/{id}/members - Add member to workspace
- DELETE /api/workspaces/{id}/members/{user_id} - Remove member
- GET /api/retention/policies - List retention policies
- POST /api/retention/policies - Create retention policy
- PUT /api/retention/policies/{id} - Update retention policy
- DELETE /api/retention/policies/{id} - Delete retention policy
- POST /api/retention/policies/{id}/execute - Execute retention policy
- GET /api/retention/expiring - Get items expiring soon
- POST /api/classify - Classify content sensitivity
- GET /api/classify/policy/{level} - Get policy for sensitivity level
- GET /api/audit/entries - Query audit entries
- GET /api/audit/report - Generate compliance report
- GET /api/audit/verify - Verify audit log integrity

SOC 2 Controls: CC6.1, CC6.3 - Logical access controls
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.http_utils import run_async

from aragora.billing.jwt_auth import extract_user_from_request

# RBAC imports - graceful fallback if not available
try:
    from aragora.rbac import AuthorizationContext, check_permission

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    AuthorizationContext = None  # type: ignore[misc]
    check_permission = None
from aragora.privacy import (
    AccessDeniedException,
    AuditAction,
    AuditOutcome,
    ClassificationConfig,
    DataIsolationManager,
    IsolationConfig,
    PrivacyAuditLog,
    RetentionAction,
    RetentionPolicyManager,
    SensitivityClassifier,
    SensitivityLevel,
    WorkspacePermission,
)
from aragora.privacy.audit_log import Actor, Resource

from .base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from .secure import SecureHandler
from .utils.rate_limit import rate_limit

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WorkspaceHandler(SecureHandler):
    """Handler for workspace and privacy management endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

    RESOURCE_TYPE = "workspace"

    ROUTES = [
        "/api/workspaces",
        "/api/retention/policies",
        "/api/retention/expiring",
        "/api/classify",
        "/api/audit/entries",
        "/api/audit/report",
        "/api/audit/verify",
        "/api/audit/actor",
        "/api/audit/resource",
        "/api/audit/denied",
    ]

    def __init__(self, server_context):
        super().__init__(server_context)
        self._isolation_manager: DataIsolationManager | None = None
        self._retention_manager: RetentionPolicyManager | None = None
        self._classifier: SensitivityClassifier | None = None
        self._audit_log: PrivacyAuditLog | None = None

    def _get_isolation_manager(self) -> DataIsolationManager:
        """Get or create isolation manager."""
        if self._isolation_manager is None:
            self._isolation_manager = DataIsolationManager(IsolationConfig())
        return self._isolation_manager

    def _get_retention_manager(self) -> RetentionPolicyManager:
        """Get or create retention manager."""
        if self._retention_manager is None:
            self._retention_manager = RetentionPolicyManager()
        return self._retention_manager

    def _get_classifier(self) -> SensitivityClassifier:
        """Get or create sensitivity classifier."""
        if self._classifier is None:
            self._classifier = SensitivityClassifier(ClassificationConfig())
        return self._classifier

    def _get_audit_log(self) -> PrivacyAuditLog:
        """Get or create audit log."""
        if self._audit_log is None:
            self._audit_log = PrivacyAuditLog()
        return self._audit_log

    def _run_async(self, coro):
        """Run an async coroutine from sync context.

        Delegates to the centralized run_async utility which handles
        event loop management and nested loop edge cases.
        """
        return run_async(coro)

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_auth_context(self, handler, auth_ctx=None) -> Optional[AuthorizationContext]:
        """Build RBAC authorization context from request."""
        if not RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        if auth_ctx is None:
            user_store = self._get_user_store()
            auth_ctx = extract_user_from_request(handler, user_store)

        if not auth_ctx.is_authenticated:
            return None

        # Get user role from user store if available
        user_store = self._get_user_store()
        user = user_store.get_user_by_id(auth_ctx.user_id) if user_store else None
        roles = set([user.role]) if user and user.role else set()

        return AuthorizationContext(
            user_id=auth_ctx.user_id,
            roles=roles,
            org_id=auth_ctx.org_id,
        )

    def _check_rbac_permission(
        self, handler, permission_key: str, auth_ctx=None
    ) -> Optional[HandlerResult]:
        """
        Check RBAC permission.

        Returns None if allowed, or an error response if denied.
        """
        if not RBAC_AVAILABLE:
            return None

        rbac_ctx = self._get_auth_context(handler, auth_ctx)
        if not rbac_ctx:
            # No auth context - rely on existing auth checks
            return None

        decision = check_permission(rbac_ctx, permission_key)
        if not decision.allowed:
            logger.warning(
                f"RBAC denied: user={rbac_ctx.user_id} permission={permission_key} "
                f"reason={decision.reason}"
            )
            return error_response(
                {"error": "Permission denied", "reason": decision.reason},
                403,
            )

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return any(path.startswith(route) for route in self.ROUTES)

    def handle(
        self, path: str, query_params: dict, handler: Any, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route GET requests."""
        if hasattr(handler, "command"):
            method = handler.command

        # Workspace endpoints
        if path.startswith("/api/workspaces"):
            return self._route_workspace(path, query_params, handler, method)

        # Retention endpoints
        if path.startswith("/api/retention"):
            return self._route_retention(path, query_params, handler, method)

        # Classification endpoints
        if path.startswith("/api/classify"):
            return self._route_classify(path, query_params, handler, method)

        # Audit endpoints
        if path.startswith("/api/audit"):
            return self._route_audit(path, query_params, handler, method)

        return None

    def handle_post(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route POST requests."""
        return self.handle(path, query_params, handler, method="POST")

    def handle_delete(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route DELETE requests."""
        return self.handle(path, query_params, handler, method="DELETE")

    def handle_put(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route PUT requests."""
        return self.handle(path, query_params, handler, method="PUT")

    # =========================================================================
    # Workspace Routing
    # =========================================================================

    def _route_workspace(
        self, path: str, query_params: dict, handler: Any, method: str
    ) -> Optional[HandlerResult]:
        """Route workspace requests."""
        parts = path.strip("/").split("/")

        # POST /api/workspaces - Create workspace
        if path == "/api/workspaces" and method == "POST":
            return self._handle_create_workspace(handler)

        # GET /api/workspaces - List workspaces
        if path == "/api/workspaces" and method == "GET":
            return self._handle_list_workspaces(handler, query_params)

        # GET /api/workspaces/{id}
        if len(parts) == 3 and method == "GET":
            workspace_id = parts[2]
            return self._handle_get_workspace(handler, workspace_id)

        # DELETE /api/workspaces/{id}
        if len(parts) == 3 and method == "DELETE":
            workspace_id = parts[2]
            return self._handle_delete_workspace(handler, workspace_id)

        # POST /api/workspaces/{id}/members - Add member
        if len(parts) == 4 and parts[3] == "members" and method == "POST":
            workspace_id = parts[2]
            return self._handle_add_member(handler, workspace_id)

        # DELETE /api/workspaces/{id}/members/{user_id} - Remove member
        if len(parts) == 5 and parts[3] == "members" and method == "DELETE":
            workspace_id = parts[2]
            user_id = parts[4]
            return self._handle_remove_member(handler, workspace_id, user_id)

        return error_response("Not found", 404)

    # =========================================================================
    # Retention Routing
    # =========================================================================

    def _route_retention(
        self, path: str, query_params: dict, handler: Any, method: str
    ) -> Optional[HandlerResult]:
        """Route retention requests."""
        parts = path.strip("/").split("/")

        # GET /api/retention/policies - List policies
        if path == "/api/retention/policies" and method == "GET":
            return self._handle_list_policies(handler, query_params)

        # POST /api/retention/policies - Create policy
        if path == "/api/retention/policies" and method == "POST":
            return self._handle_create_policy(handler)

        # GET /api/retention/policies/{id}
        if len(parts) == 4 and parts[2] == "policies" and method == "GET":
            policy_id = parts[3]
            return self._handle_get_policy(handler, policy_id)

        # PUT /api/retention/policies/{id}
        if len(parts) == 4 and parts[2] == "policies" and method == "PUT":
            policy_id = parts[3]
            return self._handle_update_policy(handler, policy_id)

        # DELETE /api/retention/policies/{id}
        if len(parts) == 4 and parts[2] == "policies" and method == "DELETE":
            policy_id = parts[3]
            return self._handle_delete_policy(handler, policy_id)

        # POST /api/retention/policies/{id}/execute
        if len(parts) == 5 and parts[4] == "execute" and method == "POST":
            policy_id = parts[3]
            return self._handle_execute_policy(handler, policy_id, query_params)

        # GET /api/retention/expiring
        if path == "/api/retention/expiring" and method == "GET":
            return self._handle_expiring_items(handler, query_params)

        return error_response("Not found", 404)

    # =========================================================================
    # Classification Routing
    # =========================================================================

    def _route_classify(
        self, path: str, query_params: dict, handler: Any, method: str
    ) -> Optional[HandlerResult]:
        """Route classification requests."""
        parts = path.strip("/").split("/")

        # POST /api/classify - Classify content
        if path == "/api/classify" and method == "POST":
            return self._handle_classify_content(handler)

        # GET /api/classify/policy/{level}
        if len(parts) == 4 and parts[2] == "policy" and method == "GET":
            level = parts[3]
            return self._handle_get_level_policy(handler, level)

        return error_response("Not found", 404)

    # =========================================================================
    # Audit Routing
    # =========================================================================

    def _route_audit(
        self, path: str, query_params: dict, handler: Any, method: str
    ) -> Optional[HandlerResult]:
        """Route audit requests."""
        parts = path.strip("/").split("/")

        # GET /api/audit/entries - Query audit entries
        if path == "/api/audit/entries" and method == "GET":
            return self._handle_query_audit(handler, query_params)

        # GET /api/audit/report - Generate compliance report
        if path == "/api/audit/report" and method == "GET":
            return self._handle_audit_report(handler, query_params)

        # GET /api/audit/verify - Verify integrity
        if path == "/api/audit/verify" and method == "GET":
            return self._handle_verify_integrity(handler, query_params)

        # GET /api/audit/actor/{id}/history
        if len(parts) >= 4 and parts[2] == "actor" and method == "GET":
            actor_id = parts[3]
            return self._handle_actor_history(handler, actor_id, query_params)

        # GET /api/audit/resource/{id}/history
        if len(parts) >= 4 and parts[2] == "resource" and method == "GET":
            resource_id = parts[3]
            return self._handle_resource_history(handler, resource_id, query_params)

        # GET /api/audit/denied - Get denied access attempts
        if path == "/api/audit/denied" and method == "GET":
            return self._handle_denied_access(handler, query_params)

        return error_response("Not found", 404)

    # =========================================================================
    # Workspace Handlers
    # =========================================================================

    @rate_limit(rpm=30, limiter_name="workspace_create")
    @handle_errors("create workspace")
    @log_request("create workspace")
    def _handle_create_workspace(self, handler) -> HandlerResult:
        """Create a new workspace."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "workspaces.create", auth_ctx)
        if rbac_error:
            return rbac_error

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        org_id = body.get("organization_id") or auth_ctx.org_id
        if not org_id:
            return error_response("organization_id is required", 400)

        initial_members = body.get("members", [])

        manager = self._get_isolation_manager()
        workspace = self._run_async(
            manager.create_workspace(
                organization_id=org_id,
                name=name,
                created_by=auth_ctx.user_id,
                initial_members=initial_members,
            )
        )

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.CREATE_WORKSPACE,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=workspace.id, type="workspace", workspace_id=workspace.id),
                outcome=AuditOutcome.SUCCESS,
                details={"name": name, "org_id": org_id},
            )
        )

        logger.info(f"Created workspace {workspace.id} for org {org_id}")

        return json_response(
            {
                "workspace": workspace.to_dict(),
                "message": "Workspace created successfully",
            },
            status=201,
        )

    @handle_errors("list workspaces")
    def _handle_list_workspaces(self, handler, query_params: dict) -> HandlerResult:
        """List workspaces accessible to user."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        org_id = query_params.get("organization_id", auth_ctx.org_id)
        manager = self._get_isolation_manager()
        workspaces = self._run_async(
            manager.list_workspaces(
                actor=auth_ctx.user_id,
                organization_id=org_id,
            )
        )

        return json_response(
            {
                "workspaces": [w.to_dict() for w in workspaces],
                "total": len(workspaces),
            }
        )

    @handle_errors("get workspace")
    def _handle_get_workspace(self, handler, workspace_id: str) -> HandlerResult:
        """Get workspace details."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        manager = self._get_isolation_manager()
        try:
            workspace = self._run_async(
                manager.get_workspace(
                    workspace_id=workspace_id,
                    actor=auth_ctx.user_id,
                )
            )
        except AccessDeniedException as e:
            return error_response(str(e), 403)

        return json_response({"workspace": workspace.to_dict()})

    @rate_limit(rpm=10, limiter_name="workspace_delete")
    @handle_errors("delete workspace")
    @log_request("delete workspace")
    def _handle_delete_workspace(self, handler, workspace_id: str) -> HandlerResult:
        """Delete a workspace."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "workspaces.delete", auth_ctx)
        if rbac_error:
            return rbac_error

        body = self.read_json_body(handler) or {}
        force = body.get("force", False)

        manager = self._get_isolation_manager()
        try:
            self._run_async(
                manager.delete_workspace(
                    workspace_id=workspace_id,
                    deleted_by=auth_ctx.user_id,
                    force=force,
                )
            )
        except AccessDeniedException as e:
            return error_response(str(e), 403)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.DELETE_WORKSPACE,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=workspace_id, type="workspace", workspace_id=workspace_id),
                outcome=AuditOutcome.SUCCESS,
            )
        )

        return json_response({"message": "Workspace deleted successfully"})

    @rate_limit(rpm=30, limiter_name="workspace_member")
    @handle_errors("add workspace member")
    @log_request("add workspace member")
    def _handle_add_member(self, handler, workspace_id: str) -> HandlerResult:
        """Add a member to a workspace."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "workspaces.members.add", auth_ctx)
        if rbac_error:
            return rbac_error

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        user_id = body.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        permissions_raw = body.get("permissions", ["read"])
        permissions = [WorkspacePermission(p) for p in permissions_raw]

        manager = self._get_isolation_manager()
        try:
            self._run_async(
                manager.add_member(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    permissions=permissions,
                    added_by=auth_ctx.user_id,
                )
            )
        except AccessDeniedException as e:
            return error_response(str(e), 403)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.ADD_MEMBER,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=workspace_id, type="workspace", workspace_id=workspace_id),
                outcome=AuditOutcome.SUCCESS,
                details={"added_user_id": user_id, "permissions": permissions_raw},
            )
        )

        return json_response({"message": f"Member {user_id} added to workspace"}, status=201)

    @rate_limit(rpm=30, limiter_name="workspace_member")
    @handle_errors("remove workspace member")
    @log_request("remove workspace member")
    def _handle_remove_member(self, handler, workspace_id: str, user_id: str) -> HandlerResult:
        """Remove a member from a workspace."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "workspaces.members.remove", auth_ctx)
        if rbac_error:
            return rbac_error

        manager = self._get_isolation_manager()
        try:
            self._run_async(
                manager.remove_member(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    removed_by=auth_ctx.user_id,
                )
            )
        except AccessDeniedException as e:
            return error_response(str(e), 403)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.REMOVE_MEMBER,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=workspace_id, type="workspace", workspace_id=workspace_id),
                outcome=AuditOutcome.SUCCESS,
                details={"removed_user_id": user_id},
            )
        )

        return json_response({"message": f"Member {user_id} removed from workspace"})

    # =========================================================================
    # Retention Policy Handlers
    # =========================================================================

    @handle_errors("list retention policies")
    def _handle_list_policies(self, handler, query_params: dict) -> HandlerResult:
        """List retention policies."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        workspace_id = query_params.get("workspace_id")
        manager = self._get_retention_manager()

        policies = manager.list_policies(workspace_id=workspace_id)

        return json_response(
            {
                "policies": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "description": p.description,
                        "retention_days": p.retention_days,
                        "action": p.action.value,
                        "enabled": p.enabled,
                        "applies_to": p.applies_to,
                        "last_run": p.last_run.isoformat() if p.last_run else None,
                    }
                    for p in policies
                ],
                "total": len(policies),
            }
        )

    @rate_limit(rpm=20, limiter_name="retention_policy")
    @handle_errors("create retention policy")
    @log_request("create retention policy")
    def _handle_create_policy(self, handler) -> HandlerResult:
        """Create a retention policy."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "retention.policies.create", auth_ctx)
        if rbac_error:
            return rbac_error

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        retention_days = body.get("retention_days", 90)
        action_str = body.get("action", "delete")

        try:
            action = RetentionAction(action_str)
        except ValueError:
            return error_response(
                f"Invalid action: {action_str}. Valid: delete, archive, anonymize, notify",
                400,
            )

        workspace_ids = body.get("workspace_ids")
        description = body.get("description", "")
        applies_to = body.get("applies_to", ["documents", "findings", "sessions"])

        manager = self._get_retention_manager()
        policy = manager.create_policy(
            name=name,
            retention_days=retention_days,
            action=action,
            workspace_ids=workspace_ids,
            description=description,
            applies_to=applies_to,
        )

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.MODIFY_POLICY,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=policy.id, type="retention_policy"),
                outcome=AuditOutcome.SUCCESS,
                details={"operation": "create", "name": name},
            )
        )

        return json_response(
            {
                "policy": {
                    "id": policy.id,
                    "name": policy.name,
                    "retention_days": policy.retention_days,
                    "action": policy.action.value,
                },
                "message": "Policy created successfully",
            },
            status=201,
        )

    @handle_errors("get retention policy")
    def _handle_get_policy(self, handler, policy_id: str) -> HandlerResult:
        """Get a retention policy."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        manager = self._get_retention_manager()
        policy = manager.get_policy(policy_id)

        if not policy:
            return error_response("Policy not found", 404)

        return json_response(
            {
                "policy": {
                    "id": policy.id,
                    "name": policy.name,
                    "description": policy.description,
                    "retention_days": policy.retention_days,
                    "action": policy.action.value,
                    "enabled": policy.enabled,
                    "applies_to": policy.applies_to,
                    "workspace_ids": policy.workspace_ids,
                    "grace_period_days": policy.grace_period_days,
                    "notify_before_days": policy.notify_before_days,
                    "exclude_sensitivity_levels": policy.exclude_sensitivity_levels,
                    "exclude_tags": policy.exclude_tags,
                    "created_at": policy.created_at.isoformat(),
                    "last_run": policy.last_run.isoformat() if policy.last_run else None,
                }
            }
        )

    @rate_limit(rpm=20, limiter_name="retention_policy")
    @handle_errors("update retention policy")
    @log_request("update retention policy")
    def _handle_update_policy(self, handler, policy_id: str) -> HandlerResult:
        """Update a retention policy."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "retention.policies.update", auth_ctx)
        if rbac_error:
            return rbac_error

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        manager = self._get_retention_manager()

        # Convert action string to enum if present
        if "action" in body:
            try:
                body["action"] = RetentionAction(body["action"])
            except ValueError:
                return error_response(f"Invalid action: {body['action']}", 400)

        try:
            policy = manager.update_policy(policy_id, **body)
        except ValueError as e:
            return error_response(str(e), 404)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.MODIFY_POLICY,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=policy_id, type="retention_policy"),
                outcome=AuditOutcome.SUCCESS,
                details={"operation": "update", "changes": list(body.keys())},
            )
        )

        return json_response(
            {
                "policy": {
                    "id": policy.id,
                    "name": policy.name,
                    "retention_days": policy.retention_days,
                },
                "message": "Policy updated successfully",
            }
        )

    @rate_limit(rpm=10, limiter_name="retention_policy")
    @handle_errors("delete retention policy")
    @log_request("delete retention policy")
    def _handle_delete_policy(self, handler, policy_id: str) -> HandlerResult:
        """Delete a retention policy."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "retention.policies.delete", auth_ctx)
        if rbac_error:
            return rbac_error

        manager = self._get_retention_manager()
        manager.delete_policy(policy_id)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.MODIFY_POLICY,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=policy_id, type="retention_policy"),
                outcome=AuditOutcome.SUCCESS,
                details={"operation": "delete"},
            )
        )

        return json_response({"message": "Policy deleted successfully"})

    @rate_limit(rpm=5, limiter_name="retention_execute")
    @handle_errors("execute retention policy")
    @log_request("execute retention policy")
    def _handle_execute_policy(self, handler, policy_id: str, query_params: dict) -> HandlerResult:
        """Execute a retention policy."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # RBAC permission check
        rbac_error = self._check_rbac_permission(handler, "retention.policies.execute", auth_ctx)
        if rbac_error:
            return rbac_error

        dry_run = query_params.get("dry_run", "false").lower() == "true"
        manager = self._get_retention_manager()

        try:
            report = self._run_async(manager.execute_policy(policy_id, dry_run=dry_run))
        except ValueError as e:
            return error_response(str(e), 404)

        # Log to audit
        audit_log = self._get_audit_log()
        self._run_async(
            audit_log.log(
                action=AuditAction.EXECUTE_RETENTION,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=policy_id, type="retention_policy"),
                outcome=AuditOutcome.SUCCESS,
                details={
                    "dry_run": dry_run,
                    "items_deleted": report.items_deleted,
                    "items_evaluated": report.items_evaluated,
                },
            )
        )

        return json_response({"report": report.to_dict(), "dry_run": dry_run})

    @handle_errors("get expiring items")
    def _handle_expiring_items(self, handler, query_params: dict) -> HandlerResult:
        """Get items expiring soon."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        workspace_id = query_params.get("workspace_id")
        days = int(query_params.get("days", "14"))

        manager = self._get_retention_manager()
        expiring = self._run_async(
            manager.check_expiring_soon(workspace_id=workspace_id, days=days)
        )

        return json_response({"expiring": expiring, "total": len(expiring), "days_ahead": days})

    # =========================================================================
    # Classification Handlers
    # =========================================================================

    @rate_limit(rpm=60, limiter_name="classify")
    @handle_errors("classify content")
    def _handle_classify_content(self, handler) -> HandlerResult:
        """Classify content sensitivity."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        content = body.get("content")
        if not content:
            return error_response("content is required", 400)

        document_id = body.get("document_id", "")
        metadata = body.get("metadata", {})

        classifier = self._get_classifier()
        result = self._run_async(
            classifier.classify(
                content=content,
                document_id=document_id,
                metadata=metadata,
            )
        )

        # Log to audit if document_id provided
        if document_id:
            audit_log = self._get_audit_log()
            self._run_async(
                audit_log.log(
                    action=AuditAction.CLASSIFY_DOCUMENT,
                    actor=Actor(id=auth_ctx.user_id, type="user"),
                    resource=Resource(
                        id=document_id,
                        type="document",
                        sensitivity_level=result.level.value,
                    ),
                    outcome=AuditOutcome.SUCCESS,
                    details={"level": result.level.value, "confidence": result.confidence},
                )
            )

        return json_response({"classification": result.to_dict()})

    @handle_errors("get level policy")
    def _handle_get_level_policy(self, handler, level: str) -> HandlerResult:
        """Get recommended policy for a sensitivity level."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        try:
            sensitivity_level = SensitivityLevel(level)
        except ValueError:
            valid_levels = [lvl.value for lvl in SensitivityLevel]
            return error_response(f"Invalid level: {level}. Valid: {', '.join(valid_levels)}", 400)

        classifier = self._get_classifier()
        policy = classifier.get_level_policy(sensitivity_level)

        return json_response({"level": level, "policy": policy})

    # =========================================================================
    # Audit Log Handlers
    # =========================================================================

    @handle_errors("query audit entries")
    def _handle_query_audit(self, handler, query_params: dict) -> HandlerResult:
        """Query audit log entries."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Parse filters
        start_date = None
        end_date = None
        if "start_date" in query_params:
            start_date = datetime.fromisoformat(query_params["start_date"])
        if "end_date" in query_params:
            end_date = datetime.fromisoformat(query_params["end_date"])

        actor_id = query_params.get("actor_id")
        resource_id = query_params.get("resource_id")
        workspace_id = query_params.get("workspace_id")
        action_str = query_params.get("action")
        outcome_str = query_params.get("outcome")
        limit = int(query_params.get("limit", "100"))

        action = AuditAction(action_str) if action_str else None
        outcome = AuditOutcome(outcome_str) if outcome_str else None

        audit_log = self._get_audit_log()
        entries = self._run_async(
            audit_log.query(
                start_date=start_date,
                end_date=end_date,
                actor_id=actor_id,
                resource_id=resource_id,
                workspace_id=workspace_id,
                action=action,
                outcome=outcome,
                limit=limit,
            )
        )

        return json_response(
            {
                "entries": [e.to_dict() for e in entries],
                "total": len(entries),
                "limit": limit,
            }
        )

    @handle_errors("generate audit report")
    def _handle_audit_report(self, handler, query_params: dict) -> HandlerResult:
        """Generate compliance report."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        start_date = None
        end_date = None
        if "start_date" in query_params:
            start_date = datetime.fromisoformat(query_params["start_date"])
        if "end_date" in query_params:
            end_date = datetime.fromisoformat(query_params["end_date"])

        workspace_id = query_params.get("workspace_id")
        format_type = query_params.get("format", "json")

        audit_log = self._get_audit_log()
        report = self._run_async(
            audit_log.generate_compliance_report(
                start_date=start_date,
                end_date=end_date,
                workspace_id=workspace_id,
                format=format_type,
            )
        )

        # Log report generation
        self._run_async(
            audit_log.log(
                action=AuditAction.GENERATE_REPORT,
                actor=Actor(id=auth_ctx.user_id, type="user"),
                resource=Resource(id=report["report_id"], type="compliance_report"),
                outcome=AuditOutcome.SUCCESS,
                details={"workspace_id": workspace_id},
            )
        )

        return json_response({"report": report})

    @handle_errors("verify audit integrity")
    def _handle_verify_integrity(self, handler, query_params: dict) -> HandlerResult:
        """Verify audit log integrity."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        start_date = None
        end_date = None
        if "start_date" in query_params:
            start_date = datetime.fromisoformat(query_params["start_date"])
        if "end_date" in query_params:
            end_date = datetime.fromisoformat(query_params["end_date"])

        audit_log = self._get_audit_log()
        is_valid, errors = self._run_async(
            audit_log.verify_integrity(start_date=start_date, end_date=end_date)
        )

        return json_response(
            {
                "valid": is_valid,
                "errors": errors,
                "error_count": len(errors),
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @handle_errors("get actor history")
    def _handle_actor_history(self, handler, actor_id: str, query_params: dict) -> HandlerResult:
        """Get all actions by a specific actor."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        days = int(query_params.get("days", "30"))
        audit_log = self._get_audit_log()
        entries = self._run_async(audit_log.get_actor_history(actor_id=actor_id, days=days))

        return json_response(
            {
                "actor_id": actor_id,
                "entries": [e.to_dict() for e in entries],
                "total": len(entries),
                "days": days,
            }
        )

    @handle_errors("get resource history")
    def _handle_resource_history(
        self, handler, resource_id: str, query_params: dict
    ) -> HandlerResult:
        """Get all actions on a specific resource."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        days = int(query_params.get("days", "30"))
        audit_log = self._get_audit_log()
        entries = self._run_async(
            audit_log.get_resource_history(resource_id=resource_id, days=days)
        )

        return json_response(
            {
                "resource_id": resource_id,
                "entries": [e.to_dict() for e in entries],
                "total": len(entries),
                "days": days,
            }
        )

    @handle_errors("get denied access attempts")
    def _handle_denied_access(self, handler, query_params: dict) -> HandlerResult:
        """Get all denied access attempts."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        days = int(query_params.get("days", "7"))
        audit_log = self._get_audit_log()
        entries = self._run_async(audit_log.get_denied_access_attempts(days=days))

        return json_response(
            {
                "denied_attempts": [e.to_dict() for e in entries],
                "total": len(entries),
                "days": days,
            }
        )


__all__ = ["WorkspaceHandler"]

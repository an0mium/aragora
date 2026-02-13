"""
HTTP request handlers for workflow API endpoints.

Provides the WorkflowHandler and WorkflowHandlers classes for integration
with the unified server.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import sys

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    PaginatedHandlerMixin,
    error_response,
    json_response,
    get_int_param,
    get_string_param,
)
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.openapi_decorator import api_endpoint

from .core import (
    logger,
    _run_async,
    WorkflowDefinition,
    RBAC_AVAILABLE,
    record_rbac_check,
    track_handler,
    _UnauthenticatedSentinel,
)

from .crud import (
    list_workflows,
    get_workflow,
    create_workflow,
    update_workflow,
    delete_workflow,
)
from .versions import (
    get_workflow_versions,
    restore_workflow_version,
)
from .execution import (
    execute_workflow,
    get_execution,
    list_executions,
    terminate_execution,
)
from .templates import list_templates
from .approvals import (
    list_pending_approvals,
    resolve_approval,
)

if TYPE_CHECKING:
    from aragora.rbac import AuthorizationContext

# Import RBAC components conditionally
if RBAC_AVAILABLE:
    from aragora.rbac import (
        AuthorizationContext,
        check_permission,
        PermissionDeniedError,
        get_role_permissions,
    )
    from aragora.billing.auth import extract_user_from_request


def _workflows_module():
    """Return workflows package module to honor test patching."""
    module = sys.modules.get("aragora.server.handlers.workflows")
    if module is None:
        from aragora.server.handlers import workflows as module
    return module


def _normalize_execute_inputs(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """Normalize workflow execute payloads.

    Accepts both of these request body shapes:
    - {"inputs": {...}} (canonical)
    - {...} (flat payload treated as inputs)

    Also merges workflow notification-related top-level fields into nested
    `inputs` for compatibility with CLI payload conventions.
    """
    if payload is None:
        return {}, None
    if not isinstance(payload, dict):
        return {}, "Request body must be a JSON object"

    raw_inputs = payload.get("inputs")
    if raw_inputs is None:
        return dict(payload), None
    if not isinstance(raw_inputs, dict):
        return {}, "inputs must be an object"

    inputs = dict(raw_inputs)
    compat_keys = (
        "channel_targets",
        "chat_targets",
        "notify_channels",
        "approval_targets",
        "notify_steps",
        "thread_id",
        "origin_thread_id",
        "thread_id_by_platform",
    )
    for key in compat_keys:
        if key in payload and key not in inputs:
            inputs[key] = payload[key]

    return inputs, None


# =============================================================================
# Legacy HTTP Route Handlers (for integration with unified_server)
# =============================================================================


class WorkflowHandlers:
    """HTTP handlers for workflow API (legacy interface)."""

    @staticmethod
    @require_permission("workflows:read")
    async def handle_list_workflows(params: dict[str, Any]) -> dict[str, Any]:
        """GET /api/workflows"""
        workflows_module = _workflows_module()
        return await workflows_module.list_workflows(
            tenant_id=params.get("tenant_id", "default"),
            category=params.get("category"),
            tags=params.get("tags"),
            search=params.get("search"),
            limit=int(params.get("limit", 50)),
            offset=int(params.get("offset", 0)),
        )

    @staticmethod
    @require_permission("workflows:read")
    async def handle_get_workflow(
        workflow_id: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """GET /api/workflows/:id"""
        workflows_module = _workflows_module()
        return await workflows_module.get_workflow(
            workflow_id,
            params.get("tenant_id", "default"),
        )

    @staticmethod
    @require_permission("workflows:write")
    async def handle_create_workflow(
        data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/workflows"""
        workflows_module = _workflows_module()
        return await workflows_module.create_workflow(
            data,
            tenant_id=params.get("tenant_id", "default"),
            created_by=params.get("user_id", ""),
        )

    @staticmethod
    @require_permission("workflows:write")
    async def handle_update_workflow(
        workflow_id: str, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """PUT /api/workflows/:id"""
        workflows_module = _workflows_module()
        return await workflows_module.update_workflow(
            workflow_id,
            data,
            params.get("tenant_id", "default"),
        )

    @staticmethod
    @require_permission("workflows:delete")
    async def handle_delete_workflow(workflow_id: str, params: dict[str, Any]) -> bool:
        """DELETE /api/workflows/:id"""
        workflows_module = _workflows_module()
        return await workflows_module.delete_workflow(
            workflow_id,
            params.get("tenant_id", "default"),
        )

    @staticmethod
    @require_permission("workflows:read")
    async def handle_execute_workflow(
        workflow_id: str, data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/workflows/:id/execute"""
        workflows_module = _workflows_module()
        inputs, input_error = _normalize_execute_inputs(data)
        if input_error:
            raise ValueError(input_error)
        return await workflows_module.execute_workflow(
            workflow_id,
            inputs=inputs,
            tenant_id=params.get("tenant_id", "default"),
            user_id=params.get("user_id"),
            org_id=params.get("org_id"),
        )

    @staticmethod
    @require_permission("workflows:read")
    async def handle_list_templates(params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET /api/workflow-templates"""
        workflows_module = _workflows_module()
        return await workflows_module.list_templates(
            category=params.get("category"),
            tags=params.get("tags"),
        )

    @staticmethod
    @require_permission("workflows:read")
    async def handle_list_approvals(params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET /api/workflow-approvals"""
        workflows_module = _workflows_module()
        return await workflows_module.list_pending_approvals(
            workflow_id=params.get("workflow_id"),
            tenant_id=params.get("tenant_id", "default"),
        )

    @staticmethod
    @require_permission("workflows:read")
    async def handle_resolve_approval(
        request_id: str, data: dict[str, Any], params: dict[str, Any]
    ) -> bool:
        """POST /api/workflow-approvals/:id/resolve"""
        workflows_module = _workflows_module()
        return await workflows_module.resolve_approval(
            request_id,
            status=data.get("status", "approved"),
            responder_id=params.get("user_id", ""),
            notes=data.get("notes", ""),
            checklist_updates=data.get("checklist"),
        )


# =============================================================================
# BaseHandler Integration for Unified Server
# =============================================================================


class WorkflowHandler(BaseHandler, PaginatedHandlerMixin):
    """
    HTTP request handler for workflow API endpoints.

    Integrates with the unified server's dispatch mechanism using BaseHandler.
    Provides REST API for managing and executing workflows.

    Routes:
        GET    /api/workflows              - List workflows
        POST   /api/workflows              - Create workflow
        GET    /api/workflows/{id}         - Get workflow details
        PATCH  /api/workflows/{id}         - Update workflow
        DELETE /api/workflows/{id}         - Delete workflow
        POST   /api/workflows/{id}/execute - Execute workflow
        POST   /api/workflows/{id}/simulate - Dry-run workflow
        GET    /api/workflows/{id}/status  - Get execution status
        GET    /api/workflows/{id}/versions - Get version history
        GET    /api/workflow-templates     - List templates
        POST   /api/workflow-approvals/{id}/resolve - Resolve approval
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/workflows",
        "/api/v1/workflows/*",
        "/api/v1/workflow-templates",
        "/api/v1/workflows/templates",
        "/api/v1/workflows/templates/*",
        "/api/v1/workflow-approvals",
        "/api/v1/workflow-approvals/*",
        "/api/v1/workflow-executions",
        "/api/v1/workflow-executions/*",
        "/api/v1/workflows/executions",
        "/api/v1/workflows/executions/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return (
            path.startswith("/api/v1/workflows")
            or path.startswith("/api/v1/workflow-templates")
            or path.startswith("/api/v1/workflows/templates")
            or path.startswith("/api/v1/workflow-approvals")
            or path.startswith("/api/v1/workflow-executions")
            or path.startswith("/api/v1/workflows/executions")
        )

    def _rbac_enabled(self) -> bool:
        """Determine if RBAC checks should be enforced."""
        rbac_enabled = RBAC_AVAILABLE
        try:
            pkg = sys.modules.get("aragora.server.handlers.workflows")
            if pkg is not None and hasattr(pkg, "RBAC_AVAILABLE"):
                if not getattr(pkg, "RBAC_AVAILABLE"):
                    rbac_enabled = False
        except Exception as e:
            logger.debug("Failed to check RBAC_AVAILABLE override: %s", e)
        return bool(rbac_enabled)

    def _list_pending_approvals_fn(self) -> Any:
        """Return list_pending_approvals, honoring test overrides."""
        try:
            pkg = sys.modules.get("aragora.server.handlers.workflows")
            override = getattr(pkg, "list_pending_approvals", None) if pkg is not None else None
            if override is not None and override is not list_pending_approvals:
                return override
        except Exception as e:
            logger.debug("Failed to resolve list_pending_approvals override: %s", e)
        return list_pending_approvals

    def _run_async_fn(self) -> Any:
        """Return _run_async, honoring test overrides."""
        try:
            pkg = sys.modules.get("aragora.server.handlers.workflows")
            override = getattr(pkg, "_run_async", None) if pkg is not None else None
            if override is not None and override is not _run_async:
                return override
        except Exception as e:
            logger.debug("Failed to resolve _run_async override: %s", e)
        return _run_async

    def _get_auth_context(
        self, handler: Any
    ) -> AuthorizationContext | _UnauthenticatedSentinel | None:
        """Build AuthorizationContext from validated JWT token.

        SECURITY: Only accepts JWT-based authentication. Header-based auth
        has been removed to prevent identity spoofing and privilege escalation.

        Returns:
            AuthorizationContext if JWT is valid and RBAC available,
            None if RBAC not available (allows request in dev mode),
            "unauthenticated" sentinel if JWT is missing/invalid
        """
        if not self._rbac_enabled():
            return None

        # JWT authentication required (secure)
        extractor = extract_user_from_request
        try:
            pkg = sys.modules.get("aragora.server.handlers.workflows")
            override = getattr(pkg, "extract_user_from_request", None) if pkg is not None else None
            if override is not None and override is not extract_user_from_request:
                extractor = override
        except Exception as e:
            logger.debug("Failed to resolve extract_user_from_request override: %s", e)

        jwt_context = extractor(handler)
        if not jwt_context.authenticated or not jwt_context.user_id:
            logger.warning(
                "workflows: JWT authentication required. Request rejected - no valid token."
            )
            # Return special sentinel to distinguish from RBAC-not-available
            return "unauthenticated"

        roles = {jwt_context.role} if jwt_context.role else {"member"}
        permissions: set[str] = set()
        for role in roles:
            permissions |= get_role_permissions(role, include_inherited=True)

        return AuthorizationContext(
            user_id=jwt_context.user_id,
            org_id=jwt_context.org_id,
            roles=roles,
            permissions=permissions,
        )

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        """Check if the request has the required permission.

        Returns None if allowed, or an error response if denied.
        If RBAC is not available, allows the request (development mode).
        """
        if not self._rbac_enabled():
            logger.debug("RBAC not available, allowing %s", permission_key)
            return None

        context = self._get_auth_context(handler)
        if context is None:
            return None  # RBAC not configured

        # Handle unauthenticated sentinel
        if context == "unauthenticated":
            return error_response("Authentication required. Please provide a valid JWT token.", 401)

        try:
            checker = check_permission
            try:
                pkg = sys.modules.get("aragora.server.handlers.workflows")
                override = getattr(pkg, "check_permission", None) if pkg is not None else None
                if override is not None and override is not check_permission:
                    checker = override
            except Exception as e:
                logger.debug("Failed to resolve check_permission override: %s", e)

            decision = checker(context, permission_key, resource_id)
            if not decision.allowed:
                logger.warning(
                    "Permission denied: %s for user %s: %s",
                    permission_key,
                    context.user_id,
                    decision.reason,
                )
                record_rbac_check(permission_key, granted=False)
                return error_response(f"Permission denied: {decision.reason}", 403)
            record_rbac_check(permission_key, granted=True)
        except PermissionDeniedError as e:
            logger.warning(
                "Permission denied: %s for user %s: %s", permission_key, context.user_id, e
            )
            record_rbac_check(permission_key, granted=False)
            return error_response(f"Permission denied: {str(e)}", 403)
        return None

    def _get_tenant_id(self, handler: Any, query_params: dict) -> str:
        """Extract tenant_id from auth context or query params."""
        if self._rbac_enabled():
            context = self._get_auth_context(handler)
            # Only use org_id from valid AuthorizationContext (not "unauthenticated" sentinel)
            if (
                context
                and context != "unauthenticated"
                and hasattr(context, "org_id")
                and context.org_id
            ):
                return context.org_id
        return get_string_param(query_params, "tenant_id", "default")

    @track_handler("workflows/main", method="GET")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests."""
        if not self.can_handle(path):
            return None

        if path.startswith("/api/v1/workflows/templates"):
            path = path.replace("/api/v1/workflows/templates", "/api/v1/workflow-templates", 1)
        if path.startswith("/api/v1/workflows/executions"):
            path = path.replace("/api/v1/workflows/executions", "/api/v1/workflow-executions", 1)

        # Authentication check
        auth_ctx = self._get_auth_context(handler) if self._rbac_enabled() else None
        if auth_ctx == "unauthenticated":
            return error_response("Authentication required", 401)

        # RBAC permission check for read operations
        if auth_ctx is not None and hasattr(auth_ctx, "permissions"):
            if (
                "workflow.read" not in auth_ctx.permissions
                and "workflow.*" not in auth_ctx.permissions
            ):
                logger.warning("User %s denied workflow.read permission", auth_ctx.user_id)
                return error_response("Permission denied: requires workflow.read", 403)

        # GET /api/workflow-executions
        if path == "/api/v1/workflow-executions":
            return self._handle_list_executions(query_params, handler)

        # GET /api/workflow-executions/{id}
        if path.startswith("/api/v1/workflow-executions/"):
            execution_id = path.split("/")[-1]
            if execution_id:
                return self._handle_get_execution(execution_id, query_params, handler)

        # GET /api/workflow-templates
        if path == "/api/v1/workflow-templates":
            return self._handle_list_templates(query_params, handler)

        # GET /api/workflow-approvals
        if path == "/api/v1/workflow-approvals":
            return self._handle_list_approvals(query_params, handler)

        # GET /api/workflows/{id}/versions
        if path.endswith("/versions"):
            workflow_id = self._extract_id(path, suffix="/versions")
            if workflow_id:
                return self._handle_get_versions(workflow_id, query_params, handler)

        # GET /api/workflows/{id}/status
        if path.endswith("/status"):
            workflow_id = self._extract_id(path, suffix="/status")
            if workflow_id:
                return self._handle_get_status(workflow_id, query_params, handler)

        # GET /api/workflows/{id}
        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_get_workflow(workflow_id, query_params, handler)

        # GET /api/workflows
        if path == "/api/v1/workflows":
            return self._handle_list_workflows(query_params, handler)

        return None

    @require_permission("workflows:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if not self.can_handle(path):
            return None

        if path.startswith("/api/v1/workflows/templates"):
            path = path.replace("/api/v1/workflows/templates", "/api/v1/workflow-templates", 1)
        if path.startswith("/api/v1/workflows/executions"):
            path = path.replace("/api/v1/workflows/executions", "/api/v1/workflow-executions", 1)

        # Authentication check
        auth_ctx = self._get_auth_context(handler)
        if auth_ctx == "unauthenticated":
            return error_response("Authentication required", 401)

        # RBAC permission check for write operations
        if auth_ctx is not None and hasattr(auth_ctx, "permissions"):
            # Execute requires workflow.execute, create requires workflow.create
            required_perm = (
                "workflows:execute"
                if path.endswith(("/execute", "/simulate"))
                else "workflows:create"
            )
            if path.endswith("/resolve"):
                required_perm = "workflows:approve"
            elif path.endswith("/restore"):
                required_perm = "workflows:update"
            if (
                required_perm not in auth_ctx.permissions
                and "workflow.*" not in auth_ctx.permissions
            ):
                logger.warning("User %s denied %s permission", auth_ctx.user_id, required_perm)
                return error_response(f"Permission denied: requires {required_perm}", 403)

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # POST /api/workflows/{id}/execute
        if path.endswith("/execute"):
            workflow_id = self._extract_id(path, suffix="/execute")
            if workflow_id:
                return self._handle_execute(workflow_id, body, query_params, handler)

        # POST /api/workflows/{id}/simulate
        if path.endswith("/simulate"):
            workflow_id = self._extract_id(path, suffix="/simulate")
            if workflow_id:
                return self._handle_simulate(workflow_id, body, query_params, handler)

        # POST /api/workflows/{id}/versions/{version}/restore
        if "/versions/" in path and path.endswith("/restore"):
            parts = path.strip("/").split("/")
            if len(parts) >= 7 and parts[4] == "versions" and parts[6] == "restore":
                workflow_id = parts[3]
                version = parts[5]
                return self._handle_restore_version(workflow_id, version, query_params, handler)

        # POST /api/workflow-approvals/{id}/resolve
        if "/workflow-approvals/" in path and path.endswith("/resolve"):
            parts = path.split("/")
            if len(parts) >= 4:
                request_id = parts[3]
                return self._handle_resolve_approval(request_id, body, query_params, handler)

        # POST /api/workflows (create)
        if path == "/api/v1/workflows":
            return self._handle_create_workflow(body, query_params, handler)

        return None

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle PATCH requests."""
        if not self.can_handle(path):
            return None

        # Authentication check
        auth_ctx = self._get_auth_context(handler)
        if auth_ctx == "unauthenticated":
            return error_response("Authentication required", 401)

        # RBAC permission check for update operations
        if auth_ctx is not None and hasattr(auth_ctx, "permissions"):
            if (
                "workflows:update" not in auth_ctx.permissions
                and "workflow.*" not in auth_ctx.permissions
            ):
                logger.warning("User %s denied workflow.update permission", auth_ctx.user_id)
                return error_response("Permission denied: requires workflow.update", 403)

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_update_workflow(workflow_id, body, query_params, handler)

        return None

    @require_permission("workflows:write")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle PUT requests (same as PATCH for workflows)."""
        return self.handle_patch(path, query_params, handler)

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        if not self.can_handle(path):
            return None

        if path.startswith("/api/v1/workflows/templates"):
            path = path.replace("/api/v1/workflows/templates", "/api/v1/workflow-templates", 1)
        if path.startswith("/api/v1/workflows/executions"):
            path = path.replace("/api/v1/workflows/executions", "/api/v1/workflow-executions", 1)

        # Authentication check
        auth_ctx = self._get_auth_context(handler)
        if auth_ctx == "unauthenticated":
            return error_response("Authentication required", 401)

        # RBAC permission check for delete operations
        if auth_ctx is not None and hasattr(auth_ctx, "permissions"):
            # Terminating executions requires execute permission, deleting workflows requires delete
            required_perm = (
                "workflows:execute" if "/workflow-executions/" in path else "workflows:delete"
            )
            if (
                required_perm not in auth_ctx.permissions
                and "workflow.*" not in auth_ctx.permissions
            ):
                logger.warning("User %s denied %s permission", auth_ctx.user_id, required_perm)
                return error_response(f"Permission denied: requires {required_perm}", 403)

        workflow_id = self._extract_id(path)
        if workflow_id:
            return self._handle_delete_workflow(workflow_id, query_params, handler)

        # DELETE /api/workflow-executions/{id} - terminate execution
        if "/workflow-executions/" in path:
            parts = path.strip("/").split("/")
            # Expected: api, v1, workflow-executions, {id}
            if len(parts) >= 4 and parts[2] == "workflow-executions":
                execution_id = parts[3]
                return self._handle_terminate_execution(execution_id, query_params, handler)

        return None

    # =========================================================================
    # Path Helpers
    # =========================================================================

    def _extract_id(self, path: str, suffix: str = "") -> str | None:
        """Extract workflow ID from path."""
        if suffix and path.endswith(suffix):
            path = path[: -len(suffix)]

        parts = path.strip("/").split("/")
        # /api/v1/workflows/{id} (4 parts: api, v1, workflows, id)
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "workflows":
            return parts[3]
        return None

    # =========================================================================
    # Request Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/v1/workflows",
        summary="List workflows",
        tags=["Workflows"],
    )
    def _handle_list_workflows(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflows."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:read"):
            return error

        coro = None
        try:
            limit, offset = self.get_pagination(query_params)
            tenant_id = self._get_tenant_id(handler, query_params)
            coro = list_workflows(
                tenant_id=tenant_id,
                category=get_string_param(query_params, "category", None),
                search=get_string_param(query_params, "search", None),
                limit=limit,
                offset=offset,
            )
            result = self._run_async_fn()(coro)
            return json_response(result)
        except OSError as e:
            if coro is not None and hasattr(coro, "close"):
                coro.close()
            logger.error("Storage error listing workflows: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            if coro is not None and hasattr(coro, "close"):
                coro.close()
            logger.error("Data error listing workflows: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflows/{workflow_id}",
        summary="Get workflow details",
        tags=["Workflows"],
    )
    def _handle_get_workflow(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = self._run_async_fn()(
                get_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if result:
                return json_response(result)
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error getting workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/workflows",
        summary="Create a new workflow",
        tags=["Workflows"],
    )
    def _handle_create_workflow(
        self, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:create"):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            # Get user_id from auth context if available
            auth_context = self._get_auth_context(handler) if self._rbac_enabled() else None
            created_by = (
                auth_context.user_id
                if auth_context and not isinstance(auth_context, str)
                else get_string_param(query_params, "user_id", "")
            )

            result = self._run_async_fn()(
                create_workflow(
                    body,
                    tenant_id=tenant_id,
                    created_by=created_by,
                )
            )
            return json_response(result, status=201)
        except ValueError as e:
            return error_response(str(e), 400)
        except OSError as e:
            logger.error("Storage error creating workflow: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error creating workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="PATCH",
        path="/api/v1/workflows/{workflow_id}",
        summary="Update a workflow",
        tags=["Workflows"],
    )
    def _handle_update_workflow(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle PATCH /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:update", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = self._run_async_fn()(
                update_workflow(
                    workflow_id,
                    body,
                    tenant_id=tenant_id,
                )
            )
            if result:
                return json_response(result)
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except ValueError as e:
            return error_response(str(e), 400)
        except OSError as e:
            logger.error("Storage error updating workflow: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error updating workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="DELETE",
        path="/api/v1/workflows/{workflow_id}",
        summary="Delete a workflow",
        tags=["Workflows"],
    )
    def _handle_delete_workflow(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/workflows/{id}."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:delete", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            deleted = self._run_async_fn()(
                delete_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if deleted:
                return json_response({"deleted": True, "id": workflow_id})
            return error_response(f"Workflow not found: {workflow_id}", 404)
        except OSError as e:
            logger.error("Storage error deleting workflow: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error deleting workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/workflows/{workflow_id}/execute",
        summary="Execute a workflow",
        tags=["Workflows"],
    )
    def _handle_execute(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows/{id}/execute."""
        # RBAC check - execute requires specific permission
        if error := self._check_permission(handler, "workflows:execute", workflow_id):
            return error

        try:
            inputs, input_error = _normalize_execute_inputs(body)
            if input_error:
                return error_response(input_error, 400)

            tenant_id = self._get_tenant_id(handler, query_params)
            auth_ctx = self._get_auth_context(handler) if self._rbac_enabled() else None
            user_id = None
            org_id = None
            if auth_ctx and auth_ctx != "unauthenticated":
                user_id = getattr(auth_ctx, "user_id", None)
                org_id = getattr(auth_ctx, "org_id", None)
            event_emitter = self.ctx.get("event_emitter") if isinstance(self.ctx, dict) else None
            result = self._run_async_fn()(
                execute_workflow(
                    workflow_id,
                    inputs=inputs,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    org_id=org_id,
                    event_emitter=event_emitter,
                )
            )
            return json_response(result)
        except ValueError as e:
            return error_response(str(e), 404)
        except (ConnectionError, TimeoutError) as e:
            logger.error("Connection error executing workflow: %s", e)
            return error_response("Execution service unavailable", 503)
        except OSError as e:
            logger.error("Storage error executing workflow: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error executing workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/workflows/{workflow_id}/simulate",
        summary="Simulate a workflow dry-run",
        tags=["Workflows"],
    )
    def _handle_simulate(
        self, workflow_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows/{id}/simulate (dry-run)."""
        # RBAC check - simulate only needs read permission
        if error := self._check_permission(handler, "workflows:read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            workflow_dict = self._run_async_fn()(
                get_workflow(
                    workflow_id,
                    tenant_id=tenant_id,
                )
            )
            if not workflow_dict:
                return error_response(f"Workflow not found: {workflow_id}", 404)

            workflow = WorkflowDefinition.from_dict(workflow_dict)
            is_valid, errors = workflow.validate()

            # Build execution plan
            plan = []
            visited = set()
            current = workflow.entry_step

            while current and current not in visited:
                step = workflow.get_step(current)
                if step:
                    plan.append(
                        {
                            "step_id": step.id,
                            "step_name": step.name,
                            "step_type": step.step_type,
                            "optional": step.optional,
                            "timeout": step.timeout_seconds,
                        }
                    )
                    visited.add(current)
                    current = step.next_steps[0] if step.next_steps else None
                else:
                    break

            return json_response(
                {
                    "workflow_id": workflow_id,
                    "is_valid": is_valid,
                    "validation_errors": errors,
                    "execution_plan": plan,
                    "estimated_steps": len(workflow.steps),
                }
            )

        except OSError as e:
            logger.error("Storage error simulating workflow: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error simulating workflow: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflows/{workflow_id}/status",
        summary="Get workflow execution status",
        tags=["Workflows"],
    )
    def _handle_get_status(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}/status."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:read", workflow_id):
            return error

        try:
            executions = self._run_async_fn()(list_executions(workflow_id=workflow_id, limit=1))
            if executions:
                return json_response(executions[0])
            return json_response(
                {
                    "workflow_id": workflow_id,
                    "status": "no_executions",
                    "message": "No executions found for this workflow",
                }
            )
        except OSError as e:
            logger.error("Storage error getting workflow status: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error getting workflow status: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflows/{workflow_id}/versions",
        summary="Get workflow version history",
        tags=["Workflows"],
    )
    def _handle_get_versions(
        self, workflow_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflows/{id}/versions."""
        # RBAC check
        if error := self._check_permission(handler, "workflows:read", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            versions = self._run_async_fn()(
                get_workflow_versions(
                    workflow_id,
                    tenant_id=tenant_id,
                    limit=get_int_param(query_params, "limit", 20),
                )
            )
            return json_response({"versions": versions, "workflow_id": workflow_id})
        except OSError as e:
            logger.error("Storage error getting workflow versions: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error getting workflow versions: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/workflows/{workflow_id}/versions/{version}/restore",
        summary="Restore workflow to a previous version",
        tags=["Workflows"],
    )
    def _handle_restore_version(
        self, workflow_id: str, version: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflows/{id}/versions/{version}/restore.

        Restores a workflow to a specific previous version.
        Requires workflows.update permission.
        """
        # RBAC check - restore requires update permission
        if error := self._check_permission(handler, "workflows:update", workflow_id):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            result = self._run_async_fn()(
                restore_workflow_version(workflow_id, version, tenant_id=tenant_id)
            )
            if result:
                logger.info("Restored workflow %s to version %s", workflow_id, version)
                return json_response({"restored": True, "workflow": result})
            return error_response(f"Version not found: {version}", 404)
        except OSError as e:
            logger.error("Storage error restoring workflow version: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error restoring workflow version: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="DELETE",
        path="/api/v1/workflow-executions/{execution_id}",
        summary="Terminate a running workflow execution",
        tags=["Workflows"],
    )
    def _handle_terminate_execution(
        self, execution_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle DELETE /api/workflow-executions/{id}.

        Terminates a running workflow execution.
        Requires workflows.execute permission.
        """
        # RBAC check - terminate requires execute permission
        if error := self._check_permission(handler, "workflows:execute", execution_id):
            return error

        try:
            success = self._run_async_fn()(terminate_execution(execution_id))
            if success:
                logger.info("Terminated execution %s", execution_id)
                return json_response({"terminated": True, "execution_id": execution_id})
            return error_response(f"Cannot terminate execution: {execution_id}", 400)
        except OSError as e:
            logger.error("Storage error terminating execution: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error terminating execution: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflow-templates",
        summary="List workflow templates",
        tags=["Workflows"],
    )
    def _handle_list_templates(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-templates."""
        # RBAC check - templates require read permission
        if error := self._check_permission(handler, "workflows:read"):
            return error

        try:
            templates = self._run_async_fn()(
                list_templates(
                    category=get_string_param(query_params, "category", None),
                )
            )
            return json_response({"templates": templates, "count": len(templates)})
        except OSError as e:
            logger.error("Storage error listing templates: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error listing templates: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflow-approvals",
        summary="List pending workflow approvals",
        tags=["Workflows"],
    )
    def _handle_list_approvals(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-approvals."""
        # RBAC check - approvals require read permission
        if error := self._check_permission(handler, "workflows:read"):
            return error

        try:
            tenant_id = self._get_tenant_id(handler, query_params)
            approvals = self._run_async_fn()(
                self._list_pending_approvals_fn()(
                    workflow_id=get_string_param(query_params, "workflow_id", None),
                    tenant_id=tenant_id,
                )
            )
            return json_response({"approvals": approvals, "count": len(approvals)})
        except OSError as e:
            logger.error("Storage error listing approvals: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error listing approvals: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflow-executions",
        summary="List workflow executions",
        tags=["Workflows"],
    )
    def _handle_list_executions(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/workflow-executions.

        Returns all workflow executions across all workflows, filtered by status.
        Used by the runtime monitoring dashboard.
        """
        # RBAC check
        if error := self._check_permission(handler, "workflows:read"):
            return error

        try:
            status_filter = get_string_param(query_params, "status", None)
            workflow_id = get_string_param(query_params, "workflow_id", None)
            limit = get_int_param(query_params, "limit", 50)
            tenant_id = self._get_tenant_id(handler, query_params)

            executions = self._run_async_fn()(
                list_executions(
                    workflow_id=workflow_id,
                    tenant_id=tenant_id,
                    limit=limit,
                )
            )

            # Apply status filter if provided
            if status_filter:
                executions = [e for e in executions if e.get("status") == status_filter]

            return json_response(
                {
                    "executions": executions,
                    "count": len(executions),
                }
            )
        except OSError as e:
            logger.error("Storage error listing executions: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error listing executions: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="GET",
        path="/api/v1/workflow-executions/{execution_id}",
        summary="Get workflow execution details",
        tags=["Workflows"],
    )
    def _handle_get_execution(
        self, execution_id: str, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle GET /api/workflow-executions/{id}.

        Returns a single execution's details and status.
        Used for polling workflow execution progress.
        """
        # RBAC check
        if error := self._check_permission(handler, "workflows:read", execution_id):
            return error

        try:
            execution = self._run_async_fn()(get_execution(execution_id))
            if not execution:
                return error_response(f"Execution not found: {execution_id}", 404)

            return json_response(execution)
        except OSError as e:
            logger.error("Storage error getting execution: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error getting execution: %s", e)
            return error_response("Internal data error", 500)

    @api_endpoint(
        method="POST",
        path="/api/v1/workflow-approvals/{request_id}/resolve",
        summary="Resolve a workflow approval request",
        tags=["Workflows"],
    )
    def _handle_resolve_approval(
        self, request_id: str, body: dict, query_params: dict, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/workflow-approvals/{id}/resolve."""
        # RBAC check - resolving approvals requires approve permission
        if error := self._check_permission(handler, "workflows:approve", request_id):
            return error

        try:
            # Get responder from auth context if available
            auth_context = self._get_auth_context(handler) if self._rbac_enabled() else None
            responder_id = (
                auth_context.user_id
                if auth_context and not isinstance(auth_context, str)
                else get_string_param(query_params, "user_id", "")
            )

            resolved = self._run_async_fn()(
                resolve_approval(
                    request_id,
                    status=body.get("status", "approved"),
                    responder_id=responder_id,
                    notes=body.get("notes", ""),
                    checklist_updates=body.get("checklist"),
                )
            )
            if resolved:
                return json_response({"resolved": True, "request_id": request_id})
            return error_response(f"Approval request not found: {request_id}", 404)
        except ValueError as e:
            return error_response(str(e), 400)
        except OSError as e:
            logger.error("Storage error resolving approval: %s", e)
            return error_response("Storage error", 503)
        except (KeyError, TypeError, AttributeError) as e:
            logger.error("Data error resolving approval: %s", e)
            return error_response("Internal data error", 500)


__all__ = [
    "WorkflowHandler",
    "WorkflowHandlers",
]

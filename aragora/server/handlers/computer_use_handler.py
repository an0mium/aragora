"""
Computer Use Handler - HTTP endpoints for computer use orchestration.

Provides API endpoints for:
- Task creation and execution
- Task status monitoring
- Action statistics
- Policy management

Routes:
    POST   /api/v1/computer-use/tasks           - Create and run a task
    GET    /api/v1/computer-use/tasks           - List recent tasks
    GET    /api/v1/computer-use/tasks/{id}      - Get task status
    POST   /api/v1/computer-use/tasks/{id}/cancel - Cancel a running task
    GET    /api/v1/computer-use/actions/stats   - Get action statistics
    GET    /api/v1/computer-use/policies        - List active policies
    POST   /api/v1/computer-use/policies        - Create a policy
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from aragora.server.extensions import get_extension_state
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.http_utils import run_async
from aragora.server.validation.query_params import safe_query_int

# RBAC imports
try:
    from aragora.rbac import AuthorizationContext, check_permission
    from aragora.billing.jwt_auth import extract_user_from_request

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    AuthorizationContext = None  # type: ignore[misc, no-redef]

# Computer Use imports
try:
    from aragora.computer_use import (
        ComputerUseOrchestrator,
        ComputerUseConfig,
        ComputerPolicy,
        create_default_computer_policy,
    )
    from aragora.computer_use.orchestrator import TaskResult, TaskStatus
    from aragora.computer_use.approval import ApprovalStatus

    COMPUTER_USE_AVAILABLE = True
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    ComputerUseOrchestrator = None  # type: ignore[misc, no-redef]
    TaskResult = None  # type: ignore[misc, no-redef]
    TaskStatus = None  # type: ignore[misc, no-redef]

logger = logging.getLogger(__name__)


class ComputerUseHandler(BaseHandler):
    """
    HTTP request handler for computer use API endpoints.

    Provides REST API for managing computer use tasks, monitoring
    execution, and configuring policies.
    """

    ROUTES = [
        "/api/v1/computer-use/tasks",
        "/api/v1/computer-use/tasks/*",
        "/api/v1/computer-use/actions",
        "/api/v1/computer-use/actions/*",
        "/api/v1/computer-use/policies",
        "/api/v1/computer-use/policies/*",
        "/api/v1/computer-use/approvals",
        "/api/v1/computer-use/approvals/*",
    ]

    def __init__(self, server_context):
        super().__init__(server_context)
        self._orchestrator: ComputerUseOrchestrator | None = None
        self._tasks: dict[str, dict[str, Any]] = {}  # In-memory task store
        self._action_stats: dict[str, dict[str, int]] = {}
        self._policies: dict[str, ComputerPolicy] = {}
        self._approval_workflow: Any | None = None

    def _get_orchestrator(self) -> ComputerUseOrchestrator | None:
        """Get or create computer use orchestrator."""
        if not COMPUTER_USE_AVAILABLE:
            return None
        state = get_extension_state()
        if state and state.computer_orchestrator:
            return state.computer_orchestrator
        if self._orchestrator is None:
            policy = create_default_computer_policy()
            config = ComputerUseConfig(max_steps=20, total_timeout_seconds=300)
            self._orchestrator = ComputerUseOrchestrator(policy=policy, config=config)
        return self._orchestrator

    def _get_approval_workflow(self) -> Any | None:
        """Get approval workflow for computer-use approvals."""
        state = get_extension_state()
        if state and state.computer_approval_workflow:
            return state.computer_approval_workflow
        return self._approval_workflow

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_auth_context(self, handler) -> AuthorizationContext | None:
        """Build AuthorizationContext from request."""
        if not RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        if not auth_ctx.is_authenticated:
            return None

        user = user_store.get_user_by_id(auth_ctx.user_id) if user_store else None
        roles = set([user.role]) if user and user.role else set()

        return AuthorizationContext(
            user_id=auth_ctx.user_id,
            roles=roles,
            org_id=auth_ctx.org_id,
        )

    def _check_rbac_permission(self, handler, permission_key: str) -> HandlerResult | None:
        """Check RBAC permission. Returns None if allowed, error response if denied."""
        if not RBAC_AVAILABLE:
            return None

        rbac_ctx = self._get_auth_context(handler)
        if not rbac_ctx:
            return error_response("Not authenticated", 401)

        decision = check_permission(rbac_ctx, permission_key)
        if not decision.allowed:
            logger.warning(f"RBAC denied: user={rbac_ctx.user_id} permission={permission_key}")
            return error_response(f"Permission denied: {decision.reason}", 403)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/computer-use/")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests."""
        if not self.can_handle(path):
            return None

        if not COMPUTER_USE_AVAILABLE:
            return error_response("Computer use module not available", 503)

        # GET /api/v1/computer-use/tasks
        if path == "/api/v1/computer-use/tasks":
            return self._handle_list_tasks(query_params, handler)

        # GET /api/v1/computer-use/tasks/{id}
        if path.startswith("/api/v1/computer-use/tasks/"):
            parts = path.strip("/").split("/")
            # parts = ["api", "v1", "computer-use", "tasks", task_id]
            if len(parts) >= 5:
                task_id = parts[4]
                return self._handle_get_task(task_id, handler)

        # GET /api/v1/computer-use/actions/stats
        if path == "/api/v1/computer-use/actions/stats":
            return self._handle_action_stats(handler)

        # GET /api/v1/computer-use/policies
        if path == "/api/v1/computer-use/policies":
            return self._handle_list_policies(handler)

        # GET /api/v1/computer-use/approvals
        if path == "/api/v1/computer-use/approvals":
            return self._handle_list_approvals(query_params, handler)

        # GET /api/v1/computer-use/approvals/{id}
        if path.startswith("/api/v1/computer-use/approvals/"):
            parts = path.strip("/").split("/")
            # parts = ["api", "v1", "computer-use", "approvals", request_id]
            if len(parts) >= 5:
                request_id = parts[4]
                return self._handle_get_approval(request_id, handler)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if not self.can_handle(path):
            return None

        if not COMPUTER_USE_AVAILABLE:
            return error_response("Computer use module not available", 503)

        # POST /api/v1/computer-use/tasks
        if path == "/api/v1/computer-use/tasks":
            return self._handle_create_task(handler)

        # POST /api/v1/computer-use/tasks/{id}/cancel
        if "/cancel" in path:
            parts = path.strip("/").split("/")
            # parts = ["api", "v1", "computer-use", "tasks", task_id, "cancel"]
            if len(parts) >= 6 and parts[5] == "cancel":
                task_id = parts[4]
                return self._handle_cancel_task(task_id, handler)

        # POST /api/v1/computer-use/policies
        if path == "/api/v1/computer-use/policies":
            return self._handle_create_policy(handler)

        # POST /api/v1/computer-use/approvals/{id}/approve|deny
        if path.startswith("/api/v1/computer-use/approvals/"):
            parts = path.strip("/").split("/")
            # parts = ["api", "v1", "computer-use", "approvals", request_id, action]
            if len(parts) >= 6:
                request_id = parts[4]
                action = parts[5]
                if action == "approve":
                    return self._handle_approve_approval(request_id, handler)
                if action == "deny":
                    return self._handle_deny_approval(request_id, handler)

        return None

    # =========================================================================
    # Task Handlers
    # =========================================================================

    @handle_errors("list tasks")
    def _handle_list_tasks(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/tasks."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:tasks:read"):
            return error

        limit = safe_query_int(query_params, "limit", default=20, min_val=1, max_val=100)
        status_filter = query_params.get("status")

        tasks = list(self._tasks.values())

        if status_filter:
            tasks = [t for t in tasks if t.get("status") == status_filter]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        tasks = tasks[:limit]

        return json_response(
            {
                "tasks": tasks,
                "total": len(tasks),
            }
        )

    @handle_errors("get task")
    def _handle_get_task(self, task_id: str, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/tasks/{id}."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:tasks:read"):
            return error

        task = self._tasks.get(task_id)
        if not task:
            return error_response(f"Task not found: {task_id}", 404)

        return json_response({"task": task})

    @rate_limit(requests_per_minute=10, limiter_name="computer_use_create")
    @handle_errors("create task")
    @log_request("create computer use task")
    def _handle_create_task(self, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/computer-use/tasks."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:tasks:create"):
            return error

        orchestrator = self._get_orchestrator()
        if not orchestrator:
            return error_response("Orchestrator not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        goal = body.get("goal")
        if not goal:
            return error_response("goal is required", 400)

        max_steps = body.get("max_steps", 10)
        dry_run = body.get("dry_run", False)

        # Create task record
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = {
            "task_id": task_id,
            "goal": goal,
            "max_steps": max_steps,
            "dry_run": dry_run,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "result": None,
        }
        self._tasks[task_id] = task

        # Execute task (in dry_run mode or mock mode for now)
        if dry_run:
            task["status"] = "completed"
            task["result"] = {
                "success": True,
                "message": "Dry run completed",
                "steps_taken": 0,
            }
        else:
            # Run the task asynchronously
            try:
                auth_ctx = self._get_auth_context(handler)
                metadata: dict[str, Any] = {}
                if auth_ctx:
                    metadata = {
                        "user_id": auth_ctx.user_id,
                        "tenant_id": auth_ctx.org_id,
                        "roles": list(auth_ctx.roles),
                    }
                result: TaskResult = run_async(
                    orchestrator.run_task(goal=goal, max_steps=max_steps, metadata=metadata)
                )
                is_success = result.status == TaskStatus.COMPLETED
                task["status"] = "completed" if is_success else "failed"
                task["result"] = {
                    "success": is_success,
                    "message": result.error or "",
                    "steps_taken": len(result.steps),
                }
                task["steps"] = [
                    {
                        "action": s.action.action_type.value,
                        "success": s.result.success,
                    }
                    for s in result.steps
                ]
            except Exception as e:
                task["status"] = "failed"
                task["result"] = {
                    "success": False,
                    "message": str(e),
                    "steps_taken": 0,
                }

        logger.info(f"Created computer use task: {task_id} - {goal}")

        return json_response(
            {
                "task_id": task_id,
                "status": task["status"],
                "message": "Task created successfully",
            },
            status=201,
        )

    @handle_errors("cancel task")
    @log_request("cancel computer use task")
    def _handle_cancel_task(self, task_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/computer-use/tasks/{id}/cancel."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:tasks:cancel"):
            return error

        task = self._tasks.get(task_id)
        if not task:
            return error_response(f"Task not found: {task_id}", 404)

        if task["status"] in ("completed", "failed", "cancelled"):
            return error_response(f"Task already {task['status']}", 400)

        task["status"] = "cancelled"
        task["cancelled_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Cancelled computer use task: {task_id}")

        return json_response({"message": "Task cancelled"})

    # =========================================================================
    # Action Handlers
    # =========================================================================

    @handle_errors("action stats")
    def _handle_action_stats(self, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/actions/stats."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:actions:read"):
            return error

        # Aggregate stats from completed tasks
        stats = {
            "click": {"total": 0, "success": 0, "failed": 0},
            "type": {"total": 0, "success": 0, "failed": 0},
            "screenshot": {"total": 0, "success": 0, "failed": 0},
            "scroll": {"total": 0, "success": 0, "failed": 0},
            "key": {"total": 0, "success": 0, "failed": 0},
        }

        for task in self._tasks.values():
            for step in task.get("steps", []):
                action = step.get("action", "").lower()
                if action in stats:
                    stats[action]["total"] += 1
                    if step.get("success"):
                        stats[action]["success"] += 1
                    else:
                        stats[action]["failed"] += 1

        return json_response({"stats": stats})

    # =========================================================================
    # Policy Handlers
    # =========================================================================

    @handle_errors("list policies")
    def _handle_list_policies(self, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/policies."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:policies:read"):
            return error

        # Return configured policies
        policies = [
            {
                "id": "default",
                "name": "Default Policy",
                "description": "Standard computer use policy",
                "allowed_actions": ["screenshot", "click", "type", "scroll", "key"],
                "blocked_domains": [],
            }
        ]

        for policy_id, policy in self._policies.items():
            policies.append(
                {
                    "id": policy_id,
                    "name": getattr(policy, "name", policy_id),
                    "description": getattr(policy, "description", ""),
                    "allowed_actions": getattr(policy, "allowed_actions", []),
                }
            )

        return json_response(
            {
                "policies": policies,
                "total": len(policies),
            }
        )

    @rate_limit(requests_per_minute=10, limiter_name="computer_use_policy")
    @handle_errors("create policy")
    @log_request("create computer use policy")
    def _handle_create_policy(self, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/computer-use/policies."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "computer_use:policies:create"):
            return error

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        policy_id = f"policy-{uuid.uuid4().hex[:8]}"

        # Create policy (simplified for now)
        policy = create_default_computer_policy()

        self._policies[policy_id] = policy

        logger.info(f"Created computer use policy: {policy_id} - {name}")

        return json_response(
            {
                "policy_id": policy_id,
                "message": "Policy created successfully",
            },
            status=201,
        )

    # =========================================================================
    # Approval Handlers
    # =========================================================================

    @handle_errors("list approvals")
    def _handle_list_approvals(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/approvals."""
        if error := self._check_rbac_permission(handler, "computer_use:admin"):
            return error

        workflow = self._get_approval_workflow()
        if not workflow:
            return error_response("Approval workflow not available", 503)

        status_filter = query_params.get("status")
        limit = safe_query_int(query_params, "limit", default=50, min_val=1, max_val=200)

        status = None
        if status_filter:
            try:
                status = ApprovalStatus(status_filter)
            except ValueError:
                return error_response(f"Invalid status: {status_filter}", 400)

        approvals = run_async(
            workflow.list_all(limit=limit, status=status)
            if status
            else workflow.list_all(limit=limit)
        )
        return json_response(
            {
                "approvals": [a.to_dict() for a in approvals],
                "count": len(approvals),
            }
        )

    @handle_errors("get approval")
    def _handle_get_approval(self, request_id: str, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/computer-use/approvals/{id}."""
        if error := self._check_rbac_permission(handler, "computer_use:admin"):
            return error

        workflow = self._get_approval_workflow()
        if not workflow:
            return error_response("Approval workflow not available", 503)

        approval = run_async(workflow.get_request(request_id))
        if not approval:
            return error_response(f"Approval request not found: {request_id}", 404)

        return json_response({"approval": approval.to_dict()})

    @handle_errors("approve approval")
    @log_request("approve computer use approval")
    def _handle_approve_approval(self, request_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/computer-use/approvals/{id}/approve."""
        if error := self._check_rbac_permission(handler, "computer_use:admin"):
            return error

        workflow = self._get_approval_workflow()
        if not workflow:
            return error_response("Approval workflow not available", 503)

        auth_ctx = self._get_auth_context(handler)
        approver_id = auth_ctx.user_id if auth_ctx else "system"
        body = self.read_json_body(handler) or {}
        reason = body.get("reason")

        approved = run_async(workflow.approve(request_id, approver_id, reason))
        if not approved:
            return error_response("Approval request not found or not pending", 404)

        return json_response({"approved": True, "request_id": request_id})

    @handle_errors("deny approval")
    @log_request("deny computer use approval")
    def _handle_deny_approval(self, request_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/computer-use/approvals/{id}/deny."""
        if error := self._check_rbac_permission(handler, "computer_use:admin"):
            return error

        workflow = self._get_approval_workflow()
        if not workflow:
            return error_response("Approval workflow not available", 503)

        auth_ctx = self._get_auth_context(handler)
        approver_id = auth_ctx.user_id if auth_ctx else "system"
        body = self.read_json_body(handler) or {}
        reason = body.get("reason")

        denied = run_async(workflow.deny(request_id, approver_id, reason))
        if not denied:
            return error_response("Approval request not found or not pending", 404)

        return json_response({"denied": True, "request_id": request_id})


__all__ = ["ComputerUseHandler"]

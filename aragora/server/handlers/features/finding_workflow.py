"""
Finding Workflow API Handler.

Enterprise workflow endpoints for managing audit findings.

Endpoints:
- PATCH /api/audit/findings/{id}/status        - Update finding workflow state
- PATCH /api/audit/findings/{id}/assign        - Assign finding to user
- POST  /api/audit/findings/{id}/unassign      - Remove assignment
- POST  /api/audit/findings/{id}/comments      - Add comment
- GET   /api/audit/findings/{id}/history       - Get workflow history
- PATCH /api/audit/findings/{id}/priority      - Set priority
- PATCH /api/audit/findings/{id}/due-date      - Set due date
- POST  /api/audit/findings/{id}/link          - Link to another finding
- POST  /api/audit/findings/{id}/duplicate     - Mark as duplicate
- POST  /api/audit/findings/bulk-action        - Bulk operations
- GET   /api/audit/findings/my-assignments     - Get current user's assignments
- GET   /api/audit/findings/overdue            - Get overdue findings
- GET   /api/audit/workflow/states             - Get valid workflow states
- GET   /api/audit/presets                     - Get available audit presets
- GET   /api/audit/types                       - Get registered audit types
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from aragora.server.handlers.base import BaseHandler
from aragora.storage.finding_workflow_store import get_finding_workflow_store
from aragora.rbac import (
    AuthorizationContext,
    check_permission,
    PermissionDeniedError,
)
from aragora.billing.auth import extract_user_from_request

logger = logging.getLogger(__name__)


class FindingWorkflowHandler(BaseHandler):
    """
    Handler for finding workflow management endpoints.

    Provides enterprise workflow capabilities including state transitions,
    assignment, comments, and bulk operations.
    """

    ROUTES = [
        "/api/audit/findings/{finding_id}/status",
        "/api/audit/findings/{finding_id}/assign",
        "/api/audit/findings/{finding_id}/unassign",
        "/api/audit/findings/{finding_id}/comments",
        "/api/audit/findings/{finding_id}/history",
        "/api/audit/findings/{finding_id}/priority",
        "/api/audit/findings/{finding_id}/due-date",
        "/api/audit/findings/{finding_id}/link",
        "/api/audit/findings/{finding_id}/duplicate",
        "/api/audit/findings/bulk-action",
        "/api/audit/findings/my-assignments",
        "/api/audit/findings/overdue",
        "/api/audit/workflow/states",
        "/api/audit/presets",
        "/api/audit/types",
    ]

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
        method = request.method
        path = str(request.path)

        # Parse finding_id from path if present
        finding_id = None
        if "/findings/" in path and not path.endswith("/bulk-action"):
            parts = path.split("/findings/")
            if len(parts) > 1:
                remaining = parts[1].split("/")
                if remaining[0] not in ("bulk-action", "my-assignments", "overdue"):
                    finding_id = remaining[0]

        # Route to handlers
        if path.endswith("/workflow/states"):
            return await self._get_workflow_states(request)
        elif path.endswith("/presets"):
            return await self._get_presets(request)
        elif path.endswith("/types"):
            return await self._get_audit_types(request)
        elif path.endswith("/bulk-action") and method == "POST":
            return await self._bulk_action(request)
        elif path.endswith("/my-assignments") and method == "GET":
            return await self._get_my_assignments(request)
        elif path.endswith("/overdue") and method == "GET":
            return await self._get_overdue(request)
        elif finding_id:
            if path.endswith("/status") and method == "PATCH":
                return await self._update_status(request, finding_id)
            elif path.endswith("/assign") and method == "PATCH":
                return await self._assign(request, finding_id)
            elif path.endswith("/unassign") and method == "POST":
                return await self._unassign(request, finding_id)
            elif path.endswith("/comments") and method == "POST":
                return await self._add_comment(request, finding_id)
            elif path.endswith("/comments") and method == "GET":
                return await self._get_comments(request, finding_id)
            elif path.endswith("/history") and method == "GET":
                return await self._get_history(request, finding_id)
            elif path.endswith("/priority") and method == "PATCH":
                return await self._set_priority(request, finding_id)
            elif path.endswith("/due-date") and method == "PATCH":
                return await self._set_due_date(request, finding_id)
            elif path.endswith("/link") and method == "POST":
                return await self._link_finding(request, finding_id)
            elif path.endswith("/duplicate") and method == "POST":
                return await self._mark_duplicate(request, finding_id)

        return self._error_response(404, "Endpoint not found")

    def _get_user_from_request(self, request: Any) -> tuple[str, str]:
        """Extract user ID and name from validated JWT token.

        Returns user info from JWT claims. Falls back to headers only
        for backward compatibility with internal service calls.
        """
        # Try JWT authentication first (secure)
        auth_context = extract_user_from_request(request)
        if auth_context.authenticated and auth_context.user_id:
            # Get display name from email or user_id
            display_name = auth_context.email or auth_context.user_id
            if "@" in display_name:
                display_name = display_name.split("@")[0]
            return auth_context.user_id, display_name

        # JWT authentication required - no header fallback for user identity
        # This prevents identity spoofing via X-User-ID header
        logger.warning(
            "finding_workflow: No authenticated user found. "
            "JWT token required for user identification."
        )
        return "anonymous", "Anonymous User"

    def _get_auth_context(self, request: Any) -> AuthorizationContext | None:
        """Build AuthorizationContext from validated JWT token.

        SECURITY: Only accepts JWT-based authentication. Header-based auth
        has been removed to prevent identity spoofing and privilege escalation.

        Returns:
            AuthorizationContext if JWT is valid, None if authentication failed
        """
        from aragora.rbac import get_role_permissions

        # JWT authentication required (secure)
        jwt_context = extract_user_from_request(request)
        if not jwt_context.authenticated or not jwt_context.user_id:
            logger.warning(
                "finding_workflow: JWT authentication required. "
                "Request rejected - no valid token."
            )
            return None

        # Build roles set from JWT role claim
        roles = {jwt_context.role} if jwt_context.role else {"member"}

        # Get resolved permissions for the roles
        permissions: set[str] = set()
        for role in roles:
            permissions |= get_role_permissions(role, include_inherited=True)

        return AuthorizationContext(
            user_id=jwt_context.user_id,
            org_id=jwt_context.org_id,
            roles=roles,
            permissions=permissions,
            ip_address=jwt_context.client_ip,
        )

    def _check_permission(
        self, request: Any, permission_key: str, resource_id: str | None = None
    ) -> dict[str, Any] | None:
        """
        Check if the request has the required permission.

        Returns None if allowed, or an error response dict if denied.
        """
        context = self._get_auth_context(request)

        # Authentication required
        if context is None:
            return self._error_response(
                401, "Authentication required. Please provide a valid JWT token."
            )

        try:
            decision = check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                logger.warning(
                    f"Permission denied: {permission_key} for user {context.user_id}: {decision.reason}"
                )
                return self._error_response(403, f"Permission denied: {decision.reason}")
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {permission_key} for user {context.user_id}: {e}")
            return self._error_response(403, f"Permission denied: {str(e)}")
        return None

    async def _get_or_create_workflow(self, finding_id: str) -> dict[str, Any]:
        """Get or create workflow data for a finding."""
        store = get_finding_workflow_store()
        workflow_dict = await store.get(finding_id)
        if workflow_dict is None:
            workflow_dict = {
                "finding_id": finding_id,
                "current_state": "open",
                "history": [],
                "assigned_to": None,
                "assigned_by": None,
                "assigned_at": None,
                "priority": 3,
                "due_date": None,
                "linked_findings": [],
                "parent_finding_id": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await store.save(workflow_dict)
        return workflow_dict

    async def _update_status(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Update finding workflow state.

        Body:
        {
            "status": "triaging" | "investigating" | "remediating" | "resolved" | ...,
            "comment": "Optional comment explaining transition"
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.update", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        new_status = body.get("status")
        if not new_status:
            return self._error_response(400, "status is required")

        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        # Get workflow and validate transition
        store = get_finding_workflow_store()
        try:
            from aragora.audit.findings.workflow import (
                FindingWorkflow,
                FindingWorkflowData,
                WorkflowState,
                InvalidTransitionError,
            )

            workflow_dict = await self._get_or_create_workflow(finding_id)
            data = FindingWorkflowData.from_dict(workflow_dict)
            workflow = FindingWorkflow(data)

            try:
                target_state = WorkflowState(new_status)
            except ValueError:
                return self._error_response(
                    400,
                    f"Invalid status: {new_status}. Valid values: {[s.value for s in WorkflowState]}",
                )

            if not workflow.can_transition_to(target_state):
                valid = workflow.get_valid_transitions()
                return self._error_response(
                    400,
                    f"Cannot transition from {workflow.state.value} to {new_status}. "
                    f"Valid transitions: {[s.value for s in valid]}",
                )

            event = workflow.transition_to(
                target_state,
                user_id=user_id,
                user_name=user_name,
                comment=comment,
            )

            # Persist workflow data
            await store.save(workflow.data.to_dict())

            logger.info(f"Finding {finding_id} transitioned to {new_status} by {user_id}")

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "previous_state": event.from_state.value if event.from_state else None,
                    "current_state": workflow.state.value,
                    "event": event.to_dict(),
                },
            )

        except InvalidTransitionError as e:
            return self._error_response(400, str(e))
        except ImportError:
            # Fallback without full workflow module
            workflow_dict = await self._get_or_create_workflow(finding_id)
            old_status = workflow_dict["current_state"]
            workflow_dict["current_state"] = new_status
            workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
            workflow_dict["history"].append(
                {
                    "id": str(uuid4()),
                    "event_type": "state_change",
                    "from_state": old_status,
                    "to_state": new_status,
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            await store.save(workflow_dict)

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "previous_state": old_status,
                    "current_state": new_status,
                },
            )

    async def _assign(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Assign finding to a user.

        Body:
        {
            "user_id": "user-456",
            "user_name": "John Doe",  // optional
            "comment": "Please review"  // optional
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.assign", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        assignee_id = body.get("user_id")
        if not assignee_id:
            return self._error_response(400, "user_id is required")

        _assignee_name = body.get("user_name", "")  # noqa: F841
        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        try:
            from aragora.audit.findings.workflow import FindingWorkflow, FindingWorkflowData

            workflow_dict = await self._get_or_create_workflow(finding_id)
            data = FindingWorkflowData.from_dict(workflow_dict)
            workflow = FindingWorkflow(data)

            event = workflow.assign(
                assignee_id,
                assigned_by=user_id,
                assigned_by_name=user_name,
                comment=comment,
            )

            await store.save(workflow.data.to_dict())

            logger.info(f"Finding {finding_id} assigned to {assignee_id} by {user_id}")

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "assigned_to": assignee_id,
                    "assigned_by": user_id,
                    "event": event.to_dict(),
                },
            )

        except ImportError:
            workflow_dict = await self._get_or_create_workflow(finding_id)
            workflow_dict["assigned_to"] = assignee_id
            workflow_dict["assigned_by"] = user_id
            workflow_dict["assigned_at"] = datetime.now(timezone.utc).isoformat()
            workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
            workflow_dict["history"].append(
                {
                    "id": str(uuid4()),
                    "event_type": "assignment",
                    "new_value": assignee_id,
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            await store.save(workflow_dict)

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "assigned_to": assignee_id,
                },
            )

    async def _unassign(self, request: Any, finding_id: str) -> dict[str, Any]:
        """Remove assignment from finding."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.assign", finding_id):
            return error

        comment = ""
        try:
            body = await self._parse_json_body(request)
            comment = body.get("comment", "")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Comment is optional, proceed with empty string if body parsing fails
            logger.debug(f"Optional comment body not provided or invalid: {e}")

        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        try:
            from aragora.audit.findings.workflow import FindingWorkflow, FindingWorkflowData

            workflow_dict = await self._get_or_create_workflow(finding_id)
            data = FindingWorkflowData.from_dict(workflow_dict)
            workflow = FindingWorkflow(data)

            event = workflow.unassign(
                user_id=user_id,
                user_name=user_name,
                comment=comment,
            )

            await store.save(workflow.data.to_dict())

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "event": event.to_dict(),
                },
            )

        except ImportError:
            workflow_dict = await self._get_or_create_workflow(finding_id)
            workflow_dict["assigned_to"] = None
            workflow_dict["assigned_by"] = None
            workflow_dict["assigned_at"] = None
            workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

            await store.save(workflow_dict)

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                },
            )

    async def _add_comment(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Add comment to finding.

        Body:
        {
            "comment": "This needs more investigation"
        }
        """
        # Check RBAC permission (read permission is sufficient for comments)
        if error := self._check_permission(request, "findings.read", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        comment = body.get("comment")
        if not comment:
            return self._error_response(400, "comment is required")

        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        try:
            from aragora.audit.findings.workflow import FindingWorkflow, FindingWorkflowData

            workflow_dict = await self._get_or_create_workflow(finding_id)
            data = FindingWorkflowData.from_dict(workflow_dict)
            workflow = FindingWorkflow(data)

            event = workflow.add_comment(
                comment,
                user_id=user_id,
                user_name=user_name,
            )

            await store.save(workflow.data.to_dict())

            return self._json_response(
                201,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "comment": event.to_dict(),
                },
            )

        except ImportError:
            workflow_dict = await self._get_or_create_workflow(finding_id)
            comment_event = {
                "id": str(uuid4()),
                "event_type": "comment",
                "user_id": user_id,
                "user_name": user_name,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            workflow_dict["history"].append(comment_event)
            workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

            await store.save(workflow_dict)

            return self._json_response(
                201,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "comment": comment_event,
                },
            )

    async def _get_comments(self, request: Any, finding_id: str) -> dict[str, Any]:
        """Get all comments for a finding."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read", finding_id):
            return error

        workflow_dict = await self._get_or_create_workflow(finding_id)
        comments = [e for e in workflow_dict.get("history", []) if e.get("event_type") == "comment"]

        return self._json_response(
            200,
            {
                "finding_id": finding_id,
                "comments": comments,
                "total": len(comments),
            },
        )

    async def _get_history(self, request: Any, finding_id: str) -> dict[str, Any]:
        """Get full workflow history for a finding."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read", finding_id):
            return error

        workflow_dict = await self._get_or_create_workflow(finding_id)

        return self._json_response(
            200,
            {
                "finding_id": finding_id,
                "current_state": workflow_dict.get("current_state"),
                "assigned_to": workflow_dict.get("assigned_to"),
                "priority": workflow_dict.get("priority"),
                "due_date": workflow_dict.get("due_date"),
                "history": workflow_dict.get("history", []),
            },
        )

    async def _set_priority(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Set finding priority.

        Body:
        {
            "priority": 1-5 (1=highest),
            "comment": "Optional reason"
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.update", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        priority = body.get("priority")
        if priority is None:
            return self._error_response(400, "priority is required")

        try:
            priority = int(priority)
            if not 1 <= priority <= 5:
                raise ValueError()
        except (ValueError, TypeError):
            return self._error_response(400, "priority must be integer 1-5")

        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        try:
            from aragora.audit.findings.workflow import FindingWorkflow, FindingWorkflowData

            workflow_dict = await self._get_or_create_workflow(finding_id)
            data = FindingWorkflowData.from_dict(workflow_dict)
            workflow = FindingWorkflow(data)

            event = workflow.set_priority(
                priority,
                user_id=user_id,
                user_name=user_name,
                comment=comment,
            )

            await store.save(workflow.data.to_dict())

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "priority": priority,
                    "event": event.to_dict(),
                },
            )

        except ImportError:
            workflow_dict = await self._get_or_create_workflow(finding_id)
            old_priority = workflow_dict.get("priority")
            workflow_dict["priority"] = priority
            workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
            workflow_dict["history"].append(
                {
                    "id": str(uuid4()),
                    "event_type": "priority_change",
                    "old_value": old_priority,
                    "new_value": priority,
                    "user_id": user_id,
                    "comment": comment,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            await store.save(workflow_dict)

            return self._json_response(
                200,
                {
                    "success": True,
                    "finding_id": finding_id,
                    "priority": priority,
                },
            )

    async def _set_due_date(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Set finding due date.

        Body:
        {
            "due_date": "2024-12-31T23:59:59Z" or null to clear,
            "comment": "Optional reason"
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.update", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        due_date_str = body.get("due_date")
        due_date = None
        if due_date_str:
            try:
                due_date = datetime.fromisoformat(due_date_str.replace("Z", "+00:00"))
            except ValueError:
                return self._error_response(400, "Invalid due_date format (use ISO 8601)")

        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        workflow_dict = await self._get_or_create_workflow(finding_id)
        old_due = workflow_dict.get("due_date")
        workflow_dict["due_date"] = due_date.isoformat() if due_date else None
        workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
        workflow_dict["history"].append(
            {
                "id": str(uuid4()),
                "event_type": "due_date_change",
                "old_value": old_due,
                "new_value": due_date.isoformat() if due_date else None,
                "user_id": user_id,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        await store.save(workflow_dict)

        return self._json_response(
            200,
            {
                "success": True,
                "finding_id": finding_id,
                "due_date": workflow_dict["due_date"],
            },
        )

    async def _link_finding(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Link this finding to another.

        Body:
        {
            "linked_finding_id": "finding-456",
            "comment": "Related issue"
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.update", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        linked_id = body.get("linked_finding_id")
        if not linked_id:
            return self._error_response(400, "linked_finding_id is required")

        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        workflow_dict = await self._get_or_create_workflow(finding_id)
        linked = workflow_dict.setdefault("linked_findings", [])
        if linked_id not in linked:
            linked.append(linked_id)

        workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
        workflow_dict["history"].append(
            {
                "id": str(uuid4()),
                "event_type": "linked",
                "new_value": linked_id,
                "user_id": user_id,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        await store.save(workflow_dict)

        return self._json_response(
            200,
            {
                "success": True,
                "finding_id": finding_id,
                "linked_findings": linked,
            },
        )

    async def _mark_duplicate(self, request: Any, finding_id: str) -> dict[str, Any]:
        """
        Mark finding as duplicate of another.

        Body:
        {
            "parent_finding_id": "finding-original",
            "comment": "Same issue as parent"
        }
        """
        # Check RBAC permission
        if error := self._check_permission(request, "findings.update", finding_id):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        parent_id = body.get("parent_finding_id")
        if not parent_id:
            return self._error_response(400, "parent_finding_id is required")

        comment = body.get("comment", f"Duplicate of {parent_id}")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        workflow_dict = await self._get_or_create_workflow(finding_id)
        old_state = workflow_dict.get("current_state", "open")
        workflow_dict["parent_finding_id"] = parent_id
        workflow_dict["current_state"] = "duplicate"
        workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        linked = workflow_dict.setdefault("linked_findings", [])
        if parent_id not in linked:
            linked.append(parent_id)

        workflow_dict["history"].append(
            {
                "id": str(uuid4()),
                "event_type": "state_change",
                "from_state": old_state,
                "to_state": "duplicate",
                "user_id": user_id,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        await store.save(workflow_dict)

        return self._json_response(
            200,
            {
                "success": True,
                "finding_id": finding_id,
                "parent_finding_id": parent_id,
                "current_state": "duplicate",
            },
        )

    async def _bulk_action(self, request: Any) -> dict[str, Any]:
        """
        Perform bulk actions on multiple findings.

        Body:
        {
            "finding_ids": ["f1", "f2", "f3"],
            "action": "assign" | "update_status" | "set_priority" | "unassign",
            "params": {
                "user_id": "...",  // for assign
                "status": "...",   // for update_status
                "priority": 1-5    // for set_priority
            },
            "comment": "Bulk update reason"
        }
        """
        # Check RBAC permission for bulk operations
        if error := self._check_permission(request, "findings.bulk"):
            return error

        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        finding_ids = body.get("finding_ids", [])
        if not finding_ids:
            return self._error_response(400, "finding_ids is required")

        action = body.get("action")
        if not action:
            return self._error_response(400, "action is required")

        params = body.get("params", {})
        comment = body.get("comment", "")
        user_id, user_name = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        results: dict[str, list[str]] = {"success": [], "failed": []}

        for fid in finding_ids:
            try:
                workflow_dict = await self._get_or_create_workflow(fid)

                if action == "assign":
                    assignee = params.get("user_id")
                    if assignee:
                        workflow_dict["assigned_to"] = assignee
                        workflow_dict["assigned_by"] = user_id
                        workflow_dict["assigned_at"] = datetime.now(timezone.utc).isoformat()

                elif action == "update_status":
                    new_status = params.get("status")
                    if new_status:
                        workflow_dict["current_state"] = new_status

                elif action == "set_priority":
                    priority = params.get("priority")
                    if priority and 1 <= int(priority) <= 5:
                        workflow_dict["priority"] = int(priority)

                elif action == "unassign":
                    workflow_dict["assigned_to"] = None
                    workflow_dict["assigned_by"] = None
                    workflow_dict["assigned_at"] = None

                workflow_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
                workflow_dict["history"].append(
                    {
                        "id": str(uuid4()),
                        "event_type": f"bulk_{action}",
                        "user_id": user_id,
                        "comment": comment,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                await store.save(workflow_dict)

                results["success"].append(fid)

            except Exception as e:  # noqa: BLE001 - Bulk operations should continue despite individual failures
                results["failed"].append({"finding_id": fid, "error": str(e)})

        return self._json_response(
            200,
            {
                "action": action,
                "total": len(finding_ids),
                "success_count": len(results["success"]),
                "failed_count": len(results["failed"]),
                "results": results,
            },
        )

    async def _get_my_assignments(self, request: Any) -> dict[str, Any]:
        """Get findings assigned to the current user."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read"):
            return error

        user_id, _ = self._get_user_from_request(request)

        store = get_finding_workflow_store()
        assignments = await store.list_by_assignee(user_id)

        # Sort by priority (1=highest first), then by due date
        assignments.sort(
            key=lambda w: (
                w.get("priority", 3),
                w.get("due_date") or "9999",
            )
        )

        return self._json_response(
            200,
            {
                "user_id": user_id,
                "findings": assignments,
                "total": len(assignments),
            },
        )

    async def _get_overdue(self, request: Any) -> dict[str, Any]:
        """Get all overdue findings."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read"):
            return error

        store = get_finding_workflow_store()
        overdue = await store.list_overdue()

        # Sort by due date (oldest first)
        overdue.sort(key=lambda w: w.get("due_date", ""))

        return self._json_response(
            200,
            {
                "findings": overdue,
                "total": len(overdue),
            },
        )

    async def _get_workflow_states(self, request: Any) -> dict[str, Any]:
        """Get valid workflow states and transitions."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read"):
            return error

        try:
            from aragora.audit.findings.workflow import (
                WorkflowState,
                VALID_TRANSITIONS,
                get_workflow_diagram,
            )

            states = []
            for state in WorkflowState:
                transitions = VALID_TRANSITIONS.get(state, set())
                states.append(
                    {
                        "id": state.value,
                        "valid_transitions": [t.value for t in transitions],
                        "is_terminal": state.value
                        in ("resolved", "false_positive", "accepted_risk", "duplicate"),
                    }
                )

            return self._json_response(
                200,
                {
                    "states": states,
                    "diagram": get_workflow_diagram(),
                },
            )

        except ImportError:
            # Fallback
            return self._json_response(
                200,
                {
                    "states": [
                        {
                            "id": "open",
                            "valid_transitions": ["triaging", "investigating", "false_positive"],
                        },
                        {
                            "id": "triaging",
                            "valid_transitions": [
                                "investigating",
                                "false_positive",
                                "accepted_risk",
                            ],
                        },
                        {
                            "id": "investigating",
                            "valid_transitions": ["remediating", "false_positive", "accepted_risk"],
                        },
                        {"id": "remediating", "valid_transitions": ["resolved", "investigating"]},
                        {"id": "resolved", "valid_transitions": ["open"], "is_terminal": True},
                        {
                            "id": "false_positive",
                            "valid_transitions": ["open"],
                            "is_terminal": True,
                        },
                        {"id": "accepted_risk", "valid_transitions": ["open"], "is_terminal": True},
                        {"id": "duplicate", "valid_transitions": ["open"], "is_terminal": True},
                    ],
                },
            )

    async def _get_presets(self, request: Any) -> dict[str, Any]:
        """Get available audit presets."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read"):
            return error

        try:
            from aragora.audit.registry import audit_registry

            audit_registry.auto_discover()
            presets = audit_registry.list_presets()

            return self._json_response(
                200,
                {
                    "presets": [
                        {
                            "name": p.name,
                            "description": p.description,
                            "audit_types": p.audit_types,
                            "consensus_threshold": p.consensus_threshold,
                            "custom_rules_count": len(p.custom_rules),
                        }
                        for p in presets
                    ],
                    "total": len(presets),
                },
            )

        except ImportError:
            return self._json_response(200, {"presets": [], "total": 0})

    async def _get_audit_types(self, request: Any) -> dict[str, Any]:
        """Get registered audit types."""
        # Check RBAC permission
        if error := self._check_permission(request, "findings.read"):
            return error

        try:
            from aragora.audit.registry import audit_registry

            audit_registry.auto_discover()
            types = audit_registry.list_audit_types()

            return self._json_response(
                200,
                {
                    "audit_types": [
                        {
                            "id": t.id,
                            "display_name": t.display_name,
                            "description": t.description,
                            "version": t.version,
                            "capabilities": t.capabilities,
                        }
                        for t in types
                    ],
                    "total": len(types),
                },
            )

        except ImportError:
            return self._json_response(
                200,
                {
                    "audit_types": [
                        {"id": "security", "display_name": "Security Analysis"},
                        {"id": "compliance", "display_name": "Compliance Check"},
                        {"id": "consistency", "display_name": "Consistency Analysis"},
                        {"id": "quality", "display_name": "Quality Assessment"},
                    ],
                    "total": 4,
                },
            )

    async def _parse_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            return await request.json()
        elif hasattr(request, "body"):
            body = await request.body()
            return json.loads(body)
        return {}

    def _json_response(self, status: int, data: Any) -> dict[str, Any]:
        """Create a JSON response."""
        return {
            "status": status,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(data, default=str),
        }

    def _error_response(self, status: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return self._json_response(status, {"error": message})


__all__ = ["FindingWorkflowHandler"]

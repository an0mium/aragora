"""
Retention policy handlers for workspace management.

Extracted from WorkspaceHandler to support package structure decomposition.

Handlers:
- handle_list_policies: List retention policies
- handle_create_policy: Create a retention policy
- handle_get_policy: Get a retention policy
- handle_update_policy: Update a retention policy
- handle_delete_policy: Delete a retention policy
- handle_execute_policy: Execute a retention policy
- handle_expiring_items: Get items expiring soon
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aragora.billing.jwt_auth import extract_user_from_request
from aragora.privacy import (
    AuditAction,
    AuditOutcome,
    RetentionAction,
)
from aragora.privacy.audit_log import Actor, Resource
from aragora.protocols import HTTPRequestHandler
from aragora.server.handlers.openapi_decorator import api_endpoint

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from ..utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from .handler import WorkspaceHandler


@api_endpoint(
    method="GET",
    path="/api/v1/retention/policies",
    summary="List retention policies",
    tags=["Retention"],
)
@handle_errors("list retention policies")
def handle_list_policies(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """List retention policies."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    workspace_id = query_params.get("workspace_id")
    manager = handler_instance._get_retention_manager()

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


@api_endpoint(
    method="POST",
    path="/api/v1/retention/policies",
    summary="Create a retention policy",
    tags=["Retention"],
)
@rate_limit(requests_per_minute=20, limiter_name="retention_policy")
@handle_errors("create retention policy")
@log_request("create retention policy")
def handle_create_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
) -> HandlerResult:
    """Create a retention policy."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "retention.policies.create", auth_ctx
    )
    if rbac_error:
        return rbac_error

    body = handler_instance.read_json_body(handler)
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

    manager = handler_instance._get_retention_manager()
    policy = manager.create_policy(
        name=name,
        retention_days=retention_days,
        action=action,
        workspace_ids=workspace_ids,
        description=description,
        applies_to=applies_to,
    )

    # Log to audit
    audit_log = handler_instance._get_audit_log()
    handler_instance._run_async(
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


@api_endpoint(
    method="GET",
    path="/api/v1/retention/policies/{policy_id}",
    summary="Get a retention policy",
    tags=["Retention"],
)
@handle_errors("get retention policy")
def handle_get_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    policy_id: str,
) -> HandlerResult:
    """Get a retention policy."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    manager = handler_instance._get_retention_manager()
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


@api_endpoint(
    method="PUT",
    path="/api/v1/retention/policies/{policy_id}",
    summary="Update a retention policy",
    tags=["Retention"],
)
@rate_limit(requests_per_minute=20, limiter_name="retention_policy")
@handle_errors("update retention policy")
@log_request("update retention policy")
def handle_update_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    policy_id: str,
) -> HandlerResult:
    """Update a retention policy."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "retention.policies.update", auth_ctx
    )
    if rbac_error:
        return rbac_error

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    manager = handler_instance._get_retention_manager()

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
    audit_log = handler_instance._get_audit_log()
    handler_instance._run_async(
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


@api_endpoint(
    method="DELETE",
    path="/api/v1/retention/policies/{policy_id}",
    summary="Delete a retention policy",
    tags=["Retention"],
)
@rate_limit(requests_per_minute=10, limiter_name="retention_policy")
@handle_errors("delete retention policy")
@log_request("delete retention policy")
def handle_delete_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    policy_id: str,
) -> HandlerResult:
    """Delete a retention policy."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "retention.policies.delete", auth_ctx
    )
    if rbac_error:
        return rbac_error

    manager = handler_instance._get_retention_manager()
    manager.delete_policy(policy_id)

    # Log to audit
    audit_log = handler_instance._get_audit_log()
    handler_instance._run_async(
        audit_log.log(
            action=AuditAction.MODIFY_POLICY,
            actor=Actor(id=auth_ctx.user_id, type="user"),
            resource=Resource(id=policy_id, type="retention_policy"),
            outcome=AuditOutcome.SUCCESS,
            details={"operation": "delete"},
        )
    )

    return json_response({"message": "Policy deleted successfully"})


@api_endpoint(
    method="POST",
    path="/api/v1/retention/policies/{policy_id}/execute",
    summary="Execute a retention policy",
    tags=["Retention"],
)
@rate_limit(requests_per_minute=5, limiter_name="retention_execute")
@handle_errors("execute retention policy")
@log_request("execute retention policy")
def handle_execute_policy(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    policy_id: str,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Execute a retention policy."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "retention.policies.execute", auth_ctx
    )
    if rbac_error:
        return rbac_error

    dry_run = query_params.get("dry_run", "false").lower() == "true"
    manager = handler_instance._get_retention_manager()

    try:
        report = handler_instance._run_async(manager.execute_policy(policy_id, dry_run=dry_run))
    except ValueError as e:
        return error_response(str(e), 404)

    # Log to audit
    audit_log = handler_instance._get_audit_log()
    handler_instance._run_async(
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


@api_endpoint(
    method="GET",
    path="/api/v1/retention/expiring",
    summary="Get items expiring soon",
    tags=["Retention"],
)
@handle_errors("get expiring items")
def handle_expiring_items(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Get items expiring soon."""
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    workspace_id = query_params.get("workspace_id")
    days = int(query_params.get("days", "14"))

    manager = handler_instance._get_retention_manager()
    expiring = handler_instance._run_async(
        manager.check_expiring_soon(workspace_id=workspace_id, days=days)
    )

    return json_response({"expiring": expiring, "total": len(expiring), "days_ahead": days})

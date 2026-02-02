"""
Audit Log Handlers for Workspace Package.

This module contains handlers for audit log operations:
- handle_query_audit: Query audit log entries
- handle_audit_report: Generate compliance audit report
- handle_verify_integrity: Verify audit log integrity
- handle_actor_history: Get all actions by a specific actor
- handle_resource_history: Get all actions on a specific resource
- handle_denied_access: Get denied access attempts

Stability: STABLE
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aragora.billing.jwt_auth import extract_user_from_request
from aragora.privacy import (
    AuditAction,
    AuditOutcome,
)
from aragora.privacy.audit_log import Actor, Resource
from aragora.protocols import HTTPRequestHandler
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.validation.query_params import safe_query_int

if TYPE_CHECKING:
    from aragora.server.handlers.workspace import WorkspaceHandler

logger = logging.getLogger(__name__)


@api_endpoint(
    method="GET",
    path="/api/v1/audit/entries",
    summary="Query audit log entries",
    tags=["Audit"],
)
@handle_errors("query audit entries")
def handle_query_audit(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Query audit log entries.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        query_params: Query parameters for filtering

    Returns:
        HandlerResult with matching audit entries
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(handler, "audit.query", auth_ctx)
    if rbac_error:
        return rbac_error

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

    audit_log = handler_instance._get_audit_log()
    entries = handler_instance._run_async(
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


@api_endpoint(
    method="GET",
    path="/api/v1/audit/report",
    summary="Generate compliance audit report",
    tags=["Audit"],
)
@handle_errors("generate audit report")
def handle_audit_report(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Generate compliance report.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        query_params: Query parameters for report generation

    Returns:
        HandlerResult with the generated report
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(handler, "audit.report", auth_ctx)
    if rbac_error:
        return rbac_error

    start_date = None
    end_date = None
    if "start_date" in query_params:
        start_date = datetime.fromisoformat(query_params["start_date"])
    if "end_date" in query_params:
        end_date = datetime.fromisoformat(query_params["end_date"])

    workspace_id = query_params.get("workspace_id")
    format_type = query_params.get("format", "json")

    audit_log = handler_instance._get_audit_log()
    report = handler_instance._run_async(
        audit_log.generate_compliance_report(
            start_date=start_date,
            end_date=end_date,
            workspace_id=workspace_id,
            format=format_type,
        )
    )

    # Log report generation
    handler_instance._run_async(
        audit_log.log(
            action=AuditAction.GENERATE_REPORT,
            actor=Actor(id=auth_ctx.user_id, type="user"),
            resource=Resource(id=report["report_id"], type="compliance_report"),
            outcome=AuditOutcome.SUCCESS,
            details={"workspace_id": workspace_id},
        )
    )

    return json_response({"report": report})


@api_endpoint(
    method="GET",
    path="/api/v1/audit/verify",
    summary="Verify audit log integrity",
    tags=["Audit"],
)
@handle_errors("verify audit integrity")
def handle_verify_integrity(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Verify audit log integrity.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        query_params: Query parameters for integrity verification

    Returns:
        HandlerResult with integrity verification results
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "audit.verify_integrity", auth_ctx
    )
    if rbac_error:
        return rbac_error

    start_date = None
    end_date = None
    if "start_date" in query_params:
        start_date = datetime.fromisoformat(query_params["start_date"])
    if "end_date" in query_params:
        end_date = datetime.fromisoformat(query_params["end_date"])

    audit_log = handler_instance._get_audit_log()
    is_valid, errors = handler_instance._run_async(
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


@api_endpoint(
    method="GET",
    path="/api/v1/audit/actor/{actor_id}/history",
    summary="Get all actions by a specific actor",
    tags=["Audit"],
)
@handle_errors("get actor history")
def handle_actor_history(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    actor_id: str,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Get all actions by a specific actor.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        actor_id: The actor ID to get history for
        query_params: Query parameters (e.g., days)

    Returns:
        HandlerResult with actor's action history
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(handler, "audit.actor_history", auth_ctx)
    if rbac_error:
        return rbac_error

    days = int(query_params.get("days", "30"))
    audit_log = handler_instance._get_audit_log()
    entries = handler_instance._run_async(audit_log.get_actor_history(actor_id=actor_id, days=days))

    return json_response(
        {
            "actor_id": actor_id,
            "entries": [e.to_dict() for e in entries],
            "total": len(entries),
            "days": days,
        }
    )


@api_endpoint(
    method="GET",
    path="/api/v1/audit/resource/{resource_id}/history",
    summary="Get all actions on a specific resource",
    tags=["Audit"],
)
@handle_errors("get resource history")
def handle_resource_history(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    resource_id: str,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Get all actions on a specific resource.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        resource_id: The resource ID to get history for
        query_params: Query parameters (e.g., days)

    Returns:
        HandlerResult with resource's action history
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(
        handler, "audit.resource_history", auth_ctx
    )
    if rbac_error:
        return rbac_error

    days = safe_query_int(query_params, "days", default=30, min_val=1, max_val=365)
    audit_log = handler_instance._get_audit_log()
    entries = handler_instance._run_async(
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


@api_endpoint(
    method="GET",
    path="/api/v1/audit/denied",
    summary="Get denied access attempts",
    tags=["Audit"],
)
@handle_errors("get denied access attempts")
def handle_denied_access(
    handler_instance: "WorkspaceHandler",
    handler: HTTPRequestHandler,
    query_params: dict[str, Any],
) -> HandlerResult:
    """Get all denied access attempts.

    Args:
        handler_instance: The WorkspaceHandler instance
        handler: The HTTP request handler
        query_params: Query parameters (e.g., days)

    Returns:
        HandlerResult with denied access attempts
    """
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)

    # RBAC permission check
    rbac_error = handler_instance._check_rbac_permission(handler, "audit.denied_access", auth_ctx)
    if rbac_error:
        return rbac_error

    days = safe_query_int(query_params, "days", default=7, min_val=1, max_val=365)
    audit_log = handler_instance._get_audit_log()
    entries = handler_instance._run_async(audit_log.get_denied_access_attempts(days=days))

    return json_response(
        {
            "denied_attempts": [e.to_dict() for e in entries],
            "total": len(entries),
            "days": days,
        }
    )


__all__ = [
    "handle_query_audit",
    "handle_audit_report",
    "handle_verify_integrity",
    "handle_actor_history",
    "handle_resource_history",
    "handle_denied_access",
]

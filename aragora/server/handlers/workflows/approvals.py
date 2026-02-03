"""
Human approval operations for workflows.

Provides operations for managing human-in-the-loop approval checkpoints.
"""

from __future__ import annotations

from typing import Any, Optional


async def list_pending_approvals(
    workflow_id: str | None = None,
    tenant_id: str = "default",
) -> list[dict[str, Any]]:
    """List pending human approvals."""
    from aragora.server.handlers import workflows as workflows_module

    if workflows_module.get_pending_approvals is None:
        return []

    approvals = workflows_module.get_pending_approvals(workflow_id)
    return [a.to_dict() for a in approvals]


async def resolve_approval(
    request_id: str,
    status: str,
    responder_id: str,
    notes: str = "",
    checklist_updates: Optional[dict[str, bool]] = None,
) -> bool:
    """Resolve a human approval request."""
    from aragora.server.handlers import workflows as workflows_module

    try:
        approval_status = workflows_module.ApprovalStatus[status.upper()]
    except KeyError:
        raise ValueError(f"Invalid status: {status}")

    if workflows_module._resolve is None:
        raise RuntimeError("Approval resolution is unavailable")

    return workflows_module._resolve(
        request_id,
        approval_status,
        responder_id,
        notes,
        checklist_updates,
    )


async def get_approval(request_id: str) -> Optional[dict[str, Any]]:
    """Get an approval request by ID."""
    from aragora.server.handlers import workflows as workflows_module

    if workflows_module.get_approval_request is None:
        return None

    approval = workflows_module.get_approval_request(request_id)
    return approval.to_dict() if approval else None


__all__ = [
    "list_pending_approvals",
    "resolve_approval",
    "get_approval",
]

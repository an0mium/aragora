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
    from aragora.workflow.nodes.human_checkpoint import get_pending_approvals

    approvals = get_pending_approvals(workflow_id)
    return [a.to_dict() for a in approvals]


async def resolve_approval(
    request_id: str,
    status: str,
    responder_id: str,
    notes: str = "",
    checklist_updates: Optional[dict[str, bool]] = None,
) -> bool:
    """Resolve a human approval request."""
    from aragora.workflow.nodes.human_checkpoint import (
        resolve_approval as _resolve,
        ApprovalStatus,
    )

    try:
        approval_status = ApprovalStatus[status.upper()]
    except KeyError:
        raise ValueError(f"Invalid status: {status}")

    return _resolve(request_id, approval_status, responder_id, notes, checklist_updates)


async def get_approval(request_id: str) -> Optional[dict[str, Any]]:
    """Get an approval request by ID."""
    from aragora.workflow.nodes.human_checkpoint import get_approval_request

    approval = get_approval_request(request_id)
    return approval.to_dict() if approval else None


__all__ = [
    "list_pending_approvals",
    "resolve_approval",
    "get_approval",
]

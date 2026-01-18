"""
Human Checkpoint Step for workflow approval gates.

Provides human-in-the-loop approval for critical workflow steps with:
- Configurable approval checklists
- Timeout with escalation
- Approval/rejection branching
- Audit logging
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from aragora.workflow.safe_eval import SafeEvalError, safe_eval_bool
from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of a human approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


@dataclass
class ChecklistItem:
    """An item in the approval checklist."""

    id: str
    label: str
    required: bool = True
    checked: bool = False
    notes: str = ""


@dataclass
class ApprovalRequest:
    """A human approval request."""

    id: str
    workflow_id: str
    step_id: str
    title: str
    description: str
    checklist: List[ChecklistItem]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    responded_at: Optional[datetime] = None
    responder_id: Optional[str] = None
    responder_notes: str = ""
    timeout_seconds: float = 3600.0
    escalation_emails: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "checklist": [
                {"id": c.id, "label": c.label, "required": c.required, "checked": c.checked, "notes": c.notes}
                for c in self.checklist
            ],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "responder_id": self.responder_id,
            "responder_notes": self.responder_notes,
            "timeout_seconds": self.timeout_seconds,
            "escalation_emails": self.escalation_emails,
        }


# In-memory store for approval requests (will be persisted in production)
_pending_approvals: Dict[str, ApprovalRequest] = {}


class HumanCheckpointStep(BaseStep):
    """
    Human checkpoint step for workflow approval gates.

    Config options:
        title: str - Title of the approval request
        description: str - Detailed description for the approver
        checklist: List[dict] - Checklist items [{label, required}]
        timeout_seconds: float - Timeout before escalation (default 3600)
        escalation_emails: List[str] - Emails to notify on timeout
        auto_approve_if: str - Python expression for auto-approval
        require_all_checklist: bool - Require all checklist items (default True)

    Usage:
        step = HumanCheckpointStep(
            name="Legal Review",
            config={
                "title": "Legal Review Required",
                "description": "Please review the contract terms",
                "checklist": [
                    {"label": "Verified compliance terms", "required": True},
                    {"label": "Checked indemnification clause", "required": True},
                ],
                "timeout_seconds": 7200,
                "escalation_emails": ["legal-lead@company.com"],
            }
        )
    """

    # Callback for notifying about new approval requests
    on_approval_requested: Optional[Callable[[ApprovalRequest], None]] = None

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the human checkpoint step."""
        config = {**self._config, **context.current_step_config}

        # Build checklist from config
        checklist_config = config.get("checklist", [])
        checklist = [
            ChecklistItem(
                id=f"item_{i}",
                label=item.get("label", f"Item {i+1}"),
                required=item.get("required", True),
            )
            for i, item in enumerate(checklist_config)
        ]

        # Check auto-approval condition
        auto_approve_condition = config.get("auto_approve_if", "")
        if auto_approve_condition:
            if self._evaluate_condition(auto_approve_condition, context):
                logger.info(f"Auto-approving checkpoint '{self.name}' based on condition")
                return {
                    "status": "approved",
                    "auto_approved": True,
                    "condition": auto_approve_condition,
                }

        # Create approval request
        request = ApprovalRequest(
            id=f"apr_{uuid.uuid4().hex[:12]}",
            workflow_id=context.workflow_id,
            step_id=context.current_step_id or self.name,
            title=config.get("title", f"Approval Required: {self.name}"),
            description=self._build_description(config, context),
            checklist=checklist,
            timeout_seconds=config.get("timeout_seconds", 3600.0),
            escalation_emails=config.get("escalation_emails", []),
        )

        # Store the request
        _pending_approvals[request.id] = request
        logger.info(f"Created approval request {request.id} for checkpoint '{self.name}'")

        # Notify listeners
        if self.on_approval_requested:
            try:
                self.on_approval_requested(request)
            except Exception as e:
                logger.warning(f"Failed to notify approval listener: {e}")

        # Store request ID in context for external resolution
        context.set_state(f"approval_request_{self.name}", request.id)

        # Wait for approval (with timeout)
        try:
            result = await asyncio.wait_for(
                self._wait_for_approval(request),
                timeout=request.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            # Handle timeout
            request.status = ApprovalStatus.TIMEOUT
            logger.warning(f"Approval request {request.id} timed out")

            # Trigger escalation
            await self._handle_escalation(request)

            return {
                "status": "timeout",
                "request_id": request.id,
                "timeout_seconds": request.timeout_seconds,
                "escalated_to": request.escalation_emails,
            }

    async def _wait_for_approval(self, request: ApprovalRequest) -> Dict[str, Any]:
        """Wait for the approval request to be resolved."""
        while request.status == ApprovalStatus.PENDING:
            await asyncio.sleep(1.0)

            # Check if request was updated
            if request.id in _pending_approvals:
                updated = _pending_approvals[request.id]
                if updated.status != ApprovalStatus.PENDING:
                    request = updated
                    break

        # Validate checklist if approved
        if request.status == ApprovalStatus.APPROVED:
            require_all = self._config.get("require_all_checklist", True)
            if require_all:
                missing = [c for c in request.checklist if c.required and not c.checked]
                if missing:
                    return {
                        "status": "rejected",
                        "reason": "Required checklist items not completed",
                        "missing_items": [c.label for c in missing],
                    }

        return {
            "status": request.status.value,
            "request_id": request.id,
            "responder_id": request.responder_id,
            "responder_notes": request.responder_notes,
            "responded_at": request.responded_at.isoformat() if request.responded_at else None,
            "checklist": [
                {"label": c.label, "checked": c.checked, "notes": c.notes}
                for c in request.checklist
            ],
        }

    async def _handle_escalation(self, request: ApprovalRequest) -> None:
        """Handle escalation when approval times out."""
        if not request.escalation_emails:
            return

        request.status = ApprovalStatus.ESCALATED
        logger.info(f"Escalating approval request {request.id} to: {request.escalation_emails}")

        # In production, this would send emails/notifications
        # For now, just log the escalation

    def _build_description(self, config: Dict[str, Any], context: WorkflowContext) -> str:
        """Build the approval request description."""
        base_description = config.get("description", "")

        # Add context information
        parts = [base_description] if base_description else []

        # Add workflow context
        parts.append(f"\nWorkflow: {context.workflow_id}")
        parts.append(f"Step: {context.current_step_id}")

        # Add input summary if relevant
        if context.inputs:
            parts.append("\nInputs:")
            for key, value in list(context.inputs.items())[:5]:  # Limit to first 5
                parts.append(f"  - {key}: {str(value)[:100]}")

        return "\n".join(parts)

    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """Safely evaluate a condition expression using AST-based evaluator."""
        try:
            namespace = {
                "inputs": context.inputs,
                "outputs": context.step_outputs,
                "state": context.state,
            }
            return safe_eval_bool(condition, namespace)
        except SafeEvalError:
            return False


# Helper functions for external approval resolution


def resolve_approval(
    request_id: str,
    status: ApprovalStatus,
    responder_id: str,
    notes: str = "",
    checklist_updates: Optional[Dict[str, bool]] = None,
) -> bool:
    """
    Resolve an approval request from external code.

    Args:
        request_id: ID of the approval request
        status: New status (APPROVED or REJECTED)
        responder_id: ID of the person responding
        notes: Optional notes from the responder
        checklist_updates: Optional dict mapping item IDs to checked status

    Returns:
        True if request was found and updated, False otherwise
    """
    if request_id not in _pending_approvals:
        return False

    request = _pending_approvals[request_id]
    request.status = status
    request.responder_id = responder_id
    request.responder_notes = notes
    request.responded_at = datetime.now(timezone.utc)

    if checklist_updates:
        for item in request.checklist:
            if item.id in checklist_updates:
                item.checked = checklist_updates[item.id]

    logger.info(f"Resolved approval request {request_id}: {status.value}")
    return True


def get_pending_approvals(workflow_id: Optional[str] = None) -> List[ApprovalRequest]:
    """Get all pending approval requests, optionally filtered by workflow."""
    approvals = [a for a in _pending_approvals.values() if a.status == ApprovalStatus.PENDING]
    if workflow_id:
        approvals = [a for a in approvals if a.workflow_id == workflow_id]
    return approvals


def get_approval_request(request_id: str) -> Optional[ApprovalRequest]:
    """Get an approval request by ID."""
    return _pending_approvals.get(request_id)

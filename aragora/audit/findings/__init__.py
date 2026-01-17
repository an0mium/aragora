"""
Finding Workflow System.

Enterprise workflow for managing audit findings from discovery to resolution.

Usage:
    from aragora.audit.findings import FindingWorkflow, WorkflowState

    # Transition finding to new state
    workflow = FindingWorkflow(finding_id="...")
    await workflow.transition_to(WorkflowState.INVESTIGATING, user_id="user_123")

    # Assign finding to user
    await workflow.assign(user_id="user_456", assigned_by="user_123")

    # Add comment
    await workflow.add_comment("Reviewed, confirmed as valid.", user_id="user_123")
"""

from .workflow import (
    WorkflowState,
    WorkflowTransition,
    FindingWorkflow,
    WorkflowEvent,
    WorkflowEventType,
)
from .assignment import (
    FindingAssignment,
    AssignmentManager,
    AssignmentPriority,
)

__all__ = [
    # Workflow
    "WorkflowState",
    "WorkflowTransition",
    "FindingWorkflow",
    "WorkflowEvent",
    "WorkflowEventType",
    # Assignment
    "FindingAssignment",
    "AssignmentManager",
    "AssignmentPriority",
]

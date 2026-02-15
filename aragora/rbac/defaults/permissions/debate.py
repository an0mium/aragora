"""
RBAC Permissions for Debate, Agent, and Orchestration resources.

Contains permissions related to:
- Debate management (create, run, fork, etc.)
- Agent management
- Orchestration and workflow execution
- Control plane operations
- Gauntlet (stress testing)
- Canvas
- Verification
- Nomic (self-improvement)
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# DEBATE PERMISSIONS
# ============================================================================

PERM_DEBATE_CREATE = _permission(
    ResourceType.DEBATE, Action.CREATE, "Create Debates", "Create new debates"
)
PERM_DEBATE_READ = _permission(
    ResourceType.DEBATE, Action.READ, "View Debates", "View debate details and history"
)
PERM_DEBATE_UPDATE = _permission(
    ResourceType.DEBATE, Action.UPDATE, "Update Debates", "Modify debate settings"
)
PERM_DEBATE_DELETE = _permission(
    ResourceType.DEBATE, Action.DELETE, "Delete Debates", "Delete debates permanently"
)
PERM_DEBATE_RUN = _permission(
    ResourceType.DEBATE, Action.RUN, "Run Debates", "Start and execute debates"
)
PERM_DEBATE_STOP = _permission(
    ResourceType.DEBATE, Action.STOP, "Stop Debates", "Stop running debates"
)
PERM_DEBATE_FORK = _permission(
    ResourceType.DEBATE, Action.FORK, "Fork Debates", "Create branches from existing debates"
)

# ============================================================================
# AGENT PERMISSIONS
# ============================================================================

PERM_AGENT_CREATE = _permission(
    ResourceType.AGENT, Action.CREATE, "Create Agents", "Create custom agent configurations"
)
PERM_AGENT_READ = _permission(
    ResourceType.AGENT, Action.READ, "View Agents", "View agent details and statistics"
)
PERM_AGENT_UPDATE = _permission(
    ResourceType.AGENT, Action.UPDATE, "Update Agents", "Modify agent configurations"
)
PERM_AGENT_DELETE = _permission(
    ResourceType.AGENT, Action.DELETE, "Delete Agents", "Remove agent configurations"
)
PERM_AGENT_DEPLOY = _permission(
    ResourceType.AGENT, Action.DEPLOY, "Deploy Agents", "Deploy agents to production"
)

# ============================================================================
# WORKFLOW PERMISSIONS
# ============================================================================

PERM_WORKFLOW_CREATE = _permission(
    ResourceType.WORKFLOW, Action.CREATE, "Create Workflows", "Create new workflows"
)
PERM_WORKFLOW_READ = _permission(
    ResourceType.WORKFLOW, Action.READ, "View Workflows", "View workflow definitions and executions"
)
PERM_WORKFLOW_RUN = _permission(
    ResourceType.WORKFLOW, Action.RUN, "Run Workflows", "Execute workflows"
)
PERM_WORKFLOW_DELETE = _permission(
    ResourceType.WORKFLOW, Action.DELETE, "Delete Workflows", "Delete workflow definitions"
)

# ============================================================================
# CONTROL PLANE PERMISSIONS
# ============================================================================

PERM_CONTROL_PLANE_READ = _permission(
    ResourceType.CONTROL_PLANE, Action.READ, "View Control Plane", "View tasks, agents, and status"
)
PERM_CONTROL_PLANE_SUBMIT = _permission(
    ResourceType.CONTROL_PLANE, Action.SUBMIT, "Submit Tasks", "Submit tasks to the control plane"
)
PERM_CONTROL_PLANE_CANCEL = _permission(
    ResourceType.CONTROL_PLANE, Action.CANCEL, "Cancel Tasks", "Cancel pending control plane tasks"
)
PERM_CONTROL_PLANE_DELIBERATE = _permission(
    ResourceType.CONTROL_PLANE,
    Action.DELIBERATE,
    "Start Deliberations",
    "Start multi-agent deliberation processes",
)
PERM_CONTROL_PLANE_AGENTS = _permission(
    ResourceType.CONTROL_PLANE,
    Action.ADMIN_OP,
    "Manage Agents",
    "Full control over agent registry",
)
PERM_CONTROL_PLANE_AGENTS_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_READ,
    "View Agents",
    "Read agent registry information",
)
PERM_CONTROL_PLANE_AGENTS_REGISTER = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_REGISTER,
    "Register Agents",
    "Register new agents in the control plane",
)
PERM_CONTROL_PLANE_AGENTS_UNREGISTER = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_UNREGISTER,
    "Unregister Agents",
    "Remove agents from the control plane",
)
PERM_CONTROL_PLANE_TASKS = _permission(
    ResourceType.CONTROL_PLANE,
    Action.MANAGE,
    "Manage Tasks",
    "Full control over task queue",
)
PERM_CONTROL_PLANE_TASKS_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_READ,
    "View Tasks",
    "Read task queue information",
)
PERM_CONTROL_PLANE_TASKS_SUBMIT = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_SUBMIT,
    "Submit Tasks",
    "Submit new tasks to the control plane",
)
PERM_CONTROL_PLANE_TASKS_CLAIM = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_CLAIM,
    "Claim Tasks",
    "Claim tasks for processing",
)
PERM_CONTROL_PLANE_TASKS_COMPLETE = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_COMPLETE,
    "Complete Tasks",
    "Mark tasks as completed",
)
PERM_CONTROL_PLANE_HEALTH_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.HEALTH_READ,
    "View Health",
    "Read control plane health status",
)

# ============================================================================
# GAUNTLET PERMISSIONS
# ============================================================================

PERM_GAUNTLET_RUN = _permission(
    ResourceType.GAUNTLET, Action.RUN, "Run Gauntlet", "Execute adversarial stress-tests"
)
PERM_GAUNTLET_READ = _permission(
    ResourceType.GAUNTLET,
    Action.READ,
    "View Gauntlet Results",
    "View gauntlet results and receipts",
)
PERM_GAUNTLET_DELETE = _permission(
    ResourceType.GAUNTLET, Action.DELETE, "Delete Gauntlet Results", "Delete gauntlet runs"
)
PERM_GAUNTLET_SIGN = _permission(
    ResourceType.GAUNTLET, Action.SIGN, "Sign Receipts", "Cryptographically sign decision receipts"
)
PERM_GAUNTLET_COMPARE = _permission(
    ResourceType.GAUNTLET, Action.COMPARE, "Compare Gauntlets", "Compare gauntlet run results"
)
PERM_GAUNTLET_EXPORT = _permission(
    ResourceType.GAUNTLET, Action.EXPORT_DATA, "Export Gauntlet Data", "Export gauntlet reports"
)

# ============================================================================
# ORCHESTRATION PERMISSIONS
# ============================================================================

PERM_ORCHESTRATION_READ = _permission(
    ResourceType.ORCHESTRATION,
    Action.READ,
    "View Orchestration",
    "View orchestration templates and status",
)
PERM_ORCHESTRATION_EXECUTE = _permission(
    ResourceType.ORCHESTRATION,
    Action.EXECUTE,
    "Execute Orchestration",
    "Run multi-agent deliberations",
)

# ============================================================================
# NOMIC PERMISSIONS
# ============================================================================

PERM_NOMIC_READ = _permission(
    ResourceType.NOMIC, Action.READ, "View Nomic", "View Nomic loop progress and results"
)
PERM_NOMIC_ADMIN = _permission(
    ResourceType.NOMIC,
    Action.ADMIN_OP,
    "Administer Nomic",
    "Control Nomic self-improvement operations",
)

# ============================================================================
# CANVAS PERMISSIONS
# ============================================================================

PERM_CANVAS_READ = _permission(
    ResourceType.CANVAS, Action.READ, "View Canvas", "View canvas documents and state"
)
PERM_CANVAS_CREATE = _permission(
    ResourceType.CANVAS, Action.CREATE, "Create Canvas", "Create new canvas documents"
)
PERM_CANVAS_UPDATE = _permission(
    ResourceType.CANVAS, Action.UPDATE, "Update Canvas", "Modify canvas content and state"
)
PERM_CANVAS_DELETE = _permission(
    ResourceType.CANVAS, Action.DELETE, "Delete Canvas", "Delete canvas documents"
)
PERM_CANVAS_RUN = _permission(
    ResourceType.CANVAS, Action.RUN, "Run Canvas", "Execute canvas operations"
)
PERM_CANVAS_WRITE = _permission(
    ResourceType.CANVAS, Action.WRITE, "Write Canvas", "Full write access to canvas"
)
PERM_CANVAS_SHARE = _permission(
    ResourceType.CANVAS, Action.SHARE, "Share Canvas", "Share canvas with others"
)

# ============================================================================
# VERIFICATION PERMISSIONS
# ============================================================================

PERM_VERIFICATION_READ = _permission(
    ResourceType.VERIFICATION, Action.READ, "View Verification", "View verification results"
)
PERM_VERIFICATION_CREATE = _permission(
    ResourceType.VERIFICATION, Action.CREATE, "Create Verification", "Run formal verification"
)

# ============================================================================
# CHECKPOINT & REPLAY PERMISSIONS
# ============================================================================

PERM_CHECKPOINT_READ = _permission(
    ResourceType.CHECKPOINT, Action.READ, "View Checkpoints", "View saved checkpoints"
)
PERM_CHECKPOINT_CREATE = _permission(
    ResourceType.CHECKPOINT, Action.CREATE, "Create Checkpoints", "Save debate checkpoints"
)
PERM_CHECKPOINT_DELETE = _permission(
    ResourceType.CHECKPOINT, Action.DELETE, "Delete Checkpoints", "Remove saved checkpoints"
)
PERM_REPLAYS_READ = _permission(
    ResourceType.REPLAY, Action.READ, "View Replays", "View debate replay recordings"
)

# ============================================================================
# DECISION PERMISSIONS
# ============================================================================

PERM_DECISION_CREATE = _permission(
    ResourceType.DECISION, Action.CREATE, "Create Decisions", "Submit decisions via unified router"
)
PERM_DECISION_READ = _permission(
    ResourceType.DECISION, Action.READ, "View Decisions", "View decision results and status"
)
PERM_DECISION_UPDATE = _permission(
    ResourceType.DECISION,
    Action.UPDATE,
    "Update Decisions",
    "Cancel or retry decision requests",
)

# ============================================================================
# EXPLAINABILITY PERMISSIONS
# ============================================================================

PERM_EXPLAINABILITY_READ = _permission(
    ResourceType.EXPLAINABILITY, Action.READ, "View Explanations", "View decision explanations"
)
PERM_EXPLAINABILITY_BATCH = _permission(
    ResourceType.EXPLAINABILITY,
    Action.BATCH,
    "Batch Explanations",
    "Process batch explanation jobs",
)

# ============================================================================
# BREAKPOINT PERMISSIONS (Admin-only - controls debate flow)
# ============================================================================

PERM_BREAKPOINT_READ = _permission(
    ResourceType.BREAKPOINT,
    Action.READ,
    "View Breakpoints",
    "View pending breakpoints and their status",
)
PERM_BREAKPOINT_UPDATE = _permission(
    ResourceType.BREAKPOINT,
    Action.UPDATE,
    "Resolve Breakpoints",
    "Approve or reject breakpoint resolutions",
)

# ============================================================================
# PLAN PERMISSIONS (Decision plan lifecycle)
# ============================================================================

PERM_PLAN_CREATE = _permission(
    ResourceType.PLANS, Action.CREATE, "Create Plans", "Create decision plans from debate results"
)
PERM_PLAN_READ = _permission(
    ResourceType.PLANS, Action.READ, "View Plans", "View decision plan details and status"
)
PERM_PLAN_APPROVE = _permission(
    ResourceType.PLANS, Action.APPROVE, "Approve Plans", "Approve decision plans for execution"
)
PERM_PLAN_REJECT = _permission(
    ResourceType.PLANS, Action.DENY, "Reject Plans", "Reject decision plans with reason"
)

# All debate-related permission exports
__all__ = [
    # Debate
    "PERM_DEBATE_CREATE",
    "PERM_DEBATE_READ",
    "PERM_DEBATE_UPDATE",
    "PERM_DEBATE_DELETE",
    "PERM_DEBATE_RUN",
    "PERM_DEBATE_STOP",
    "PERM_DEBATE_FORK",
    # Agent
    "PERM_AGENT_CREATE",
    "PERM_AGENT_READ",
    "PERM_AGENT_UPDATE",
    "PERM_AGENT_DELETE",
    "PERM_AGENT_DEPLOY",
    # Workflow
    "PERM_WORKFLOW_CREATE",
    "PERM_WORKFLOW_READ",
    "PERM_WORKFLOW_RUN",
    "PERM_WORKFLOW_DELETE",
    # Control Plane
    "PERM_CONTROL_PLANE_READ",
    "PERM_CONTROL_PLANE_SUBMIT",
    "PERM_CONTROL_PLANE_CANCEL",
    "PERM_CONTROL_PLANE_DELIBERATE",
    "PERM_CONTROL_PLANE_AGENTS",
    "PERM_CONTROL_PLANE_AGENTS_READ",
    "PERM_CONTROL_PLANE_AGENTS_REGISTER",
    "PERM_CONTROL_PLANE_AGENTS_UNREGISTER",
    "PERM_CONTROL_PLANE_TASKS",
    "PERM_CONTROL_PLANE_TASKS_READ",
    "PERM_CONTROL_PLANE_TASKS_SUBMIT",
    "PERM_CONTROL_PLANE_TASKS_CLAIM",
    "PERM_CONTROL_PLANE_TASKS_COMPLETE",
    "PERM_CONTROL_PLANE_HEALTH_READ",
    # Gauntlet
    "PERM_GAUNTLET_RUN",
    "PERM_GAUNTLET_READ",
    "PERM_GAUNTLET_DELETE",
    "PERM_GAUNTLET_SIGN",
    "PERM_GAUNTLET_COMPARE",
    "PERM_GAUNTLET_EXPORT",
    # Orchestration
    "PERM_ORCHESTRATION_READ",
    "PERM_ORCHESTRATION_EXECUTE",
    # Nomic
    "PERM_NOMIC_READ",
    "PERM_NOMIC_ADMIN",
    # Canvas
    "PERM_CANVAS_READ",
    "PERM_CANVAS_CREATE",
    "PERM_CANVAS_UPDATE",
    "PERM_CANVAS_DELETE",
    "PERM_CANVAS_RUN",
    "PERM_CANVAS_WRITE",
    "PERM_CANVAS_SHARE",
    # Verification
    "PERM_VERIFICATION_READ",
    "PERM_VERIFICATION_CREATE",
    # Checkpoint & Replay
    "PERM_CHECKPOINT_READ",
    "PERM_CHECKPOINT_CREATE",
    "PERM_CHECKPOINT_DELETE",
    "PERM_REPLAYS_READ",
    # Decision
    "PERM_DECISION_CREATE",
    "PERM_DECISION_READ",
    "PERM_DECISION_UPDATE",
    # Explainability
    "PERM_EXPLAINABILITY_READ",
    "PERM_EXPLAINABILITY_BATCH",
    # Breakpoints
    "PERM_BREAKPOINT_READ",
    "PERM_BREAKPOINT_UPDATE",
    # Plans
    "PERM_PLAN_CREATE",
    "PERM_PLAN_READ",
    "PERM_PLAN_APPROVE",
    "PERM_PLAN_REJECT",
]

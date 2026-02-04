"""
Decision Plan - Bridge from DebateResult to executable implementation.

The DecisionPlan is the central artifact in the gold path:
    input → debate → DECISION_PLAN → implementation → verification → learning

It bundles all the artifacts needed to go from a debate conclusion to
executable implementation tasks, with risk-aware routing and human
checkpoint support.

Unlike DecisionIntegrityPackage (which bundles receipt + plan),
DecisionPlan adds:
- Risk analysis with risk-aware routing
- Verification plan for post-implementation checks
- Budget tracking and limits
- Human checkpoint configuration
- WorkflowDefinition generation for the workflow engine
- Status tracking across the full lifecycle

Usage:
    # From a debate result
    plan = DecisionPlanFactory.from_debate_result(result)

    # Check if human approval is required
    if plan.requires_human_approval:
        await plan.request_approval(approver_id="user-123")

    # Generate executable workflow
    workflow = plan.to_workflow_definition()

    # Execute via workflow engine
    engine = WorkflowEngine()
    result = await engine.execute(workflow)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.workflow.types import WorkflowDefinition

from aragora.core_types import DebateResult
from aragora.implement.types import ImplementPlan
from aragora.pipeline.risk_register import RiskLevel, RiskRegister
from aragora.pipeline.verification_plan import (
    CasePriority,
    VerificationPlan,
)


# ---------------------------------------------------------------------------
# Workflow-build helpers (used by DecisionPlan.to_workflow_definition)
# ---------------------------------------------------------------------------

_DIRECT_ACTION_TYPES = frozenset(
    {
        "shell",
        "file_read",
        "file_write",
        "file_delete",
        "browser",
        "screenshot",
        "api",
        "keyboard",
        "mouse",
    }
)


@dataclass
class _WorkflowBuildContext:
    """Mutable context threaded through workflow-build helper methods."""

    steps: list[Any] = field(default_factory=list)
    transitions: list[Any] = field(default_factory=list)
    step_idx: int = 0
    prev_step_id: str | None = None

    def next_step_id(self) -> str:
        self.step_idx += 1
        return f"step-{self.step_idx:03d}"

    def add_step(self, step: Any) -> None:
        self.steps.append(step)

    def link(self, to_step: str) -> None:
        """Add a transition from *prev_step_id* → *to_step* (if prev exists)."""
        if self.prev_step_id is None:
            return
        from aragora.workflow.types import TransitionRule

        self.transitions.append(
            TransitionRule(
                id=f"tr-{len(self.transitions) + 1}",
                from_step=self.prev_step_id,
                to_step=to_step,
                condition="True",
            )
        )

    def advance(self, step_id: str) -> None:
        """Link *prev* → *step_id*, then set *prev* to *step_id*."""
        self.link(step_id)
        self.prev_step_id = step_id


def _normalize_computer_action(
    action: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalise a single ``computer_use_actions`` entry.

    Returns ``(openclaw_actions, manual_actions)`` – each a list of 0-or-1
    items so the caller can simply ``extend`` its accumulators.
    """
    if not isinstance(action, dict):
        return [], []

    action_type = action.get("action") or action.get("action_type") or action.get("type")
    if not action_type:
        return [], []

    atv = str(action_type).lower().strip()

    # --- direct pass-through types ---
    if atv in _DIRECT_ACTION_TYPES:
        return [action], []

    # --- navigate / browser_navigate ---
    if atv in {"navigate", "browser_navigate"}:
        url = action.get("url") or (action.get("params") or {}).get("url") or action.get("target")
        if url:
            return [
                {
                    "action_type": "browser",
                    "url": url,
                    "description": action.get("description") or f"Navigate to {url}",
                    "params": action.get("params") or {},
                    "require_approval": bool(action.get("require_approval", False)),
                }
            ], []
        return [], [_manual_entry(action, "missing_url")]

    # --- screenshot / browser_screenshot ---
    if atv in {"screenshot", "browser_screenshot"}:
        return [
            {
                "action_type": "screenshot",
                "url": action.get("url", ""),
                "description": action.get("description") or "Capture screenshot",
                "params": action.get("params") or {},
                "require_approval": bool(action.get("require_approval", False)),
            }
        ], []

    # --- click / browser_click / mouse ---
    if atv in {"click", "browser_click", "mouse"}:
        coords = (
            action.get("coordinate")
            or action.get("coords")
            or action.get("position")
            or (action.get("params") or {}).get("coordinate")
            or (action.get("params") or {}).get("coords")
        )
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return [
                {
                    "action_type": "mouse",
                    "params": {"x": coords[0], "y": coords[1]},
                    "description": action.get("description") or f"Click at {coords[0]},{coords[1]}",
                    "require_approval": bool(action.get("require_approval", True)),
                }
            ], []
        return [], [_manual_entry(action, "missing_coordinates")]

    # --- type / browser_type / key / keyboard ---
    if atv in {"type", "browser_type", "key", "keyboard"}:
        text = (
            action.get("text")
            or (action.get("params") or {}).get("text")
            or action.get("key")
            or (action.get("params") or {}).get("key")
        )
        if text:
            return [
                {
                    "action_type": "keyboard",
                    "params": {"text": text},
                    "description": action.get("description") or f"Type '{text}'",
                    "require_approval": bool(action.get("require_approval", True)),
                }
            ], []
        return [], [_manual_entry(action, "missing_text")]

    # --- unsupported ---
    return [], [_manual_entry(action, "unsupported_action")]


def _manual_entry(action: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "description": action.get("description")
        or action.get("name")
        or f"Manual action required ({reason})",
        "reason": reason,
        "action": action,
    }


class PlanStatus(Enum):
    """Lifecycle status of a DecisionPlan."""

    CREATED = "created"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ApprovalMode(Enum):
    """How human approval is determined."""

    ALWAYS = "always"  # Always require human approval
    RISK_BASED = "risk_based"  # Auto-approve if no critical/high risks
    CONFIDENCE_BASED = "confidence_based"  # Auto-approve above confidence threshold
    NEVER = "never"  # Skip approval (for automated pipelines)


@dataclass
class ApprovalRecord:
    """Records a human approval decision."""

    approved: bool
    approver_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "approver_id": self.approver_id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "conditions": self.conditions,
        }


@dataclass
class BudgetAllocation:
    """Budget tracking for plan execution."""

    limit_usd: float | None = None
    estimated_usd: float = 0.0
    spent_usd: float = 0.0

    # Per-phase budget breakdown
    debate_cost_usd: float = 0.0
    implementation_cost_usd: float = 0.0
    verification_cost_usd: float = 0.0

    @property
    def remaining_usd(self) -> float | None:
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def over_budget(self) -> bool:
        if self.limit_usd is None:
            return False
        return self.spent_usd > self.limit_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit_usd": self.limit_usd,
            "estimated_usd": self.estimated_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "over_budget": self.over_budget,
            "debate_cost_usd": self.debate_cost_usd,
            "implementation_cost_usd": self.implementation_cost_usd,
            "verification_cost_usd": self.verification_cost_usd,
        }


@dataclass
class DecisionPlan:
    """Bridges DebateResult to executable implementation with full decision trail.

    This is the central data structure in the gold path:
        input → debate → DECISION_PLAN → implementation → verification → learning

    It bundles all the artifacts needed to go from a debate conclusion to
    executable implementation tasks, with risk-aware routing, human
    checkpoints, and budget tracking.

    Attributes:
        id: Unique plan identifier.
        debate_id: ID of the source debate.
        task: The original task/question debated.
        created_at: When the plan was created.
        status: Current lifecycle status.

        debate_result: Source DebateResult from Arena.run().
        risk_register: Risks identified from debate analysis.
        verification_plan: Post-implementation verification strategy.
        implement_plan: Decomposed implementation tasks.

        budget: Budget allocation and tracking.
        approval_mode: How approval is determined.
        approval_record: Human approval decision (if applicable).

        max_auto_risk: Maximum risk level for auto-execution.
    """

    # Identity
    id: str = field(default_factory=lambda: f"dp-{uuid.uuid4().hex[:12]}")
    debate_id: str = ""
    task: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: PlanStatus = PlanStatus.CREATED

    # Source debate
    debate_result: DebateResult | None = None

    # Decision artifacts
    risk_register: RiskRegister | None = None
    verification_plan: VerificationPlan | None = None
    implement_plan: ImplementPlan | None = None

    # Budget
    budget: BudgetAllocation = field(default_factory=BudgetAllocation)

    # Approval
    approval_mode: ApprovalMode = ApprovalMode.RISK_BASED
    approval_record: ApprovalRecord | None = None

    # Risk-aware routing
    max_auto_risk: RiskLevel = RiskLevel.LOW

    # Execution tracking
    workflow_id: str | None = None
    execution_started_at: datetime | None = None
    execution_completed_at: datetime | None = None
    execution_error: str | None = None

    # Memory feedback
    memory_written: bool = False
    bead_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.debate_result and not self.debate_id:
            self.debate_id = self.debate_result.debate_id
        if self.debate_result and not self.task:
            self.task = self.debate_result.task

    # -------------------------------------------------------------------------
    # Risk assessment
    # -------------------------------------------------------------------------

    @property
    def has_critical_risks(self) -> bool:
        """Whether the plan has critical or high-severity risks."""
        if not self.risk_register:
            return False
        return len(self.risk_register.get_critical_risks()) > 0

    @property
    def highest_risk_level(self) -> RiskLevel:
        """Return the highest risk level in the register."""
        if not self.risk_register or not self.risk_register.risks:
            return RiskLevel.LOW
        level_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1,
        }
        return max(self.risk_register.risks, key=lambda r: level_order.get(r.level, 0)).level

    # -------------------------------------------------------------------------
    # Approval logic
    # -------------------------------------------------------------------------

    @property
    def requires_human_approval(self) -> bool:
        """Whether the plan requires human approval before execution."""
        if self.approval_mode == ApprovalMode.ALWAYS:
            return True
        if self.approval_mode == ApprovalMode.NEVER:
            return False
        if self.approval_mode == ApprovalMode.RISK_BASED:
            level_order = {
                RiskLevel.LOW: 1,
                RiskLevel.MEDIUM: 2,
                RiskLevel.HIGH: 3,
                RiskLevel.CRITICAL: 4,
            }
            return level_order.get(self.highest_risk_level, 0) > level_order.get(
                self.max_auto_risk, 1
            )
        if self.approval_mode == ApprovalMode.CONFIDENCE_BASED:
            if not self.debate_result:
                return True
            return self.debate_result.confidence < 0.8
        return True

    @property
    def is_approved(self) -> bool:
        """Whether the plan has been approved for execution."""
        if not self.requires_human_approval:
            return True
        return self.approval_record is not None and self.approval_record.approved

    def approve(
        self, approver_id: str, reason: str = "", conditions: list[str] | None = None
    ) -> None:
        """Record approval and advance status."""
        self.approval_record = ApprovalRecord(
            approved=True,
            approver_id=approver_id,
            reason=reason,
            conditions=conditions or [],
        )
        self.status = PlanStatus.APPROVED

    def reject(self, approver_id: str, reason: str = "") -> None:
        """Record rejection."""
        self.approval_record = ApprovalRecord(
            approved=False,
            approver_id=approver_id,
            reason=reason,
        )
        self.status = PlanStatus.REJECTED

    # -------------------------------------------------------------------------
    # Workflow generation
    # -------------------------------------------------------------------------

    def to_workflow_definition(self, *, parallelize: bool = False) -> "WorkflowDefinition":
        """Generate a WorkflowDefinition for the workflow engine.

        Creates a DAG of steps that implements the full gold path:
        1. Human approval checkpoint (if required)
        2. Implementation tasks (from implement_plan)
        3. Verification steps (from verification_plan)
        4. Memory write-back (feedback loop)

        Risk-aware routing: critical risks get additional human
        checkpoints before their corresponding implementation steps.

        Args:
            parallelize: When True, execute independent implementation tasks
                in parallel via a workflow parallel step.
        """
        from aragora.workflow.types import WorkflowCategory, WorkflowDefinition

        ctx = _WorkflowBuildContext()

        # Parse OpenClaw / computer-use actions from metadata
        openclaw_actions, manual_actions, openclaw_session = self._parse_openclaw_metadata()

        # Phase 1 – approval checkpoint
        self._build_approval_step(ctx)

        # Phase 2 – implementation tasks
        self._build_implementation_steps(ctx, parallelize=parallelize)

        # Phase 2b – OpenClaw / computer-use actions
        self._build_openclaw_steps(ctx, openclaw_actions, manual_actions, openclaw_session)

        # Phase 3 – verification
        self._build_verification_step(ctx)

        # Phase 4 – memory write-back
        self._build_memory_step(ctx)

        workflow_id = f"wf-{self.id}"
        self.workflow_id = workflow_id

        return WorkflowDefinition(
            id=workflow_id,
            name=f"Decision Plan: {self.task[:60]}",
            description=f"Auto-generated workflow from debate {self.debate_id}",
            steps=ctx.steps,
            transitions=ctx.transitions,
            category=WorkflowCategory.GENERAL,
            tags=["decision-plan", "auto-generated"],
            metadata={
                "decision_plan_id": self.id,
                "debate_id": self.debate_id,
                "debate_confidence": self.debate_result.confidence if self.debate_result else 0,
                "risk_count": len(self.risk_register.risks) if self.risk_register else 0,
            },
        )

    # -- workflow-build helpers (private) ------------------------------------

    def _parse_openclaw_metadata(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        """Extract OpenClaw / computer-use actions from ``self.metadata``.

        Returns ``(openclaw_actions, manual_actions, openclaw_session)``.
        """
        openclaw_actions: list[dict[str, Any]] = []
        manual_actions: list[dict[str, Any]] = []
        openclaw_session: dict[str, Any] | None = None

        if not isinstance(self.metadata, dict):
            return openclaw_actions, manual_actions, openclaw_session

        # Explicit openclaw actions
        raw_actions = self.metadata.get("openclaw_actions") or []
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if isinstance(action, dict) and action.get("action_type"):
                    openclaw_actions.append(action)

        # Computer-use actions (need normalisation)
        raw_computer = self.metadata.get("computer_use_actions") or []
        if isinstance(raw_computer, list):
            for action in raw_computer:
                oc, ma = _normalize_computer_action(action)
                openclaw_actions.extend(oc)
                manual_actions.extend(ma)

        raw_session = self.metadata.get("openclaw_session")
        if isinstance(raw_session, dict):
            openclaw_session = raw_session

        return openclaw_actions, manual_actions, openclaw_session

    def _build_approval_step(self, ctx: _WorkflowBuildContext) -> None:
        """Add a human-approval checkpoint if the plan requires it."""
        if not self.requires_human_approval:
            return

        from aragora.workflow.types import StepDefinition

        approval_id = ctx.next_step_id()
        ctx.add_step(
            StepDefinition(
                id=approval_id,
                name="Human Approval",
                step_type="human_checkpoint",
                config={
                    "prompt": f"Approve implementation of: {self.task[:200]}",
                    "context": {
                        "debate_confidence": self.debate_result.confidence
                        if self.debate_result
                        else 0,
                        "risk_summary": self.risk_register.summary if self.risk_register else {},
                        "task_count": len(self.implement_plan.tasks) if self.implement_plan else 0,
                    },
                    "timeout_seconds": 86400,  # 24h default
                },
                description="Review and approve the implementation plan",
            )
        )
        ctx.prev_step_id = approval_id

    def _find_critical_task_ids(self) -> set[str]:
        """Return IDs of implementation tasks that touch files named in critical risks."""
        critical: set[str] = set()
        if not self.risk_register or not self.implement_plan:
            return critical
        for risk in self.risk_register.get_critical_risks():
            for task in self.implement_plan.tasks:
                for file_path in task.files:
                    if file_path in risk.description or file_path in risk.title:
                        critical.add(task.id)
                        break
        return critical

    def _build_implementation_steps(self, ctx: _WorkflowBuildContext, *, parallelize: bool) -> None:
        """Add implementation-task steps to *ctx*."""
        if not self.implement_plan:
            return

        from aragora.workflow.types import ExecutionPattern, StepDefinition

        task_to_step: dict[str, str] = {}
        task_steps: list[StepDefinition] = []
        critical_task_ids = self._find_critical_task_ids()

        parallel_impl = (
            parallelize
            and len(self.implement_plan.tasks) > 1
            and not any(task.dependencies for task in self.implement_plan.tasks)
            and not critical_task_ids
        )

        for task in self.implement_plan.tasks:
            impl_step_id = ctx.next_step_id()
            task_to_step[task.id] = impl_step_id
            task_has_critical_risk = task.id in critical_task_ids

            # Risk checkpoint before high-risk tasks (sequential only)
            if task_has_critical_risk and not parallel_impl:
                risk_checkpoint_id = ctx.next_step_id()
                ctx.add_step(
                    StepDefinition(
                        id=risk_checkpoint_id,
                        name=f"Risk Review: {task.description[:40]}",
                        step_type="human_checkpoint",
                        config={
                            "prompt": f"High-risk task requires review: {task.description}",
                            "risk_level": "critical",
                        },
                        description="Review high-risk implementation step",
                    )
                )
                ctx.advance(risk_checkpoint_id)

            task_step = StepDefinition(
                id=impl_step_id,
                name=f"Implement: {task.description[:50]}",
                step_type="implementation",
                config={
                    "task_id": task.id,
                    "description": task.description,
                    "files": task.files,
                    "complexity": task.complexity,
                },
                description=task.description,
                timeout_seconds=300.0 if task.complexity == "complex" else 120.0,
            )

            if parallel_impl:
                task_steps.append(task_step)
            else:
                ctx.add_step(task_step)
                # Wire dependencies
                if task.dependencies:
                    for dep_id in task.dependencies:
                        dep_step_id = task_to_step.get(dep_id)
                        if dep_step_id:
                            ctx.transitions.append(self._transition(ctx, dep_step_id, impl_step_id))
                elif ctx.prev_step_id and not task_has_critical_risk:
                    ctx.link(impl_step_id)
                ctx.prev_step_id = impl_step_id

        if parallel_impl:
            parallel_step_id = ctx.next_step_id()
            ctx.add_step(
                StepDefinition(
                    id=parallel_step_id,
                    name="Parallel Implementation",
                    step_type="task",
                    execution_pattern=ExecutionPattern.PARALLEL,
                    config={"parallel_steps": list(task_to_step.values())},
                    description="Execute implementation tasks in parallel",
                )
            )
            ctx.advance(parallel_step_id)
            ctx.steps.extend(task_steps)

    @staticmethod
    def _transition(ctx: _WorkflowBuildContext, from_step: str, to_step: str) -> Any:
        from aragora.workflow.types import TransitionRule

        return TransitionRule(
            id=f"tr-{len(ctx.transitions) + 1}",
            from_step=from_step,
            to_step=to_step,
            condition="True",
        )

    def _build_openclaw_steps(
        self,
        ctx: _WorkflowBuildContext,
        openclaw_actions: list[dict[str, Any]],
        manual_actions: list[dict[str, Any]],
        openclaw_session: dict[str, Any] | None,
    ) -> None:
        """Add OpenClaw session + action steps and manual-action checkpoints."""
        if not openclaw_actions and not manual_actions:
            return

        from aragora.workflow.types import StepDefinition

        session_config = openclaw_session or {}

        if openclaw_actions:
            self._build_openclaw_session_steps(ctx, openclaw_actions, session_config)

        for manual in manual_actions:
            checkpoint_id = ctx.next_step_id()
            ctx.add_step(
                StepDefinition(
                    id=checkpoint_id,
                    name="Manual Action Required",
                    step_type="human_checkpoint",
                    config={
                        "prompt": manual.get("description", "Manual action required"),
                        "reason": manual.get("reason", ""),
                        "action": manual.get("action", {}),
                    },
                    description="Manual action required for unsupported automation",
                )
            )
            ctx.advance(checkpoint_id)

    def _build_openclaw_session_steps(
        self,
        ctx: _WorkflowBuildContext,
        openclaw_actions: list[dict[str, Any]],
        session_config: dict[str, Any],
    ) -> None:
        """Create → execute actions → end an OpenClaw session."""
        from aragora.workflow.types import StepDefinition

        # Open session
        session_step_id = ctx.next_step_id()
        ctx.add_step(
            StepDefinition(
                id=session_step_id,
                name="OpenClaw Session",
                step_type="openclaw_session",
                config={
                    "operation": "create",
                    "workspace_id": session_config.get("workspace_id", "/workspace"),
                    "roles": session_config.get("roles", ["developer"]),
                    "user_id": session_config.get("user_id"),
                    "tenant_id": session_config.get("tenant_id"),
                },
                description="Create OpenClaw session for automated actions",
            )
        )
        ctx.advance(session_step_id)

        session_ref = f"{{step.{session_step_id}.session_id}}"

        # Individual actions
        for action in openclaw_actions:
            if not isinstance(action, dict):
                continue
            self._build_single_openclaw_action(ctx, action, session_ref)

        # End session
        end_session_id = ctx.next_step_id()
        ctx.add_step(
            StepDefinition(
                id=end_session_id,
                name="End OpenClaw Session",
                step_type="openclaw_session",
                config={"operation": "end", "session_id": session_ref},
                description="Terminate OpenClaw session",
            )
        )
        ctx.advance(end_session_id)

    def _build_single_openclaw_action(
        self,
        ctx: _WorkflowBuildContext,
        action: dict[str, Any],
        session_ref: str,
    ) -> None:
        """Emit one OpenClaw action step (with optional approval checkpoint)."""
        from aragora.workflow.types import StepDefinition

        action_type = str(action.get("action_type", "shell"))
        action_desc = (
            action.get("description") or action.get("name") or f"OpenClaw action ({action_type})"
        )

        if bool(action.get("require_approval", False)):
            checkpoint_id = ctx.next_step_id()
            ctx.add_step(
                StepDefinition(
                    id=checkpoint_id,
                    name=f"Approve OpenClaw: {action_desc[:40]}",
                    step_type="human_checkpoint",
                    config={
                        "prompt": f"Approve OpenClaw action: {action_desc}",
                        "action_type": action_type,
                    },
                    description="Approve OpenClaw action execution",
                )
            )
            ctx.advance(checkpoint_id)

        action_step_id = ctx.next_step_id()
        action_config: dict[str, Any] = {
            "action_type": action_type,
            "session_id": session_ref,
        }
        for key in ("command", "path", "content", "url", "params", "timeout_seconds", "on_failure"):
            if action.get(key) is not None:
                action_config[key] = action[key]

        ctx.add_step(
            StepDefinition(
                id=action_step_id,
                name=f"OpenClaw: {action_desc[:50]}",
                step_type="openclaw_action",
                config=action_config,
                description=action_desc,
            )
        )
        ctx.advance(action_step_id)

    def _build_verification_step(self, ctx: _WorkflowBuildContext) -> None:
        """Add the verification step if there are test cases."""
        if not self.verification_plan or not self.verification_plan.test_cases:
            return

        from aragora.workflow.types import StepDefinition

        verify_step_id = ctx.next_step_id()
        ctx.add_step(
            StepDefinition(
                id=verify_step_id,
                name="Run Verification",
                step_type="verification",
                config={
                    "action": "verify",
                    "test_count": len(self.verification_plan.test_cases),
                    "critical_count": len(self.verification_plan.get_by_priority(CasePriority.P0)),
                },
                description="Execute verification plan against implementation",
            )
        )
        ctx.advance(verify_step_id)

    def _build_memory_step(self, ctx: _WorkflowBuildContext) -> None:
        """Add the memory write-back (feedback loop) step."""
        from aragora.workflow.types import StepDefinition

        memory_step_id = ctx.next_step_id()
        ctx.add_step(
            StepDefinition(
                id=memory_step_id,
                name="Write to Memory",
                step_type="memory_write",
                config={
                    "action": "record_outcome",
                    "debate_id": self.debate_id,
                    "plan_id": self.id,
                },
                description="Record implementation outcome to organizational memory",
                optional=True,
            )
        )
        ctx.advance(memory_step_id)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "task": self.task,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "debate_result": self.debate_result.to_dict() if self.debate_result else None,
            "risk_register": self.risk_register.to_dict() if self.risk_register else None,
            "verification_plan": self.verification_plan.to_dict()
            if self.verification_plan
            else None,
            "implement_plan": self.implement_plan.to_dict() if self.implement_plan else None,
            "budget": self.budget.to_dict(),
            "approval_mode": self.approval_mode.value,
            "approval_record": self.approval_record.to_dict() if self.approval_record else None,
            "max_auto_risk": self.max_auto_risk.value,
            "workflow_id": self.workflow_id,
            "execution_started_at": self.execution_started_at.isoformat()
            if self.execution_started_at
            else None,
            "execution_completed_at": self.execution_completed_at.isoformat()
            if self.execution_completed_at
            else None,
            "execution_error": self.execution_error,
            "memory_written": self.memory_written,
            "bead_id": self.bead_id,
            "has_critical_risks": self.has_critical_risks,
            "requires_human_approval": self.requires_human_approval,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        risk_str = "none"
        if self.risk_register:
            s = self.risk_register.summary
            risk_str = f"{s['total_risks']} total ({s['critical']} critical, {s['high']} high)"

        verify_str = "none"
        if self.verification_plan:
            verify_str = f"{len(self.verification_plan.test_cases)} cases"

        task_str = "none"
        if self.implement_plan:
            task_str = f"{len(self.implement_plan.tasks)} tasks"

        budget_str = "unlimited"
        if self.budget.limit_usd is not None:
            budget_str = f"${self.budget.limit_usd:.2f} (${self.budget.spent_usd:.2f} spent)"

        confidence_str = f"{self.debate_result.confidence:.0%}" if self.debate_result else "N/A"

        return f"""Decision Plan ({self.id})
Status: {self.status.value}
Task: {self.task[:100]}
Debate: {self.debate_id} (confidence: {confidence_str})
Risks: {risk_str}
Verification: {verify_str}
Implementation: {task_str}
Budget: {budget_str}
Approval: {"required" if self.requires_human_approval else "auto"} ({self.approval_mode.value})"""

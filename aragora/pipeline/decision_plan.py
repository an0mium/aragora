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

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.workflow.types import WorkflowDefinition

from aragora.core_types import DebateResult
from aragora.implement.types import ImplementPlan, ImplementTask
from aragora.pipeline.risk_register import Risk, RiskCategory, RiskLevel, RiskRegister
from aragora.pipeline.verification_plan import (
    CasePriority,
    VerificationCase,
    VerificationPlan,
    VerificationType,
)


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

    def to_workflow_definition(self) -> "WorkflowDefinition":
        """Generate a WorkflowDefinition for the workflow engine.

        Creates a DAG of steps that implements the full gold path:
        1. Human approval checkpoint (if required)
        2. Implementation tasks (from implement_plan)
        3. Verification steps (from verification_plan)
        4. Memory write-back (feedback loop)

        Risk-aware routing: critical risks get additional human
        checkpoints before their corresponding implementation steps.
        """
        from aragora.workflow.types import (
            StepDefinition,
            TransitionRule,
            WorkflowCategory,
            WorkflowDefinition,
        )

        steps: list[StepDefinition] = []
        transitions: list[TransitionRule] = []
        step_idx = 0

        def _step_id() -> str:
            nonlocal step_idx
            step_idx += 1
            return f"step-{step_idx:03d}"

        # Step 1: Approval checkpoint (if needed)
        prev_step_id: str | None = None
        if self.requires_human_approval:
            approval_id = _step_id()
            steps.append(
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
                            "risk_summary": self.risk_register.summary
                            if self.risk_register
                            else {},
                            "task_count": len(self.implement_plan.tasks)
                            if self.implement_plan
                            else 0,
                        },
                        "timeout_seconds": 86400,  # 24h default
                    },
                    description="Review and approve the implementation plan",
                )
            )
            prev_step_id = approval_id

        # Step 2: Implementation tasks
        if self.implement_plan:
            # Map task IDs to workflow step IDs for dependency resolution
            task_to_step: dict[str, str] = {}

            for task in self.implement_plan.tasks:
                impl_step_id = _step_id()
                task_to_step[task.id] = impl_step_id

                # Check if this task has related critical risks
                task_has_critical_risk = False
                if self.risk_register:
                    for risk in self.risk_register.get_critical_risks():
                        # Match risks to tasks by file overlap or keyword match
                        for file_path in task.files:
                            if file_path in risk.description or file_path in risk.title:
                                task_has_critical_risk = True
                                break

                # Insert human checkpoint before high-risk tasks
                if task_has_critical_risk:
                    risk_checkpoint_id = _step_id()
                    steps.append(
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
                    # Wire previous step → risk checkpoint
                    if prev_step_id:
                        transitions.append(
                            TransitionRule(
                                id=f"tr-{len(transitions) + 1}",
                                from_step=prev_step_id,
                                to_step=risk_checkpoint_id,
                                condition="True",
                            )
                        )
                    prev_step_id = risk_checkpoint_id

                steps.append(
                    StepDefinition(
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
                )

                # Wire dependencies
                if task.dependencies:
                    for dep_id in task.dependencies:
                        dep_step_id = task_to_step.get(dep_id)
                        if dep_step_id:
                            transitions.append(
                                TransitionRule(
                                    id=f"tr-{len(transitions) + 1}",
                                    from_step=dep_step_id,
                                    to_step=impl_step_id,
                                    condition="True",
                                )
                            )
                elif prev_step_id and not task_has_critical_risk:
                    # Sequential fallback when no explicit dependencies
                    transitions.append(
                        TransitionRule(
                            id=f"tr-{len(transitions) + 1}",
                            from_step=prev_step_id,
                            to_step=impl_step_id,
                            condition="True",
                        )
                    )

                prev_step_id = impl_step_id

        # Step 3: Verification
        if self.verification_plan and self.verification_plan.test_cases:
            verify_step_id = _step_id()
            steps.append(
                StepDefinition(
                    id=verify_step_id,
                    name="Run Verification",
                    step_type="verification",
                    config={
                        "action": "verify",
                        "test_count": len(self.verification_plan.test_cases),
                        "critical_count": len(
                            self.verification_plan.get_by_priority(CasePriority.P0)
                        ),
                    },
                    description="Execute verification plan against implementation",
                )
            )
            if prev_step_id:
                transitions.append(
                    TransitionRule(
                        id=f"tr-{len(transitions) + 1}",
                        from_step=prev_step_id,
                        to_step=verify_step_id,
                        condition="True",
                    )
                )
            prev_step_id = verify_step_id

        # Step 4: Memory write-back (feedback loop)
        memory_step_id = _step_id()
        steps.append(
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
        if prev_step_id:
            transitions.append(
                TransitionRule(
                    id=f"tr-{len(transitions) + 1}",
                    from_step=prev_step_id,
                    to_step=memory_step_id,
                    condition="True",
                )
            )

        workflow_id = f"wf-{self.id}"
        self.workflow_id = workflow_id

        return WorkflowDefinition(
            id=workflow_id,
            name=f"Decision Plan: {self.task[:60]}",
            description=f"Auto-generated workflow from debate {self.debate_id}",
            steps=steps,
            transitions=transitions,
            category=WorkflowCategory.GENERAL,
            tags=["decision-plan", "auto-generated"],
            metadata={
                "decision_plan_id": self.id,
                "debate_id": self.debate_id,
                "debate_confidence": self.debate_result.confidence if self.debate_result else 0,
                "risk_count": len(self.risk_register.risks) if self.risk_register else 0,
            },
        )

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


class DecisionPlanFactory:
    """Factory for creating DecisionPlan from DebateResult.

    Generates all sub-artifacts (risk register, verification plan,
    implementation plan) from the debate result and its metadata.

    Usage:
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=5.00,
            approval_mode=ApprovalMode.RISK_BASED,
        )
    """

    @staticmethod
    def from_debate_result(
        result: DebateResult,
        *,
        budget_limit_usd: float | None = None,
        approval_mode: ApprovalMode = ApprovalMode.RISK_BASED,
        max_auto_risk: RiskLevel = RiskLevel.LOW,
        repo_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionPlan:
        """Create a DecisionPlan from a DebateResult.

        This is the primary entry point for the gold path. It:
        1. Analyzes the debate result for risks
        2. Generates a verification plan
        3. Decomposes the conclusion into implementation tasks
        4. Sets up budget tracking from debate costs
        5. Configures approval based on risk assessment

        Args:
            result: The completed DebateResult from Arena.run().
            budget_limit_usd: Optional budget cap for the full plan.
            approval_mode: How human approval is determined.
            max_auto_risk: Max risk level for auto-execution.
            repo_path: Repository root for implementation planning.
            metadata: Additional metadata to attach.

        Returns:
            A fully populated DecisionPlan ready for approval/execution.
        """
        plan = DecisionPlan(
            debate_id=result.debate_id,
            task=result.task,
            debate_result=result,
            approval_mode=approval_mode,
            max_auto_risk=max_auto_risk,
            metadata=metadata or {},
        )

        # Budget setup
        plan.budget = BudgetAllocation(
            limit_usd=budget_limit_usd,
            debate_cost_usd=result.total_cost_usd,
            spent_usd=result.total_cost_usd,
        )

        # Risk analysis from debate result
        plan.risk_register = DecisionPlanFactory._build_risk_register(result)

        # Verification plan from debate result
        plan.verification_plan = DecisionPlanFactory._build_verification_plan(result)

        # Implementation plan from debate conclusion
        plan.implement_plan = DecisionPlanFactory._build_implement_plan(result, repo_path)

        # Set status based on approval needs
        if plan.requires_human_approval:
            plan.status = PlanStatus.AWAITING_APPROVAL
        else:
            plan.status = PlanStatus.APPROVED

        return plan

    @staticmethod
    def _build_risk_register(result: DebateResult) -> RiskRegister:
        """Build risk register directly from DebateResult."""
        register = RiskRegister(debate_id=result.debate_id)

        # Low confidence → risk
        if result.confidence < 0.7:
            register.add_risk(
                Risk(
                    id=f"risk-confidence-{result.debate_id[:8]}",
                    title="Low consensus confidence",
                    description=(
                        f"Debate reached {result.confidence:.0%} confidence. "
                        "Implementation may face challenges or require revision."
                    ),
                    level=RiskLevel.MEDIUM if result.confidence >= 0.5 else RiskLevel.HIGH,
                    category=RiskCategory.UNKNOWN,
                    source="consensus_analysis",
                    impact=0.6,
                    likelihood=1.0 - result.confidence,
                )
            )

        # No consensus → risk
        if not result.consensus_reached:
            register.add_risk(
                Risk(
                    id=f"risk-no-consensus-{result.debate_id[:8]}",
                    title="No consensus reached",
                    description="Agents did not reach consensus. Decision may be contested.",
                    level=RiskLevel.HIGH,
                    category=RiskCategory.UNKNOWN,
                    source="consensus_analysis",
                    impact=0.8,
                    likelihood=0.7,
                )
            )

        # High-severity critiques → risks
        for i, critique in enumerate(result.critiques):
            if critique.severity >= 7.0:
                for j, issue in enumerate(critique.issues[:2]):
                    register.add_risk(
                        Risk(
                            id=f"risk-critique-{i}-{j}",
                            title=issue[:60],
                            description=issue,
                            level=RiskLevel.HIGH if critique.severity >= 8.0 else RiskLevel.MEDIUM,
                            category=_categorize_issue(issue),
                            source=f"critique:{critique.agent}",
                            impact=critique.severity / 10.0,
                            likelihood=0.7,
                            mitigation=", ".join(critique.suggestions[:2]),
                        )
                    )

        # Dissenting views → risks
        for i, view in enumerate(result.dissenting_views[:3]):
            register.add_risk(
                Risk(
                    id=f"risk-dissent-{i}",
                    title=f"Dissenting view: {view[:50]}",
                    description=view,
                    level=RiskLevel.MEDIUM,
                    category=RiskCategory.UNKNOWN,
                    source="dissent_analysis",
                    impact=0.5,
                    likelihood=0.4,
                )
            )

        # Debate cruxes → risks
        for i, crux in enumerate(result.debate_cruxes[:3]):
            claim = crux.get("claim", crux.get("text", "Unknown crux"))
            register.add_risk(
                Risk(
                    id=f"risk-crux-{i}",
                    title=f"Unresolved crux: {str(claim)[:50]}",
                    description=f"Key disagreement driver: {claim}",
                    level=RiskLevel.MEDIUM,
                    category=RiskCategory.TECHNICAL,
                    source="belief_network",
                    impact=0.5,
                    likelihood=0.5,
                )
            )

        return register

    @staticmethod
    def _build_verification_plan(result: DebateResult) -> VerificationPlan:
        """Build verification plan directly from DebateResult."""
        plan = VerificationPlan(
            debate_id=result.debate_id,
            title=f"Verify: {result.task[:60]}",
            description=f"Verification plan for debate {result.debate_id}",
        )

        # Extract testable claims from final answer
        test_num = 1
        if result.final_answer:
            for line in result.final_answer.split("\n"):
                line = line.strip()
                if not line or len(line) < 15:
                    continue
                keywords = ["implement", "use", "add", "create", "ensure", "should", "must"]
                if any(kw in line.lower() for kw in keywords):
                    plan.add_test(
                        VerificationCase(
                            id=f"consensus-{test_num}",
                            title=f"Verify: {line[:50]}",
                            description=f"Confirm implementation satisfies: {line}",
                            test_type=VerificationType.INTEGRATION,
                            priority=CasePriority.P1,
                            steps=[
                                "Set up environment",
                                "Execute functionality",
                                "Verify expected behavior",
                            ],
                            expected_result="Functionality works as described",
                        )
                    )
                    test_num += 1
                    if test_num > 5:
                        break

        # Edge cases from high-severity critiques
        for i, critique in enumerate(result.critiques[:5]):
            if critique.severity >= 5.0:
                for j, issue in enumerate(critique.issues[:1]):
                    plan.add_test(
                        VerificationCase(
                            id=f"critique-edge-{i}-{j}",
                            title=f"Edge case: {issue[:50]}",
                            description=f"Verify handling of: {issue}",
                            test_type=VerificationType.UNIT,
                            priority=CasePriority.P2,
                            steps=["Set up edge case", "Execute", "Verify graceful handling"],
                            expected_result="Edge case handled",
                        )
                    )

        # Smoke test
        plan.add_test(
            VerificationCase(
                id="smoke-1",
                title="Smoke test: Basic functionality",
                description="Verify basic functionality after implementation",
                test_type=VerificationType.E2E,
                priority=CasePriority.P0,
                steps=["Deploy changes", "Execute happy path", "Verify success"],
                expected_result="Basic use case succeeds",
            )
        )

        # Regression
        plan.add_test(
            VerificationCase(
                id="regression-1",
                title="Regression: Existing functionality",
                description="Verify no regressions in existing functionality",
                test_type=VerificationType.REGRESSION,
                priority=CasePriority.P1,
                steps=["Run existing test suite", "Verify all pass"],
                expected_result="No regressions",
            )
        )

        return plan

    @staticmethod
    def _build_implement_plan(result: DebateResult, repo_path: Path | None = None) -> ImplementPlan:
        """Build implementation plan from debate conclusion.

        Uses heuristic extraction from the final answer. For richer
        decomposition, callers should use generate_implement_plan()
        from aragora.implement.planner with an LLM.
        """
        design = result.final_answer or result.task
        design_hash = hashlib.sha256(design.encode()).hexdigest()

        tasks: list[ImplementTask] = []
        task_num = 1

        # Extract numbered steps from the final answer
        if result.final_answer:
            for line in result.final_answer.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match numbered items or bullet points with action verbs
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                    clean = line.lstrip("0123456789.-*) ").strip()
                    if len(clean) > 15:
                        # Infer file paths mentioned
                        import re

                        files = re.findall(r"`([a-zA-Z0-9_/\-\.]+\.[a-z]+)`", clean)

                        tasks.append(
                            ImplementTask(
                                id=f"task-{task_num}",
                                description=clean[:200],
                                files=files[:5],
                                complexity="moderate",
                                dependencies=[f"task-{task_num - 1}"] if task_num > 1 else [],
                            )
                        )
                        task_num += 1
                        if task_num > 10:
                            break

        # Fallback: single task if no structured steps found
        if not tasks:
            tasks.append(
                ImplementTask(
                    id="task-1",
                    description="Implement the debated solution",
                    files=[],
                    complexity="complex",
                    dependencies=[],
                )
            )

        return ImplementPlan(design_hash=design_hash, tasks=tasks)


def _categorize_issue(issue: str) -> RiskCategory:
    """Categorize a risk issue by keywords."""
    lower = issue.lower()
    if any(k in lower for k in ["security", "auth", "permission", "vulnerable", "injection"]):
        return RiskCategory.SECURITY
    if any(k in lower for k in ["performance", "slow", "latency", "speed", "timeout"]):
        return RiskCategory.PERFORMANCE
    if any(k in lower for k in ["scale", "load", "capacity", "throughput"]):
        return RiskCategory.SCALABILITY
    if any(k in lower for k in ["maintain", "complex", "readab", "test", "debt"]):
        return RiskCategory.MAINTAINABILITY
    if any(k in lower for k in ["compat", "version", "depend", "integrat", "migrat"]):
        return RiskCategory.COMPATIBILITY
    return RiskCategory.TECHNICAL


# =========================================================================
# Memory feedback loop
# =========================================================================


@dataclass
class PlanOutcome:
    """Records the outcome of executing a DecisionPlan.

    This is the data structure that feeds back into organizational memory,
    closing the gold-path loop: debate → plan → execute → verify → LEARN.
    """

    plan_id: str
    debate_id: str
    task: str
    success: bool
    tasks_completed: int = 0
    tasks_total: int = 0
    verification_passed: int = 0
    verification_total: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None
    duration_seconds: float = 0.0
    lessons: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        if self.tasks_total == 0:
            return 0.0
        return self.tasks_completed / self.tasks_total

    @property
    def verification_rate(self) -> float:
        if self.verification_total == 0:
            return 0.0
        return self.verification_passed / self.verification_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "debate_id": self.debate_id,
            "task": self.task,
            "success": self.success,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "verification_passed": self.verification_passed,
            "verification_total": self.verification_total,
            "total_cost_usd": self.total_cost_usd,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "completion_rate": self.completion_rate,
            "verification_rate": self.verification_rate,
            "lessons": self.lessons,
        }

    def to_memory_content(self) -> str:
        """Format outcome as structured text for memory storage."""
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [
            f"[Decision Plan Outcome: {status}]",
            f"Task: {self.task}",
            f"Debate: {self.debate_id}",
            f"Completion: {self.tasks_completed}/{self.tasks_total} tasks",
            f"Verification: {self.verification_passed}/{self.verification_total} cases",
            f"Cost: ${self.total_cost_usd:.4f}",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.lessons:
            lines.append("Lessons learned:")
            for lesson in self.lessons:
                lines.append(f"  - {lesson}")
        return "\n".join(lines)


async def record_plan_outcome(
    plan: DecisionPlan,
    outcome: PlanOutcome,
    *,
    continuum_memory: Any | None = None,
    knowledge_mound: Any | None = None,
) -> dict[str, Any]:
    """Write implementation outcome back to organizational memory.

    This closes the gold-path feedback loop by recording what happened
    when a debate's conclusions were actually implemented.

    The recorded data feeds future debates via:
    - ContinuumMemory: Pattern learning (what debate patterns lead to
      successful implementations)
    - KnowledgeMound: Organizational knowledge (lessons learned,
      successful approaches)

    Args:
        plan: The executed DecisionPlan.
        outcome: The PlanOutcome recording what happened.
        continuum_memory: Optional ContinuumMemory instance.
        knowledge_mound: Optional KnowledgeMound instance.

    Returns:
        Dict with write results: {continuum_id, mound_id, errors}.
    """
    import logging

    logger = logging.getLogger(__name__)
    results: dict[str, Any] = {"continuum_id": None, "mound_id": None, "errors": []}

    # Update plan state
    plan.status = PlanStatus.COMPLETED if outcome.success else PlanStatus.FAILED
    plan.execution_completed_at = datetime.now()
    plan.execution_error = outcome.error
    plan.budget.spent_usd = outcome.total_cost_usd

    # 1. Write to ContinuumMemory (pattern learning)
    if continuum_memory is not None:
        try:
            # Determine importance based on outcome
            # Failures are more important to learn from than successes
            importance = 0.8 if not outcome.success else 0.6
            # Low verification rates are surprising and important
            if outcome.verification_rate < 0.5:
                importance = min(1.0, importance + 0.2)

            from aragora.memory.continuum import MemoryTier

            entry = await continuum_memory.store(
                key=f"plan_outcome:{plan.id}",
                content=outcome.to_memory_content(),
                tier=MemoryTier.SLOW,  # Cross-session learning
                importance=importance,
                metadata={
                    "type": "plan_outcome",
                    "plan_id": plan.id,
                    "debate_id": plan.debate_id,
                    "success": outcome.success,
                    "completion_rate": outcome.completion_rate,
                    "verification_rate": outcome.verification_rate,
                    "cost_usd": outcome.total_cost_usd,
                },
            )
            results["continuum_id"] = entry.id
            logger.info(
                "Recorded plan outcome to ContinuumMemory: %s (importance=%.2f)",
                entry.id,
                importance,
            )

            # Update the original debate memory entry with outcome data
            if plan.debate_result and plan.debate_result.debate_id:
                try:
                    continuum_memory.update_outcome(
                        id=f"debate:{plan.debate_result.debate_id}",
                        success=outcome.success,
                        agent_prediction_error=1.0 - outcome.completion_rate,
                    )
                except Exception:
                    pass  # Best-effort update of original debate entry

        except Exception as e:
            err = f"ContinuumMemory write failed: {e}"
            logger.warning(err)
            results["errors"].append(err)

    # 2. Write to KnowledgeMound (organizational knowledge)
    if knowledge_mound is not None:
        try:
            confidence = outcome.verification_rate if outcome.success else 0.3

            item_id = await knowledge_mound.store_knowledge(
                content=outcome.to_memory_content(),
                source="decision_plan",
                source_id=plan.id,
                confidence=confidence,
                metadata={
                    "type": "implementation_outcome",
                    "plan_id": plan.id,
                    "debate_id": plan.debate_id,
                    "task": plan.task[:200],
                    "success": outcome.success,
                    "lessons": outcome.lessons[:5],
                },
            )
            results["mound_id"] = item_id
            logger.info("Recorded plan outcome to KnowledgeMound: %s", item_id)

        except Exception as e:
            err = f"KnowledgeMound write failed: {e}"
            logger.warning(err)
            results["errors"].append(err)

    # Mark feedback written
    plan.memory_written = len(results["errors"]) == 0

    return results


__all__ = [
    "ApprovalMode",
    "ApprovalRecord",
    "BudgetAllocation",
    "DecisionPlan",
    "DecisionPlanFactory",
    "PlanOutcome",
    "PlanStatus",
    "record_plan_outcome",
]

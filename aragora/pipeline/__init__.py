"""
Pipeline module for aragora - Decision-to-implementation artifacts.

Transforms debate outcomes into actionable development artifacts:
- DecisionPlan: Full gold-path bridge from DebateResult to executable workflow
- DecisionMemo: Summary of debate conclusions
- RiskRegister: Identified risks and mitigations
- VerificationPlan: Verification strategy
- PatchPlan: Implementation steps
- DecisionIntegrityPackage: Receipt + implementation plan bundle
"""

from aragora.pipeline.decision_integrity import (
    ContextSnapshot,
    DecisionIntegrityPackage,
    build_decision_integrity_package,
    capture_context_snapshot,
)
from aragora.pipeline.decision_plan import (
    ApprovalMode,
    ApprovalRecord,
    BudgetAllocation,
    DecisionPlan,
    DecisionPlanFactory,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)
from aragora.pipeline.execution_notifier import ExecutionNotifier, ExecutionProgress
from aragora.pipeline.notifications import (
    notify_plan_created,
    notify_plan_approved,
    notify_plan_rejected,
    notify_execution_started,
    notify_execution_completed,
    notify_execution_failed,
)
from aragora.pipeline.executor import PlanExecutor, get_plan, list_plans, store_plan
from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)
from aragora.pipeline.pr_generator import DecisionMemo, PatchPlan, PRGenerator
from aragora.pipeline.risk_register import Risk, RiskRegister
from aragora.pipeline.verification_plan import (
    VerificationCase,
    VerificationPlan,
    VerificationPlanGenerator,
)

# Backward compatibility aliases (old names triggered pytest discovery)
TestPlan = VerificationPlan
TestCase = VerificationCase
TestPlanGenerator = VerificationPlanGenerator

__all__ = [
    # Gold path
    "DecisionPlan",
    "DecisionPlanFactory",
    "PlanStatus",
    "PlanOutcome",
    "ApprovalMode",
    "ApprovalRecord",
    "BudgetAllocation",
    "record_plan_outcome",
    # Execution notifications
    "ExecutionNotifier",
    "ExecutionProgress",
    # Plan lifecycle notifications
    "notify_plan_created",
    "notify_plan_approved",
    "notify_plan_rejected",
    "notify_execution_started",
    "notify_execution_completed",
    "notify_execution_failed",
    # Executor
    "PlanExecutor",
    "get_plan",
    "list_plans",
    "store_plan",
    # Decision integrity
    "ContextSnapshot",
    "DecisionIntegrityPackage",
    "build_decision_integrity_package",
    "capture_context_snapshot",
    "PRGenerator",
    "DecisionMemo",
    "PatchPlan",
    "RiskRegister",
    "Risk",
    "VerificationPlan",
    "VerificationCase",
    # Idea-to-Execution pipeline
    "IdeaToExecutionPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StageResult",
    # Backward compatibility
    "TestPlan",
    "TestCase",
]

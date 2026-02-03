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
from aragora.pipeline.executor import PlanExecutor, get_plan, list_plans, store_plan
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
    # Backward compatibility
    "TestPlan",
    "TestCase",
]

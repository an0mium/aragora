"""
Decision Plan - Bridge from DebateResult to executable implementation.

The DecisionPlan is the central artifact in the gold path:
    input -> debate -> DECISION_PLAN -> implementation -> verification -> learning

Modules:
- core: DecisionPlan class and supporting types
- factory: DecisionPlanFactory for creating plans from debate results
- memory: PlanOutcome and feedback loop for organizational learning
"""

from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    ApprovalRecord,
    BudgetAllocation,
    DecisionPlan,
    PlanStatus,
)
from aragora.pipeline.decision_plan.factory import DecisionPlanFactory
from aragora.pipeline.decision_plan.memory import PlanOutcome, record_plan_outcome

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

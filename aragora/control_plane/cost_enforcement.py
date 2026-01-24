"""
Cost-Aware Scheduling with Budget Enforcement.

Provides cost constraints for the TaskScheduler:
- Pre-submission budget validation
- Cost estimation for tasks
- Auto-throttle when approaching limits
- Budget-aware priority adjustment
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from aragora.observability import get_logger

if TYPE_CHECKING:
    from aragora.billing.cost_tracker import CostTracker

logger = get_logger(__name__)


class CostEnforcementMode(str, Enum):
    """How to handle cost limit violations."""

    HARD = "hard"  # Reject task immediately
    SOFT = "soft"  # Allow with warning
    THROTTLE = "throttle"  # Queue with reduced priority
    ESTIMATE = "estimate"  # Warn based on estimated cost


class ThrottleLevel(str, Enum):
    """Throttle severity based on budget consumption."""

    NONE = "none"  # < 50% of budget
    LIGHT = "light"  # 50-75% - reduce priority slightly
    MEDIUM = "medium"  # 75-90% - reduce priority moderately
    HEAVY = "heavy"  # 90-100% - reduce priority significantly
    BLOCKED = "blocked"  # > 100% - block new tasks


@dataclass
class CostEstimate:
    """Estimated cost for a task before execution."""

    task_type: str
    estimated_cost_usd: Decimal
    estimated_tokens: int
    confidence: float  # 0.0-1.0, how confident we are in estimate
    based_on_samples: int  # How many historical samples used
    model_suggestion: Optional[str] = None  # Cheaper model that could work
    estimated_savings_usd: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "estimated_cost_usd": str(self.estimated_cost_usd),
            "estimated_tokens": self.estimated_tokens,
            "confidence": self.confidence,
            "based_on_samples": self.based_on_samples,
            "model_suggestion": self.model_suggestion,
            "estimated_savings_usd": str(self.estimated_savings_usd)
            if self.estimated_savings_usd
            else None,
        }


@dataclass
class CostConstraintResult:
    """Result of a cost constraint check."""

    allowed: bool
    reason: Optional[str] = None
    throttle_level: ThrottleLevel = ThrottleLevel.NONE
    enforcement_mode: CostEnforcementMode = CostEnforcementMode.SOFT
    budget_percentage_used: float = 0.0
    remaining_budget_usd: Optional[Decimal] = None
    estimated_cost: Optional[CostEstimate] = None
    priority_adjustment: int = 0  # Negative = lower priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "throttle_level": self.throttle_level.value,
            "enforcement_mode": self.enforcement_mode.value,
            "budget_percentage_used": self.budget_percentage_used,
            "remaining_budget_usd": str(self.remaining_budget_usd)
            if self.remaining_budget_usd
            else None,
            "estimated_cost": self.estimated_cost.to_dict() if self.estimated_cost else None,
            "priority_adjustment": self.priority_adjustment,
        }


# Default cost estimates by task type (USD per task)
DEFAULT_TASK_COST_ESTIMATES: Dict[str, Decimal] = {
    "debate": Decimal("0.50"),
    "analysis": Decimal("0.10"),
    "summary": Decimal("0.05"),
    "critique": Decimal("0.15"),
    "consensus": Decimal("0.20"),
    "search": Decimal("0.02"),
    "embedding": Decimal("0.01"),
    "default": Decimal("0.10"),
}


@dataclass
class CostEnforcementConfig:
    """Configuration for cost enforcement."""

    # Enforcement mode
    mode: CostEnforcementMode = CostEnforcementMode.THROTTLE

    # Throttle thresholds (percentage of budget)
    throttle_light_threshold: float = 50.0
    throttle_medium_threshold: float = 75.0
    throttle_heavy_threshold: float = 90.0
    throttle_block_threshold: float = 100.0

    # Priority adjustments for throttle levels
    light_priority_adjustment: int = -1
    medium_priority_adjustment: int = -2
    heavy_priority_adjustment: int = -3

    # Cost estimation
    enable_cost_estimation: bool = True
    min_samples_for_estimate: int = 5
    estimate_confidence_threshold: float = 0.7

    # Alerts
    alert_on_throttle: bool = True
    alert_on_block: bool = True

    # Grace period after budget reset (hours)
    grace_period_hours: float = 1.0


class CostEnforcer:
    """
    Enforces cost constraints for task scheduling.

    Integrates with CostTracker to:
    1. Check budget before task submission
    2. Estimate task costs
    3. Apply throttling based on budget consumption
    4. Block tasks when budget exceeded (hard mode)
    """

    def __init__(
        self,
        cost_tracker: Optional["CostTracker"] = None,
        config: Optional[CostEnforcementConfig] = None,
    ):
        """
        Initialize cost enforcer.

        Args:
            cost_tracker: CostTracker instance for budget info
            config: Enforcement configuration
        """
        self._cost_tracker = cost_tracker
        self._config = config or CostEnforcementConfig()

        # Historical cost data for estimation (task_type -> list of costs)
        self._task_cost_history: Dict[str, List[Decimal]] = {}
        self._max_history_per_type = 100

        # Callbacks for enforcement events
        self._throttle_callbacks: List[Callable[[str, ThrottleLevel], None]] = []
        self._block_callbacks: List[Callable[[str, str], None]] = []

        logger.info(
            "CostEnforcer initialized",
            mode=self._config.mode.value,
            throttle_thresholds={
                "light": self._config.throttle_light_threshold,
                "medium": self._config.throttle_medium_threshold,
                "heavy": self._config.throttle_heavy_threshold,
                "block": self._config.throttle_block_threshold,
            },
        )

    def set_cost_tracker(self, cost_tracker: "CostTracker") -> None:
        """Set the cost tracker for budget lookups."""
        self._cost_tracker = cost_tracker
        logger.info("CostEnforcer connected to CostTracker")

    def check_budget_constraint(
        self,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        task_type: str = "default",
        estimated_cost: Optional[Decimal] = None,
    ) -> CostConstraintResult:
        """
        Check if a task can be submitted given budget constraints.

        Args:
            workspace_id: Workspace to check budget for
            org_id: Organization to check budget for
            task_type: Type of task for cost estimation
            estimated_cost: Optional pre-calculated cost estimate

        Returns:
            CostConstraintResult with decision and details
        """
        # No cost tracker = allow all
        if not self._cost_tracker:
            return CostConstraintResult(allowed=True)

        # Get budget for workspace/org
        budget = self._cost_tracker.get_budget(workspace_id=workspace_id, org_id=org_id)

        # No budget configured = allow all
        if not budget:
            return CostConstraintResult(allowed=True)

        # Calculate budget usage percentage
        if budget.monthly_limit_usd and budget.monthly_limit_usd > 0:
            percentage = float((budget.current_monthly_spend / budget.monthly_limit_usd) * 100)
            remaining = budget.monthly_limit_usd - budget.current_monthly_spend
        elif budget.daily_limit_usd and budget.daily_limit_usd > 0:
            percentage = float((budget.current_daily_spend / budget.daily_limit_usd) * 100)
            remaining = budget.daily_limit_usd - budget.current_daily_spend
        else:
            return CostConstraintResult(allowed=True)

        # Get cost estimate
        cost_estimate = self._estimate_task_cost(task_type, estimated_cost)

        # Determine throttle level
        throttle_level = self._get_throttle_level(percentage)

        # Calculate priority adjustment
        priority_adjustment = self._get_priority_adjustment(throttle_level)

        # Determine if allowed based on mode
        if self._config.mode == CostEnforcementMode.HARD:
            # Hard mode: block if over budget
            if percentage >= self._config.throttle_block_threshold:
                self._notify_block(workspace_id or org_id or "unknown", "Budget exceeded")
                return CostConstraintResult(
                    allowed=False,
                    reason=f"Budget exceeded ({percentage:.1f}%)",
                    throttle_level=ThrottleLevel.BLOCKED,
                    enforcement_mode=CostEnforcementMode.HARD,
                    budget_percentage_used=percentage,
                    remaining_budget_usd=remaining if remaining > 0 else Decimal("0"),
                    estimated_cost=cost_estimate,
                )

        elif self._config.mode == CostEnforcementMode.THROTTLE:
            # Throttle mode: adjust priority based on usage
            if throttle_level == ThrottleLevel.BLOCKED:
                self._notify_block(
                    workspace_id or org_id or "unknown",
                    "Budget exceeded - task blocked",
                )
                return CostConstraintResult(
                    allowed=False,
                    reason=f"Budget exceeded ({percentage:.1f}%) - tasks blocked until budget resets",
                    throttle_level=ThrottleLevel.BLOCKED,
                    enforcement_mode=CostEnforcementMode.THROTTLE,
                    budget_percentage_used=percentage,
                    remaining_budget_usd=Decimal("0"),
                    estimated_cost=cost_estimate,
                )

            if throttle_level != ThrottleLevel.NONE:
                self._notify_throttle(workspace_id or org_id or "unknown", throttle_level)

        # Allowed with possible throttling
        reason = None
        if throttle_level != ThrottleLevel.NONE:
            reason = f"Throttled ({throttle_level.value}): {percentage:.1f}% of budget used"

        return CostConstraintResult(
            allowed=True,
            reason=reason,
            throttle_level=throttle_level,
            enforcement_mode=self._config.mode,
            budget_percentage_used=percentage,
            remaining_budget_usd=remaining if remaining > 0 else Decimal("0"),
            estimated_cost=cost_estimate,
            priority_adjustment=priority_adjustment,
        )

    def _get_throttle_level(self, percentage: float) -> ThrottleLevel:
        """Determine throttle level based on budget percentage used."""
        if percentage >= self._config.throttle_block_threshold:
            return ThrottleLevel.BLOCKED
        elif percentage >= self._config.throttle_heavy_threshold:
            return ThrottleLevel.HEAVY
        elif percentage >= self._config.throttle_medium_threshold:
            return ThrottleLevel.MEDIUM
        elif percentage >= self._config.throttle_light_threshold:
            return ThrottleLevel.LIGHT
        return ThrottleLevel.NONE

    def _get_priority_adjustment(self, throttle_level: ThrottleLevel) -> int:
        """Get priority adjustment for throttle level."""
        if throttle_level == ThrottleLevel.LIGHT:
            return self._config.light_priority_adjustment
        elif throttle_level == ThrottleLevel.MEDIUM:
            return self._config.medium_priority_adjustment
        elif throttle_level == ThrottleLevel.HEAVY:
            return self._config.heavy_priority_adjustment
        return 0

    def _estimate_task_cost(
        self, task_type: str, provided_cost: Optional[Decimal] = None
    ) -> Optional[CostEstimate]:
        """Estimate cost for a task type."""
        if not self._config.enable_cost_estimation:
            return None

        if provided_cost is not None:
            return CostEstimate(
                task_type=task_type,
                estimated_cost_usd=provided_cost,
                estimated_tokens=0,  # Unknown
                confidence=1.0,
                based_on_samples=0,
            )

        # Check historical data
        history = self._task_cost_history.get(task_type, [])
        if len(history) >= self._config.min_samples_for_estimate:
            avg_cost = float(sum(history)) / len(history)
            # Calculate confidence based on variance
            if len(history) > 1:
                variance = sum((float(c) - avg_cost) ** 2 for c in history) / len(history)
                std_dev = variance**0.5
                # Lower confidence if high variance
                confidence = min(1.0, 1.0 / (1.0 + std_dev / avg_cost))
            else:
                confidence = 0.5

            return CostEstimate(
                task_type=task_type,
                estimated_cost_usd=Decimal(str(avg_cost)),
                estimated_tokens=0,  # Would need token tracking
                confidence=confidence,
                based_on_samples=len(history),
            )

        # Fall back to default estimates
        default_cost = DEFAULT_TASK_COST_ESTIMATES.get(
            task_type, DEFAULT_TASK_COST_ESTIMATES["default"]
        )
        return CostEstimate(
            task_type=task_type,
            estimated_cost_usd=default_cost,
            estimated_tokens=0,
            confidence=0.3,  # Low confidence for defaults
            based_on_samples=0,
        )

    def record_task_cost(self, task_type: str, actual_cost: Decimal) -> None:
        """Record actual task cost for improving estimates."""
        if task_type not in self._task_cost_history:
            self._task_cost_history[task_type] = []

        history = self._task_cost_history[task_type]
        history.append(actual_cost)

        # Keep history bounded
        if len(history) > self._max_history_per_type:
            self._task_cost_history[task_type] = history[-self._max_history_per_type :]

    def add_throttle_callback(self, callback: Callable[[str, ThrottleLevel], None]) -> None:
        """Add callback for throttle events."""
        self._throttle_callbacks.append(callback)

    def add_block_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for block events."""
        self._block_callbacks.append(callback)

    def _notify_throttle(self, identifier: str, level: ThrottleLevel) -> None:
        """Notify throttle callbacks."""
        if not self._config.alert_on_throttle:
            return

        logger.warning(
            "Cost throttle applied",
            identifier=identifier,
            throttle_level=level.value,
        )

        for callback in self._throttle_callbacks:
            try:
                callback(identifier, level)
            except Exception as e:
                logger.error(f"Throttle callback error: {e}")

    def _notify_block(self, identifier: str, reason: str) -> None:
        """Notify block callbacks."""
        if not self._config.alert_on_block:
            return

        logger.error(
            "Cost block applied",
            identifier=identifier,
            reason=reason,
        )

        for callback in self._block_callbacks:
            try:
                callback(identifier, reason)
            except Exception as e:
                logger.error(f"Block callback error: {e}")

    def get_budget_status(
        self, workspace_id: Optional[str] = None, org_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current budget status for a workspace/org."""
        if not self._cost_tracker:
            return {"error": "No cost tracker configured"}

        budget = self._cost_tracker.get_budget(workspace_id=workspace_id, org_id=org_id)
        if not budget:
            return {"error": "No budget configured"}

        # Calculate percentage
        if budget.monthly_limit_usd and budget.monthly_limit_usd > 0:
            percentage = float((budget.current_monthly_spend / budget.monthly_limit_usd) * 100)
            remaining = budget.monthly_limit_usd - budget.current_monthly_spend
            limit = budget.monthly_limit_usd
            period = "monthly"
        elif budget.daily_limit_usd and budget.daily_limit_usd > 0:
            percentage = float((budget.current_daily_spend / budget.daily_limit_usd) * 100)
            remaining = budget.daily_limit_usd - budget.current_daily_spend
            limit = budget.daily_limit_usd
            period = "daily"
        else:
            return {
                "budget_configured": True,
                "limits_set": False,
            }

        throttle_level = self._get_throttle_level(percentage)

        return {
            "budget_id": budget.id,
            "workspace_id": workspace_id,
            "org_id": org_id,
            "period": period,
            "limit_usd": str(limit),
            "spent_usd": str(
                budget.current_monthly_spend if period == "monthly" else budget.current_daily_spend
            ),
            "remaining_usd": str(remaining) if remaining > 0 else "0",
            "percentage_used": round(percentage, 2),
            "throttle_level": throttle_level.value,
            "enforcement_mode": self._config.mode.value,
        }


class BudgetAwareSchedulerMixin:
    """
    Mixin for TaskScheduler to add cost-aware scheduling.

    Usage:
        class CostAwareScheduler(BudgetAwareSchedulerMixin, TaskScheduler):
            pass

    Or monkey-patch existing scheduler:
        scheduler._cost_enforcer = CostEnforcer(cost_tracker)
    """

    _cost_enforcer: Optional[CostEnforcer] = None

    def set_cost_enforcer(self, enforcer: CostEnforcer) -> None:
        """Set the cost enforcer for budget checks."""
        self._cost_enforcer = enforcer

    async def check_cost_constraint_before_submit(
        self,
        task_type: str,
        workspace_id: Optional[str] = None,
        estimated_cost: Optional[Decimal] = None,
    ) -> CostConstraintResult:
        """
        Check cost constraints before submitting a task.

        Should be called by submit() before accepting the task.
        """
        if not self._cost_enforcer:
            return CostConstraintResult(allowed=True)

        return self._cost_enforcer.check_budget_constraint(
            workspace_id=workspace_id,
            task_type=task_type,
            estimated_cost=estimated_cost,
        )


# Exception for cost-related rejections
class CostLimitExceededError(Exception):
    """Raised when a task is rejected due to cost limits."""

    def __init__(self, result: CostConstraintResult, task_type: str):
        self.result = result
        self.task_type = task_type
        super().__init__(f"Cost limit exceeded for task '{task_type}': {result.reason}")

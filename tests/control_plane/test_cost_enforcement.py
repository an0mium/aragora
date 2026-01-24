"""Tests for cost-aware scheduling with budget enforcement."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aragora.control_plane.cost_enforcement import (
    CostEnforcer,
    CostEnforcementConfig,
    CostEnforcementMode,
    CostEstimate,
    CostConstraintResult,
    CostLimitExceededError,
    ThrottleLevel,
    DEFAULT_TASK_COST_ESTIMATES,
)


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_to_dict(self):
        """Test CostEstimate serialization."""
        estimate = CostEstimate(
            task_type="debate",
            estimated_cost_usd=Decimal("0.50"),
            estimated_tokens=1000,
            confidence=0.85,
            based_on_samples=10,
            model_suggestion="claude-haiku",
            estimated_savings_usd=Decimal("0.30"),
        )

        result = estimate.to_dict()

        assert result["task_type"] == "debate"
        assert result["estimated_cost_usd"] == "0.50"
        assert result["estimated_tokens"] == 1000
        assert result["confidence"] == 0.85
        assert result["based_on_samples"] == 10
        assert result["model_suggestion"] == "claude-haiku"
        assert result["estimated_savings_usd"] == "0.30"

    def test_to_dict_no_savings(self):
        """Test CostEstimate serialization without savings."""
        estimate = CostEstimate(
            task_type="search",
            estimated_cost_usd=Decimal("0.02"),
            estimated_tokens=100,
            confidence=0.5,
            based_on_samples=0,
        )

        result = estimate.to_dict()

        assert result["model_suggestion"] is None
        assert result["estimated_savings_usd"] is None


class TestCostConstraintResult:
    """Tests for CostConstraintResult dataclass."""

    def test_to_dict_allowed(self):
        """Test allowed result serialization."""
        result = CostConstraintResult(
            allowed=True,
            throttle_level=ThrottleLevel.LIGHT,
            enforcement_mode=CostEnforcementMode.THROTTLE,
            budget_percentage_used=55.0,
            remaining_budget_usd=Decimal("450.00"),
            priority_adjustment=-1,
        )

        data = result.to_dict()

        assert data["allowed"] is True
        assert data["throttle_level"] == "light"
        assert data["enforcement_mode"] == "throttle"
        assert data["budget_percentage_used"] == 55.0
        assert data["remaining_budget_usd"] == "450.00"
        assert data["priority_adjustment"] == -1

    def test_to_dict_blocked(self):
        """Test blocked result serialization."""
        result = CostConstraintResult(
            allowed=False,
            reason="Budget exceeded (102%)",
            throttle_level=ThrottleLevel.BLOCKED,
            enforcement_mode=CostEnforcementMode.HARD,
            budget_percentage_used=102.0,
            remaining_budget_usd=Decimal("0"),
        )

        data = result.to_dict()

        assert data["allowed"] is False
        assert data["reason"] == "Budget exceeded (102%)"
        assert data["throttle_level"] == "blocked"


class TestCostEnforcer:
    """Tests for CostEnforcer class."""

    def test_init_default_config(self):
        """Test CostEnforcer initialization with default config."""
        enforcer = CostEnforcer()

        assert enforcer._config.mode == CostEnforcementMode.THROTTLE
        assert enforcer._config.throttle_light_threshold == 50.0
        assert enforcer._config.throttle_medium_threshold == 75.0
        assert enforcer._config.throttle_heavy_threshold == 90.0
        assert enforcer._config.throttle_block_threshold == 100.0

    def test_init_custom_config(self):
        """Test CostEnforcer with custom config."""
        config = CostEnforcementConfig(
            mode=CostEnforcementMode.HARD,
            throttle_block_threshold=95.0,
        )
        enforcer = CostEnforcer(config=config)

        assert enforcer._config.mode == CostEnforcementMode.HARD
        assert enforcer._config.throttle_block_threshold == 95.0

    def test_check_budget_no_tracker(self):
        """Test budget check without cost tracker - allows all."""
        enforcer = CostEnforcer()

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.NONE

    def test_check_budget_no_budget_configured(self):
        """Test budget check when no budget exists for workspace."""
        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = None

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        cost_tracker.get_budget.assert_called_once_with(workspace_id="ws-123", org_id=None)

    def test_check_budget_under_threshold(self):
        """Test budget check when under all thresholds."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("300")  # 30%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.NONE
        assert result.budget_percentage_used == 30.0
        assert result.remaining_budget_usd == Decimal("700")
        assert result.priority_adjustment == 0

    def test_check_budget_light_throttle(self):
        """Test budget check in light throttle zone (50-75%)."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("600")  # 60%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.LIGHT
        assert result.priority_adjustment == -1
        assert "Throttled" in (result.reason or "")

    def test_check_budget_medium_throttle(self):
        """Test budget check in medium throttle zone (75-90%)."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("800")  # 80%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.MEDIUM
        assert result.priority_adjustment == -2

    def test_check_budget_heavy_throttle(self):
        """Test budget check in heavy throttle zone (90-100%)."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("950")  # 95%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.HEAVY
        assert result.priority_adjustment == -3

    def test_check_budget_blocked_throttle_mode(self):
        """Test budget check when blocked in throttle mode."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("1050")  # 105%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is False
        assert result.throttle_level == ThrottleLevel.BLOCKED
        assert "exceeded" in (result.reason or "").lower()

    def test_check_budget_blocked_hard_mode(self):
        """Test budget check when blocked in hard mode."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("1050")  # 105%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        config = CostEnforcementConfig(mode=CostEnforcementMode.HARD)
        enforcer = CostEnforcer(cost_tracker=cost_tracker, config=config)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is False
        assert result.enforcement_mode == CostEnforcementMode.HARD

    def test_check_budget_uses_daily_limit(self):
        """Test budget check uses daily limit when monthly not set."""
        budget = MagicMock()
        budget.monthly_limit_usd = None
        budget.daily_limit_usd = Decimal("100")
        budget.current_daily_spend = Decimal("80")  # 80%

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        result = enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.MEDIUM
        assert result.budget_percentage_used == 80.0
        assert result.remaining_budget_usd == Decimal("20")

    def test_estimate_task_cost_default(self):
        """Test task cost estimation with defaults."""
        enforcer = CostEnforcer()

        estimate = enforcer._estimate_task_cost("debate", None)

        assert estimate is not None
        assert estimate.task_type == "debate"
        assert estimate.estimated_cost_usd == DEFAULT_TASK_COST_ESTIMATES["debate"]
        assert estimate.confidence == 0.3  # Low confidence for defaults
        assert estimate.based_on_samples == 0

    def test_estimate_task_cost_provided(self):
        """Test task cost estimation with provided cost."""
        enforcer = CostEnforcer()

        estimate = enforcer._estimate_task_cost("debate", Decimal("0.75"))

        assert estimate is not None
        assert estimate.estimated_cost_usd == Decimal("0.75")
        assert estimate.confidence == 1.0
        assert estimate.based_on_samples == 0

    def test_estimate_task_cost_from_history(self):
        """Test task cost estimation from historical data."""
        enforcer = CostEnforcer()

        # Record some historical costs
        for _ in range(10):
            enforcer.record_task_cost("debate", Decimal("0.45"))

        estimate = enforcer._estimate_task_cost("debate", None)

        assert estimate is not None
        assert estimate.estimated_cost_usd == Decimal("0.45")
        assert estimate.based_on_samples == 10
        assert estimate.confidence > 0.5

    def test_record_task_cost_history_bounded(self):
        """Test that cost history is bounded."""
        enforcer = CostEnforcer()
        enforcer._max_history_per_type = 10

        # Record more than max
        for i in range(20):
            enforcer.record_task_cost("debate", Decimal(str(i)))

        assert len(enforcer._task_cost_history["debate"]) == 10
        # Should keep the most recent ones
        assert enforcer._task_cost_history["debate"][-1] == Decimal("19")

    def test_get_budget_status_no_tracker(self):
        """Test budget status without tracker."""
        enforcer = CostEnforcer()

        status = enforcer.get_budget_status(workspace_id="ws-123")

        assert "error" in status

    def test_get_budget_status_no_budget(self):
        """Test budget status when no budget configured."""
        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = None

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        status = enforcer.get_budget_status(workspace_id="ws-123")

        assert "error" in status

    def test_get_budget_status_with_budget(self):
        """Test budget status with configured budget."""
        budget = MagicMock()
        budget.id = "budget-123"
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("750")  # 75%
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        enforcer = CostEnforcer(cost_tracker=cost_tracker)

        status = enforcer.get_budget_status(workspace_id="ws-123")

        assert status["budget_id"] == "budget-123"
        assert status["period"] == "monthly"
        assert status["limit_usd"] == "1000"
        assert status["spent_usd"] == "750"
        assert status["remaining_usd"] == "250"
        assert status["percentage_used"] == 75.0
        assert status["throttle_level"] == "medium"

    def test_throttle_callback(self):
        """Test throttle callbacks are invoked."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("600")  # 60% - light throttle
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        callback = MagicMock()
        enforcer = CostEnforcer(cost_tracker=cost_tracker)
        enforcer.add_throttle_callback(callback)

        enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        callback.assert_called_once_with("ws-123", ThrottleLevel.LIGHT)

    def test_block_callback(self):
        """Test block callbacks are invoked."""
        budget = MagicMock()
        budget.monthly_limit_usd = Decimal("1000")
        budget.current_monthly_spend = Decimal("1050")  # 105% - blocked
        budget.daily_limit_usd = None

        cost_tracker = MagicMock()
        cost_tracker.get_budget.return_value = budget

        callback = MagicMock()
        enforcer = CostEnforcer(cost_tracker=cost_tracker)
        enforcer.add_block_callback(callback)

        enforcer.check_budget_constraint(
            workspace_id="ws-123",
            task_type="debate",
        )

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "ws-123"
        assert "blocked" in args[1].lower()


class TestCostLimitExceededError:
    """Tests for CostLimitExceededError exception."""

    def test_error_message(self):
        """Test error message format."""
        result = CostConstraintResult(
            allowed=False,
            reason="Budget exceeded (105%)",
            throttle_level=ThrottleLevel.BLOCKED,
        )

        error = CostLimitExceededError(result=result, task_type="debate")

        assert "debate" in str(error)
        assert "105%" in str(error)
        assert error.result == result
        assert error.task_type == "debate"


class TestCostEnforcementConfig:
    """Tests for CostEnforcementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CostEnforcementConfig()

        assert config.mode == CostEnforcementMode.THROTTLE
        assert config.throttle_light_threshold == 50.0
        assert config.throttle_medium_threshold == 75.0
        assert config.throttle_heavy_threshold == 90.0
        assert config.throttle_block_threshold == 100.0
        assert config.light_priority_adjustment == -1
        assert config.medium_priority_adjustment == -2
        assert config.heavy_priority_adjustment == -3
        assert config.enable_cost_estimation is True
        assert config.alert_on_throttle is True
        assert config.alert_on_block is True

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = CostEnforcementConfig(
            throttle_light_threshold=40.0,
            throttle_heavy_threshold=85.0,
            throttle_block_threshold=95.0,
        )

        assert config.throttle_light_threshold == 40.0
        assert config.throttle_heavy_threshold == 85.0
        assert config.throttle_block_threshold == 95.0

"""
Security-focused tests for cost enforcement module.

Tests critical security paths including:
- Budget boundary conditions (exact thresholds, near-zero limits)
- Division-by-zero protection (zero budget limits)
- Negative budget exploitation
- Decimal precision in cost math
- Cost estimation edge cases (zero-cost history, high variance)
- Provided cost validation
- Soft/estimate mode behavior
- Remaining budget clamping
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest

from aragora.control_plane.cost_enforcement import (
    CostEnforcementConfig,
    CostEnforcementMode,
    CostEnforcer,
    CostEstimate,
    ThrottleLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget(
    monthly_limit: Decimal | None = Decimal("100.00"),
    monthly_spend: Decimal = Decimal("0.00"),
    daily_limit: Decimal | None = None,
    daily_spend: Decimal = Decimal("0.00"),
    budget_id: str = "budget-1",
):
    """Create a mock budget object."""
    b = mock.MagicMock()
    b.id = budget_id
    b.monthly_limit_usd = monthly_limit
    b.current_monthly_spend = monthly_spend
    b.daily_limit_usd = daily_limit
    b.current_daily_spend = daily_spend
    return b


def _make_tracker(budget=None):
    """Create a mock cost tracker."""
    tracker = mock.MagicMock()
    tracker.get_budget.return_value = budget
    return tracker


def _make_enforcer(
    mode=CostEnforcementMode.HARD,
    budget=None,
    **config_kwargs,
):
    """Create enforcer with budget."""
    config = CostEnforcementConfig(mode=mode, **config_kwargs)
    tracker = _make_tracker(budget) if budget else None
    return CostEnforcer(cost_tracker=tracker, config=config)


# ===========================================================================
# Budget Boundary Conditions
# ===========================================================================


class TestThrottleBoundaries:
    """Test exact boundary values for throttle thresholds."""

    @pytest.mark.parametrize(
        "spend,expected_level",
        [
            (Decimal("0.00"), ThrottleLevel.NONE),
            (Decimal("49.99"), ThrottleLevel.NONE),
            (Decimal("50.00"), ThrottleLevel.LIGHT),
            (Decimal("50.01"), ThrottleLevel.LIGHT),
            (Decimal("74.99"), ThrottleLevel.LIGHT),
            (Decimal("75.00"), ThrottleLevel.MEDIUM),
            (Decimal("75.01"), ThrottleLevel.MEDIUM),
            (Decimal("89.99"), ThrottleLevel.MEDIUM),
            (Decimal("90.00"), ThrottleLevel.HEAVY),
            (Decimal("90.01"), ThrottleLevel.HEAVY),
            (Decimal("99.99"), ThrottleLevel.HEAVY),
            (Decimal("100.00"), ThrottleLevel.BLOCKED),
            (Decimal("100.01"), ThrottleLevel.BLOCKED),
            (Decimal("200.00"), ThrottleLevel.BLOCKED),
        ],
    )
    def test_throttle_level_at_exact_boundary(self, spend, expected_level):
        """Throttle level transitions at exact percentage boundaries."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=spend)
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.throttle_level == expected_level

    @pytest.mark.parametrize(
        "spend,expected_allowed",
        [
            (Decimal("99.99"), True),
            (Decimal("100.00"), False),
            (Decimal("100.01"), False),
        ],
    )
    def test_hard_mode_blocks_at_100_percent(self, spend, expected_allowed):
        """Hard mode blocks exactly at 100% threshold."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=spend)
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed == expected_allowed

    def test_custom_thresholds(self):
        """Custom threshold configuration is respected."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("30.00"))
        enforcer = _make_enforcer(
            mode=CostEnforcementMode.THROTTLE,
            budget=budget,
            throttle_light_threshold=25.0,
            throttle_medium_threshold=50.0,
            throttle_heavy_threshold=75.0,
            throttle_block_threshold=90.0,
        )
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # 30% should be LIGHT with custom threshold of 25%
        assert result.throttle_level == ThrottleLevel.LIGHT


# ===========================================================================
# Division by Zero / Zero Budget
# ===========================================================================


class TestZeroBudgetProtection:
    """Test behavior when budget limits are zero or None."""

    def test_zero_monthly_limit_allows_all(self):
        """Zero monthly limit should not cause division by zero."""
        budget = _make_budget(monthly_limit=Decimal("0"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        # Should not raise, should allow (no valid limit = allow)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True

    def test_none_monthly_limit_allows_all(self):
        """None monthly limit falls through to daily or allows."""
        budget = _make_budget(
            monthly_limit=None,
            daily_limit=None,
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True

    def test_zero_monthly_with_daily_limit(self):
        """Zero monthly but valid daily limit uses daily."""
        budget = _make_budget(
            monthly_limit=Decimal("0"),
            monthly_spend=Decimal("0"),
            daily_limit=Decimal("10.00"),
            daily_spend=Decimal("8.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # 80% of daily → MEDIUM throttle
        assert result.throttle_level == ThrottleLevel.MEDIUM

    def test_no_budget_configured_allows_all(self):
        """No budget object at all allows everything."""
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=None)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True

    def test_no_cost_tracker_allows_all(self):
        """No cost tracker configured allows everything."""
        config = CostEnforcementConfig(mode=CostEnforcementMode.HARD)
        enforcer = CostEnforcer(cost_tracker=None, config=config)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True


# ===========================================================================
# Negative Budget Exploitation
# ===========================================================================


class TestNegativeBudgetExploitation:
    """Test that negative budget values don't create bypasses."""

    def test_negative_monthly_limit_allows_all(self):
        """Negative monthly limit should not invert enforcement logic."""
        budget = _make_budget(
            monthly_limit=Decimal("-100.00"),
            monthly_spend=Decimal("50.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        # Negative limit → percentage will be negative → level = NONE → allowed
        # This is a potential bypass vector
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # Document the actual behavior: negative limit falls through
        # (since -100 is not > 0, it falls through to the else branch = allow)
        assert result.allowed is True

    def test_negative_spend_percentage(self):
        """Negative spend shouldn't cause unexpected throttle behavior."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("-10.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # -10% should be NONE
        assert result.throttle_level == ThrottleLevel.NONE
        assert result.allowed is True


# ===========================================================================
# Remaining Budget Clamping
# ===========================================================================


class TestRemainingBudgetClamping:
    """Test that remaining budget is properly clamped to zero."""

    def test_overspent_budget_remaining_is_zero(self):
        """When spend exceeds limit, remaining is clamped to 0."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("150.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.remaining_budget_usd == Decimal("0")

    def test_exactly_at_limit_remaining_is_zero(self):
        """When spend equals limit, remaining is zero."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("100.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.remaining_budget_usd == Decimal("0")

    def test_under_budget_remaining_positive(self):
        """When under budget, remaining is positive."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("60.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.remaining_budget_usd == Decimal("40.00")

    def test_slightly_overspent_remaining_clamped(self):
        """Slightly over budget (e.g., due to concurrent tasks) clamps to zero."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("100.01"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.remaining_budget_usd == Decimal("0")


# ===========================================================================
# Decimal Precision
# ===========================================================================


class TestDecimalPrecision:
    """Test decimal arithmetic precision in budget calculations."""

    def test_fractional_percentage_calculation(self):
        """Percentage calculation with non-round numbers preserves precision."""
        budget = _make_budget(
            monthly_limit=Decimal("33.33"),
            monthly_spend=Decimal("16.67"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # 16.67 / 33.33 * 100 = ~50.015%
        assert result.budget_percentage_used > 50.0
        assert result.throttle_level == ThrottleLevel.LIGHT

    def test_very_small_budget(self):
        """Very small budget values work correctly."""
        budget = _make_budget(
            monthly_limit=Decimal("0.01"),
            monthly_spend=Decimal("0.005"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # 0.005 / 0.01 = 50% → LIGHT
        assert result.throttle_level == ThrottleLevel.LIGHT

    def test_large_budget_values(self):
        """Large budget values don't cause overflow."""
        budget = _make_budget(
            monthly_limit=Decimal("999999999.99"),
            monthly_spend=Decimal("500000000.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.budget_percentage_used == pytest.approx(50.0, abs=0.1)
        assert result.throttle_level == ThrottleLevel.LIGHT

    def test_repeating_decimal_division(self):
        """Division resulting in repeating decimal doesn't crash."""
        budget = _make_budget(
            monthly_limit=Decimal("3.00"),
            monthly_spend=Decimal("1.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        # 1/3 * 100 = 33.333...
        assert result.budget_percentage_used == pytest.approx(33.33, abs=0.1)
        assert result.throttle_level == ThrottleLevel.NONE


# ===========================================================================
# Cost Estimation Edge Cases
# ===========================================================================


class TestCostEstimationSecurity:
    """Test cost estimation security edge cases."""

    def test_provided_negative_cost_accepted(self):
        """Negative provided cost is accepted (potential bypass)."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("90.00"))
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(
            workspace_id="ws-1",
            task_type="debate",
            estimated_cost=Decimal("-1.00"),
        )
        # Document behavior: negative cost is accepted in estimate
        # Budget check uses percentage (not estimated cost), so this doesn't bypass
        assert result.estimated_cost is not None
        assert result.estimated_cost.estimated_cost_usd == Decimal("-1.00")

    def test_provided_zero_cost(self):
        """Zero provided cost works correctly."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("50.00"))
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(
            workspace_id="ws-1",
            estimated_cost=Decimal("0"),
        )
        assert result.estimated_cost.estimated_cost_usd == Decimal("0")
        assert result.estimated_cost.confidence == 1.0

    def test_zero_cost_history_division_safety(self):
        """History of zero-cost tasks doesn't cause division by zero in confidence."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(
            mode=CostEnforcementMode.THROTTLE,
            budget=budget,
            min_samples_for_estimate=3,
        )

        # Record several zero-cost tasks
        for _ in range(5):
            enforcer.record_task_cost("free_task", Decimal("0"))

        # avg_cost = 0.0, std_dev = 0.0 → confidence should be 0.5 (zero cost, no variance)
        result = enforcer.check_budget_constraint(workspace_id="ws-1", task_type="free_task")
        assert result.estimated_cost is not None
        assert result.estimated_cost.estimated_cost_usd == Decimal("0")
        assert result.estimated_cost.confidence == 0.5

    def test_high_variance_cost_history(self):
        """High variance in cost history produces low confidence."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(
            mode=CostEnforcementMode.THROTTLE,
            budget=budget,
            min_samples_for_estimate=3,
        )

        # Record wildly varying costs
        costs = [
            Decimal("0.01"),
            Decimal("100.00"),
            Decimal("0.02"),
            Decimal("50.00"),
            Decimal("0.01"),
        ]
        for cost in costs:
            enforcer.record_task_cost("volatile_task", cost)

        result = enforcer.check_budget_constraint(workspace_id="ws-1", task_type="volatile_task")
        assert result.estimated_cost is not None
        # High variance should produce low confidence
        assert result.estimated_cost.confidence < 0.5

    def test_single_sample_confidence(self):
        """Single sample uses 0.5 confidence."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(
            mode=CostEnforcementMode.THROTTLE,
            budget=budget,
            min_samples_for_estimate=1,
        )
        enforcer.record_task_cost("single_task", Decimal("1.50"))

        result = enforcer.check_budget_constraint(workspace_id="ws-1", task_type="single_task")
        assert result.estimated_cost is not None
        assert result.estimated_cost.confidence == 0.5

    def test_history_bounded_at_max(self):
        """Cost history doesn't grow unbounded."""
        enforcer = _make_enforcer(mode=CostEnforcementMode.SOFT)
        # Record 200 costs (max is 100)
        for i in range(200):
            enforcer.record_task_cost("bulk_task", Decimal(str(i)))

        history = enforcer._task_cost_history["bulk_task"]
        assert len(history) <= 100
        # Should keep the most recent entries
        assert history[-1] == Decimal("199")

    def test_unknown_task_type_uses_default(self):
        """Unknown task type falls back to default cost estimate."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(
            workspace_id="ws-1", task_type="completely_unknown_task"
        )
        assert result.estimated_cost is not None
        assert result.estimated_cost.estimated_cost_usd == Decimal("0.10")  # default
        assert result.estimated_cost.confidence == 0.3  # low confidence for defaults

    def test_estimation_disabled(self):
        """Cost estimation can be disabled."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=Decimal("10.00"))
        enforcer = _make_enforcer(
            mode=CostEnforcementMode.THROTTLE,
            budget=budget,
            enable_cost_estimation=False,
        )
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.estimated_cost is None


# ===========================================================================
# Enforcement Mode Behavior
# ===========================================================================


class TestEnforcementModes:
    """Test all enforcement mode behaviors."""

    def test_soft_mode_always_allows(self):
        """Soft mode allows even when over budget."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("200.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.SOFT, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.BLOCKED  # Still reports level
        assert result.budget_percentage_used == 200.0

    def test_estimate_mode_allows_over_budget(self):
        """Estimate mode allows even when over budget."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("150.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.ESTIMATE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True

    def test_hard_mode_blocks_over_budget(self):
        """Hard mode blocks when at/over budget."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("100.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is False
        assert result.enforcement_mode == CostEnforcementMode.HARD

    def test_throttle_mode_blocks_at_100_percent(self):
        """Throttle mode blocks at 100% (not just hard mode)."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("100.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is False
        assert result.throttle_level == ThrottleLevel.BLOCKED

    def test_throttle_mode_allows_under_100_percent(self):
        """Throttle mode allows with adjustment under 100%."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("92.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.HEAVY
        assert result.priority_adjustment == -3


# ===========================================================================
# Budget Status
# ===========================================================================


class TestBudgetStatus:
    """Test budget status reporting edge cases."""

    def test_status_no_tracker(self):
        """Status without tracker returns error."""
        config = CostEnforcementConfig()
        enforcer = CostEnforcer(cost_tracker=None, config=config)
        status = enforcer.get_budget_status(workspace_id="ws-1")
        assert "error" in status

    def test_status_no_budget(self):
        """Status without budget returns error."""
        tracker = _make_tracker(budget=None)
        enforcer = CostEnforcer(cost_tracker=tracker)
        status = enforcer.get_budget_status(workspace_id="ws-1")
        assert "error" in status

    def test_status_no_limits_set(self):
        """Status with budget but no limits returns limits_set=False."""
        budget = _make_budget(monthly_limit=None, daily_limit=None)
        tracker = _make_tracker(budget=budget)
        enforcer = CostEnforcer(cost_tracker=tracker)
        status = enforcer.get_budget_status(workspace_id="ws-1")
        assert status["limits_set"] is False

    def test_status_overspent_remaining_zero(self):
        """Status with overspent budget shows remaining as '0'."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("150.00"),
        )
        tracker = _make_tracker(budget=budget)
        enforcer = CostEnforcer(cost_tracker=tracker)
        status = enforcer.get_budget_status(workspace_id="ws-1")
        assert status["remaining_usd"] == "0"
        assert status["percentage_used"] == 150.0

    def test_status_daily_limit_used(self):
        """Status falls back to daily limit when no monthly."""
        budget = _make_budget(
            monthly_limit=Decimal("0"),
            daily_limit=Decimal("10.00"),
            daily_spend=Decimal("7.50"),
        )
        tracker = _make_tracker(budget=budget)
        enforcer = CostEnforcer(cost_tracker=tracker)
        status = enforcer.get_budget_status(workspace_id="ws-1")
        assert status["period"] == "daily"
        assert status["percentage_used"] == 75.0


# ===========================================================================
# Callback Safety
# ===========================================================================


class TestCallbackSafety:
    """Test that callback exceptions don't crash the enforcer."""

    def test_throttle_callback_exception_handled(self):
        """Exception in throttle callback doesn't crash check."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("60.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)

        def bad_callback(identifier, level):
            raise RuntimeError("Callback crashed!")

        enforcer.add_throttle_callback(bad_callback)
        # Should not raise
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is True
        assert result.throttle_level == ThrottleLevel.LIGHT

    def test_block_callback_exception_handled(self):
        """Exception in block callback doesn't crash check."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("100.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.HARD, budget=budget)

        def bad_callback(identifier, reason):
            raise RuntimeError("Block callback crashed!")

        enforcer.add_block_callback(bad_callback)
        # Should not raise
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.allowed is False

    def test_multiple_callbacks_all_called(self):
        """All callbacks are called even if one fails."""
        budget = _make_budget(
            monthly_limit=Decimal("100.00"),
            monthly_spend=Decimal("60.00"),
        )
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)

        called = []

        def good_callback_1(identifier, level):
            called.append("cb1")

        def bad_callback(identifier, level):
            called.append("bad")
            raise RuntimeError("fail")

        def good_callback_2(identifier, level):
            called.append("cb2")

        enforcer.add_throttle_callback(good_callback_1)
        enforcer.add_throttle_callback(bad_callback)
        enforcer.add_throttle_callback(good_callback_2)

        enforcer.check_budget_constraint(workspace_id="ws-1")
        assert "cb1" in called
        assert "bad" in called
        assert "cb2" in called


# ===========================================================================
# Priority Adjustment
# ===========================================================================


class TestPriorityAdjustment:
    """Test priority adjustments for throttle levels."""

    @pytest.mark.parametrize(
        "spend,expected_adjustment",
        [
            (Decimal("30.00"), 0),  # NONE → no adjustment
            (Decimal("60.00"), -1),  # LIGHT → -1
            (Decimal("80.00"), -2),  # MEDIUM → -2
            (Decimal("95.00"), -3),  # HEAVY → -3
        ],
    )
    def test_priority_adjustment_by_level(self, spend, expected_adjustment):
        """Priority adjustment matches throttle level."""
        budget = _make_budget(monthly_limit=Decimal("100.00"), monthly_spend=spend)
        enforcer = _make_enforcer(mode=CostEnforcementMode.THROTTLE, budget=budget)
        result = enforcer.check_budget_constraint(workspace_id="ws-1")
        assert result.priority_adjustment == expected_adjustment

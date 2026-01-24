"""Tests for Budget Overage Handling."""

import pytest
import time

from aragora.billing.budget_manager import (
    Budget,
    BudgetAction,
    BudgetPeriod,
    BudgetStatus,
    BudgetThreshold,
    SpendResult,
)


class TestSpendResult:
    """Tests for SpendResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SpendResult(
            allowed=True,
            message="OK",
            is_overage=True,
            overage_amount_usd=25.0,
            overage_rate_multiplier=1.5,
        )
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["is_overage"] is True
        assert d["overage_amount_usd"] == 25.0
        assert d["overage_rate_multiplier"] == 1.5

    def test_default_values(self):
        """Test default values."""
        result = SpendResult(allowed=True)
        assert result.is_overage is False
        assert result.overage_amount_usd == 0.0
        assert result.overage_rate_multiplier == 1.0


class TestBudgetOverageFields:
    """Tests for overage fields on Budget."""

    def test_default_overage_settings(self):
        """Test that overage is disabled by default."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test Budget",
            amount_usd=100.0,
        )
        assert budget.allow_overage is False
        assert budget.overage_rate_multiplier == 1.5
        assert budget.overage_spent_usd == 0.0
        assert budget.max_overage_usd is None

    def test_overage_in_to_dict(self):
        """Test that overage fields are in to_dict output."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test Budget",
            amount_usd=100.0,
            allow_overage=True,
            overage_rate_multiplier=2.0,
            overage_spent_usd=25.0,
            max_overage_usd=50.0,
        )
        d = budget.to_dict()
        assert d["allow_overage"] is True
        assert d["overage_rate_multiplier"] == 2.0
        assert d["overage_spent_usd"] == 25.0
        assert d["max_overage_usd"] == 50.0


class TestBudgetCanSpendExtended:
    """Tests for can_spend_extended with overage support."""

    def test_within_budget_returns_ok(self):
        """Test spending within budget returns OK."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=50.0,
        )
        result = budget.can_spend_extended(25.0)
        assert result.allowed is True
        assert result.is_overage is False
        assert result.message == "OK"

    def test_exceed_without_overage_blocked(self):
        """Test that exceeding budget without overage is blocked."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,  # Already at 100% to trigger HARD_LIMIT action
            allow_overage=False,
            thresholds=[BudgetThreshold(1.0, BudgetAction.HARD_LIMIT)],
        )
        result = budget.can_spend_extended(20.0)
        assert result.allowed is False
        assert "exceeded" in result.message.lower()

    def test_exceed_with_overage_allowed(self):
        """Test that exceeding budget with overage enabled is allowed."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=90.0,
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )
        result = budget.can_spend_extended(20.0)
        assert result.allowed is True
        assert result.is_overage is True
        assert result.overage_amount_usd == 10.0  # 90 + 20 - 100 = 10 overage
        assert result.overage_rate_multiplier == 1.5

    def test_overage_cap_respected(self):
        """Test that max overage cap is enforced."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,
            allow_overage=True,
            overage_spent_usd=40.0,  # Already 40 in overage
            max_overage_usd=50.0,  # Cap at 50
        )
        # Trying to spend 20 more would put us at 60 overage, exceeding cap
        result = budget.can_spend_extended(20.0)
        assert result.allowed is False
        assert "cap exceeded" in result.message.lower()

    def test_overage_within_cap_allowed(self):
        """Test that overage within cap is allowed."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,
            allow_overage=True,
            overage_spent_usd=30.0,  # Already 30 in overage
            max_overage_usd=50.0,  # Cap at 50
        )
        # Trying to spend 10 more would put us at 40 overage, within cap
        result = budget.can_spend_extended(10.0)
        assert result.allowed is True
        assert result.is_overage is True

    def test_allow_with_charges_action(self):
        """Test ALLOW_WITH_CHARGES action enables overage."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,  # Already at 100% to trigger ALLOW_WITH_CHARGES action
            allow_overage=False,  # Disabled globally
            thresholds=[BudgetThreshold(1.0, BudgetAction.ALLOW_WITH_CHARGES)],
        )
        result = budget.can_spend_extended(20.0)
        assert result.allowed is True
        assert result.is_overage is True

    def test_suspended_budget_blocked(self):
        """Test that suspended budget blocks all spending."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=50.0,
            status=BudgetStatus.SUSPENDED,
            allow_overage=True,
        )
        result = budget.can_spend_extended(10.0)
        assert result.allowed is False
        assert "suspended" in result.message.lower()

    def test_override_user_bypasses(self):
        """Test that override users bypass all checks."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=150.0,  # Already over
            status=BudgetStatus.EXCEEDED,
            allow_overage=False,
            override_user_ids=["admin_user"],
        )
        result = budget.can_spend_extended(50.0, user_id="admin_user")
        assert result.allowed is True
        assert result.message == "Override active"


class TestBudgetCanSpendLegacy:
    """Tests for legacy can_spend interface."""

    def test_returns_tuple(self):
        """Test that can_spend returns tuple for backward compatibility."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=50.0,
        )
        result = budget.can_spend(25.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is True
        assert result[1] == "OK"

    def test_legacy_blocked_when_exceeded(self):
        """Test legacy interface blocks when exceeded."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,  # Already at 100% to trigger HARD_LIMIT action
            allow_overage=False,
            thresholds=[BudgetThreshold(1.0, BudgetAction.HARD_LIMIT)],
        )
        allowed, message = budget.can_spend(20.0)
        assert allowed is False
        assert "exceeded" in message.lower()


class TestRecordOverage:
    """Tests for recording overage spending."""

    def test_record_overage_adds_to_total(self):
        """Test that recording overage increases overage_spent_usd."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            overage_spent_usd=0.0,
        )
        budget.record_overage(25.0)
        assert budget.overage_spent_usd == 25.0

    def test_record_overage_accumulates(self):
        """Test that overage accumulates correctly."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            overage_spent_usd=10.0,
        )
        budget.record_overage(15.0)
        assert budget.overage_spent_usd == 25.0

    def test_record_overage_updates_timestamp(self):
        """Test that recording overage updates the timestamp."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
        )
        old_updated = budget.updated_at
        time.sleep(0.01)
        budget.record_overage(10.0)
        assert budget.updated_at > old_updated


class TestOverageRateCalculation:
    """Tests for overage rate calculation scenarios."""

    def test_overage_rate_in_result(self):
        """Test that overage rate is included in result."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=80.0,
            allow_overage=True,
            overage_rate_multiplier=2.0,
        )
        result = budget.can_spend_extended(30.0)  # 10 will be overage
        assert result.overage_rate_multiplier == 2.0
        # Caller should calculate: 10 * 2.0 = 20 for overage portion

    def test_partial_overage_amount(self):
        """Test correct calculation of partial overage amount."""
        budget = Budget(
            budget_id="test",
            org_id="org_1",
            name="Test",
            amount_usd=100.0,
            spent_usd=75.0,
            allow_overage=True,
        )
        result = budget.can_spend_extended(50.0)
        # Total would be 125, budget is 100, so 25 is overage
        assert result.overage_amount_usd == 25.0

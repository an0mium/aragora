"""Tests for budget manager circuit breaker integration.

Verifies that when a budget hits EXCEEDED or SUSPENDED status,
a circuit breaker trips to prevent further API spend.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from aragora.billing.budget_manager import (
    Budget,
    BudgetAction,
    BudgetManager,
    BudgetPeriod,
    BudgetStatus,
    BudgetThreshold,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def manager(temp_db):
    """Create a budget manager with temp database."""
    return BudgetManager(temp_db)


class TestBudgetCircuitBreakerTrips:
    """Tests that circuit breaker trips on exceeded/suspended budgets."""

    def test_circuit_breaker_trips_on_exceeded_budget(self, manager):
        """Circuit breaker should trip when budget status becomes EXCEEDED."""
        budget = manager.create_budget(
            org_id="org-1",
            name="Test Budget",
            amount_usd=100.0,
            auto_suspend=False,  # Don't auto-suspend, just exceed
        )

        # Spend exactly the budget amount to trigger EXCEEDED via threshold check
        # With auto_suspend=False, the budget won't suspend but the status
        # transitions happen via _check_budget_circuit_breaker checking the
        # budget object state. We manually set exceeded to test.
        budget_obj = Budget(
            budget_id=budget.budget_id,
            org_id="org-1",
            name="Test Budget",
            amount_usd=100.0,
            spent_usd=150.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-1") is True

    def test_circuit_breaker_trips_on_suspended_budget(self, manager):
        """Circuit breaker should trip when budget status is SUSPENDED."""
        budget_obj = Budget(
            budget_id="budget-test",
            org_id="org-2",
            name="Suspended Budget",
            amount_usd=50.0,
            spent_usd=60.0,
            status=BudgetStatus.SUSPENDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-2") is True

    def test_circuit_breaker_does_not_trip_on_active_budget(self, manager):
        """Circuit breaker should NOT trip for ACTIVE budgets."""
        budget_obj = Budget(
            budget_id="budget-active",
            org_id="org-3",
            name="Active Budget",
            amount_usd=100.0,
            spent_usd=10.0,
            status=BudgetStatus.ACTIVE,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-3") is False

    def test_circuit_breaker_does_not_trip_on_warning_budget(self, manager):
        """Circuit breaker should NOT trip for WARNING budgets."""
        budget_obj = Budget(
            budget_id="budget-warn",
            org_id="org-4",
            name="Warning Budget",
            amount_usd=100.0,
            spent_usd=80.0,
            status=BudgetStatus.WARNING,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-4") is False

    def test_circuit_breaker_does_not_trip_on_critical_budget(self, manager):
        """Circuit breaker should NOT trip for CRITICAL budgets."""
        budget_obj = Budget(
            budget_id="budget-crit",
            org_id="org-5",
            name="Critical Budget",
            amount_usd=100.0,
            spent_usd=95.0,
            status=BudgetStatus.CRITICAL,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-5") is False


class TestIsBudgetCircuitOpen:
    """Tests for the is_budget_circuit_open public method."""

    def test_returns_true_when_breaker_tripped(self, manager):
        """Should return True when circuit breaker has been tripped."""
        budget_obj = Budget(
            budget_id="budget-test",
            org_id="org-open",
            name="Exceeded Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert manager.is_budget_circuit_open("org-open") is True

    def test_returns_false_for_unknown_org(self, manager):
        """Should return False for an org with no circuit breaker."""
        assert manager.is_budget_circuit_open("nonexistent-org") is False

    def test_returns_false_before_any_breaker_created(self, manager):
        """Should return False when no breakers have been created at all."""
        # Fresh manager has no _budget_breakers attribute
        assert manager.is_budget_circuit_open("any-org") is False

    def test_returns_false_after_cooldown_expires(self, manager):
        """Should return False after circuit breaker cooldown expires."""
        budget_obj = Budget(
            budget_id="budget-test",
            org_id="org-cooldown",
            name="Exceeded Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)
        assert manager.is_budget_circuit_open("org-cooldown") is True

        # Simulate cooldown expiry by backdating the open_at timestamp
        breaker = manager._budget_breakers["budget_org-cooldown"]
        breaker._single_open_at -= 400.0  # 400s > 300s cooldown

        assert manager.is_budget_circuit_open("org-cooldown") is False


class TestCircuitBreakerGracefulDegradation:
    """Tests that circuit breaker integration degrades gracefully."""

    def test_graceful_when_circuit_breaker_module_unavailable(self, manager):
        """Should not raise when circuit breaker module can't be imported."""
        budget_obj = Budget(
            budget_id="budget-test",
            org_id="org-noimport",
            name="Test Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        with patch.dict("sys.modules", {"aragora.resilience.circuit_breaker": None}):
            # Should not raise, just log debug message
            manager._check_budget_circuit_breaker(budget_obj)

        # Circuit should not be open since import failed
        assert manager.is_budget_circuit_open("org-noimport") is False

    def test_graceful_on_runtime_error(self, manager):
        """Should not raise when circuit breaker raises RuntimeError."""
        budget_obj = Budget(
            budget_id="budget-test",
            org_id="org-err",
            name="Test Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        with patch(
            "aragora.resilience.circuit_breaker.CircuitBreaker.record_failure",
            side_effect=RuntimeError("test error"),
        ):
            # Should not raise
            manager._check_budget_circuit_breaker(budget_obj)


class TestBudgetBreakerLazyInit:
    """Tests that the budget breaker dict is lazily initialized."""

    def test_budget_breakers_not_present_initially(self, manager):
        """_budget_breakers should not exist on a fresh manager."""
        assert not hasattr(manager, "_budget_breakers")

    def test_budget_breakers_created_on_first_trip(self, manager):
        """_budget_breakers should be created when first breaker trips."""
        assert not hasattr(manager, "_budget_breakers")

        budget_obj = Budget(
            budget_id="budget-lazy",
            org_id="org-lazy",
            name="Lazy Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        assert hasattr(manager, "_budget_breakers")
        assert "budget_org-lazy" in manager._budget_breakers

    def test_budget_breakers_reused_on_subsequent_trips(self, manager):
        """Same breaker should be reused for the same org."""
        budget_obj = Budget(
            budget_id="budget-reuse",
            org_id="org-reuse",
            name="Reuse Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)
        first_breaker = manager._budget_breakers["budget_org-reuse"]

        manager._check_budget_circuit_breaker(budget_obj)
        second_breaker = manager._budget_breakers["budget_org-reuse"]

        assert first_breaker is second_breaker


class TestRecordSpendTripsCircuitBreaker:
    """Tests that record_spend triggers circuit breaker on budget exceed."""

    def test_record_spend_trips_breaker_on_auto_suspend(self, manager):
        """record_spend should trip circuit breaker when budget auto-suspends."""
        budget = manager.create_budget(
            org_id="org-spend",
            name="Spend Budget",
            amount_usd=100.0,
            auto_suspend=True,
        )

        # Spend enough to exceed the budget
        manager.record_spend("org-spend", 150.0, description="Big spend")

        # The budget should be auto-suspended and circuit breaker tripped
        assert manager.is_budget_circuit_open("org-spend") is True

    def test_record_spend_no_breaker_for_normal_spend(self, manager):
        """record_spend should NOT trip circuit breaker for normal spending."""
        manager.create_budget(
            org_id="org-normal",
            name="Normal Budget",
            amount_usd=1000.0,
        )

        manager.record_spend("org-normal", 10.0, description="Small spend")

        assert manager.is_budget_circuit_open("org-normal") is False

    def test_record_spend_trips_breaker_different_orgs(self, manager):
        """Circuit breakers should be independent per org."""
        manager.create_budget(
            org_id="org-a",
            name="Budget A",
            amount_usd=100.0,
            auto_suspend=True,
        )
        manager.create_budget(
            org_id="org-b",
            name="Budget B",
            amount_usd=1000.0,
        )

        # Exceed org-a but not org-b
        manager.record_spend("org-a", 200.0, description="Over budget")
        manager.record_spend("org-b", 10.0, description="Normal spend")

        assert manager.is_budget_circuit_open("org-a") is True
        assert manager.is_budget_circuit_open("org-b") is False


class TestCircuitBreakerConfiguration:
    """Tests for circuit breaker configuration details."""

    def test_breaker_has_threshold_of_one(self, manager):
        """Budget circuit breaker should trip immediately (threshold=1)."""
        budget_obj = Budget(
            budget_id="budget-cfg",
            org_id="org-cfg",
            name="Config Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        breaker = manager._budget_breakers["budget_org-cfg"]
        assert breaker.failure_threshold == 1

    def test_breaker_has_five_minute_cooldown(self, manager):
        """Budget circuit breaker should have 300s (5 min) cooldown."""
        budget_obj = Budget(
            budget_id="budget-cd",
            org_id="org-cd",
            name="Cooldown Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        breaker = manager._budget_breakers["budget_org-cd"]
        assert breaker.cooldown_seconds == 300.0

    def test_breaker_named_after_org(self, manager):
        """Budget circuit breaker name should include org_id."""
        budget_obj = Budget(
            budget_id="budget-name",
            org_id="my-org-123",
            name="Named Budget",
            amount_usd=100.0,
            spent_usd=200.0,
            status=BudgetStatus.EXCEEDED,
        )

        manager._check_budget_circuit_breaker(budget_obj)

        breaker = manager._budget_breakers["budget_my-org-123"]
        assert breaker.name == "budget_my-org-123"

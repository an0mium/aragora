"""Tests for the Budget Manager."""

import os
import tempfile
import pytest

from aragora.billing.budget_manager import (
    BudgetManager,
    Budget,
    BudgetAlert,
    BudgetPeriod,
    BudgetStatus,
    BudgetAction,
    BudgetThreshold,
    DEFAULT_THRESHOLDS,
    get_budget_manager,
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


class TestBudgetCreation:
    """Tests for budget creation."""

    def test_create_budget_basic(self, manager):
        """Test basic budget creation."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test Budget",
            amount_usd=100.0,
            period=BudgetPeriod.MONTHLY,
        )

        assert budget.budget_id.startswith("budget-")
        assert budget.org_id == "test-org"
        assert budget.name == "Test Budget"
        assert budget.amount_usd == 100.0
        assert budget.period == BudgetPeriod.MONTHLY
        assert budget.status == BudgetStatus.ACTIVE
        assert budget.spent_usd == 0.0
        assert len(budget.thresholds) == 4

    def test_create_budget_with_custom_thresholds(self, manager):
        """Test budget creation with custom thresholds."""
        thresholds = [
            BudgetThreshold(0.80, BudgetAction.WARN),
            BudgetThreshold(0.95, BudgetAction.HARD_LIMIT),
        ]

        budget = manager.create_budget(
            org_id="test-org",
            name="Custom Budget",
            amount_usd=500.0,
            period=BudgetPeriod.WEEKLY,
            thresholds=thresholds,
        )

        assert len(budget.thresholds) == 2
        assert budget.thresholds[0].percentage == 0.80

    def test_create_budget_all_periods(self, manager):
        """Test budget creation with all period types."""
        periods = [
            BudgetPeriod.DAILY,
            BudgetPeriod.WEEKLY,
            BudgetPeriod.MONTHLY,
            BudgetPeriod.QUARTERLY,
            BudgetPeriod.ANNUAL,
            BudgetPeriod.UNLIMITED,
        ]

        for period in periods:
            budget = manager.create_budget(
                org_id="test-org",
                name=f"{period.value} Budget",
                amount_usd=100.0,
                period=period,
            )
            assert budget.period == period
            assert budget.period_start > 0
            assert budget.period_end > budget.period_start


class TestBudgetRetrieval:
    """Tests for budget retrieval."""

    def test_get_budget(self, manager):
        """Test getting a budget by ID."""
        created = manager.create_budget(
            org_id="test-org",
            name="Test Budget",
            amount_usd=100.0,
        )

        retrieved = manager.get_budget(created.budget_id)
        assert retrieved is not None
        assert retrieved.budget_id == created.budget_id
        assert retrieved.name == "Test Budget"

    def test_get_nonexistent_budget(self, manager):
        """Test getting a budget that doesn't exist."""
        result = manager.get_budget("budget-nonexistent")
        assert result is None

    def test_get_budgets_for_org(self, manager):
        """Test getting all budgets for an organization."""
        manager.create_budget(org_id="org-1", name="Budget 1", amount_usd=100.0)
        manager.create_budget(org_id="org-1", name="Budget 2", amount_usd=200.0)
        manager.create_budget(org_id="org-2", name="Budget 3", amount_usd=300.0)

        budgets = manager.get_budgets_for_org("org-1")
        assert len(budgets) == 2
        assert all(b.org_id == "org-1" for b in budgets)


class TestBudgetUpdate:
    """Tests for budget updates."""

    def test_update_budget_name(self, manager):
        """Test updating budget name."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Original Name",
            amount_usd=100.0,
        )

        updated = manager.update_budget(budget.budget_id, name="New Name")
        assert updated.name == "New Name"

    def test_update_budget_amount(self, manager):
        """Test updating budget amount."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        updated = manager.update_budget(budget.budget_id, amount_usd=200.0)
        assert updated.amount_usd == 200.0

    def test_update_budget_status(self, manager):
        """Test updating budget status."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        updated = manager.update_budget(budget.budget_id, status=BudgetStatus.PAUSED)
        assert updated.status == BudgetStatus.PAUSED

    def test_delete_budget(self, manager):
        """Test deleting (closing) a budget."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        manager.delete_budget(budget.budget_id)
        closed = manager.get_budget(budget.budget_id)
        assert closed.status == BudgetStatus.CLOSED


class TestBudgetEnforcement:
    """Tests for budget enforcement."""

    def test_check_budget_allowed(self, manager):
        """Test budget check when spending is allowed."""
        manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        allowed, reason, action = manager.check_budget("test-org", 10.0)
        assert allowed is True
        assert action is None

    def test_check_budget_exceeds(self, manager):
        """Test budget check when spending would exceed."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        # Spend most of the budget
        manager.record_spend("test-org", 95.0, "Test spend")

        # Try to spend more than remaining
        allowed, reason, action = manager.check_budget("test-org", 10.0)
        # Should still be allowed until hard limit
        assert allowed is True

    def test_check_budget_no_budget(self, manager):
        """Test budget check when no budget configured."""
        allowed, reason, action = manager.check_budget("no-budget-org", 100.0)
        assert allowed is True
        assert "No budget" in reason

    def test_record_spend(self, manager):
        """Test recording spending."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        manager.record_spend("test-org", 25.0, "First spend")
        manager.record_spend("test-org", 25.0, "Second spend")

        updated = manager.get_budget(budget.budget_id)
        assert updated.spent_usd == 50.0
        assert updated.usage_percentage == 0.5

    def test_auto_suspend_on_exceed(self, manager):
        """Test auto-suspension when budget is exceeded."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
            auto_suspend=True,
        )

        # Exceed the budget
        manager.record_spend("test-org", 110.0, "Exceed spend")

        updated = manager.get_budget(budget.budget_id)
        assert updated.status == BudgetStatus.SUSPENDED


class TestBudgetOverrides:
    """Tests for budget overrides."""

    def test_add_override(self, manager):
        """Test adding a budget override."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        result = manager.add_override(budget.budget_id, "user-123")
        assert result is True

        updated = manager.get_budget(budget.budget_id)
        assert "user-123" in updated.override_user_ids

    def test_override_allows_exceed(self, manager):
        """Test that override allows exceeding budget."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        # Add override for user
        manager.add_override(budget.budget_id, "admin-user")

        # Get updated budget with override applied
        budget = manager.get_budget(budget.budget_id)

        # Simulate already exceeded budget
        budget.spent_usd = 100.0

        # Check that override user can still spend
        allowed, reason = budget.can_spend(10.0, user_id="admin-user")
        assert allowed is True
        assert "Override" in reason

        # Check that non-override user cannot spend when exceeded
        allowed_no_override, reason_no_override = budget.can_spend(10.0, user_id="other-user")
        assert allowed_no_override is False

    def test_remove_override(self, manager):
        """Test removing a budget override."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        manager.add_override(budget.budget_id, "user-123")
        manager.remove_override(budget.budget_id, "user-123")

        updated = manager.get_budget(budget.budget_id)
        assert "user-123" not in updated.override_user_ids


class TestBudgetAlerts:
    """Tests for budget alerts."""

    def test_alert_on_threshold_cross(self, manager):
        """Test that alerts are created when thresholds are crossed."""
        alerts_received = []
        manager.register_alert_callback(lambda a: alerts_received.append(a))

        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        # Cross 50% threshold
        manager.record_spend("test-org", 55.0, "Test spend")

        assert len(alerts_received) >= 1
        assert alerts_received[0].threshold_percentage == 0.5

    def test_get_alerts(self, manager):
        """Test retrieving alerts."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        # Trigger an alert
        manager.record_spend("test-org", 55.0)

        alerts = manager.get_alerts(org_id="test-org")
        assert len(alerts) >= 1

    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        manager.record_spend("test-org", 55.0)
        alerts = manager.get_alerts(org_id="test-org")

        if alerts:
            manager.acknowledge_alert(alerts[0].alert_id, "user-123")
            updated_alerts = manager.get_alerts(org_id="test-org")
            assert updated_alerts[0].acknowledged is True


class TestBudgetPeriodReset:
    """Tests for budget period reset."""

    def test_reset_period(self, manager):
        """Test resetting a budget period."""
        budget = manager.create_budget(
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
        )

        # Spend some
        manager.record_spend("test-org", 50.0)

        # Reset
        reset = manager.reset_period(budget.budget_id)
        assert reset.spent_usd == 0.0
        assert reset.status == BudgetStatus.ACTIVE


class TestBudgetSummary:
    """Tests for budget summary."""

    def test_get_summary(self, manager):
        """Test getting org budget summary."""
        manager.create_budget(
            org_id="test-org",
            name="Budget 1",
            amount_usd=100.0,
        )
        manager.create_budget(
            org_id="test-org",
            name="Budget 2",
            amount_usd=200.0,
        )

        manager.record_spend("test-org", 50.0)

        summary = manager.get_summary("test-org")
        assert summary["total_budget_usd"] == 300.0
        assert summary["total_spent_usd"] == 100.0  # 50 per budget
        assert summary["active_budgets"] == 2


class TestDefaultThresholds:
    """Tests for default threshold constants."""

    def test_default_thresholds_exist(self):
        """Test that default thresholds are defined."""
        assert len(DEFAULT_THRESHOLDS) == 4

    def test_default_thresholds_order(self):
        """Test that default thresholds are in order."""
        percentages = [t.percentage for t in DEFAULT_THRESHOLDS]
        assert percentages == sorted(percentages)

    def test_default_threshold_actions(self):
        """Test that default thresholds have expected actions."""
        actions = {t.percentage: t.action for t in DEFAULT_THRESHOLDS}
        assert actions[0.50] == BudgetAction.NOTIFY
        assert actions[0.75] == BudgetAction.WARN
        assert actions[0.90] == BudgetAction.SOFT_LIMIT
        assert actions[1.00] == BudgetAction.HARD_LIMIT


class TestBudgetProperties:
    """Tests for Budget dataclass properties."""

    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        budget = Budget(
            budget_id="test",
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
            spent_usd=25.0,
        )
        assert budget.usage_percentage == 0.25

    def test_remaining_usd(self):
        """Test remaining USD calculation."""
        budget = Budget(
            budget_id="test",
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
            spent_usd=25.0,
        )
        assert budget.remaining_usd == 75.0

    def test_is_exceeded(self):
        """Test is_exceeded flag."""
        budget = Budget(
            budget_id="test",
            org_id="test-org",
            name="Test",
            amount_usd=100.0,
            spent_usd=100.0,
        )
        assert budget.is_exceeded is True

    def test_to_dict(self):
        """Test Budget serialization."""
        budget = Budget(
            budget_id="test-123",
            org_id="test-org",
            name="Test Budget",
            amount_usd=100.0,
            spent_usd=50.0,
        )

        data = budget.to_dict()
        assert data["budget_id"] == "test-123"
        assert data["name"] == "Test Budget"
        assert data["amount_usd"] == 100.0
        assert data["spent_usd"] == 50.0
        assert data["usage_percentage"] == 0.5
        assert data["remaining_usd"] == 50.0

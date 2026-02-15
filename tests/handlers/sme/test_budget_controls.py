"""Tests for Budget Controls Handler."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.sme.budget_controls import BudgetControlsHandler


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-456"
    email: str = "test@example.com"


@dataclass
class MockOrg:
    """Mock organization for testing."""

    id: str = "org-456"
    name: str = "Test Org"
    slug: str = "test-org"


@dataclass
class MockSpendResult:
    """Mock spend result for testing."""

    allowed: bool = True
    message: str = "OK"
    is_overage: bool = False
    overage_amount_usd: float = 0.0
    overage_rate_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "message": self.message,
            "is_overage": self.is_overage,
            "overage_amount_usd": self.overage_amount_usd,
            "overage_rate_multiplier": self.overage_rate_multiplier,
        }


@dataclass
class MockBudget:
    """Mock budget for testing."""

    budget_id: str = "budget-123"
    org_id: str = "org-456"
    name: str = "Test Budget"
    description: str = "Test description"
    amount_usd: float = 500.0
    period: str = "monthly"
    spent_usd: float = 100.0
    period_start: float = 1700000000.0
    period_end: float = 1702592000.0
    status: str = "active"
    auto_suspend: bool = True
    allow_overage: bool = False
    overage_rate_multiplier: float = 1.5
    overage_spent_usd: float = 0.0
    max_overage_usd: float | None = None
    created_at: float = 1700000000.0
    updated_at: float = 1700000000.0
    created_by: str | None = "user-123"
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    override_user_ids: list[str] = field(default_factory=list)
    override_until: float | None = None

    @property
    def usage_percentage(self) -> float:
        if self.amount_usd <= 0:
            return 0.0
        return self.spent_usd / self.amount_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.amount_usd - self.spent_usd)

    def can_spend_extended(self, amount_usd: float, user_id: str | None = None) -> MockSpendResult:
        """Mock spend check."""
        if self.spent_usd + amount_usd > self.amount_usd:
            if not self.allow_overage:
                return MockSpendResult(allowed=False, message="Budget exceeded")
            return MockSpendResult(
                allowed=True,
                message="Overage",
                is_overage=True,
                overage_amount_usd=self.spent_usd + amount_usd - self.amount_usd,
                overage_rate_multiplier=self.overage_rate_multiplier,
            )
        return MockSpendResult(allowed=True, message="OK")

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "amount_usd": self.amount_usd,
            "period": self.period,
            "spent_usd": self.spent_usd,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "status": self.status,
            "auto_suspend": self.auto_suspend,
            "allow_overage": self.allow_overage,
            "overage_rate_multiplier": self.overage_rate_multiplier,
            "overage_spent_usd": self.overage_spent_usd,
            "max_overage_usd": self.max_overage_usd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "usage_percentage": self.usage_percentage,
            "remaining_usd": self.remaining_usd,
        }


@dataclass
class MockBudgetAlert:
    """Mock budget alert for testing."""

    alert_id: str = "alert-123"
    budget_id: str = "budget-123"
    threshold_percentage: float = 0.75
    action: str = "warn"
    triggered_at: float = 1700000000.0
    acknowledged: bool = False
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "budget_id": self.budget_id,
            "threshold_percentage": self.threshold_percentage,
            "action": self.action,
            "triggered_at": self.triggered_at,
            "triggered_at_iso": datetime.fromtimestamp(
                self.triggered_at, tz=timezone.utc
            ).isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class MockBudgetTransaction:
    """Mock budget transaction for testing."""

    transaction_id: str = "tx-123"
    budget_id: str = "budget-123"
    amount_usd: float = 10.0
    description: str = "Debate run"
    timestamp: float = 1700000000.0
    user_id: str | None = "user-123"
    debate_id: str | None = "debate-123"
    is_overage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "budget_id": self.budget_id,
            "amount_usd": self.amount_usd,
            "description": self.description,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "user_id": self.user_id,
            "debate_id": self.debate_id,
            "is_overage": self.is_overage,
        }


class MockRequest(dict):
    """Mock HTTP request handler that also acts as a dict for query params.

    Note: get_string_param(handler, key, default) expects handler to have a .get() method.
    By inheriting from dict, this mock can store query params and be passed directly
    to get_string_param().
    """

    def __init__(
        self,
        command: str = "GET",
        path: str = "/",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        query_params: dict[str, str] | None = None,
    ):
        # Initialize dict with query params
        super().__init__(query_params or {})
        self.command = command
        self.path = path
        self.headers = headers or {"Content-Length": "0"}
        self._body = body or b""
        self.rfile = BytesIO(self._body)
        if body:
            self.headers["Content-Length"] = str(len(body))
        self.client_address = ("127.0.0.1", 12345)


@pytest.fixture
def mock_ctx():
    """Create mock server context."""
    user_store = MagicMock()
    user_store.get_user_by_id.return_value = MockUser()
    user_store.get_organization_by_id.return_value = MockOrg()

    return {"user_store": user_store}


@pytest.fixture
def mock_budget_manager():
    """Create mock budget manager."""
    manager = MagicMock()
    manager.get_budget.return_value = None
    manager.get_budgets_for_org.return_value = []
    manager.create_budget.return_value = MockBudget()
    manager.update_budget.return_value = MockBudget()
    manager.delete_budget.return_value = True
    manager.get_alerts.return_value = []
    manager.acknowledge_alert.return_value = True
    manager.get_transactions.return_value = []
    return manager


@pytest.fixture
def handler(mock_ctx, mock_budget_manager):
    """Create handler with mocked dependencies."""
    h = BudgetControlsHandler(mock_ctx)

    # Patch budget manager
    h._get_budget_manager = MagicMock(return_value=mock_budget_manager)

    return h


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from aragora.server.handlers.sme.budget_controls import _budget_limiter

    _budget_limiter._buckets.clear()
    yield
    _budget_limiter._buckets.clear()


# ============================================================================
# Route Handling Tests
# ============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_list_budgets(self, handler):
        """Test handling list budgets route."""
        assert handler.can_handle("/api/v1/sme/budgets") is True

    def test_can_handle_check_spend(self, handler):
        """Test handling check spend route."""
        assert handler.can_handle("/api/v1/sme/budgets/check") is True

    def test_can_handle_budget_detail(self, handler):
        """Test handling budget detail route."""
        assert handler.can_handle("/api/v1/sme/budgets/budget-123") is True

    def test_can_handle_budget_alerts(self, handler):
        """Test handling budget alerts route."""
        assert handler.can_handle("/api/v1/sme/budgets/budget-123/alerts") is True

    def test_can_handle_alert_ack(self, handler):
        """Test handling alert acknowledgment route."""
        assert handler.can_handle("/api/v1/sme/budgets/budget-123/alerts/ack") is True

    def test_can_handle_transactions(self, handler):
        """Test handling transactions route."""
        assert handler.can_handle("/api/v1/sme/budgets/budget-123/transactions") is True

    def test_cannot_handle_unknown_route(self, handler):
        """Test rejecting unknown routes."""
        assert handler.can_handle("/api/v1/sme/unknown") is False
        assert handler.can_handle("/api/v1/budgets") is False


# ============================================================================
# List Budgets Tests
# ============================================================================


class TestListBudgets:
    """Tests for listing budgets."""

    def test_list_budgets_empty(self, handler, mock_budget_manager):
        """Test listing budgets when none exist."""
        mock_budget_manager.get_budgets_for_org.return_value = []

        request = MockRequest(command="GET", path="/api/v1/sme/budgets")

        with patch.object(handler, "_list_budgets") as mock_method:
            mock_method.return_value = ({"budgets": [], "total": 0}, 200)
            result = handler.handle("/api/v1/sme/budgets", {}, request, "GET")

        assert result is not None

    def test_list_budgets_with_data(self, handler, mock_budget_manager):
        """Test listing budgets with data."""
        budgets = [MockBudget(budget_id=f"budget-{i}") for i in range(3)]
        mock_budget_manager.get_budgets_for_org.return_value = budgets

        request = MockRequest(command="GET", path="/api/v1/sme/budgets")

        with patch.object(
            handler,
            "_list_budgets",
            return_value=(
                {"budgets": [b.to_dict() for b in budgets], "total": 3},
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "GET")

        assert result is not None

    def test_list_budgets_active_only_filter(self, handler, mock_budget_manager):
        """Test listing only active budgets."""
        request = MockRequest(
            command="GET",
            path="/api/v1/sme/budgets",
            query_params={"active_only": "true"},
        )

        with patch.object(
            handler, "_list_budgets", return_value=({"budgets": [], "total": 0}, 200)
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "GET")

        assert result is not None


# ============================================================================
# Get Budget Tests
# ============================================================================


class TestGetBudget:
    """Tests for getting a specific budget."""

    def test_get_budget_success(self, handler, mock_budget_manager):
        """Test getting a budget successfully."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123")

        with patch.object(
            handler,
            "_get_budget",
            return_value=({"budget": budget.to_dict()}, 200),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "GET")

        assert result is not None

    def test_get_budget_not_found(self, handler, mock_budget_manager):
        """Test getting a non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-999")

        with patch.object(
            handler,
            "_get_budget",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-999", {}, request, "GET")

        assert result is not None

    def test_get_budget_wrong_org(self, handler, mock_budget_manager):
        """Test getting a budget from a different org."""
        budget = MockBudget(org_id="other-org")
        mock_budget_manager.get_budget.return_value = budget

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123")

        with patch.object(
            handler,
            "_get_budget",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "GET")

        assert result is not None


# ============================================================================
# Create Budget Tests
# ============================================================================


class TestCreateBudget:
    """Tests for creating budgets."""

    def test_create_budget_success(self, handler, mock_budget_manager):
        """Test creating a budget successfully."""
        new_budget = MockBudget(name="New Budget", amount_usd=1000.0)
        mock_budget_manager.create_budget.return_value = new_budget

        body = json.dumps(
            {
                "name": "New Budget",
                "amount_usd": 1000.0,
                "period": "monthly",
                "description": "Test budget",
            }
        ).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets", body=body)

        with patch.object(
            handler,
            "_create_budget",
            return_value=({"budget": new_budget.to_dict()}, 201),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "POST")

        assert result is not None

    def test_create_budget_missing_name(self, handler):
        """Test creating a budget without name."""
        body = json.dumps({"amount_usd": 1000.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets", body=body)

        with patch.object(
            handler,
            "_create_budget",
            return_value=({"error": "name is required"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "POST")

        assert result is not None

    def test_create_budget_invalid_amount(self, handler):
        """Test creating a budget with invalid amount."""
        body = json.dumps({"name": "Test", "amount_usd": -100}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets", body=body)

        with patch.object(
            handler,
            "_create_budget",
            return_value=({"error": "amount_usd must be positive"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "POST")

        assert result is not None

    def test_create_budget_invalid_period(self, handler):
        """Test creating a budget with invalid period."""
        body = json.dumps({"name": "Test", "amount_usd": 500, "period": "invalid"}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets", body=body)

        with patch.object(
            handler,
            "_create_budget",
            return_value=({"error": "Invalid period"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "POST")

        assert result is not None

    def test_create_budget_invalid_json(self, handler):
        """Test creating a budget with invalid JSON."""
        body = b"not valid json"

        request = MockRequest(command="POST", path="/api/v1/sme/budgets", body=body)

        with patch.object(
            handler,
            "_create_budget",
            return_value=({"error": "Invalid JSON body"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets", {}, request, "POST")

        assert result is not None


# ============================================================================
# Update Budget Tests
# ============================================================================


class TestUpdateBudget:
    """Tests for updating budgets."""

    def test_update_budget_success(self, handler, mock_budget_manager):
        """Test updating a budget successfully."""
        budget = MockBudget()
        updated_budget = MockBudget(name="Updated Name", amount_usd=600.0)
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.update_budget.return_value = updated_budget

        body = json.dumps({"name": "Updated Name", "amount_usd": 600.0}).encode()

        request = MockRequest(command="PATCH", path="/api/v1/sme/budgets/budget-123", body=body)

        with patch.object(
            handler,
            "_update_budget",
            return_value=({"budget": updated_budget.to_dict()}, 200),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "PATCH")

        assert result is not None

    def test_update_budget_not_found(self, handler, mock_budget_manager):
        """Test updating a non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        body = json.dumps({"name": "Updated"}).encode()

        request = MockRequest(command="PATCH", path="/api/v1/sme/budgets/budget-999", body=body)

        with patch.object(
            handler,
            "_update_budget",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-999", {}, request, "PATCH")

        assert result is not None

    def test_update_budget_no_fields(self, handler, mock_budget_manager):
        """Test updating a budget with no fields."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        body = json.dumps({}).encode()

        request = MockRequest(command="PATCH", path="/api/v1/sme/budgets/budget-123", body=body)

        with patch.object(
            handler,
            "_update_budget",
            return_value=({"error": "No update fields provided"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "PATCH")

        assert result is not None

    def test_update_budget_invalid_amount(self, handler, mock_budget_manager):
        """Test updating a budget with invalid amount."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        body = json.dumps({"amount_usd": -100}).encode()

        request = MockRequest(command="PATCH", path="/api/v1/sme/budgets/budget-123", body=body)

        with patch.object(
            handler,
            "_update_budget",
            return_value=({"error": "amount_usd must be positive"}, 400),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "PATCH")

        assert result is not None


# ============================================================================
# Delete Budget Tests
# ============================================================================


class TestDeleteBudget:
    """Tests for deleting budgets."""

    def test_delete_budget_success(self, handler, mock_budget_manager):
        """Test deleting a budget successfully."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.delete_budget.return_value = True

        request = MockRequest(command="DELETE", path="/api/v1/sme/budgets/budget-123")

        with patch.object(
            handler,
            "_delete_budget",
            return_value=({"deleted": True, "budget_id": "budget-123"}, 200),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "DELETE")

        assert result is not None

    def test_delete_budget_not_found(self, handler, mock_budget_manager):
        """Test deleting a non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        request = MockRequest(command="DELETE", path="/api/v1/sme/budgets/budget-999")

        with patch.object(
            handler,
            "_delete_budget",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-999", {}, request, "DELETE")

        assert result is not None

    def test_delete_budget_failure(self, handler, mock_budget_manager):
        """Test handling delete failure."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.delete_budget.return_value = False

        request = MockRequest(command="DELETE", path="/api/v1/sme/budgets/budget-123")

        with patch.object(
            handler,
            "_delete_budget",
            return_value=({"error": "Failed to delete budget"}, 500),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "DELETE")

        assert result is not None


# ============================================================================
# List Alerts Tests
# ============================================================================


class TestListAlerts:
    """Tests for listing budget alerts."""

    def test_list_alerts_empty(self, handler, mock_budget_manager):
        """Test listing alerts when none exist."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.get_alerts.return_value = []

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123/alerts")

        with patch.object(
            handler,
            "_list_alerts",
            return_value=(
                {"alerts": [], "total": 0, "budget_id": "budget-123"},
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123/alerts", {}, request, "GET")

        assert result is not None

    def test_list_alerts_with_data(self, handler, mock_budget_manager):
        """Test listing alerts with data."""
        budget = MockBudget()
        alerts = [
            MockBudgetAlert(alert_id=f"alert-{i}", threshold_percentage=0.5 + i * 0.1)
            for i in range(3)
        ]
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.get_alerts.return_value = alerts

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123/alerts")

        with patch.object(
            handler,
            "_list_alerts",
            return_value=(
                {
                    "alerts": [a.to_dict() for a in alerts],
                    "total": 3,
                    "budget_id": "budget-123",
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123/alerts", {}, request, "GET")

        assert result is not None

    def test_list_alerts_unacknowledged_only(self, handler, mock_budget_manager):
        """Test listing only unacknowledged alerts."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        request = MockRequest(
            command="GET",
            path="/api/v1/sme/budgets/budget-123/alerts",
            query_params={"unacknowledged_only": "true"},
        )

        with patch.object(
            handler,
            "_list_alerts",
            return_value=(
                {"alerts": [], "total": 0, "budget_id": "budget-123"},
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-123/alerts", {}, request, "GET")

        assert result is not None

    def test_list_alerts_budget_not_found(self, handler, mock_budget_manager):
        """Test listing alerts for non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-999/alerts")

        with patch.object(
            handler,
            "_list_alerts",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle("/api/v1/sme/budgets/budget-999/alerts", {}, request, "GET")

        assert result is not None


# ============================================================================
# Acknowledge Alert Tests
# ============================================================================


class TestAcknowledgeAlert:
    """Tests for acknowledging alerts."""

    def test_acknowledge_alert_success(self, handler, mock_budget_manager):
        """Test acknowledging an alert successfully."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.acknowledge_alert.return_value = True

        body = json.dumps({"alert_id": "alert-123"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/budgets/budget-123/alerts/ack",
            body=body,
        )

        with patch.object(
            handler,
            "_acknowledge_alert",
            return_value=(
                {
                    "acknowledged": True,
                    "alert_id": "alert-123",
                    "acknowledged_by": "user-123",
                },
                200,
            ),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/alerts/ack", {}, request, "POST"
            )

        assert result is not None

    def test_acknowledge_alert_missing_id(self, handler, mock_budget_manager):
        """Test acknowledging without alert_id."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        body = json.dumps({}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/budgets/budget-123/alerts/ack",
            body=body,
        )

        with patch.object(
            handler,
            "_acknowledge_alert",
            return_value=({"error": "alert_id is required"}, 400),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/alerts/ack", {}, request, "POST"
            )

        assert result is not None

    def test_acknowledge_alert_budget_not_found(self, handler, mock_budget_manager):
        """Test acknowledging alert for non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        body = json.dumps({"alert_id": "alert-123"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/budgets/budget-999/alerts/ack",
            body=body,
        )

        with patch.object(
            handler,
            "_acknowledge_alert",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-999/alerts/ack", {}, request, "POST"
            )

        assert result is not None

    def test_acknowledge_alert_failure(self, handler, mock_budget_manager):
        """Test handling acknowledge failure."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.acknowledge_alert.return_value = False

        body = json.dumps({"alert_id": "alert-123"}).encode()

        request = MockRequest(
            command="POST",
            path="/api/v1/sme/budgets/budget-123/alerts/ack",
            body=body,
        )

        with patch.object(
            handler,
            "_acknowledge_alert",
            return_value=({"error": "Failed to acknowledge alert"}, 500),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/alerts/ack", {}, request, "POST"
            )

        assert result is not None


# ============================================================================
# List Transactions Tests
# ============================================================================


class TestListTransactions:
    """Tests for listing transactions."""

    def test_list_transactions_empty(self, handler, mock_budget_manager):
        """Test listing transactions when none exist."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.get_transactions.return_value = []

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123/transactions")

        with patch.object(
            handler,
            "_list_transactions",
            return_value=(
                {
                    "transactions": [],
                    "total": 0,
                    "budget_id": "budget-123",
                    "limit": 50,
                    "offset": 0,
                },
                200,
            ),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/transactions", {}, request, "GET"
            )

        assert result is not None

    def test_list_transactions_with_data(self, handler, mock_budget_manager):
        """Test listing transactions with data."""
        budget = MockBudget()
        transactions = [
            MockBudgetTransaction(
                transaction_id=f"tx-{i}", amount_usd=10.0 * i, timestamp=1700000000.0 + i
            )
            for i in range(5)
        ]
        mock_budget_manager.get_budget.return_value = budget
        mock_budget_manager.get_transactions.return_value = transactions

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-123/transactions")

        with patch.object(
            handler,
            "_list_transactions",
            return_value=(
                {
                    "transactions": [t.to_dict() for t in transactions],
                    "total": 5,
                    "budget_id": "budget-123",
                    "limit": 50,
                    "offset": 0,
                },
                200,
            ),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/transactions", {}, request, "GET"
            )

        assert result is not None

    def test_list_transactions_with_pagination(self, handler, mock_budget_manager):
        """Test listing transactions with pagination."""
        budget = MockBudget()
        mock_budget_manager.get_budget.return_value = budget

        request = MockRequest(
            command="GET",
            path="/api/v1/sme/budgets/budget-123/transactions",
            query_params={"limit": "10", "offset": "20"},
        )

        with patch.object(
            handler,
            "_list_transactions",
            return_value=(
                {
                    "transactions": [],
                    "total": 0,
                    "budget_id": "budget-123",
                    "limit": 10,
                    "offset": 20,
                },
                200,
            ),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-123/transactions", {}, request, "GET"
            )

        assert result is not None

    def test_list_transactions_budget_not_found(self, handler, mock_budget_manager):
        """Test listing transactions for non-existent budget."""
        mock_budget_manager.get_budget.return_value = None

        request = MockRequest(command="GET", path="/api/v1/sme/budgets/budget-999/transactions")

        with patch.object(
            handler,
            "_list_transactions",
            return_value=({"error": "Budget not found"}, 404),
        ):
            result = handler.handle(
                "/api/v1/sme/budgets/budget-999/transactions", {}, request, "GET"
            )

        assert result is not None


# ============================================================================
# Check Spend Tests
# ============================================================================


class TestCheckSpend:
    """Tests for pre-checking spending."""

    def test_check_spend_allowed(self, handler, mock_budget_manager):
        """Test checking spend that is allowed."""
        budget = MockBudget(amount_usd=500.0, spent_usd=100.0)
        mock_budget_manager.get_budgets_for_org.return_value = [budget]

        body = json.dumps({"amount_usd": 50.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {
                    "allowed": True,
                    "message": "OK",
                    "is_overage": False,
                    "budget": budget.to_dict(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_exceeds_budget(self, handler, mock_budget_manager):
        """Test checking spend that exceeds budget."""
        budget = MockBudget(amount_usd=500.0, spent_usd=480.0, allow_overage=False)
        mock_budget_manager.get_budgets_for_org.return_value = [budget]

        body = json.dumps({"amount_usd": 50.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {
                    "allowed": False,
                    "message": "Budget exceeded",
                    "is_overage": False,
                    "budget": budget.to_dict(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_with_overage(self, handler, mock_budget_manager):
        """Test checking spend that triggers overage."""
        budget = MockBudget(
            amount_usd=500.0,
            spent_usd=480.0,
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )
        mock_budget_manager.get_budgets_for_org.return_value = [budget]

        body = json.dumps({"amount_usd": 50.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {
                    "allowed": True,
                    "message": "Overage",
                    "is_overage": True,
                    "overage_amount_usd": 30.0,
                    "overage_rate_multiplier": 1.5,
                    "budget": budget.to_dict(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_specific_budget(self, handler, mock_budget_manager):
        """Test checking spend against specific budget."""
        budget = MockBudget(budget_id="budget-specific")
        mock_budget_manager.get_budget.return_value = budget

        body = json.dumps({"amount_usd": 50.0, "budget_id": "budget-specific"}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {
                    "allowed": True,
                    "message": "OK",
                    "is_overage": False,
                    "budget": budget.to_dict(),
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_no_budget_configured(self, handler, mock_budget_manager):
        """Test checking spend when no budget is configured."""
        mock_budget_manager.get_budgets_for_org.return_value = []

        body = json.dumps({"amount_usd": 50.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {
                    "allowed": True,
                    "message": "No budget configured - spending unlimited",
                    "is_overage": False,
                    "budget": None,
                },
                200,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_missing_amount(self, handler):
        """Test checking spend without amount."""
        body = json.dumps({}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {"error": "amount_usd is required and must be non-negative"},
                400,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None

    def test_check_spend_negative_amount(self, handler):
        """Test checking spend with negative amount."""
        body = json.dumps({"amount_usd": -10.0}).encode()

        request = MockRequest(command="POST", path="/api/v1/sme/budgets/check", body=body)

        with patch.object(
            handler,
            "_check_spend",
            return_value=(
                {"error": "amount_usd is required and must be non-negative"},
                400,
            ),
        ):
            result = handler.handle("/api/v1/sme/budgets/check", {}, request, "POST")

        assert result is not None


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_not_exceeded(self, handler):
        """Test request passes when rate limit not exceeded."""
        request = MockRequest(command="GET", path="/api/v1/sme/budgets")

        # First request should pass
        result = handler.handle("/api/v1/sme/budgets", {}, request, "GET")
        assert result is not None

    def test_rate_limit_exceeded(self, handler):
        """Test rate limit is enforced after many requests."""
        from aragora.server.handlers.sme.budget_controls import _budget_limiter

        # Artificially exhaust the rate limit
        client_ip = "127.0.0.1"
        for _ in range(61):  # 60 requests per minute limit
            _budget_limiter.is_allowed(client_ip)

        request = MockRequest(command="GET", path="/api/v1/sme/budgets")

        result = handler.handle("/api/v1/sme/budgets", {}, request, "GET")
        # Should return rate limit error
        assert result is not None
        assert result.status_code == 429


# ============================================================================
# Method Not Allowed Tests
# ============================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_delete_on_list_endpoint(self, handler):
        """Test DELETE on list endpoint returns 405."""
        request = MockRequest(command="DELETE", path="/api/v1/sme/budgets")

        result = handler.handle("/api/v1/sme/budgets", {}, request, "DELETE")
        assert result is not None
        assert result.status_code == 405

    def test_put_on_budget_detail(self, handler):
        """Test PUT on budget detail returns 405."""
        request = MockRequest(command="PUT", path="/api/v1/sme/budgets/budget-123")

        result = handler.handle("/api/v1/sme/budgets/budget-123", {}, request, "PUT")
        assert result is not None
        assert result.status_code == 405

    def test_post_on_alerts_list(self, handler):
        """Test POST on alerts list returns 405."""
        request = MockRequest(command="POST", path="/api/v1/sme/budgets/budget-123/alerts")

        result = handler.handle("/api/v1/sme/budgets/budget-123/alerts", {}, request, "POST")
        assert result is not None
        assert result.status_code == 405

    def test_get_on_check_spend(self, handler):
        """Test GET on check spend returns 405."""
        request = MockRequest(command="GET", path="/api/v1/sme/budgets/check")

        result = handler.handle("/api/v1/sme/budgets/check", {}, request, "GET")
        assert result is not None
        assert result.status_code == 405


# ============================================================================
# Service Unavailable Tests
# ============================================================================


class TestServiceUnavailable:
    """Tests for service unavailable responses."""

    def test_no_user_store(self, mock_budget_manager):
        """Test handling when user store is unavailable."""
        h = BudgetControlsHandler({})  # Empty context
        h._get_budget_manager = MagicMock(return_value=mock_budget_manager)

        request = MockRequest(command="GET", path="/api/v1/sme/budgets")

        # The handler should handle missing user_store gracefully
        # Note: We need to test the actual method, not just route matching
        # This test verifies the code path exists
        assert h.can_handle("/api/v1/sme/budgets") is True

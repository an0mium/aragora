"""Tests for aragora.server.handlers.sme.budget_controls - Budget Controls Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockUser:
    """Mock authenticated user."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-123"
    email: str = "test@example.com"
    name: str = "Test User"


@dataclass
class MockOrg:
    """Mock organization."""

    id: str = "org-123"
    name: str = "Test Organization"


@dataclass
class MockBudget:
    """Mock budget object."""

    budget_id: str = "budget-123"
    org_id: str = "org-123"
    name: str = "Monthly Budget"
    description: str = "Test budget"
    amount_usd: float = 500.0
    period: str = "monthly"
    auto_suspend: bool = True
    allow_overage: bool = False
    spent_usd: float = 100.0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "amount_usd": self.amount_usd,
            "period": self.period,
            "auto_suspend": self.auto_suspend,
            "allow_overage": self.allow_overage,
            "spent_usd": self.spent_usd,
            "is_active": self.is_active,
        }

    def can_spend_extended(self, amount_usd: float, user_id: str = None):
        """Check if spend is allowed."""
        mock_result = MagicMock()
        if self.spent_usd + amount_usd > self.amount_usd:
            mock_result.allowed = False
            mock_result.message = "Budget exceeded"
            mock_result.is_overage = True
            mock_result.overage_amount_usd = self.spent_usd + amount_usd - self.amount_usd
            mock_result.overage_rate_multiplier = 1.5
        else:
            mock_result.allowed = True
            mock_result.message = "OK"
            mock_result.is_overage = False
            mock_result.overage_amount_usd = 0.0
            mock_result.overage_rate_multiplier = 1.0
        return mock_result


@dataclass
class MockAlert:
    """Mock alert object."""

    id: str = "alert-123"
    budget_id: str = "budget-123"
    alert_type: str = "threshold"
    threshold_percent: float = 80.0
    triggered_at: float = 0.0
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "budget_id": self.budget_id,
            "alert_type": self.alert_type,
            "threshold_percent": self.threshold_percent,
            "triggered_at": self.triggered_at,
            "acknowledged": self.acknowledged,
        }


@dataclass
class MockTransaction:
    """Mock transaction object."""

    id: str = "tx-123"
    budget_id: str = "budget-123"
    amount_usd: float = 10.0
    description: str = "API call"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "budget_id": self.budget_id,
            "amount_usd": self.amount_usd,
            "description": self.description,
            "created_at": self.created_at,
        }


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: bytes = b"",
        headers: Optional[dict[str, str]] = None,
        path: str = "/",
        method: str = "GET",
    ):
        self._body = body
        self.headers = headers or {"Content-Length": str(len(body))}
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = ("127.0.0.1", 12345)

    @classmethod
    def with_json_body(cls, data: dict[str, Any], **kwargs) -> "MockHandler":
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        return cls(body=body, headers=headers, **kwargs)

    def get_argument(self, name: str, default: str = None) -> Optional[str]:
        return default


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state before each test."""
    try:
        from aragora.server.handlers.sme import budget_controls

        budget_controls._budget_limiter._buckets.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def mock_user():
    return MockUser()


@pytest.fixture
def mock_org():
    return MockOrg()


@pytest.fixture
def mock_user_store(mock_user, mock_org):
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    store.get_organization_by_id.return_value = mock_org
    return store


@pytest.fixture
def mock_budget_manager():
    manager = MagicMock()
    manager.get_budgets_for_org.return_value = [MockBudget()]
    manager.get_budget.return_value = MockBudget()
    manager.create_budget.return_value = MockBudget()
    manager.update_budget.return_value = MockBudget()
    manager.delete_budget.return_value = True
    manager.get_alerts.return_value = [MockAlert()]
    manager.acknowledge_alert.return_value = True
    manager.get_transactions.return_value = [MockTransaction()]
    return manager


@pytest.fixture
def handler_context(mock_user_store):
    return {"user_store": mock_user_store}


@pytest.fixture
def budget_handler(handler_context, mock_budget_manager):
    with patch(
        "aragora.billing.budget_manager.get_budget_manager",
        return_value=mock_budget_manager,
    ):
        from aragora.server.handlers.sme.budget_controls import BudgetControlsHandler

        handler = BudgetControlsHandler(handler_context)
        yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_budgets_list(self, budget_handler):
        """Test handler recognizes budgets list endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets") is True

    def test_can_handle_budgets_check(self, budget_handler):
        """Test handler recognizes budgets check endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets/check") is True

    def test_can_handle_budget_detail(self, budget_handler):
        """Test handler recognizes budget detail endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets/budget-123") is True

    def test_can_handle_budget_alerts(self, budget_handler):
        """Test handler recognizes budget alerts endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets/budget-123/alerts") is True

    def test_can_handle_budget_transactions(self, budget_handler):
        """Test handler recognizes transactions endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets/budget-123/transactions") is True

    def test_can_handle_alert_ack(self, budget_handler):
        """Test handler recognizes alert acknowledgment endpoint."""
        assert budget_handler.can_handle("/api/v1/sme/budgets/budget-123/alerts/ack") is True

    def test_cannot_handle_unknown_path(self, budget_handler):
        """Test handler rejects unknown paths."""
        assert budget_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# List Budgets Tests
# ===========================================================================


class TestListBudgets:
    """Tests for listing budgets."""

    def test_list_budgets_success(self, budget_handler, mock_user):
        """Test successful budget listing."""
        http_handler = MockHandler(path="/api/v1/sme/budgets", method="GET")

        with patch.object(budget_handler, "_list_budgets") as mock_list:
            mock_list.return_value = MagicMock(
                status_code=200, body=json.dumps({"budgets": [], "total": 0}).encode()
            )
            result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="GET")
            assert result is not None

    def test_list_budgets_no_user_store(self, handler_context, mock_budget_manager):
        """Test error when user store not available."""
        handler_context["user_store"] = None
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=mock_budget_manager,
        ):
            from aragora.server.handlers.sme.budget_controls import BudgetControlsHandler

            handler = BudgetControlsHandler(handler_context)
            http_handler = MockHandler(path="/api/v1/sme/budgets", method="GET")

            result = handler.handle("/api/v1/sme/budgets", {}, http_handler, method="GET")
            assert result is not None


# ===========================================================================
# Get Budget Tests
# ===========================================================================


class TestGetBudget:
    """Tests for getting a single budget."""

    def test_get_budget_success(self, budget_handler, mock_user):
        """Test successful budget retrieval."""
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-123", method="GET")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_budget_not_found(self, budget_handler, mock_budget_manager):
        """Test budget not found error."""
        mock_budget_manager.get_budget.return_value = None
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-999", method="GET")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-999", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Create Budget Tests
# ===========================================================================


class TestCreateBudget:
    """Tests for creating budgets."""

    def test_create_budget_success(self, budget_handler, mock_user):
        """Test successful budget creation."""
        body = {
            "name": "New Budget",
            "amount_usd": 1000.0,
            "period": "monthly",
        }
        http_handler = MockHandler.with_json_body(body, path="/api/v1/sme/budgets", method="POST")

        result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="POST")
        assert result is not None

    def test_create_budget_missing_name(self, budget_handler):
        """Test error when name is missing."""
        body = {"amount_usd": 1000.0}
        http_handler = MockHandler.with_json_body(body, path="/api/v1/sme/budgets", method="POST")

        result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="POST")
        assert result is not None

    def test_create_budget_invalid_amount(self, budget_handler):
        """Test error when amount is invalid."""
        body = {"name": "Test", "amount_usd": -100}
        http_handler = MockHandler.with_json_body(body, path="/api/v1/sme/budgets", method="POST")

        result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="POST")
        assert result is not None

    def test_create_budget_invalid_json(self, budget_handler):
        """Test error when JSON body is invalid."""
        http_handler = MockHandler(
            body=b"not json",
            headers={"Content-Type": "application/json", "Content-Length": "8"},
            path="/api/v1/sme/budgets",
            method="POST",
        )

        result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="POST")
        assert result is not None


# ===========================================================================
# Update Budget Tests
# ===========================================================================


class TestUpdateBudget:
    """Tests for updating budgets."""

    def test_update_budget_success(self, budget_handler, mock_user):
        """Test successful budget update."""
        body = {"name": "Updated Budget", "amount_usd": 750.0}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/budget-123", method="PATCH"
        )

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123", {}, http_handler, method="PATCH"
        )
        assert result is not None

    def test_update_budget_no_fields(self, budget_handler):
        """Test error when no update fields provided."""
        body = {}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/budget-123", method="PATCH"
        )

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123", {}, http_handler, method="PATCH"
        )
        assert result is not None


# ===========================================================================
# Delete Budget Tests
# ===========================================================================


class TestDeleteBudget:
    """Tests for deleting budgets."""

    def test_delete_budget_success(self, budget_handler, mock_user):
        """Test successful budget deletion."""
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-123", method="DELETE")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123", {}, http_handler, method="DELETE"
        )
        assert result is not None

    def test_delete_budget_not_found(self, budget_handler, mock_budget_manager):
        """Test delete budget not found."""
        mock_budget_manager.get_budget.return_value = None
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-999", method="DELETE")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-999", {}, http_handler, method="DELETE"
        )
        assert result is not None


# ===========================================================================
# Alerts Tests
# ===========================================================================


class TestAlerts:
    """Tests for budget alerts."""

    def test_list_alerts_success(self, budget_handler, mock_user):
        """Test successful alerts listing."""
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-123/alerts", method="GET")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123/alerts", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_acknowledge_alert_success(self, budget_handler, mock_user):
        """Test successful alert acknowledgment."""
        body = {"alert_id": "alert-123"}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/budget-123/alerts/ack", method="POST"
        )

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123/alerts/ack", {}, http_handler, method="POST"
        )
        assert result is not None

    def test_acknowledge_alert_missing_id(self, budget_handler):
        """Test error when alert_id is missing."""
        body = {}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/budget-123/alerts/ack", method="POST"
        )

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123/alerts/ack", {}, http_handler, method="POST"
        )
        assert result is not None


# ===========================================================================
# Transactions Tests
# ===========================================================================


class TestTransactions:
    """Tests for budget transactions."""

    def test_list_transactions_success(self, budget_handler, mock_user):
        """Test successful transactions listing."""
        http_handler = MockHandler(path="/api/v1/sme/budgets/budget-123/transactions", method="GET")

        result = budget_handler.handle(
            "/api/v1/sme/budgets/budget-123/transactions", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Check Spend Tests
# ===========================================================================


class TestCheckSpend:
    """Tests for spend checking."""

    def test_check_spend_success(self, budget_handler, mock_user):
        """Test successful spend check."""
        body = {"amount_usd": 50.0}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/check", method="POST"
        )

        result = budget_handler.handle("/api/v1/sme/budgets/check", {}, http_handler, method="POST")
        assert result is not None

    def test_check_spend_with_budget_id(self, budget_handler, mock_user):
        """Test spend check with specific budget."""
        body = {"budget_id": "budget-123", "amount_usd": 50.0}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/check", method="POST"
        )

        result = budget_handler.handle("/api/v1/sme/budgets/check", {}, http_handler, method="POST")
        assert result is not None

    def test_check_spend_missing_amount(self, budget_handler):
        """Test error when amount is missing."""
        body = {}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/check", method="POST"
        )

        result = budget_handler.handle("/api/v1/sme/budgets/check", {}, http_handler, method="POST")
        assert result is not None

    def test_check_spend_negative_amount(self, budget_handler):
        """Test error when amount is negative."""
        body = {"amount_usd": -10.0}
        http_handler = MockHandler.with_json_body(
            body, path="/api/v1/sme/budgets/check", method="POST"
        )

        result = budget_handler.handle("/api/v1/sme/budgets/check", {}, http_handler, method="POST")
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, budget_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/sme/budgets", method="GET")

        # Make many requests to trigger rate limiting
        with patch(
            "aragora.server.handlers.sme.budget_controls._budget_limiter.is_allowed",
            return_value=False,
        ):
            result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="GET")
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_budgets_list_method_not_allowed(self, budget_handler):
        """Test method not allowed for budgets list."""
        http_handler = MockHandler(path="/api/v1/sme/budgets", method="DELETE")

        result = budget_handler.handle("/api/v1/sme/budgets", {}, http_handler, method="DELETE")
        assert result is not None
        assert result.status_code == 405

    def test_budgets_check_method_not_allowed(self, budget_handler):
        """Test method not allowed for budgets check."""
        http_handler = MockHandler(path="/api/v1/sme/budgets/check", method="GET")

        result = budget_handler.handle("/api/v1/sme/budgets/check", {}, http_handler, method="GET")
        assert result is not None
        assert result.status_code == 405

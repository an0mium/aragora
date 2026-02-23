"""Tests for budget management handler (aragora/server/handlers/budgets.py).

Covers all routes and behavior of the BudgetHandler class:
- can_handle() routing
- GET    /api/v1/budgets              - List budgets for org
- POST   /api/v1/budgets              - Create a budget
- GET    /api/v1/budgets/:id          - Get budget details
- PATCH  /api/v1/budgets/:id          - Update a budget
- DELETE /api/v1/budgets/:id          - Delete (close) a budget
- GET    /api/v1/budgets/:id/alerts   - Get alerts for a budget
- POST   /api/v1/budgets/:id/alerts/:alert_id/acknowledge - Acknowledge alert
- POST   /api/v1/budgets/:id/override - Add override for user
- DELETE /api/v1/budgets/:id/override/:user_id - Remove override
- POST   /api/v1/budgets/:id/reset    - Reset budget period
- GET    /api/v1/budgets/:id/transactions - Get transaction history
- GET    /api/v1/budgets/:id/trends   - Get spending trends
- GET    /api/v1/budgets/summary      - Get org budget summary
- GET    /api/v1/budgets/trends       - Get org-wide spending trends
- POST   /api/v1/budgets/check        - Pre-flight cost check
- GET    /api/v1/costs/agents         - Per-agent cost breakdown
- GET    /api/v1/costs/anomalies      - Cost anomaly detection
- Circuit breaker integration
- Input validation, edge cases, error paths
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.budgets import (
    BudgetHandler,
    get_budget_circuit_breaker,
    reset_budget_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract body dict from a HandlerResult."""
    if hasattr(result, "body"):
        # HandlerResult dataclass with bytes body
        return json.loads(result.body.decode("utf-8"))
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return 200


class _MockHTTPHandler:
    """Minimal mock HTTP handler that simulates BaseHTTPRequestHandler."""

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        path: str = "/api/v1/budgets",
        method: str = "GET",
        org_id: str = "test-org-001",
        user_id: str = "test-user-001",
    ):
        self.path = path
        self.command = method
        self.org_id = org_id
        self.user_id = user_id
        # Set up rfile for read_json_body
        if body is not None:
            body_bytes = json.dumps(body).encode()
        else:
            body_bytes = b"{}"
        self.rfile = MagicMock()
        self.rfile.read.return_value = body_bytes
        self.headers = {"Content-Length": str(len(body_bytes))}


# Mock budget dataclass
@dataclass
class _MockBudget:
    budget_id: str = "budget-001"
    org_id: str = "test-org-001"
    name: str = "Test Budget"
    amount_usd: float = 1000.0
    period: str = "monthly"
    description: str = "Test description"
    status: str = "active"
    auto_suspend: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "org_id": self.org_id,
            "name": self.name,
            "amount_usd": self.amount_usd,
            "period": self.period,
            "description": self.description,
            "status": self.status,
            "auto_suspend": self.auto_suspend,
        }


@dataclass
class _MockAlert:
    alert_id: str = "alert-001"
    budget_id: str = "budget-001"
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "budget_id": self.budget_id,
            "severity": self.severity,
        }


@dataclass
class _MockTransaction:
    transaction_id: str = "txn-001"
    budget_id: str = "budget-001"
    amount_usd: float = 10.0
    user_id: str = "test-user-001"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "budget_id": self.budget_id,
            "amount_usd": self.amount_usd,
            "user_id": self.user_id,
        }


class _MockAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a BudgetHandler with empty context."""
    return BudgetHandler({})


@pytest.fixture(autouse=True)
def _reset_cb():
    """Reset the circuit breaker between tests."""
    reset_budget_circuit_breaker()
    yield
    reset_budget_circuit_breaker()


@pytest.fixture
def mock_manager():
    """Create a mock budget manager."""
    mgr = MagicMock()
    mgr.get_budgets_for_org.return_value = []
    mgr.get_budget.return_value = _MockBudget()
    mgr.create_budget.return_value = _MockBudget()
    mgr.update_budget.return_value = _MockBudget()
    mgr.delete_budget.return_value = True
    mgr.get_summary.return_value = {"total": 0, "active": 0}
    mgr.check_budget.return_value = (True, "Within budget", _MockAction.ALLOW)
    mgr.get_alerts.return_value = []
    mgr.acknowledge_alert.return_value = True
    mgr.add_override.return_value = True
    mgr.remove_override.return_value = True
    mgr.reset_period.return_value = _MockBudget()
    mgr.get_transactions.return_value = []
    mgr.count_transactions.return_value = 0
    mgr.get_spending_trends.return_value = []
    mgr.get_org_spending_trends.return_value = []
    return mgr


def _make_auth_ctx(
    authenticated: bool = True,
    user_id: str = "test-user-001",
    email: str = "test@example.com",
    org_id: str = "test-org-001",
    role: str = "admin",
):
    """Create a mock UserAuthContext."""
    ctx = MagicMock()
    ctx.authenticated = authenticated
    ctx.is_authenticated = authenticated
    ctx.user_id = user_id
    ctx.email = email
    ctx.org_id = org_id
    ctx.role = role
    return ctx


def _make_permission_decision(allowed: bool = True, reason: str = "granted"):
    """Create a mock permission decision."""
    d = MagicMock()
    d.allowed = allowed
    d.reason = reason
    return d


@pytest.fixture
def _patch_auth():
    """Patch JWT auth to return an authenticated admin user."""
    auth_ctx = _make_auth_ctx()
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=auth_ctx,
    ):
        yield auth_ctx


@pytest.fixture
def _patch_auth_and_rbac():
    """Patch JWT auth and RBAC checker to allow access."""
    auth_ctx = _make_auth_ctx()
    decision = _make_permission_decision(allowed=True)
    checker = MagicMock()
    checker.check_permission.return_value = decision
    with (
        patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ),
        patch(
            "aragora.rbac.checker.get_permission_checker",
            return_value=checker,
        ),
    ):
        yield auth_ctx, checker


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify can_handle accepts or rejects paths correctly."""

    def test_budgets_root(self, handler):
        assert handler.can_handle("/api/v1/budgets", "GET")

    def test_budgets_with_id(self, handler):
        assert handler.can_handle("/api/v1/budgets/budget-001", "GET")

    def test_budgets_summary(self, handler):
        assert handler.can_handle("/api/v1/budgets/summary", "GET")

    def test_budgets_check(self, handler):
        assert handler.can_handle("/api/v1/budgets/check", "POST")

    def test_budgets_trends(self, handler):
        assert handler.can_handle("/api/v1/budgets/trends", "GET")

    def test_budgets_alerts(self, handler):
        assert handler.can_handle("/api/v1/budgets/budget-001/alerts", "GET")

    def test_budgets_transactions(self, handler):
        assert handler.can_handle("/api/v1/budgets/budget-001/transactions", "GET")

    def test_costs_agents(self, handler):
        assert handler.can_handle("/api/v1/costs/agents", "GET")

    def test_costs_anomalies(self, handler):
        assert handler.can_handle("/api/v1/costs/anomalies", "GET")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/users", "GET")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("", "GET")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/", "GET")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/budget", "GET")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = BudgetHandler({})
        assert h.ctx == {}

    def test_init_with_none(self):
        h = BudgetHandler(None)
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"key": "value"}
        h = BudgetHandler(ctx)
        assert h.ctx == ctx

    def test_has_circuit_breaker(self, handler):
        assert handler._circuit_breaker is not None

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v1/budgets" in handler.ROUTES

    def test_validation_constants(self, handler):
        assert handler.MAX_NAME_LENGTH == 200
        assert handler.MAX_DESCRIPTION_LENGTH == 2000
        assert handler.MAX_AMOUNT_USD == 1_000_000_000
        assert handler.MIN_AMOUNT_USD == 0.01


# ============================================================================
# Authentication
# ============================================================================


class TestAuthentication:
    """Test authentication enforcement."""

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, handler):
        auth_ctx = _make_auth_ctx(authenticated=False)
        mock_handler = _MockHTTPHandler()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_ctx,
        ):
            result = await handler.handle("/api/v1/budgets", {}, mock_handler)
        assert _status(result) == 401
        assert "Authentication required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_none_user_returns_401(self, handler):
        mock_handler = _MockHTTPHandler()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=None,
        ):
            result = await handler.handle("/api/v1/budgets", {}, mock_handler)
        assert _status(result) == 401


# ============================================================================
# RBAC Permission Checks
# ============================================================================


class TestRBACPermissions:
    """Test RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_read_denied_returns_403(self, handler):
        auth_ctx = _make_auth_ctx()
        decision = _make_permission_decision(allowed=False, reason="no read")
        checker = MagicMock()
        checker.check_permission.return_value = decision
        mock_handler = _MockHTTPHandler()
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=checker,
            ),
        ):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 403
        assert "Permission denied" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_write_denied_returns_403(self, handler):
        auth_ctx = _make_auth_ctx()
        decision = _make_permission_decision(allowed=False, reason="no write")
        checker = MagicMock()
        checker.check_permission.return_value = decision
        mock_handler = _MockHTTPHandler(body={"name": "Test", "amount_usd": 100}, method="POST")
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=checker,
            ),
        ):
            result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_delete_denied_returns_403(self, handler):
        auth_ctx = _make_auth_ctx()
        decision = _make_permission_decision(allowed=False, reason="no delete")
        checker = MagicMock()
        checker.check_permission.return_value = decision
        mock_handler = _MockHTTPHandler(method="DELETE")
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=checker,
            ),
        ):
            result = await handler.handle("/api/v1/budgets/budget-001", "DELETE", mock_handler)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_rbac_import_error_allows_access(self, handler, mock_manager):
        """When RBAC module is not available, access should still be granted."""
        auth_ctx = _make_auth_ctx()
        mock_handler = _MockHTTPHandler()
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                side_effect=ImportError("no rbac"),
            ),
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
        ):
            result = await handler.handle("/api/v1/budgets/summary", "GET", mock_handler)
        assert _status(result) == 200


# ============================================================================
# Circuit Breaker
# ============================================================================


class TestCircuitBreaker:
    """Test circuit breaker integration."""

    def test_get_circuit_breaker(self):
        cb = get_budget_circuit_breaker()
        assert cb is not None
        assert cb.name == "budget"

    def test_reset_circuit_breaker(self):
        cb = get_budget_circuit_breaker()
        # Force failures
        for _ in range(10):
            cb.record_failure()
        assert cb.state != "closed"
        reset_budget_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_open_circuit_breaker_returns_503(self, handler):
        auth_ctx = _make_auth_ctx()
        decision = _make_permission_decision(allowed=True)
        checker = MagicMock()
        checker.check_permission.return_value = decision
        mock_handler = _MockHTTPHandler()
        # Open the circuit breaker
        cb = get_budget_circuit_breaker()
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=checker,
            ),
        ):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"]

    def test_circuit_breaker_status(self, handler):
        status = handler.get_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status
        assert status["state"] == "closed"


# ============================================================================
# GET /api/v1/budgets - List Budgets
# ============================================================================


class TestListBudgets:
    """Test listing budgets."""

    @pytest.mark.asyncio
    async def test_list_empty(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["budgets"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_with_budgets(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budgets_for_org.return_value = [_MockBudget(), _MockBudget(budget_id="budget-002")]
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["budgets"]) == 2

    @pytest.mark.asyncio
    async def test_list_with_active_only_false(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets?active_only=false")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 200
        mock_manager.get_budgets_for_org.assert_called_once_with("test-org-001", active_only=False)

    @pytest.mark.asyncio
    async def test_list_manager_error(self, handler, _patch_auth_and_rbac):
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("no module")):
            mock_handler = _MockHTTPHandler(path="/api/v1/budgets")
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        assert _status(result) == 500
        assert "Failed to list budgets" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_includes_org_id(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets", "GET", mock_handler)
        body = _body(result)
        assert body["org_id"] == "test-org-001"


# ============================================================================
# POST /api/v1/budgets - Create Budget
# ============================================================================


class TestCreateBudget:
    """Test budget creation."""

    @pytest.mark.asyncio
    async def test_create_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "New Budget", "amount_usd": 500, "period": "monthly"},
            method="POST",
        )
        with (
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
            patch("aragora.billing.budget_manager.BudgetPeriod", side_effect=lambda v: v),
        ):
            result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_missing_name(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"amount_usd": 500},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "name" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_empty_name(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "   ", "amount_usd": 500},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "empty" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_name_not_string(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": 123, "amount_usd": 500},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_name_too_long(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "x" * 201, "amount_usd": 500},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "maximum length" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_missing_amount(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget"},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "amount_usd" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_invalid_amount_type(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": "not_a_number"},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "must be a number" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_amount_too_small(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 0.001},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "at least" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_amount_too_large(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 2_000_000_000},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "maximum" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_invalid_period(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "period": "biweekly"},
            method="POST",
        )
        with patch(
            "aragora.billing.budget_manager.BudgetPeriod",
            side_effect=ValueError("invalid period"),
        ):
            result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "Invalid period" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_period_not_string(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "period": 42},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "period must be a string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_description_not_string(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "description": 123},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "description must be a string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_description_too_long(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "description": "x" * 2001},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "maximum length" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_auto_suspend_not_bool(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "auto_suspend": "yes"},
            method="POST",
        )
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 400
        assert "boolean" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_empty_body(self, handler, _patch_auth_and_rbac):
        # read_json_body returns {} for empty Content-Length, but name is missing
        mock_handler = _MockHTTPHandler(body=None, method="POST")
        # Override to return empty body
        mock_handler.headers = {"Content-Length": "0"}
        result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        # read_json_body returns {} for content_length <= 0, then
        # body = {} is truthy, so it proceeds to check name
        # But body.get("name") is None -> "Missing required field: name"
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"name": "Budget", "amount_usd": 100, "period": "monthly"},
            method="POST",
        )
        with (
            patch.object(handler, "_get_budget_manager", side_effect=RuntimeError("fail")),
            patch("aragora.billing.budget_manager.BudgetPeriod", side_effect=lambda v: v),
        ):
            result = await handler.handle("/api/v1/budgets", "POST", mock_handler)
        assert _status(result) == 500
        assert "creation failed" in _body(result)["error"].lower()


# ============================================================================
# GET /api/v1/budgets/:id - Get Budget
# ============================================================================


class TestGetBudget:
    """Test getting a specific budget."""

    @pytest.mark.asyncio
    async def test_get_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["budget_id"] == "budget-001"

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/nonexistent", "GET", mock_handler)
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_get_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "GET", mock_handler)
        assert _status(result) == 403
        assert "Access denied" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_get_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle("/api/v1/budgets/budget-001", "GET", mock_handler)
        assert _status(result) == 500
        assert "Failed to retrieve budget" in _body(result)["error"]


# ============================================================================
# PATCH /api/v1/budgets/:id - Update Budget
# ============================================================================


class TestUpdateBudget:
    """Test budget updates."""

    @pytest.mark.asyncio
    async def test_update_name(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"name": "Updated Name"}, method="PATCH")
        with (
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
            patch("aragora.billing.budget_manager.BudgetStatus", side_effect=lambda v: v),
        ):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(body={"name": "Updated"}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(body={"name": "Updated"}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_update_empty_body(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body=None, method="PATCH")
        mock_handler.headers = {"Content-Length": "0"}
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        # read_json_body returns {} for CL=0, which is falsy? No, {} is truthy.
        # The handler checks "if not body" - empty dict is falsy in Python? No, {} is falsy.
        # Actually, {} is falsy. So it returns 400 "Invalid request body"
        # Wait no, {} is falsy in Python. Let me check: bool({}) is False.
        # But read_json_body returns {} for CL<=0. And "if not body" -> True for {}.
        # So this should return 400.
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_name_not_string(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"name": 123}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_name_empty_after_strip(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"name": "   "}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "empty" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_name_too_long(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"name": "x" * 201}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "maximum length" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_description_not_string(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"description": 99}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "description must be a string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_description_too_long(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"description": "x" * 2001}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_invalid_amount(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"amount_usd": "abc"}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "must be a number" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_amount_too_small(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"amount_usd": 0.001}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "at least" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_amount_too_large(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"amount_usd": 2_000_000_000}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "maximum" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_auto_suspend_not_bool(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"auto_suspend": "no"}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "boolean" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_invalid_status(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"status": "invalid_status"}, method="PATCH")
        with (
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
            patch(
                "aragora.billing.budget_manager.BudgetStatus",
                side_effect=ValueError("bad status"),
            ),
        ):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "Invalid status" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_status_not_string(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"status": 42}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 400
        assert "status must be a string" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_returns_none(self, handler, mock_manager, _patch_auth_and_rbac):
        """When update_budget returns None, should return 500."""
        mock_manager.update_budget.return_value = None
        mock_handler = _MockHTTPHandler(body={"name": "Updated"}, method="PATCH")
        with (
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
            patch("aragora.billing.budget_manager.BudgetStatus", side_effect=lambda v: v),
        ):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_update_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"name": "Updated"}, method="PATCH")
        with patch.object(handler, "_get_budget_manager", side_effect=RuntimeError("fail")):
            result = await handler.handle("/api/v1/budgets/budget-001", "PATCH", mock_handler)
        assert _status(result) == 500


# ============================================================================
# DELETE /api/v1/budgets/:id - Delete Budget
# ============================================================================


class TestDeleteBudget:
    """Test budget deletion."""

    @pytest.mark.asyncio
    async def test_delete_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "DELETE", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True
        assert body["budget_id"] == "budget-001"

    @pytest.mark.asyncio
    async def test_delete_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "DELETE", mock_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001", "DELETE", mock_handler)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_delete_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle("/api/v1/budgets/budget-001", "DELETE", mock_handler)
        assert _status(result) == 500
        assert "deletion failed" in _body(result)["error"].lower()


# ============================================================================
# GET /api/v1/budgets/summary - Budget Summary
# ============================================================================


class TestGetSummary:
    """Test budget summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_summary.return_value = {"total": 5000, "used": 1200}
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/summary", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5000

    @pytest.mark.asyncio
    async def test_summary_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", side_effect=RuntimeError("fail")):
            result = await handler.handle("/api/v1/budgets/summary", "GET", mock_handler)
        assert _status(result) == 500
        assert "summary" in _body(result)["error"].lower()


# ============================================================================
# POST /api/v1/budgets/check - Pre-flight Cost Check
# ============================================================================


class TestCheckBudget:
    """Test pre-flight cost check."""

    @pytest.mark.asyncio
    async def test_check_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"estimated_cost_usd": 10.0}, method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/check", "POST", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["allowed"] is True

    @pytest.mark.asyncio
    async def test_check_no_body(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body=None, method="POST")
        mock_handler.headers = {"Content-Length": "0"}
        result = await handler.handle("/api/v1/budgets/check", "POST", mock_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_check_zero_cost(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"estimated_cost_usd": 0}, method="POST")
        result = await handler.handle("/api/v1/budgets/check", "POST", mock_handler)
        assert _status(result) == 400
        assert "positive" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_check_negative_cost(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"estimated_cost_usd": -5}, method="POST")
        result = await handler.handle("/api/v1/budgets/check", "POST", mock_handler)
        assert _status(result) == 400
        assert "positive" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_check_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"estimated_cost_usd": 10}, method="POST")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle("/api/v1/budgets/check", "POST", mock_handler)
        assert _status(result) == 500
        assert "check failed" in _body(result)["error"].lower()


# ============================================================================
# GET /api/v1/budgets/:id/alerts - Get Alerts
# ============================================================================


class TestGetAlerts:
    """Test getting alerts for a budget."""

    @pytest.mark.asyncio
    async def test_alerts_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_alerts.return_value = [_MockAlert(), _MockAlert(alert_id="alert-002")]
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001/alerts", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert body["budget_id"] == "budget-001"

    @pytest.mark.asyncio
    async def test_alerts_budget_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001/alerts", "GET", mock_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_alerts_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001/alerts", "GET", mock_handler)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_alerts_empty(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/budget-001/alerts", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["alerts"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_alerts_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle("/api/v1/budgets/budget-001/alerts", "GET", mock_handler)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/budgets/:id/alerts/:alert_id/acknowledge - Acknowledge Alert
# ============================================================================


class TestAcknowledgeAlert:
    """Test acknowledging a budget alert."""

    @pytest.mark.asyncio
    async def test_acknowledge_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/alerts/alert-001/acknowledge",
                "POST",
                mock_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["acknowledged"] is True
        assert body["alert_id"] == "alert-001"

    @pytest.mark.asyncio
    async def test_acknowledge_no_user_id(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="POST")
        mock_handler.user_id = None
        # Also patch _get_user_id to return None
        with patch.object(handler, "_get_user_id", return_value=None):
            auth_ctx = _make_auth_ctx()
            decision = _make_permission_decision(allowed=True)
            checker = MagicMock()
            checker.check_permission.return_value = decision
            with (
                patch(
                    "aragora.billing.jwt_auth.extract_user_from_request",
                    return_value=auth_ctx,
                ),
                patch(
                    "aragora.rbac.checker.get_permission_checker",
                    return_value=checker,
                ),
            ):
                result = await handler.handle(
                    "/api/v1/budgets/budget-001/alerts/alert-001/acknowledge",
                    "POST",
                    mock_handler,
                )
        assert _status(result) == 400
        assert "User ID required" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_acknowledge_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/alerts/alert-001/acknowledge",
                "POST",
                mock_handler,
            )
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/budgets/:id/override - Add Override
# ============================================================================


class TestAddOverride:
    """Test adding a budget override."""

    @pytest.mark.asyncio
    async def test_override_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"user_id": "user-002", "duration_hours": 24},
            method="POST",
        )
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["override_added"] is True
        assert body["user_id"] == "user-002"

    @pytest.mark.asyncio
    async def test_override_budget_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(body={"user_id": "user-002"}, method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_override_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(body={"user_id": "user-002"}, method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_override_missing_user_id(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"duration_hours": 24}, method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 400
        assert "user_id" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_override_invalid_duration(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(
            body={"user_id": "user-002", "duration_hours": "not_a_number"},
            method="POST",
        )
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 400
        assert "duration_hours" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_override_no_body(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body=None, method="POST")
        mock_handler.headers = {"Content-Length": "0"}
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_override_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(body={"user_id": "user-002"}, method="POST")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_override_without_duration(self, handler, mock_manager, _patch_auth_and_rbac):
        """Override without duration_hours should pass None."""
        mock_handler = _MockHTTPHandler(body={"user_id": "user-002"}, method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override", "POST", mock_handler
            )
        assert _status(result) == 200
        mock_manager.add_override.assert_called_once_with(
            budget_id="budget-001",
            user_id="user-002",
            duration_hours=None,
        )


# ============================================================================
# DELETE /api/v1/budgets/:id/override/:user_id - Remove Override
# ============================================================================


class TestRemoveOverride:
    """Test removing a budget override."""

    @pytest.mark.asyncio
    async def test_remove_override_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override/user-002", "DELETE", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["override_removed"] is True
        assert body["user_id"] == "user-002"

    @pytest.mark.asyncio
    async def test_remove_override_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override/user-002", "DELETE", mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_remove_override_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override/user-002", "DELETE", mock_handler
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_remove_override_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="DELETE")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/override/user-002", "DELETE", mock_handler
            )
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/budgets/:id/reset - Reset Budget
# ============================================================================


class TestResetBudget:
    """Test resetting a budget period."""

    @pytest.mark.asyncio
    async def test_reset_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/reset", "POST", mock_handler
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reset_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/reset", "POST", mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_reset_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/reset", "POST", mock_handler
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_reset_returns_none(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.reset_period.return_value = None
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/reset", "POST", mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_reset_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="POST")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/reset", "POST", mock_handler
            )
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/budgets/:id/transactions - Get Transactions
# ============================================================================


class TestGetTransactions:
    """Test getting transaction history."""

    @pytest.mark.asyncio
    async def test_transactions_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_transactions.return_value = [_MockTransaction()]
        mock_manager.count_transactions.return_value = 1
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/transactions")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["budget_id"] == "budget-001"
        assert "pagination" in body

    @pytest.mark.asyncio
    async def test_transactions_with_query_params(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_transactions.return_value = []
        mock_manager.count_transactions.return_value = 0
        mock_handler = _MockHTTPHandler(
            path="/api/v1/budgets/budget-001/transactions?limit=10&offset=5"
        )
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["pagination"]["limit"] == 10
        assert body["pagination"]["offset"] == 5

    @pytest.mark.asyncio
    async def test_transactions_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/transactions")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_transactions_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/transactions")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_transactions_has_more(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_transactions.return_value = [_MockTransaction()] * 50
        mock_manager.count_transactions.return_value = 100
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/transactions")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_transactions_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/transactions")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_transactions_with_date_filters(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_transactions.return_value = []
        mock_manager.count_transactions.return_value = 0
        mock_handler = _MockHTTPHandler(
            path="/api/v1/budgets/budget-001/transactions?date_from=1000.0&date_to=2000.0"
        )
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_transactions_with_user_filter(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_transactions.return_value = []
        mock_manager.count_transactions.return_value = 0
        mock_handler = _MockHTTPHandler(
            path="/api/v1/budgets/budget-001/transactions?user_id=user-002"
        )
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/transactions", "GET", mock_handler
            )
        assert _status(result) == 200


# ============================================================================
# GET /api/v1/budgets/:id/trends - Budget Trends
# ============================================================================


class TestBudgetTrends:
    """Test spending trends for a budget."""

    @pytest.mark.asyncio
    async def test_trends_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_spending_trends.return_value = [{"day": "2026-01-01", "amount": 100}]
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["budget_id"] == "budget-001"
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_trends_with_period(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_spending_trends.return_value = []
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends?period=week")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 200
        assert _body(result)["period"] == "week"

    @pytest.mark.asyncio
    async def test_trends_invalid_period(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends?period=year")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 400
        assert "Invalid period" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_trends_not_found(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = None
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_trends_wrong_org(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_budget.return_value = _MockBudget(org_id="other-org")
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_trends_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/budget-001/trends")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle(
                "/api/v1/budgets/budget-001/trends", "GET", mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_trends_all_valid_periods(self, handler, mock_manager, _patch_auth_and_rbac):
        """All valid periods should be accepted."""
        for period in ("hour", "day", "week", "month"):
            mock_manager.get_spending_trends.return_value = []
            mock_handler = _MockHTTPHandler(
                path=f"/api/v1/budgets/budget-001/trends?period={period}"
            )
            with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
                result = await handler.handle(
                    "/api/v1/budgets/budget-001/trends", "GET", mock_handler
                )
            assert _status(result) == 200, f"Failed for period: {period}"


# ============================================================================
# GET /api/v1/budgets/trends - Org-wide Trends
# ============================================================================


class TestOrgTrends:
    """Test org-wide spending trends."""

    @pytest.mark.asyncio
    async def test_org_trends_success(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_manager.get_org_spending_trends.return_value = [{"day": "2026-01-01", "amount": 500}]
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/trends")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/trends", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert "org_id" in body

    @pytest.mark.asyncio
    async def test_org_trends_invalid_period(self, handler, mock_manager, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/trends?period=year")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/trends", "GET", mock_handler)
        assert _status(result) == 400
        assert "Invalid period" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_org_trends_manager_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/trends")
        with patch.object(handler, "_get_budget_manager", side_effect=ImportError("fail")):
            result = await handler.handle("/api/v1/budgets/trends", "GET", mock_handler)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/costs/agents - Agent Costs
# ============================================================================


class TestAgentCosts:
    """Test per-agent cost breakdown."""

    @pytest.mark.asyncio
    async def test_agent_costs_success(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "cost_by_agent": {"claude": 50.0, "gpt-4": 30.0},
            "total_cost_usd": "80.0",
        }
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/agents")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle("/api/v1/costs/agents", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["agents"]) == 2
        assert body["total_cost_usd"] == "80.0"

    @pytest.mark.asyncio
    async def test_agent_costs_with_workspace_id(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "cost_by_agent": {},
            "total_cost_usd": "0",
        }
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/agents?workspace_id=ws-001")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle("/api/v1/costs/agents", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"

    @pytest.mark.asyncio
    async def test_agent_costs_import_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/agents")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no module"),
        ):
            result = await handler.handle("/api/v1/costs/agents", "GET", mock_handler)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_agent_costs_runtime_error(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.side_effect = RuntimeError("db error")
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/agents")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            return_value=mock_tracker,
        ):
            result = await handler.handle("/api/v1/costs/agents", "GET", mock_handler)
        assert _status(result) == 500


# ============================================================================
# GET /api/v1/costs/anomalies - Cost Anomalies
# ============================================================================


class TestCostAnomalies:
    """Test cost anomaly detection."""

    @pytest.mark.asyncio
    async def test_anomalies_none_detected(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/anomalies")
        with (
            patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=([], None),
            ),
        ):
            result = await handler.handle("/api/v1/costs/anomalies", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["anomalies"] == []
        assert body["count"] == 0
        assert body["cost_advisory"] is None
        assert "No anomalies" in body["advisory"]

    @pytest.mark.asyncio
    async def test_anomalies_detected(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_advisory = MagicMock()
        mock_advisory.to_dict.return_value = {"severity": "high", "message": "Spike detected"}
        anomalies = [{"type": "spike", "agent": "claude"}]
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/anomalies")
        with (
            patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=(anomalies, mock_advisory),
            ),
        ):
            result = await handler.handle("/api/v1/costs/anomalies", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["cost_advisory"]["severity"] == "high"
        assert "anomalies detected" in body["advisory"]

    @pytest.mark.asyncio
    async def test_anomalies_import_error(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/anomalies")
        with patch(
            "aragora.billing.cost_tracker.get_cost_tracker",
            side_effect=ImportError("no module"),
        ):
            result = await handler.handle("/api/v1/costs/anomalies", "GET", mock_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_anomalies_with_workspace_id(self, handler, _patch_auth_and_rbac):
        mock_tracker = MagicMock()
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/anomalies?workspace_id=ws-002")
        with (
            patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.server.http_utils.run_async",
                return_value=([], None),
            ),
        ):
            result = await handler.handle("/api/v1/costs/anomalies", "GET", mock_handler)
        assert _status(result) == 200
        assert _body(result)["workspace_id"] == "ws-002"

    @pytest.mark.asyncio
    async def test_anomalies_runtime_error_in_detect(self, handler, _patch_auth_and_rbac):
        """When run_async raises RuntimeError, should return empty anomalies."""
        mock_tracker = MagicMock()
        mock_handler = _MockHTTPHandler(path="/api/v1/costs/anomalies")
        with (
            patch(
                "aragora.billing.cost_tracker.get_cost_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "aragora.server.http_utils.run_async",
                side_effect=RuntimeError("event loop error"),
            ),
        ):
            result = await handler.handle("/api/v1/costs/anomalies", "GET", mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["anomalies"] == []
        assert body["cost_advisory"] is None


# ============================================================================
# Not Found / Method Routing
# ============================================================================


class TestNotFound:
    """Test fallback for unknown routes."""

    @pytest.mark.asyncio
    async def test_unknown_path(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler()
        result = await handler.handle("/api/v1/budgets/foo/bar/baz/qux/extra", "GET", mock_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unsupported_method_on_root(self, handler, _patch_auth_and_rbac):
        mock_handler = _MockHTTPHandler(method="PUT")
        result = await handler.handle("/api/v1/budgets", "PUT", mock_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_trailing_slash(self, handler, mock_manager, _patch_auth_and_rbac):
        """Trailing slashes should be stripped before matching."""
        mock_handler = _MockHTTPHandler(path="/api/v1/budgets/")
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/", "GET", mock_handler)
        # After rstrip("/"), path becomes "/api/v1/budgets" which matches list
        assert _status(result) == 200


# ============================================================================
# Helper Methods
# ============================================================================


class TestHelperMethods:
    """Test internal helper methods."""

    def test_get_org_id_from_handler(self, handler):
        mock_handler = _MockHTTPHandler(org_id="org-123")
        assert handler._get_org_id(mock_handler) == "org-123"

    def test_get_org_id_none_handler(self, handler):
        result = handler._get_org_id(None)
        # Falls back to extract_user_from_request or "default"
        assert isinstance(result, str)

    def test_get_user_id_from_handler(self, handler):
        mock_handler = _MockHTTPHandler(user_id="user-123")
        assert handler._get_user_id(mock_handler) == "user-123"

    def test_get_user_id_none_handler(self, handler):
        result = handler._get_user_id(None)
        # Returns None when handler is None
        assert result is None

    def test_get_budget_manager_import_error(self, handler):
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=ImportError("no module"),
        ):
            with pytest.raises(ImportError):
                handler._get_budget_manager()

    def test_get_budget_manager_success(self, handler):
        mock_mgr = MagicMock()
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=mock_mgr,
        ):
            result = handler._get_budget_manager()
        assert result is mock_mgr

    def test_get_budget_manager_records_failure(self, handler):
        cb = handler._circuit_breaker
        initial_failures = cb._failure_count
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            side_effect=RuntimeError("error"),
        ):
            with pytest.raises(RuntimeError):
                handler._get_budget_manager()
        assert cb._failure_count > initial_failures

    def test_get_budget_manager_records_success(self, handler):
        mock_mgr = MagicMock()
        with patch(
            "aragora.billing.budget_manager.get_budget_manager",
            return_value=mock_mgr,
        ):
            handler._get_budget_manager()
        # After success, failure count should be 0
        assert handler._circuit_breaker._failure_count == 0

    def test_circuit_breaker_status_structure(self, handler):
        status = handler.get_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status


# ============================================================================
# Method Extraction
# ============================================================================


class TestMethodExtraction:
    """Test method extraction from query_params / handler."""

    @pytest.mark.asyncio
    async def test_method_from_string_query_params(self, handler, mock_manager, _patch_auth_and_rbac):
        """When query_params is a string, use it as the method."""
        mock_handler = _MockHTTPHandler()
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/summary", "GET", mock_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_method_from_handler_command(self, handler, mock_manager, _patch_auth_and_rbac):
        """Method should be extracted from handler.command when query_params is dict."""
        mock_handler = _MockHTTPHandler()
        mock_handler.command = "GET"
        with patch.object(handler, "_get_budget_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/budgets/summary", {}, mock_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_method_default_get(self, handler, mock_manager, _patch_auth_and_rbac):
        """Default method should be GET when no handler."""
        mock_handler = None
        auth_ctx = _make_auth_ctx()
        # With handler=None, extract_user_from_request may fail, so we need careful patching
        with (
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=auth_ctx,
            ),
            patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=MagicMock(
                    check_permission=MagicMock(return_value=_make_permission_decision(True))
                ),
            ),
            patch.object(handler, "_get_budget_manager", return_value=mock_manager),
        ):
            result = await handler.handle("/api/v1/budgets/summary", {}, None)
        assert _status(result) == 200


# ============================================================================
# Factory Function
# ============================================================================


class TestFactory:
    """Test handler factory function."""

    def test_create_budget_handler(self):
        from aragora.server.handlers.budgets import create_budget_handler

        ctx = {"server": "context"}
        h = create_budget_handler(ctx)
        assert isinstance(h, BudgetHandler)
        assert h.ctx == ctx

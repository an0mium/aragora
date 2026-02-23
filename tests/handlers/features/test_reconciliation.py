"""Tests for bank reconciliation handler.

Tests the reconciliation API endpoints including:
- GET  /api/v1/reconciliation/status           - Circuit breaker status
- POST /api/v1/reconciliation/run              - Run new reconciliation
- GET  /api/v1/reconciliation/list             - List past reconciliations
- GET  /api/v1/reconciliation/{id}             - Get reconciliation details
- GET  /api/v1/reconciliation/{id}/report      - Generate report (JSON/CSV)
- POST /api/v1/reconciliation/{id}/resolve     - Resolve a discrepancy
- POST /api/v1/reconciliation/{id}/approve     - Approve reconciliation
- GET  /api/v1/reconciliation/discrepancies    - Get pending discrepancies
- POST /api/v1/reconciliation/discrepancies/bulk-resolve - Bulk resolve
- GET  /api/v1/reconciliation/demo             - Demo data
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock HTTP request for testing the reconciliation handler."""

    path: str = "/api/v1/reconciliation/status"
    method: str = "GET"
    query: dict[str, Any] = field(default_factory=dict)
    _body: dict[str, Any] | None = None
    tenant_id: str = "default"
    user_id: str = "test-user-001"

    async def json(self) -> dict[str, Any]:
        return self._body or {}


class ResolutionStatus(Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class DiscrepancyType(Enum):
    MISSING_BANK = "missing_bank"
    MISSING_BOOK = "missing_book"
    AMOUNT_MISMATCH = "amount_mismatch"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MockDiscrepancy:
    """Mock discrepancy for reconciliation results."""

    discrepancy_id: str = "disc_001"
    discrepancy_type: DiscrepancyType = DiscrepancyType.AMOUNT_MISMATCH
    description: str = "Amount mismatch on payment"
    bank_amount: float = 100.0
    book_amount: float = 95.0
    bank_date: str = "2024-01-15"
    book_date: str = "2024-01-15"
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    severity: Severity = Severity.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "discrepancy_id": self.discrepancy_id,
            "discrepancy_type": self.discrepancy_type.value,
            "description": self.description,
            "bank_amount": self.bank_amount,
            "book_amount": self.book_amount,
            "bank_date": self.bank_date,
            "book_date": self.book_date,
            "resolution_status": self.resolution_status.value,
            "severity": self.severity.value,
        }


@dataclass
class MockMatchedTransaction:
    """Mock matched transaction."""

    transaction_id: str = "txn_001"
    amount: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {"transaction_id": self.transaction_id, "amount": self.amount}


@dataclass
class MockReconciliationResult:
    """Mock reconciliation result."""

    reconciliation_id: str = "rec_001"
    account_name: str = "Business Checking"
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    end_date: date = field(default_factory=lambda: date(2024, 1, 31))
    bank_total: float = 10000.0
    book_total: float = 9500.0
    difference: float = 500.0
    matched_count: int = 42
    discrepancy_count: int = 2
    match_rate: float = 0.95
    is_reconciled: bool = False
    reconciled_at: datetime | None = None
    reconciled_by: str | None = None
    discrepancies: list[MockDiscrepancy] = field(default_factory=lambda: [MockDiscrepancy()])
    matched_transactions: list[MockMatchedTransaction] = field(
        default_factory=lambda: [MockMatchedTransaction()]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "account_name": self.account_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "bank_total": self.bank_total,
            "book_total": self.book_total,
            "difference": self.difference,
            "matched_count": self.matched_count,
            "discrepancy_count": self.discrepancy_count,
            "match_rate": self.match_rate,
            "is_reconciled": self.is_reconciled,
        }


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult."""
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


def _data(result) -> dict[str, Any]:
    """Extract the 'data' field from a success response."""
    body = _body(result)
    return body.get("data", {})


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    body = _body(result)
    return body.get("error", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ReconciliationHandler instance with empty context."""
    from aragora.server.handlers.features.reconciliation import (
        ReconciliationHandler,
        _clear_reconciliation_circuit_breaker,
    )

    _clear_reconciliation_circuit_breaker()
    h = ReconciliationHandler(server_context={})
    return h


@pytest.fixture(autouse=True)
def reset_reconciliation_state():
    """Reset global state before/after each test."""
    from aragora.server.handlers.features.reconciliation import (
        _clear_reconciliation_circuit_breaker,
        _service_instances,
    )

    _clear_reconciliation_circuit_breaker()
    _service_instances.clear()
    yield
    _clear_reconciliation_circuit_breaker()
    _service_instances.clear()


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def mock_service():
    """Create a mock ReconciliationService."""
    service = MagicMock()
    service.list_reconciliations = MagicMock(return_value=[MockReconciliationResult()])
    service.get_reconciliation = MagicMock(return_value=MockReconciliationResult())
    service.resolve_discrepancy = AsyncMock(return_value=True)
    service.reconcile = AsyncMock(return_value=MockReconciliationResult())
    return service


@pytest.fixture
def mock_service_patched(mock_service):
    """Patch get_reconciliation_service to return our mock."""
    with patch(
        "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
        return_value=mock_service,
    ):
        yield mock_service


# ---------------------------------------------------------------------------
# can_handle() Routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle routing for all reconciliation paths."""

    def test_status_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/status")

    def test_run_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/run", "POST")

    def test_list_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/list")

    def test_detail_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/rec_001")

    def test_report_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/rec_001/report")

    def test_resolve_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/rec_001/resolve", "POST")

    def test_approve_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/rec_001/approve", "POST")

    def test_discrepancies_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/discrepancies")

    def test_bulk_resolve_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/discrepancies/bulk-resolve", "POST")

    def test_demo_path(self, handler):
        assert handler.can_handle("/api/v1/reconciliation/demo")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/ecommerce/platforms")

    def test_unrelated_path_partial(self, handler):
        assert not handler.can_handle("/api/v1/billing/invoices")


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/status
# ---------------------------------------------------------------------------


class TestStatus:
    """Test circuit breaker status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_healthy(self, handler):
        req = MockRequest(path="/api/v1/reconciliation/status", method="GET")
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["status"] == "healthy"
        assert "circuit_breaker" in data

    @pytest.mark.asyncio
    async def test_status_circuit_breaker_fields(self, handler):
        req = MockRequest(path="/api/v1/reconciliation/status", method="GET")
        result = await handler.handle(req, req.path, req.method)
        cb = _data(result)["circuit_breaker"]
        assert "state" in cb
        assert "failure_count" in cb
        assert cb["state"] == "closed"


# ---------------------------------------------------------------------------
# POST /api/v1/reconciliation/run (no plaid token -> demo)
# ---------------------------------------------------------------------------


class TestRunReconciliationDemo:
    """Test running reconciliation without plaid token (returns demo data)."""

    @pytest.mark.asyncio
    async def test_run_demo_success(self, handler, mock_service_patched):
        mock_mod = MagicMock()
        mock_mod.get_mock_reconciliation_result.return_value = MockReconciliationResult()
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": mock_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200
            data = _data(result)
            assert data["is_demo"] is True
            assert "reconciliation" in data
            assert "discrepancies" in data

    @pytest.mark.asyncio
    async def test_run_missing_dates(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "start_date" in _error(result)

    @pytest.mark.asyncio
    async def test_run_missing_start_date(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"end_date": "2024-01-31"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_missing_end_date(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "2024-01-01"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_invalid_date_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "not-a-date", "end_date": "2024-01-31"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "date format" in _error(result).lower() or "date" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_run_end_before_start(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "2024-02-01", "end_date": "2024-01-01"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "end_date" in _error(result)

    @pytest.mark.asyncio
    async def test_run_start_date_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "2024-01-01-extra", "end_date": "2024-01-31"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_end_date_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "2024-01-01", "end_date": "2024-01-31-extra"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_start_date_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": 12345, "end_date": "2024-01-31"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_invalid_account_id_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "account_id": "x" * 65,
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_invalid_account_id_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "account_id": 12345,
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_use_agents_not_bool(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "use_agents": "yes",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "use_agents" in _error(result)

    @pytest.mark.asyncio
    async def test_run_valid_account_id(self, handler, mock_service_patched):
        mock_mod = MagicMock()
        mock_mod.get_mock_reconciliation_result.return_value = MockReconciliationResult()
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": mock_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "account_id": "acct_001",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_run_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={"start_date": "2024-01-01", "end_date": "2024-01-31"},
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# POST /api/v1/reconciliation/run (with plaid token -> real reconciliation)
# ---------------------------------------------------------------------------


class TestRunReconciliationWithPlaid:
    """Test running reconciliation with plaid token."""

    @pytest.mark.asyncio
    async def test_run_with_plaid_success(self, handler, mock_service_patched):
        mock_result = MockReconciliationResult()
        mock_service_patched.reconcile = AsyncMock(return_value=mock_result)

        mock_plaid_mod = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.connectors.accounting.plaid": mock_plaid_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "plaid_access_token": "access-sandbox-abc123",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200
            data = _data(result)
            assert "reconciliation" in data
            assert "matched_count" in data

    @pytest.mark.asyncio
    async def test_run_with_plaid_token_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "plaid_access_token": "x" * 257,
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_with_plaid_token_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "plaid_access_token": 12345,
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_run_service_error_records_failure(self, handler, mock_service_patched):
        mock_service_patched.reconcile = AsyncMock(side_effect=ValueError("Bad data"))

        mock_plaid_mod = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.connectors.accounting.plaid": mock_plaid_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "plaid_access_token": "access-sandbox-abc123",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_run_with_institution_info(self, handler, mock_service_patched):
        mock_result = MockReconciliationResult()
        mock_service_patched.reconcile = AsyncMock(return_value=mock_result)

        mock_plaid_mod = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.connectors.accounting.plaid": mock_plaid_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "plaid_access_token": "access-sandbox-abc123",
                    "institution_id": "ins_12345",
                    "institution_name": "Chase",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Circuit Breaker for Run
# ---------------------------------------------------------------------------


class TestRunCircuitBreaker:
    """Test circuit breaker behavior during reconciliation run."""

    @pytest.mark.asyncio
    async def test_run_circuit_breaker_open(self, handler, mock_service_patched):
        # Force circuit breaker open
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None  # Prevent transition

        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="POST",
            _body={"start_date": "2024-01-01", "end_date": "2024-01-31"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 503
        assert "circuit breaker" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_run_import_error_records_failure(self, handler, mock_service_patched):
        # Simulate ImportError when importing the reconciliation module
        # Remove the module from sys.modules so the local import fails
        import sys

        saved = sys.modules.pop("aragora.services.accounting.reconciliation", None)
        # Make the import raise ImportError by inserting a broken module
        broken_mod = MagicMock()
        broken_mod.get_mock_reconciliation_result = property(
            lambda self: (_ for _ in ()).throw(ImportError("No module"))
        )

        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": None},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/run",
                method="POST",
                _body={"start_date": "2024-01-01", "end_date": "2024-01-31"},
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503

        if saved is not None:
            sys.modules["aragora.services.accounting.reconciliation"] = saved


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/list
# ---------------------------------------------------------------------------


class TestListReconciliations:
    """Test listing past reconciliations."""

    @pytest.mark.asyncio
    async def test_list_success(self, handler, mock_service_patched):
        req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert "reconciliations" in data
        assert "total" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_list_with_account_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"account_id": "acct_001"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once_with(
            account_id="acct_001",
            limit=20,
        )

    @pytest.mark.asyncio
    async def test_list_with_limit(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"limit": "5"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once_with(
            account_id=None,
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_list_limit_clamped_to_max(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"limit": "500"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once_with(
            account_id=None,
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_list_limit_clamped_to_min(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"limit": "0"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once_with(
            account_id=None,
            limit=1,
        )

    @pytest.mark.asyncio
    async def test_list_invalid_limit_defaults(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"limit": "abc"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once_with(
            account_id=None,
            limit=20,
        )

    @pytest.mark.asyncio
    async def test_list_invalid_account_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            query={"account_id": "../etc/passwd"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_no_service_returns_empty(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200
            data = _data(result)
            assert data["reconciliations"] == []
            assert data["total"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/{id}
# ---------------------------------------------------------------------------


class TestGetReconciliation:
    """Test getting reconciliation details."""

    @pytest.mark.asyncio
    async def test_get_success(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert "reconciliation" in data
        assert "discrepancies" in data
        assert "matched_transactions" in data

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.return_value = None
        req = MockRequest(
            path="/api/v1/reconciliation/rec_999",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_invalid_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/../etc/passwd",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        # The path splitting makes ".." the id at index 4, which fails validation
        assert _status(result) in (400, 404)

    @pytest.mark.asyncio
    async def test_get_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/rec_001",
                method="GET",
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/{id}/report
# ---------------------------------------------------------------------------


class TestReport:
    """Test report generation endpoint."""

    @pytest.mark.asyncio
    async def test_report_json_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
            query={"format": "json"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        report = data["report"]
        assert "title" in report
        assert "period" in report
        assert "summary" in report
        assert "discrepancies" in report
        assert "generated_at" in report

    @pytest.mark.asyncio
    async def test_report_json_default_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert "report" in data

    @pytest.mark.asyncio
    async def test_report_json_summary_fields(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        summary = _data(result)["report"]["summary"]
        assert "bank_balance" in summary
        assert "book_balance" in summary
        assert "difference" in summary
        assert "transactions_matched" in summary
        assert "discrepancies_found" in summary
        assert "match_rate" in summary
        assert "is_reconciled" in summary

    @pytest.mark.asyncio
    async def test_report_csv_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
            query={"format": "csv"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        assert result.content_type == "text/csv"
        csv_content = result.body.decode("utf-8")
        assert "Type,Description,Bank Amount,Book Amount" in csv_content
        assert result.headers["Content-Disposition"].startswith("attachment;")

    @pytest.mark.asyncio
    async def test_report_unsupported_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
            query={"format": "pdf"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "Unsupported format" in _error(result)

    @pytest.mark.asyncio
    async def test_report_not_found(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.return_value = None
        req = MockRequest(
            path="/api/v1/reconciliation/rec_999/report",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_report_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/rec_001/report",
                method="GET",
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_report_csv_filename_contains_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report",
            method="GET",
            query={"format": "csv"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert "rec_001" in result.headers["Content-Disposition"]


# ---------------------------------------------------------------------------
# POST /api/v1/reconciliation/{id}/resolve
# ---------------------------------------------------------------------------


class TestResolve:
    """Test discrepancy resolution endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_success(self, handler, mock_service_patched):
        # Return a result where discrepancy is resolved
        resolved_result = MockReconciliationResult(
            discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
        )
        mock_service_patched.get_reconciliation.return_value = resolved_result
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Created expense entry",
                "action": "create_entry",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["status"] == "resolved"
        assert data["discrepancy_id"] == "disc_001"

    @pytest.mark.asyncio
    async def test_resolve_missing_discrepancy_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={"resolution": "Fixed", "action": "resolve"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "discrepancy_id" in _error(result)

    @pytest.mark.asyncio
    async def test_resolve_invalid_discrepancy_id_format(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "x" * 65,
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resolve_discrepancy_id_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": 12345,
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resolve_resolution_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "x" * 1001,
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "1000" in _error(result)

    @pytest.mark.asyncio
    async def test_resolve_resolution_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": 12345,
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resolve_invalid_action(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Fixed",
                "action": "invalid_action",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "Invalid action" in _error(result)

    @pytest.mark.asyncio
    async def test_resolve_valid_actions(self, handler, mock_service_patched):
        """Test all valid action types."""
        for action in ["create_entry", "ignore", "match_manual", "resolve"]:
            mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)
            resolved_result = MockReconciliationResult(
                discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
            )
            mock_service_patched.get_reconciliation.return_value = resolved_result

            req = MockRequest(
                path="/api/v1/reconciliation/rec_001/resolve",
                method="POST",
                _body={
                    "discrepancy_id": "disc_001",
                    "resolution": "Fixed",
                    "action": action,
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200, f"Action {action} failed"

    @pytest.mark.asyncio
    async def test_resolve_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/rec_001/resolve",
                method="POST",
                _body={
                    "discrepancy_id": "disc_001",
                    "resolution": "Fixed",
                    "action": "resolve",
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_resolve_failure(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=False)

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Fixed",
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resolve_service_error(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(
            side_effect=ValueError("Service error")
        )

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Fixed",
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_resolve_pending_discrepancy_count(self, handler, mock_service_patched):
        # After resolve, 1 pending discrepancy remains
        pending_result = MockReconciliationResult(
            discrepancies=[
                MockDiscrepancy(
                    discrepancy_id="disc_002",
                    resolution_status=ResolutionStatus.PENDING,
                ),
            ],
        )
        mock_service_patched.get_reconciliation.return_value = pending_result
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Fixed",
                "action": "resolve",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert data["reconciliation_status"]["pending_discrepancies"] == 1

    @pytest.mark.asyncio
    async def test_resolve_default_action(self, handler, mock_service_patched):
        """If action is not provided, defaults to 'resolve'."""
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)
        resolved_result = MockReconciliationResult(
            discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
        )
        mock_service_patched.get_reconciliation.return_value = resolved_result

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="POST",
            _body={
                "discrepancy_id": "disc_001",
                "resolution": "Resolved it",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/v1/reconciliation/{id}/approve
# ---------------------------------------------------------------------------


class TestApprove:
    """Test reconciliation approval endpoint."""

    @pytest.mark.asyncio
    async def test_approve_success(self, handler, mock_service_patched):
        # No pending discrepancies
        mock_service_patched.get_reconciliation.return_value = MockReconciliationResult(
            discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
        )

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={"notes": "Reviewed and approved"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["status"] == "approved"
        assert data["reconciliation_id"] == "rec_001"
        assert "approved_by" in data
        assert "approved_at" in data

    @pytest.mark.asyncio
    async def test_approve_with_pending_discrepancies(self, handler, mock_service_patched):
        # Has pending discrepancies - should fail
        mock_service_patched.get_reconciliation.return_value = MockReconciliationResult(
            discrepancies=[
                MockDiscrepancy(resolution_status=ResolutionStatus.PENDING),
                MockDiscrepancy(
                    discrepancy_id="disc_002",
                    resolution_status=ResolutionStatus.PENDING,
                ),
            ],
        )

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={"notes": "Trying to approve"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "unresolved" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_approve_not_found(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.return_value = None

        req = MockRequest(
            path="/api/v1/reconciliation/rec_999/approve",
            method="POST",
            _body={"notes": "Approve"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_approve_notes_too_long(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={"notes": "x" * 501},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "500" in _error(result)

    @pytest.mark.asyncio
    async def test_approve_notes_not_string(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={"notes": 12345},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_approve_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/rec_001/approve",
                method="POST",
                _body={"notes": "Approve"},
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_approve_service_error(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.side_effect = ValueError("DB error")

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={"notes": "Approve"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_approve_default_empty_notes(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.return_value = MockReconciliationResult(
            discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
        )

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_approve_sets_user_id(self, handler, mock_service_patched):
        mock_service_patched.get_reconciliation.return_value = MockReconciliationResult(
            discrepancies=[MockDiscrepancy(resolution_status=ResolutionStatus.RESOLVED)],
        )

        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="POST",
            _body={},
            user_id="admin-user",
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert data["approved_by"] == "admin-user"


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/discrepancies
# ---------------------------------------------------------------------------


class TestDiscrepancies:
    """Test discrepancy listing endpoint."""

    @pytest.mark.asyncio
    async def test_discrepancies_success(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert "discrepancies" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_discrepancies_filtered_by_status(self, handler, mock_service_patched):
        # The mock has a pending discrepancy by default
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
            query={"status": "pending"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_discrepancies_filtered_by_severity(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
            query={"severity": "medium"},
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_discrepancies_no_service_returns_empty(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/discrepancies",
                method="GET",
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200
            data = _data(result)
            assert data["discrepancies"] == []
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_discrepancies_sorted_by_severity(self, handler, mock_service_patched):
        # Return multiple reconciliations with different severity discrepancies
        result1 = MockReconciliationResult(
            reconciliation_id="rec_001",
            discrepancies=[MockDiscrepancy(severity=Severity.LOW)],
        )
        result2 = MockReconciliationResult(
            reconciliation_id="rec_002",
            discrepancies=[MockDiscrepancy(severity=Severity.CRITICAL, discrepancy_id="disc_002")],
        )
        mock_service_patched.list_reconciliations.return_value = [result1, result2]

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        if len(data["discrepancies"]) >= 2:
            # Critical should come before low
            assert data["discrepancies"][0]["severity"] == "critical"
            assert data["discrepancies"][1]["severity"] == "low"

    @pytest.mark.asyncio
    async def test_discrepancies_include_reconciliation_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        if data["discrepancies"]:
            disc = data["discrepancies"][0]
            assert "reconciliation_id" in disc
            assert "account_name" in disc
            assert "period" in disc

    @pytest.mark.asyncio
    async def test_discrepancies_with_limit(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
            query={"limit": "1"},
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert len(data["discrepancies"]) <= 1

    @pytest.mark.asyncio
    async def test_discrepancies_filter_excludes_non_matching(self, handler, mock_service_patched):
        """Filtering by status should exclude non-matching discrepancies."""
        recon = MockReconciliationResult(
            discrepancies=[
                MockDiscrepancy(
                    discrepancy_id="disc_resolved",
                    resolution_status=ResolutionStatus.RESOLVED,
                ),
            ],
        )
        mock_service_patched.list_reconciliations.return_value = [recon]

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
            query={"status": "pending"},
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_discrepancies_severity_filter_excludes_non_matching(
        self, handler, mock_service_patched
    ):
        recon = MockReconciliationResult(
            discrepancies=[
                MockDiscrepancy(severity=Severity.LOW),
            ],
        )
        mock_service_patched.list_reconciliations.return_value = [recon]

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies",
            method="GET",
            query={"severity": "critical"},
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# POST /api/v1/reconciliation/discrepancies/bulk-resolve
# ---------------------------------------------------------------------------


class TestBulkResolve:
    """Test bulk discrepancy resolution endpoint."""

    @pytest.mark.asyncio
    async def test_bulk_resolve_success(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {
                        "discrepancy_id": "disc_001",
                        "resolution": "Created entry",
                        "action": "create_entry",
                    },
                    {
                        "discrepancy_id": "disc_002",
                        "resolution": "Ignored",
                        "action": "ignore",
                    },
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["resolved_count"] == 2
        assert data["error_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_resolve_missing_reconciliation_id(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "resolutions": [
                    {"discrepancy_id": "disc_001", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "reconciliation_id" in _error(result)

    @pytest.mark.asyncio
    async def test_bulk_resolve_invalid_reconciliation_id_too_long(
        self, handler, mock_service_patched
    ):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "x" * 65,
                "resolutions": [],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_resolve_invalid_reconciliation_id_not_string(
        self, handler, mock_service_patched
    ):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": 12345,
                "resolutions": [],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_resolve_resolutions_not_array(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": "not-an-array",
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "array" in _error(result)

    @pytest.mark.asyncio
    async def test_bulk_resolve_too_many_resolutions(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"discrepancy_id": f"disc_{i}", "action": "resolve"} for i in range(101)
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 400
        assert "100" in _error(result)

    @pytest.mark.asyncio
    async def test_bulk_resolve_service_not_available(self, handler):
        with patch(
            "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
            return_value=None,
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/discrepancies/bulk-resolve",
                method="POST",
                _body={
                    "reconciliation_id": "rec_001",
                    "resolutions": [],
                },
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_bulk_resolve_partial_failures(self, handler, mock_service_patched):
        # First resolution succeeds, second fails
        mock_service_patched.resolve_discrepancy = AsyncMock(side_effect=[True, False])

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "action": "resolve"},
                    {"discrepancy_id": "disc_002", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["resolved_count"] == 1
        assert data["error_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_resolve_invalid_item_skipped(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    "not-a-dict",
                    {"discrepancy_id": "disc_001", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        # The non-dict item is skipped, only the valid one counts
        assert data["resolved_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_resolve_invalid_action_in_item(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "action": "bad_action"},
                    {"discrepancy_id": "disc_002", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["resolved_count"] == 1
        assert data["error_count"] == 1
        assert data["errors"][0]["error"] == "Invalid action"

    @pytest.mark.asyncio
    async def test_bulk_resolve_missing_discrepancy_id_in_item(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"resolution": "Fixed", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["error_count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_resolve_empty_resolutions(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        data = _data(result)
        assert data["resolved_count"] == 0
        assert data["error_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_resolve_service_error(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(side_effect=ValueError("DB error"))

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_bulk_resolve_reconciliation_status_returned(self, handler, mock_service_patched):
        mock_service_patched.resolve_discrepancy = AsyncMock(return_value=True)

        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="POST",
            _body={
                "reconciliation_id": "rec_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "action": "resolve"},
                ],
            },
        )
        result = await handler.handle(req, req.path, req.method)
        data = _data(result)
        assert "reconciliation_status" in data
        assert "is_reconciled" in data["reconciliation_status"]


# ---------------------------------------------------------------------------
# GET /api/v1/reconciliation/demo
# ---------------------------------------------------------------------------


class TestDemo:
    """Test demo data endpoint."""

    @pytest.mark.asyncio
    async def test_demo_success(self, handler):
        mock_mod = MagicMock()
        mock_mod.get_mock_reconciliation_result.return_value = MockReconciliationResult()
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": mock_mod},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/demo",
                method="GET",
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 200
            data = _data(result)
            assert data["is_demo"] is True
            assert "reconciliation" in data
            assert "discrepancies" in data
            assert "matched_transactions" in data

    @pytest.mark.asyncio
    async def test_demo_import_error(self, handler):
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": None},
        ):
            req = MockRequest(
                path="/api/v1/reconciliation/demo",
                method="GET",
            )
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# 404 / Not Found
# ---------------------------------------------------------------------------


class TestNotFound:
    """Test that unmatched routes return 404."""

    @pytest.mark.asyncio
    async def test_unknown_path(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/unknown-action",
            method="POST",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_run(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/run",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_list(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="POST",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_resolve(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/resolve",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_approve(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/approve",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_status(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/status",
            method="POST",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_demo(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/demo",
            method="POST",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_bulk_resolve(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/discrepancies/bulk-resolve",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_too_many_path_segments(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/report/extra/stuff",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_action_on_reconciliation(self, handler):
        req = MockRequest(
            path="/api/v1/reconciliation/rec_001/unknown",
            method="GET",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Tenant Extraction
# ---------------------------------------------------------------------------


class TestTenantExtraction:
    """Test tenant ID extraction from request."""

    @pytest.mark.asyncio
    async def test_default_tenant(self, handler, mock_service_patched):
        req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
        # Remove tenant_id attribute
        del req.tenant_id
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_custom_tenant(self, handler, mock_service_patched):
        req = MockRequest(
            path="/api/v1/reconciliation/list",
            method="GET",
            tenant_id="tenant_abc",
        )
        result = await handler.handle(req, req.path, req.method)
        assert _status(result) == 200
        mock_service_patched.list_reconciliations.assert_called_once()


# ---------------------------------------------------------------------------
# Service Instance Management
# ---------------------------------------------------------------------------


class TestServiceManagement:
    """Test service instance management."""

    def test_get_service_import_error(self):
        from aragora.server.handlers.features.reconciliation import (
            get_reconciliation_service,
            _service_instances,
        )

        _service_instances.clear()

        # Set module to None in sys.modules to trigger ImportError on local import
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": None},
        ):
            svc = get_reconciliation_service("test-tenant")
            assert svc is None

    def test_get_service_caches_instance(self):
        from aragora.server.handlers.features.reconciliation import (
            get_reconciliation_service,
            _service_instances,
        )

        _service_instances.clear()
        mock_svc = MagicMock()

        mock_mod = MagicMock()
        mock_mod.ReconciliationService.return_value = mock_svc
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": mock_mod},
        ):
            svc1 = get_reconciliation_service("tenant-a")
            svc2 = get_reconciliation_service("tenant-a")
            assert svc1 is svc2

    def test_get_service_different_tenants(self):
        from aragora.server.handlers.features.reconciliation import (
            get_reconciliation_service,
            _service_instances,
        )

        _service_instances.clear()

        mock_svc_a = MagicMock()
        mock_svc_b = MagicMock()
        mock_mod = MagicMock()
        mock_mod.ReconciliationService.side_effect = [mock_svc_a, mock_svc_b]
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": mock_mod},
        ):
            svc1 = get_reconciliation_service("tenant-a")
            svc2 = get_reconciliation_service("tenant-b")
            assert svc1 is not svc2


# ---------------------------------------------------------------------------
# Circuit Breaker Management
# ---------------------------------------------------------------------------


class TestCircuitBreakerManagement:
    """Test circuit breaker utility functions."""

    def test_get_status(self):
        from aragora.server.handlers.features.reconciliation import (
            get_reconciliation_circuit_breaker_status,
        )

        status = get_reconciliation_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"

    def test_clear_circuit_breaker(self):
        from aragora.server.handlers.features.reconciliation import (
            _clear_reconciliation_circuit_breaker,
            _get_reconciliation_circuit_breaker,
        )

        cb1 = _get_reconciliation_circuit_breaker()
        _clear_reconciliation_circuit_breaker()
        cb2 = _get_reconciliation_circuit_breaker()
        assert cb1 is not cb2


# ---------------------------------------------------------------------------
# Handler Registration
# ---------------------------------------------------------------------------


class TestHandlerRegistration:
    """Test handler registration and entry point."""

    def test_get_handler_singleton(self):
        from aragora.server.handlers.features.reconciliation import (
            get_reconciliation_handler,
            _handler_instance,
        )

        import aragora.server.handlers.features.reconciliation as mod

        mod._handler_instance = None
        h1 = get_reconciliation_handler()
        h2 = get_reconciliation_handler()
        assert h1 is h2
        mod._handler_instance = None

    @pytest.mark.asyncio
    async def test_handle_reconciliation_entry_point(self):
        from aragora.server.handlers.features.reconciliation import (
            handle_reconciliation,
        )

        import aragora.server.handlers.features.reconciliation as mod

        mod._handler_instance = None

        req = MockRequest(path="/api/v1/reconciliation/status", method="GET")
        result = await handle_reconciliation(req, req.path, req.method)
        assert _status(result) == 200

        mod._handler_instance = None

    def test_handler_init_default_context(self):
        from aragora.server.handlers.features.reconciliation import ReconciliationHandler

        h = ReconciliationHandler()
        assert h.ctx == {}

    def test_handler_init_with_context(self):
        from aragora.server.handlers.features.reconciliation import ReconciliationHandler

        ctx = {"db": "mock_db"}
        h = ReconciliationHandler(server_context=ctx)
        assert h.ctx == ctx


# ---------------------------------------------------------------------------
# Error Handling in Handle Method
# ---------------------------------------------------------------------------


class TestHandleErrorHandling:
    """Test error handling in the top-level handle method."""

    @pytest.mark.asyncio
    async def test_handle_catches_value_error(self, handler):
        """ValueError in _get_tenant_id is caught by handle()."""
        with patch.object(handler, "_get_tenant_id", side_effect=ValueError("bad")):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_catches_key_error(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=KeyError("missing")):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_catches_type_error(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=TypeError("type")):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_catches_runtime_error(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=RuntimeError("runtime")):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_catches_os_error(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=OSError("os")):
            req = MockRequest(path="/api/v1/reconciliation/list", method="GET")
            result = await handler.handle(req, req.path, req.method)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Utility Methods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self, handler):
        req = MockRequest(_body={"key": "value"})
        body = await handler._get_json_body(req)
        assert body == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_empty(self, handler):
        req = MockRequest()
        body = await handler._get_json_body(req)
        assert body == {}

    @pytest.mark.asyncio
    async def test_get_json_body_non_dict(self, handler):
        """If json() returns a non-dict, return empty dict."""
        req = MagicMock()
        req.json = MagicMock(return_value="not-a-dict")
        body = await handler._get_json_body(req)
        assert body == {}

    @pytest.mark.asyncio
    async def test_get_json_body_property(self, handler):
        """Test json as property (non-callable)."""

        class PropRequest:
            @property
            def json(self):
                return {"prop": "value"}

        req = PropRequest()
        body = await handler._get_json_body(req)
        assert body == {"prop": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_no_json(self, handler):
        """Request without json attribute returns empty dict."""

        class BareRequest:
            pass

        req = BareRequest()
        body = await handler._get_json_body(req)
        assert body == {}

    def test_get_query_params_with_query(self, handler):
        req = MockRequest(query={"key": "value"})
        params = handler._get_query_params(req)
        assert params == {"key": "value"}

    def test_get_query_params_with_args(self, handler):
        """Test request with .args attribute instead of .query."""

        class ArgsRequest:
            args = {"key": "value"}

        req = ArgsRequest()
        params = handler._get_query_params(req)
        assert params == {"key": "value"}

    def test_get_query_params_empty(self, handler):
        """Test request without query or args."""

        class BareRequest:
            pass

        req = BareRequest()
        params = handler._get_query_params(req)
        assert params == {}


# ---------------------------------------------------------------------------
# ROUTES and RESOURCE_TYPE
# ---------------------------------------------------------------------------


class TestHandlerMetadata:
    """Test handler class-level metadata."""

    def test_routes_list(self, handler):
        assert "/api/v1/reconciliation/run" in handler.ROUTES
        assert "/api/v1/reconciliation/list" in handler.ROUTES
        assert "/api/v1/reconciliation/discrepancies" in handler.ROUTES
        assert "/api/v1/reconciliation/discrepancies/bulk-resolve" in handler.ROUTES
        assert "/api/v1/reconciliation/demo" in handler.ROUTES
        assert "/api/v1/reconciliation/status" in handler.ROUTES

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "reconciliation"

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 10

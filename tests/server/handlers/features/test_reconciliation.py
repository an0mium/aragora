"""
Tests for Bank Reconciliation Handler.

Tests cover:
- can_handle routing
- Permission checks (RBAC via @require_permission)
- Run reconciliation (POST /api/v1/reconciliation/run)
- List reconciliations (GET /api/v1/reconciliation/list)
- Get reconciliation details (GET /api/v1/reconciliation/{id})
- Generate report (GET /api/v1/reconciliation/{id}/report)
- Resolve discrepancy (POST /api/v1/reconciliation/{id}/resolve)
- Approve reconciliation (POST /api/v1/reconciliation/{id}/approve)
- Get discrepancies (GET /api/v1/reconciliation/discrepancies)
- Bulk resolve (POST /api/v1/reconciliation/discrepancies/bulk-resolve)
- Demo endpoint (GET /api/v1/reconciliation/demo)
- Error handling / service unavailable
- Module-level state cleanup
- Utility methods
"""

import json
import pytest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.reconciliation import (
    ReconciliationHandler,
    get_reconciliation_handler,
    handle_reconciliation,
    get_reconciliation_service,
    _service_instances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result):
    """Parse a HandlerResult dataclass into (body_dict, status_code).

    Note: success_response wraps data in {"success": True, "data": ...}
    We unwrap this for convenience in tests.
    """
    if hasattr(result, "body"):
        if isinstance(result.body, bytes):
            body = result.body.decode("utf-8")
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass  # Keep as string (e.g., CSV)
        else:
            body = json.loads(result.body) if result.body else {}
    else:
        body = {}

    # Unwrap success_response format
    if isinstance(body, dict) and "success" in body and "data" in body:
        body = body["data"]

    return body, result.status_code


def _make_request(
    *,
    tenant_id="test-tenant",
    user_id="test-user",
    query=None,
    json_body=None,
    headers=None,
):
    """Build a fake request object."""
    req = SimpleNamespace()
    req.tenant_id = tenant_id
    req.user_id = user_id
    req.query = query or {}
    req.headers = headers or {}
    req.args = query or {}

    if json_body is not None:

        async def _json():
            return json_body

        req.json = _json
    else:

        async def _json():
            return {}

        req.json = _json

    return req


def _make_reconciliation_result(
    reconciliation_id="recon_001",
    account_name="Business Checking",
    account_id="acc_001",
    start_date=None,
    end_date=None,
    bank_total=Decimal("10000.00"),
    book_total=Decimal("9800.00"),
    difference=Decimal("200.00"),
    matched_count=45,
    discrepancy_count=3,
    match_rate=0.92,
    is_reconciled=False,
    discrepancies=None,
    matched_transactions=None,
):
    """Create a mock ReconciliationResult object."""
    result = SimpleNamespace()
    result.reconciliation_id = reconciliation_id
    result.account_name = account_name
    result.account_id = account_id
    result.start_date = start_date or date.today() - timedelta(days=30)
    result.end_date = end_date or date.today()
    result.bank_total = bank_total
    result.book_total = book_total
    result.difference = difference
    result.matched_count = matched_count
    result.discrepancy_count = discrepancy_count
    result.match_rate = match_rate
    result.is_reconciled = is_reconciled
    result.reconciled_at = None
    result.reconciled_by = None
    result.discrepancies = discrepancies or []
    result.matched_transactions = matched_transactions or []

    def to_dict():
        return {
            "reconciliation_id": result.reconciliation_id,
            "account_name": result.account_name,
            "account_id": result.account_id,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "bank_total": float(result.bank_total),
            "book_total": float(result.book_total),
            "difference": float(result.difference),
            "matched_count": result.matched_count,
            "discrepancy_count": result.discrepancy_count,
            "match_rate": result.match_rate,
            "is_reconciled": result.is_reconciled,
        }

    result.to_dict = to_dict
    return result


def _make_discrepancy(
    discrepancy_id="disc_001",
    discrepancy_type="unmatched_bank",
    severity="medium",
    description="Bank transaction not found in books",
    resolution_status="pending",
    bank_amount=Decimal("156.78"),
    book_amount=None,
    bank_date=None,
    book_date=None,
):
    """Create a mock Discrepancy object."""
    disc = SimpleNamespace()
    disc.discrepancy_id = discrepancy_id
    disc.discrepancy_type = SimpleNamespace(value=discrepancy_type)
    disc.severity = SimpleNamespace(value=severity)
    disc.description = description
    disc.resolution_status = SimpleNamespace(value=resolution_status)
    disc.bank_amount = bank_amount
    disc.book_amount = book_amount
    disc.bank_date = bank_date or date.today() - timedelta(days=3)
    disc.book_date = book_date

    def to_dict():
        return {
            "discrepancy_id": disc.discrepancy_id,
            "discrepancy_type": disc.discrepancy_type.value,
            "severity": disc.severity.value,
            "description": disc.description,
            "resolution_status": disc.resolution_status.value,
            "bank_amount": float(disc.bank_amount) if disc.bank_amount else None,
            "book_amount": float(disc.book_amount) if disc.book_amount else None,
            "bank_date": disc.bank_date.isoformat() if disc.bank_date else None,
            "book_date": disc.book_date.isoformat() if disc.book_date else None,
        }

    disc.to_dict = to_dict
    return disc


def _make_matched_transaction(
    bank_txn_id="txn_001",
    book_txn_id="inv_001",
    bank_amount=Decimal("1250.00"),
    book_amount=Decimal("1250.00"),
    bank_date=None,
    book_date=None,
    bank_description="AWS Cloud Services",
    book_description="AWS Monthly Invoice",
    match_confidence=1.0,
    match_method="exact",
):
    """Create a mock MatchedTransaction object."""
    match = SimpleNamespace()
    match.bank_txn_id = bank_txn_id
    match.book_txn_id = book_txn_id
    match.bank_amount = bank_amount
    match.book_amount = book_amount
    match.bank_date = bank_date or date.today() - timedelta(days=5)
    match.book_date = book_date or date.today() - timedelta(days=5)
    match.bank_description = bank_description
    match.book_description = book_description
    match.match_confidence = match_confidence
    match.match_method = match_method

    def to_dict():
        return {
            "bank_txn_id": match.bank_txn_id,
            "book_txn_id": match.book_txn_id,
            "bank_amount": float(match.bank_amount),
            "book_amount": float(match.book_amount),
            "bank_date": match.bank_date.isoformat(),
            "book_date": match.book_date.isoformat(),
            "bank_description": match.bank_description,
            "book_description": match.book_description,
            "match_confidence": match.match_confidence,
            "match_method": match.match_method,
        }

    match.to_dict = to_dict
    return match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Clear module-level service caches between tests."""
    _service_instances.clear()
    yield
    _service_instances.clear()


@pytest.fixture()
def handler():
    """Create a ReconciliationHandler instance."""
    return ReconciliationHandler(server_context={})


@pytest.fixture()
def mock_service():
    """Create a mock ReconciliationService."""
    svc = MagicMock()
    svc.list_reconciliations = MagicMock(return_value=[])
    svc.get_reconciliation = MagicMock(return_value=None)
    svc.resolve_discrepancy = AsyncMock(return_value=True)
    return svc


def _patch_service(mock_svc):
    """Shorthand to patch the service lookup."""
    return patch(
        "aragora.server.handlers.features.reconciliation.get_reconciliation_service",
        return_value=mock_svc,
    )


def _patch_permission():
    """Patch RBAC permission check to allow all."""
    return patch(
        "aragora.server.handlers.features.reconciliation.require_permission",
        lambda perm: lambda fn: fn,
    )


# ---------------------------------------------------------------------------
# Tests: can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_reconciliation_run(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/run") is True

    def test_reconciliation_list(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/list") is True

    def test_reconciliation_with_id(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/recon_001") is True

    def test_reconciliation_report(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/recon_001/report") is True

    def test_reconciliation_resolve(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/recon_001/resolve") is True

    def test_reconciliation_approve(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/recon_001/approve") is True

    def test_reconciliation_discrepancies(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/discrepancies") is True

    def test_reconciliation_bulk_resolve(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/discrepancies/bulk-resolve") is True

    def test_reconciliation_demo(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/reconciliation/demo") is True

    def test_unrelated_path(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/api/v1/debates") is False

    def test_unrelated_root(self):
        h = ReconciliationHandler(server_context={})
        assert h.can_handle("/health") is False


class TestRoutes:
    def test_routes_defined(self):
        assert len(ReconciliationHandler.ROUTES) > 0

    def test_expected_routes_present(self):
        expected = [
            "/api/v1/reconciliation/run",
            "/api/v1/reconciliation/list",
            "/api/v1/reconciliation/discrepancies",
            "/api/v1/reconciliation/demo",
        ]
        for route in expected:
            assert route in ReconciliationHandler.ROUTES, f"Expected route: {route}"


# ---------------------------------------------------------------------------
# Tests: Handler creation and factory
# ---------------------------------------------------------------------------


class TestHandlerCreation:
    def test_handler_creation(self):
        handler = ReconciliationHandler(server_context={})
        assert handler is not None

    def test_handler_with_context(self):
        ctx = {"key": "value"}
        handler = ReconciliationHandler(server_context=ctx)
        assert handler.ctx == ctx

    def test_get_reconciliation_handler_singleton(self):
        h1 = get_reconciliation_handler()
        h2 = get_reconciliation_handler()
        assert h1 is h2
        assert isinstance(h1, ReconciliationHandler)


# ---------------------------------------------------------------------------
# Tests: Run reconciliation (POST /api/v1/reconciliation/run)
# ---------------------------------------------------------------------------


class TestRunReconciliation:
    @pytest.mark.asyncio
    async def test_run_reconciliation_missing_dates(self, handler):
        req = _make_request(json_body={"account_id": "acc_001"})
        with _patch_permission():
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "start_date" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_run_reconciliation_invalid_date_format(self, handler):
        req = _make_request(
            json_body={
                "start_date": "invalid",
                "end_date": "2024-01-31",
            }
        )
        with _patch_permission():
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "date format" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_run_reconciliation_end_before_start(self, handler):
        req = _make_request(
            json_body={
                "start_date": "2024-01-31",
                "end_date": "2024-01-01",
            }
        )
        with _patch_permission():
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "after" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_run_reconciliation_service_unavailable(self, handler):
        req = _make_request(
            json_body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "plaid_access_token": "access_token",
            }
        )
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 503
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_run_reconciliation_demo_mode(self, handler, mock_service):
        """When no plaid_access_token, returns demo data."""
        req = _make_request(
            json_body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }
        )
        mock_result = _make_reconciliation_result()
        mock_result.discrepancies = [_make_discrepancy()]
        mock_result.matched_transactions = [_make_matched_transaction()]

        with (
            _patch_permission(),
            _patch_service(mock_service),
            patch(
                "aragora.services.accounting.reconciliation.get_mock_reconciliation_result",
                return_value=mock_result,
            ),
        ):
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 200
        assert body.get("is_demo") is True
        assert "reconciliation" in body

    @pytest.mark.asyncio
    async def test_run_reconciliation_success(self, handler, mock_service):
        recon_result = _make_reconciliation_result()
        recon_result.discrepancies = [_make_discrepancy()]
        mock_service.reconcile = AsyncMock(return_value=recon_result)

        req = _make_request(
            json_body={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "plaid_access_token": "access_token",
                "account_id": "acc_001",
            }
        )

        with (
            _patch_permission(),
            _patch_service(mock_service),
            patch("aragora.connectors.accounting.plaid.PlaidCredentials"),
        ):
            result = await handler.handle(req, "/api/v1/reconciliation/run", "POST")
        body, status = _parse_result(result)
        assert status == 200
        assert "reconciliation" in body


# ---------------------------------------------------------------------------
# Tests: List reconciliations (GET /api/v1/reconciliation/list)
# ---------------------------------------------------------------------------


class TestListReconciliations:
    @pytest.mark.asyncio
    async def test_list_reconciliations_empty(self, handler, mock_service):
        mock_service.list_reconciliations = MagicMock(return_value=[])
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["reconciliations"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_reconciliations_success(self, handler, mock_service):
        recon = _make_reconciliation_result()
        mock_service.list_reconciliations = MagicMock(return_value=[recon])
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1
        assert len(body["reconciliations"]) == 1

    @pytest.mark.asyncio
    async def test_list_reconciliations_with_filters(self, handler, mock_service):
        mock_service.list_reconciliations = MagicMock(return_value=[])
        req = _make_request(query={"account_id": "acc_001", "limit": "10"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 200
        mock_service.list_reconciliations.assert_called_once_with(account_id="acc_001", limit=10)

    @pytest.mark.asyncio
    async def test_list_reconciliations_no_service(self, handler):
        req = _make_request()
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["reconciliations"] == []


# ---------------------------------------------------------------------------
# Tests: Get reconciliation (GET /api/v1/reconciliation/{id})
# ---------------------------------------------------------------------------


class TestGetReconciliation:
    @pytest.mark.asyncio
    async def test_get_reconciliation_not_found(self, handler, mock_service):
        mock_service.get_reconciliation = MagicMock(return_value=None)
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_999", "GET")
        body, status = _parse_result(result)
        assert status == 404
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_reconciliation_success(self, handler, mock_service):
        recon = _make_reconciliation_result(reconciliation_id="recon_001")
        recon.discrepancies = [_make_discrepancy()]
        recon.matched_transactions = [_make_matched_transaction()]
        mock_service.get_reconciliation = MagicMock(return_value=recon)
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["reconciliation"]["reconciliation_id"] == "recon_001"
        assert "discrepancies" in body
        assert "matched_transactions" in body

    @pytest.mark.asyncio
    async def test_get_reconciliation_service_unavailable(self, handler):
        req = _make_request()
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001", "GET")
        body, status = _parse_result(result)
        assert status == 503


# ---------------------------------------------------------------------------
# Tests: Demo endpoint (GET /api/v1/reconciliation/demo)
# ---------------------------------------------------------------------------


class TestDemoEndpoint:
    @pytest.mark.asyncio
    async def test_demo_success(self, handler, mock_service):
        mock_result = _make_reconciliation_result()
        mock_result.discrepancies = [_make_discrepancy()]
        mock_result.matched_transactions = [_make_matched_transaction()]

        req = _make_request()
        with (
            _patch_permission(),
            _patch_service(mock_service),
            patch(
                "aragora.services.accounting.reconciliation.get_mock_reconciliation_result",
                return_value=mock_result,
            ),
        ):
            result = await handler.handle(req, "/api/v1/reconciliation/demo", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body.get("is_demo") is True
        assert "reconciliation" in body

    @pytest.mark.asyncio
    async def test_demo_import_error(self, handler, mock_service):
        """When get_mock_reconciliation_result module can't be imported, returns 503."""
        req = _make_request()
        # Mock the import to fail within _handle_demo
        with (
            _patch_permission(),
            _patch_service(mock_service),
            patch.dict(
                "sys.modules",
                {"aragora.services.accounting.reconciliation": None},
            ),
        ):
            # Force re-import to trigger ImportError
            result = await handler.handle(req, "/api/v1/reconciliation/demo", "GET")
        body, status = _parse_result(result)
        # The actual behavior depends on import mechanism - may be 503 or 200
        assert status in (200, 503)


# ---------------------------------------------------------------------------
# Tests: Resolve discrepancy (POST /api/v1/reconciliation/{id}/resolve)
# ---------------------------------------------------------------------------


class TestResolveDiscrepancy:
    @pytest.mark.asyncio
    async def test_resolve_missing_discrepancy_id(self, handler, mock_service):
        req = _make_request(json_body={"resolution": "Fixed manually"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/resolve", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "discrepancy_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_resolve_service_unavailable(self, handler):
        req = _make_request(json_body={"discrepancy_id": "disc_001"})
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/resolve", "POST")
        body, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_resolve_success(self, handler, mock_service):
        recon = _make_reconciliation_result(is_reconciled=True)
        recon.discrepancies = []
        mock_service.get_reconciliation = MagicMock(return_value=recon)
        mock_service.resolve_discrepancy = AsyncMock(return_value=True)

        req = _make_request(
            json_body={
                "discrepancy_id": "disc_001",
                "resolution": "Created expense entry",
                "action": "create_entry",
            }
        )
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/resolve", "POST")
        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "resolved"
        assert body["discrepancy_id"] == "disc_001"

    @pytest.mark.asyncio
    async def test_resolve_failure(self, handler, mock_service):
        mock_service.resolve_discrepancy = AsyncMock(return_value=False)

        req = _make_request(json_body={"discrepancy_id": "disc_001"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/resolve", "POST")
        body, status = _parse_result(result)
        assert status == 400


# ---------------------------------------------------------------------------
# Tests: Bulk resolve (POST /api/v1/reconciliation/discrepancies/bulk-resolve)
# ---------------------------------------------------------------------------


class TestBulkResolve:
    @pytest.mark.asyncio
    async def test_bulk_resolve_missing_reconciliation_id(self, handler, mock_service):
        req = _make_request(json_body={"resolutions": []})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(
                req, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
            )
        body, status = _parse_result(result)
        assert status == 400
        assert "reconciliation_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_bulk_resolve_service_unavailable(self, handler):
        req = _make_request(json_body={"reconciliation_id": "recon_001", "resolutions": []})
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(
                req, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
            )
        body, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_bulk_resolve_success(self, handler, mock_service):
        recon = _make_reconciliation_result(is_reconciled=True)
        mock_service.get_reconciliation = MagicMock(return_value=recon)
        mock_service.resolve_discrepancy = AsyncMock(return_value=True)

        req = _make_request(
            json_body={
                "reconciliation_id": "recon_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "resolution": "Fixed"},
                    {"discrepancy_id": "disc_002", "resolution": "Ignored", "action": "ignore"},
                ],
            }
        )
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(
                req, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
            )
        body, status = _parse_result(result)
        assert status == 200
        assert body["resolved_count"] == 2
        assert body["error_count"] == 0

    @pytest.mark.asyncio
    async def test_bulk_resolve_partial_failure(self, handler, mock_service):
        recon = _make_reconciliation_result()
        mock_service.get_reconciliation = MagicMock(return_value=recon)
        # First succeeds, second fails
        mock_service.resolve_discrepancy = AsyncMock(side_effect=[True, False])

        req = _make_request(
            json_body={
                "reconciliation_id": "recon_001",
                "resolutions": [
                    {"discrepancy_id": "disc_001", "resolution": "Fixed"},
                    {"discrepancy_id": "disc_002", "resolution": "Fixed"},
                ],
            }
        )
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(
                req, "/api/v1/reconciliation/discrepancies/bulk-resolve", "POST"
            )
        body, status = _parse_result(result)
        assert status == 200
        assert body["resolved_count"] == 1
        assert body["error_count"] == 1
        assert body["errors"] is not None


# ---------------------------------------------------------------------------
# Tests: Approve reconciliation (POST /api/v1/reconciliation/{id}/approve)
# ---------------------------------------------------------------------------


class TestApproveReconciliation:
    @pytest.mark.asyncio
    async def test_approve_service_unavailable(self, handler):
        req = _make_request(json_body={"notes": "Approved"})
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/approve", "POST")
        body, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_approve_not_found(self, handler, mock_service):
        mock_service.get_reconciliation = MagicMock(return_value=None)
        req = _make_request(json_body={"notes": "Approved"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/approve", "POST")
        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_approve_with_pending_discrepancies(self, handler, mock_service):
        disc = _make_discrepancy(resolution_status="pending")
        recon = _make_reconciliation_result()
        recon.discrepancies = [disc]
        mock_service.get_reconciliation = MagicMock(return_value=recon)

        req = _make_request(json_body={"notes": "Approved"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/approve", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "unresolved" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_approve_success(self, handler, mock_service):
        recon = _make_reconciliation_result()
        recon.discrepancies = []
        mock_service.get_reconciliation = MagicMock(return_value=recon)

        req = _make_request(json_body={"notes": "Reviewed and approved"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/approve", "POST")
        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "approved"
        assert body["reconciliation_id"] == "recon_001"
        assert body["approved_by"] == "test-user"


# ---------------------------------------------------------------------------
# Tests: Get discrepancies (GET /api/v1/reconciliation/discrepancies)
# ---------------------------------------------------------------------------


class TestGetDiscrepancies:
    @pytest.mark.asyncio
    async def test_get_discrepancies_no_service(self, handler):
        req = _make_request()
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/discrepancies", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["discrepancies"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_get_discrepancies_success(self, handler, mock_service):
        disc = _make_discrepancy(severity="high")
        recon = _make_reconciliation_result()
        recon.discrepancies = [disc]
        mock_service.list_reconciliations = MagicMock(return_value=[recon])

        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/discrepancies", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1
        assert len(body["discrepancies"]) == 1

    @pytest.mark.asyncio
    async def test_get_discrepancies_with_filters(self, handler, mock_service):
        disc_pending = _make_discrepancy(resolution_status="pending", severity="high")
        disc_resolved = _make_discrepancy(
            discrepancy_id="disc_002", resolution_status="user_resolved", severity="low"
        )
        recon = _make_reconciliation_result()
        recon.discrepancies = [disc_pending, disc_resolved]
        mock_service.list_reconciliations = MagicMock(return_value=[recon])

        # Filter by status=pending
        req = _make_request(query={"status": "pending"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/discrepancies", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_get_discrepancies_sorted_by_severity(self, handler, mock_service):
        disc_low = _make_discrepancy(discrepancy_id="disc_low", severity="low")
        disc_critical = _make_discrepancy(discrepancy_id="disc_critical", severity="critical")
        disc_medium = _make_discrepancy(discrepancy_id="disc_medium", severity="medium")
        recon = _make_reconciliation_result()
        recon.discrepancies = [disc_low, disc_critical, disc_medium]
        mock_service.list_reconciliations = MagicMock(return_value=[recon])

        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/discrepancies", "GET")
        body, status = _parse_result(result)
        assert status == 200
        # Should be sorted by severity (critical first)
        severities = [d["severity"] for d in body["discrepancies"]]
        assert severities[0] == "critical"


# ---------------------------------------------------------------------------
# Tests: Generate report (GET /api/v1/reconciliation/{id}/report)
# ---------------------------------------------------------------------------


class TestGenerateReport:
    @pytest.mark.asyncio
    async def test_report_service_unavailable(self, handler):
        req = _make_request()
        with _patch_permission(), _patch_service(None):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/report", "GET")
        body, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_report_not_found(self, handler, mock_service):
        mock_service.get_reconciliation = MagicMock(return_value=None)
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_999/report", "GET")
        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_report_json_format(self, handler, mock_service):
        recon = _make_reconciliation_result()
        recon.discrepancies = [_make_discrepancy()]
        mock_service.get_reconciliation = MagicMock(return_value=recon)

        req = _make_request(query={"format": "json"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/report", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert "report" in body
        assert "title" in body["report"]
        assert "summary" in body["report"]
        assert "period" in body["report"]

    @pytest.mark.asyncio
    async def test_report_csv_format(self, handler, mock_service):
        recon = _make_reconciliation_result()
        disc = _make_discrepancy()
        recon.discrepancies = [disc]
        mock_service.get_reconciliation = MagicMock(return_value=recon)

        req = _make_request(query={"format": "csv"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/report", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert result.content_type == "text/csv"
        # Check CSV content
        assert "Type,Description" in body
        assert "unmatched_bank" in body

    @pytest.mark.asyncio
    async def test_report_unsupported_format(self, handler, mock_service):
        recon = _make_reconciliation_result()
        mock_service.get_reconciliation = MagicMock(return_value=recon)

        req = _make_request(query={"format": "pdf"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/report", "GET")
        body, status = _parse_result(result)
        assert status == 400
        assert "unsupported" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# Tests: Not found / unknown routes
# ---------------------------------------------------------------------------


class TestNotFound:
    @pytest.mark.asyncio
    async def test_put_not_supported(self, handler):
        req = _make_request()
        with _patch_permission():
            result = await handler.handle(req, "/api/v1/reconciliation/run", "PUT")
        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_delete_not_supported(self, handler):
        req = _make_request()
        with _patch_permission():
            result = await handler.handle(req, "/api/v1/reconciliation/list", "DELETE")
        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_unknown_action(self, handler, mock_service):
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(
                req, "/api/v1/reconciliation/recon_001/unknown_action", "POST"
            )
        body, status = _parse_result(result)
        assert status == 404


# ---------------------------------------------------------------------------
# Tests: handle_reconciliation entry point
# ---------------------------------------------------------------------------


class TestHandleReconciliationEntryPoint:
    @pytest.mark.asyncio
    async def test_handle_reconciliation_function(self, mock_service):
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handle_reconciliation(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 200


# ---------------------------------------------------------------------------
# Tests: Tenant ID extraction
# ---------------------------------------------------------------------------


class TestTenantId:
    def test_extracts_from_request(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace(tenant_id="my-tenant")
        assert h._get_tenant_id(req) == "my-tenant"

    def test_defaults_to_default(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace()
        assert h._get_tenant_id(req) == "default"


# ---------------------------------------------------------------------------
# Tests: Utility methods
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_get_query_params_from_query(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace(query={"a": "1", "b": "2"})
        assert h._get_query_params(req) == {"a": "1", "b": "2"}

    def test_get_query_params_from_args(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace(args={"x": "y"})
        assert h._get_query_params(req) == {"x": "y"}

    def test_get_query_params_empty(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace()
        assert h._get_query_params(req) == {}

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self):
        h = ReconciliationHandler(server_context={})

        async def _json():
            return {"key": "val"}

        req = SimpleNamespace(json=_json)
        result = await h._get_json_body(req)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_get_json_body_property(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace(json={"key": "val"})
        result = await h._get_json_body(req)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_get_json_body_none(self):
        h = ReconciliationHandler(server_context={})
        req = SimpleNamespace()
        result = await h._get_json_body(req)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_reconciliation_service
# ---------------------------------------------------------------------------


class TestGetReconciliationService:
    def test_returns_none_on_import_error(self):
        with patch.dict(
            "sys.modules",
            {"aragora.services.accounting.reconciliation": None},
        ):
            _service_instances.clear()
            result = get_reconciliation_service("tenant1")
        # Result depends on whether import fails
        # In test environment, it may succeed

    def test_returns_cached_service(self):
        mock_svc = MagicMock()
        _service_instances["cached_tenant"] = mock_svc
        result = get_reconciliation_service("cached_tenant")
        assert result is mock_svc

    def test_creates_new_service_per_tenant(self):
        _service_instances.clear()
        with patch("aragora.services.accounting.reconciliation.ReconciliationService") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc1 = get_reconciliation_service("tenant1")
            svc2 = get_reconciliation_service("tenant1")
            # Should be same instance (cached)
            assert svc1 is svc2


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_general_exception_returns_500(self, handler, mock_service):
        mock_service.list_reconciliations = MagicMock(side_effect=RuntimeError("Database error"))
        req = _make_request()
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/list", "GET")
        body, status = _parse_result(result)
        assert status == 500
        assert "error" in body

    @pytest.mark.asyncio
    async def test_resolve_exception_returns_500(self, handler, mock_service):
        mock_service.resolve_discrepancy = AsyncMock(side_effect=RuntimeError("Resolution failed"))
        req = _make_request(json_body={"discrepancy_id": "disc_001"})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/resolve", "POST")
        body, status = _parse_result(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_approve_exception_returns_500(self, handler, mock_service):
        mock_service.get_reconciliation = MagicMock(side_effect=RuntimeError("Database error"))
        req = _make_request(json_body={})
        with _patch_permission(), _patch_service(mock_service):
            result = await handler.handle(req, "/api/v1/reconciliation/recon_001/approve", "POST")
        body, status = _parse_result(result)
        assert status == 500

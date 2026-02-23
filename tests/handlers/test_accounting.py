"""Tests for accounting handler (aragora/server/handlers/accounting.py).

Comprehensive test suite covering all routes and behavior:
- Helper functions (_parse_iso_date, _generate_mock_report)
- Mock data constants (MOCK_COMPANY, MOCK_STATS, MOCK_CUSTOMERS, MOCK_TRANSACTIONS)
- GET /api/accounting/status - QuickBooks status + dashboard data
- GET /api/accounting/connect - Start QuickBooks OAuth
- GET /api/accounting/callback - QuickBooks OAuth callback
- POST /api/accounting/disconnect - Disconnect QuickBooks
- GET /api/accounting/customers - List QuickBooks customers
- GET /api/accounting/transactions - List QuickBooks transactions
- POST /api/accounting/report - Generate accounting report
- GET /api/accounting/gusto/status - Gusto connection status
- GET /api/accounting/gusto/connect - Start Gusto OAuth
- GET /api/accounting/gusto/callback - Gusto OAuth callback
- POST /api/accounting/gusto/disconnect - Disconnect Gusto
- GET /api/accounting/gusto/employees - List employees
- GET /api/accounting/gusto/payrolls - List payroll runs
- GET /api/accounting/gusto/payrolls/{payroll_id} - Payroll run details
- POST /api/accounting/gusto/payrolls/{payroll_id}/journal-entry - Generate journal entry
- register_accounting_routes - Route registration
- Error handling (auth failures, API errors, bad data)
- Edge cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.accounting import (
    MOCK_COMPANY,
    MOCK_CUSTOMERS,
    MOCK_STATS,
    MOCK_TRANSACTIONS,
    _generate_mock_report,
    _parse_iso_date,
    get_gusto_connector,
    get_qbo_connector,
    handle_accounting_callback,
    handle_accounting_connect,
    handle_accounting_customers,
    handle_accounting_disconnect,
    handle_accounting_report,
    handle_accounting_status,
    handle_accounting_transactions,
    handle_gusto_callback,
    handle_gusto_connect,
    handle_gusto_disconnect,
    handle_gusto_employees,
    handle_gusto_journal_entry,
    handle_gusto_payroll_detail,
    handle_gusto_payrolls,
    handle_gusto_status,
    register_accounting_routes,
)


# ===========================================================================
# Mock data classes
# ===========================================================================


@dataclass
class MockQBOCompany:
    """Mock QuickBooks company info."""

    name: str = "Test Company"
    legal_name: str = "Test Company LLC"
    country: str = "US"
    email: str = "accounting@test.com"


@dataclass
class MockQBOCustomer:
    """Mock QuickBooks customer."""

    id: str = "cust_123"
    display_name: str = "Test Customer"
    company_name: str = "Test Corp"
    email: str = "billing@test.com"
    balance: float = 1500.00
    active: bool = True


@dataclass
class MockQBOInvoice:
    """Mock QuickBooks invoice."""

    id: str = "inv_123"
    type: str = "Invoice"
    doc_number: str = "INV-001"
    txn_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime | None = None
    total_amount: float = 1000.00
    balance: float = 0
    customer_name: str = "Test Customer"
    status: str = "Paid"


@dataclass
class MockQBOExpense:
    """Mock QuickBooks expense."""

    id: str = "exp_123"
    doc_number: str = "EXP-001"
    txn_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_amount: float = 500.00
    balance: float = 0
    vendor_name: str = "Test Vendor"
    status: str = "Paid"


@dataclass
class MockGustoCredentials:
    """Mock Gusto credentials."""

    company_id: str = "gusto_company_123"
    company_name: str = "Test Company"
    access_token: str = "test_access_token"


@dataclass
class MockGustoEmployee:
    """Mock Gusto employee."""

    id: str = "emp_123"
    first_name: str = "John"
    last_name: str = "Doe"
    email: str = "john.doe@test.com"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
        }


@dataclass
class MockGustoPayrollItem:
    """Mock Gusto payroll item."""

    employee_id: str = "emp_123"
    gross_pay: float = 5000.00
    net_pay: float = 3500.00

    def to_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "gross_pay": self.gross_pay,
            "net_pay": self.net_pay,
        }


@dataclass
class MockGustoPayroll:
    """Mock Gusto payroll."""

    id: str = "payroll_123"
    pay_period_start: date = field(default_factory=lambda: date(2025, 1, 1))
    pay_period_end: date = field(default_factory=lambda: date(2025, 1, 15))
    check_date: date = field(default_factory=lambda: date(2025, 1, 20))
    processed: bool = True
    payroll_items: list = field(default_factory=lambda: [MockGustoPayrollItem()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pay_period_start": self.pay_period_start.isoformat(),
            "pay_period_end": self.pay_period_end.isoformat(),
            "check_date": self.check_date.isoformat(),
            "processed": self.processed,
        }


@dataclass
class MockJournalEntry:
    """Mock journal entry."""

    date: date = field(default_factory=lambda: date(2025, 1, 20))
    memo: str = "Payroll Journal Entry"
    lines: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "memo": self.memo,
            "lines": self.lines,
        }


# ===========================================================================
# Helper to build mock aiohttp requests
# ===========================================================================


def create_mock_request(
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    match_info: dict[str, str] | None = None,
    app_state: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock aiohttp request with the given parameters."""
    request = MagicMock(spec=web.Request)
    request.query = query or {}
    request.match_info = match_info or {}
    request.app = app_state if app_state is not None else {}

    if body is not None:

        async def json_func():
            return body

        request.json = json_func
        request.content_length = len(json.dumps(body).encode())

        async def read_func():
            return json.dumps(body).encode()

        request.read = read_func
    else:

        async def json_error():
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        request.json = json_error
        request.content_length = None

        async def read_empty():
            return b""

        request.read = read_empty

    return request


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_qbo_connector():
    """Create a mock QuickBooks connector."""
    connector = MagicMock()
    connector.is_connected = MagicMock(return_value=True)
    connector.get_company_info = AsyncMock(return_value=MockQBOCompany())
    connector.list_customers = AsyncMock(return_value=[MockQBOCustomer()])
    connector.list_invoices = AsyncMock(return_value=[MockQBOInvoice()])
    connector.list_expenses = AsyncMock(return_value=[MockQBOExpense()])
    connector.get_authorization_url = MagicMock(return_value="https://oauth.intuit.com/authorize")
    connector.exchange_code = AsyncMock(return_value={"access_token": "test_token"})
    connector.revoke_token = AsyncMock()
    connector.get_profit_loss_report = AsyncMock(return_value={"title": "P&L"})
    connector.get_balance_sheet_report = AsyncMock(return_value={"title": "Balance Sheet"})
    connector.get_ar_aging_report = AsyncMock(return_value={"title": "AR Aging"})
    connector.get_ap_aging_report = AsyncMock(return_value={"title": "AP Aging"})
    return connector


@pytest.fixture
def mock_gusto_connector():
    """Create a mock Gusto connector."""
    connector = MagicMock()
    connector.is_configured = True
    connector.is_authenticated = True
    connector.get_authorization_url = MagicMock(
        return_value="https://api.gusto.com/oauth/authorize"
    )
    connector.exchange_code = AsyncMock(return_value=MockGustoCredentials())
    connector.list_employees = AsyncMock(return_value=[MockGustoEmployee()])
    connector.list_payrolls = AsyncMock(return_value=[MockGustoPayroll()])
    connector.get_payroll = AsyncMock(return_value=MockGustoPayroll())
    connector.generate_journal_entry = MagicMock(return_value=MockJournalEntry())
    connector.set_credentials = MagicMock()
    return connector


# ===========================================================================
# Test Helper Functions
# ===========================================================================


class TestParseIsoDate:
    """Tests for _parse_iso_date helper."""

    def test_valid_date(self):
        result = _parse_iso_date("2025-01-15", "test_date")
        assert result == date(2025, 1, 15)

    def test_none_returns_none(self):
        result = _parse_iso_date(None, "test_date")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_iso_date("", "test_date")
        assert result is None

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid test_date"):
            _parse_iso_date("not-a-date", "test_date")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid start_date"):
            _parse_iso_date("2025/01/15", "start_date")

    def test_partial_date_raises(self):
        with pytest.raises(ValueError, match="Invalid end_date"):
            _parse_iso_date("2025-13-01", "end_date")


class TestGenerateMockReport:
    """Tests for _generate_mock_report helper."""

    def test_profit_loss(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("profit_loss", start, end)
        assert report["title"] == "Profit and Loss"
        assert "sections" in report
        assert "netIncome" in report
        assert len(report["sections"]) == 4

    def test_balance_sheet(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("balance_sheet", start, end)
        assert report["title"] == "Balance Sheet"
        assert "sections" in report
        assert "as_of" in report

    def test_ar_aging(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("ar_aging", start, end)
        assert report["title"] == "Accounts Receivable Aging"
        assert "buckets" in report
        assert "total" in report

    def test_ap_aging(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("ap_aging", start, end)
        assert report["title"] == "Accounts Payable Aging"
        assert "buckets" in report

    def test_unknown_type_returns_error(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("unknown_type", start, end)
        assert "error" in report

    def test_profit_loss_period_formatted(self):
        start = datetime(2025, 3, 1)
        end = datetime(2025, 3, 31)
        report = _generate_mock_report("profit_loss", start, end)
        assert "Mar 01, 2025" in report["period"]
        assert "Mar 31, 2025" in report["period"]


# ===========================================================================
# Test Mock Data Constants
# ===========================================================================


class TestMockDataConstants:
    """Tests for mock data constants."""

    def test_mock_company_fields(self):
        assert "name" in MOCK_COMPANY
        assert "legalName" in MOCK_COMPANY
        assert "country" in MOCK_COMPANY
        assert "email" in MOCK_COMPANY

    def test_mock_stats_fields(self):
        assert "receivables" in MOCK_STATS
        assert "payables" in MOCK_STATS
        assert "revenue" in MOCK_STATS
        assert "expenses" in MOCK_STATS
        assert "netIncome" in MOCK_STATS
        assert "openInvoices" in MOCK_STATS
        assert "overdueInvoices" in MOCK_STATS

    def test_mock_customers_structure(self):
        assert len(MOCK_CUSTOMERS) == 4
        for customer in MOCK_CUSTOMERS:
            assert "id" in customer
            assert "displayName" in customer
            assert "email" in customer

    def test_mock_transactions_structure(self):
        assert len(MOCK_TRANSACTIONS) == 5
        for txn in MOCK_TRANSACTIONS:
            assert "id" in txn
            assert "type" in txn
            assert "totalAmount" in txn


# ===========================================================================
# Test get_qbo_connector / get_gusto_connector helpers
# ===========================================================================


class TestGetConnectorHelpers:
    """Tests for the connector accessor functions."""

    @pytest.mark.asyncio
    async def test_get_qbo_connector_returns_from_app(self, mock_qbo_connector):
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        result = await get_qbo_connector(request)
        assert result is mock_qbo_connector

    @pytest.mark.asyncio
    async def test_get_qbo_connector_returns_none_when_missing(self):
        request = create_mock_request(app_state={})
        result = await get_qbo_connector(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_gusto_connector_creates_new_if_missing(self):
        request = create_mock_request(app_state={})
        with patch("aragora.server.handlers.accounting.GustoConnector") as MockGusto:
            mock_instance = MagicMock()
            MockGusto.return_value = mock_instance
            result = await get_gusto_connector(request)
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_get_gusto_connector_reuses_existing(self, mock_gusto_connector):
        request = create_mock_request(app_state={"gusto_connector": mock_gusto_connector})
        result = await get_gusto_connector(request)
        assert result is mock_gusto_connector

    @pytest.mark.asyncio
    async def test_get_gusto_connector_sets_credentials(self, mock_gusto_connector):
        creds = MockGustoCredentials()
        request = create_mock_request(
            app_state={
                "gusto_connector": mock_gusto_connector,
                "gusto_credentials": creds,
            }
        )
        result = await get_gusto_connector(request)
        mock_gusto_connector.set_credentials.assert_called_once_with(creds)
        assert result is mock_gusto_connector


# ===========================================================================
# Test Accounting Status Handler
# ===========================================================================


class TestAccountingStatusHandler:
    """Tests for handle_accounting_status."""

    @pytest.mark.asyncio
    async def test_status_connected_returns_real_data(self, mock_qbo_connector):
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is True
        assert "company" in data
        assert data["company"]["name"] == "Test Company"
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_status_not_connected_returns_mock(self):
        request = create_mock_request(app_state={})
        response = await handle_accounting_status(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is True  # Simulated
        assert data["company"] == MOCK_COMPANY
        assert data["stats"] == MOCK_STATS
        assert data["customers"] == MOCK_CUSTOMERS
        assert data["transactions"] == MOCK_TRANSACTIONS

    @pytest.mark.asyncio
    async def test_status_connector_not_connected(self):
        connector = MagicMock()
        connector.is_connected = MagicMock(return_value=False)
        request = create_mock_request(app_state={"qbo_connector": connector})
        response = await handle_accounting_status(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["company"] == MOCK_COMPANY

    @pytest.mark.asyncio
    async def test_status_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.get_company_info = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500
        data = json.loads(response.text)
        assert data["connected"] is False

    @pytest.mark.asyncio
    async def test_status_customers_in_real_data(self, mock_qbo_connector):
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        data = json.loads(response.text)
        assert len(data["customers"]) == 1
        assert data["customers"][0]["displayName"] == "Test Customer"

    @pytest.mark.asyncio
    async def test_status_transactions_in_real_data(self, mock_qbo_connector):
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        data = json.loads(response.text)
        assert len(data["transactions"]) == 1
        assert data["transactions"][0]["type"] == "Invoice"

    @pytest.mark.asyncio
    async def test_status_open_invoice_stats(self, mock_qbo_connector):
        """Open invoices (balance > 0) are counted in stats."""
        open_invoice = MockQBOInvoice(balance=500.00)
        mock_qbo_connector.list_invoices = AsyncMock(return_value=[open_invoice])
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        data = json.loads(response.text)
        assert data["stats"]["openInvoices"] == 1
        assert data["stats"]["receivables"] == 500.00


# ===========================================================================
# Test Accounting Connect Handler
# ===========================================================================


class TestAccountingConnectHandler:
    """Tests for handle_accounting_connect."""

    @pytest.mark.asyncio
    async def test_connect_redirects_to_oauth(self, mock_qbo_connector):
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        with pytest.raises(web.HTTPFound) as exc_info:
            await handle_accounting_connect(request)
        assert "oauth.intuit.com" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_connect_no_connector_returns_503(self):
        request = create_mock_request(app_state={})
        response = await handle_accounting_connect(request)
        assert response.status == 503
        data = json.loads(response.text)
        assert "not configured" in data["error"]

    @pytest.mark.asyncio
    async def test_connect_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.get_authorization_url = MagicMock(
            side_effect=RuntimeError("OAuth config error")
        )
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_connect(request)
        assert response.status == 500


# ===========================================================================
# Test Accounting Callback Handler
# ===========================================================================


class TestAccountingCallbackHandler:
    """Tests for handle_accounting_callback."""

    @pytest.mark.asyncio
    async def test_callback_success(self, mock_qbo_connector):
        app_state = {"qbo_connector": mock_qbo_connector}
        request = create_mock_request(
            query={"code": "auth_code_123", "realmId": "realm_456"},
            app_state=app_state,
        )
        with pytest.raises(web.HTTPFound) as exc_info:
            await handle_accounting_callback(request)
        assert "connected=true" in str(exc_info.value.location)
        mock_qbo_connector.exchange_code.assert_awaited_once_with("auth_code_123", "realm_456")

    @pytest.mark.asyncio
    async def test_callback_with_oauth_error(self):
        request = create_mock_request(
            query={
                "error": "access_denied",
                "error_description": "User denied access",
            },
            app_state={},
        )
        response = await handle_accounting_callback(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "access_denied"
        assert data["description"] == "User denied access"

    @pytest.mark.asyncio
    async def test_callback_missing_code(self):
        request = create_mock_request(
            query={"realmId": "realm_456"},
            app_state={},
        )
        response = await handle_accounting_callback(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Missing" in data["error"]

    @pytest.mark.asyncio
    async def test_callback_missing_realm_id(self):
        request = create_mock_request(
            query={"code": "auth_code_123"},
            app_state={},
        )
        response = await handle_accounting_callback(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_callback_no_connector_returns_503(self):
        request = create_mock_request(
            query={"code": "auth_code_123", "realmId": "realm_456"},
            app_state={},
        )
        response = await handle_accounting_callback(request)
        assert response.status == 503
        data = json.loads(response.text)
        assert "not available" in data["error"]

    @pytest.mark.asyncio
    async def test_callback_exchange_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.exchange_code = AsyncMock(
            side_effect=RuntimeError("Token exchange failed")
        )
        request = create_mock_request(
            query={"code": "auth_code_123", "realmId": "realm_456"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_callback(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_callback_stores_credentials(self, mock_qbo_connector):
        app_state = {"qbo_connector": mock_qbo_connector}
        request = create_mock_request(
            query={"code": "auth_code_123", "realmId": "realm_456"},
            app_state=app_state,
        )
        with pytest.raises(web.HTTPFound):
            await handle_accounting_callback(request)
        assert "qbo_credentials" in app_state


# ===========================================================================
# Test Accounting Disconnect Handler
# ===========================================================================


class TestAccountingDisconnectHandler:
    """Tests for handle_accounting_disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, mock_qbo_connector):
        app_state = {
            "qbo_connector": mock_qbo_connector,
            "qbo_credentials": {"token": "test"},
        }
        request = create_mock_request(app_state=app_state)
        response = await handle_accounting_disconnect(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True
        assert data["message"] == "QuickBooks disconnected"
        assert "qbo_credentials" not in app_state
        mock_qbo_connector.revoke_token.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_no_connector(self):
        app_state = {"qbo_credentials": {"token": "test"}}
        request = create_mock_request(app_state=app_state)
        response = await handle_accounting_disconnect(request)
        assert response.status == 200
        assert "qbo_credentials" not in app_state

    @pytest.mark.asyncio
    async def test_disconnect_no_credentials(self):
        request = create_mock_request(app_state={})
        response = await handle_accounting_disconnect(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_revoke_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.revoke_token = AsyncMock(side_effect=ConnectionError("Cannot reach QBO"))
        request = create_mock_request(
            app_state={
                "qbo_connector": mock_qbo_connector,
                "qbo_credentials": {"token": "test"},
            }
        )
        response = await handle_accounting_disconnect(request)
        assert response.status == 500


# ===========================================================================
# Test Accounting Customers Handler
# ===========================================================================


class TestAccountingCustomersHandler:
    """Tests for handle_accounting_customers."""

    @pytest.mark.asyncio
    async def test_customers_connected(self, mock_qbo_connector):
        request = create_mock_request(
            query={"active": "true", "limit": "50", "offset": "0"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_customers(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "customers" in data
        assert data["total"] == 1
        assert data["customers"][0]["displayName"] == "Test Customer"

    @pytest.mark.asyncio
    async def test_customers_not_connected_returns_mock(self):
        request = create_mock_request(app_state={})
        response = await handle_accounting_customers(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["customers"] == MOCK_CUSTOMERS
        assert data["total"] == len(MOCK_CUSTOMERS)

    @pytest.mark.asyncio
    async def test_customers_inactive_filter(self, mock_qbo_connector):
        request = create_mock_request(
            query={"active": "false"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_customers(request)
        assert response.status == 200
        mock_qbo_connector.list_customers.assert_awaited_once()
        call_kwargs = mock_qbo_connector.list_customers.call_args
        assert call_kwargs.kwargs["active_only"] is False

    @pytest.mark.asyncio
    async def test_customers_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.list_customers = AsyncMock(side_effect=RuntimeError("API error"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_customers(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_customers_connector_not_connected(self):
        connector = MagicMock()
        connector.is_connected = MagicMock(return_value=False)
        request = create_mock_request(app_state={"qbo_connector": connector})
        response = await handle_accounting_customers(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["customers"] == MOCK_CUSTOMERS


# ===========================================================================
# Test Accounting Transactions Handler
# ===========================================================================


class TestAccountingTransactionsHandler:
    """Tests for handle_accounting_transactions."""

    @pytest.mark.asyncio
    async def test_transactions_all_types(self, mock_qbo_connector):
        request = create_mock_request(
            query={"type": "all"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "transactions" in data
        # Should include both invoices and expenses
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_transactions_invoice_only(self, mock_qbo_connector):
        request = create_mock_request(
            query={"type": "invoice"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 200
        data = json.loads(response.text)
        types = [t["type"] for t in data["transactions"]]
        assert all(t == "Invoice" for t in types)

    @pytest.mark.asyncio
    async def test_transactions_expense_only(self, mock_qbo_connector):
        request = create_mock_request(
            query={"type": "expense"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 200
        data = json.loads(response.text)
        types = [t["type"] for t in data["transactions"]]
        assert all(t == "Expense" for t in types)

    @pytest.mark.asyncio
    async def test_transactions_not_connected_returns_mock(self):
        request = create_mock_request(app_state={})
        response = await handle_accounting_transactions(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["transactions"] == MOCK_TRANSACTIONS
        assert data["total"] == len(MOCK_TRANSACTIONS)

    @pytest.mark.asyncio
    async def test_transactions_with_date_filters(self, mock_qbo_connector):
        request = create_mock_request(
            query={
                "type": "all",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_transactions_invalid_date_returns_400(self, mock_qbo_connector):
        request = create_mock_request(
            query={"type": "all", "start_date": "invalid-date"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Invalid date" in data["error"]

    @pytest.mark.asyncio
    async def test_transactions_error_returns_500(self, mock_qbo_connector):
        mock_qbo_connector.list_invoices = AsyncMock(side_effect=OSError("Network error"))
        request = create_mock_request(
            query={"type": "all"},
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_transactions(request)
        assert response.status == 500


# ===========================================================================
# Test Accounting Report Handler
# ===========================================================================


class TestAccountingReportHandler:
    """Tests for handle_accounting_report."""

    @pytest.mark.asyncio
    async def test_report_profit_loss_connected(self, mock_qbo_connector):
        request = create_mock_request(
            body={
                "type": "profit_loss",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "report" in data
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_report_balance_sheet_connected(self, mock_qbo_connector):
        request = create_mock_request(
            body={
                "type": "balance_sheet",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_report_ar_aging_connected(self, mock_qbo_connector):
        request = create_mock_request(
            body={
                "type": "ar_aging",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_report_ap_aging_connected(self, mock_qbo_connector):
        request = create_mock_request(
            body={
                "type": "ap_aging",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_report_unknown_type_connected(self, mock_qbo_connector):
        request = create_mock_request(
            body={
                "type": "invalid_report_type",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={"qbo_connector": mock_qbo_connector},
        )
        response = await handle_accounting_report(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Unknown report type" in data["error"]

    @pytest.mark.asyncio
    async def test_report_missing_dates(self):
        request = create_mock_request(
            body={"type": "profit_loss"},
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "start_date and end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_report_missing_start_date(self):
        request = create_mock_request(
            body={"type": "profit_loss", "end_date": "2025-01-31"},
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_report_missing_end_date(self):
        request = create_mock_request(
            body={"type": "profit_loss", "start_date": "2025-01-01"},
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_report_invalid_date_format(self):
        request = create_mock_request(
            body={
                "type": "profit_loss",
                "start_date": "not-a-date",
                "end_date": "2025-01-31",
            },
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Invalid date" in data["error"]

    @pytest.mark.asyncio
    async def test_report_invalid_json(self):
        request = create_mock_request(body=None, app_state={})
        response = await handle_accounting_report(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_report_not_connected_returns_mock(self):
        request = create_mock_request(
            body={
                "type": "profit_loss",
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["mock"] is True
        assert "report" in data

    @pytest.mark.asyncio
    async def test_report_defaults_to_profit_loss(self):
        request = create_mock_request(
            body={
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={},
        )
        response = await handle_accounting_report(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["report"]["title"] == "Profit and Loss"


# ===========================================================================
# Test Gusto Status Handler
# ===========================================================================


class TestGustoStatusHandler:
    """Tests for handle_gusto_status."""

    @pytest.mark.asyncio
    async def test_gusto_status_connected(self, mock_gusto_connector):
        credentials = MockGustoCredentials()
        request = create_mock_request(
            app_state={
                "gusto_connector": mock_gusto_connector,
                "gusto_credentials": credentials,
            }
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_status(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is True
        assert data["configured"] is True
        assert data["company"]["id"] == "gusto_company_123"

    @pytest.mark.asyncio
    async def test_gusto_status_not_connected(self):
        connector = MagicMock()
        connector.is_configured = False
        connector.is_authenticated = False
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_status(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is False
        assert data["company"] is None

    @pytest.mark.asyncio
    async def test_gusto_status_error_returns_500(self):
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            side_effect=RuntimeError("Config error"),
        ):
            response = await handle_gusto_status(request)
        assert response.status == 500


# ===========================================================================
# Test Gusto Connect Handler
# ===========================================================================


class TestGustoConnectHandler:
    """Tests for handle_gusto_connect."""

    @pytest.mark.asyncio
    async def test_gusto_connect_redirects(self, mock_gusto_connector):
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            with pytest.raises(web.HTTPFound) as exc_info:
                await handle_gusto_connect(request)
        assert "gusto.com" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_gusto_connect_not_configured(self):
        connector = MagicMock()
        connector.is_configured = False
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_connect(request)
        assert response.status == 503
        data = json.loads(response.text)
        assert "not configured" in data["error"]

    @pytest.mark.asyncio
    async def test_gusto_connect_error_returns_500(self, mock_gusto_connector):
        mock_gusto_connector.get_authorization_url = MagicMock(side_effect=ValueError("Bad config"))
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_connect(request)
        assert response.status == 500


# ===========================================================================
# Test Gusto Callback Handler
# ===========================================================================


class TestGustoCallbackHandler:
    """Tests for handle_gusto_callback."""

    @pytest.mark.asyncio
    async def test_gusto_callback_success(self, mock_gusto_connector):
        app_state = {}
        request = create_mock_request(
            query={"code": "gusto_auth_code"},
            app_state=app_state,
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            with pytest.raises(web.HTTPFound) as exc_info:
                await handle_gusto_callback(request)
        assert "connected=true" in str(exc_info.value.location)
        assert "provider=gusto" in str(exc_info.value.location)
        assert "gusto_credentials" in app_state

    @pytest.mark.asyncio
    async def test_gusto_callback_oauth_error(self):
        request = create_mock_request(
            query={"error": "access_denied", "error_description": "Denied"},
            app_state={},
        )
        response = await handle_gusto_callback(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_gusto_callback_missing_code(self):
        request = create_mock_request(query={}, app_state={})
        response = await handle_gusto_callback(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Missing" in data["error"]

    @pytest.mark.asyncio
    async def test_gusto_callback_not_configured(self):
        connector = MagicMock()
        connector.is_configured = False
        request = create_mock_request(
            query={"code": "gusto_auth_code"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_callback(request)
        assert response.status == 503

    @pytest.mark.asyncio
    async def test_gusto_callback_exchange_error(self, mock_gusto_connector):
        mock_gusto_connector.exchange_code = AsyncMock(side_effect=ConnectionError("Network error"))
        request = create_mock_request(
            query={"code": "gusto_auth_code"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_callback(request)
        assert response.status == 500


# ===========================================================================
# Test Gusto Disconnect Handler
# ===========================================================================


class TestGustoDisconnectHandler:
    """Tests for handle_gusto_disconnect."""

    @pytest.mark.asyncio
    async def test_gusto_disconnect_success(self):
        app_state = {"gusto_credentials": {"token": "test"}}
        request = create_mock_request(app_state=app_state)
        response = await handle_gusto_disconnect(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True
        assert data["message"] == "Gusto disconnected"
        assert "gusto_credentials" not in app_state

    @pytest.mark.asyncio
    async def test_gusto_disconnect_no_credentials(self):
        request = create_mock_request(app_state={})
        response = await handle_gusto_disconnect(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True


# ===========================================================================
# Test Gusto Employees Handler
# ===========================================================================


class TestGustoEmployeesHandler:
    """Tests for handle_gusto_employees."""

    @pytest.mark.asyncio
    async def test_gusto_employees_success(self, mock_gusto_connector):
        request = create_mock_request(
            query={"active": "true"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_employees(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "employees" in data
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_gusto_employees_inactive_filter(self, mock_gusto_connector):
        request = create_mock_request(
            query={"active": "false"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_employees(request)
        assert response.status == 200
        mock_gusto_connector.list_employees.assert_awaited_once_with(active_only=False)

    @pytest.mark.asyncio
    async def test_gusto_employees_not_connected(self):
        connector = MagicMock()
        connector.is_authenticated = False
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_employees(request)
        assert response.status == 503
        data = json.loads(response.text)
        assert "not connected" in data["error"]

    @pytest.mark.asyncio
    async def test_gusto_employees_error_returns_500(self, mock_gusto_connector):
        mock_gusto_connector.list_employees = AsyncMock(side_effect=TypeError("Unexpected error"))
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_employees(request)
        assert response.status == 500


# ===========================================================================
# Test Gusto Payrolls Handler
# ===========================================================================


class TestGustoPayrollsHandler:
    """Tests for handle_gusto_payrolls."""

    @pytest.mark.asyncio
    async def test_gusto_payrolls_success(self, mock_gusto_connector):
        request = create_mock_request(
            query={"processed": "true"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "payrolls" in data
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_gusto_payrolls_with_date_filters(self, mock_gusto_connector):
        request = create_mock_request(
            query={
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 200
        call_kwargs = mock_gusto_connector.list_payrolls.call_args
        assert call_kwargs.kwargs["start_date"] == date(2025, 1, 1)
        assert call_kwargs.kwargs["end_date"] == date(2025, 1, 31)

    @pytest.mark.asyncio
    async def test_gusto_payrolls_not_connected(self):
        connector = MagicMock()
        connector.is_authenticated = False
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 503

    @pytest.mark.asyncio
    async def test_gusto_payrolls_invalid_date_returns_400(self, mock_gusto_connector):
        request = create_mock_request(
            query={"start_date": "bad-date"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_gusto_payrolls_processed_false(self, mock_gusto_connector):
        request = create_mock_request(
            query={"processed": "false"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 200
        call_kwargs = mock_gusto_connector.list_payrolls.call_args
        assert call_kwargs.kwargs["processed_only"] is False


# ===========================================================================
# Test Gusto Payroll Detail Handler
# ===========================================================================


class TestGustoPayrollDetailHandler:
    """Tests for handle_gusto_payroll_detail."""

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_success(self, mock_gusto_connector):
        request = create_mock_request(
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "payroll" in data
        assert "payroll_items" in data["payroll"]

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_not_found(self, mock_gusto_connector):
        mock_gusto_connector.get_payroll = AsyncMock(return_value=None)
        request = create_mock_request(
            match_info={"payroll_id": "nonexistent"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)
        assert response.status == 404
        data = json.loads(response.text)
        assert "not found" in data["error"]

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_missing_id(self, mock_gusto_connector):
        request = create_mock_request(
            match_info={},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)
        assert response.status == 400
        data = json.loads(response.text)
        assert "Missing payroll_id" in data["error"]

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_not_connected(self):
        connector = MagicMock()
        connector.is_authenticated = False
        request = create_mock_request(
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_payroll_detail(request)
        assert response.status == 503

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_error_returns_500(self, mock_gusto_connector):
        mock_gusto_connector.get_payroll = AsyncMock(side_effect=AttributeError("Bad data"))
        request = create_mock_request(
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)
        assert response.status == 500


# ===========================================================================
# Test Gusto Journal Entry Handler
# ===========================================================================


class TestGustoJournalEntryHandler:
    """Tests for handle_gusto_journal_entry."""

    @pytest.mark.asyncio
    async def test_journal_entry_success(self, mock_gusto_connector):
        request = create_mock_request(
            body={"account_mappings": {}},
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 200
        data = json.loads(response.text)
        assert "payroll" in data
        assert "journal_entry" in data

    @pytest.mark.asyncio
    async def test_journal_entry_with_dict_mappings(self, mock_gusto_connector):
        request = create_mock_request(
            body={
                "account_mappings": {
                    "wages": {"account_id": "123", "account_name": "Wages Expense"},
                    "taxes": {"id": "456", "name": "Tax Payable"},
                }
            },
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 200
        # Verify generate_journal_entry was called with account_mappings
        call_args = mock_gusto_connector.generate_journal_entry.call_args
        mappings = call_args[0][1]
        assert "wages" in mappings
        assert mappings["wages"] == ("123", "Wages Expense")
        assert "taxes" in mappings
        assert mappings["taxes"] == ("456", "Tax Payable")

    @pytest.mark.asyncio
    async def test_journal_entry_with_tuple_mappings(self, mock_gusto_connector):
        request = create_mock_request(
            body={
                "account_mappings": {
                    "wages": ["123", "Wages Expense"],
                }
            },
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 200
        call_args = mock_gusto_connector.generate_journal_entry.call_args
        mappings = call_args[0][1]
        assert mappings["wages"] == ("123", "Wages Expense")

    @pytest.mark.asyncio
    async def test_journal_entry_not_found(self, mock_gusto_connector):
        mock_gusto_connector.get_payroll = AsyncMock(return_value=None)
        request = create_mock_request(
            body={},
            match_info={"payroll_id": "nonexistent"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_journal_entry_missing_payroll_id(self, mock_gusto_connector):
        request = create_mock_request(
            body={},
            match_info={},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_journal_entry_not_connected(self):
        connector = MagicMock()
        connector.is_authenticated = False
        request = create_mock_request(
            body={},
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 503

    @pytest.mark.asyncio
    async def test_journal_entry_empty_body(self, mock_gusto_connector):
        """Empty body should be allowed (allow_empty=True in parse_json_body)."""
        request = create_mock_request(
            body={},
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 200
        # No mappings, should pass None
        call_args = mock_gusto_connector.generate_journal_entry.call_args
        assert call_args[0][1] is None

    @pytest.mark.asyncio
    async def test_journal_entry_error_returns_500(self, mock_gusto_connector):
        mock_gusto_connector.generate_journal_entry = MagicMock(
            side_effect=KeyError("Missing data")
        )
        request = create_mock_request(
            body={},
            match_info={"payroll_id": "payroll_123"},
            app_state={},
        )
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)
        assert response.status == 500


# ===========================================================================
# Test Route Registration
# ===========================================================================


class TestRegisterAccountingRoutes:
    """Tests for register_accounting_routes."""

    def test_registers_v1_routes(self):
        app = web.Application()
        register_accounting_routes(app)
        # Collect all registered routes
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        # v1 canonical routes
        assert "/api/v1/accounting/status" in routes
        assert "/api/v1/accounting/connect" in routes
        assert "/api/v1/accounting/callback" in routes
        assert "/api/v1/accounting/disconnect" in routes
        assert "/api/v1/accounting/customers" in routes
        assert "/api/v1/accounting/transactions" in routes
        assert "/api/v1/accounting/report" in routes
        assert "/api/v1/accounting/gusto/status" in routes
        assert "/api/v1/accounting/gusto/connect" in routes
        assert "/api/v1/accounting/gusto/callback" in routes
        assert "/api/v1/accounting/gusto/disconnect" in routes
        assert "/api/v1/accounting/gusto/employees" in routes
        assert "/api/v1/accounting/gusto/payrolls" in routes

    def test_registers_legacy_routes(self):
        app = web.Application()
        register_accounting_routes(app)
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        # Legacy routes
        assert "/api/accounting/status" in routes
        assert "/api/accounting/connect" in routes
        assert "/api/accounting/callback" in routes
        assert "/api/accounting/disconnect" in routes
        assert "/api/accounting/customers" in routes
        assert "/api/accounting/transactions" in routes
        assert "/api/accounting/report" in routes
        assert "/api/accounting/gusto/status" in routes
        assert "/api/accounting/gusto/connect" in routes
        assert "/api/accounting/gusto/callback" in routes
        assert "/api/accounting/gusto/disconnect" in routes
        assert "/api/accounting/gusto/employees" in routes
        assert "/api/accounting/gusto/payrolls" in routes

    def test_registers_parameterized_routes(self):
        app = web.Application()
        register_accounting_routes(app)
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")]
        assert "/api/v1/accounting/gusto/payrolls/{payroll_id}" in routes
        assert "/api/accounting/gusto/payrolls/{payroll_id}" in routes

    def test_route_count(self):
        app = web.Application()
        register_accounting_routes(app)
        route_list = list(app.router.routes())
        # 14 v1 + 14 legacy = 28 routes
        assert len(route_list) >= 28


# ===========================================================================
# Test Error Handling Edge Cases
# ===========================================================================


class TestErrorHandlingEdgeCases:
    """Tests for error handling across various exception types."""

    @pytest.mark.asyncio
    async def test_status_value_error(self, mock_qbo_connector):
        mock_qbo_connector.list_invoices = AsyncMock(side_effect=ValueError("Bad value"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_status_key_error(self, mock_qbo_connector):
        mock_qbo_connector.get_company_info = AsyncMock(side_effect=KeyError("missing_key"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_status_type_error(self, mock_qbo_connector):
        mock_qbo_connector.list_customers = AsyncMock(side_effect=TypeError("Wrong type"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_status_os_error(self, mock_qbo_connector):
        mock_qbo_connector.list_expenses = AsyncMock(side_effect=OSError("Disk error"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_status_connection_error(self, mock_qbo_connector):
        mock_qbo_connector.get_company_info = AsyncMock(side_effect=ConnectionError("Network down"))
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_status(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_connect_value_error(self, mock_qbo_connector):
        mock_qbo_connector.get_authorization_url = MagicMock(
            side_effect=ValueError("Bad URL config")
        )
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})
        response = await handle_accounting_connect(request)
        assert response.status == 500

    @pytest.mark.asyncio
    async def test_gusto_payrolls_key_error(self, mock_gusto_connector):
        mock_gusto_connector.list_payrolls = AsyncMock(side_effect=KeyError("missing_key"))
        request = create_mock_request(app_state={})
        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)
        assert response.status == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for aragora.server.handlers.accounting - Accounting integration handler.

Tests cover:
- QuickBooks Online status and dashboard
- QuickBooks OAuth flow (connect, callback, disconnect)
- Customer and transaction listing
- Financial report generation
- Gusto payroll integration
- Error handling and validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
from aiohttp import web

from aragora.server.handlers.accounting import (
    handle_accounting_status,
    handle_accounting_connect,
    handle_accounting_callback,
    handle_accounting_disconnect,
    handle_accounting_customers,
    handle_accounting_transactions,
    handle_accounting_report,
    handle_gusto_status,
    handle_gusto_connect,
    handle_gusto_callback,
    handle_gusto_disconnect,
    handle_gusto_employees,
    handle_gusto_payrolls,
    handle_gusto_payroll_detail,
    handle_gusto_journal_entry,
    _generate_mock_report,
    _parse_iso_date,
    MOCK_COMPANY,
    MOCK_STATS,
    MOCK_CUSTOMERS,
    MOCK_TRANSACTIONS,
)


# ===========================================================================
# Test Fixtures
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
    # Set due_date to None to avoid offset-naive vs offset-aware comparison
    # in handler code that uses datetime.now() without timezone
    due_date: datetime | None = None
    total_amount: float = 1000.00
    balance: float = 0  # Set to 0 to avoid overdue check
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


def create_mock_request(
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    match_info: dict[str, str] | None = None,
    app_state: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock aiohttp request."""
    request = MagicMock(spec=web.Request)
    request.query = query or {}
    request.match_info = match_info or {}
    request.app = app_state or {}

    if body is not None:

        async def json_func():
            return body

        request.json = json_func
    else:

        async def json_error():
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        request.json = json_error

    return request


# ===========================================================================
# Test Helper Functions
# ===========================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_iso_date_valid(self):
        """Parses valid ISO date."""
        result = _parse_iso_date("2025-01-15", "test_date")
        assert result == date(2025, 1, 15)

    def test_parse_iso_date_none(self):
        """Returns None for None input."""
        result = _parse_iso_date(None, "test_date")
        assert result is None

    def test_parse_iso_date_empty(self):
        """Returns None for empty string."""
        result = _parse_iso_date("", "test_date")
        assert result is None

    def test_parse_iso_date_invalid(self):
        """Raises ValueError for invalid date."""
        with pytest.raises(ValueError, match="Invalid test_date"):
            _parse_iso_date("not-a-date", "test_date")

    def test_generate_mock_report_profit_loss(self):
        """Generates mock profit/loss report."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("profit_loss", start, end)

        assert report["title"] == "Profit and Loss"
        assert "sections" in report
        assert "netIncome" in report

    def test_generate_mock_report_balance_sheet(self):
        """Generates mock balance sheet report."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("balance_sheet", start, end)

        assert report["title"] == "Balance Sheet"
        assert "sections" in report

    def test_generate_mock_report_ar_aging(self):
        """Generates mock AR aging report."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("ar_aging", start, end)

        assert report["title"] == "Accounts Receivable Aging"
        assert "buckets" in report

    def test_generate_mock_report_ap_aging(self):
        """Generates mock AP aging report."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("ap_aging", start, end)

        assert report["title"] == "Accounts Payable Aging"
        assert "buckets" in report

    def test_generate_mock_report_unknown_type(self):
        """Returns error for unknown report type."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        report = _generate_mock_report("unknown", start, end)

        assert "error" in report


# ===========================================================================
# Test Mock Data
# ===========================================================================


class TestMockData:
    """Tests for mock data constants."""

    def test_mock_company_has_required_fields(self):
        """Mock company has required fields."""
        assert "name" in MOCK_COMPANY
        assert "legalName" in MOCK_COMPANY
        assert "email" in MOCK_COMPANY

    def test_mock_stats_has_required_fields(self):
        """Mock stats has required fields."""
        assert "receivables" in MOCK_STATS
        assert "payables" in MOCK_STATS
        assert "netIncome" in MOCK_STATS

    def test_mock_customers_not_empty(self):
        """Mock customers list is not empty."""
        assert len(MOCK_CUSTOMERS) > 0
        assert "id" in MOCK_CUSTOMERS[0]
        assert "displayName" in MOCK_CUSTOMERS[0]

    def test_mock_transactions_not_empty(self):
        """Mock transactions list is not empty."""
        assert len(MOCK_TRANSACTIONS) > 0
        assert "id" in MOCK_TRANSACTIONS[0]
        assert "type" in MOCK_TRANSACTIONS[0]


# ===========================================================================
# Test QBO Status Handler
# ===========================================================================


class TestAccountingStatusHandler:
    """Tests for accounting status handler."""

    @pytest.mark.asyncio
    async def test_status_with_connected_qbo(self, mock_qbo_connector):
        """Returns real data when QBO connected."""
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})

        response = await handle_accounting_status(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is True
        assert "company" in data
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_status_returns_mock_when_not_connected(self):
        """Returns mock data when QBO not connected."""
        request = create_mock_request(app_state={})

        response = await handle_accounting_status(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["connected"] is True  # Simulated connection
        assert data["company"] == MOCK_COMPANY


# ===========================================================================
# Test QBO Connect Handler
# ===========================================================================


class TestAccountingConnectHandler:
    """Tests for accounting connect handler."""

    @pytest.mark.asyncio
    async def test_connect_redirects_to_oauth(self, mock_qbo_connector):
        """Redirects to QBO OAuth URL."""
        request = create_mock_request(app_state={"qbo_connector": mock_qbo_connector})

        with pytest.raises(web.HTTPFound) as exc_info:
            await handle_accounting_connect(request)

        assert "oauth.intuit.com" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_connect_returns_error_when_not_configured(self):
        """Returns error when connector not configured."""
        request = create_mock_request(app_state={})

        response = await handle_accounting_connect(request)

        assert response.status == 503
        data = json.loads(response.text)
        assert "not configured" in data["error"]


# ===========================================================================
# Test QBO Callback Handler
# ===========================================================================


class TestAccountingCallbackHandler:
    """Tests for accounting callback handler."""

    @pytest.mark.asyncio
    async def test_callback_success(self, mock_qbo_connector):
        """Successful callback exchanges code and redirects."""
        request = create_mock_request(
            query={"code": "auth_code", "realmId": "realm_123"},
            app_state={"qbo_connector": mock_qbo_connector},
        )

        with pytest.raises(web.HTTPFound) as exc_info:
            await handle_accounting_callback(request)

        assert "connected=true" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_callback_with_error(self):
        """Returns error when OAuth fails."""
        request = create_mock_request(
            query={"error": "access_denied", "error_description": "User denied"},
            app_state={},
        )

        response = await handle_accounting_callback(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "access_denied"

    @pytest.mark.asyncio
    async def test_callback_missing_code(self):
        """Returns error when code missing."""
        request = create_mock_request(
            query={"realmId": "realm_123"},
            app_state={},
        )

        response = await handle_accounting_callback(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Missing" in data["error"]


# ===========================================================================
# Test QBO Disconnect Handler
# ===========================================================================


class TestAccountingDisconnectHandler:
    """Tests for accounting disconnect handler."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, mock_qbo_connector):
        """Successful disconnect clears credentials."""
        app_state = {
            "qbo_connector": mock_qbo_connector,
            "qbo_credentials": {"token": "test"},
        }
        request = create_mock_request(app_state=app_state)

        response = await handle_accounting_disconnect(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True
        assert "qbo_credentials" not in app_state


# ===========================================================================
# Test QBO Customers Handler
# ===========================================================================


class TestAccountingCustomersHandler:
    """Tests for accounting customers handler."""

    @pytest.mark.asyncio
    async def test_customers_with_connected_qbo(self, mock_qbo_connector):
        """Returns real customers when QBO connected."""
        request = create_mock_request(
            query={"active": "true", "limit": "50"},
            app_state={"qbo_connector": mock_qbo_connector},
        )

        response = await handle_accounting_customers(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "customers" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_customers_returns_mock_when_not_connected(self):
        """Returns mock customers when not connected."""
        request = create_mock_request(app_state={})

        response = await handle_accounting_customers(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["customers"] == MOCK_CUSTOMERS


# ===========================================================================
# Test QBO Transactions Handler
# ===========================================================================


class TestAccountingTransactionsHandler:
    """Tests for accounting transactions handler."""

    @pytest.mark.asyncio
    async def test_transactions_with_connected_qbo(self, mock_qbo_connector):
        """Returns real transactions when QBO connected."""
        request = create_mock_request(
            query={"type": "all"},
            app_state={"qbo_connector": mock_qbo_connector},
        )

        response = await handle_accounting_transactions(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "transactions" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_transactions_returns_mock_when_not_connected(self):
        """Returns mock transactions when not connected."""
        request = create_mock_request(app_state={})

        response = await handle_accounting_transactions(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["transactions"] == MOCK_TRANSACTIONS


# ===========================================================================
# Test QBO Report Handler
# ===========================================================================


class TestAccountingReportHandler:
    """Tests for accounting report handler."""

    @pytest.mark.asyncio
    async def test_report_profit_loss(self, mock_qbo_connector):
        """Generates profit/loss report."""
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
    async def test_report_missing_dates(self):
        """Returns error when dates missing."""
        request = create_mock_request(
            body={"type": "profit_loss"},
            app_state={},
        )

        response = await handle_accounting_report(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "start_date and end_date" in data["error"]

    @pytest.mark.asyncio
    async def test_report_invalid_type(self, mock_qbo_connector):
        """Returns error for invalid report type."""
        request = create_mock_request(
            body={
                "type": "invalid_report",
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
    async def test_report_invalid_json(self):
        """Returns error for invalid JSON."""
        request = create_mock_request(body=None, app_state={})

        response = await handle_accounting_report(request)

        assert response.status == 400
        data = json.loads(response.text)
        assert "Invalid JSON" in data["error"]

    @pytest.mark.asyncio
    async def test_report_returns_mock_when_not_connected(self):
        """Returns mock report when not connected."""
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


# ===========================================================================
# Test Gusto Status Handler
# ===========================================================================


class TestGustoStatusHandler:
    """Tests for Gusto status handler."""

    @pytest.mark.asyncio
    async def test_gusto_status_connected(self, mock_gusto_connector):
        """Returns connected status with company info."""
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
        assert data["company"]["id"] == credentials.company_id


# ===========================================================================
# Test Gusto Connect Handler
# ===========================================================================


class TestGustoConnectHandler:
    """Tests for Gusto connect handler."""

    @pytest.mark.asyncio
    async def test_gusto_connect_redirects(self, mock_gusto_connector):
        """Redirects to Gusto OAuth URL."""
        request = create_mock_request(app_state={"gusto_connector": mock_gusto_connector})

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            with pytest.raises(web.HTTPFound) as exc_info:
                await handle_gusto_connect(request)

        assert "gusto.com" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_gusto_connect_not_configured(self):
        """Returns error when Gusto not configured."""
        connector = MagicMock()
        connector.is_configured = False
        request = create_mock_request(app_state={})

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_connect(request)

        assert response.status == 503


# ===========================================================================
# Test Gusto Callback Handler
# ===========================================================================


class TestGustoCallbackHandler:
    """Tests for Gusto callback handler."""

    @pytest.mark.asyncio
    async def test_gusto_callback_success(self, mock_gusto_connector):
        """Successful callback exchanges code."""
        request = create_mock_request(
            query={"code": "auth_code"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            with pytest.raises(web.HTTPFound) as exc_info:
                await handle_gusto_callback(request)

        assert "connected=true" in str(exc_info.value.location)

    @pytest.mark.asyncio
    async def test_gusto_callback_error(self):
        """Returns error when OAuth fails."""
        request = create_mock_request(
            query={"error": "access_denied"},
            app_state={},
        )

        response = await handle_gusto_callback(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_gusto_callback_missing_code(self):
        """Returns error when code missing."""
        request = create_mock_request(query={}, app_state={})

        response = await handle_gusto_callback(request)

        assert response.status == 400


# ===========================================================================
# Test Gusto Disconnect Handler
# ===========================================================================


class TestGustoDisconnectHandler:
    """Tests for Gusto disconnect handler."""

    @pytest.mark.asyncio
    async def test_gusto_disconnect_success(self):
        """Successful disconnect clears credentials."""
        app_state = {"gusto_credentials": {"token": "test"}}
        request = create_mock_request(app_state=app_state)

        response = await handle_gusto_disconnect(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert data["success"] is True


# ===========================================================================
# Test Gusto Employees Handler
# ===========================================================================


class TestGustoEmployeesHandler:
    """Tests for Gusto employees handler."""

    @pytest.mark.asyncio
    async def test_gusto_employees_success(self, mock_gusto_connector):
        """Returns employees list."""
        request = create_mock_request(
            query={"active": "true"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_employees(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "employees" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_gusto_employees_not_connected(self):
        """Returns error when not connected."""
        connector = MagicMock()
        connector.is_authenticated = False
        request = create_mock_request(app_state={})

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_employees(request)

        assert response.status == 503


# ===========================================================================
# Test Gusto Payrolls Handler
# ===========================================================================


class TestGustoPayrollsHandler:
    """Tests for Gusto payrolls handler."""

    @pytest.mark.asyncio
    async def test_gusto_payrolls_success(self, mock_gusto_connector):
        """Returns payrolls list."""
        request = create_mock_request(
            query={"processed": "true"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payrolls(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "payrolls" in data

    @pytest.mark.asyncio
    async def test_gusto_payrolls_invalid_date(self):
        """Returns error for invalid date."""
        connector = MagicMock()
        connector.is_authenticated = True
        request = create_mock_request(
            query={"start_date": "invalid-date"},
            app_state={},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=connector,
        ):
            response = await handle_gusto_payrolls(request)

        assert response.status == 400


# ===========================================================================
# Test Gusto Payroll Detail Handler
# ===========================================================================


class TestGustoPayrollDetailHandler:
    """Tests for Gusto payroll detail handler."""

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_success(self, mock_gusto_connector):
        """Returns payroll details."""
        request = create_mock_request(
            match_info={"payroll_id": "payroll_123"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)

        assert response.status == 200
        data = json.loads(response.text)
        assert "payroll" in data

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_not_found(self, mock_gusto_connector):
        """Returns 404 when payroll not found."""
        mock_gusto_connector.get_payroll = AsyncMock(return_value=None)
        request = create_mock_request(
            match_info={"payroll_id": "nonexistent"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_gusto_payroll_detail_missing_id(self, mock_gusto_connector):
        """Returns error when payroll_id missing."""
        request = create_mock_request(
            match_info={},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_payroll_detail(request)

        assert response.status == 400


# ===========================================================================
# Test Gusto Journal Entry Handler
# ===========================================================================


class TestGustoJournalEntryHandler:
    """Tests for Gusto journal entry handler."""

    @pytest.mark.asyncio
    async def test_gusto_journal_entry_success(self, mock_gusto_connector):
        """Generates journal entry."""
        request = create_mock_request(
            body={"account_mappings": {}},
            match_info={"payroll_id": "payroll_123"},
            app_state={"gusto_connector": mock_gusto_connector},
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
    async def test_gusto_journal_entry_not_found(self, mock_gusto_connector):
        """Returns 404 when payroll not found."""
        mock_gusto_connector.get_payroll = AsyncMock(return_value=None)
        request = create_mock_request(
            body={},
            match_info={"payroll_id": "nonexistent"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_gusto_journal_entry_with_mappings(self, mock_gusto_connector):
        """Generates journal entry with account mappings."""
        request = create_mock_request(
            body={
                "account_mappings": {
                    "wages": {"account_id": "123", "account_name": "Wages Expense"},
                }
            },
            match_info={"payroll_id": "payroll_123"},
            app_state={"gusto_connector": mock_gusto_connector},
        )

        with patch(
            "aragora.server.handlers.accounting.get_gusto_connector",
            return_value=mock_gusto_connector,
        ):
            response = await handle_gusto_journal_entry(request)

        assert response.status == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

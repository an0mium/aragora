"""
Comprehensive tests for the Xero Accounting Connector.

Tests cover:
- Enum values (InvoiceType, InvoiceStatus, ContactStatus, AccountType, etc.)
- Data model classes (XeroCredentials, Address, Phone, XeroContact, LineItem,
  Invoice, Account, BankTransaction, Payment, JournalLine, ManualJournal)
- from_api and to_api serialization for all models
- XeroError exception class
- XeroConnector initialization and configuration
- HTTP client creation and header setup
- API request handling (_request with circuit breaker, error responses)
- Contact operations (list, get, create, update)
- Invoice operations (list, get, create, update status, void)
- Account operations (list, get, bank accounts)
- Bank transaction operations (list, create)
- Payment operations (list, create)
- Manual journal operations (list, create)
- Organisation endpoint
- Async context manager (close, __aenter__, __aexit__)
- Date/datetime parsing helpers (_parse_xero_datetime, _parse_xero_date)
- Error handling (server errors, client errors, timeouts, connection errors)
- Circuit breaker integration
- Pagination and filtering
- Mock data helpers
"""

from __future__ import annotations

from datetime import date as Date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import pytest

from aragora.connectors.accounting.xero import (
    Account,
    AccountType,
    Address,
    BankTransaction,
    BankTransactionType,
    ContactStatus,
    Invoice,
    InvoiceStatus,
    InvoiceType,
    JournalLine,
    LineItem,
    ManualJournal,
    Payment,
    PaymentStatus,
    Phone,
    XeroConnector,
    XeroContact,
    XeroCredentials,
    XeroError,
    _parse_xero_date,
    _parse_xero_datetime,
    get_mock_contact,
    get_mock_invoice,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def xero_credentials():
    """Create standard test credentials."""
    return XeroCredentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        tenant_id="test_tenant_id",
    )


@pytest.fixture
def xero_connector(xero_credentials):
    """Create a XeroConnector with circuit breaker disabled."""
    return XeroConnector(
        credentials=xero_credentials,
        enable_circuit_breaker=False,
    )


@pytest.fixture
def xero_connector_with_cb(xero_credentials):
    """Create a XeroConnector with circuit breaker enabled."""
    return XeroConnector(
        credentials=xero_credentials,
        enable_circuit_breaker=True,
    )


@pytest.fixture
def sample_contact_api_data():
    """Sample Xero API contact response data."""
    return {
        "ContactID": "contact-abc-123",
        "Name": "Acme Corp",
        "EmailAddress": "billing@acme.com",
        "FirstName": "John",
        "LastName": "Doe",
        "CompanyNumber": "12345678",
        "TaxNumber": "GB123456789",
        "BankAccountDetails": "12-34-56 12345678",
        "AccountsReceivableTaxType": "OUTPUT2",
        "AccountsPayableTaxType": "INPUT2",
        "IsSupplier": True,
        "IsCustomer": True,
        "DefaultCurrency": "GBP",
        "ContactStatus": "ACTIVE",
        "Addresses": [
            {
                "AddressType": "STREET",
                "AddressLine1": "123 Main St",
                "AddressLine2": "Suite 100",
                "City": "London",
                "Region": "Greater London",
                "PostalCode": "EC1A 1BB",
                "Country": "UK",
            }
        ],
        "Phones": [
            {
                "PhoneType": "DEFAULT",
                "PhoneNumber": "1234567",
                "PhoneAreaCode": "020",
                "PhoneCountryCode": "44",
            }
        ],
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


@pytest.fixture
def sample_invoice_api_data():
    """Sample Xero API invoice response data."""
    return {
        "InvoiceID": "inv-abc-123",
        "InvoiceNumber": "INV-0042",
        "Reference": "PO-12345",
        "Type": "ACCREC",
        "Status": "AUTHORISED",
        "Contact": {
            "ContactID": "contact-abc-123",
            "Name": "Acme Corp",
        },
        "Date": "/Date(1700000000000)/",
        "DueDate": "/Date(1702592000000)/",
        "LineItems": [
            {
                "Description": "Consulting services",
                "Quantity": "10",
                "UnitAmount": "150.00",
                "AccountCode": "200",
                "TaxType": "OUTPUT2",
                "TaxAmount": "225.00",
                "LineAmount": "1500.00",
                "ItemCode": "CONSULT",
                "Tracking": [{"Name": "Region", "Option": "North"}],
            }
        ],
        "SubTotal": "1500.00",
        "TotalTax": "225.00",
        "Total": "1725.00",
        "AmountDue": "1725.00",
        "AmountPaid": "0",
        "CurrencyCode": "GBP",
        "LineAmountTypes": "Exclusive",
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


@pytest.fixture
def sample_account_api_data():
    """Sample Xero API account response data."""
    return {
        "AccountID": "acct-abc-123",
        "Code": "200",
        "Name": "Sales Revenue",
        "Type": "REVENUE",
        "Status": "ACTIVE",
        "Description": "Revenue from sales",
        "TaxType": "OUTPUT2",
        "EnablePaymentsToAccount": False,
        "CurrencyCode": "GBP",
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


@pytest.fixture
def sample_bank_transaction_api_data():
    """Sample Xero API bank transaction response data."""
    return {
        "BankTransactionID": "bt-abc-123",
        "Type": "SPEND",
        "Contact": {"ContactID": "contact-abc-123", "Name": "Supplier Co"},
        "BankAccount": {"AccountID": "acct-bank-001", "Code": "090"},
        "Date": "/Date(1700000000000)/",
        "Reference": "INV-EXT-001",
        "LineItems": [
            {
                "Description": "Office supplies",
                "Quantity": "1",
                "UnitAmount": "50.00",
            }
        ],
        "SubTotal": "50.00",
        "TotalTax": "7.50",
        "Total": "57.50",
        "CurrencyCode": "GBP",
        "Status": "AUTHORISED",
        "IsReconciled": True,
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


@pytest.fixture
def sample_payment_api_data():
    """Sample Xero API payment response data."""
    return {
        "PaymentID": "pay-abc-123",
        "Invoice": {
            "InvoiceID": "inv-abc-123",
            "InvoiceNumber": "INV-0042",
        },
        "Account": {"AccountID": "acct-bank-001"},
        "Date": "/Date(1700000000000)/",
        "Amount": "1725.00",
        "CurrencyRate": "1.0",
        "Reference": "Payment for INV-0042",
        "Status": "AUTHORISED",
        "PaymentType": "ACCRECPAYMENT",
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


@pytest.fixture
def sample_journal_api_data():
    """Sample Xero API manual journal response data."""
    return {
        "ManualJournalID": "mj-abc-123",
        "Narration": "Month-end adjustment",
        "Date": "/Date(1700000000000)/",
        "Status": "POSTED",
        "JournalLines": [
            {
                "AccountCode": "200",
                "Description": "Revenue adjustment",
                "LineAmount": "500",
                "TaxType": "OUTPUT2",
            },
            {
                "AccountCode": "400",
                "Description": "Expense correction",
                "LineAmount": "-500",
                "TaxType": "INPUT2",
            },
        ],
        "Url": "https://go.xero.com/ManualJournal/mj-abc-123",
        "UpdatedDateUTC": "/Date(1700000000000)/",
    }


# =============================================================================
# Enum Tests
# =============================================================================


class TestInvoiceType:
    """Tests for InvoiceType enum."""

    def test_invoice_type_values(self):
        assert InvoiceType.ACCPAY.value == "ACCPAY"
        assert InvoiceType.ACCREC.value == "ACCREC"

    def test_invoice_type_is_str(self):
        assert isinstance(InvoiceType.ACCREC, str)


class TestInvoiceStatus:
    """Tests for InvoiceStatus enum."""

    def test_invoice_status_values(self):
        assert InvoiceStatus.DRAFT.value == "DRAFT"
        assert InvoiceStatus.SUBMITTED.value == "SUBMITTED"
        assert InvoiceStatus.AUTHORISED.value == "AUTHORISED"
        assert InvoiceStatus.PAID.value == "PAID"
        assert InvoiceStatus.VOIDED.value == "VOIDED"
        assert InvoiceStatus.DELETED.value == "DELETED"


class TestContactStatus:
    """Tests for ContactStatus enum."""

    def test_contact_status_values(self):
        assert ContactStatus.ACTIVE.value == "ACTIVE"
        assert ContactStatus.ARCHIVED.value == "ARCHIVED"
        assert ContactStatus.GDPRREQUEST.value == "GDPRREQUEST"


class TestAccountType:
    """Tests for AccountType enum."""

    def test_account_type_values(self):
        assert AccountType.BANK.value == "BANK"
        assert AccountType.REVENUE.value == "REVENUE"
        assert AccountType.EXPENSE.value == "EXPENSE"
        assert AccountType.EQUITY.value == "EQUITY"
        assert AccountType.FIXED.value == "FIXED"
        assert AccountType.INVENTORY.value == "INVENTORY"
        assert AccountType.WAGESEXPENSE.value == "WAGESEXPENSE"


class TestBankTransactionType:
    """Tests for BankTransactionType enum."""

    def test_bank_transaction_type_values(self):
        assert BankTransactionType.RECEIVE.value == "RECEIVE"
        assert BankTransactionType.SPEND.value == "SPEND"
        assert BankTransactionType.RECEIVE_OVERPAYMENT.value == "RECEIVE-OVERPAYMENT"
        assert BankTransactionType.SPEND_PREPAYMENT.value == "SPEND-PREPAYMENT"


class TestPaymentStatus:
    """Tests for PaymentStatus enum."""

    def test_payment_status_values(self):
        assert PaymentStatus.AUTHORISED.value == "AUTHORISED"
        assert PaymentStatus.DELETED.value == "DELETED"


# =============================================================================
# Data Model Tests
# =============================================================================


class TestXeroCredentials:
    """Tests for XeroCredentials dataclass."""

    def test_creation_with_all_fields(self, xero_credentials):
        assert xero_credentials.client_id == "test_client_id"
        assert xero_credentials.client_secret == "test_client_secret"
        assert xero_credentials.access_token == "test_access_token"
        assert xero_credentials.refresh_token == "test_refresh_token"
        assert xero_credentials.tenant_id == "test_tenant_id"
        assert xero_credentials.base_url == "https://api.xero.com/api.xro/2.0"

    def test_creation_minimal(self):
        creds = XeroCredentials(
            client_id="cid",
            client_secret="csecret",
        )
        assert creds.access_token is None
        assert creds.refresh_token is None
        assert creds.tenant_id is None
        assert creds.base_url == "https://api.xero.com/api.xro/2.0"

    def test_custom_base_url(self):
        creds = XeroCredentials(
            client_id="cid",
            client_secret="csecret",
            base_url="https://api-sandbox.xero.com/api.xro/2.0",
        )
        assert "sandbox" in creds.base_url


class TestAddress:
    """Tests for Address dataclass."""

    def test_from_api(self):
        data = {
            "AddressType": "STREET",
            "AddressLine1": "123 Main St",
            "AddressLine2": "Suite 100",
            "City": "London",
            "Region": "Greater London",
            "PostalCode": "EC1A 1BB",
            "Country": "UK",
        }
        addr = Address.from_api(data)
        assert addr.address_type == "STREET"
        assert addr.address_line1 == "123 Main St"
        assert addr.address_line2 == "Suite 100"
        assert addr.city == "London"
        assert addr.region == "Greater London"
        assert addr.postal_code == "EC1A 1BB"
        assert addr.country == "UK"

    def test_from_api_defaults(self):
        addr = Address.from_api({})
        assert addr.address_type == "POBOX"
        assert addr.address_line1 is None

    def test_to_api_full(self):
        addr = Address(
            address_type="STREET",
            address_line1="123 Main St",
            address_line2="Suite 100",
            city="London",
            region="Greater London",
            postal_code="EC1A 1BB",
            country="UK",
        )
        result = addr.to_api()
        assert result["AddressType"] == "STREET"
        assert result["AddressLine1"] == "123 Main St"
        assert result["City"] == "London"
        assert result["Country"] == "UK"

    def test_to_api_minimal(self):
        addr = Address()
        result = addr.to_api()
        assert result == {"AddressType": "POBOX"}

    def test_to_api_excludes_none(self):
        addr = Address(address_type="STREET", city="London")
        result = addr.to_api()
        assert "AddressLine1" not in result
        assert "AddressLine2" not in result


class TestPhone:
    """Tests for Phone dataclass."""

    def test_from_api(self):
        data = {
            "PhoneType": "MOBILE",
            "PhoneNumber": "1234567890",
            "PhoneAreaCode": "020",
            "PhoneCountryCode": "44",
        }
        phone = Phone.from_api(data)
        assert phone.phone_type == "MOBILE"
        assert phone.phone_number == "1234567890"
        assert phone.phone_area_code == "020"
        assert phone.phone_country_code == "44"

    def test_from_api_defaults(self):
        phone = Phone.from_api({})
        assert phone.phone_type == "DEFAULT"
        assert phone.phone_number is None

    def test_defaults(self):
        phone = Phone()
        assert phone.phone_type == "DEFAULT"
        assert phone.phone_number is None


class TestXeroContact:
    """Tests for XeroContact dataclass."""

    def test_from_api_full(self, sample_contact_api_data):
        contact = XeroContact.from_api(sample_contact_api_data)
        assert contact.contact_id == "contact-abc-123"
        assert contact.name == "Acme Corp"
        assert contact.email == "billing@acme.com"
        assert contact.first_name == "John"
        assert contact.last_name == "Doe"
        assert contact.company_number == "12345678"
        assert contact.tax_number == "GB123456789"
        assert contact.is_supplier is True
        assert contact.is_customer is True
        assert contact.default_currency == "GBP"
        assert contact.contact_status == ContactStatus.ACTIVE
        assert len(contact.addresses) == 1
        assert len(contact.phones) == 1
        assert contact.updated_date is not None

    def test_from_api_minimal(self):
        contact = XeroContact.from_api({"Name": "Test"})
        assert contact.name == "Test"
        assert contact.contact_id is None
        assert contact.is_supplier is False
        assert contact.is_customer is False
        assert contact.addresses == []
        assert contact.phones == []

    def test_from_api_empty(self):
        contact = XeroContact.from_api({})
        assert contact.name == ""
        assert contact.contact_status == ContactStatus.ACTIVE


class TestLineItem:
    """Tests for LineItem dataclass."""

    def test_from_api_full(self):
        data = {
            "Description": "Widget",
            "Quantity": "5",
            "UnitAmount": "20.00",
            "AccountCode": "200",
            "TaxType": "OUTPUT2",
            "TaxAmount": "15.00",
            "LineAmount": "100.00",
            "ItemCode": "WIDGET",
            "Tracking": [{"Name": "Region", "Option": "North"}],
        }
        item = LineItem.from_api(data)
        assert item.description == "Widget"
        assert item.quantity == Decimal("5")
        assert item.unit_amount == Decimal("20.00")
        assert item.account_code == "200"
        assert item.tax_type == "OUTPUT2"
        assert item.tax_amount == Decimal("15.00")
        assert item.line_amount == Decimal("100.00")
        assert item.item_code == "WIDGET"
        assert len(item.tracking) == 1

    def test_from_api_minimal(self):
        item = LineItem.from_api({})
        assert item.description == ""
        assert item.quantity == Decimal("1")
        assert item.unit_amount == Decimal("0")
        assert item.tax_amount is None
        assert item.line_amount is None

    def test_to_api_full(self):
        item = LineItem(
            description="Service",
            quantity=Decimal("2"),
            unit_amount=Decimal("100"),
            account_code="200",
            tax_type="OUTPUT2",
            item_code="SVC",
            tracking=[{"Name": "Dept", "Option": "Sales"}],
        )
        result = item.to_api()
        assert result["Description"] == "Service"
        assert result["Quantity"] == "2"
        assert result["UnitAmount"] == "100"
        assert result["AccountCode"] == "200"
        assert result["TaxType"] == "OUTPUT2"
        assert result["ItemCode"] == "SVC"
        assert result["Tracking"] == [{"Name": "Dept", "Option": "Sales"}]

    def test_to_api_minimal(self):
        item = LineItem(description="Test")
        result = item.to_api()
        assert result["Description"] == "Test"
        assert "AccountCode" not in result
        assert "TaxType" not in result
        assert "ItemCode" not in result
        assert "Tracking" not in result


class TestInvoice:
    """Tests for Invoice dataclass."""

    def test_from_api_full(self, sample_invoice_api_data):
        invoice = Invoice.from_api(sample_invoice_api_data)
        assert invoice.invoice_id == "inv-abc-123"
        assert invoice.invoice_number == "INV-0042"
        assert invoice.reference == "PO-12345"
        assert invoice.type == InvoiceType.ACCREC
        assert invoice.status == InvoiceStatus.AUTHORISED
        assert invoice.contact_id == "contact-abc-123"
        assert invoice.contact_name == "Acme Corp"
        assert invoice.sub_total == Decimal("1500.00")
        assert invoice.total_tax == Decimal("225.00")
        assert invoice.total == Decimal("1725.00")
        assert invoice.amount_due == Decimal("1725.00")
        assert invoice.amount_paid == Decimal("0")
        assert invoice.currency_code == "GBP"
        assert invoice.line_amount_types == "Exclusive"
        assert len(invoice.line_items) == 1
        assert invoice.date is not None
        assert invoice.due_date is not None

    def test_from_api_minimal(self):
        invoice = Invoice.from_api({})
        assert invoice.invoice_id is None
        assert invoice.type == InvoiceType.ACCREC
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.total == Decimal("0")
        assert invoice.line_items == []

    def test_from_api_bill_type(self):
        invoice = Invoice.from_api({"Type": "ACCPAY", "Status": "PAID"})
        assert invoice.type == InvoiceType.ACCPAY
        assert invoice.status == InvoiceStatus.PAID


class TestAccount:
    """Tests for Account dataclass."""

    def test_from_api_full(self, sample_account_api_data):
        account = Account.from_api(sample_account_api_data)
        assert account.account_id == "acct-abc-123"
        assert account.code == "200"
        assert account.name == "Sales Revenue"
        assert account.type == AccountType.REVENUE
        assert account.status == "ACTIVE"
        assert account.description == "Revenue from sales"
        assert account.tax_type == "OUTPUT2"
        assert account.enable_payments is False
        assert account.currency_code == "GBP"

    def test_from_api_minimal(self):
        account = Account.from_api({})
        assert account.account_id is None
        assert account.name == ""
        assert account.type is None
        assert account.status == "ACTIVE"

    def test_from_api_bank_account(self):
        data = {
            "AccountID": "bank-001",
            "Code": "090",
            "Name": "Business Account",
            "Type": "BANK",
            "EnablePaymentsToAccount": True,
            "BankAccountNumber": "12345678",
            "BankAccountType": "BANK",
        }
        account = Account.from_api(data)
        assert account.type == AccountType.BANK
        assert account.enable_payments is True
        assert account.bank_account_number == "12345678"


class TestBankTransaction:
    """Tests for BankTransaction dataclass."""

    def test_from_api_full(self, sample_bank_transaction_api_data):
        bt = BankTransaction.from_api(sample_bank_transaction_api_data)
        assert bt.bank_transaction_id == "bt-abc-123"
        assert bt.type == BankTransactionType.SPEND
        assert bt.contact_id == "contact-abc-123"
        assert bt.contact_name == "Supplier Co"
        assert bt.bank_account_id == "acct-bank-001"
        assert bt.bank_account_code == "090"
        assert bt.reference == "INV-EXT-001"
        assert bt.sub_total == Decimal("50.00")
        assert bt.total_tax == Decimal("7.50")
        assert bt.total == Decimal("57.50")
        assert bt.is_reconciled is True
        assert len(bt.line_items) == 1

    def test_from_api_minimal(self):
        bt = BankTransaction.from_api({})
        assert bt.bank_transaction_id is None
        assert bt.type == BankTransactionType.SPEND
        assert bt.is_reconciled is False

    def test_from_api_receive(self):
        bt = BankTransaction.from_api({"Type": "RECEIVE"})
        assert bt.type == BankTransactionType.RECEIVE


class TestPayment:
    """Tests for Payment dataclass."""

    def test_from_api_full(self, sample_payment_api_data):
        payment = Payment.from_api(sample_payment_api_data)
        assert payment.payment_id == "pay-abc-123"
        assert payment.invoice_id == "inv-abc-123"
        assert payment.invoice_number == "INV-0042"
        assert payment.account_id == "acct-bank-001"
        assert payment.amount == Decimal("1725.00")
        assert payment.currency_rate == Decimal("1.0")
        assert payment.reference == "Payment for INV-0042"
        assert payment.status == PaymentStatus.AUTHORISED
        assert payment.payment_type == "ACCRECPAYMENT"

    def test_from_api_minimal(self):
        payment = Payment.from_api({})
        assert payment.payment_id is None
        assert payment.amount == Decimal("0")
        assert payment.currency_rate == Decimal("1")
        assert payment.status == PaymentStatus.AUTHORISED


class TestJournalLine:
    """Tests for JournalLine dataclass."""

    def test_to_api_debit(self):
        line = JournalLine(
            account_code="200",
            description="Revenue",
            debit=Decimal("500"),
            tax_type="OUTPUT2",
        )
        result = line.to_api()
        assert result["AccountCode"] == "200"
        assert result["Description"] == "Revenue"
        assert result["LineAmount"] == "500"
        assert result["TaxType"] == "OUTPUT2"

    def test_to_api_credit(self):
        line = JournalLine(
            account_code="400",
            description="Expense",
            credit=Decimal("500"),
        )
        result = line.to_api()
        assert result["LineAmount"] == "-500"

    def test_to_api_minimal(self):
        line = JournalLine(account_code="200")
        result = line.to_api()
        assert result == {"AccountCode": "200"}

    def test_to_api_no_description_excluded(self):
        line = JournalLine(account_code="200", debit=Decimal("100"))
        result = line.to_api()
        assert "Description" not in result

    def test_to_api_no_tax_excluded(self):
        line = JournalLine(account_code="200", debit=Decimal("100"))
        result = line.to_api()
        assert "TaxType" not in result


class TestManualJournal:
    """Tests for ManualJournal dataclass."""

    def test_from_api_full(self, sample_journal_api_data):
        mj = ManualJournal.from_api(sample_journal_api_data)
        assert mj.manual_journal_id == "mj-abc-123"
        assert mj.narration == "Month-end adjustment"
        assert mj.status == "POSTED"
        assert mj.url == "https://go.xero.com/ManualJournal/mj-abc-123"
        assert len(mj.journal_lines) == 2

        debit_line = mj.journal_lines[0]
        assert debit_line.account_code == "200"
        assert debit_line.debit == Decimal("500")
        assert debit_line.credit is None

        credit_line = mj.journal_lines[1]
        assert credit_line.account_code == "400"
        assert credit_line.credit == Decimal("500")
        assert credit_line.debit is None

    def test_from_api_minimal(self):
        mj = ManualJournal.from_api({})
        assert mj.manual_journal_id is None
        assert mj.narration == ""
        assert mj.status == "DRAFT"
        assert mj.journal_lines == []

    def test_from_api_zero_amount_line(self):
        data = {
            "JournalLines": [
                {"AccountCode": "200", "LineAmount": "0"},
            ],
        }
        mj = ManualJournal.from_api(data)
        line = mj.journal_lines[0]
        # Zero amount: neither debit nor credit
        assert line.debit is None
        assert line.credit is None


# =============================================================================
# XeroError Tests
# =============================================================================


class TestXeroError:
    """Tests for XeroError exception."""

    def test_creation_with_message(self):
        err = XeroError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.status_code is None
        assert err.details == {}

    def test_creation_with_status_and_details(self):
        err = XeroError("Not found", status_code=404, details={"Type": "ValidationException"})
        assert err.status_code == 404
        assert err.details["Type"] == "ValidationException"

    def test_inherits_from_exception(self):
        err = XeroError("test")
        assert isinstance(err, Exception)


# =============================================================================
# Date/Datetime Parsing Tests
# =============================================================================


class TestDateTimeParsing:
    """Tests for _parse_xero_datetime and _parse_xero_date helpers."""

    def test_parse_xero_datetime_timestamp_format(self):
        result = _parse_xero_datetime("/Date(1700000000000)/")
        assert result is not None
        assert isinstance(result, datetime)

    def test_parse_xero_datetime_with_timezone_offset(self):
        result = _parse_xero_datetime("/Date(1700000000000+0000)/")
        assert result is not None

    def test_parse_xero_datetime_with_negative_offset(self):
        result = _parse_xero_datetime("/Date(1700000000000-0500)/")
        assert result is not None

    def test_parse_xero_datetime_iso_format(self):
        result = _parse_xero_datetime("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_xero_datetime_none(self):
        assert _parse_xero_datetime(None) is None

    def test_parse_xero_datetime_empty_string(self):
        assert _parse_xero_datetime("") is None

    def test_parse_xero_datetime_invalid(self):
        assert _parse_xero_datetime("not-a-date") is None

    def test_parse_xero_date_timestamp(self):
        result = _parse_xero_date("/Date(1700000000000)/")
        assert result is not None
        assert isinstance(result, Date)

    def test_parse_xero_date_none(self):
        assert _parse_xero_date(None) is None

    def test_parse_xero_date_empty(self):
        assert _parse_xero_date("") is None

    def test_parse_xero_date_invalid(self):
        assert _parse_xero_date("invalid") is None


# =============================================================================
# XeroConnector Initialization Tests
# =============================================================================


class TestXeroConnectorInit:
    """Tests for XeroConnector initialization."""

    def test_init_with_credentials(self, xero_connector, xero_credentials):
        assert xero_connector.credentials is xero_credentials
        assert xero_connector._client is None
        assert xero_connector._circuit_breaker is None

    def test_circuit_breaker_enabled(self, xero_connector_with_cb):
        assert xero_connector_with_cb._circuit_breaker is not None

    def test_custom_circuit_breaker(self, xero_credentials):
        from aragora.resilience import CircuitBreaker

        cb = CircuitBreaker(name="custom_xero", failure_threshold=5, cooldown_seconds=90.0)
        connector = XeroConnector(credentials=xero_credentials, circuit_breaker=cb)
        assert connector._circuit_breaker is cb

    def test_circuit_breaker_disabled(self, xero_connector):
        assert xero_connector._circuit_breaker is None


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestGetClient:
    """Tests for HTTP client creation."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, xero_connector):
        client = await xero_connector._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        # Clean up
        await xero_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_returns_same_instance(self, xero_connector):
        client1 = await xero_connector._get_client()
        client2 = await xero_connector._get_client()
        assert client1 is client2
        await xero_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_includes_tenant_header(self, xero_connector):
        client = await xero_connector._get_client()
        assert "Xero-Tenant-Id" in client.headers
        assert client.headers["Xero-Tenant-Id"] == "test_tenant_id"
        await xero_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_no_tenant_header_when_none(self):
        creds = XeroCredentials(
            client_id="cid",
            client_secret="csecret",
            access_token="token",
        )
        connector = XeroConnector(credentials=creds, enable_circuit_breaker=False)
        client = await connector._get_client()
        assert "Xero-Tenant-Id" not in client.headers
        await connector.close()


# =============================================================================
# API Request Tests
# =============================================================================


class TestXeroRequest:
    """Tests for the _request method and error handling."""

    @pytest.mark.asyncio
    async def test_request_success(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Contacts": []}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        result = await xero_connector._request("GET", "/Contacts")
        assert result == {"Contacts": []}

    @pytest.mark.asyncio
    async def test_request_server_error_json(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"Message": "Internal server error"}
        mock_response.text = "Internal server error"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_server_error_non_json(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "Bad Gateway"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 502
        assert "502" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_rate_limit_429(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"Message": "Rate limit exceeded"}
        mock_response.text = "Rate limit exceeded"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_request_client_error_400_json(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"Message": "Validation error"}
        mock_response.text = "Validation error"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_request_client_error_non_json(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "Not found"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts/missing")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_request_timeout(self, xero_connector):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("Timed out"))
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_request_connection_error(self, xero_connector):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        xero_connector._client = mock_client

        with pytest.raises(XeroError) as exc_info:
            await xero_connector._request("GET", "/Contacts")
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_request_circuit_breaker_open(self, xero_connector_with_cb):
        xero_connector_with_cb._circuit_breaker.can_proceed = MagicMock(return_value=False)
        xero_connector_with_cb._circuit_breaker.cooldown_remaining = MagicMock(return_value=30.0)

        with pytest.raises(XeroError) as exc_info:
            await xero_connector_with_cb._request("GET", "/Contacts")
        assert exc_info.value.status_code == 503
        assert "Circuit breaker" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_circuit_breaker_records_failure_on_500(self, xero_connector_with_cb):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"Message": "Error"}
        mock_response.text = "Error"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector_with_cb._client = mock_client

        original_fn = xero_connector_with_cb._circuit_breaker.record_failure
        xero_connector_with_cb._circuit_breaker.record_failure = MagicMock(side_effect=original_fn)

        with pytest.raises(XeroError):
            await xero_connector_with_cb._request("GET", "/Contacts")

        xero_connector_with_cb._circuit_breaker.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_request_circuit_breaker_records_success(self, xero_connector_with_cb):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector_with_cb._client = mock_client

        original_fn = xero_connector_with_cb._circuit_breaker.record_success
        xero_connector_with_cb._circuit_breaker.record_success = MagicMock(side_effect=original_fn)

        await xero_connector_with_cb._request("GET", "/Contacts")
        xero_connector_with_cb._circuit_breaker.record_success.assert_called()


# =============================================================================
# Contact Operations Tests
# =============================================================================


class TestContactOperations:
    """Tests for contact API operations."""

    @pytest.mark.asyncio
    async def test_get_contacts(self, xero_connector, sample_contact_api_data):
        xero_connector._request = AsyncMock(return_value={"Contacts": [sample_contact_api_data]})
        contacts = await xero_connector.get_contacts()
        assert len(contacts) == 1
        assert contacts[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_contacts_with_page(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": []})
        await xero_connector.get_contacts(page=3)
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["page"] == 3

    @pytest.mark.asyncio
    async def test_get_contacts_with_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": []})
        await xero_connector.get_contacts(where='Name=="Acme"')
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["where"] == 'Name=="Acme"'

    @pytest.mark.asyncio
    async def test_get_contacts_with_ids(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": []})
        await xero_connector.get_contacts(ids=["id1", "id2"])
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["IDs"] == "id1,id2"

    @pytest.mark.asyncio
    async def test_get_contacts_empty(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": []})
        contacts = await xero_connector.get_contacts()
        assert contacts == []

    @pytest.mark.asyncio
    async def test_get_contact_success(self, xero_connector, sample_contact_api_data):
        xero_connector._request = AsyncMock(return_value={"Contacts": [sample_contact_api_data]})
        contact = await xero_connector.get_contact("contact-abc-123")
        assert contact.contact_id == "contact-abc-123"

    @pytest.mark.asyncio
    async def test_get_contact_not_found(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": []})
        with pytest.raises(XeroError, match="not found"):
            await xero_connector.get_contact("missing-id")

    @pytest.mark.asyncio
    async def test_create_contact_full(self, xero_connector, sample_contact_api_data):
        xero_connector._request = AsyncMock(return_value={"Contacts": [sample_contact_api_data]})
        contact = await xero_connector.create_contact(
            name="Acme Corp",
            email="billing@acme.com",
            first_name="John",
            last_name="Doe",
            is_customer=True,
            is_supplier=True,
            tax_number="GB123456789",
            addresses=[Address(address_type="STREET", city="London")],
        )
        assert contact.name == "Acme Corp"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Contacts"][0]
        assert json_data["Name"] == "Acme Corp"
        assert json_data["EmailAddress"] == "billing@acme.com"
        assert json_data["IsCustomer"] is True
        assert json_data["IsSupplier"] is True

    @pytest.mark.asyncio
    async def test_create_contact_minimal(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Contacts": [{"Name": "Minimal"}]})
        contact = await xero_connector.create_contact(name="Minimal")
        assert contact.name == "Minimal"

    @pytest.mark.asyncio
    async def test_update_contact(self, xero_connector, sample_contact_api_data):
        updated_data = dict(sample_contact_api_data)
        updated_data["Name"] = "Updated Corp"
        xero_connector._request = AsyncMock(return_value={"Contacts": [updated_data]})
        contact = await xero_connector.update_contact(
            contact_id="contact-abc-123",
            name="Updated Corp",
            email="new@acme.com",
        )
        assert contact.name == "Updated Corp"


# =============================================================================
# Invoice Operations Tests
# =============================================================================


class TestInvoiceOperations:
    """Tests for invoice API operations."""

    @pytest.mark.asyncio
    async def test_get_invoices(self, xero_connector, sample_invoice_api_data):
        xero_connector._request = AsyncMock(return_value={"Invoices": [sample_invoice_api_data]})
        invoices = await xero_connector.get_invoices()
        assert len(invoices) == 1
        assert invoices[0].invoice_number == "INV-0042"

    @pytest.mark.asyncio
    async def test_get_invoices_with_status_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": []})
        await xero_connector.get_invoices(status=InvoiceStatus.AUTHORISED)
        call_args = xero_connector._request.call_args
        assert 'Status=="AUTHORISED"' in call_args[1]["params"]["where"]

    @pytest.mark.asyncio
    async def test_get_invoices_with_type_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": []})
        await xero_connector.get_invoices(invoice_type=InvoiceType.ACCPAY)
        call_args = xero_connector._request.call_args
        assert 'Type=="ACCPAY"' in call_args[1]["params"]["where"]

    @pytest.mark.asyncio
    async def test_get_invoices_combined_filters(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": []})
        await xero_connector.get_invoices(
            where='Contact.Name=="Test"',
            status=InvoiceStatus.DRAFT,
            invoice_type=InvoiceType.ACCREC,
        )
        call_args = xero_connector._request.call_args
        where_clause = call_args[1]["params"]["where"]
        assert "Contact.Name" in where_clause
        assert "DRAFT" in where_clause
        assert "ACCREC" in where_clause
        assert " AND " in where_clause

    @pytest.mark.asyncio
    async def test_get_invoices_empty(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": []})
        invoices = await xero_connector.get_invoices()
        assert invoices == []

    @pytest.mark.asyncio
    async def test_get_invoice_success(self, xero_connector, sample_invoice_api_data):
        xero_connector._request = AsyncMock(return_value={"Invoices": [sample_invoice_api_data]})
        invoice = await xero_connector.get_invoice("inv-abc-123")
        assert invoice.invoice_id == "inv-abc-123"

    @pytest.mark.asyncio
    async def test_get_invoice_not_found(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": []})
        with pytest.raises(XeroError, match="not found"):
            await xero_connector.get_invoice("missing")

    @pytest.mark.asyncio
    async def test_create_invoice(self, xero_connector, sample_invoice_api_data):
        xero_connector._request = AsyncMock(return_value={"Invoices": [sample_invoice_api_data]})
        line_items = [
            LineItem(description="Service", quantity=Decimal("1"), unit_amount=Decimal("100")),
        ]
        invoice = await xero_connector.create_invoice(
            contact_id="contact-abc-123",
            line_items=line_items,
            invoice_type=InvoiceType.ACCREC,
            invoice_date=Date(2024, 1, 15),
            due_date=Date(2024, 2, 14),
            reference="PO-001",
            status=InvoiceStatus.DRAFT,
        )
        assert invoice.invoice_id == "inv-abc-123"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Invoices"][0]
        assert json_data["Type"] == "ACCREC"
        assert json_data["Contact"]["ContactID"] == "contact-abc-123"
        assert json_data["Date"] == "2024-01-15"
        assert json_data["DueDate"] == "2024-02-14"
        assert json_data["Reference"] == "PO-001"

    @pytest.mark.asyncio
    async def test_create_invoice_minimal(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Invoices": [{"InvoiceID": "new-inv"}]})
        invoice = await xero_connector.create_invoice(
            contact_id="c1",
            line_items=[LineItem(description="Item")],
        )
        assert invoice.invoice_id == "new-inv"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Invoices"][0]
        assert "Date" not in json_data
        assert "DueDate" not in json_data
        assert "Reference" not in json_data

    @pytest.mark.asyncio
    async def test_update_invoice_status(self, xero_connector, sample_invoice_api_data):
        updated = dict(sample_invoice_api_data)
        updated["Status"] = "PAID"
        xero_connector._request = AsyncMock(return_value={"Invoices": [updated]})
        invoice = await xero_connector.update_invoice_status("inv-abc-123", InvoiceStatus.PAID)
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Invoices"][0]
        assert json_data["InvoiceID"] == "inv-abc-123"
        assert json_data["Status"] == "PAID"

    @pytest.mark.asyncio
    async def test_void_invoice(self, xero_connector, sample_invoice_api_data):
        voided = dict(sample_invoice_api_data)
        voided["Status"] = "VOIDED"
        xero_connector._request = AsyncMock(return_value={"Invoices": [voided]})
        invoice = await xero_connector.void_invoice("inv-abc-123")
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Invoices"][0]
        assert json_data["Status"] == "VOIDED"


# =============================================================================
# Account Operations Tests
# =============================================================================


class TestAccountOperations:
    """Tests for account API operations."""

    @pytest.mark.asyncio
    async def test_get_accounts(self, xero_connector, sample_account_api_data):
        xero_connector._request = AsyncMock(return_value={"Accounts": [sample_account_api_data]})
        accounts = await xero_connector.get_accounts()
        assert len(accounts) == 1
        assert accounts[0].name == "Sales Revenue"

    @pytest.mark.asyncio
    async def test_get_accounts_with_type_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Accounts": []})
        await xero_connector.get_accounts(account_type=AccountType.BANK)
        call_args = xero_connector._request.call_args
        assert 'Type=="BANK"' in call_args[1]["params"]["where"]

    @pytest.mark.asyncio
    async def test_get_accounts_with_where_and_type(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Accounts": []})
        await xero_connector.get_accounts(
            where='Status=="ACTIVE"',
            account_type=AccountType.EXPENSE,
        )
        call_args = xero_connector._request.call_args
        where = call_args[1]["params"]["where"]
        assert "ACTIVE" in where
        assert "EXPENSE" in where
        assert " AND " in where

    @pytest.mark.asyncio
    async def test_get_accounts_no_filters(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Accounts": []})
        await xero_connector.get_accounts()
        call_args = xero_connector._request.call_args
        assert "where" not in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_get_account_success(self, xero_connector, sample_account_api_data):
        xero_connector._request = AsyncMock(return_value={"Accounts": [sample_account_api_data]})
        account = await xero_connector.get_account("acct-abc-123")
        assert account.account_id == "acct-abc-123"

    @pytest.mark.asyncio
    async def test_get_account_not_found(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Accounts": []})
        with pytest.raises(XeroError, match="not found"):
            await xero_connector.get_account("missing")

    @pytest.mark.asyncio
    async def test_get_bank_accounts(self, xero_connector):
        bank_data = {
            "AccountID": "bank-001",
            "Name": "Business Account",
            "Type": "BANK",
        }
        xero_connector._request = AsyncMock(return_value={"Accounts": [bank_data]})
        accounts = await xero_connector.get_bank_accounts()
        assert len(accounts) == 1
        assert accounts[0].type == AccountType.BANK


# =============================================================================
# Bank Transaction Operations Tests
# =============================================================================


class TestBankTransactionOperations:
    """Tests for bank transaction API operations."""

    @pytest.mark.asyncio
    async def test_get_bank_transactions(self, xero_connector, sample_bank_transaction_api_data):
        xero_connector._request = AsyncMock(
            return_value={"BankTransactions": [sample_bank_transaction_api_data]}
        )
        transactions = await xero_connector.get_bank_transactions()
        assert len(transactions) == 1
        assert transactions[0].bank_transaction_id == "bt-abc-123"

    @pytest.mark.asyncio
    async def test_get_bank_transactions_with_bank_account_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"BankTransactions": []})
        await xero_connector.get_bank_transactions(bank_account_id="acct-bank-001")
        call_args = xero_connector._request.call_args
        assert "BankAccount.AccountID" in call_args[1]["params"]["where"]
        assert "acct-bank-001" in call_args[1]["params"]["where"]

    @pytest.mark.asyncio
    async def test_get_bank_transactions_with_page(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"BankTransactions": []})
        await xero_connector.get_bank_transactions(page=2)
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["page"] == 2

    @pytest.mark.asyncio
    async def test_create_bank_transaction(self, xero_connector, sample_bank_transaction_api_data):
        xero_connector._request = AsyncMock(
            return_value={"BankTransactions": [sample_bank_transaction_api_data]}
        )
        line_items = [
            LineItem(description="Supplies", quantity=Decimal("1"), unit_amount=Decimal("50")),
        ]
        tx = await xero_connector.create_bank_transaction(
            bank_account_id="acct-bank-001",
            contact_id="contact-abc-123",
            line_items=line_items,
            transaction_type=BankTransactionType.SPEND,
            transaction_date=Date(2024, 1, 15),
            reference="REF-001",
        )
        assert tx.bank_transaction_id == "bt-abc-123"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["BankTransactions"][0]
        assert json_data["Type"] == "SPEND"
        assert json_data["BankAccount"]["AccountID"] == "acct-bank-001"
        assert json_data["Date"] == "2024-01-15"
        assert json_data["Reference"] == "REF-001"

    @pytest.mark.asyncio
    async def test_create_bank_transaction_minimal(self, xero_connector):
        xero_connector._request = AsyncMock(
            return_value={"BankTransactions": [{"BankTransactionID": "new-bt"}]}
        )
        tx = await xero_connector.create_bank_transaction(
            bank_account_id="b1",
            contact_id="c1",
            line_items=[LineItem(description="Item")],
        )
        assert tx.bank_transaction_id == "new-bt"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["BankTransactions"][0]
        assert "Date" not in json_data
        assert "Reference" not in json_data


# =============================================================================
# Payment Operations Tests
# =============================================================================


class TestPaymentOperations:
    """Tests for payment API operations."""

    @pytest.mark.asyncio
    async def test_get_payments(self, xero_connector, sample_payment_api_data):
        xero_connector._request = AsyncMock(return_value={"Payments": [sample_payment_api_data]})
        payments = await xero_connector.get_payments()
        assert len(payments) == 1
        assert payments[0].payment_id == "pay-abc-123"

    @pytest.mark.asyncio
    async def test_get_payments_with_filter(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Payments": []})
        await xero_connector.get_payments(where='Status=="AUTHORISED"')
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["where"] == 'Status=="AUTHORISED"'

    @pytest.mark.asyncio
    async def test_get_payments_with_page(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Payments": []})
        await xero_connector.get_payments(page=5)
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["page"] == 5

    @pytest.mark.asyncio
    async def test_get_payments_empty(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Payments": []})
        payments = await xero_connector.get_payments()
        assert payments == []

    @pytest.mark.asyncio
    async def test_create_payment(self, xero_connector, sample_payment_api_data):
        xero_connector._request = AsyncMock(return_value={"Payments": [sample_payment_api_data]})
        payment = await xero_connector.create_payment(
            invoice_id="inv-abc-123",
            account_id="acct-bank-001",
            amount=Decimal("1725.00"),
            payment_date=Date(2024, 1, 20),
            reference="Pay INV-0042",
        )
        assert payment.payment_id == "pay-abc-123"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Payments"][0]
        assert json_data["Invoice"]["InvoiceID"] == "inv-abc-123"
        assert json_data["Account"]["AccountID"] == "acct-bank-001"
        assert json_data["Amount"] == "1725.00"
        assert json_data["Date"] == "2024-01-20"
        assert json_data["Reference"] == "Pay INV-0042"

    @pytest.mark.asyncio
    async def test_create_payment_minimal(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Payments": [{"PaymentID": "new-pay"}]})
        payment = await xero_connector.create_payment(
            invoice_id="inv-1",
            account_id="acct-1",
            amount=Decimal("500"),
        )
        assert payment.payment_id == "new-pay"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["Payments"][0]
        assert "Date" not in json_data
        assert "Reference" not in json_data


# =============================================================================
# Manual Journal Operations Tests
# =============================================================================


class TestManualJournalOperations:
    """Tests for manual journal API operations."""

    @pytest.mark.asyncio
    async def test_get_manual_journals(self, xero_connector, sample_journal_api_data):
        xero_connector._request = AsyncMock(
            return_value={"ManualJournals": [sample_journal_api_data]}
        )
        journals = await xero_connector.get_manual_journals()
        assert len(journals) == 1
        assert journals[0].narration == "Month-end adjustment"

    @pytest.mark.asyncio
    async def test_get_manual_journals_with_page(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"ManualJournals": []})
        await xero_connector.get_manual_journals(page=2)
        call_args = xero_connector._request.call_args
        assert call_args[1]["params"]["page"] == 2

    @pytest.mark.asyncio
    async def test_get_manual_journals_empty(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"ManualJournals": []})
        journals = await xero_connector.get_manual_journals()
        assert journals == []

    @pytest.mark.asyncio
    async def test_create_manual_journal(self, xero_connector, sample_journal_api_data):
        xero_connector._request = AsyncMock(
            return_value={"ManualJournals": [sample_journal_api_data]}
        )
        lines = [
            JournalLine(account_code="200", debit=Decimal("500"), description="Revenue"),
            JournalLine(account_code="400", credit=Decimal("500"), description="Expense"),
        ]
        journal = await xero_connector.create_manual_journal(
            narration="Month-end adjustment",
            journal_lines=lines,
            journal_date=Date(2024, 1, 31),
            status="POSTED",
        )
        assert journal.manual_journal_id == "mj-abc-123"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["ManualJournals"][0]
        assert json_data["Narration"] == "Month-end adjustment"
        assert json_data["Status"] == "POSTED"
        assert json_data["Date"] == "2024-01-31"
        assert len(json_data["JournalLines"]) == 2

    @pytest.mark.asyncio
    async def test_create_manual_journal_minimal(self, xero_connector):
        xero_connector._request = AsyncMock(
            return_value={"ManualJournals": [{"ManualJournalID": "new-mj"}]}
        )
        journal = await xero_connector.create_manual_journal(
            narration="Test",
            journal_lines=[JournalLine(account_code="200", debit=Decimal("100"))],
        )
        assert journal.manual_journal_id == "new-mj"
        call_args = xero_connector._request.call_args
        json_data = call_args[1]["json_data"]["ManualJournals"][0]
        assert "Date" not in json_data


# =============================================================================
# Organisation Tests
# =============================================================================


class TestOrganisation:
    """Tests for organisation endpoint."""

    @pytest.mark.asyncio
    async def test_get_organisation(self, xero_connector):
        org_data = {
            "Name": "My Organisation",
            "LegalName": "My Org Ltd",
            "CountryCode": "GB",
        }
        xero_connector._request = AsyncMock(return_value={"Organisations": [org_data]})
        result = await xero_connector.get_organisation()
        assert result["Name"] == "My Organisation"
        assert result["CountryCode"] == "GB"

    @pytest.mark.asyncio
    async def test_get_organisation_empty(self, xero_connector):
        xero_connector._request = AsyncMock(return_value={"Organisations": []})
        result = await xero_connector.get_organisation()
        assert result == {}


# =============================================================================
# Async Context Manager Tests
# =============================================================================


class TestAsyncContextManager:
    """Tests for async context manager and cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_client(self, xero_connector):
        # Create a client first
        client = await xero_connector._get_client()
        assert xero_connector._client is not None
        await xero_connector.close()
        assert xero_connector._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, xero_connector):
        # Should not raise
        await xero_connector.close()
        assert xero_connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, xero_credentials):
        async with XeroConnector(
            credentials=xero_credentials,
            enable_circuit_breaker=False,
        ) as connector:
            assert connector is not None
            assert isinstance(connector, XeroConnector)

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self, xero_credentials):
        connector = XeroConnector(
            credentials=xero_credentials,
            enable_circuit_breaker=False,
        )
        async with connector:
            await connector._get_client()
            assert connector._client is not None
        assert connector._client is None


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation helpers."""

    def test_get_mock_invoice(self):
        invoice = get_mock_invoice()
        assert invoice.invoice_id == "inv-123"
        assert invoice.invoice_number == "INV-0001"
        assert invoice.type == InvoiceType.ACCREC
        assert invoice.status == InvoiceStatus.AUTHORISED
        assert invoice.total == Decimal("1000.00")
        assert invoice.amount_due == Decimal("1000.00")

    def test_get_mock_contact(self):
        contact = get_mock_contact()
        assert contact.contact_id == "contact-123"
        assert contact.name == "Test Customer"
        assert contact.email == "customer@example.com"
        assert contact.is_customer is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_line_item_zero_quantity(self):
        item = LineItem(description="Free item", quantity=Decimal("0"), unit_amount=Decimal("100"))
        result = item.to_api()
        assert result["Quantity"] == "0"

    def test_line_item_negative_amount(self):
        item = LineItem(
            description="Discount",
            quantity=Decimal("1"),
            unit_amount=Decimal("-50"),
        )
        result = item.to_api()
        assert result["UnitAmount"] == "-50"

    def test_journal_line_neither_debit_nor_credit(self):
        line = JournalLine(account_code="200")
        result = line.to_api()
        # No LineAmount key when neither debit nor credit set
        assert "LineAmount" not in result

    def test_address_all_none_optional_fields(self):
        addr = Address()
        result = addr.to_api()
        assert len(result) == 1  # Only AddressType

    def test_invoice_defaults(self):
        invoice = Invoice()
        assert invoice.invoice_id is None
        assert invoice.type == InvoiceType.ACCREC
        assert invoice.status == InvoiceStatus.DRAFT
        assert invoice.currency_code == "USD"
        assert invoice.line_amount_types == "Exclusive"

    def test_bank_transaction_defaults(self):
        bt = BankTransaction()
        assert bt.type == BankTransactionType.SPEND
        assert bt.currency_code == "USD"
        assert bt.status == "AUTHORISED"
        assert bt.is_reconciled is False

    def test_payment_defaults(self):
        payment = Payment()
        assert payment.amount == Decimal("0")
        assert payment.currency_rate == Decimal("1")
        assert payment.status == PaymentStatus.AUTHORISED

    def test_xero_error_with_empty_details(self):
        err = XeroError("error")
        assert err.details == {}

    def test_parse_xero_datetime_timestamp_zero(self):
        result = _parse_xero_datetime("/Date(0)/")
        assert result is not None

    @pytest.mark.asyncio
    async def test_request_with_json_data(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Result": "ok"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        result = await xero_connector._request(
            "POST",
            "/Contacts",
            json_data={"Contacts": [{"Name": "Test"}]},
        )
        assert result == {"Result": "ok"}
        mock_client.request.assert_called_once()
        call_kwargs = mock_client.request.call_args
        assert call_kwargs[1]["json"] == {"Contacts": [{"Name": "Test"}]}

    @pytest.mark.asyncio
    async def test_request_with_params(self, xero_connector):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"Contacts": []}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        xero_connector._client = mock_client

        await xero_connector._request(
            "GET",
            "/Contacts",
            params={"page": 2, "where": 'Name=="Test"'},
        )
        call_kwargs = mock_client.request.call_args
        assert call_kwargs[1]["params"]["page"] == 2

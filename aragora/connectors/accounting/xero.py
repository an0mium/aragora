"""
Xero Accounting Connector.

Integration with Xero API:
- Contacts (customers and suppliers)
- Invoices (sales and bills)
- Bank transactions and reconciliation
- Accounts and chart of accounts
- Journal entries
- Payments
- Reporting

Requires Xero OAuth2 credentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class InvoiceType(str, Enum):
    """Invoice type."""

    ACCPAY = "ACCPAY"  # Bill (accounts payable)
    ACCREC = "ACCREC"  # Invoice (accounts receivable)


class InvoiceStatus(str, Enum):
    """Invoice status."""

    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    AUTHORISED = "AUTHORISED"
    PAID = "PAID"
    VOIDED = "VOIDED"
    DELETED = "DELETED"


class ContactStatus(str, Enum):
    """Contact status."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    GDPRREQUEST = "GDPRREQUEST"


class AccountType(str, Enum):
    """Account type."""

    BANK = "BANK"
    CURRENT = "CURRENT"
    CURRLIAB = "CURRLIAB"
    DEPRECIATN = "DEPRECIATN"
    DIRECTCOSTS = "DIRECTCOSTS"
    EQUITY = "EQUITY"
    EXPENSE = "EXPENSE"
    FIXED = "FIXED"
    INVENTORY = "INVENTORY"
    LIABILITY = "LIABILITY"
    NONCURRENT = "NONCURRENT"
    OTHERINCOME = "OTHERINCOME"
    OVERHEADS = "OVERHEADS"
    PREPAYMENT = "PREPAYMENT"
    REVENUE = "REVENUE"
    SALES = "SALES"
    TERMLIAB = "TERMLIAB"
    PAYGLIABILITY = "PAYGLIABILITY"
    SUPERANNUATIONEXPENSE = "SUPERANNUATIONEXPENSE"
    SUPERANNUATIONLIABILITY = "SUPERANNUATIONLIABILITY"
    WAGESEXPENSE = "WAGESEXPENSE"


class BankTransactionType(str, Enum):
    """Bank transaction type."""

    RECEIVE = "RECEIVE"  # Money in
    SPEND = "SPEND"  # Money out
    RECEIVE_OVERPAYMENT = "RECEIVE-OVERPAYMENT"
    RECEIVE_PREPAYMENT = "RECEIVE-PREPAYMENT"
    SPEND_OVERPAYMENT = "SPEND-OVERPAYMENT"
    SPEND_PREPAYMENT = "SPEND-PREPAYMENT"


class PaymentStatus(str, Enum):
    """Payment status."""

    AUTHORISED = "AUTHORISED"
    DELETED = "DELETED"


@dataclass
class XeroCredentials:
    """Xero API credentials."""

    client_id: str
    client_secret: str
    access_token: str | None = None
    refresh_token: str | None = None
    tenant_id: str | None = None
    base_url: str = "https://api.xero.com/api.xro/2.0"


@dataclass
class Address:
    """Xero address."""

    address_type: str = "POBOX"  # POBOX, STREET, DELIVERY
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    region: str | None = None
    postal_code: str | None = None
    country: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Address:
        """Create from API response."""
        return cls(
            address_type=data.get("AddressType", "POBOX"),
            address_line1=data.get("AddressLine1"),
            address_line2=data.get("AddressLine2"),
            city=data.get("City"),
            region=data.get("Region"),
            postal_code=data.get("PostalCode"),
            country=data.get("Country"),
        )

    def to_api(self) -> dict[str, Any]:
        """Convert to API format."""
        result: dict[str, Any] = {"AddressType": self.address_type}
        if self.address_line1:
            result["AddressLine1"] = self.address_line1
        if self.address_line2:
            result["AddressLine2"] = self.address_line2
        if self.city:
            result["City"] = self.city
        if self.region:
            result["Region"] = self.region
        if self.postal_code:
            result["PostalCode"] = self.postal_code
        if self.country:
            result["Country"] = self.country
        return result


@dataclass
class Phone:
    """Xero phone number."""

    phone_type: str = "DEFAULT"  # DEFAULT, DDI, MOBILE, FAX
    phone_number: str | None = None
    phone_area_code: str | None = None
    phone_country_code: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Phone:
        """Create from API response."""
        return cls(
            phone_type=data.get("PhoneType", "DEFAULT"),
            phone_number=data.get("PhoneNumber"),
            phone_area_code=data.get("PhoneAreaCode"),
            phone_country_code=data.get("PhoneCountryCode"),
        )


@dataclass
class XeroContact:
    """Xero contact (customer or supplier)."""

    contact_id: str | None = None
    name: str = ""
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    company_number: str | None = None
    tax_number: str | None = None
    bank_account_details: str | None = None
    accounts_receivable_tax_type: str | None = None
    accounts_payable_tax_type: str | None = None
    is_supplier: bool = False
    is_customer: bool = False
    default_currency: str | None = None
    contact_status: ContactStatus = ContactStatus.ACTIVE
    addresses: list[Address] = field(default_factory=list)
    phones: list[Phone] = field(default_factory=list)
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> XeroContact:
        """Create from API response."""
        return cls(
            contact_id=data.get("ContactID"),
            name=data.get("Name", ""),
            email=data.get("EmailAddress"),
            first_name=data.get("FirstName"),
            last_name=data.get("LastName"),
            company_number=data.get("CompanyNumber"),
            tax_number=data.get("TaxNumber"),
            bank_account_details=data.get("BankAccountDetails"),
            accounts_receivable_tax_type=data.get("AccountsReceivableTaxType"),
            accounts_payable_tax_type=data.get("AccountsPayableTaxType"),
            is_supplier=data.get("IsSupplier", False),
            is_customer=data.get("IsCustomer", False),
            default_currency=data.get("DefaultCurrency"),
            contact_status=ContactStatus(data.get("ContactStatus", "ACTIVE")),
            addresses=[Address.from_api(a) for a in data.get("Addresses", [])],
            phones=[Phone.from_api(p) for p in data.get("Phones", [])],
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


@dataclass
class LineItem:
    """Invoice line item."""

    description: str
    quantity: Decimal = Decimal("1")
    unit_amount: Decimal = Decimal("0")
    account_code: str | None = None
    tax_type: str | None = None
    tax_amount: Decimal | None = None
    line_amount: Decimal | None = None
    item_code: str | None = None
    tracking: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> LineItem:
        """Create from API response."""
        return cls(
            description=data.get("Description", ""),
            quantity=Decimal(str(data.get("Quantity", 1))),
            unit_amount=Decimal(str(data.get("UnitAmount", 0))),
            account_code=data.get("AccountCode"),
            tax_type=data.get("TaxType"),
            tax_amount=Decimal(str(data["TaxAmount"])) if data.get("TaxAmount") else None,
            line_amount=Decimal(str(data["LineAmount"])) if data.get("LineAmount") else None,
            item_code=data.get("ItemCode"),
            tracking=data.get("Tracking", []),
        )

    def to_api(self) -> dict[str, Any]:
        """Convert to API format."""
        result: dict[str, Any] = {
            "Description": self.description,
            "Quantity": str(self.quantity),
            "UnitAmount": str(self.unit_amount),
        }
        if self.account_code:
            result["AccountCode"] = self.account_code
        if self.tax_type:
            result["TaxType"] = self.tax_type
        if self.item_code:
            result["ItemCode"] = self.item_code
        if self.tracking:
            result["Tracking"] = self.tracking
        return result


@dataclass
class Invoice:
    """Xero invoice."""

    invoice_id: str | None = None
    invoice_number: str | None = None
    reference: str | None = None
    type: InvoiceType = InvoiceType.ACCREC
    status: InvoiceStatus = InvoiceStatus.DRAFT
    contact_id: str | None = None
    contact_name: str | None = None
    date: date | None = None
    due_date: date | None = None
    line_items: list[LineItem] = field(default_factory=list)
    sub_total: Decimal = Decimal("0")
    total_tax: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    amount_due: Decimal = Decimal("0")
    amount_paid: Decimal = Decimal("0")
    currency_code: str = "USD"
    line_amount_types: str = "Exclusive"  # Exclusive, Inclusive, NoTax
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Invoice:
        """Create from API response."""
        contact = data.get("Contact", {})
        return cls(
            invoice_id=data.get("InvoiceID"),
            invoice_number=data.get("InvoiceNumber"),
            reference=data.get("Reference"),
            type=InvoiceType(data.get("Type", "ACCREC")),
            status=InvoiceStatus(data.get("Status", "DRAFT")),
            contact_id=contact.get("ContactID"),
            contact_name=contact.get("Name"),
            date=_parse_xero_date(data.get("Date")),
            due_date=_parse_xero_date(data.get("DueDate")),
            line_items=[LineItem.from_api(li) for li in data.get("LineItems", [])],
            sub_total=Decimal(str(data.get("SubTotal", 0))),
            total_tax=Decimal(str(data.get("TotalTax", 0))),
            total=Decimal(str(data.get("Total", 0))),
            amount_due=Decimal(str(data.get("AmountDue", 0))),
            amount_paid=Decimal(str(data.get("AmountPaid", 0))),
            currency_code=data.get("CurrencyCode", "USD"),
            line_amount_types=data.get("LineAmountTypes", "Exclusive"),
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


@dataclass
class Account:
    """Xero account (chart of accounts)."""

    account_id: str | None = None
    code: str | None = None
    name: str = ""
    type: AccountType | None = None
    status: str = "ACTIVE"
    description: str | None = None
    tax_type: str | None = None
    enable_payments: bool = False
    bank_account_number: str | None = None
    bank_account_type: str | None = None
    currency_code: str | None = None
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Account:
        """Create from API response."""
        return cls(
            account_id=data.get("AccountID"),
            code=data.get("Code"),
            name=data.get("Name", ""),
            type=AccountType(data["Type"]) if data.get("Type") else None,
            status=data.get("Status", "ACTIVE"),
            description=data.get("Description"),
            tax_type=data.get("TaxType"),
            enable_payments=data.get("EnablePaymentsToAccount", False),
            bank_account_number=data.get("BankAccountNumber"),
            bank_account_type=data.get("BankAccountType"),
            currency_code=data.get("CurrencyCode"),
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


@dataclass
class BankTransaction:
    """Xero bank transaction."""

    bank_transaction_id: str | None = None
    type: BankTransactionType = BankTransactionType.SPEND
    contact_id: str | None = None
    contact_name: str | None = None
    bank_account_id: str | None = None
    bank_account_code: str | None = None
    date: date | None = None
    reference: str | None = None
    line_items: list[LineItem] = field(default_factory=list)
    sub_total: Decimal = Decimal("0")
    total_tax: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    currency_code: str = "USD"
    status: str = "AUTHORISED"
    is_reconciled: bool = False
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> BankTransaction:
        """Create from API response."""
        contact = data.get("Contact", {})
        bank_account = data.get("BankAccount", {})
        return cls(
            bank_transaction_id=data.get("BankTransactionID"),
            type=BankTransactionType(data.get("Type", "SPEND")),
            contact_id=contact.get("ContactID"),
            contact_name=contact.get("Name"),
            bank_account_id=bank_account.get("AccountID"),
            bank_account_code=bank_account.get("Code"),
            date=_parse_xero_date(data.get("Date")),
            reference=data.get("Reference"),
            line_items=[LineItem.from_api(li) for li in data.get("LineItems", [])],
            sub_total=Decimal(str(data.get("SubTotal", 0))),
            total_tax=Decimal(str(data.get("TotalTax", 0))),
            total=Decimal(str(data.get("Total", 0))),
            currency_code=data.get("CurrencyCode", "USD"),
            status=data.get("Status", "AUTHORISED"),
            is_reconciled=data.get("IsReconciled", False),
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


@dataclass
class Payment:
    """Xero payment."""

    payment_id: str | None = None
    invoice_id: str | None = None
    invoice_number: str | None = None
    account_id: str | None = None
    date: date | None = None
    amount: Decimal = Decimal("0")
    currency_rate: Decimal = Decimal("1")
    reference: str | None = None
    status: PaymentStatus = PaymentStatus.AUTHORISED
    payment_type: str | None = None
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Payment:
        """Create from API response."""
        invoice = data.get("Invoice", {})
        return cls(
            payment_id=data.get("PaymentID"),
            invoice_id=invoice.get("InvoiceID"),
            invoice_number=invoice.get("InvoiceNumber"),
            account_id=data.get("Account", {}).get("AccountID"),
            date=_parse_xero_date(data.get("Date")),
            amount=Decimal(str(data.get("Amount", 0))),
            currency_rate=Decimal(str(data.get("CurrencyRate", 1))),
            reference=data.get("Reference"),
            status=PaymentStatus(data.get("Status", "AUTHORISED")),
            payment_type=data.get("PaymentType"),
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


@dataclass
class JournalLine:
    """Journal entry line."""

    account_code: str
    description: str | None = None
    debit: Decimal | None = None
    credit: Decimal | None = None
    tax_type: str | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert to API format."""
        result: dict[str, Any] = {"AccountCode": self.account_code}
        if self.description:
            result["Description"] = self.description
        if self.debit is not None:
            result["LineAmount"] = str(self.debit)
        elif self.credit is not None:
            result["LineAmount"] = str(-self.credit)
        if self.tax_type:
            result["TaxType"] = self.tax_type
        return result


@dataclass
class ManualJournal:
    """Xero manual journal entry."""

    manual_journal_id: str | None = None
    narration: str = ""
    date: date | None = None
    status: str = "DRAFT"
    journal_lines: list[JournalLine] = field(default_factory=list)
    url: str | None = None
    updated_date: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ManualJournal:
        """Create from API response."""
        lines = []
        for line in data.get("JournalLines", []):
            amount = Decimal(str(line.get("LineAmount", 0)))
            lines.append(
                JournalLine(
                    account_code=line.get("AccountCode", ""),
                    description=line.get("Description"),
                    debit=amount if amount > 0 else None,
                    credit=abs(amount) if amount < 0 else None,
                    tax_type=line.get("TaxType"),
                )
            )

        return cls(
            manual_journal_id=data.get("ManualJournalID"),
            narration=data.get("Narration", ""),
            date=_parse_xero_date(data.get("Date")),
            status=data.get("Status", "DRAFT"),
            journal_lines=lines,
            url=data.get("Url"),
            updated_date=_parse_xero_datetime(data.get("UpdatedDateUTC")),
        )


class XeroError(Exception):
    """Xero API error."""

    def __init__(self, message: str, status_code: int | None = None, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class XeroConnector:
    """
    Xero Accounting API connector.

    Provides integration with Xero for:
    - Contact management (customers/suppliers)
    - Invoice and bill management
    - Bank transactions and reconciliation
    - Chart of accounts
    - Payments
    - Journal entries
    """

    def __init__(self, credentials: XeroCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.credentials.tenant_id:
                headers["Xero-Tenant-Id"] = self.credentials.tenant_id

            self._client = httpx.AsyncClient(
                base_url=self.credentials.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make API request."""
        client = await self._get_client()
        response = await client.request(method, path, params=params, json=json_data)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise XeroError(
                    message=error_data.get("Message", response.text),
                    status_code=response.status_code,
                    details=error_data,
                )
            except ValueError:
                raise XeroError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

        return response.json()

    # =========================================================================
    # Contacts
    # =========================================================================

    async def get_contacts(
        self,
        page: int = 1,
        where: str | None = None,
        ids: list[str] | None = None,
    ) -> list[XeroContact]:
        """Get contacts with optional filtering."""
        params: dict[str, Any] = {"page": page}
        if where:
            params["where"] = where
        if ids:
            params["IDs"] = ",".join(ids)

        data = await self._request("GET", "/Contacts", params=params)
        return [XeroContact.from_api(c) for c in data.get("Contacts", [])]

    async def get_contact(self, contact_id: str) -> XeroContact:
        """Get a single contact."""
        data = await self._request("GET", f"/Contacts/{contact_id}")
        contacts = data.get("Contacts", [])
        if not contacts:
            raise XeroError(f"Contact {contact_id} not found", status_code=404)
        return XeroContact.from_api(contacts[0])

    async def create_contact(
        self,
        name: str,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        is_customer: bool = True,
        is_supplier: bool = False,
        tax_number: str | None = None,
        addresses: list[Address] | None = None,
    ) -> XeroContact:
        """Create a new contact."""
        contact_data: dict[str, Any] = {"Name": name}

        if email:
            contact_data["EmailAddress"] = email
        if first_name:
            contact_data["FirstName"] = first_name
        if last_name:
            contact_data["LastName"] = last_name
        if is_customer:
            contact_data["IsCustomer"] = True
        if is_supplier:
            contact_data["IsSupplier"] = True
        if tax_number:
            contact_data["TaxNumber"] = tax_number
        if addresses:
            contact_data["Addresses"] = [a.to_api() for a in addresses]

        data = await self._request("POST", "/Contacts", json_data={"Contacts": [contact_data]})
        return XeroContact.from_api(data.get("Contacts", [{}])[0])

    async def update_contact(
        self,
        contact_id: str,
        name: str | None = None,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> XeroContact:
        """Update a contact."""
        contact_data: dict[str, Any] = {"ContactID": contact_id}

        if name:
            contact_data["Name"] = name
        if email:
            contact_data["EmailAddress"] = email
        if first_name:
            contact_data["FirstName"] = first_name
        if last_name:
            contact_data["LastName"] = last_name

        data = await self._request("POST", "/Contacts", json_data={"Contacts": [contact_data]})
        return XeroContact.from_api(data.get("Contacts", [{}])[0])

    # =========================================================================
    # Invoices
    # =========================================================================

    async def get_invoices(
        self,
        page: int = 1,
        where: str | None = None,
        status: InvoiceStatus | None = None,
        invoice_type: InvoiceType | None = None,
    ) -> list[Invoice]:
        """Get invoices with optional filtering."""
        params: dict[str, Any] = {"page": page}

        filters = []
        if where:
            filters.append(where)
        if status:
            filters.append(f'Status=="{status.value}"')
        if invoice_type:
            filters.append(f'Type=="{invoice_type.value}"')

        if filters:
            params["where"] = " AND ".join(filters)

        data = await self._request("GET", "/Invoices", params=params)
        return [Invoice.from_api(i) for i in data.get("Invoices", [])]

    async def get_invoice(self, invoice_id: str) -> Invoice:
        """Get a single invoice."""
        data = await self._request("GET", f"/Invoices/{invoice_id}")
        invoices = data.get("Invoices", [])
        if not invoices:
            raise XeroError(f"Invoice {invoice_id} not found", status_code=404)
        return Invoice.from_api(invoices[0])

    async def create_invoice(
        self,
        contact_id: str,
        line_items: list[LineItem],
        invoice_type: InvoiceType = InvoiceType.ACCREC,
        invoice_date: date | None = None,
        due_date: date | None = None,
        reference: str | None = None,
        status: InvoiceStatus = InvoiceStatus.DRAFT,
        line_amount_types: str = "Exclusive",
        currency_code: str = "USD",
    ) -> Invoice:
        """Create a new invoice."""
        invoice_data: dict[str, Any] = {
            "Type": invoice_type.value,
            "Contact": {"ContactID": contact_id},
            "LineItems": [li.to_api() for li in line_items],
            "Status": status.value,
            "LineAmountTypes": line_amount_types,
            "CurrencyCode": currency_code,
        }

        if invoice_date:
            invoice_data["Date"] = invoice_date.isoformat()
        if due_date:
            invoice_data["DueDate"] = due_date.isoformat()
        if reference:
            invoice_data["Reference"] = reference

        data = await self._request("POST", "/Invoices", json_data={"Invoices": [invoice_data]})
        return Invoice.from_api(data.get("Invoices", [{}])[0])

    async def update_invoice_status(
        self,
        invoice_id: str,
        status: InvoiceStatus,
    ) -> Invoice:
        """Update invoice status."""
        invoice_data = {
            "InvoiceID": invoice_id,
            "Status": status.value,
        }

        data = await self._request("POST", "/Invoices", json_data={"Invoices": [invoice_data]})
        return Invoice.from_api(data.get("Invoices", [{}])[0])

    async def void_invoice(self, invoice_id: str) -> Invoice:
        """Void an invoice."""
        return await self.update_invoice_status(invoice_id, InvoiceStatus.VOIDED)

    # =========================================================================
    # Accounts
    # =========================================================================

    async def get_accounts(
        self,
        where: str | None = None,
        account_type: AccountType | None = None,
    ) -> list[Account]:
        """Get chart of accounts."""
        params: dict[str, Any] = {}

        filters = []
        if where:
            filters.append(where)
        if account_type:
            filters.append(f'Type=="{account_type.value}"')

        if filters:
            params["where"] = " AND ".join(filters)

        data = await self._request("GET", "/Accounts", params=params)
        return [Account.from_api(a) for a in data.get("Accounts", [])]

    async def get_account(self, account_id: str) -> Account:
        """Get a single account."""
        data = await self._request("GET", f"/Accounts/{account_id}")
        accounts = data.get("Accounts", [])
        if not accounts:
            raise XeroError(f"Account {account_id} not found", status_code=404)
        return Account.from_api(accounts[0])

    async def get_bank_accounts(self) -> list[Account]:
        """Get bank accounts only."""
        return await self.get_accounts(account_type=AccountType.BANK)

    # =========================================================================
    # Bank Transactions
    # =========================================================================

    async def get_bank_transactions(
        self,
        page: int = 1,
        where: str | None = None,
        bank_account_id: str | None = None,
    ) -> list[BankTransaction]:
        """Get bank transactions."""
        params: dict[str, Any] = {"page": page}

        filters = []
        if where:
            filters.append(where)
        if bank_account_id:
            filters.append(f'BankAccount.AccountID==Guid("{bank_account_id}")')

        if filters:
            params["where"] = " AND ".join(filters)

        data = await self._request("GET", "/BankTransactions", params=params)
        return [BankTransaction.from_api(bt) for bt in data.get("BankTransactions", [])]

    async def create_bank_transaction(
        self,
        bank_account_id: str,
        contact_id: str,
        line_items: list[LineItem],
        transaction_type: BankTransactionType = BankTransactionType.SPEND,
        transaction_date: date | None = None,
        reference: str | None = None,
    ) -> BankTransaction:
        """Create a bank transaction."""
        tx_data: dict[str, Any] = {
            "Type": transaction_type.value,
            "BankAccount": {"AccountID": bank_account_id},
            "Contact": {"ContactID": contact_id},
            "LineItems": [li.to_api() for li in line_items],
        }

        if transaction_date:
            tx_data["Date"] = transaction_date.isoformat()
        if reference:
            tx_data["Reference"] = reference

        data = await self._request(
            "POST", "/BankTransactions", json_data={"BankTransactions": [tx_data]}
        )
        return BankTransaction.from_api(data.get("BankTransactions", [{}])[0])

    # =========================================================================
    # Payments
    # =========================================================================

    async def get_payments(self, page: int = 1, where: str | None = None) -> list[Payment]:
        """Get payments."""
        params: dict[str, Any] = {"page": page}
        if where:
            params["where"] = where

        data = await self._request("GET", "/Payments", params=params)
        return [Payment.from_api(p) for p in data.get("Payments", [])]

    async def create_payment(
        self,
        invoice_id: str,
        account_id: str,
        amount: Decimal,
        payment_date: date | None = None,
        reference: str | None = None,
    ) -> Payment:
        """Create a payment against an invoice."""
        payment_data: dict[str, Any] = {
            "Invoice": {"InvoiceID": invoice_id},
            "Account": {"AccountID": account_id},
            "Amount": str(amount),
        }

        if payment_date:
            payment_data["Date"] = payment_date.isoformat()
        if reference:
            payment_data["Reference"] = reference

        data = await self._request("POST", "/Payments", json_data={"Payments": [payment_data]})
        return Payment.from_api(data.get("Payments", [{}])[0])

    # =========================================================================
    # Manual Journals
    # =========================================================================

    async def get_manual_journals(self, page: int = 1) -> list[ManualJournal]:
        """Get manual journals."""
        params: dict[str, Any] = {"page": page}
        data = await self._request("GET", "/ManualJournals", params=params)
        return [ManualJournal.from_api(mj) for mj in data.get("ManualJournals", [])]

    async def create_manual_journal(
        self,
        narration: str,
        journal_lines: list[JournalLine],
        journal_date: date | None = None,
        status: str = "DRAFT",
    ) -> ManualJournal:
        """Create a manual journal entry."""
        journal_data: dict[str, Any] = {
            "Narration": narration,
            "JournalLines": [jl.to_api() for jl in journal_lines],
            "Status": status,
        }

        if journal_date:
            journal_data["Date"] = journal_date.isoformat()

        data = await self._request(
            "POST", "/ManualJournals", json_data={"ManualJournals": [journal_data]}
        )
        return ManualJournal.from_api(data.get("ManualJournals", [{}])[0])

    # =========================================================================
    # Organization
    # =========================================================================

    async def get_organisation(self) -> dict[str, Any]:
        """Get organisation details."""
        data = await self._request("GET", "/Organisation")
        organisations = data.get("Organisations", [])
        return organisations[0] if organisations else {}

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> XeroConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _parse_xero_datetime(value: str | None) -> datetime | None:
    """Parse Xero datetime format: /Date(1234567890000)/"""
    if not value:
        return None
    try:
        # Handle /Date(timestamp)/ format
        if value.startswith("/Date(") and value.endswith(")/"):
            # Extract timestamp (might have timezone offset like +0000)
            inner = value[6:-2]
            if "+" in inner:
                inner = inner.split("+")[0]
            elif "-" in inner and inner.count("-") == 1:
                inner = inner.rsplit("-", 1)[0]
            timestamp = int(inner) / 1000
            return datetime.fromtimestamp(timestamp)
        # Handle ISO format
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_xero_date(value: str | None) -> date | None:
    """Parse Xero date format."""
    if not value:
        return None
    try:
        dt = _parse_xero_datetime(value)
        return dt.date() if dt else None
    except (ValueError, AttributeError):
        return None


def get_mock_invoice() -> Invoice:
    """Get a mock invoice for testing."""
    return Invoice(
        invoice_id="inv-123",
        invoice_number="INV-0001",
        type=InvoiceType.ACCREC,
        status=InvoiceStatus.AUTHORISED,
        contact_name="Test Customer",
        date=date.today(),
        total=Decimal("1000.00"),
        amount_due=Decimal("1000.00"),
    )


def get_mock_contact() -> XeroContact:
    """Get a mock contact for testing."""
    return XeroContact(
        contact_id="contact-123",
        name="Test Customer",
        email="customer@example.com",
        is_customer=True,
    )

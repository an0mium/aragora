"""
QuickBooks Online Entity Operations Mixin.

Extracted from qbo.py. Contains all entity-specific operations:
customers, invoices, expenses, payments, vendors, accounts, reports,
and mock data generators.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from aragora.connectors.accounting.qbo_models import (
    QBOAccount,
    QBOCustomer,
    QBOTransaction,
    TransactionType,
)
from aragora.connectors.accounting.qbo_query import QBOQueryBuilder

logger = logging.getLogger(__name__)


class QBOOperationsMixin:
    """Mixin providing entity-specific operations for QuickBooksConnector."""

    if TYPE_CHECKING:
        _request: Any
        _validate_numeric_id: Any
        _sanitize_query_value: Any
        _credentials: Any

    # =========================================================================
    # Customer Operations
    # =========================================================================

    async def list_customers(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QBOCustomer]:
        """List customers."""
        # Use QBOQueryBuilder for safe query construction
        query = (
            QBOQueryBuilder("Customer")
            .select(
                "Id",
                "DisplayName",
                "CompanyName",
                "PrimaryEmailAddr",
                "PrimaryPhone",
                "Balance",
                "Active",
            )
            .where_eq("Active", active_only)
            .limit(limit)
            .offset(offset)
            .build()
        )

        response = await self._request("GET", f"query?query={query}")

        customers = []
        for item in response.get("QueryResponse", {}).get("Customer", []):
            customers.append(
                QBOCustomer(
                    id=item["Id"],
                    display_name=item.get("DisplayName", ""),
                    company_name=item.get("CompanyName"),
                    email=item.get("PrimaryEmailAddr", {}).get("Address"),
                    phone=item.get("PrimaryPhone", {}).get("FreeFormNumber"),
                    balance=float(item.get("Balance", 0)),
                    active=item.get("Active", True),
                )
            )

        return customers

    async def get_customer(self, customer_id: str) -> QBOCustomer | None:
        """Get customer by ID."""
        # Validate customer_id is numeric to prevent path traversal/injection
        safe_customer_id = self._validate_numeric_id(customer_id, "customer_id")
        response = await self._request("GET", f"customer/{safe_customer_id}")

        item = response.get("Customer")
        if not item:
            return None

        return QBOCustomer(
            id=item["Id"],
            display_name=item.get("DisplayName", ""),
            company_name=item.get("CompanyName"),
            email=item.get("PrimaryEmailAddr", {}).get("Address"),
            phone=item.get("PrimaryPhone", {}).get("FreeFormNumber"),
            balance=float(item.get("Balance", 0)),
            active=item.get("Active", True),
        )

    # =========================================================================
    # Transaction Operations
    # =========================================================================

    async def list_invoices(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        customer_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QBOTransaction]:
        """List invoices."""
        # Use QBOQueryBuilder for safe query construction
        builder = (
            QBOQueryBuilder("Invoice")
            .select(
                "Id",
                "DocNumber",
                "TxnDate",
                "DueDate",
                "TotalAmt",
                "Balance",
                "CustomerRef",
                "VendorRef",
                "PrivateNote",
                "Line",
            )
            .limit(limit)
            .offset(offset)
        )

        if start_date:
            builder = builder.where_gte("TxnDate", start_date)
        if end_date:
            builder = builder.where_lte("TxnDate", end_date)
        if customer_id:
            builder = builder.where_ref("CustomerRef", customer_id)

        query = builder.build()

        response = await self._request("GET", f"query?query={query}")

        transactions = []
        for item in response.get("QueryResponse", {}).get("Invoice", []):
            transactions.append(self._parse_transaction(item, TransactionType.INVOICE))

        return transactions

    async def list_expenses(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[QBOTransaction]:
        """List expenses (purchases)."""
        # Use QBOQueryBuilder for safe query construction
        builder = (
            QBOQueryBuilder("Purchase")
            .select(
                "Id",
                "DocNumber",
                "TxnDate",
                "DueDate",
                "TotalAmt",
                "Balance",
                "CustomerRef",
                "VendorRef",
                "PrivateNote",
                "Line",
            )
            .limit(limit)
            .offset(offset)
        )

        if start_date:
            builder = builder.where_gte("TxnDate", start_date)
        if end_date:
            builder = builder.where_lte("TxnDate", end_date)

        query = builder.build()

        response = await self._request("GET", f"query?query={query}")

        transactions = []
        for item in response.get("QueryResponse", {}).get("Purchase", []):
            transactions.append(self._parse_transaction(item, TransactionType.EXPENSE))

        return transactions

    def _parse_transaction(self, item: dict[str, Any], txn_type: TransactionType) -> QBOTransaction:
        """Parse transaction from API response."""
        customer_ref = item.get("CustomerRef", {})
        vendor_ref = item.get("VendorRef", {})

        return QBOTransaction(
            id=item["Id"],
            type=txn_type,
            doc_number=item.get("DocNumber"),
            txn_date=datetime.fromisoformat(item["TxnDate"]) if item.get("TxnDate") else None,
            due_date=datetime.fromisoformat(item["DueDate"]) if item.get("DueDate") else None,
            total_amount=float(item.get("TotalAmt", 0)),
            balance=float(item.get("Balance", 0)),
            customer_id=customer_ref.get("value"),
            customer_name=customer_ref.get("name"),
            vendor_id=vendor_ref.get("value"),
            vendor_name=vendor_ref.get("name"),
            memo=item.get("PrivateNote"),
            status="Paid" if float(item.get("Balance", 0)) == 0 else "Open",
            line_items=item.get("Line", []),
        )

    # =========================================================================
    # Report Operations
    # =========================================================================

    async def get_profit_loss_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get profit and loss report."""
        params = (
            f"start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"
        )

        response = await self._request("GET", f"reports/ProfitAndLoss?{params}")
        return response

    async def get_balance_sheet_report(
        self,
        as_of_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get balance sheet report."""
        date_str = (as_of_date or datetime.now()).strftime("%Y-%m-%d")
        response = await self._request("GET", f"reports/BalanceSheet?as_of={date_str}")
        return response

    async def get_accounts_receivable_aging(self) -> dict[str, Any]:
        """Get AR aging report."""
        response = await self._request("GET", "reports/AgedReceivables")
        return response

    async def get_accounts_payable_aging(self) -> dict[str, Any]:
        """Get AP aging report."""
        response = await self._request("GET", "reports/AgedPayables")
        return response

    # =========================================================================
    # Account Operations
    # =========================================================================

    async def list_accounts(
        self,
        account_type: str | None = None,
        active_only: bool = True,
    ) -> list[QBOAccount]:
        """List chart of accounts."""
        # Use QBOQueryBuilder for safe query construction
        builder = (
            QBOQueryBuilder("Account")
            .select("Id", "Name", "AccountType", "AccountSubType", "CurrentBalance", "Active")
            .where_eq("Active", active_only)
        )

        if account_type:
            safe_type = self._sanitize_query_value(account_type)
            builder = builder.where_raw(f"AccountType = '{safe_type}'")

        query = builder.build()

        response = await self._request("GET", f"query?query={query}")

        accounts = []
        for item in response.get("QueryResponse", {}).get("Account", []):
            accounts.append(
                QBOAccount(
                    id=item["Id"],
                    name=item.get("Name", ""),
                    account_type=item.get("AccountType", ""),
                    account_sub_type=item.get("AccountSubType"),
                    current_balance=float(item.get("CurrentBalance", 0)),
                    active=item.get("Active", True),
                )
            )

        return accounts

    # =========================================================================
    # Company Info
    # =========================================================================

    async def get_company_info(self) -> dict[str, Any]:
        """Get company information."""
        response = await self._request("GET", f"companyinfo/{self._credentials.realm_id}")
        return response.get("CompanyInfo", {})

    # =========================================================================
    # Vendor Operations
    # =========================================================================

    async def list_vendors(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List vendors."""
        # Use QBOQueryBuilder for safe query construction
        query = (
            QBOQueryBuilder("Vendor")
            .select(
                "Id",
                "DisplayName",
                "CompanyName",
                "PrimaryEmailAddr",
                "PrimaryPhone",
                "Balance",
                "Active",
            )
            .where_eq("Active", active_only)
            .limit(limit)
            .offset(offset)
            .build()
        )

        response = await self._request("GET", f"query?query={query}")
        return response.get("QueryResponse", {}).get("Vendor", [])

    async def get_vendor_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Get vendor by display name.

        Args:
            name: Vendor display name to search

        Returns:
            Vendor data or None if not found
        """
        # Use QBOQueryBuilder for safe query construction
        query = (
            QBOQueryBuilder("Vendor")
            .select(
                "Id",
                "DisplayName",
                "CompanyName",
                "PrimaryEmailAddr",
                "PrimaryPhone",
                "Balance",
                "Active",
            )
            .where_raw(f"DisplayName = '{self._sanitize_query_value(name)}'")
            .build()
        )

        response = await self._request("GET", f"query?query={query}")
        vendors = response.get("QueryResponse", {}).get("Vendor", [])

        return vendors[0] if vendors else None

    async def create_vendor(
        self,
        display_name: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new vendor.

        Args:
            display_name: Vendor display name
            email: Vendor email
            phone: Vendor phone

        Returns:
            Created vendor data
        """
        vendor_data: dict[str, Any] = {
            "DisplayName": display_name,
        }

        if email:
            vendor_data["PrimaryEmailAddr"] = {"Address": email}
        if phone:
            vendor_data["PrimaryPhone"] = {"FreeFormNumber": phone}

        response = await self._request("POST", "vendor", vendor_data)
        return response.get("Vendor", {})

    # =========================================================================
    # Expense/Purchase Creation
    # =========================================================================

    async def create_expense(
        self,
        vendor_id: str,
        account_id: str,
        amount: float,
        description: str | None = None,
        txn_date: datetime | None = None,
        payment_type: str = "Cash",
    ) -> dict[str, Any]:
        """
        Create an expense (Purchase) in QuickBooks.

        Args:
            vendor_id: QBO Vendor ID
            account_id: Account to charge (e.g., bank account)
            amount: Expense amount
            description: Expense description/memo
            txn_date: Transaction date
            payment_type: 'Cash', 'Check', or 'CreditCard'

        Returns:
            Created purchase data
        """
        purchase_data: dict[str, Any] = {
            "PaymentType": payment_type,
            "AccountRef": {"value": account_id},
            "TotalAmt": amount,
            "Line": [
                {
                    "Amount": amount,
                    "DetailType": "AccountBasedExpenseLineDetail",
                    "AccountBasedExpenseLineDetail": {
                        "AccountRef": {"value": account_id},
                    },
                }
            ],
        }

        if vendor_id:
            purchase_data["EntityRef"] = {"value": vendor_id, "type": "Vendor"}

        if description:
            purchase_data["PrivateNote"] = description

        if txn_date:
            purchase_data["TxnDate"] = txn_date.strftime("%Y-%m-%d")

        response = await self._request("POST", "purchase", purchase_data)
        return response.get("Purchase", {})

    async def create_bill(
        self,
        vendor_id: str,
        account_id: str,
        amount: float,
        due_date: datetime | None = None,
        description: str | None = None,
        line_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Create a bill (accounts payable) in QuickBooks.

        Args:
            vendor_id: QBO Vendor ID
            account_id: Expense account to charge
            amount: Bill total amount
            due_date: Payment due date
            description: Bill description
            line_items: Optional line items

        Returns:
            Created bill data
        """
        lines = line_items or [
            {
                "Amount": amount,
                "DetailType": "AccountBasedExpenseLineDetail",
                "AccountBasedExpenseLineDetail": {
                    "AccountRef": {"value": account_id},
                },
            }
        ]

        bill_data: dict[str, Any] = {
            "VendorRef": {"value": vendor_id},
            "Line": lines,
        }

        if due_date:
            bill_data["DueDate"] = due_date.strftime("%Y-%m-%d")

        if description:
            bill_data["PrivateNote"] = description

        response = await self._request("POST", "bill", bill_data)
        return response.get("Bill", {})

    # =========================================================================
    # Invoice Creation
    # =========================================================================

    async def create_invoice(
        self,
        customer_id: str,
        line_items: list[dict[str, Any]],
        due_date: datetime | None = None,
        memo: str | None = None,
    ) -> dict[str, Any]:
        """
        Create an invoice in QuickBooks.

        Args:
            customer_id: QBO Customer ID
            line_items: Invoice line items
            due_date: Payment due date
            memo: Invoice memo

        Returns:
            Created invoice data
        """
        invoice_data: dict[str, Any] = {
            "CustomerRef": {"value": customer_id},
            "Line": line_items,
        }

        if due_date:
            invoice_data["DueDate"] = due_date.strftime("%Y-%m-%d")

        if memo:
            invoice_data["CustomerMemo"] = {"value": memo}

        response = await self._request("POST", "invoice", invoice_data)
        return response.get("Invoice", {})

    # =========================================================================
    # Payment Operations
    # =========================================================================

    async def create_payment(
        self,
        customer_id: str,
        amount: float,
        invoice_ids: list[str] | None = None,
        payment_method: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a payment received from a customer.

        Args:
            customer_id: QBO Customer ID
            amount: Payment amount
            invoice_ids: Optional list of invoice IDs this pays
            payment_method: Payment method reference

        Returns:
            Created payment data
        """
        payment_data: dict[str, Any] = {
            "CustomerRef": {"value": customer_id},
            "TotalAmt": amount,
        }

        if invoice_ids:
            payment_data["Line"] = [
                {
                    "Amount": amount,
                    "LinkedTxn": [
                        {"TxnId": inv_id, "TxnType": "Invoice"} for inv_id in invoice_ids
                    ],
                }
            ]

        response = await self._request("POST", "payment", payment_data)
        return response.get("Payment", {})

    async def create_bill_payment(
        self,
        vendor_id: str,
        amount: float,
        bank_account_id: str,
        bill_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Record a payment to a vendor.

        Args:
            vendor_id: QBO Vendor ID
            amount: Payment amount
            bank_account_id: Bank account to pay from
            bill_ids: Optional list of bill IDs this pays

        Returns:
            Created bill payment data
        """
        payment_data: dict[str, Any] = {
            "VendorRef": {"value": vendor_id},
            "TotalAmt": amount,
            "PayType": "Check",
            "CheckPayment": {
                "BankAccountRef": {"value": bank_account_id},
            },
        }

        if bill_ids:
            payment_data["Line"] = [
                {
                    "Amount": amount,
                    "LinkedTxn": [{"TxnId": bill_id, "TxnType": "Bill"} for bill_id in bill_ids],
                }
            ]

        response = await self._request("POST", "billpayment", payment_data)
        return response.get("BillPayment", {})


# =============================================================================
# Mock Data for Demo
# =============================================================================


def get_mock_customers() -> list[QBOCustomer]:
    """Generate mock customer data for demo."""
    return [
        QBOCustomer(
            id="1",
            display_name="Acme Corporation",
            company_name="Acme Corp",
            email="billing@acme.com",
            phone="555-0100",
            balance=15420.50,
            active=True,
        ),
        QBOCustomer(
            id="2",
            display_name="TechStart Inc",
            company_name="TechStart",
            email="ap@techstart.io",
            phone="555-0200",
            balance=8750.00,
            active=True,
        ),
        QBOCustomer(
            id="3",
            display_name="Green Energy Solutions",
            company_name="Green Energy",
            email="finance@greenenergy.com",
            phone="555-0300",
            balance=22100.00,
            active=True,
        ),
    ]


def get_mock_transactions() -> list[QBOTransaction]:
    """Generate mock transaction data for demo."""
    now = datetime.now(timezone.utc)
    return [
        QBOTransaction(
            id="1001",
            type=TransactionType.INVOICE,
            doc_number="INV-1001",
            txn_date=now - timedelta(days=5),
            due_date=now + timedelta(days=25),
            total_amount=5250.00,
            balance=5250.00,
            customer_id="1",
            customer_name="Acme Corporation",
            status="Open",
        ),
        QBOTransaction(
            id="1002",
            type=TransactionType.INVOICE,
            doc_number="INV-1002",
            txn_date=now - timedelta(days=12),
            due_date=now + timedelta(days=18),
            total_amount=3800.00,
            balance=0.00,
            customer_id="2",
            customer_name="TechStart Inc",
            status="Paid",
        ),
        QBOTransaction(
            id="2001",
            type=TransactionType.EXPENSE,
            doc_number="EXP-2001",
            txn_date=now - timedelta(days=3),
            total_amount=1250.00,
            vendor_name="Office Supplies Co",
            memo="Office supplies Q1",
            status="Paid",
        ),
        QBOTransaction(
            id="2002",
            type=TransactionType.EXPENSE,
            doc_number="EXP-2002",
            txn_date=now - timedelta(days=7),
            total_amount=4500.00,
            vendor_name="Cloud Services Inc",
            memo="Monthly infrastructure",
            status="Paid",
        ),
    ]


__all__ = [
    "QBOOperationsMixin",
    "get_mock_customers",
    "get_mock_transactions",
]

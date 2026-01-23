"""
QuickBooks Online Connector.

Provides integration with QuickBooks Online for accounting operations:
- OAuth 2.0 authentication flow
- Transaction sync (invoices, payments, expenses)
- Customer and vendor management
- Report generation
- Multi-company support

Dependencies:
    pip install intuit-oauth quickbooks-python

Environment Variables:
    QBO_CLIENT_ID - QuickBooks OAuth client ID
    QBO_CLIENT_SECRET - QuickBooks OAuth client secret
    QBO_REDIRECT_URI - OAuth callback URL
    QBO_ENVIRONMENT - 'sandbox' or 'production'
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QBOEnvironment(str, Enum):
    """QuickBooks environment."""

    SANDBOX = "sandbox"
    PRODUCTION = "production"


class TransactionType(str, Enum):
    """Transaction types."""

    INVOICE = "Invoice"
    PAYMENT = "Payment"
    EXPENSE = "Expense"
    BILL = "Bill"
    CREDIT_MEMO = "CreditMemo"
    SALES_RECEIPT = "SalesReceipt"
    PURCHASE = "Purchase"
    JOURNAL_ENTRY = "JournalEntry"


@dataclass
class QBOCredentials:
    """OAuth credentials for QuickBooks."""

    access_token: str
    refresh_token: str
    realm_id: str  # Company ID
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.expires_at:
            return True
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class QBOCustomer:
    """QuickBooks customer."""

    id: str
    display_name: str
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    balance: float = 0.0
    active: bool = True
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "displayName": self.display_name,
            "companyName": self.company_name,
            "email": self.email,
            "phone": self.phone,
            "balance": self.balance,
            "active": self.active,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class QBOTransaction:
    """QuickBooks transaction."""

    id: str
    type: TransactionType
    doc_number: Optional[str] = None
    txn_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    total_amount: float = 0.0
    balance: float = 0.0
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    memo: Optional[str] = None
    status: str = "Open"
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "docNumber": self.doc_number,
            "txnDate": self.txn_date.isoformat() if self.txn_date else None,
            "dueDate": self.due_date.isoformat() if self.due_date else None,
            "totalAmount": self.total_amount,
            "balance": self.balance,
            "customerId": self.customer_id,
            "customerName": self.customer_name,
            "vendorId": self.vendor_id,
            "vendorName": self.vendor_name,
            "memo": self.memo,
            "status": self.status,
            "lineItems": self.line_items,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class QBOAccount:
    """QuickBooks account (chart of accounts)."""

    id: str
    name: str
    account_type: str
    account_sub_type: Optional[str] = None
    current_balance: float = 0.0
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "accountType": self.account_type,
            "accountSubType": self.account_sub_type,
            "currentBalance": self.current_balance,
            "active": self.active,
        }


class QuickBooksConnector:
    """
    QuickBooks Online integration connector.

    Handles OAuth authentication and API operations.
    """

    BASE_URL_SANDBOX = "https://sandbox-quickbooks.api.intuit.com"
    BASE_URL_PRODUCTION = "https://quickbooks.api.intuit.com"
    AUTH_URL = "https://appcenter.intuit.com/connect/oauth2"
    TOKEN_URL = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        environment: QBOEnvironment = QBOEnvironment.SANDBOX,
    ):
        """
        Initialize QuickBooks connector.

        Args:
            client_id: OAuth client ID (or from QBO_CLIENT_ID env var)
            client_secret: OAuth client secret (or from QBO_CLIENT_SECRET env var)
            redirect_uri: OAuth callback URL (or from QBO_REDIRECT_URI env var)
            environment: Sandbox or production
        """
        self.client_id = client_id or os.getenv("QBO_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("QBO_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("QBO_REDIRECT_URI")
        self.environment = environment

        env_str = os.getenv("QBO_ENVIRONMENT", "sandbox").lower()
        if env_str == "production":
            self.environment = QBOEnvironment.PRODUCTION

        self.base_url = (
            self.BASE_URL_PRODUCTION
            if self.environment == QBOEnvironment.PRODUCTION
            else self.BASE_URL_SANDBOX
        )

        self._credentials: Optional[QBOCredentials] = None
        self._http_client: Optional[Any] = None

    @property
    def is_configured(self) -> bool:
        """Check if connector is configured."""
        return bool(self.client_id and self.client_secret and self.redirect_uri)

    @property
    def is_authenticated(self) -> bool:
        """Check if connector has valid credentials."""
        return self._credentials is not None and not self._credentials.is_expired

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL to redirect user to
        """
        import urllib.parse

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": "com.intuit.quickbooks.accounting",
            "redirect_uri": self.redirect_uri,
        }
        if state:
            params["state"] = state

        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    async def exchange_code(
        self,
        authorization_code: str,
        realm_id: str,
    ) -> QBOCredentials:
        """
        Exchange authorization code for tokens.

        Args:
            authorization_code: Code from OAuth callback
            realm_id: QuickBooks company ID

        Returns:
            OAuth credentials
        """
        import aiohttp
        import base64

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": self.redirect_uri,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token exchange failed: {error_text}")

                data = await response.json()

                self._credentials = QBOCredentials(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    realm_id=realm_id,
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=data.get("expires_in", 3600)),
                )

                return self._credentials

    async def refresh_tokens(self) -> QBOCredentials:
        """Refresh OAuth tokens."""
        if not self._credentials:
            raise Exception("No credentials to refresh")

        import aiohttp
        import base64

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._credentials.refresh_token,
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token refresh failed: {error_text}")

                data = await response.json()

                self._credentials = QBOCredentials(
                    access_token=data["access_token"],
                    refresh_token=data["refresh_token"],
                    realm_id=self._credentials.realm_id,
                    token_type=data.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=data.get("expires_in", 3600)),
                )

                return self._credentials

    def set_credentials(self, credentials: QBOCredentials) -> None:
        """Set credentials (e.g., from storage)."""
        self._credentials = credentials

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        if not self._credentials:
            raise Exception("Not authenticated")

        # Refresh if expired
        if self._credentials.is_expired:
            await self.refresh_tokens()

        import aiohttp

        url = f"{self.base_url}/v3/company/{self._credentials.realm_id}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self._credentials.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=data,
            ) as response:
                response_data = await response.json()

                if response.status >= 400:
                    error = response_data.get("Fault", {}).get("Error", [{}])[0]
                    raise Exception(f"QBO API error: {error.get('Message', 'Unknown error')}")

                return response_data

    # =========================================================================
    # Customer Operations
    # =========================================================================

    async def list_customers(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QBOCustomer]:
        """List customers."""
        query = f"SELECT * FROM Customer WHERE Active = {str(active_only).lower()} MAXRESULTS {limit} STARTPOSITION {offset + 1}"

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

    async def get_customer(self, customer_id: str) -> Optional[QBOCustomer]:
        """Get customer by ID."""
        response = await self._request("GET", f"customer/{customer_id}")

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
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        customer_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QBOTransaction]:
        """List invoices."""
        conditions = []

        if start_date:
            conditions.append(f"TxnDate >= '{start_date.strftime('%Y-%m-%d')}'")
        if end_date:
            conditions.append(f"TxnDate <= '{end_date.strftime('%Y-%m-%d')}'")
        if customer_id:
            conditions.append(f"CustomerRef = '{customer_id}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM Invoice WHERE {where_clause} MAXRESULTS {limit} STARTPOSITION {offset + 1}"

        response = await self._request("GET", f"query?query={query}")

        transactions = []
        for item in response.get("QueryResponse", {}).get("Invoice", []):
            transactions.append(self._parse_transaction(item, TransactionType.INVOICE))

        return transactions

    async def list_expenses(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QBOTransaction]:
        """List expenses (purchases)."""
        conditions = []

        if start_date:
            conditions.append(f"TxnDate >= '{start_date.strftime('%Y-%m-%d')}'")
        if end_date:
            conditions.append(f"TxnDate <= '{end_date.strftime('%Y-%m-%d')}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM Purchase WHERE {where_clause} MAXRESULTS {limit} STARTPOSITION {offset + 1}"

        response = await self._request("GET", f"query?query={query}")

        transactions = []
        for item in response.get("QueryResponse", {}).get("Purchase", []):
            transactions.append(self._parse_transaction(item, TransactionType.EXPENSE))

        return transactions

    def _parse_transaction(self, item: Dict[str, Any], txn_type: TransactionType) -> QBOTransaction:
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
    ) -> Dict[str, Any]:
        """Get profit and loss report."""
        params = (
            f"start_date={start_date.strftime('%Y-%m-%d')}"
            f"&end_date={end_date.strftime('%Y-%m-%d')}"
        )

        response = await self._request("GET", f"reports/ProfitAndLoss?{params}")
        return response

    async def get_balance_sheet_report(
        self,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get balance sheet report."""
        date_str = (as_of_date or datetime.now()).strftime("%Y-%m-%d")
        response = await self._request("GET", f"reports/BalanceSheet?as_of={date_str}")
        return response

    async def get_accounts_receivable_aging(self) -> Dict[str, Any]:
        """Get AR aging report."""
        response = await self._request("GET", "reports/AgedReceivables")
        return response

    async def get_accounts_payable_aging(self) -> Dict[str, Any]:
        """Get AP aging report."""
        response = await self._request("GET", "reports/AgedPayables")
        return response

    # =========================================================================
    # Account Operations
    # =========================================================================

    async def list_accounts(
        self,
        account_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[QBOAccount]:
        """List chart of accounts."""
        conditions = [f"Active = {str(active_only).lower()}"]
        if account_type:
            conditions.append(f"AccountType = '{account_type}'")

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM Account WHERE {where_clause}"

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

    async def get_company_info(self) -> Dict[str, Any]:
        """Get company information."""
        response = await self._request("GET", f"companyinfo/{self._credentials.realm_id}")  # type: ignore
        return response.get("CompanyInfo", {})


# =============================================================================
# Mock Data for Demo
# =============================================================================


def get_mock_customers() -> List[QBOCustomer]:
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


def get_mock_transactions() -> List[QBOTransaction]:
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

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
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Make authenticated API request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            max_retries: Maximum retry attempts (default 3)
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            API response data

        Raises:
            Exception: If request fails after all retries
        """
        import asyncio

        import aiohttp

        if not self._credentials:
            raise Exception("Not authenticated")

        # Refresh if expired
        if self._credentials.is_expired:
            await self.refresh_tokens()

        url = f"{self.base_url}/v3/company/{self._credentials.realm_id}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self._credentials.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Retryable status codes
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        # Handle rate limiting with Retry-After header
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            if retry_after and attempt < max_retries:
                                delay = float(retry_after)
                                logger.warning(f"QBO rate limited, waiting {delay}s")
                                await asyncio.sleep(delay)
                                continue

                        # Retry on server errors
                        if response.status in retryable_statuses and attempt < max_retries:
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                f"QBO request failed ({response.status}), "
                                f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue

                        response_data = await response.json()

                        if response.status >= 400:
                            error = response_data.get("Fault", {}).get("Error", [{}])[0]
                            raise Exception(f"QBO API error: {error.get('Message', 'Unknown error')}")

                        return response_data

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"QBO connection error: {e}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise Exception(f"QBO connection failed after {max_retries} retries: {e}")

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError("Request timed out")
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"QBO request timeout, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise Exception(f"QBO request timed out after {max_retries} retries")

        # Should not reach here, but just in case
        raise last_error or Exception("QBO request failed")

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

    # =========================================================================
    # Vendor Operations
    # =========================================================================

    async def list_vendors(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List vendors."""
        query = f"SELECT * FROM Vendor WHERE Active = {str(active_only).lower()} MAXRESULTS {limit} STARTPOSITION {offset + 1}"
        response = await self._request("GET", f"query?query={query}")
        return response.get("QueryResponse", {}).get("Vendor", [])

    async def get_vendor_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get vendor by display name.

        Args:
            name: Vendor display name to search

        Returns:
            Vendor data or None if not found
        """
        # Escape single quotes in name
        safe_name = name.replace("'", "\\'")
        query = f"SELECT * FROM Vendor WHERE DisplayName = '{safe_name}'"

        response = await self._request("GET", f"query?query={query}")
        vendors = response.get("QueryResponse", {}).get("Vendor", [])

        return vendors[0] if vendors else None

    async def create_vendor(
        self,
        display_name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new vendor.

        Args:
            display_name: Vendor display name
            email: Vendor email
            phone: Vendor phone

        Returns:
            Created vendor data
        """
        vendor_data: Dict[str, Any] = {
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
        description: Optional[str] = None,
        txn_date: Optional[datetime] = None,
        payment_type: str = "Cash",
    ) -> Dict[str, Any]:
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
        purchase_data: Dict[str, Any] = {
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
        due_date: Optional[datetime] = None,
        description: Optional[str] = None,
        line_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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

        bill_data: Dict[str, Any] = {
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
        line_items: List[Dict[str, Any]],
        due_date: Optional[datetime] = None,
        memo: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        invoice_data: Dict[str, Any] = {
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
        invoice_ids: Optional[List[str]] = None,
        payment_method: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        payment_data: Dict[str, Any] = {
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
        bill_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
        payment_data: Dict[str, Any] = {
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

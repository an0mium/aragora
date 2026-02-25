"""
QuickBooks Online Data Models and Enums.

Extracted from qbo.py. Contains all dataclass models and enums used by
the QuickBooks connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.connectors.model_base import ConnectorDataclass


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
    token_type: str = "Bearer"  # noqa: S105 -- OAuth2 token type
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.expires_at:
            return True
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class QBOCustomer(ConnectorDataclass):
    """QuickBooks customer."""

    _field_mapping = {
        "display_name": "displayName",
        "company_name": "companyName",
        "created_at": "createdAt",
    }
    _include_none = True

    id: str
    display_name: str
    company_name: str | None = None
    email: str | None = None
    phone: str | None = None
    balance: float = 0.0
    active: bool = True
    created_at: datetime | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class QBOTransaction(ConnectorDataclass):
    """QuickBooks transaction."""

    _field_mapping = {
        "doc_number": "docNumber",
        "txn_date": "txnDate",
        "due_date": "dueDate",
        "total_amount": "totalAmount",
        "customer_id": "customerId",
        "customer_name": "customerName",
        "vendor_id": "vendorId",
        "vendor_name": "vendorName",
        "line_items": "lineItems",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
    }
    _include_none = True

    id: str
    type: TransactionType
    doc_number: str | None = None
    txn_date: datetime | None = None
    due_date: datetime | None = None
    total_amount: float = 0.0
    balance: float = 0.0
    customer_id: str | None = None
    customer_name: str | None = None
    vendor_id: str | None = None
    vendor_name: str | None = None
    memo: str | None = None
    status: str = "Open"
    line_items: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


@dataclass
class QBOAccount(ConnectorDataclass):
    """QuickBooks account (chart of accounts)."""

    _field_mapping = {
        "account_type": "accountType",
        "account_sub_type": "accountSubType",
        "current_balance": "currentBalance",
    }
    _include_none = True

    id: str
    name: str
    account_type: str
    account_sub_type: str | None = None
    current_balance: float = 0.0
    active: bool = True

    def to_dict(self, exclude=None, use_api_names=True) -> dict[str, Any]:
        return super().to_dict(exclude=exclude, use_api_names=use_api_names)


__all__ = [
    "QBOEnvironment",
    "TransactionType",
    "QBOCredentials",
    "QBOCustomer",
    "QBOTransaction",
    "QBOAccount",
]

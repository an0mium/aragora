"""
Square Payment Connector.

Full integration with Square API:
- Payments (create, complete, cancel)
- Customers (create, update, search)
- Subscriptions (create, manage, cancel)
- Invoices (create, publish, pay)
- Catalog (items, variations, categories)
- Webhooks for real-time notifications
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class SquareEnvironment(str, Enum):
    """Square environment."""

    SANDBOX = "sandbox"
    PRODUCTION = "production"


class PaymentStatus(str, Enum):
    """Square payment status."""

    APPROVED = "APPROVED"
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"


class CardBrand(str, Enum):
    """Card brands."""

    VISA = "VISA"
    MASTERCARD = "MASTERCARD"
    AMERICAN_EXPRESS = "AMERICAN_EXPRESS"
    DISCOVER = "DISCOVER"
    DISCOVER_DINERS = "DISCOVER_DINERS"
    JCB = "JCB"
    CHINA_UNIONPAY = "CHINA_UNIONPAY"
    SQUARE_GIFT_CARD = "SQUARE_GIFT_CARD"
    SQUARE_CAPITAL_CARD = "SQUARE_CAPITAL_CARD"
    OTHER_BRAND = "OTHER_BRAND"


class SubscriptionStatus(str, Enum):
    """Square subscription status."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CANCELED = "CANCELED"
    DEACTIVATED = "DEACTIVATED"
    PAUSED = "PAUSED"


class InvoiceStatus(str, Enum):
    """Square invoice status."""

    DRAFT = "DRAFT"
    SCHEDULED = "SCHEDULED"
    PUBLISHED = "PUBLISHED"
    PARTIALLY_PAID = "PARTIALLY_PAID"
    PAID = "PAID"
    PARTIALLY_REFUNDED = "PARTIALLY_REFUNDED"
    REFUNDED = "REFUNDED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"
    UNPAID = "UNPAID"


class CatalogObjectType(str, Enum):
    """Catalog object types."""

    ITEM = "ITEM"
    ITEM_VARIATION = "ITEM_VARIATION"
    CATEGORY = "CATEGORY"
    DISCOUNT = "DISCOUNT"
    TAX = "TAX"
    MODIFIER = "MODIFIER"
    MODIFIER_LIST = "MODIFIER_LIST"
    IMAGE = "IMAGE"


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class SquareCredentials:
    """Square API credentials."""

    access_token: str
    environment: SquareEnvironment = SquareEnvironment.SANDBOX
    application_id: Optional[str] = None
    location_id: Optional[str] = None
    webhook_signature_key: Optional[str] = None

    @property
    def base_url(self) -> str:
        """Get base URL for environment."""
        if self.environment == SquareEnvironment.PRODUCTION:
            return "https://connect.squareup.com"
        return "https://connect.squareupsandbox.com"

    @classmethod
    def from_env(cls, prefix: str = "SQUARE_") -> "SquareCredentials":
        """Load credentials from environment variables."""
        import os

        access_token = os.environ.get(f"{prefix}ACCESS_TOKEN", "")
        environment = os.environ.get(f"{prefix}ENVIRONMENT", "sandbox").lower()
        application_id = os.environ.get(f"{prefix}APPLICATION_ID")
        location_id = os.environ.get(f"{prefix}LOCATION_ID")
        webhook_signature_key = os.environ.get(f"{prefix}WEBHOOK_SIGNATURE_KEY")

        if not access_token:
            raise ValueError(f"Missing {prefix}ACCESS_TOKEN")

        return cls(
            access_token=access_token,
            environment=SquareEnvironment(environment),
            application_id=application_id,
            location_id=location_id,
            webhook_signature_key=webhook_signature_key,
        )


# =============================================================================
# Error Handling
# =============================================================================


class SquareError(Exception):
    """Square API error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        category: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.category = category
        self.detail = detail


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Money:
    """Square money amount (in smallest currency unit)."""

    amount: int  # In smallest currency unit (e.g., cents for USD)
    currency: str = "USD"

    @classmethod
    def from_api(cls, data: Optional[Dict[str, Any]]) -> Optional["Money"]:
        if not data:
            return None
        return cls(
            amount=data.get("amount", 0),
            currency=data.get("currency", "USD"),
        )

    def to_api(self) -> Dict[str, Any]:
        return {"amount": self.amount, "currency": self.currency}

    @classmethod
    def usd(cls, dollars: float) -> "Money":
        """Create USD money from dollars."""
        return cls(amount=int(dollars * 100), currency="USD")

    @property
    def as_dollars(self) -> float:
        """Get amount in dollars."""
        return self.amount / 100


@dataclass
class Address:
    """Address."""

    address_line_1: Optional[str] = None
    address_line_2: Optional[str] = None
    locality: Optional[str] = None  # City
    administrative_district_level_1: Optional[str] = None  # State
    postal_code: Optional[str] = None
    country: str = "US"

    @classmethod
    def from_api(cls, data: Optional[Dict[str, Any]]) -> Optional["Address"]:
        if not data:
            return None
        return cls(
            address_line_1=data.get("address_line_1"),
            address_line_2=data.get("address_line_2"),
            locality=data.get("locality"),
            administrative_district_level_1=data.get("administrative_district_level_1"),
            postal_code=data.get("postal_code"),
            country=data.get("country", "US"),
        )

    def to_api(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"country": self.country}
        if self.address_line_1:
            result["address_line_1"] = self.address_line_1
        if self.address_line_2:
            result["address_line_2"] = self.address_line_2
        if self.locality:
            result["locality"] = self.locality
        if self.administrative_district_level_1:
            result["administrative_district_level_1"] = self.administrative_district_level_1
        if self.postal_code:
            result["postal_code"] = self.postal_code
        return result


@dataclass
class Card:
    """Card details."""

    id: Optional[str] = None
    card_brand: Optional[CardBrand] = None
    last_4: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    cardholder_name: Optional[str] = None
    billing_address: Optional[Address] = None
    fingerprint: Optional[str] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Card":
        return cls(
            id=data.get("id"),
            card_brand=CardBrand(data["card_brand"]) if data.get("card_brand") else None,
            last_4=data.get("last_4"),
            exp_month=data.get("exp_month"),
            exp_year=data.get("exp_year"),
            cardholder_name=data.get("cardholder_name"),
            billing_address=Address.from_api(data.get("billing_address")),
            fingerprint=data.get("fingerprint"),
        )


@dataclass
class Customer:
    """Square customer."""

    id: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    email_address: Optional[str] = None
    phone_number: Optional[str] = None
    company_name: Optional[str] = None
    address: Optional[Address] = None
    note: Optional[str] = None
    reference_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    creation_source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Customer":
        return cls(
            id=data["id"],
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            email_address=data.get("email_address"),
            phone_number=data.get("phone_number"),
            company_name=data.get("company_name"),
            address=Address.from_api(data.get("address")),
            note=data.get("note"),
            reference_id=data.get("reference_id"),
            preferences=data.get("preferences"),
            creation_source=data.get("creation_source"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )

    @property
    def full_name(self) -> str:
        parts = [p for p in [self.given_name, self.family_name] if p]
        return " ".join(parts)


@dataclass
class Payment:
    """Square payment."""

    id: str
    status: PaymentStatus
    amount_money: Optional[Money] = None
    tip_money: Optional[Money] = None
    total_money: Optional[Money] = None
    app_fee_money: Optional[Money] = None
    processing_fee: Optional[List[Dict[str, Any]]] = None
    source_type: Optional[str] = None
    card_details: Optional[Dict[str, Any]] = None
    location_id: Optional[str] = None
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    reference_id: Optional[str] = None
    note: Optional[str] = None
    receipt_number: Optional[str] = None
    receipt_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Payment":
        return cls(
            id=data["id"],
            status=PaymentStatus(data.get("status", "PENDING")),
            amount_money=Money.from_api(data.get("amount_money")),
            tip_money=Money.from_api(data.get("tip_money")),
            total_money=Money.from_api(data.get("total_money")),
            app_fee_money=Money.from_api(data.get("app_fee_money")),
            processing_fee=data.get("processing_fee"),
            source_type=data.get("source_type"),
            card_details=data.get("card_details"),
            location_id=data.get("location_id"),
            order_id=data.get("order_id"),
            customer_id=data.get("customer_id"),
            reference_id=data.get("reference_id"),
            note=data.get("note"),
            receipt_number=data.get("receipt_number"),
            receipt_url=data.get("receipt_url"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Refund:
    """Square refund."""

    id: str
    payment_id: str
    status: str
    amount_money: Optional[Money] = None
    processing_fee: Optional[List[Dict[str, Any]]] = None
    reason: Optional[str] = None
    location_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Refund":
        return cls(
            id=data["id"],
            payment_id=data.get("payment_id", ""),
            status=data.get("status", "PENDING"),
            amount_money=Money.from_api(data.get("amount_money")),
            processing_fee=data.get("processing_fee"),
            reason=data.get("reason"),
            location_id=data.get("location_id"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class SubscriptionPlan:
    """Square subscription plan (catalog item with subscription data)."""

    id: str
    name: str
    subscription_plan_data: Optional[Dict[str, Any]] = None
    phases: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "SubscriptionPlan":
        plan_data = data.get("subscription_plan_data", {})
        return cls(
            id=data["id"],
            name=plan_data.get("name", data.get("id", "")),
            subscription_plan_data=plan_data,
            phases=plan_data.get("phases", []),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Subscription:
    """Square subscription."""

    id: str
    status: SubscriptionStatus
    plan_id: str
    customer_id: str
    location_id: Optional[str] = None
    start_date: Optional[str] = None
    canceled_date: Optional[str] = None
    charged_through_date: Optional[str] = None
    invoice_ids: List[str] = field(default_factory=list)
    price_override_money: Optional[Money] = None
    card_id: Optional[str] = None
    timezone: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Subscription":
        return cls(
            id=data["id"],
            status=SubscriptionStatus(data.get("status", "PENDING")),
            plan_id=data.get("plan_id", ""),
            customer_id=data.get("customer_id", ""),
            location_id=data.get("location_id"),
            start_date=data.get("start_date"),
            canceled_date=data.get("canceled_date"),
            charged_through_date=data.get("charged_through_date"),
            invoice_ids=data.get("invoice_ids", []),
            price_override_money=Money.from_api(data.get("price_override_money")),
            card_id=data.get("card_id"),
            timezone=data.get("timezone"),
            source=data.get("source"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Invoice:
    """Square invoice."""

    id: str
    version: int
    status: InvoiceStatus
    location_id: str
    order_id: Optional[str] = None
    primary_recipient: Optional[Dict[str, Any]] = None
    payment_requests: List[Dict[str, Any]] = field(default_factory=list)
    invoice_number: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_at: Optional[str] = None
    public_url: Optional[str] = None
    next_payment_amount_money: Optional[Money] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Invoice":
        return cls(
            id=data["id"],
            version=data.get("version", 0),
            status=InvoiceStatus(data.get("status", "DRAFT")),
            location_id=data.get("location_id", ""),
            order_id=data.get("order_id"),
            primary_recipient=data.get("primary_recipient"),
            payment_requests=data.get("payment_requests", []),
            invoice_number=data.get("invoice_number"),
            title=data.get("title"),
            description=data.get("description"),
            scheduled_at=data.get("scheduled_at"),
            public_url=data.get("public_url"),
            next_payment_amount_money=Money.from_api(data.get("next_payment_amount_money")),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class CatalogItem:
    """Square catalog item."""

    id: str
    type: CatalogObjectType
    name: Optional[str] = None
    description: Optional[str] = None
    abbreviation: Optional[str] = None
    variations: List[Dict[str, Any]] = field(default_factory=list)
    category_id: Optional[str] = None
    tax_ids: List[str] = field(default_factory=list)
    modifier_list_ids: List[str] = field(default_factory=list)
    is_deleted: bool = False
    present_at_all_locations: bool = True
    present_at_location_ids: List[str] = field(default_factory=list)
    version: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "CatalogItem":
        item_data = data.get("item_data", {})
        return cls(
            id=data["id"],
            type=CatalogObjectType(data.get("type", "ITEM")),
            name=item_data.get("name"),
            description=item_data.get("description"),
            abbreviation=item_data.get("abbreviation"),
            variations=item_data.get("variations", []),
            category_id=item_data.get("category_id"),
            tax_ids=item_data.get("tax_ids", []),
            modifier_list_ids=item_data.get("modifier_list_ids", []),
            is_deleted=data.get("is_deleted", False),
            present_at_all_locations=data.get("present_at_all_locations", True),
            present_at_location_ids=data.get("present_at_location_ids", []),
            version=data.get("version"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse RFC 3339 datetime from API response."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _generate_idempotency_key() -> str:
    """Generate a unique idempotency key."""
    return str(uuid.uuid4())


# =============================================================================
# Square Client
# =============================================================================


class SquareClient:
    """
    Square API client.

    Provides full access to Square payments including:
    - Payments (create, complete, cancel, refund)
    - Customers (create, update, search)
    - Subscriptions (create, manage, cancel)
    - Invoices (create, publish, pay)
    - Catalog (items, variations, categories)
    - Webhook verification

    Example:
        async with SquareClient(credentials) as client:
            # Create a customer
            customer = await client.create_customer(
                given_name="John",
                family_name="Doe",
                email_address="john@example.com",
            )

            # Create a payment
            payment = await client.create_payment(
                source_id="cnon:card-nonce-ok",
                amount=Money.usd(19.99),
                customer_id=customer.id,
            )
    """

    API_VERSION = "2024-01-18"

    def __init__(self, credentials: SquareCredentials):
        self.credentials = credentials
        self._client: Optional["httpx.AsyncClient"] = None

    async def __aenter__(self) -> "SquareClient":
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self.credentials.base_url,
            headers={
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Square-Version": self.API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        response = await self._client.request(
            method,
            endpoint,
            json=json,
            params=params,
        )

        result = response.json() if response.content else {}

        if response.status_code >= 400:
            errors = result.get("errors", [])
            error = errors[0] if errors else {}
            raise SquareError(
                message=error.get("detail", f"HTTP {response.status_code}"),
                status_code=response.status_code,
                code=error.get("code"),
                category=error.get("category"),
                detail=error.get("detail"),
            )

        return result

    # -------------------------------------------------------------------------
    # Payments
    # -------------------------------------------------------------------------

    async def create_payment(
        self,
        source_id: str,
        amount: Money,
        idempotency_key: Optional[str] = None,
        customer_id: Optional[str] = None,
        location_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        note: Optional[str] = None,
        autocomplete: bool = True,
        tip_money: Optional[Money] = None,
        app_fee_money: Optional[Money] = None,
        buyer_email_address: Optional[str] = None,
    ) -> Payment:
        """
        Create a payment.

        Args:
            source_id: Payment source (card nonce, customer card ID, etc.)
            amount: Payment amount
            idempotency_key: Unique key for request (auto-generated if not provided)
            customer_id: Associated customer
            location_id: Location for the payment
            reference_id: Your reference ID
            note: Payment note
            autocomplete: Automatically complete the payment
            tip_money: Tip amount
            app_fee_money: Application fee
            buyer_email_address: Buyer's email for receipt

        Returns:
            Created Payment
        """
        body: Dict[str, Any] = {
            "source_id": source_id,
            "amount_money": amount.to_api(),
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
            "autocomplete": autocomplete,
        }

        if location_id or self.credentials.location_id:
            body["location_id"] = location_id or self.credentials.location_id
        if customer_id:
            body["customer_id"] = customer_id
        if reference_id:
            body["reference_id"] = reference_id
        if note:
            body["note"] = note
        if tip_money:
            body["tip_money"] = tip_money.to_api()
        if app_fee_money:
            body["app_fee_money"] = app_fee_money.to_api()
        if buyer_email_address:
            body["buyer_email_address"] = buyer_email_address

        data = await self._request("POST", "/v2/payments", json=body)
        return Payment.from_api(data["payment"])

    async def get_payment(self, payment_id: str) -> Payment:
        """Get payment details."""
        data = await self._request("GET", f"/v2/payments/{payment_id}")
        return Payment.from_api(data["payment"])

    async def list_payments(
        self,
        begin_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sort_order: str = "DESC",
        cursor: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[List[Payment], Optional[str]]:
        """
        List payments.

        Returns:
            Tuple of (payments, next_cursor)
        """
        params: Dict[str, Any] = {
            "sort_order": sort_order,
            "limit": min(limit, 100),
        }

        if begin_time:
            params["begin_time"] = begin_time
        if end_time:
            params["end_time"] = end_time
        if cursor:
            params["cursor"] = cursor
        if location_id or self.credentials.location_id:
            params["location_id"] = location_id or self.credentials.location_id

        data = await self._request("GET", "/v2/payments", params=params)
        payments = [Payment.from_api(p) for p in data.get("payments", [])]
        return payments, data.get("cursor")

    async def complete_payment(self, payment_id: str) -> Payment:
        """Complete a delayed payment."""
        data = await self._request("POST", f"/v2/payments/{payment_id}/complete")
        return Payment.from_api(data["payment"])

    async def cancel_payment(self, payment_id: str) -> Payment:
        """Cancel a payment."""
        data = await self._request("POST", f"/v2/payments/{payment_id}/cancel")
        return Payment.from_api(data["payment"])

    async def refund_payment(
        self,
        payment_id: str,
        amount: Money,
        idempotency_key: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Refund:
        """
        Refund a payment.

        Args:
            payment_id: Payment to refund
            amount: Refund amount
            idempotency_key: Unique key for request
            reason: Refund reason

        Returns:
            Refund details
        """
        body: Dict[str, Any] = {
            "payment_id": payment_id,
            "amount_money": amount.to_api(),
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
        }

        if reason:
            body["reason"] = reason

        data = await self._request("POST", "/v2/refunds", json=body)
        return Refund.from_api(data["refund"])

    async def get_refund(self, refund_id: str) -> Refund:
        """Get refund details."""
        data = await self._request("GET", f"/v2/refunds/{refund_id}")
        return Refund.from_api(data["refund"])

    # -------------------------------------------------------------------------
    # Customers
    # -------------------------------------------------------------------------

    async def create_customer(
        self,
        idempotency_key: Optional[str] = None,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        company_name: Optional[str] = None,
        address: Optional[Address] = None,
        note: Optional[str] = None,
        reference_id: Optional[str] = None,
    ) -> Customer:
        """Create a customer."""
        body: Dict[str, Any] = {
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
        }

        if given_name:
            body["given_name"] = given_name
        if family_name:
            body["family_name"] = family_name
        if email_address:
            body["email_address"] = email_address
        if phone_number:
            body["phone_number"] = phone_number
        if company_name:
            body["company_name"] = company_name
        if address:
            body["address"] = address.to_api()
        if note:
            body["note"] = note
        if reference_id:
            body["reference_id"] = reference_id

        data = await self._request("POST", "/v2/customers", json=body)
        return Customer.from_api(data["customer"])

    async def get_customer(self, customer_id: str) -> Customer:
        """Get customer details."""
        data = await self._request("GET", f"/v2/customers/{customer_id}")
        return Customer.from_api(data["customer"])

    async def update_customer(
        self,
        customer_id: str,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        company_name: Optional[str] = None,
        address: Optional[Address] = None,
        note: Optional[str] = None,
    ) -> Customer:
        """Update a customer."""
        body: Dict[str, Any] = {}

        if given_name:
            body["given_name"] = given_name
        if family_name:
            body["family_name"] = family_name
        if email_address:
            body["email_address"] = email_address
        if phone_number:
            body["phone_number"] = phone_number
        if company_name:
            body["company_name"] = company_name
        if address:
            body["address"] = address.to_api()
        if note:
            body["note"] = note

        data = await self._request("PUT", f"/v2/customers/{customer_id}", json=body)
        return Customer.from_api(data["customer"])

    async def delete_customer(self, customer_id: str) -> None:
        """Delete a customer."""
        await self._request("DELETE", f"/v2/customers/{customer_id}")

    async def search_customers(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[Customer], Optional[str]]:
        """
        Search customers.

        Args:
            query: Search query (filter, sort)
            limit: Maximum results
            cursor: Pagination cursor

        Returns:
            Tuple of (customers, next_cursor)
        """
        body: Dict[str, Any] = {"limit": min(limit, 100)}

        if query:
            body["query"] = query
        if cursor:
            body["cursor"] = cursor

        data = await self._request("POST", "/v2/customers/search", json=body)
        customers = [Customer.from_api(c) for c in data.get("customers", [])]
        return customers, data.get("cursor")

    # -------------------------------------------------------------------------
    # Customer Cards
    # -------------------------------------------------------------------------

    async def create_customer_card(
        self,
        customer_id: str,
        card_nonce: str,
        billing_address: Optional[Address] = None,
        cardholder_name: Optional[str] = None,
    ) -> Card:
        """Add a card to a customer."""
        body: Dict[str, Any] = {"card_nonce": card_nonce}

        if billing_address:
            body["billing_address"] = billing_address.to_api()
        if cardholder_name:
            body["cardholder_name"] = cardholder_name

        data = await self._request("POST", f"/v2/customers/{customer_id}/cards", json=body)
        return Card.from_api(data["card"])

    async def delete_customer_card(self, customer_id: str, card_id: str) -> None:
        """Delete a customer's card."""
        await self._request("DELETE", f"/v2/customers/{customer_id}/cards/{card_id}")

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def create_subscription(
        self,
        plan_id: str,
        customer_id: str,
        location_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        start_date: Optional[str] = None,
        card_id: Optional[str] = None,
        timezone: Optional[str] = None,
        price_override_money: Optional[Money] = None,
    ) -> Subscription:
        """
        Create a subscription.

        Args:
            plan_id: Subscription plan ID
            customer_id: Customer ID
            location_id: Location ID
            idempotency_key: Unique key for request
            start_date: Start date (YYYY-MM-DD)
            card_id: Customer card ID for billing
            timezone: Billing timezone
            price_override_money: Override the plan price

        Returns:
            Created Subscription
        """
        body: Dict[str, Any] = {
            "plan_id": plan_id,
            "customer_id": customer_id,
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
        }

        if location_id or self.credentials.location_id:
            body["location_id"] = location_id or self.credentials.location_id
        if start_date:
            body["start_date"] = start_date
        if card_id:
            body["card_id"] = card_id
        if timezone:
            body["timezone"] = timezone
        if price_override_money:
            body["price_override_money"] = price_override_money.to_api()

        data = await self._request("POST", "/v2/subscriptions", json=body)
        return Subscription.from_api(data["subscription"])

    async def get_subscription(self, subscription_id: str) -> Subscription:
        """Get subscription details."""
        data = await self._request("GET", f"/v2/subscriptions/{subscription_id}")
        return Subscription.from_api(data["subscription"])

    async def search_subscriptions(
        self,
        customer_ids: Optional[List[str]] = None,
        location_ids: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[List[Subscription], Optional[str]]:
        """
        Search subscriptions.

        Returns:
            Tuple of (subscriptions, next_cursor)
        """
        body: Dict[str, Any] = {"limit": min(limit, 100)}

        query: Dict[str, Any] = {"filter": {}}
        if customer_ids:
            query["filter"]["customer_ids"] = customer_ids
        if location_ids:
            query["filter"]["location_ids"] = location_ids

        if query["filter"]:
            body["query"] = query

        if cursor:
            body["cursor"] = cursor

        data = await self._request("POST", "/v2/subscriptions/search", json=body)
        subscriptions = [Subscription.from_api(s) for s in data.get("subscriptions", [])]
        return subscriptions, data.get("cursor")

    async def cancel_subscription(self, subscription_id: str) -> tuple[Subscription, List[str]]:
        """
        Cancel a subscription.

        Returns:
            Tuple of (subscription, affected_actions)
        """
        data = await self._request("POST", f"/v2/subscriptions/{subscription_id}/cancel")
        subscription = Subscription.from_api(data["subscription"])
        actions = data.get("actions", [])
        return subscription, actions

    async def pause_subscription(
        self,
        subscription_id: str,
        pause_effective_date: Optional[str] = None,
        pause_length_in_billing_cycles: Optional[int] = None,
    ) -> Subscription:
        """Pause a subscription."""
        body: Dict[str, Any] = {}

        if pause_effective_date:
            body["pause_effective_date"] = pause_effective_date
        if pause_length_in_billing_cycles:
            body["pause_cycle_duration"] = pause_length_in_billing_cycles

        data = await self._request("POST", f"/v2/subscriptions/{subscription_id}/pause", json=body)
        return Subscription.from_api(data["subscription"])

    async def resume_subscription(
        self,
        subscription_id: str,
        resume_effective_date: Optional[str] = None,
    ) -> Subscription:
        """Resume a paused subscription."""
        body: Dict[str, Any] = {}

        if resume_effective_date:
            body["resume_effective_date"] = resume_effective_date

        data = await self._request("POST", f"/v2/subscriptions/{subscription_id}/resume", json=body)
        return Subscription.from_api(data["subscription"])

    # -------------------------------------------------------------------------
    # Invoices
    # -------------------------------------------------------------------------

    async def create_invoice(
        self,
        order_id: str,
        location_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        primary_recipient_customer_id: Optional[str] = None,
        payment_requests: Optional[List[Dict[str, Any]]] = None,
        invoice_number: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        scheduled_at: Optional[str] = None,
    ) -> Invoice:
        """Create an invoice."""
        invoice_data: Dict[str, Any] = {
            "order_id": order_id,
            "location_id": location_id or self.credentials.location_id,
        }

        if primary_recipient_customer_id:
            invoice_data["primary_recipient"] = {"customer_id": primary_recipient_customer_id}
        if payment_requests:
            invoice_data["payment_requests"] = payment_requests
        if invoice_number:
            invoice_data["invoice_number"] = invoice_number
        if title:
            invoice_data["title"] = title
        if description:
            invoice_data["description"] = description
        if scheduled_at:
            invoice_data["scheduled_at"] = scheduled_at

        body: Dict[str, Any] = {
            "invoice": invoice_data,
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
        }

        data = await self._request("POST", "/v2/invoices", json=body)
        return Invoice.from_api(data["invoice"])

    async def get_invoice(self, invoice_id: str) -> Invoice:
        """Get invoice details."""
        data = await self._request("GET", f"/v2/invoices/{invoice_id}")
        return Invoice.from_api(data["invoice"])

    async def publish_invoice(
        self,
        invoice_id: str,
        version: int,
        idempotency_key: Optional[str] = None,
    ) -> Invoice:
        """Publish a draft invoice."""
        body: Dict[str, Any] = {
            "version": version,
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
        }

        data = await self._request("POST", f"/v2/invoices/{invoice_id}/publish", json=body)
        return Invoice.from_api(data["invoice"])

    async def cancel_invoice(
        self,
        invoice_id: str,
        version: int,
    ) -> Invoice:
        """Cancel an invoice."""
        body: Dict[str, Any] = {"version": version}

        data = await self._request("POST", f"/v2/invoices/{invoice_id}/cancel", json=body)
        return Invoice.from_api(data["invoice"])

    # -------------------------------------------------------------------------
    # Catalog
    # -------------------------------------------------------------------------

    async def batch_upsert_catalog_objects(
        self,
        objects: List[Dict[str, Any]],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch create or update catalog objects."""
        body: Dict[str, Any] = {
            "idempotency_key": idempotency_key or _generate_idempotency_key(),
            "batches": [{"objects": objects}],
        }

        return await self._request("POST", "/v2/catalog/batch-upsert", json=body)

    async def get_catalog_object(
        self,
        object_id: str,
        include_related_objects: bool = False,
    ) -> CatalogItem:
        """Get a catalog object."""
        params: Dict[str, Any] = {}
        if include_related_objects:
            params["include_related_objects"] = "true"

        data = await self._request("GET", f"/v2/catalog/object/{object_id}", params=params)
        return CatalogItem.from_api(data["object"])

    async def search_catalog_objects(
        self,
        object_types: Optional[List[str]] = None,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[List[CatalogItem], Optional[str]]:
        """
        Search catalog objects.

        Returns:
            Tuple of (items, next_cursor)
        """
        body: Dict[str, Any] = {"limit": min(limit, 100)}

        if object_types:
            body["object_types"] = object_types
        if query:
            body["query"] = query
        if cursor:
            body["cursor"] = cursor

        data = await self._request("POST", "/v2/catalog/search", json=body)
        items = [CatalogItem.from_api(obj) for obj in data.get("objects", [])]
        return items, data.get("cursor")

    async def delete_catalog_object(self, object_id: str) -> List[str]:
        """
        Delete a catalog object.

        Returns:
            List of deleted object IDs
        """
        data = await self._request("DELETE", f"/v2/catalog/object/{object_id}")
        return data.get("deleted_object_ids", [])

    # -------------------------------------------------------------------------
    # Webhooks
    # -------------------------------------------------------------------------

    def verify_webhook_signature(
        self,
        request_body: str,
        signature: str,
        signature_key: Optional[str] = None,
        notification_url: str = "",
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            request_body: Raw request body
            signature: X-Square-Signature header
            signature_key: Webhook signature key (from credentials if not provided)
            notification_url: Your webhook URL

        Returns:
            True if signature is valid
        """
        key = signature_key or self.credentials.webhook_signature_key
        if not key:
            logger.warning("Webhook signature key not configured")
            return True

        # Square uses HMAC-SHA1
        expected_signature = base64.b64encode(
            hmac.new(
                key.encode(),
                (notification_url + request_body).encode(),
                hashlib.sha1,
            ).digest()
        ).decode()

        return hmac.compare_digest(expected_signature, signature)


# =============================================================================
# Mock Data Generators
# =============================================================================


def get_mock_customer() -> Customer:
    """Get a mock customer for testing."""
    return Customer(
        id="CUSTOMER_ID_12345",
        given_name="John",
        family_name="Doe",
        email_address="john.doe@example.com",
        phone_number="+15551234567",
        created_at=datetime.now(timezone.utc),
    )


def get_mock_payment() -> Payment:
    """Get a mock payment for testing."""
    return Payment(
        id="PAYMENT_ID_12345",
        status=PaymentStatus.COMPLETED,
        amount_money=Money.usd(99.99),
        total_money=Money.usd(99.99),
        source_type="CARD",
        created_at=datetime.now(timezone.utc),
    )


def get_mock_subscription() -> Subscription:
    """Get a mock subscription for testing."""
    return Subscription(
        id="SUBSCRIPTION_ID_12345",
        status=SubscriptionStatus.ACTIVE,
        plan_id="PLAN_ID_12345",
        customer_id="CUSTOMER_ID_12345",
        start_date="2025-01-01",
        created_at=datetime.now(timezone.utc),
    )

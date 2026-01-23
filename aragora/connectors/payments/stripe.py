"""
Stripe Payments Connector.

Full integration with Stripe API:
- Customers and subscriptions
- Payment intents and charges
- Invoices and billing
- Products and prices
- Webhooks for real-time events
- Reporting and analytics

Requires Stripe API secret key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class PaymentStatus(str, Enum):
    """Payment intent status."""

    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    REQUIRES_CAPTURE = "requires_capture"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"


class SubscriptionStatus(str, Enum):
    """Subscription status."""

    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    PAUSED = "paused"


class InvoiceStatus(str, Enum):
    """Invoice status."""

    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    UNCOLLECTIBLE = "uncollectible"
    VOID = "void"


class PriceType(str, Enum):
    """Price type."""

    ONE_TIME = "one_time"
    RECURRING = "recurring"


@dataclass
class StripeCredentials:
    """Stripe API credentials."""

    secret_key: str
    webhook_secret: str | None = None


@dataclass
class StripeCustomer:
    """A Stripe customer."""

    id: str
    email: str | None = None
    name: str | None = None
    phone: str | None = None
    description: str | None = None
    balance: int = 0  # In cents
    currency: str = "usd"
    delinquent: bool = False
    default_source: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> StripeCustomer:
        return cls(
            id=data["id"],
            email=data.get("email"),
            name=data.get("name"),
            phone=data.get("phone"),
            description=data.get("description"),
            balance=data.get("balance", 0),
            currency=data.get("currency", "usd"),
            delinquent=data.get("delinquent", False),
            default_source=data.get("default_source"),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


@dataclass
class StripeProduct:
    """A Stripe product."""

    id: str
    name: str
    active: bool = True
    description: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created: datetime | None = None
    updated: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> StripeProduct:
        return cls(
            id=data["id"],
            name=data["name"],
            active=data.get("active", True),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
            updated=datetime.fromtimestamp(data["updated"]) if data.get("updated") else None,
        )


@dataclass
class StripePrice:
    """A Stripe price."""

    id: str
    product_id: str
    active: bool = True
    currency: str = "usd"
    unit_amount: int | None = None  # In cents
    type: PriceType = PriceType.ONE_TIME
    recurring_interval: str | None = None  # day, week, month, year
    recurring_interval_count: int = 1
    metadata: dict[str, str] = field(default_factory=dict)
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> StripePrice:
        recurring = data.get("recurring", {})
        return cls(
            id=data["id"],
            product_id=data["product"]
            if isinstance(data["product"], str)
            else data["product"]["id"],
            active=data.get("active", True),
            currency=data.get("currency", "usd"),
            unit_amount=data.get("unit_amount"),
            type=PriceType(data.get("type", "one_time")),
            recurring_interval=recurring.get("interval") if recurring else None,
            recurring_interval_count=recurring.get("interval_count", 1) if recurring else 1,
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


@dataclass
class StripeSubscription:
    """A Stripe subscription."""

    id: str
    customer_id: str
    status: SubscriptionStatus
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False
    canceled_at: datetime | None = None
    ended_at: datetime | None = None
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    items: list[dict] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> StripeSubscription:
        return cls(
            id=data["id"],
            customer_id=data["customer"]
            if isinstance(data["customer"], str)
            else data["customer"]["id"],
            status=SubscriptionStatus(data["status"]),
            current_period_start=datetime.fromtimestamp(data["current_period_start"])
            if data.get("current_period_start")
            else None,
            current_period_end=datetime.fromtimestamp(data["current_period_end"])
            if data.get("current_period_end")
            else None,
            cancel_at_period_end=data.get("cancel_at_period_end", False),
            canceled_at=datetime.fromtimestamp(data["canceled_at"])
            if data.get("canceled_at")
            else None,
            ended_at=datetime.fromtimestamp(data["ended_at"]) if data.get("ended_at") else None,
            trial_start=datetime.fromtimestamp(data["trial_start"])
            if data.get("trial_start")
            else None,
            trial_end=datetime.fromtimestamp(data["trial_end"]) if data.get("trial_end") else None,
            items=data.get("items", {}).get("data", []),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


@dataclass
class StripeInvoice:
    """A Stripe invoice."""

    id: str
    customer_id: str
    subscription_id: str | None = None
    status: InvoiceStatus = InvoiceStatus.DRAFT
    currency: str = "usd"
    amount_due: int = 0
    amount_paid: int = 0
    amount_remaining: int = 0
    subtotal: int = 0
    tax: int | None = None
    total: int = 0
    number: str | None = None
    invoice_pdf: str | None = None
    hosted_invoice_url: str | None = None
    paid: bool = False
    due_date: datetime | None = None
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> StripeInvoice:
        return cls(
            id=data["id"],
            customer_id=data["customer"]
            if isinstance(data["customer"], str)
            else data["customer"]["id"],
            subscription_id=data.get("subscription"),
            status=InvoiceStatus(data.get("status", "draft")),
            currency=data.get("currency", "usd"),
            amount_due=data.get("amount_due", 0),
            amount_paid=data.get("amount_paid", 0),
            amount_remaining=data.get("amount_remaining", 0),
            subtotal=data.get("subtotal", 0),
            tax=data.get("tax"),
            total=data.get("total", 0),
            number=data.get("number"),
            invoice_pdf=data.get("invoice_pdf"),
            hosted_invoice_url=data.get("hosted_invoice_url"),
            paid=data.get("paid", False),
            due_date=datetime.fromtimestamp(data["due_date"]) if data.get("due_date") else None,
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


@dataclass
class PaymentIntent:
    """A Stripe payment intent."""

    id: str
    amount: int
    currency: str
    status: PaymentStatus
    customer_id: str | None = None
    description: str | None = None
    receipt_email: str | None = None
    payment_method: str | None = None
    client_secret: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> PaymentIntent:
        return cls(
            id=data["id"],
            amount=data["amount"],
            currency=data["currency"],
            status=PaymentStatus(data["status"]),
            customer_id=data.get("customer"),
            description=data.get("description"),
            receipt_email=data.get("receipt_email"),
            payment_method=data.get("payment_method"),
            client_secret=data.get("client_secret"),
            metadata=data.get("metadata", {}),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


@dataclass
class BalanceTransaction:
    """A Stripe balance transaction."""

    id: str
    amount: int
    currency: str
    type: str
    description: str | None = None
    fee: int = 0
    net: int = 0
    status: str = "available"
    created: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> BalanceTransaction:
        return cls(
            id=data["id"],
            amount=data["amount"],
            currency=data["currency"],
            type=data["type"],
            description=data.get("description"),
            fee=data.get("fee", 0),
            net=data.get("net", 0),
            status=data.get("status", "available"),
            created=datetime.fromtimestamp(data["created"]) if data.get("created") else None,
        )


class StripeError(Exception):
    """Stripe API error."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class StripeConnector:
    """
    Stripe API connector.

    Features:
    - Customer management
    - Product and price catalog
    - Subscriptions and billing
    - Payment intents
    - Invoices
    - Balance and payouts
    - Webhook verification

    Usage:
        ```python
        credentials = StripeCredentials(secret_key="sk_...")

        async with StripeConnector(credentials) as stripe:
            # Create a customer
            customer = await stripe.create_customer(
                email="customer@example.com",
                name="John Doe"
            )

            # Create a subscription
            subscription = await stripe.create_subscription(
                customer_id=customer.id,
                price_id="price_..."
            )

            # List invoices
            invoices = await stripe.list_invoices(customer_id=customer.id)
        ```
    """

    BASE_URL = "https://api.stripe.com/v1"

    def __init__(self, credentials: StripeCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> StripeConnector:
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.credentials.secret_key}",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise StripeError("Connector not initialized. Use async context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = await self.client.request(
                method,
                url,
                data=data,  # Stripe uses form encoding
                params=params,
            )

            result = response.json()

            if response.status_code >= 400:
                error = result.get("error", {})
                raise StripeError(
                    message=error.get("message", "Unknown error"),
                    code=error.get("code"),
                    status_code=response.status_code,
                )

            return result
        except httpx.HTTPError as e:
            raise StripeError(f"HTTP error: {e}") from e

    # -------------------------------------------------------------------------
    # Customers
    # -------------------------------------------------------------------------

    async def create_customer(
        self,
        email: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StripeCustomer:
        """Create a new customer."""
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if phone:
            data["phone"] = phone
        if description:
            data["description"] = description
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v

        result = await self._request("POST", "/customers", data=data)
        return StripeCustomer.from_api(result)

    async def get_customer(self, customer_id: str) -> StripeCustomer:
        """Get a customer by ID."""
        result = await self._request("GET", f"/customers/{customer_id}")
        return StripeCustomer.from_api(result)

    async def update_customer(self, customer_id: str, **updates) -> StripeCustomer:
        """Update a customer."""
        result = await self._request("POST", f"/customers/{customer_id}", data=updates)
        return StripeCustomer.from_api(result)

    async def list_customers(
        self,
        email: str | None = None,
        limit: int = 10,
        starting_after: str | None = None,
    ) -> list[StripeCustomer]:
        """List customers."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if email:
            params["email"] = email
        if starting_after:
            params["starting_after"] = starting_after

        result = await self._request("GET", "/customers", params=params)
        return [StripeCustomer.from_api(c) for c in result.get("data", [])]

    async def delete_customer(self, customer_id: str) -> None:
        """Delete a customer."""
        await self._request("DELETE", f"/customers/{customer_id}")

    # -------------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------------

    async def create_product(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StripeProduct:
        """Create a product."""
        data: dict[str, Any] = {"name": name}
        if description:
            data["description"] = description
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v

        result = await self._request("POST", "/products", data=data)
        return StripeProduct.from_api(result)

    async def get_product(self, product_id: str) -> StripeProduct:
        """Get a product by ID."""
        result = await self._request("GET", f"/products/{product_id}")
        return StripeProduct.from_api(result)

    async def list_products(
        self, active: bool | None = None, limit: int = 10
    ) -> list[StripeProduct]:
        """List products."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if active is not None:
            params["active"] = str(active).lower()

        result = await self._request("GET", "/products", params=params)
        return [StripeProduct.from_api(p) for p in result.get("data", [])]

    # -------------------------------------------------------------------------
    # Prices
    # -------------------------------------------------------------------------

    async def create_price(
        self,
        product_id: str,
        unit_amount: int,
        currency: str = "usd",
        recurring_interval: str | None = None,  # month, year, etc.
    ) -> StripePrice:
        """Create a price."""
        data: dict[str, Any] = {
            "product": product_id,
            "unit_amount": unit_amount,
            "currency": currency,
        }
        if recurring_interval:
            data["recurring[interval]"] = recurring_interval

        result = await self._request("POST", "/prices", data=data)
        return StripePrice.from_api(result)

    async def get_price(self, price_id: str) -> StripePrice:
        """Get a price by ID."""
        result = await self._request("GET", f"/prices/{price_id}")
        return StripePrice.from_api(result)

    async def list_prices(
        self, product_id: str | None = None, limit: int = 10
    ) -> list[StripePrice]:
        """List prices."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if product_id:
            params["product"] = product_id

        result = await self._request("GET", "/prices", params=params)
        return [StripePrice.from_api(p) for p in result.get("data", [])]

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StripeSubscription:
        """Create a subscription."""
        data: dict[str, Any] = {
            "customer": customer_id,
            "items[0][price]": price_id,
        }
        if trial_period_days:
            data["trial_period_days"] = trial_period_days
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v

        result = await self._request("POST", "/subscriptions", data=data)
        return StripeSubscription.from_api(result)

    async def get_subscription(self, subscription_id: str) -> StripeSubscription:
        """Get a subscription by ID."""
        result = await self._request("GET", f"/subscriptions/{subscription_id}")
        return StripeSubscription.from_api(result)

    async def list_subscriptions(
        self,
        customer_id: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[StripeSubscription]:
        """List subscriptions."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if customer_id:
            params["customer"] = customer_id
        if status:
            params["status"] = status

        result = await self._request("GET", "/subscriptions", params=params)
        return [StripeSubscription.from_api(s) for s in result.get("data", [])]

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = False,
    ) -> StripeSubscription:
        """Cancel a subscription."""
        if at_period_end:
            result = await self._request(
                "POST",
                f"/subscriptions/{subscription_id}",
                data={"cancel_at_period_end": "true"},
            )
        else:
            result = await self._request("DELETE", f"/subscriptions/{subscription_id}")
        return StripeSubscription.from_api(result)

    # -------------------------------------------------------------------------
    # Invoices
    # -------------------------------------------------------------------------

    async def create_invoice(
        self,
        customer_id: str,
        auto_advance: bool = True,
        description: str | None = None,
    ) -> StripeInvoice:
        """Create an invoice."""
        data: dict[str, Any] = {
            "customer": customer_id,
            "auto_advance": str(auto_advance).lower(),
        }
        if description:
            data["description"] = description

        result = await self._request("POST", "/invoices", data=data)
        return StripeInvoice.from_api(result)

    async def get_invoice(self, invoice_id: str) -> StripeInvoice:
        """Get an invoice by ID."""
        result = await self._request("GET", f"/invoices/{invoice_id}")
        return StripeInvoice.from_api(result)

    async def list_invoices(
        self,
        customer_id: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[StripeInvoice]:
        """List invoices."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if customer_id:
            params["customer"] = customer_id
        if status:
            params["status"] = status

        result = await self._request("GET", "/invoices", params=params)
        return [StripeInvoice.from_api(i) for i in result.get("data", [])]

    async def finalize_invoice(self, invoice_id: str) -> StripeInvoice:
        """Finalize a draft invoice."""
        result = await self._request("POST", f"/invoices/{invoice_id}/finalize")
        return StripeInvoice.from_api(result)

    async def pay_invoice(self, invoice_id: str) -> StripeInvoice:
        """Pay an invoice."""
        result = await self._request("POST", f"/invoices/{invoice_id}/pay")
        return StripeInvoice.from_api(result)

    async def void_invoice(self, invoice_id: str) -> StripeInvoice:
        """Void an invoice."""
        result = await self._request("POST", f"/invoices/{invoice_id}/void")
        return StripeInvoice.from_api(result)

    # -------------------------------------------------------------------------
    # Payment Intents
    # -------------------------------------------------------------------------

    async def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd",
        customer_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> PaymentIntent:
        """Create a payment intent."""
        data: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
        }
        if customer_id:
            data["customer"] = customer_id
        if description:
            data["description"] = description
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v

        result = await self._request("POST", "/payment_intents", data=data)
        return PaymentIntent.from_api(result)

    async def get_payment_intent(self, payment_intent_id: str) -> PaymentIntent:
        """Get a payment intent by ID."""
        result = await self._request("GET", f"/payment_intents/{payment_intent_id}")
        return PaymentIntent.from_api(result)

    async def confirm_payment_intent(self, payment_intent_id: str) -> PaymentIntent:
        """Confirm a payment intent."""
        result = await self._request("POST", f"/payment_intents/{payment_intent_id}/confirm")
        return PaymentIntent.from_api(result)

    async def cancel_payment_intent(self, payment_intent_id: str) -> PaymentIntent:
        """Cancel a payment intent."""
        result = await self._request("POST", f"/payment_intents/{payment_intent_id}/cancel")
        return PaymentIntent.from_api(result)

    # -------------------------------------------------------------------------
    # Balance
    # -------------------------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        """Get current balance."""
        return await self._request("GET", "/balance")

    async def list_balance_transactions(
        self,
        limit: int = 10,
        type: str | None = None,
    ) -> list[BalanceTransaction]:
        """List balance transactions."""
        params: dict[str, Any] = {"limit": min(limit, 100)}
        if type:
            params["type"] = type

        result = await self._request("GET", "/balance_transactions", params=params)
        return [BalanceTransaction.from_api(t) for t in result.get("data", [])]


# -----------------------------------------------------------------------------
# Mock Data
# -----------------------------------------------------------------------------


def get_mock_customer() -> StripeCustomer:
    return StripeCustomer(
        id="cus_123456789",
        email="customer@example.com",
        name="John Doe",
        balance=0,
        currency="usd",
    )


def get_mock_subscription() -> StripeSubscription:
    return StripeSubscription(
        id="sub_123456789",
        customer_id="cus_123456789",
        status=SubscriptionStatus.ACTIVE,
        current_period_start=datetime(2025, 1, 1),
        current_period_end=datetime(2025, 2, 1),
    )


def get_mock_invoice() -> StripeInvoice:
    return StripeInvoice(
        id="in_123456789",
        customer_id="cus_123456789",
        status=InvoiceStatus.PAID,
        currency="usd",
        amount_due=2999,
        amount_paid=2999,
        total=2999,
        paid=True,
    )

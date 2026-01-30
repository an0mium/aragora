# Stripe Connector Guide

This guide covers the Stripe payments connector for integrating billing and subscriptions.

## Overview

The `StripeConnector` provides full integration with Stripe API:

- Customer management
- Product and price catalog
- Subscriptions and billing
- Payment intents
- Invoices
- Balance and payouts
- Webhook verification

**Location:** `aragora/connectors/payments/stripe.py`

---

## Quick Start

```python
from aragora.connectors.payments.stripe import (
    StripeConnector,
    StripeCredentials,
)

credentials = StripeCredentials(
    secret_key="sk_test_...",
    webhook_secret="whsec_...",  # Optional, for webhooks
)

async with StripeConnector(credentials) as stripe:
    # Create a customer
    customer = await stripe.create_customer(
        email="customer@example.com",
        name="John Doe",
    )
    print(f"Created customer: {customer.id}")
```

---

## Configuration

### Credentials

```python
from dataclasses import dataclass

@dataclass
class StripeCredentials:
    secret_key: str              # Required: sk_test_... or sk_live_...
    webhook_secret: str | None   # Optional: whsec_... for webhook verification
```

### Environment Variables

```bash
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### Getting Your Keys

1. Go to [Stripe Dashboard](https://dashboard.stripe.com/apikeys)
2. Copy your **Secret key** (starts with `sk_test_` or `sk_live_`)
3. For webhooks, create an endpoint and copy the **Signing secret**

---

## Data Models

### Customer

```python
@dataclass
class StripeCustomer:
    id: str
    email: str | None
    name: str | None
    phone: str | None
    description: str | None
    balance: int  # In cents
    currency: str
    delinquent: bool
    default_source: str | None
    metadata: dict[str, str]
    created: datetime | None
```

### Product

```python
@dataclass
class StripeProduct:
    id: str
    name: str
    active: bool
    description: str | None
    metadata: dict[str, str]
    created: datetime | None
    updated: datetime | None
```

### Price

```python
@dataclass
class StripePrice:
    id: str
    product_id: str
    active: bool
    currency: str
    unit_amount: int | None  # In cents
    type: PriceType          # ONE_TIME or RECURRING
    recurring_interval: str | None  # day, week, month, year
    recurring_interval_count: int
    metadata: dict[str, str]
    created: datetime | None
```

### Subscription

```python
@dataclass
class StripeSubscription:
    id: str
    customer_id: str
    status: SubscriptionStatus
    current_period_start: datetime | None
    current_period_end: datetime | None
    cancel_at_period_end: bool
    canceled_at: datetime | None
    ended_at: datetime | None
    trial_start: datetime | None
    trial_end: datetime | None
    items: list[dict]
    metadata: dict[str, str]
    created: datetime | None
```

### Status Enums

```python
class SubscriptionStatus(str, Enum):
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    PAUSED = "paused"

class PaymentStatus(str, Enum):
    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    REQUIRES_CAPTURE = "requires_capture"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"

class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    UNCOLLECTIBLE = "uncollectible"
    VOID = "void"
```

---

## Customer Operations

### Create Customer

```python
customer = await stripe.create_customer(
    email="customer@example.com",
    name="John Doe",
    phone="+1234567890",
    description="Premium customer",
    metadata={"plan": "enterprise"},
)
```

### Get Customer

```python
customer = await stripe.get_customer("cus_xxx")
```

### Update Customer

```python
customer = await stripe.update_customer(
    customer_id="cus_xxx",
    name="Jane Doe",
    email="jane@example.com",
)
```

### List Customers

```python
# List all customers
customers = await stripe.list_customers(limit=50)

# Filter by email
customers = await stripe.list_customers(email="john@example.com")

# Paginate
customers = await stripe.list_customers(
    limit=50,
    starting_after="cus_xxx",
)
```

### Delete Customer

```python
await stripe.delete_customer("cus_xxx")
```

---

## Product Catalog

### Create Product

```python
product = await stripe.create_product(
    name="Pro Plan",
    description="Full access to all features",
    metadata={"tier": "pro"},
)
```

### Create Price

```python
# One-time price
price = await stripe.create_price(
    product_id=product.id,
    unit_amount=2999,  # $29.99 in cents
    currency="usd",
)

# Recurring price (monthly)
monthly_price = await stripe.create_price(
    product_id=product.id,
    unit_amount=2999,
    currency="usd",
    recurring_interval="month",
)

# Recurring price (yearly)
yearly_price = await stripe.create_price(
    product_id=product.id,
    unit_amount=29900,  # $299.00/year
    currency="usd",
    recurring_interval="year",
)
```

### List Products and Prices

```python
# List active products
products = await stripe.list_products(active=True)

# List prices for a product
prices = await stripe.list_prices(product_id="prod_xxx")
```

---

## Subscriptions

### Create Subscription

```python
subscription = await stripe.create_subscription(
    customer_id="cus_xxx",
    price_id="price_xxx",
    trial_period_days=14,  # Optional trial
    metadata={"source": "website"},
)
```

### Get Subscription

```python
subscription = await stripe.get_subscription("sub_xxx")
```

### List Subscriptions

```python
# All subscriptions for a customer
subs = await stripe.list_subscriptions(customer_id="cus_xxx")

# Filter by status
active_subs = await stripe.list_subscriptions(
    customer_id="cus_xxx",
    status="active",
)
```

### Cancel Subscription

```python
# Cancel immediately
sub = await stripe.cancel_subscription("sub_xxx")

# Cancel at period end
sub = await stripe.cancel_subscription(
    "sub_xxx",
    at_period_end=True,
)
```

---

## Invoices

### Create Invoice

```python
invoice = await stripe.create_invoice(
    customer_id="cus_xxx",
    auto_advance=True,  # Auto-finalize
    description="January 2026 Invoice",
)
```

### Invoice Lifecycle

```python
# Get invoice
invoice = await stripe.get_invoice("in_xxx")

# Finalize draft invoice
invoice = await stripe.finalize_invoice("in_xxx")

# Pay invoice
invoice = await stripe.pay_invoice("in_xxx")

# Void invoice
invoice = await stripe.void_invoice("in_xxx")
```

### List Invoices

```python
# All invoices for customer
invoices = await stripe.list_invoices(customer_id="cus_xxx")

# Filter by status
paid_invoices = await stripe.list_invoices(
    customer_id="cus_xxx",
    status="paid",
)
```

---

## Payment Intents

### Create Payment Intent

```python
intent = await stripe.create_payment_intent(
    amount=5000,  # $50.00 in cents
    currency="usd",
    customer_id="cus_xxx",
    description="Order #12345",
    metadata={"order_id": "12345"},
)

# Get client_secret for frontend
print(intent.client_secret)
```

### Confirm/Cancel Payment

```python
# Confirm (if not auto-confirmed)
intent = await stripe.confirm_payment_intent("pi_xxx")

# Cancel
intent = await stripe.cancel_payment_intent("pi_xxx")
```

---

## Balance

```python
# Get current balance
balance = await stripe.get_balance()
print(f"Available: {balance['available']}")
print(f"Pending: {balance['pending']}")

# List balance transactions
transactions = await stripe.list_balance_transactions(
    limit=50,
    type="charge",  # Filter by type
)
```

---

## Webhooks

### Verify Webhook Signature

```python
from aiohttp import web

async def webhook_handler(request: web.Request) -> web.Response:
    payload = await request.read()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = await stripe.construct_webhook_event(payload, sig_header)
    except Exception as e:
        return web.Response(status=400, text=str(e))

    # Handle the event
    if event.type == "customer.subscription.created":
        subscription = event.data.object
        print(f"New subscription: {subscription.id}")

    elif event.type == "invoice.paid":
        invoice = event.data.object
        print(f"Invoice paid: {invoice.get('id')}")

    elif event.type == "invoice.payment_failed":
        invoice = event.data.object
        print(f"Payment failed for invoice: {invoice.get('id')}")

    return web.Response(status=200)
```

### Common Webhook Events

| Event | Description |
|-------|-------------|
| `customer.created` | New customer created |
| `customer.subscription.created` | New subscription started |
| `customer.subscription.updated` | Subscription changed |
| `customer.subscription.deleted` | Subscription canceled |
| `invoice.created` | New invoice created |
| `invoice.paid` | Invoice successfully paid |
| `invoice.payment_failed` | Invoice payment failed |
| `charge.succeeded` | Charge successful |
| `charge.failed` | Charge failed |
| `payment_intent.succeeded` | Payment intent successful |

---

## Error Handling

```python
from aragora.connectors.payments.stripe import StripeError

try:
    customer = await stripe.create_customer(email="invalid")
except StripeError as e:
    print(f"Stripe error: {e}")
    print(f"Code: {e.code}")
    print(f"Status: {e.status_code}")
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `card_declined` | Card was declined |
| `expired_card` | Card has expired |
| `incorrect_cvc` | CVC is incorrect |
| `processing_error` | Error processing card |
| `rate_limit` | Too many API requests |

---

## Testing

### Mock Data

```python
from aragora.connectors.payments.stripe import (
    get_mock_customer,
    get_mock_subscription,
    get_mock_invoice,
)

# Get mock objects for testing
customer = get_mock_customer()
subscription = get_mock_subscription()
invoice = get_mock_invoice()
```

### Test Mode

Use test keys (prefixed with `sk_test_`) during development:

```python
credentials = StripeCredentials(secret_key="sk_test_...")
```

Stripe provides [test card numbers](https://stripe.com/docs/testing):

| Number | Description |
|--------|-------------|
| `4242424242424242` | Succeeds |
| `4000000000000002` | Declines |
| `4000002500003155` | Requires 3D Secure |

---

## Best Practices

1. **Always use test keys in development** - Never use live keys locally
2. **Verify webhooks** - Always verify webhook signatures
3. **Handle idempotency** - Use idempotency keys for retries
4. **Store metadata** - Use metadata for tracking internal references
5. **Handle async payments** - Payment intents may require additional steps

---

## Related Documentation

- [Stripe API Reference](https://stripe.com/docs/api)
- [Stripe Webhooks Guide](https://stripe.com/docs/webhooks)
- [Connector Patterns Guide](./CONNECTOR_PATTERNS.md)

---

*Last updated: 2026-01-30*

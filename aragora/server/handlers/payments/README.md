# Payment Handlers

Payment processing APIs supporting Stripe and Authorize.net with PCI-compliant transaction handling, subscription management, and webhook processing.

## Modules

| Module | Purpose |
|--------|---------|
| `handler.py` | Core infrastructure, models, circuit breakers, rate limiters |
| `stripe.py` | Stripe payment processing and webhook handlers |
| `billing.py` | Customer and subscription management |
| `plans.py` | Route registration utilities |

## Endpoints

### Transactions
- `POST /api/v1/payments/charge` - Process payment
- `POST /api/v1/payments/authorize` - Authorize hold
- `POST /api/v1/payments/capture` - Capture authorization
- `POST /api/v1/payments/refund` - Process refund
- `POST /api/v1/payments/void` - Void authorization
- `GET /api/v1/payments/transactions/{id}` - Get transaction

### Customers
- `POST /api/v1/payments/customers` - Create customer
- `GET /api/v1/payments/customers/{id}` - Get customer
- `PUT /api/v1/payments/customers/{id}` - Update customer
- `DELETE /api/v1/payments/customers/{id}` - Delete customer

### Subscriptions
- `POST /api/v1/payments/subscriptions` - Create subscription
- `GET /api/v1/payments/subscriptions/{id}` - Get subscription
- `PUT /api/v1/payments/subscriptions/{id}` - Update subscription
- `POST /api/v1/payments/subscriptions/{id}/cancel` - Cancel subscription

### Webhooks
- `POST /api/v1/webhooks/stripe` - Stripe webhook endpoint
- `POST /api/v1/webhooks/authnet` - Authorize.net webhook

## RBAC Permissions

### Payment Operations
| Permission | Description |
|------------|-------------|
| `payments:read` | View transaction details |
| `payments:charge` | Process payments |
| `payments:authorize` | Create authorization holds |
| `payments:capture` | Capture authorizations |
| `payments:refund` | Process refunds |
| `payments:void` | Void authorizations |
| `payments:admin` | Full payment administration |

### Customer Management
| Permission | Description |
|------------|-------------|
| `customer:read` | View customer profiles |
| `customer:create` | Create customers |
| `customer:update` | Update customers |
| `customer:delete` | Delete customers |

### Subscription Management
| Permission | Description |
|------------|-------------|
| `subscription:read` | View subscriptions |
| `subscription:create` | Create subscriptions |
| `subscription:update` | Modify subscriptions |
| `subscription:cancel` | Cancel subscriptions |

### Webhooks
| Permission | Description |
|------------|-------------|
| `webhook:stripe` | Process Stripe webhooks |
| `webhook:authnet` | Process Authorize.net webhooks |

## Providers

| Provider | Status | Features |
|----------|--------|----------|
| Stripe | STABLE | Full API, webhooks, subscriptions |
| Authorize.net | STABLE | Transactions, customers, webhooks |

## Usage

```python
from aragora.server.handlers.payments import (
    handle_charge,
    handle_create_customer,
    handle_stripe_webhook,
    PaymentProvider,
)

# Process a charge
result = await handle_charge(request_handler, workspace_id)

# Handle Stripe webhook
result = await handle_stripe_webhook(request_handler)
```

## Features

- **Multi-Provider**: Stripe and Authorize.net support
- **PCI Compliance**: No card data stored server-side
- **Webhook Security**: Signature verification, deduplication
- **Circuit Breakers**: Per-provider fault isolation
- **Rate Limiting**: Read/write separation, webhook protection
- **Idempotency**: Duplicate request handling
- **Retry Logic**: Configurable retry with exponential backoff
- **Audit Logging**: Complete transaction audit trail

## Tests

166 tests covering transactions, webhooks, subscriptions, and error handling.

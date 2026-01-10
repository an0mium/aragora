# Billing System

Aragora's billing system provides user authentication, subscription management, usage tracking, and Stripe integration for monetization.

## Overview

The billing module consists of four components:

| Module | Purpose |
|--------|---------|
| `models.py` | Data models for User, Organization, Subscription |
| `jwt_auth.py` | JWT token authentication |
| `stripe_client.py` | Stripe API integration |
| `usage.py` | Usage tracking and cost calculation |

## Quick Start

### Enable Billing

Set the required environment variables:

```bash
# JWT Authentication
export ARAGORA_JWT_SECRET="your-secure-secret-key"
export ARAGORA_JWT_EXPIRY_HOURS=24

# Stripe Integration (optional for paid tiers)
export STRIPE_SECRET_KEY="sk_test_xxx"
export STRIPE_WEBHOOK_SECRET="whsec_xxx"
export STRIPE_PRICE_STARTER="price_xxx"
export STRIPE_PRICE_PROFESSIONAL="price_xxx"
export STRIPE_PRICE_ENTERPRISE="price_xxx"
```

## Subscription Tiers

| Tier | Debates/Month | Users | API Access | Price |
|------|---------------|-------|------------|-------|
| FREE | 10 | 1 | No | $0 |
| STARTER | 50 | 2 | No | $99/mo |
| PROFESSIONAL | 200 | 10 | Yes | $299/mo |
| ENTERPRISE | Unlimited | Unlimited | Yes | $999/mo |

### Tier Features

```python
from aragora.billing.models import SubscriptionTier, TIER_LIMITS

# Get limits for a tier
limits = TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
print(limits.debates_per_month)  # 200
print(limits.api_access)         # True
print(limits.all_agents)         # True
```

## Authentication

### JWT Tokens

Create and validate JWT tokens for user sessions:

```python
from aragora.billing.jwt_auth import (
    create_access_token,
    create_refresh_token,
    validate_access_token,
    create_token_pair,
)

# Create access token (24-hour default)
token = create_access_token(
    user_id="user_123",
    email="user@example.com",
    org_id="org_456",
    role="admin",
)

# Create token pair (access + refresh)
pair = create_token_pair(
    user_id="user_123",
    email="user@example.com",
    org_id="org_456",
)
print(pair.access_token)   # JWT for API calls
print(pair.refresh_token)  # JWT for token refresh
print(pair.expires_in)     # Seconds until expiry

# Validate token
payload = validate_access_token(token)
if payload:
    print(payload.user_id)  # "user_123"
    print(payload.email)    # "user@example.com"
    print(payload.is_expired)  # False
```

### API Key Authentication

Users can generate API keys for programmatic access:

```python
from aragora.billing.models import User

user = User(email="dev@example.com")
api_key = user.generate_api_key()  # Returns "ara_xxxxx..."

# Validate API key format
# Keys must start with "ara_" and be at least 15 characters
if api_key.startswith("ara_") and len(api_key) >= 15:
    # Valid format (production should validate against database)
    pass
```

### Request Authentication

Extract authentication from HTTP requests:

```python
from aragora.billing.jwt_auth import extract_user_from_request

# In a request handler
context = extract_user_from_request(handler)

if context.authenticated:
    print(f"User: {context.user_id}")
    print(f"Role: {context.role}")
    print(f"Token type: {context.token_type}")  # "access" or "api_key"

    if context.is_admin:
        # Allow admin operations
        pass
```

## User Management

### Creating Users

```python
from aragora.billing.models import User

# Create user
user = User(email="new@example.com", name="New User")
user.set_password("secure_password")

# User ID is auto-generated
print(user.id)  # UUID string

# Verify password
if user.verify_password("secure_password"):
    print("Password correct")
```

### Serialization

```python
# Safe serialization (excludes sensitive data)
data = user.to_dict()
# {"id": "...", "email": "...", "has_api_key": True, ...}

# Include sensitive data (for admin views)
data = user.to_dict(include_sensitive=True)
# Includes api_key field

# Restore from dict
restored = User.from_dict(data)
```

## Organization Management

### Creating Organizations

```python
from aragora.billing.models import Organization, SubscriptionTier, generate_slug

# Create organization
org = Organization(
    name="Acme Corp",
    slug=generate_slug("Acme Corp"),  # "acme-corp"
    tier=SubscriptionTier.PROFESSIONAL,
    owner_id="user_123",
)

# Check limits
print(org.limits.debates_per_month)  # 200
print(org.debates_remaining)          # 200 (at start of month)
print(org.is_at_limit)               # False
```

### Usage Tracking

```python
# Increment debate count
if org.increment_debates():
    print("Debate started")
else:
    print("At limit - upgrade required")

# Reset monthly usage (call at billing cycle)
org.reset_monthly_usage()
```

## Usage Tracking

Track token usage and costs per user/organization:

```python
from aragora.billing.usage import (
    UsageTracker,
    UsageEvent,
    UsageEventType,
    calculate_token_cost,
)

# Initialize tracker
tracker = UsageTracker("./data/usage.db")

# Record a debate event
tracker.record_event(
    org_id="org_123",
    user_id="user_456",
    event_type=UsageEventType.DEBATE,
    provider="anthropic",
    model="claude-sonnet-4",
    tokens_in=1500,
    tokens_out=800,
)

# Calculate token cost
cost = calculate_token_cost(
    provider="anthropic",
    model="claude-sonnet-4",
    tokens_in=1500,
    tokens_out=800,
)
print(f"Cost: ${cost}")  # Based on provider pricing

# Get usage summary
summary = tracker.get_usage_summary(
    org_id="org_123",
    start_date=datetime.now() - timedelta(days=30),
)
print(summary["total_debates"])
print(summary["total_tokens"])
print(summary["total_cost"])
```

### Provider Pricing

Pricing per 1M tokens (as of Jan 2025):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Anthropic | Claude Opus 4 | $15.00 | $75.00 |
| Anthropic | Claude Sonnet 4 | $3.00 | $15.00 |
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o Mini | $0.15 | $0.60 |
| Google | Gemini Pro | $1.25 | $5.00 |
| DeepSeek | DeepSeek V3 | $0.14 | $0.28 |
| OpenRouter | Default | $2.00 | $8.00 |

## Stripe Integration

### Setup

1. Create products and prices in Stripe Dashboard
2. Set environment variables with price IDs
3. Configure webhook endpoint

### Creating Checkout Sessions

```python
from aragora.billing.stripe_client import (
    create_checkout_session,
    StripeConfigError,
)
from aragora.billing.models import SubscriptionTier

try:
    session = create_checkout_session(
        customer_email="user@example.com",
        tier=SubscriptionTier.PROFESSIONAL,
        success_url="https://app.example.com/success",
        cancel_url="https://app.example.com/cancel",
        metadata={"org_id": "org_123"},
    )
    # Redirect user to session.url
except StripeConfigError as e:
    print(f"Stripe not configured: {e}")
```

### Handling Webhooks

```python
from aragora.billing.stripe_client import (
    verify_webhook_signature,
    WebhookEvent,
)

def handle_stripe_webhook(request):
    payload = request.body
    signature = request.headers.get("Stripe-Signature")

    # Verify signature
    event = verify_webhook_signature(payload, signature)

    if event.type == "checkout.session.completed":
        # Upgrade organization tier
        org_id = event.data.metadata.get("org_id")
        # Update organization...

    elif event.type == "customer.subscription.deleted":
        # Downgrade to free tier
        pass
```

### Managing Subscriptions

```python
from aragora.billing.stripe_client import (
    get_subscription,
    cancel_subscription,
    update_subscription,
)

# Get subscription details
sub = get_subscription(subscription_id)
print(sub.status)  # "active", "canceled", etc.
print(sub.current_period_end)

# Cancel at period end
cancel_subscription(subscription_id, at_period_end=True)

# Upgrade/downgrade
update_subscription(
    subscription_id,
    new_price_id="price_enterprise_xxx",
)
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ARAGORA_JWT_SECRET` | Secret key for JWT signing | Yes (auto-generated if missing) |
| `ARAGORA_JWT_EXPIRY_HOURS` | Access token expiry (default: 24) | No |
| `ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS` | Refresh token expiry (default: 30) | No |
| `STRIPE_SECRET_KEY` | Stripe API secret key | For paid tiers |
| `STRIPE_WEBHOOK_SECRET` | Webhook signing secret | For webhooks |
| `STRIPE_PRICE_STARTER` | Stripe price ID for Starter | For paid tiers |
| `STRIPE_PRICE_PROFESSIONAL` | Stripe price ID for Pro | For paid tiers |
| `STRIPE_PRICE_ENTERPRISE` | Stripe price ID for Enterprise | For paid tiers |

## Security Considerations

1. **JWT Secret**: Always set `ARAGORA_JWT_SECRET` in production. Auto-generated secrets are invalidated on restart.

2. **Password Hashing**: Passwords are hashed using SHA-256 with unique salts. Never store plain text passwords.

3. **API Keys**: Keys are prefixed with `ara_` for easy identification. Implement database validation for production.

4. **Stripe Webhooks**: Always verify webhook signatures to prevent spoofing.

5. **Token Expiry**: Access tokens expire after 24 hours by default. Use refresh tokens for long-lived sessions.

## Database Schema

The usage tracker creates these tables:

```sql
-- Usage events
CREATE TABLE usage_events (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT,
    event_type TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Monthly summaries (materialized for performance)
CREATE TABLE usage_monthly (
    org_id TEXT NOT NULL,
    month TEXT NOT NULL,  -- YYYY-MM format
    debates INTEGER DEFAULT 0,
    api_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0,
    PRIMARY KEY (org_id, month)
);
```

## See Also

- [API Reference](API_REFERENCE.md) - Authentication endpoints
- [Environment Variables](ENVIRONMENT.md) - Full configuration reference
- [Architecture](ARCHITECTURE.md) - System overview

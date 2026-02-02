# Aragora Billing Module

The Aragora billing module provides comprehensive monetization infrastructure for multi-tenant SaaS deployments. It handles subscription management, usage metering, cost tracking, Stripe integration, and budget enforcement.

## Architecture Overview

```
aragora/billing/
├── models.py              # Core data models (User, Organization, Subscription, Tiers)
├── stripe_client.py       # Stripe API integration
├── metering.py            # Tenant-aware usage metering
├── enterprise_metering.py # Enterprise-grade token-level metering
├── cost_tracker.py        # Real-time cost attribution and budgets
├── budget_manager.py      # Budget CRUD and enforcement
├── usage.py               # Basic usage tracking
├── discounts.py           # Promo codes and volume discounts
├── credits.py             # Prepaid credit management
├── notifications.py       # Email/webhook billing alerts
├── forecaster.py          # Usage forecasting
├── roi_calculator.py      # ROI metrics for debates
├── invoice_export.py      # PDF/HTML invoice generation
├── multi_org.py           # Multi-organization management
├── partner.py             # Partner/reseller billing
├── payment_recovery.py    # Failed payment handling
└── auth/                  # Billing authentication
```

## Subscription Tiers

Aragora supports five subscription tiers with progressively more features:

| Tier | Price/Month | Debates | Users | Key Features |
|------|-------------|---------|-------|--------------|
| **Free** | $0 | 10 | 1 | Basic access |
| **Starter** | $99 | 50 | 2 | - |
| **Professional** | $299 | 200 | 10 | API access, all agents, audit logs |
| **Enterprise** | $999 | Unlimited | Unlimited | SSO, custom agents, priority support |
| **Enterprise+** | $5,000 | Unlimited | Unlimited | Dedicated infra, SLA, custom models, compliance |

## Quick Start

### Basic Usage Tracking

```python
from aragora.billing import UsageTracker, UsageEvent, UsageEventType

# Initialize tracker
tracker = UsageTracker()

# Record a debate
tracker.record_debate(
    user_id="user_123",
    org_id="org_456",
    debate_id="debate_789",
    tokens_in=5000,
    tokens_out=2000,
    provider="anthropic",
    model="claude-opus-4"
)

# Get usage summary
summary = tracker.get_summary(org_id="org_456")
print(f"Total cost: ${summary.total_cost_usd}")
```

### Tenant-Aware Metering

```python
from aragora.billing import UsageMeter, get_usage_meter

# Get or create the global meter
meter = get_usage_meter()

# Start the metering system (enables background flushing)
await meter.start()

# Record various events
await meter.record_api_call(resource="debates/create")
await meter.record_debate(
    debate_id="debate_123",
    tokens_in=10000,
    tokens_out=5000,
    rounds=3
)
await meter.record_tokens(
    tokens_in=1000,
    tokens_out=500,
    provider="openai",
    model="gpt-4o"
)

# Get billing summary
summary = await meter.get_usage_summary(
    start_date=datetime(2025, 1, 1),
    end_date=datetime.now()
)
print(f"Total events: {summary.total_events}")
print(f"Total cost: ${summary.total_cost}")

# Stop metering (flushes remaining events)
await meter.stop()
```

### Enterprise Token Metering

For Enterprise+ tier with granular token-level billing:

```python
from aragora.billing import EnterpriseMeter, get_enterprise_meter

meter = get_enterprise_meter()
await meter.initialize()

# Record token usage with full attribution
record = await meter.record_token_usage(
    provider="anthropic",
    model="claude-opus-4",
    tokens_in=15000,
    tokens_out=3000,
    tenant_id="tenant_123",
    debate_id="debate_456",
    agent_id="agent_claude",
    request_type="debate",
    cached_tokens=2000,  # Cached tokens get discounts
    latency_ms=1250
)

# Get detailed cost breakdown
breakdown = await meter.get_cost_breakdown(
    tenant_id="tenant_123",
    start_date=datetime(2025, 1, 1)
)
print(f"Total cost: ${breakdown.total_cost}")
print(f"By provider: {breakdown.cost_by_provider}")
print(f"By model: {breakdown.cost_by_model}")

# Forecast future usage
forecast = await meter.forecast_usage(
    tenant_id="tenant_123",
    days_ahead=30
)
print(f"Projected cost: ${forecast.projected_cost}")
print(f"Will exceed budget: {forecast.will_exceed_budget}")

# Generate monthly invoice
invoice = await meter.generate_invoice(
    tenant_id="tenant_123",
    period="2025-01",
    tax_rate=Decimal("0.1")  # 10% tax
)
print(f"Invoice {invoice.id}: ${invoice.total}")
```

### Cost Tracking with Budgets

```python
from aragora.billing import CostTracker, Budget, get_cost_tracker, record_usage
from decimal import Decimal

tracker = get_cost_tracker()

# Set a budget
budget = Budget(
    name="Q1 Budget",
    workspace_id="ws_123",
    monthly_limit_usd=Decimal("1000"),
    daily_limit_usd=Decimal("50"),
    per_debate_limit_usd=Decimal("5")
)
tracker.set_budget(budget)

# Register alert callback
def on_budget_alert(alert):
    print(f"Budget alert: {alert.level.value} - {alert.message}")

tracker.add_alert_callback(on_budget_alert)

# Record usage (auto-calculates cost)
usage = await record_usage(
    workspace_id="ws_123",
    agent_name="claude",
    provider="anthropic",
    model="claude-opus-4",
    tokens_in=10000,
    tokens_out=5000,
    debate_id="debate_789"
)

# Check budget status
status = tracker.check_debate_budget("debate_789")
if not status["allowed"]:
    raise Exception(status["message"])

# Generate cost report
report = await tracker.generate_report(
    workspace_id="ws_123",
    granularity=CostGranularity.DAILY
)
```

### Per-Debate Budget Enforcement

```python
from aragora.billing import CostTracker, DebateBudgetExceededError
from decimal import Decimal

tracker = get_cost_tracker()

# Set a hard limit for a specific debate
tracker.set_debate_limit("debate_123", Decimal("10.00"))

# Before each agent call, check budget
status = tracker.check_debate_budget("debate_123", estimated_cost_usd=Decimal("0.50"))
if not status["allowed"]:
    raise DebateBudgetExceededError(
        debate_id="debate_123",
        current_cost=Decimal(status["current_cost"]),
        limit=Decimal(status["limit"])
    )

# Record cost after successful call
tracker.record_debate_cost("debate_123", Decimal("0.45"))

# Clean up when debate ends
tracker.clear_debate_budget("debate_123")
```

## Stripe Integration

### Configuration

Set the following environment variables:

```bash
# Required
STRIPE_SECRET_KEY=sk_live_xxx          # Stripe secret key
STRIPE_WEBHOOK_SECRET=whsec_xxx        # Webhook signing secret

# Price IDs for each tier
STRIPE_PRICE_STARTER=price_xxx
STRIPE_PRICE_PROFESSIONAL=price_xxx
STRIPE_PRICE_ENTERPRISE=price_xxx
STRIPE_PRICE_ENTERPRISE_PLUS=price_xxx

# Metered billing price IDs (for Enterprise+)
STRIPE_METERED_TOKENS_INPUT=price_xxx
STRIPE_METERED_TOKENS_OUTPUT=price_xxx
STRIPE_METERED_DEBATES=price_xxx
```

### Creating Checkout Sessions

```python
from aragora.billing import StripeClient, SubscriptionTier, get_stripe_client

client = get_stripe_client()

# Create a checkout session for subscription
session = client.create_checkout_session(
    tier=SubscriptionTier.PROFESSIONAL,
    customer_email="user@example.com",
    success_url="https://app.example.com/success?session_id={CHECKOUT_SESSION_ID}",
    cancel_url="https://app.example.com/pricing",
    metadata={"org_id": "org_123"},
    trial_days=14
)

# Redirect user to session.url
print(f"Checkout URL: {session.url}")
```

### Managing Subscriptions

```python
# Get subscription details
subscription = client.get_subscription("sub_xxx")
print(f"Status: {subscription.status}")
print(f"Period ends: {subscription.current_period_end}")

# Cancel subscription at period end
updated = client.cancel_subscription("sub_xxx", at_period_end=True)

# Resume a canceled subscription
resumed = client.resume_subscription("sub_xxx")
```

### Billing Portal

```python
# Create portal session for self-service billing management
portal = client.create_portal_session(
    customer_id="cus_xxx",
    return_url="https://app.example.com/settings"
)
# Redirect user to portal.url
```

### Metered Billing (Enterprise+)

```python
# Find the subscription item for token billing
token_item_id = client.find_metered_subscription_item(
    subscription_id="sub_xxx",
    price_id="price_tokens_input"
)

# Report usage (e.g., every hour or at debate end)
record = client.report_usage(
    subscription_item_id=token_item_id,
    quantity=15000,  # Tokens used
    idempotency_key=f"usage-{debate_id}-{datetime.now().hour}"
)

# Get usage summary
summary = client.get_usage_summary(token_item_id)
print(f"Total usage this period: {summary['total_usage']}")
```

### Webhook Handling

```python
from aragora.billing import parse_webhook_event, verify_webhook_signature

# In your webhook endpoint
@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("stripe-signature")

    # Parse and verify webhook
    event = parse_webhook_event(payload, signature)
    if not event:
        return {"error": "Invalid signature"}, 400

    # Handle events
    if event.type == "customer.subscription.created":
        subscription_id = event.subscription_id
        metadata = event.metadata
        # Activate subscription in your system

    elif event.type == "customer.subscription.deleted":
        # Handle cancellation

    elif event.type == "invoice.payment_failed":
        # Handle failed payment

    return {"status": "ok"}
```

## Discount Codes

```python
from aragora.billing import DiscountManager, get_discount_manager
from datetime import datetime, timedelta

manager = get_discount_manager()

# Create a promo code
code = await manager.create_code(
    code="WELCOME50",
    discount_percent=50,
    max_uses=1000,
    max_uses_per_org=1,
    expires_at=datetime.now() + timedelta(days=30),
    description="50% off first month"
)

# Validate before checkout
result = await manager.validate_code(
    code="WELCOME50",
    org_id="org_123",
    purchase_amount_cents=29900,
    tier="professional"
)

if result.valid:
    print(f"Discount: ${result.discount_amount_cents / 100}")
else:
    print(f"Invalid: {result.message}")

# Apply the code (records usage)
result = await manager.apply_code(
    code="WELCOME50",
    org_id="org_123",
    purchase_amount_cents=29900,
    user_id="user_456"
)
```

### Volume Discounts

```python
# Get current volume discount tier
volume = await manager.get_volume_discount(org_id="org_123")
print(f"Current tier: {volume.current_discount_percent}%")
print(f"Cumulative spend: ${volume.cumulative_spend_cents / 100}")

# Update after purchase (advances tier automatically)
updated = await manager.update_volume_spend(
    org_id="org_123",
    spend_cents=50000_00  # $50,000
)
print(f"New discount tier: {updated.current_discount_percent}%")
```

Default volume tiers:
- $10k+ cumulative: 5% discount
- $50k+ cumulative: 10% discount
- $100k+ cumulative: 15% discount
- $500k+ cumulative: 20% discount

## Credits System

```python
from aragora.billing import CreditManager, CreditTransactionType, get_credit_manager
from datetime import timedelta

manager = get_credit_manager()

# Issue promotional credits
await manager.issue_credit(
    org_id="org_123",
    amount_cents=5000,  # $50
    credit_type=CreditTransactionType.PROMOTIONAL,
    description="Welcome bonus",
    expires_in_days=90
)

# Check balance
balance = await manager.get_balance("org_123")
print(f"Available credits: ${balance.balance_cents / 100}")

# Deduct credits during billing
result = await manager.deduct_credits(
    org_id="org_123",
    amount_cents=2500,  # $25
    description="Monthly subscription"
)
print(f"Deducted: ${result.amount_deducted_cents / 100}")
print(f"Still owed: ${result.remaining_amount_cents / 100}")

# Get transaction history
transactions = await manager.get_transactions(org_id="org_123", limit=50)
```

## Budget Management

```python
from aragora.billing.budget_manager import BudgetManager, Budget, BudgetPeriod, BudgetAction

manager = BudgetManager()

# Create a monthly budget with thresholds
budget = await manager.create_budget(
    org_id="org_123",
    name="Engineering Team Budget",
    amount_usd=5000.0,
    period=BudgetPeriod.MONTHLY,
    auto_suspend=True,  # Stop operations when exceeded
    thresholds=[
        BudgetThreshold(0.50, BudgetAction.NOTIFY),
        BudgetThreshold(0.75, BudgetAction.WARN),
        BudgetThreshold(0.90, BudgetAction.SOFT_LIMIT),
        BudgetThreshold(1.00, BudgetAction.HARD_LIMIT)
    ]
)

# Check if spend is allowed
result = await manager.can_spend(
    org_id="org_123",
    amount_usd=150.0,
    user_id="user_456"
)
if not result.allowed:
    raise Exception(result.message)

# Record spend
await manager.record_spend(
    org_id="org_123",
    amount_usd=150.0
)

# Grant temporary override (for admins)
await manager.grant_override(
    org_id="org_123",
    user_id="admin_789",
    duration_hours=24
)
```

## Notifications

Configure SMTP for email notifications:

```bash
ARAGORA_SMTP_HOST=smtp.sendgrid.net
ARAGORA_SMTP_PORT=587
ARAGORA_SMTP_USER=apikey
ARAGORA_SMTP_PASSWORD=SG.xxx
ARAGORA_SMTP_FROM=billing@aragora.ai

# Optional webhook for Slack/Discord
ARAGORA_NOTIFICATION_WEBHOOK=https://hooks.slack.com/services/xxx
```

```python
from aragora.billing import BillingNotifier, get_billing_notifier

notifier = get_billing_notifier()

# Send payment failure notice
result = notifier.notify_payment_failed(
    email="user@example.com",
    org_name="Acme Corp",
    amount="$299.00",
    retry_url="https://app.example.com/billing"
)

# Send trial expiration warning
result = notifier.notify_trial_expiring(
    email="user@example.com",
    org_name="Acme Corp",
    days_remaining=3,
    upgrade_url="https://app.example.com/upgrade"
)

# Send budget alert
result = notifier.notify_budget_alert(
    tenant_id="tenant_123",
    email="admin@example.com",
    alert_level="warning",
    current_spend="$750.00",
    budget_limit="$1,000.00",
    percent_used=75.0
)
```

## Invoice Export

```python
from aragora.billing import InvoiceExporter, InvoiceCompanyInfo, InvoiceCustomerInfo

# Requires: pip install reportlab

exporter = InvoiceExporter(
    company=InvoiceCompanyInfo(
        name="Aragora Inc.",
        address="123 AI Street, San Francisco, CA 94105",
        email="billing@aragora.ai"
    )
)

# Export to PDF
pdf_path = await exporter.export_pdf(
    invoice_id="INV-2025-001",
    customer=InvoiceCustomerInfo(
        name="Acme Corp",
        email="billing@acme.com",
        address="456 Tech Blvd"
    ),
    line_items=[
        {"description": "Professional Plan", "amount": "$299.00"},
        {"description": "API Overage (50k tokens)", "amount": "$25.00"}
    ],
    subtotal="$324.00",
    tax="$32.40",
    total="$356.40"
)

# Export to HTML
html = await exporter.export_html(invoice_id="INV-2025-001", ...)
```

## Multi-Organization Support

```python
from aragora.billing import MultiOrgManager, MembershipRole, get_multi_org_manager

manager = get_multi_org_manager()

# Create organization
org = await manager.create_organization(
    name="Acme Corp",
    owner_id="user_123"
)

# Invite members
await manager.invite_member(
    org_id=org.id,
    email="colleague@acme.com",
    role=MembershipRole.ADMIN,
    invited_by="user_123"
)

# Get user's organizations
orgs = await manager.get_user_organizations("user_123")

# Check permissions
permissions = manager.get_role_permissions(MembershipRole.ADMIN)
```

## Provider Pricing

The module includes current pricing for major LLM providers (per 1M tokens):

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Anthropic | claude-opus-4 | $15.00 | $75.00 |
| Anthropic | claude-sonnet-4 | $3.00 | $15.00 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Google | gemini-pro | $1.25 | $5.00 |
| DeepSeek | deepseek-v3 | $0.14 | $0.28 |
| OpenRouter | default | $2.00 | $8.00 |

## API Reference

### Key Exports

```python
from aragora.billing import (
    # Models
    User, Organization, Subscription, SubscriptionTier, TierLimits, TIER_LIMITS,

    # Usage Tracking
    UsageTracker, UsageEvent, UsageEventType, UsageSummary,

    # Tenant Metering
    UsageMeter, BillingEvent, BillingEventType, MeteringConfig,
    get_usage_meter, record_billing_usage,

    # Enterprise Metering
    EnterpriseMeter, TokenUsageRecord, BudgetConfig, CostBreakdown,
    Invoice, InvoiceStatus, UsageForecast, get_enterprise_meter,

    # Cost Tracking
    CostTracker, TokenUsage, Budget, BudgetAlert, BudgetAlertLevel,
    CostReport, CostGranularity, DebateBudgetExceededError,
    get_cost_tracker, record_usage,

    # Stripe Integration
    StripeClient, StripeCustomer, StripeSubscription,
    CheckoutSession, BillingPortalSession,
    get_stripe_client, parse_webhook_event, verify_webhook_signature,

    # Discounts
    DiscountManager, DiscountCode, DiscountType, ApplyCodeResult,
    VolumeDiscount, VolumeTier, get_discount_manager,

    # Credits
    CreditManager, CreditTransaction, CreditTransactionType,
    CreditAccount, DeductionResult, get_credit_manager,

    # Multi-Org
    MultiOrgManager, MembershipRole, MembershipStatus,
    OrganizationMembership, get_multi_org_manager,

    # Notifications
    BillingNotifier, NotificationResult, get_billing_notifier,

    # ROI
    ROICalculator, ROIMetrics, DebateROIInput, get_roi_calculator,

    # Invoice Export (optional dependency)
    InvoiceExporter, InvoiceCompanyInfo, InvoiceCustomerInfo,
    export_invoice_pdf, export_invoice_html,
)
```

### Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `STRIPE_SECRET_KEY` | Stripe API secret key | - |
| `STRIPE_WEBHOOK_SECRET` | Webhook signing secret | - |
| `STRIPE_PRICE_*` | Price IDs for each tier | - |
| `ARAGORA_SMTP_HOST` | SMTP server host | - |
| `ARAGORA_SMTP_PORT` | SMTP server port | 587 |
| `ARAGORA_SMTP_USER` | SMTP username | - |
| `ARAGORA_SMTP_PASSWORD` | SMTP password | - |
| `ARAGORA_SMTP_FROM` | From email address | billing@aragora.ai |
| `ARAGORA_NOTIFICATION_WEBHOOK` | Webhook URL for alerts | - |
| `ARAGORA_DATA_DIR` | Data directory for SQLite | ~/.nomic |

## Testing

The billing module is designed for testability:

```python
from aragora.billing import MeteringConfig, UsageMeter

# Create meter with persistence disabled for testing
config = MeteringConfig(persist_events=False)
meter = UsageMeter(config)

# Use in-memory tracking only
await meter.record_api_call(resource="test")
events = await meter.get_billing_events(start, end)
```

## Knowledge Mound Integration

The CostTracker integrates with Knowledge Mound for persistent cost analytics:

```python
from aragora.billing import get_cost_tracker

tracker = get_cost_tracker()

# KM adapter is auto-wired if available
# Query historical cost patterns
patterns = tracker.query_km_cost_patterns(workspace_id="ws_123")

# Get workspace alerts from KM
alerts = tracker.query_km_workspace_alerts(
    workspace_id="ws_123",
    min_level="warning"
)

# Detect and store cost anomalies
anomalies = await tracker.detect_and_store_anomalies(workspace_id="ws_123")
```

## Related Documentation

- [Enterprise Features](/docs/ENTERPRISE_FEATURES.md) - Full enterprise capabilities
- [API Reference](/docs/API_REFERENCE.md) - REST API endpoints
- [Multi-Tenancy](/aragora/tenancy/README.md) - Tenant isolation

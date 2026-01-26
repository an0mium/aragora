# Operational Connectors Guide

Business operations and SaaS integrations.

## Overview

Operational connectors integrate Aragora with day-to-day business tools for analytics, support, payments, e-commerce, and marketing operations.

---

## Chat Platform Connectors

### Discord Connector

Discord bot integration for community debates.

```python
from aragora.connectors.chat.discord import DiscordConnector

connector = DiscordConnector(
    bot_token="xxx",  # or DISCORD_BOT_TOKEN env var
)

# Send message
await connector.send_message(
    channel_id="123456789",
    content="Decision reached!",
    embed={
        "title": "Debate Result",
        "description": "The consensus was...",
        "color": 0x00ff00,
    },
)

# Reply in thread
await connector.reply(
    channel_id="123456789",
    message_id="987654321",
    content="Here's the detailed analysis...",
)
```

### Google Chat Connector

```python
from aragora.connectors.chat.google_chat import GoogleChatConnector

connector = GoogleChatConnector(
    credentials_file="/path/to/credentials.json",
)

# Send message
await connector.send_message(
    space="spaces/xxx",
    text="Decision reached!",
)

# Send card
await connector.send_card(
    space="spaces/xxx",
    card={
        "header": {"title": "Debate Result"},
        "sections": [...],
    },
)
```

### Telegram Connector

```python
from aragora.connectors.chat.telegram import TelegramConnector

connector = TelegramConnector(
    bot_token="xxx:xxx",  # or TELEGRAM_BOT_TOKEN env var
)

# Send message
await connector.send_message(
    chat_id=123456789,
    text="Decision reached!",
    parse_mode="Markdown",
)

# Send with inline keyboard
await connector.send_message(
    chat_id=123456789,
    text="Do you approve?",
    reply_markup={
        "inline_keyboard": [
            [{"text": "Yes", "callback_data": "approve"}],
            [{"text": "No", "callback_data": "reject"}],
        ],
    },
)
```

### WhatsApp Connector

```python
from aragora.connectors.chat.whatsapp import WhatsAppConnector

connector = WhatsAppConnector(
    account_sid="xxx",
    auth_token="xxx",
    from_number="whatsapp:+1234567890",
)

# Send message
await connector.send_message(
    to="whatsapp:+0987654321",
    body="Decision reached: Approved",
)
```

---

## Analytics Connectors

### Google Analytics Connector

```python
from aragora.connectors.analytics.google_analytics import GoogleAnalyticsConnector

connector = GoogleAnalyticsConnector(
    credentials_file="/path/to/credentials.json",
    property_id="123456789",
)

# Run report
report = await connector.run_report(
    date_ranges=[{"start_date": "30daysAgo", "end_date": "today"}],
    dimensions=[{"name": "country"}, {"name": "city"}],
    metrics=[{"name": "activeUsers"}, {"name": "sessions"}],
)

# Get realtime data
realtime = await connector.get_realtime(
    dimensions=["country"],
    metrics=["activeUsers"],
)
```

### Mixpanel Connector

```python
from aragora.connectors.analytics.mixpanel import MixpanelConnector

connector = MixpanelConnector(
    project_token="xxx",
    api_secret="xxx",
)

# Export events
events = await connector.export_events(
    from_date="2024-01-01",
    to_date="2024-01-31",
    event="Decision Made",
)

# Query insights
insights = await connector.insights_query(
    script="""
    function main() {
        return Events({
            from_date: '2024-01-01',
            to_date: '2024-01-31',
            event_selectors: [{event: 'Decision Made'}]
        }).groupBy(['properties.$browser'], mixpanel.reducer.count());
    }
    """,
)
```

### Segment Connector

```python
from aragora.connectors.analytics.segment import SegmentConnector

connector = SegmentConnector(write_key="xxx")

# Track event
await connector.track(
    user_id="user_123",
    event="Decision Made",
    properties={
        "debate_id": "debate_456",
        "decision": "approved",
    },
)

# Identify user
await connector.identify(
    user_id="user_123",
    traits={
        "name": "John Doe",
        "email": "john@example.com",
    },
)
```

---

## Support Connectors

### Zendesk Connector

```python
from aragora.connectors.support.zendesk import ZendeskConnector

connector = ZendeskConnector(
    subdomain="company",
    email="user@company.com",
    api_token="xxx",
)

# Create ticket
ticket = await connector.create_ticket(
    subject="Decision Follow-up Required",
    description="Based on debate #123...",
    priority="normal",
    tags=["aragora", "decision"],
)

# Search tickets
tickets = await connector.search(
    query="type:ticket status:open tags:aragora",
)

# Add comment
await connector.add_comment(
    ticket_id=12345,
    body="Update from Aragora debate...",
    public=True,
)
```

### Freshdesk Connector

```python
from aragora.connectors.support.freshdesk import FreshdeskConnector

connector = FreshdeskConnector(
    domain="company.freshdesk.com",
    api_key="xxx",
)

# Create ticket
ticket = await connector.create_ticket(
    subject="Decision Implementation",
    description="...",
    email="requester@example.com",
    priority=2,
)
```

### Intercom Connector

```python
from aragora.connectors.support.intercom import IntercomConnector

connector = IntercomConnector(access_token="xxx")

# Create conversation
conversation = await connector.create_conversation(
    user_id="user_123",
    body="Decision notification...",
)

# Send message
await connector.send_message(
    conversation_id="123",
    body="Follow-up on decision...",
)
```

### Help Scout Connector

```python
from aragora.connectors.support.helpscout import HelpScoutConnector

connector = HelpScoutConnector(
    client_id="xxx",
    client_secret="xxx",
)

# Create conversation
conversation = await connector.create_conversation(
    mailbox_id=12345,
    subject="Decision Follow-up",
    customer={"email": "customer@example.com"},
    threads=[{"type": "customer", "text": "..."}],
)
```

---

## Payment Connectors

### Stripe Connector

```python
from aragora.connectors.payments.stripe import StripeConnector

connector = StripeConnector(api_key="sk_xxx")

# List payments
payments = await connector.list_payments(
    limit=100,
    created={"gte": 1704067200},  # After 2024-01-01
)

# Get customer
customer = await connector.get_customer(customer_id="cus_xxx")

# Create invoice
invoice = await connector.create_invoice(
    customer="cus_xxx",
    items=[{"price": "price_xxx", "quantity": 1}],
)
```

### PayPal Connector

```python
from aragora.connectors.payments.paypal import PayPalConnector

connector = PayPalConnector(
    client_id="xxx",
    client_secret="xxx",
    sandbox=False,
)

# List transactions
transactions = await connector.list_transactions(
    start_date="2024-01-01T00:00:00Z",
    end_date="2024-01-31T23:59:59Z",
)
```

### Square Connector

```python
from aragora.connectors.payments.square import SquareConnector

connector = SquareConnector(
    access_token="xxx",
    location_id="xxx",
)

# List payments
payments = await connector.list_payments(
    begin_time="2024-01-01T00:00:00Z",
    end_time="2024-01-31T23:59:59Z",
)
```

---

## E-commerce Connectors

### Shopify Connector

```python
from aragora.connectors.ecommerce.shopify import ShopifyConnector

connector = ShopifyConnector(
    shop_name="my-store",
    access_token="xxx",
)

# List orders
orders = await connector.list_orders(
    status="any",
    created_at_min="2024-01-01",
)

# Get product
product = await connector.get_product(product_id=123456789)

# Update inventory
await connector.update_inventory(
    inventory_item_id=12345,
    quantity=100,
)
```

### ShipStation Connector

```python
from aragora.connectors.ecommerce.shipstation import ShipStationConnector

connector = ShipStationConnector(
    api_key="xxx",
    api_secret="xxx",
)

# List orders
orders = await connector.list_orders(
    order_status="awaiting_shipment",
)

# Create shipment
shipment = await connector.create_shipment(
    order_id=12345,
    carrier_code="fedex",
    service_code="fedex_ground",
)
```

### WooCommerce Connector

```python
from aragora.connectors.ecommerce.woocommerce import WooCommerceConnector

connector = WooCommerceConnector(
    url="https://mystore.com",
    consumer_key="ck_xxx",
    consumer_secret="cs_xxx",
)

# List orders
orders = await connector.list_orders(status="processing")

# Update order
await connector.update_order(
    order_id=123,
    data={"status": "completed"},
)
```

---

## Accounting Connectors

### QuickBooks Connector

```python
from aragora.connectors.accounting.quickbooks import QuickBooksConnector

connector = QuickBooksConnector(
    client_id="xxx",
    client_secret="xxx",
    refresh_token="xxx",
    company_id="xxx",
)

# List invoices
invoices = await connector.list_invoices(
    query="SELECT * FROM Invoice WHERE TotalAmt > '1000'",
)

# Create invoice
invoice = await connector.create_invoice(
    customer_ref={"value": "123"},
    line=[{
        "Amount": 100.00,
        "DetailType": "SalesItemLineDetail",
        "SalesItemLineDetail": {"ItemRef": {"value": "1"}},
    }],
)
```

### Xero Connector

```python
from aragora.connectors.accounting.xero import XeroConnector

connector = XeroConnector(
    client_id="xxx",
    client_secret="xxx",
    refresh_token="xxx",
)

# List invoices
invoices = await connector.list_invoices(
    statuses=["AUTHORISED", "PAID"],
)

# Get bank transactions
transactions = await connector.list_bank_transactions(
    bank_account_id="xxx",
)
```

---

## Advertising Connectors

### Google Ads Connector

```python
from aragora.connectors.advertising.google_ads import GoogleAdsConnector

connector = GoogleAdsConnector(
    developer_token="xxx",
    client_id="xxx",
    client_secret="xxx",
    refresh_token="xxx",
    customer_id="123-456-7890",
)

# Get campaign performance
campaigns = await connector.get_campaigns(
    date_range="LAST_30_DAYS",
    metrics=["impressions", "clicks", "cost_micros"],
)
```

### Meta Ads Connector

```python
from aragora.connectors.advertising.meta import MetaAdsConnector

connector = MetaAdsConnector(
    access_token="xxx",
    ad_account_id="act_123456789",
)

# Get campaign insights
insights = await connector.get_insights(
    level="campaign",
    fields=["impressions", "clicks", "spend"],
    date_preset="last_30d",
)
```

---

## Calendar Connectors

### Google Calendar Connector

```python
from aragora.connectors.calendar.google_calendar import GoogleCalendarConnector

connector = GoogleCalendarConnector(
    credentials_file="/path/to/credentials.json",
)

# List events
events = await connector.list_events(
    calendar_id="primary",
    time_min="2024-01-01T00:00:00Z",
    max_results=100,
)

# Create event
event = await connector.create_event(
    calendar_id="primary",
    summary="Decision Review Meeting",
    start={"dateTime": "2024-02-01T10:00:00-07:00"},
    end={"dateTime": "2024-02-01T11:00:00-07:00"},
    attendees=[{"email": "team@company.com"}],
)
```

### Outlook Calendar Connector

```python
from aragora.connectors.calendar.outlook_calendar import OutlookCalendarConnector

connector = OutlookCalendarConnector(
    client_id="xxx",
    client_secret="xxx",
)

# List events
events = await connector.list_events(
    start="2024-01-01T00:00:00Z",
    end="2024-01-31T23:59:59Z",
)
```

---

## Low-Code Platform Connectors

### Airtable Connector

```python
from aragora.connectors.lowcode.airtable import AirtableConnector

connector = AirtableConnector(api_key="xxx")

# List records
records = await connector.list_records(
    base_id="appXXX",
    table_name="Decisions",
    filter_by_formula="{Status} = 'Active'",
)

# Create record
record = await connector.create_record(
    base_id="appXXX",
    table_name="Decisions",
    fields={"Name": "New Decision", "Status": "Pending"},
)
```

### Knack Connector

```python
from aragora.connectors.lowcode.knack import KnackConnector

connector = KnackConnector(
    application_id="xxx",
    api_key="xxx",
)

# Get records
records = await connector.get_records(
    object_key="object_1",
    filters=[{"field": "field_1", "operator": "is", "value": "Active"}],
)
```

---

## Webhook Security

All chat connectors support webhook verification:

```python
from aragora.connectors.chat.webhook_security import verify_webhook

# Slack
is_valid = verify_webhook(
    provider="slack",
    signature=request.headers["X-Slack-Signature"],
    timestamp=request.headers["X-Slack-Request-Timestamp"],
    body=request.body,
    secret=signing_secret,
)

# Discord
is_valid = verify_webhook(
    provider="discord",
    signature=request.headers["X-Signature-Ed25519"],
    timestamp=request.headers["X-Signature-Timestamp"],
    body=request.body,
    public_key=public_key,
)
```

---

## Environment Variables

```bash
# Chat platforms
DISCORD_BOT_TOKEN=xxx
TELEGRAM_BOT_TOKEN=xxx
SLACK_BOT_TOKEN=xxx

# Analytics
GOOGLE_ANALYTICS_PROPERTY_ID=xxx
MIXPANEL_PROJECT_TOKEN=xxx
SEGMENT_WRITE_KEY=xxx

# Payments
STRIPE_API_KEY=sk_xxx
PAYPAL_CLIENT_ID=xxx
PAYPAL_CLIENT_SECRET=xxx

# E-commerce
SHOPIFY_ACCESS_TOKEN=xxx
SHIPSTATION_API_KEY=xxx
SHIPSTATION_API_SECRET=xxx

# Accounting
QUICKBOOKS_CLIENT_ID=xxx
QUICKBOOKS_CLIENT_SECRET=xxx
XERO_CLIENT_ID=xxx
XERO_CLIENT_SECRET=xxx
```

---

## See Also

- [Connector Integration Index](../CONNECTOR_INTEGRATION_INDEX.md) - Master connector list
- [Evidence Connectors Guide](EVIDENCE_CONNECTORS.md) - Evidence sources
- [Enterprise Connectors Guide](ENTERPRISE_CONNECTORS.md) - Enterprise integrations

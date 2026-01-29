# Connectors - External Integration Hub

67+ connectors across 22 categories for evidence collection, chat integration, enterprise data sync, and Knowledge Mound ingestion.

## Quick Start

```python
# Evidence collection
from aragora.connectors import GitHubConnector

github = GitHubConnector(token="ghp_...")
evidence = await github.fetch_evidence(
    query="authentication bug",
    repo="org/repo"
)

# Chat integration
from aragora.connectors.chat import SlackConnector

slack = SlackConnector(bot_token="xoxb-...")
await slack.send_message(
    channel="#debates",
    text="Consensus reached: Use JWT authentication"
)

# Enterprise sync
from aragora.connectors.enterprise.collaboration import JiraConnector

jira = JiraConnector(url="https://org.atlassian.net", token="...")
await jira.sync_to_knowledge_mound(project="PROJ")
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `BaseConnector` | `base.py` | Evidence connector with caching, retry |
| `ChatPlatformConnector` | `chat/base.py` | Chat platform abstraction |
| `EnterpriseConnector` | `enterprise/base.py` | Enterprise sync with state tracking |
| `Evidence` | `base.py` | Evidence with provenance tracking |
| `SyncState` | `enterprise/base.py` | Incremental sync cursor |

## Architecture

```
connectors/
├── base.py                  # BaseConnector, Evidence, exceptions
├── Chat Platforms (8)
│   ├── chat/
│   │   ├── slack.py         # Slack
│   │   ├── teams.py         # Microsoft Teams
│   │   ├── discord.py       # Discord
│   │   ├── telegram.py      # Telegram
│   │   ├── whatsapp.py      # WhatsApp
│   │   ├── google_chat.py   # Google Chat
│   │   ├── imessage.py      # iMessage
│   │   └── signal.py        # Signal
├── Enterprise (19+)
│   ├── enterprise/
│   │   ├── collaboration/   # Jira, Confluence, Asana, Linear, Monday, Notion
│   │   ├── communication/   # Gmail, Outlook
│   │   ├── crm/             # Salesforce
│   │   ├── database/        # PostgreSQL, MySQL, MongoDB, Snowflake, SQL Server
│   │   ├── documents/       # Google Drive, Sheets, OneDrive, SharePoint, Dropbox, S3
│   │   ├── git/             # GitHub Enterprise
│   │   ├── healthcare/      # FHIR, HL7v2, Epic, Cerner
│   │   ├── itsm/            # ServiceNow
│   │   └── streaming/       # Kafka, RabbitMQ
├── Evidence Sources (12)
│   ├── github.py            # GitHub issues/PRs
│   ├── web.py               # Web search/scraping
│   ├── arxiv.py             # Academic papers
│   ├── hackernews.py        # HackerNews
│   ├── wikipedia.py         # Wikipedia
│   ├── reddit.py            # Reddit
│   ├── twitter.py           # Twitter/X
│   ├── newsapi.py           # News articles
│   ├── sec.py               # SEC filings
│   ├── sql.py               # Direct SQL queries
│   ├── whisper.py           # Audio transcription
│   └── local_docs.py        # Local documentation
├── Business (20+)
│   ├── accounting/          # Xero, QuickBooks, Gusto, Plaid
│   ├── advertising/         # Google Ads, Meta, LinkedIn, Twitter, Microsoft, TikTok
│   ├── analytics/           # Google Analytics, Mixpanel, Segment, Metabase
│   ├── ecommerce/           # Shopify, WooCommerce, Amazon, ShipStation
│   ├── payments/            # Stripe, PayPal, Square, Authorize.Net
│   ├── crm/                 # HubSpot, Pipedrive
│   ├── marketing/           # Mailchimp, Klaviyo
│   └── support/             # Zendesk, Freshdesk, Intercom, Help Scout
├── Specialized
│   ├── devices/             # Alexa, Google Home, Push notifications
│   ├── automation/          # Zapier, n8n
│   ├── legal/               # DocuSign
│   ├── devops/              # PagerDuty
│   ├── calendar/            # Google Calendar, Outlook Calendar
│   ├── browser/             # Playwright automation
│   └── feeds/               # RSS/Atom ingestion
└── Utilities
    ├── twitter_poster.py    # Post to Twitter
    ├── youtube_uploader.py  # Upload to YouTube
    └── repository_crawler.py  # Code analysis
```

## Chat Integration

```python
from aragora.connectors.chat import (
    SlackConnector, TeamsConnector, TelegramConnector,
    ChatMessage, ChannelContext
)

# Bidirectional debate routing
connector = SlackConnector(bot_token="xoxb-...")

# Receive messages
@connector.on_message
async def handle(msg: ChatMessage, ctx: ChannelContext):
    # Route to debate engine
    result = await arena.run(task=msg.text)
    await connector.reply(ctx, result.conclusion)

# Thread management
await connector.create_thread(channel, "Debate: API Design")

# Voice/TTS integration
from aragora.connectors.chat import TtsBridge
tts = TtsBridge(connector)
await tts.speak(channel, "Consensus reached")
```

## Enterprise Sync

```python
from aragora.connectors.enterprise.collaboration import JiraConnector
from aragora.connectors.enterprise.database import PostgreSQLConnector
from aragora.connectors.enterprise.streaming import KafkaConnector

# Incremental sync with state tracking
jira = JiraConnector(url="...", token="...")
sync_result = await jira.sync(
    project="PROJ",
    since=last_sync_time,
    to_knowledge_mound=True
)

# Database CDC (Change Data Capture)
postgres = PostgreSQLConnector(connection_string="...")
await postgres.enable_cdc(tables=["orders", "customers"])

# Event streaming
kafka = KafkaConnector(brokers=["localhost:9092"])
async for event in kafka.consume(topic="debate-events"):
    await process_event(event)
```

## Healthcare (HIPAA Compliant)

```python
from aragora.connectors.enterprise.healthcare import (
    FHIRConnector, HL7v2Connector, EpicAdapter
)

# FHIR R4 compliant
fhir = FHIRConnector(base_url="https://fhir.epic.com")
patient = await fhir.get_patient(patient_id="...")

# HL7 v2 message parsing
hl7 = HL7v2Connector()
message = hl7.parse(hl7_string)

# PHI redaction built-in
from aragora.connectors.enterprise.healthcare import PHIRedactor
redactor = PHIRedactor()
safe_text = redactor.redact(patient_notes)
```

## Resilience Features

All connectors include:

```python
# Circuit breaker (auto-enabled)
connector = GitHubConnector(
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60
)

# Retry with exponential backoff
connector = JiraConnector(
    max_retries=3,
    retry_backoff=2.0
)

# LRU caching with TTL
connector = WikipediaConnector(
    cache_ttl=3600  # 1 hour
)

# Rate limit handling
# Automatic Retry-After header support
```

## Exception Hierarchy

```python
from aragora.connectors import (
    ConnectorError,           # Base exception
    ConnectorAuthError,       # Authentication failures
    ConnectorAPIError,        # API errors
    ConnectorRateLimitError,  # Rate limiting
    ConnectorTimeoutError,    # Timeout errors
    ConnectorNetworkError,    # Network issues
    ConnectorCircuitOpenError # Circuit breaker open
)

try:
    evidence = await connector.fetch_evidence(query)
except ConnectorRateLimitError as e:
    await asyncio.sleep(e.retry_after)
except ConnectorCircuitOpenError:
    # Use fallback connector
    pass
```

## Connector Categories

| Category | Count | Examples |
|----------|-------|----------|
| Chat Platforms | 8 | Slack, Teams, Discord, Telegram, WhatsApp |
| Enterprise Collaboration | 8 | Jira, Confluence, Asana, Linear, Notion |
| Databases | 5 | PostgreSQL, MySQL, MongoDB, Snowflake |
| Document Storage | 6 | Google Drive, OneDrive, SharePoint, S3 |
| Healthcare | 4 | FHIR, HL7v2, Epic, Cerner |
| Evidence Sources | 12 | GitHub, arXiv, Wikipedia, HackerNews |
| Streaming | 2 | Kafka, RabbitMQ |
| Business | 20+ | Stripe, Salesforce, Shopify, Xero |

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Knowledge Mound](../knowledge/mound/README.md) - Knowledge storage
- [Control Plane](../control_plane/README.md) - Notifications to channels
- [Server Handlers](../server/handlers/social/README.md) - Chat HTTP endpoints

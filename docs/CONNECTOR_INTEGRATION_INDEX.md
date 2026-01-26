# Connector Integration Index

Master index of all connectors available in Aragora.

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Evidence Connectors | 14 | External knowledge sources |
| Enterprise Connectors | 34 | Business systems integration |
| Operational Connectors | 40 | Business operations tools |
| **Total** | **88** | |

---

## Quick Start

```python
from aragora.connectors import (
    GitHubConnector,
    SlackConnector,
    GoogleDriveConnector,
)

# Initialize a connector
github = GitHubConnector(
    token="ghp_xxx",
    organization="my-org",
)

# Fetch evidence
evidence = await github.fetch_issues(repo="my-repo", state="open")
```

---

## Evidence Connectors

Sources for gathering evidence for debates.

| Connector | Module | Purpose |
|-----------|--------|---------|
| ArXiv | `aragora.connectors.arxiv` | Academic papers |
| GitHub | `aragora.connectors.github` | Repository content, issues, PRs |
| HackerNews | `aragora.connectors.hackernews` | Tech news and discussions |
| Local Docs | `aragora.connectors.local_docs` | Local file system documents |
| NewsAPI | `aragora.connectors.newsapi` | News articles |
| Reddit | `aragora.connectors.reddit` | Subreddit discussions |
| Repository Crawler | `aragora.connectors.repository_crawler` | Git repository analysis |
| SEC | `aragora.connectors.sec` | SEC filings (10-K, 10-Q) |
| SQL | `aragora.connectors.sql` | Database queries |
| Twitter | `aragora.connectors.twitter` | Twitter/X posts |
| Web | `aragora.connectors.web` | Web page scraping |
| Whisper | `aragora.connectors.whisper` | Audio transcription |
| Wikipedia | `aragora.connectors.wikipedia` | Wikipedia articles |
| YouTube | `aragora.connectors.youtube_uploader` | Video content |

**See:** [Evidence Connectors Guide](connectors/EVIDENCE_CONNECTORS.md)

---

## Enterprise Connectors

Business system integrations for enterprise deployments.

### Collaboration (8)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Asana | `aragora.connectors.enterprise.collaboration.asana` | Task management |
| Confluence | `aragora.connectors.enterprise.collaboration.confluence` | Documentation |
| Jira | `aragora.connectors.enterprise.collaboration.jira` | Issue tracking |
| Linear | `aragora.connectors.enterprise.collaboration.linear` | Project management |
| Monday | `aragora.connectors.enterprise.collaboration.monday` | Work management |
| Notion | `aragora.connectors.enterprise.collaboration.notion` | Workspace |
| Slack | `aragora.connectors.enterprise.collaboration.slack` | Team communication |
| Teams | `aragora.connectors.enterprise.collaboration.teams` | Microsoft Teams |

### Documents (6)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Dropbox | `aragora.connectors.enterprise.documents.dropbox` | Cloud storage |
| Google Drive | `aragora.connectors.enterprise.documents.gdrive` | Cloud storage |
| OneDrive | `aragora.connectors.enterprise.documents.onedrive` | Cloud storage |
| S3 | `aragora.connectors.enterprise.documents.s3` | AWS object storage |
| SharePoint | `aragora.connectors.enterprise.documents.sharepoint` | Document management |
| Document Parser | `aragora.connectors.documents.parser` | Multi-format parsing |

### Database (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| PostgreSQL | `aragora.connectors.enterprise.database.postgres` | SQL database |
| MongoDB | `aragora.connectors.enterprise.database.mongodb` | NoSQL database |
| Snowflake | `aragora.connectors.enterprise.database.snowflake` | Data warehouse |
| MySQL | `aragora.connectors.enterprise.database.mysql` | SQL database |

### Healthcare (3)

| Connector | Module | Purpose |
|-----------|--------|---------|
| FHIR | `aragora.connectors.enterprise.healthcare.fhir` | Healthcare data |
| HL7v2 | `aragora.connectors.enterprise.healthcare.hl7v2` | Healthcare messaging |
| EHR Base | `aragora.connectors.enterprise.healthcare.ehr.base` | EHR integration |

### Communication (3)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Gmail | `aragora.connectors.enterprise.communication.gmail` | Email |
| Outlook | `aragora.connectors.enterprise.communication.outlook` | Email & Calendar |
| Email | `aragora.connectors.email` | Generic email |

### Streaming (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Kafka | `aragora.connectors.enterprise.streaming.kafka` | Event streaming |
| RabbitMQ | `aragora.connectors.enterprise.streaming.rabbitmq` | Message queue |

### CRM (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Salesforce | `aragora.connectors.enterprise.crm.salesforce` | CRM |
| Pipedrive | `aragora.connectors.crm.pipedrive` | Sales CRM |

### ITSM (1)

| Connector | Module | Purpose |
|-----------|--------|---------|
| ServiceNow | `aragora.connectors.enterprise.itsm.servicenow` | IT service management |

### Git (1)

| Connector | Module | Purpose |
|-----------|--------|---------|
| GitHub Enterprise | `aragora.connectors.enterprise.git.github` | Enterprise GitHub |

### DevOps (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| DevOps | `aragora.connectors.devops` | CI/CD integration |
| Metrics | `aragora.connectors.metrics` | Metrics collection |

### Legal (1)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Legal | `aragora.connectors.legal` | Legal document processing |

**See:** [Enterprise Connectors Guide](connectors/ENTERPRISE_CONNECTORS.md)

---

## Operational Connectors

Business operations and SaaS integrations.

### Chat Platforms (6)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Discord | `aragora.connectors.chat.discord` | Discord bot |
| Google Chat | `aragora.connectors.chat.google_chat` | Google Chat integration |
| Teams Chat | `aragora.connectors.chat.teams` | Teams messaging |
| Teams Conversations | `aragora.connectors.chat.teams_conversations` | Teams channels |
| Telegram | `aragora.connectors.chat.telegram` | Telegram bot |
| WhatsApp | `aragora.connectors.chat.whatsapp` | WhatsApp messaging |

### Advertising (6)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Google Ads | `aragora.connectors.advertising.google_ads` | Google advertising |
| Meta Ads | `aragora.connectors.advertising.meta` | Facebook/Instagram ads |
| LinkedIn Ads | `aragora.connectors.advertising.linkedin` | LinkedIn advertising |
| Microsoft Ads | `aragora.connectors.advertising.microsoft` | Bing advertising |
| TikTok Ads | `aragora.connectors.advertising.tiktok` | TikTok advertising |
| Twitter Ads | `aragora.connectors.advertising.twitter` | Twitter/X advertising |

### Analytics (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Google Analytics | `aragora.connectors.analytics.google_analytics` | Web analytics |
| Mixpanel | `aragora.connectors.analytics.mixpanel` | Product analytics |
| Segment | `aragora.connectors.analytics.segment` | Customer data platform |
| Metabase | `aragora.connectors.analytics.metabase` | BI dashboards |

### Accounting (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| QuickBooks | `aragora.connectors.accounting.quickbooks` | Accounting |
| Xero | `aragora.connectors.accounting.xero` | Accounting |
| Gusto | `aragora.connectors.accounting.gusto` | Payroll |
| Plaid | `aragora.connectors.accounting.plaid` | Banking |

### E-commerce (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Shopify | `aragora.connectors.ecommerce.shopify` | E-commerce platform |
| Amazon Seller | `aragora.connectors.marketplace.amazon` | Amazon marketplace |
| WooCommerce | `aragora.connectors.ecommerce.woocommerce` | WordPress e-commerce |
| ShipStation | `aragora.connectors.ecommerce.shipstation` | Shipping management |

### Support (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Zendesk | `aragora.connectors.support.zendesk` | Customer support |
| Freshdesk | `aragora.connectors.support.freshdesk` | Help desk |
| Intercom | `aragora.connectors.support.intercom` | Customer messaging |
| Help Scout | `aragora.connectors.support.helpscout` | Help desk |

### Payments (4)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Stripe | `aragora.connectors.payments.stripe` | Payments |
| PayPal | `aragora.connectors.payments.paypal` | Payments |
| Square | `aragora.connectors.payments.square` | Payments & POS |
| Authorize.net | `aragora.connectors.payments.authorize` | Payment gateway |

### Marketplace (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Walmart | `aragora.connectors.marketplace.walmart` | Walmart marketplace |
| Amazon | `aragora.connectors.marketplace.amazon` | Amazon marketplace |

### Calendar (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Google Calendar | `aragora.connectors.calendar.google_calendar` | Google Calendar |
| Outlook Calendar | `aragora.connectors.calendar.outlook_calendar` | Outlook Calendar |

### Marketing (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Marketing | `aragora.connectors.marketing` | Marketing automation |
| Twitter Poster | `aragora.connectors.twitter_poster` | Social posting |

### Low-Code (2)

| Connector | Module | Purpose |
|-----------|--------|---------|
| Airtable | `aragora.connectors.lowcode.airtable` | Database/spreadsheet |
| Knack | `aragora.connectors.lowcode.knack` | Low-code apps |

**See:** [Operational Connectors Guide](connectors/OPERATIONAL_CONNECTORS.md)

---

## Common Patterns

### Authentication

Most connectors support multiple authentication methods:

```python
# API Key
connector = Connector(api_key="xxx")

# OAuth 2.0
connector = Connector(
    client_id="xxx",
    client_secret="xxx",
    refresh_token="xxx",
)

# Service Account
connector = Connector(
    service_account_file="/path/to/credentials.json"
)
```

### Error Handling

```python
from aragora.connectors.exceptions import (
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
)

try:
    result = await connector.fetch()
except ConnectorAuthError:
    # Re-authenticate
    pass
except ConnectorRateLimitError as e:
    # Wait and retry
    await asyncio.sleep(e.retry_after)
except ConnectorTimeoutError:
    # Retry with longer timeout
    pass
```

### Credential Management

```python
from aragora.connectors.credentials import CredentialProvider

# Load from environment
provider = CredentialProvider.from_env()

# Load from vault
provider = CredentialProvider.from_vault(
    vault_url="https://vault.example.com",
    secret_path="aragora/connectors/slack",
)

connector = SlackConnector(credentials=provider)
```

---

## Connector Base Class

All connectors inherit from `BaseConnector`:

```python
from aragora.connectors.base import BaseConnector

class CustomConnector(BaseConnector):
    """Custom connector implementation."""

    async def connect(self) -> None:
        """Establish connection."""
        pass

    async def disconnect(self) -> None:
        """Clean up connection."""
        pass

    async def fetch(self, query: str) -> list[dict]:
        """Fetch data based on query."""
        pass

    async def health_check(self) -> bool:
        """Check connector health."""
        pass
```

---

## Configuration

### Environment Variables

```bash
# Evidence connectors
GITHUB_TOKEN=ghp_xxx
NEWSAPI_KEY=xxx
TWITTER_BEARER_TOKEN=xxx

# Enterprise connectors
SLACK_BOT_TOKEN=xoxb-xxx
SALESFORCE_CLIENT_ID=xxx
SALESFORCE_CLIENT_SECRET=xxx

# Database connectors
POSTGRES_URL=postgresql://user:pass@host:5432/db
MONGODB_URI=mongodb://user:pass@host:27017/db
```

### Connector Registry

```python
from aragora.connectors import ConnectorRegistry

# Register custom connector
registry = ConnectorRegistry()
registry.register("custom", CustomConnector)

# Get connector instance
connector = registry.get("custom", config={...})
```

---

## See Also

- [Evidence Connectors Guide](connectors/EVIDENCE_CONNECTORS.md) - Detailed evidence connector docs
- [Enterprise Connectors Guide](connectors/ENTERPRISE_CONNECTORS.md) - Enterprise integration docs
- [Operational Connectors Guide](connectors/OPERATIONAL_CONNECTORS.md) - Operations connector docs
- [API Reference](API_REFERENCE.md) - REST API for connector management
- [ENVIRONMENT.md](ENVIRONMENT.md) - Environment variable reference

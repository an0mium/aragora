# Data Connectors Reference

Aragora is the **control plane for multi-agent robust decisionmaking across organizational knowledge and channels**. It is **omnivorous by design**—ingesting information from many sources to fuel multi-agent decision making. Evidence connectors ground debates in real-world data from external sources.

**Input Sources**: Documents, APIs, databases, web searches, academic papers, news, social media, financial filings, and more.

> **Full documentation**: See [Evidence System Guide](EVIDENCE.md) for detailed usage examples.

## Quick Reference (Evidence Connectors)

| Connector | Source | API Key | Rate Limit | Best For |
|-----------|--------|---------|------------|----------|
| `LocalDocsConnector` | Local files | No | N/A | Documentation, code |
| `ArXivConnector` | arXiv.org | No | 3/sec | Academic papers |
| `WikipediaConnector` | Wikipedia | No | 200/min | Reference knowledge |
| `HackerNewsConnector` | Hacker News | No | 500/day | Tech discussions |
| `WebConnector` | Web search | No | 10/min | Live web content |
| `GitHubConnector` | GitHub API | Optional | 60/hr (5000 w/token) | Issues, PRs, code |
| `RedditConnector` | Reddit API | Yes | 60/min | Community sentiment |
| `TwitterConnector` | Twitter/X API | Yes | Varies | Real-time discourse |
| `NewsAPIConnector` | NewsAPI | Yes | 1000/day | News articles |
| `SECConnector` | SEC EDGAR | No | 10/sec | Financial filings |
| `SQLConnector` | SQL databases | No | N/A | Structured data |

## Operational Platform Connectors

These connectors power workflow automation and platform APIs (advertising,
analytics, CRM, ecommerce, support, marketing, calendar, payments). They are
distinct from evidence connectors and are used by feature handlers and
workflow templates.

| Category | Subpackage | Platforms (current / planned) | Notes |
|----------|------------|-------------------------------|-------|
| Advertising | `aragora.connectors.advertising` | Google Ads, Meta Ads, LinkedIn Ads, Microsoft Ads, TikTok Ads, X Ads | Campaigns, performance, budgeting |
| Analytics | `aragora.connectors.analytics` | Metabase, Google Analytics 4, Mixpanel, Segment | Dashboards, reports, funnels |
| CRM | `aragora.connectors.crm` | HubSpot / Salesforce, Pipedrive (planned) | Contacts, deals, pipeline |
| Ecommerce | `aragora.connectors.ecommerce` | Shopify, Amazon Seller, WooCommerce, ShipStation / eBay, Magento, TikTok Shop (planned) | Orders, inventory, fulfillment |
| Support | `aragora.connectors.support` | Zendesk, Freshdesk, Intercom, Help Scout | Tickets, triage, responses |
| Marketing | `aragora.connectors.marketing` | Mailchimp, Klaviyo | Campaigns, flows, audiences |
| Calendar | `aragora.connectors.calendar` | Google Calendar, Outlook Calendar | Events, availability |
| Email Sync | `aragora.connectors.email` | Gmail Sync, Outlook Sync | Background sync + webhooks |
| Payments | `aragora.connectors.payments` | Stripe / PayPal, Square, Adyen (planned) | Subscriptions, invoices |

## Environment Variables

```bash
# Required for specific connectors
NEWSAPI_KEY=your_newsapi_key
GITHUB_TOKEN=ghp_your_token
TWITTER_BEARER_TOKEN=your_bearer_token
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

Operational platform connectors typically use OAuth credentials passed to the
connector constructor or managed via the enterprise credential provider stack.
See the connector subpackage for required fields.

## Quick Start

```python
from aragora.connectors import (
    ArXivConnector,
    HackerNewsConnector,
    WikipediaConnector,
)

# Academic papers
arxiv = ArXivConnector(categories=["cs.AI", "cs.LG"])
papers = arxiv.search("transformer attention mechanisms", max_results=10)

# Tech discussions
hn = HackerNewsConnector(min_score=50)
posts = hn.search("Claude AI", max_results=20)

# Reference knowledge
wiki = WikipediaConnector(language="en")
articles = wiki.search("artificial general intelligence")
```

## Connector Details

### LocalDocsConnector
Search local files (Markdown, Python, text).

```python
from aragora.connectors import LocalDocsConnector

connector = LocalDocsConnector(
    root_paths=["./docs", "./src"],
    extensions=[".md", ".py", ".txt"],
)
results = connector.search("authentication flow")
```

### ArXivConnector
Search academic papers on arXiv.org.

```python
from aragora.connectors import ArXivConnector, ARXIV_CATEGORIES

connector = ArXivConnector(
    categories=["cs.AI", "cs.LG", "cs.CL"],
    max_age_days=365,
)
results = connector.search("reinforcement learning from human feedback")

# Available categories: cs.AI, cs.LG, cs.CL, cs.CV, math.OC, stat.ML, etc.
```

### NewsAPIConnector
Fetch news from trusted sources with credibility scoring.

```python
from aragora.connectors import (
    NewsAPIConnector,
    HIGH_CREDIBILITY_SOURCES,
    MEDIUM_CREDIBILITY_SOURCES,
)

connector = NewsAPIConnector(
    api_key="your-key",
    preferred_sources=HIGH_CREDIBILITY_SOURCES,  # Reuters, AP, BBC
)
results = connector.search("AI regulation")
```

### SECConnector
Access SEC EDGAR financial filings.

```python
from aragora.connectors import SECConnector, FORM_TYPES

connector = SECConnector()
filings = connector.search("Apple Inc", form_types=["10-K", "10-Q"])

# Available forms: 10-K, 10-Q, 8-K, DEF 14A, S-1, etc.
```

### SQLConnector
Query SQL databases directly.

```python
from aragora.connectors import SQLConnector

connector = SQLConnector(
    connection_string="postgresql://user:pass@localhost/db"
)
result = connector.query(
    "SELECT * FROM debates WHERE status = %s",
    params=["completed"],
)
```

### GitHubConnector
Fetch GitHub issues, PRs, and discussions.

```python
from aragora.connectors import GitHubConnector

connector = GitHubConnector(
    token="ghp_...",  # Optional, increases rate limits
    repos=["anthropics/claude-code"],
)
issues = connector.search("memory leak", content_type="issues")
```

## Error Handling

```python
from aragora.connectors import (
    ConnectorError,
    ConnectorRateLimitError,
    ConnectorAuthError,
    is_retryable_error,
    get_retry_delay,
)

try:
    results = connector.search("query")
except ConnectorRateLimitError as e:
    delay = get_retry_delay(e)
    print(f"Rate limited. Retry after {delay}s")
except ConnectorAuthError:
    print("Check API credentials")
except ConnectorError as e:
    if is_retryable_error(e):
        # Retry with exponential backoff
        pass
```

## Creating Custom Connectors

```python
from aragora.connectors import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

class MyConnector(BaseConnector):
    source_type = SourceType.CUSTOM
    default_confidence = 0.7

    def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        # Implement search logic
        pass

    def fetch(self, evidence_id: str) -> Evidence | None:
        # Implement fetch by ID
        pass
```

## See Also

- [Evidence System Guide](EVIDENCE.md) - Full connector documentation with examples
- [Pulse System](PULSE.md) - Trending topic integration using connectors
- [API Endpoints](API_ENDPOINTS.md) - REST API for evidence collection

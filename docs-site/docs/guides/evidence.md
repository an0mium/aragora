---
title: Evidence System Guide
description: Evidence System Guide
---

# Evidence System Guide

The Aragora Evidence system provides connectors for grounding debates in real-world data. Each connector integrates with the provenance system for full traceability and reliability scoring.

## Overview

Evidence connectors fetch data from external sources and return `Evidence` objects that include:
- Content and metadata
- Confidence and authority scores
- Provenance tracking for debate traceability

## Available Connectors

| Connector | Source | API Key Required | Use Case |
|-----------|--------|------------------|----------|
| `LocalDocsConnector` | Local files | No | Documentation, code, markdown |
| `GitHubConnector` | GitHub API | Optional | Issues, PRs, discussions |
| `WebConnector` | Web search | No | Live web content |
| `ArXivConnector` | arXiv.org | No | Academic papers |
| `HackerNewsConnector` | Hacker News | No | Tech community discussions |
| `WikipediaConnector` | Wikipedia | No | Encyclopedia articles |
| `RedditConnector` | Reddit API | Optional | Community discussions |
| `TwitterConnector` | Twitter/X API | Yes | Real-time discourse |
| `SQLConnector` | SQL databases | No | PostgreSQL, MySQL, SQLite |
| `NewsAPIConnector` | NewsAPI | Yes | News articles |
| `SECConnector` | SEC EDGAR | No | Financial filings |

## Quick Start

```python
from aragora.connectors import ArXivConnector, Evidence

# Create connector
connector = ArXivConnector()

# Search for evidence
results = connector.search("large language model safety", max_results=5)

for evidence in results:
    print(f"Title: {evidence.title}")
    print(f"Score: {evidence.reliability_score:.2f}")
    print(f"URL: {evidence.url}")
    print("---")
```

## Evidence Object

Every connector returns `Evidence` objects with standardized fields:

```python
@dataclass
class Evidence:
    id: str                    # Unique identifier
    source_type: SourceType    # Type of source (paper, article, etc.)
    source_id: str             # URL, file path, or identifier
    content: str               # Main text content
    title: str                 # Title or headline

    # Metadata
    created_at: str | None     # Publication/creation date
    author: str | None         # Author or source
    url: str | None            # Link to original

    # Reliability scores (0.0 to 1.0)
    confidence: float          # Base confidence in source
    freshness: float           # How recent (decays over time)
    authority: float           # Source authority rating

    metadata: dict             # Additional connector-specific data
```

### Reliability Score

The combined reliability score is computed as:
```
reliability = 0.4 * confidence + 0.3 * freshness + 0.3 * authority
```

## Connector Details

### LocalDocsConnector

Search local documentation, markdown files, and code.

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
    categories=["cs.AI", "cs.LG"],  # Filter by category
    max_age_days=365,               # Only recent papers
)

results = connector.search("transformer architecture")

# Available categories
print(ARXIV_CATEGORIES)  # {'cs.AI': 'Artificial Intelligence', ...}
```

### NewsAPIConnector

Fetch news articles from multiple sources with credibility scoring.

```python
from aragora.connectors import (
    NewsAPIConnector,
    HIGH_CREDIBILITY_SOURCES,
    MEDIUM_CREDIBILITY_SOURCES,
)

connector = NewsAPIConnector(
    api_key="your-newsapi-key",
    preferred_sources=HIGH_CREDIBILITY_SOURCES,  # Reuters, AP, etc.
)

results = connector.search("artificial intelligence regulation")
```

**Credibility tiers:**
- `HIGH_CREDIBILITY_SOURCES`: Major news agencies (Reuters, AP, BBC, etc.)
- `MEDIUM_CREDIBILITY_SOURCES`: Quality publications with editorial standards

### SECConnector

Access SEC EDGAR financial filings.

```python
from aragora.connectors import SECConnector, FORM_TYPES

connector = SECConnector()

# Search by company
results = connector.search("Apple Inc", form_types=["10-K", "10-Q"])

# Fetch specific filing
filing = connector.fetch("0000320193-24-000123")

# Available form types
print(FORM_TYPES)  # {'10-K': 'Annual report', '10-Q': 'Quarterly report', ...}
```

### GitHubConnector

Fetch issues, pull requests, and discussions.

```python
from aragora.connectors import GitHubConnector

connector = GitHubConnector(
    token="ghp_...",  # Optional, increases rate limits
    repos=["anthropics/claude-code", "langchain-ai/langchain"],
)

# Search issues
issues = connector.search("memory leak", content_type="issues")

# Fetch specific issue
issue = connector.fetch("anthropics/claude-code/issues/123")
```

### WebConnector

Search and fetch live web content.

```python
from aragora.connectors import WebConnector

connector = WebConnector(
    user_agent="AragonaBot/1.0",
    timeout=30,
)

results = connector.search("AI safety research 2024")
```

### SQLConnector

Query SQL databases directly.

```python
from aragora.connectors import SQLConnector, SQLQueryResult

connector = SQLConnector(
    connection_string="postgresql://user:pass@localhost/db"
)

# Execute query
result: SQLQueryResult = connector.query(
    "SELECT * FROM debates WHERE topic LIKE %s",
    params=["%AI%"],
)

print(f"Found {result.row_count} rows")
for row in result.rows:
    print(row)
```

### WikipediaConnector

Search Wikipedia articles.

```python
from aragora.connectors import WikipediaConnector

connector = WikipediaConnector(
    language="en",
    max_section_depth=2,  # Include subsections
)

results = connector.search("artificial general intelligence")
```

### HackerNewsConnector

Search Hacker News stories and comments.

```python
from aragora.connectors import HackerNewsConnector

connector = HackerNewsConnector(
    include_comments=True,
    min_score=50,  # Filter by points
)

results = connector.search("Claude AI")
```

### RedditConnector

Search Reddit posts and comments.

```python
from aragora.connectors import RedditConnector

connector = RedditConnector(
    client_id="...",
    client_secret="...",
    subreddits=["MachineLearning", "LocalLLaMA"],
)

results = connector.search("fine-tuning techniques")
```

### TwitterConnector

Search Twitter/X posts.

```python
from aragora.connectors import TwitterConnector

connector = TwitterConnector(
    bearer_token="...",
    max_results=100,
)

results = connector.search("AI safety debate")
```

## Error Handling

All connectors use a standardized exception hierarchy:

```python
from aragora.connectors import (
    ConnectorError,           # Base exception
    ConnectorAuthError,       # Authentication failed
    ConnectorRateLimitError,  # Rate limit exceeded
    ConnectorTimeoutError,    # Request timed out
    ConnectorNetworkError,    # Network connectivity issue
    ConnectorAPIError,        # API returned error
    ConnectorValidationError, # Invalid input
    ConnectorNotFoundError,   # Resource not found
    ConnectorQuotaError,      # Quota exceeded
    ConnectorParseError,      # Failed to parse response
    is_retryable_error,       # Check if error is retryable
    get_retry_delay,          # Get recommended retry delay
)

try:
    results = connector.search("query")
except ConnectorRateLimitError as e:
    delay = get_retry_delay(e)
    print(f"Rate limited. Retry after \{delay\}s")
except ConnectorAuthError:
    print("Check your API credentials")
except ConnectorError as e:
    if is_retryable_error(e):
        # Retry with backoff
        pass
```

## Provenance Integration

Evidence automatically integrates with the provenance system:

```python
from aragora.connectors import ArXivConnector
from aragora.reasoning.provenance import ProvenanceManager

manager = ProvenanceManager()
connector = ArXivConnector(provenance_manager=manager)

# Evidence is automatically recorded
results = connector.search("reinforcement learning")

# View provenance chain
for evidence in results:
    record = manager.get_record(evidence.id)
    print(f"Source: {record.source_type}")
    print(f"Confidence: {record.confidence}")
```

## Custom Connectors

Create custom connectors by extending `BaseConnector`:

```python
from aragora.connectors import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

class MyCustomConnector(BaseConnector):
    """Custom connector for my data source."""

    source_type = SourceType.CUSTOM
    default_confidence = 0.7

    def search(self, query: str, max_results: int = 10) -> list[Evidence]:
        # Implement search logic
        results = self._fetch_from_api(query, max_results)
        return [self._to_evidence(r) for r in results]

    def fetch(self, evidence_id: str) -> Evidence | None:
        # Implement fetch by ID
        data = self._fetch_by_id(evidence_id)
        return self._to_evidence(data) if data else None

    def _to_evidence(self, data: dict) -> Evidence:
        return Evidence(
            id=data["id"],
            source_type=self.source_type,
            source_id=data["url"],
            content=data["text"],
            title=data["title"],
            confidence=self.default_confidence,
            authority=data.get("authority", 0.5),
            url=data["url"],
        )
```

## Environment Variables

Configure connectors via environment variables:

| Variable | Connector | Description |
|----------|-----------|-------------|
| `NEWSAPI_KEY` | NewsAPIConnector | NewsAPI API key |
| `GITHUB_TOKEN` | GitHubConnector | GitHub personal access token |
| `TWITTER_BEARER_TOKEN` | TwitterConnector | Twitter API bearer token |
| `REDDIT_CLIENT_ID` | RedditConnector | Reddit OAuth client ID |
| `REDDIT_CLIENT_SECRET` | RedditConnector | Reddit OAuth client secret |

## Best Practices

1. **Rate Limiting**: Respect API rate limits. Use `get_retry_delay()` for backoff.

2. **Caching**: Enable caching for repeated queries:
   ```python
   connector = ArXivConnector(cache_ttl=3600)  # 1 hour cache
   ```

3. **Filtering**: Use source-specific filters to improve relevance:
   ```python
   connector = NewsAPIConnector(
       preferred_sources=HIGH_CREDIBILITY_SOURCES,
       language="en",
   )
   ```

4. **Error Recovery**: Always handle errors gracefully:
   ```python
   try:
       results = connector.search(query)
   except ConnectorError:
       results = []  # Fallback to empty results
   ```

5. **Provenance**: Always use provenance tracking in production:
   ```python
   connector = ArXivConnector(provenance_manager=manager)
   ```

## Testing

Mock connectors for testing:

```python
from unittest.mock import Mock
from aragora.connectors import Evidence
from aragora.reasoning.provenance import SourceType

def test_with_mock_connector():
    mock_connector = Mock()
    mock_connector.search.return_value = [
        Evidence(
            id="test-1",
            source_type=SourceType.PAPER,
            source_id="https://example.com",
            content="Test content",
            title="Test Paper",
            confidence=0.9,
        )
    ]

    results = mock_connector.search("test query")
    assert len(results) == 1
    assert results[0].confidence == 0.9
```

## REST API Reference

The Evidence API provides HTTP endpoints for managing evidence collection and storage.

### Authentication

All endpoints require authentication via `Authorization: Bearer <token>` header when `ARAGORA_API_TOKEN` is set.

### Rate Limits

| Operation | Limit |
|-----------|-------|
| Read operations (GET) | 60 requests/minute |
| Write operations (POST, DELETE) | 10 requests/minute |

### Endpoints

#### List Evidence

```http
GET /api/evidence
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Maximum results to return |
| `offset` | int | 0 | Pagination offset |
| `source` | string | - | Filter by source name |
| `min_reliability` | float | 0.0 | Minimum reliability score (0.0-1.0) |

**Response:**
```json
{
  "evidence": [...],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

#### Get Evidence by ID

```http
GET /api/evidence/:id
```

**Response:**
```json
{
  "evidence": {
    "id": "evid-abc123",
    "source": "arxiv",
    "title": "Large Language Model Safety",
    "snippet": "...",
    "url": "https://arxiv.org/abs/...",
    "reliability_score": 0.85,
    "metadata": {...}
  }
}
```

**Errors:**
- `404` - Evidence not found

#### Search Evidence

```http
POST /api/evidence/search
```

**Request Body:**
```json
{
  "query": "AI safety techniques",
  "limit": 20,
  "source": "arxiv",
  "min_reliability": 0.5,
  "context": {
    "topic": "AI alignment",
    "keywords": ["safety", "alignment"],
    "preferred_sources": ["arxiv", "nature"],
    "max_age_days": 365
  }
}
```

**Response:**
```json
{
  "query": "AI safety techniques",
  "results": [...],
  "count": 15
}
```

#### Collect Evidence

Collect new evidence from configured connectors.

```http
POST /api/evidence/collect
```

**Request Body:**
```json
{
  "task": "What are the best practices for AI safety?",
  "connectors": ["arxiv", "hackernews"],
  "debate_id": "debate-123",
  "round": 2
}
```

**Response:**
```json
{
  "task": "What are the best practices for AI safety?",
  "keywords": ["AI", "safety", "best practices"],
  "snippets": [...],
  "count": 12,
  "total_searched": 45,
  "average_reliability": 0.78,
  "average_freshness": 0.65,
  "saved_ids": ["evid-1", "evid-2"],
  "debate_id": "debate-123"
}
```

#### Get Debate Evidence

```http
GET /api/evidence/debate/:debate_id
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `round` | int | Filter by debate round (optional) |

**Response:**
```json
{
  "debate_id": "debate-123",
  "round": 2,
  "evidence": [...],
  "count": 8
}
```

#### Associate Evidence with Debate

```http
POST /api/evidence/debate/:debate_id
```

**Request Body:**
```json
{
  "evidence_ids": ["evid-1", "evid-2", "evid-3"],
  "round": 2
}
```

**Response:**
```json
{
  "debate_id": "debate-123",
  "associated": ["evid-1", "evid-2", "evid-3"],
  "count": 3
}
```

#### Delete Evidence

```http
DELETE /api/evidence/:id
```

**Response:**
```json
{
  "deleted": true,
  "evidence_id": "evid-abc123"
}
```

**Errors:**
- `404` - Evidence not found

#### Get Statistics

```http
GET /api/evidence/statistics
```

**Response:**
```json
{
  "statistics": {
    "total_evidence": 1250,
    "sources": {
      "arxiv": 450,
      "hackernews": 320,
      "wikipedia": 280,
      "newsapi": 200
    },
    "average_reliability": 0.72,
    "debates_with_evidence": 45
  }
}
```

### Error Responses

All errors return JSON in this format:

```json
{
  "error": "Evidence not found: evid-invalid",
  "code": "NOT_FOUND"
}
```

| Status | Code | Description |
|--------|------|-------------|
| 400 | VALIDATION_ERROR | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid authentication |
| 404 | NOT_FOUND | Resource not found |
| 429 | RATE_LIMITED | Rate limit exceeded |
| 500 | INTERNAL_ERROR | Server error |

### Python SDK Usage

```python
from aragora.client import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")

# Search evidence
results = client._request("POST", "/api/evidence/search", json={
    "query": "transformer architecture",
    "limit": 10
})

# Collect evidence for a debate
evidence = client._request("POST", "/api/evidence/collect", json={
    "task": "Compare GPT-4 vs Claude 3.5",
    "debate_id": "debate-123",
    "connectors": ["arxiv", "hackernews"]
})
```

---

## Attribution & Reputation System

The attribution module tracks source reputation across debates and provides cross-debate evidence tracking.

### ReputationTier

Source reputation classification:

| Tier | Score Range | Description |
|------|-------------|-------------|
| `AUTHORITATIVE` | >= 0.85 | Highly trusted source |
| `RELIABLE` | >= 0.70 | Generally trustworthy |
| `STANDARD` | >= 0.50 | Default/neutral reputation |
| `UNCERTAIN` | >= 0.30 | Questionable reliability |
| `UNRELIABLE` | < 0.30 | Poor track record |

### VerificationOutcome

Track verification results for evidence:

```python
from aragora.evidence import VerificationOutcome

VerificationOutcome.VERIFIED        # Confirmed accurate
VerificationOutcome.PARTIALLY_VERIFIED  # Some claims verified
VerificationOutcome.UNVERIFIED      # Could not verify (neutral)
VerificationOutcome.CONTESTED       # Disputed by other evidence
VerificationOutcome.REFUTED         # Shown to be incorrect
```

### SourceReputationManager

Track source reputation across debates:

```python
from aragora.evidence import SourceReputationManager, VerificationOutcome

manager = SourceReputationManager()

# Record a verification
manager.record_verification(
    record_id="v-001",
    source_id="arxiv.org",
    debate_id="debate-123",
    outcome=VerificationOutcome.VERIFIED,
    confidence=0.9,
)

# Get reputation
rep = manager.get_reputation("arxiv.org")
print(rep.reputation_score)  # 0-1 score
print(rep.tier)              # ReputationTier.RELIABLE
print(rep.verification_rate) # Fraction verified
print(rep.trend)             # Recent vs overall trend

# Query sources
top_sources = manager.get_top_sources(limit=10)
unreliable = manager.get_unreliable_sources(threshold=0.3)

# Export/import state
state = manager.export_state()
new_manager.import_state(state)
```

### ReputationScorer

Algorithms for computing reputation with time decay:

```python
from aragora.evidence import ReputationScorer

scorer = ReputationScorer(
    decay_half_life=30.0,     # Days for time decay
    recent_window=7.0,        # Days for "recent" window
)

# Compute from history
overall, recent, trend = scorer.compute_score(verifications, current_score=0.5)
```

**Outcome Impact Weights:**
| Outcome | Impact |
|---------|--------|
| `VERIFIED` | +0.15 |
| `PARTIALLY_VERIFIED` | +0.05 |
| `UNVERIFIED` | 0.00 |
| `CONTESTED` | -0.05 |
| `REFUTED` | -0.20 |

### AttributionChain

Track evidence usage across debates:

```python
from aragora.evidence import AttributionChain

chain = AttributionChain()

# Track evidence usage
entry = chain.add_entry(
    evidence_id="ev-001",
    source_id="arxiv.org",
    debate_id="debate-123",
    content="Evidence content...",
)

# Record verification
chain.record_verification(
    evidence_id="ev-001",
    outcome=VerificationOutcome.VERIFIED,
)

# Query chain
evidence_history = chain.get_evidence_chain("ev-001")
source_usage = chain.get_source_chain("arxiv.org")
debate_attributions = chain.get_debate_attributions("debate-123")

# Find reused evidence
reused = chain.find_reused_evidence(min_uses=2)

# Compute debate reliability
metrics = chain.compute_debate_reliability("debate-123")
# {
#     "evidence_count": 5,
#     "avg_reputation": 0.75,
#     "min_reputation": 0.60,
#     "verified_count": 3,
#     "reliability_score": 0.72,
# }
```

---

## Evidence Metadata Enrichment

### MetadataEnricher

Enriches evidence with source classification and provenance:

```python
from aragora.evidence import MetadataEnricher

enricher = MetadataEnricher()

metadata = enricher.enrich(
    content="Research paper content...",
    url="https://arxiv.org/abs/2024.12345",
    existing_metadata={"author": "John Smith", "doi": "10.1234/example"},
)

print(metadata.source_type)      # SourceType.ACADEMIC
print(metadata.confidence)       # 0.75
print(metadata.has_citations)    # True
print(metadata.topics)           # ["Machine Learning", ...]
```

### Source Types

| Type | Domains/Patterns |
|------|------------------|
| `ACADEMIC` | arxiv.org, doi.org, scholar.google.com |
| `DOCUMENTATION` | docs.python.org, readthedocs.io |
| `NEWS` | reuters.com, bbc.com, nytimes.com |
| `SOCIAL` | stackoverflow.com, reddit.com, twitter.com |
| `CODE` | github.com, gitlab.com |

### Confidence Scoring

Confidence is computed from multiple factors:

| Factor | Impact |
|--------|--------|
| Academic source | +0.20 |
| Has DOI | +0.10 |
| Peer reviewed | +0.15 |
| Has citations | +0.10 |
| Recent (&lt;30 days) | +0.10 |
| Short content (&lt;50 words) | -0.10 |

---

## See Also

- [Pulse System Guide](./pulse) - Trending topic integration
- [Provenance Documentation](./features#provenancemanager) - Full provenance system details
- [API Endpoints](../api/endpoints) - Full API endpoint reference
- [REASONING.md](../core-concepts/reasoning) - Belief networks and claims

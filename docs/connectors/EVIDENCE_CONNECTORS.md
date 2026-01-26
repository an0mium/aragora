# Evidence Connectors Guide

Connectors for gathering external evidence to inform debates.

## Overview

Evidence connectors fetch data from external sources to provide context and supporting information for AI debates. Each connector implements the `BaseConnector` interface.

---

## ArXiv Connector

Fetch academic papers from arXiv.org.

### Configuration

```python
from aragora.connectors.arxiv import ArxivConnector

connector = ArxivConnector(
    max_results=10,
    sort_by="relevance",  # or "lastUpdatedDate", "submittedDate"
)
```

### Usage

```python
# Search for papers
papers = await connector.search(
    query="transformer architecture",
    categories=["cs.AI", "cs.LG"],
)

# Get paper by ID
paper = await connector.get_paper("2103.14030")
```

### Response

```python
{
    "id": "2103.14030",
    "title": "An Image is Worth 16x16 Words",
    "authors": ["Alexey Dosovitskiy", ...],
    "summary": "...",
    "published": "2021-03-01",
    "categories": ["cs.CV", "cs.AI"],
    "pdf_url": "https://arxiv.org/pdf/2103.14030",
}
```

---

## GitHub Connector

Fetch repository content, issues, and pull requests.

### Configuration

```python
from aragora.connectors.github import GitHubConnector

connector = GitHubConnector(
    token="ghp_xxx",  # or GITHUB_TOKEN env var
    organization="my-org",  # optional default org
)
```

### Usage

```python
# Fetch repository content
content = await connector.fetch_file(
    repo="owner/repo",
    path="src/main.py",
    ref="main",  # branch or commit
)

# Search issues
issues = await connector.search_issues(
    repo="owner/repo",
    query="bug label:critical",
    state="open",
)

# Get pull request
pr = await connector.get_pull_request(
    repo="owner/repo",
    number=123,
    include_diff=True,
)

# Search code
results = await connector.search_code(
    query="class Arena",
    repo="owner/repo",
)
```

### Environment Variables

```bash
GITHUB_TOKEN=ghp_xxx
GITHUB_ORGANIZATION=my-org
```

---

## HackerNews Connector

Fetch discussions and posts from Hacker News.

### Configuration

```python
from aragora.connectors.hackernews import HackerNewsConnector

connector = HackerNewsConnector(
    min_score=10,  # Minimum post score
    max_age_hours=24,  # Maximum age
)
```

### Usage

```python
# Fetch top stories
stories = await connector.fetch_top_stories(limit=30)

# Search stories
results = await connector.search(
    query="GPT-4 release",
    tags="story",  # or "comment", "ask_hn", "show_hn"
)

# Get story with comments
story = await connector.get_story(
    story_id=12345,
    include_comments=True,
    comment_depth=2,
)
```

---

## Local Docs Connector

Fetch documents from local file system.

### Configuration

```python
from aragora.connectors.local_docs import LocalDocsConnector

connector = LocalDocsConnector(
    base_path="/path/to/docs",
    extensions=[".md", ".txt", ".pdf", ".docx"],
    max_file_size_mb=10,
)
```

### Usage

```python
# Search documents
docs = await connector.search(
    query="architecture decisions",
    path_filter="docs/adr/*.md",
)

# Read document
content = await connector.read(
    path="docs/README.md",
    extract_text=True,  # Parse PDFs, DOCX
)

# List documents
files = await connector.list_documents(
    path="docs/",
    recursive=True,
)
```

---

## NewsAPI Connector

Fetch news articles from NewsAPI.org.

### Configuration

```python
from aragora.connectors.newsapi import NewsAPIConnector

connector = NewsAPIConnector(
    api_key="xxx",  # or NEWSAPI_KEY env var
)
```

### Usage

```python
# Search articles
articles = await connector.search(
    query="artificial intelligence",
    sources=["techcrunch", "wired"],
    from_date="2024-01-01",
    language="en",
)

# Get top headlines
headlines = await connector.get_headlines(
    country="us",
    category="technology",
)
```

### Environment Variables

```bash
NEWSAPI_KEY=xxx
```

---

## Reddit Connector

Fetch discussions from Reddit.

### Configuration

```python
from aragora.connectors.reddit import RedditConnector

connector = RedditConnector(
    client_id="xxx",
    client_secret="xxx",
    user_agent="Aragora/1.0",
)
```

### Usage

```python
# Search subreddit
posts = await connector.search(
    subreddit="MachineLearning",
    query="LLM fine-tuning",
    sort="relevance",  # or "hot", "new", "top"
    time_filter="month",
)

# Get post with comments
post = await connector.get_post(
    post_id="abc123",
    include_comments=True,
    comment_sort="best",
)

# Get subreddit top posts
top = await connector.get_top(
    subreddit="technology",
    limit=25,
    time_filter="week",
)
```

### Environment Variables

```bash
REDDIT_CLIENT_ID=xxx
REDDIT_CLIENT_SECRET=xxx
```

---

## SEC Connector

Fetch SEC filings (10-K, 10-Q, 8-K, etc.).

### Configuration

```python
from aragora.connectors.sec import SECConnector

connector = SECConnector(
    user_agent="Company contact@example.com",
)
```

### Usage

```python
# Get company filings
filings = await connector.get_filings(
    ticker="AAPL",
    form_type="10-K",
    limit=5,
)

# Get specific filing
filing = await connector.get_filing(
    accession_number="0000320193-23-000077",
    sections=["item1", "item7", "item8"],  # Specific sections
)

# Search filings
results = await connector.search(
    query="artificial intelligence",
    form_types=["10-K", "10-Q"],
    date_range=("2023-01-01", "2024-01-01"),
)
```

---

## SQL Connector

Execute queries against SQL databases.

### Configuration

```python
from aragora.connectors.sql import SQLConnector

# PostgreSQL
connector = SQLConnector(
    connection_string="postgresql://user:pass@host:5432/db"
)

# MySQL
connector = SQLConnector(
    connection_string="mysql://user:pass@host:3306/db"
)

# SQLite
connector = SQLConnector(
    connection_string="sqlite:///path/to/db.sqlite"
)
```

### Usage

```python
# Execute query
results = await connector.query(
    sql="SELECT * FROM users WHERE created_at > %s",
    params=["2024-01-01"],
    limit=100,
)

# Get schema
schema = await connector.get_schema(
    tables=["users", "orders"],
    include_sample_data=True,
)
```

### Security Note

The SQL connector implements query sanitization to prevent SQL injection. Only SELECT queries are allowed by default.

---

## Twitter Connector

Fetch tweets and threads from Twitter/X.

### Configuration

```python
from aragora.connectors.twitter import TwitterConnector

connector = TwitterConnector(
    bearer_token="xxx",  # or TWITTER_BEARER_TOKEN env var
)
```

### Usage

```python
# Search tweets
tweets = await connector.search(
    query="AI safety",
    max_results=100,
    start_time="2024-01-01T00:00:00Z",
)

# Get tweet thread
thread = await connector.get_thread(tweet_id="123456789")

# Get user timeline
timeline = await connector.get_user_timeline(
    username="OpenAI",
    max_results=50,
)
```

### Environment Variables

```bash
TWITTER_BEARER_TOKEN=xxx
```

---

## Web Connector

Scrape and parse web pages.

### Configuration

```python
from aragora.connectors.web import WebConnector

connector = WebConnector(
    timeout=30,
    user_agent="Aragora/1.0",
    respect_robots_txt=True,
)
```

### Usage

```python
# Fetch page
page = await connector.fetch(
    url="https://example.com/article",
    extract_text=True,  # Remove HTML
    extract_links=True,
)

# Fetch multiple pages
pages = await connector.fetch_many(
    urls=["https://a.com", "https://b.com"],
    concurrency=5,
)

# Search within page
results = await connector.search_page(
    url="https://docs.example.com",
    query="authentication",
)
```

---

## Whisper Connector

Transcribe audio using OpenAI Whisper.

### Configuration

```python
from aragora.connectors.whisper import WhisperConnector

# Local Whisper
connector = WhisperConnector(
    model="base",  # tiny, base, small, medium, large
    device="cuda",  # or "cpu"
)

# OpenAI API
connector = WhisperConnector(
    api_key="xxx",  # or OPENAI_API_KEY env var
    use_api=True,
)
```

### Usage

```python
# Transcribe file
transcript = await connector.transcribe(
    audio_path="/path/to/audio.mp3",
    language="en",  # optional, auto-detected
)

# Transcribe URL
transcript = await connector.transcribe_url(
    url="https://example.com/podcast.mp3",
)

# Get timestamps
transcript = await connector.transcribe(
    audio_path="/path/to/audio.mp3",
    word_timestamps=True,
)
```

---

## Wikipedia Connector

Fetch Wikipedia articles.

### Configuration

```python
from aragora.connectors.wikipedia import WikipediaConnector

connector = WikipediaConnector(
    language="en",  # Wikipedia language
)
```

### Usage

```python
# Get article
article = await connector.get_article(
    title="Machine learning",
    sections=["History", "Applications"],
)

# Search articles
results = await connector.search(
    query="neural networks",
    limit=10,
)

# Get summary
summary = await connector.get_summary(
    title="Deep learning",
    sentences=3,
)
```

---

## YouTube Connector

Fetch video content and metadata.

### Configuration

```python
from aragora.connectors.youtube_uploader import YouTubeConnector

connector = YouTubeConnector(
    api_key="xxx",  # or YOUTUBE_API_KEY env var
)
```

### Usage

```python
# Get video details
video = await connector.get_video(
    video_id="dQw4w9WgXcQ",
    include_transcript=True,
)

# Search videos
videos = await connector.search(
    query="machine learning tutorial",
    max_results=25,
    order="relevance",  # or "date", "viewCount"
)

# Get transcript
transcript = await connector.get_transcript(
    video_id="dQw4w9WgXcQ",
    language="en",
)
```

---

## Common Integration Patterns

### Batch Evidence Collection

```python
async def gather_evidence(topic: str) -> list:
    """Gather evidence from multiple sources."""
    connectors = [
        ArxivConnector(),
        HackerNewsConnector(),
        NewsAPIConnector(api_key=os.getenv("NEWSAPI_KEY")),
    ]

    tasks = [c.search(query=topic) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    evidence = []
    for result in results:
        if not isinstance(result, Exception):
            evidence.extend(result)

    return evidence
```

### Evidence with Caching

```python
from aragora.connectors.base import CachedConnector

connector = CachedConnector(
    connector=GitHubConnector(token="xxx"),
    cache_ttl=3600,  # 1 hour
    cache_backend="redis",
)
```

### Rate Limiting

```python
from aragora.connectors.base import RateLimitedConnector

connector = RateLimitedConnector(
    connector=TwitterConnector(bearer_token="xxx"),
    requests_per_minute=60,
)
```

---

## See Also

- [Connector Integration Index](../CONNECTOR_INTEGRATION_INDEX.md) - Master connector list
- [Enterprise Connectors Guide](ENTERPRISE_CONNECTORS.md) - Business integrations
- [Operational Connectors Guide](OPERATIONAL_CONNECTORS.md) - Operations tools

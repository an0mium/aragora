# Pulse - Trending Topic Intelligence

Real-time trending topic ingestion from social platforms with quality filtering, source weighting, and automated debate scheduling.

## Overview

Pulse continuously monitors social platforms for trending topics, filters for quality, clusters related discussions, and automatically schedules relevant debates.

## Architecture

```
pulse/
├── __init__.py         # Public API exports
├── ingestor.py         # Platform-specific ingestors (13 sources)
├── quality.py          # Clickbait/spam filtering
├── freshness.py        # Time-based relevance decay
├── weighting.py        # Source credibility scoring
├── consensus.py        # Cross-source trend detection
├── scheduler.py        # Automated debate creation
├── store.py            # Topic persistence (in-memory)
└── postgres_store.py   # Topic persistence (PostgreSQL)
```

## Supported Sources

| Platform | Ingestor | Content Type |
|----------|----------|--------------|
| Hacker News | `HackerNewsIngestor` | Tech news, discussions |
| Reddit | `RedditIngestor` | Subreddit trends |
| Twitter/X | `TwitterIngestor` | Social trends |
| GitHub | `GitHubTrendingIngestor` | Trending repos |
| arXiv | `ArxivIngestor` | Research papers |
| Product Hunt | `ProductHuntIngestor` | Product launches |
| Dev.to | `DevToIngestor` | Developer articles |
| Lobsters | `LobstersIngestor` | Tech curation |
| Substack | `SubstackIngestor` | Newsletter content |
| Google Trends | `GoogleTrendsIngestor` | Search trends |

## Usage

### Basic Ingestion

```python
from aragora.pulse import PulseManager, HackerNewsIngestor, RedditIngestor

# Create manager with ingestors
manager = PulseManager([
    HackerNewsIngestor(),
    RedditIngestor(subreddits=["programming", "machinelearning"]),
])

# Fetch trending topics
topics = await manager.fetch_all()

for topic in topics:
    print(f"{topic.title} (score: {topic.score}, source: {topic.source})")
```

### Quality Filtering

```python
from aragora.pulse import TopicQualityFilter, QualityScore

filter = TopicQualityFilter(
    min_score=0.6,
    block_clickbait=True,
    block_spam=True,
)

filtered_topics = filter.filter(topics)
```

### Freshness Decay

```python
from aragora.pulse import FreshnessCalculator, DecayModel

calculator = FreshnessCalculator(
    decay_model=DecayModel.EXPONENTIAL,
    half_life_hours=24,
)

for topic in topics:
    freshness = calculator.calculate(topic)
    print(f"{topic.title}: {freshness.score:.2f} freshness")
```

### Cross-Source Consensus

```python
from aragora.pulse import CrossSourceConsensus

consensus = CrossSourceConsensus(
    min_sources=2,
    similarity_threshold=0.8,
)

clusters = consensus.cluster(topics)
for cluster in clusters:
    print(f"Trend: {cluster.representative_title}")
    print(f"  Sources: {[t.source for t in cluster.topics]}")
```

### Automated Debate Scheduling

```python
from aragora.pulse import PulseDebateScheduler, SchedulerConfig

scheduler = PulseDebateScheduler(
    config=SchedulerConfig(
        min_quality_score=0.7,
        min_freshness_score=0.5,
        max_debates_per_hour=5,
        require_cross_source=True,
    ),
    arena=arena,  # Your Arena instance
)

# Start automatic scheduling
await scheduler.start()
```

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PULSE_FETCH_INTERVAL` | `300` | Seconds between fetches |
| `PULSE_MIN_QUALITY` | `0.6` | Minimum quality score |
| `PULSE_ENABLE_TWITTER` | `false` | Enable Twitter API |
| `TWITTER_BEARER_TOKEN` | - | Twitter API token |
| `REDDIT_CLIENT_ID` | - | Reddit API client ID |
| `REDDIT_CLIENT_SECRET` | - | Reddit API secret |

### Platform Half-Lives

Default freshness decay half-lives by platform:

| Platform | Half-Life |
|----------|-----------|
| Twitter | 2 hours |
| Hacker News | 12 hours |
| Reddit | 24 hours |
| GitHub | 48 hours |
| arXiv | 168 hours (1 week) |

## Integration with Debates

Pulse integrates with the debate engine through:

1. **Context Gathering**: Injects trending context into debate prompts
2. **Debate Scheduling**: Automatically creates debates for high-quality trends
3. **Knowledge Mound**: Stores trending data as knowledge nodes

```python
# In ArenaConfig
config = ArenaConfig(
    enable_pulse_context=True,
    pulse_sources=["hackernews", "reddit"],
    pulse_freshness_weight=0.3,
)
```

## Circuit Breaker

Each ingestor has built-in circuit breaker protection:

```python
# Automatic failure handling
ingestor = HackerNewsIngestor(
    circuit_breaker_threshold=5,  # Failures before opening
    circuit_breaker_timeout=60,   # Seconds before retry
)
```

## Quality Indicators

### Clickbait Detection
- ALL CAPS titles
- Excessive punctuation (!!!, ???)
- Clickbait phrases ("You won't believe", "This one trick")

### Spam Detection
- Promotional language
- Excessive links
- Low engagement ratios
- Known spam domains

## Persistence

### In-Memory Store
```python
from aragora.pulse import PulseStore

store = PulseStore()
await store.save(topic)
recent = await store.get_recent(hours=24)
```

### PostgreSQL Store
```python
from aragora.pulse import PostgresPulseStore

store = PostgresPulseStore(connection_url)
await store.initialize()
await store.save(topic)
```

## Metrics

Pulse exposes Prometheus metrics:
- `pulse_topics_fetched_total` - Total topics fetched by source
- `pulse_topics_filtered_total` - Topics filtered by reason
- `pulse_fetch_duration_seconds` - Fetch latency by source
- `pulse_circuit_breaker_state` - Circuit breaker states

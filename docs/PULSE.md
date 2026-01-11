# Pulse: Trending Topic Automation

Pulse automatically fetches trending topics from social media platforms and schedules AI debates on the most interesting ones.

## Overview

Pulse consists of three components:

1. **Ingestors** - Fetch trending topics from platforms (HackerNews, Reddit, GitHub, Twitter)
2. **Scheduler** - Selects debate-worthy topics and schedules debates
3. **Store** - Persists scheduled debates and tracks outcomes

## Quick Start

```python
import asyncio
from aragora.pulse import (
    PulseManager,
    HackerNewsIngestor,
    RedditIngestor,
    PulseDebateScheduler,
    ScheduledDebateStore,
    SchedulerConfig,
)
from aragora.debate import Arena, DebateProtocol
from aragora.core import Environment

# 1. Setup ingestors
pulse_manager = PulseManager()
pulse_manager.add_ingestor("hackernews", HackerNewsIngestor())
pulse_manager.add_ingestor("reddit", RedditIngestor(subreddits=["programming", "MachineLearning"]))

# 2. Setup store
store = ScheduledDebateStore("data/scheduled_debates.db")

# 3. Configure scheduler
config = SchedulerConfig(
    poll_interval_seconds=300,  # Check every 5 minutes
    platforms=["hackernews", "reddit"],
    max_debates_per_hour=6,
    min_volume_threshold=100,
    allowed_categories=["tech", "ai", "programming"],
)

scheduler = PulseDebateScheduler(pulse_manager, store, config)

# 4. Define debate creator
async def create_debate(topic: str, rounds: int, threshold: float):
    env = Environment(task=topic)
    protocol = DebateProtocol(rounds=rounds, consensus_threshold=threshold)
    arena = Arena.from_config(env, agents, protocol)
    result = await arena.run()
    return {"debate_id": result.debate_id, "consensus": result.consensus_reached}

scheduler.set_debate_creator(create_debate)

# 5. Start scheduler
await scheduler.start()
```

## Ingestors

### Available Ingestors

| Ingestor | Platform | API Key Required |
|----------|----------|------------------|
| `HackerNewsIngestor` | Hacker News | No |
| `RedditIngestor` | Reddit | Optional |
| `GitHubTrendingIngestor` | GitHub | Optional |
| `TwitterIngestor` | Twitter/X | Yes |

### HackerNews Ingestor

Fetches top stories from HackerNews.

```python
from aragora.pulse import HackerNewsIngestor

ingestor = HackerNewsIngestor(
    rate_limit_delay=1.0,    # Seconds between requests
    max_retries=3,           # Retry failed requests
    base_retry_delay=1.0,    # Initial retry delay
)

topics = await ingestor.fetch_trending(limit=20)
for topic in topics:
    print(f"{topic.topic} - {topic.volume} points")
```

### Reddit Ingestor

Fetches trending posts from specified subreddits.

```python
from aragora.pulse import RedditIngestor

ingestor = RedditIngestor(
    api_key="optional-reddit-api-key",
    subreddits=["programming", "MachineLearning", "technology"],
    rate_limit_delay=2.0,
)

topics = await ingestor.fetch_trending(limit=30)
```

### GitHub Trending

Fetches trending repositories.

```python
from aragora.pulse import GitHubTrendingIngestor

ingestor = GitHubTrendingIngestor(
    api_key="github-token",  # Optional, increases rate limits
    languages=["python", "rust", "go"],
)

topics = await ingestor.fetch_trending(limit=10)
```

### Twitter/X Ingestor

Requires Twitter API credentials.

```python
from aragora.pulse import TwitterIngestor

ingestor = TwitterIngestor(
    api_key="twitter-bearer-token",
    rate_limit_delay=2.0,
)

topics = await ingestor.fetch_trending(limit=50)
```

## Pulse Manager

Coordinates multiple ingestors.

```python
from aragora.pulse import PulseManager, HackerNewsIngestor, RedditIngestor

manager = PulseManager()

# Add ingestors
manager.add_ingestor("hackernews", HackerNewsIngestor())
manager.add_ingestor("reddit", RedditIngestor())

# Fetch from all platforms
all_topics = await manager.fetch_all_trending()

# Fetch from specific platform
hn_topics = await manager.fetch_trending("hackernews")

# Get combined, deduplicated list
combined = await manager.get_combined_trending(limit=50)
```

## Scheduler Configuration

```python
from aragora.pulse import SchedulerConfig

config = SchedulerConfig(
    # Polling
    poll_interval_seconds=300,              # How often to check for topics
    platforms=["hackernews", "reddit"],     # Which platforms to poll

    # Rate limiting
    max_debates_per_hour=6,                 # Maximum debates per hour
    min_interval_between_debates=600,       # Minimum seconds between debates

    # Topic filtering
    min_volume_threshold=100,               # Minimum engagement to consider
    min_controversy_score=0.3,              # Controversy score threshold (0-1)
    allowed_categories=["tech", "ai", "science", "programming"],
    blocked_categories=["politics", "religion"],

    # Deduplication
    dedup_window_hours=24,                  # Don't repeat topics within this window

    # Debate settings
    debate_rounds=3,                        # Rounds per debate
    consensus_threshold=0.7,                # Minimum consensus threshold
)
```

### Category Configuration

Filter topics by category:

```python
config = SchedulerConfig(
    # Only debate these categories
    allowed_categories=["tech", "ai", "science", "programming", "startups"],

    # Never debate these categories
    blocked_categories=["politics", "religion", "celebrity"],
)
```

Common categories detected by ingestors:
- `tech`, `ai`, `programming`, `science`
- `business`, `startups`, `finance`
- `gaming`, `entertainment`
- `politics`, `world`, `sports`

## Topic Scoring

The scheduler uses `TopicSelector` to score topics for debate suitability.

### Scoring Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Category | +0.3 | Allowed category |
| Volume | up to +0.3 | Higher engagement = higher score |
| Controversy | up to +0.4 | Keywords indicating debate-worthiness |
| Boost | up to +0.2 | AI/tech keywords get bonus |

### Controversy Keywords

Topics containing these words score higher:
- `should`, `vs`, `versus`, `debate`
- `controversy`, `opinion`, `disagree`
- `argument`, `battle`, `challenge`
- `question`, `problem`, `issue`

### Boost Keywords

Topics containing these get a bonus:
- `ai`, `artificial intelligence`, `machine learning`
- `ethics`, `future`, `impact`
- `breakthrough`, `revolutionary`

### Manual Topic Scoring

```python
from aragora.pulse import TopicSelector, SchedulerConfig, TrendingTopic

selector = TopicSelector(SchedulerConfig())

topic = TrendingTopic(
    platform="hackernews",
    topic="Should AI models be regulated?",
    volume=500,
    category="ai",
)

score = selector.score_topic(topic)
print(f"Score: {score.score:.2f}")
print(f"Viable: {score.is_viable}")
for reason in score.reasons:
    print(f"  - {reason}")
```

## Scheduler Operations

### Starting and Stopping

```python
from aragora.pulse import PulseDebateScheduler, SchedulerState

# Start the scheduler
await scheduler.start()
assert scheduler.state == SchedulerState.RUNNING

# Pause (stops polling but keeps state)
await scheduler.pause()
assert scheduler.state == SchedulerState.PAUSED

# Resume
await scheduler.resume()

# Stop completely
await scheduler.stop()
assert scheduler.state == SchedulerState.STOPPED
```

### Metrics

```python
metrics = scheduler.get_metrics()
print(f"Polls completed: {metrics.polls_completed}")
print(f"Topics evaluated: {metrics.topics_evaluated}")
print(f"Topics filtered: {metrics.topics_filtered}")
print(f"Debates created: {metrics.debates_created}")
print(f"Debates failed: {metrics.debates_failed}")
print(f"Duplicates skipped: {metrics.duplicates_skipped}")
print(f"Uptime: {metrics.to_dict()['uptime_seconds']:.0f}s")
```

### Manual Trigger

Force an immediate poll:

```python
# Trigger a poll without waiting for interval
await scheduler.trigger_poll()
```

## Persistence

### Scheduled Debate Store

Persists scheduled debates to SQLite.

```python
from aragora.pulse import ScheduledDebateStore, ScheduledDebateRecord

store = ScheduledDebateStore("data/scheduled_debates.db")

# Save a scheduled debate
record = ScheduledDebateRecord(
    record_id="abc123",
    topic_hash="hash",
    topic="Should we regulate AI?",
    platform="hackernews",
    debate_id="debate-001",
    created_at=time.time(),
    status="completed",
    consensus_reached=True,
)
store.save(record)

# Check if topic was already debated
exists = store.topic_exists("hash", window_hours=24)

# Get recent scheduled debates
recent = store.get_recent(limit=10)

# Get stats
stats = store.get_stats()
print(f"Total: {stats['total']}")
print(f"Completed: {stats['completed']}")
print(f"Consensus rate: {stats['consensus_rate']:.0%}")
```

## API Endpoints

When running the Aragora server, these Pulse endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pulse/topics` | GET | Get current trending topics |
| `/api/v1/pulse/topics/{platform}` | GET | Get topics from specific platform |
| `/api/v1/pulse/scheduled` | GET | List scheduled debates |
| `/api/v1/pulse/scheduled/{id}` | GET | Get scheduled debate details |
| `/api/v1/pulse/schedule` | POST | Manually schedule a topic |
| `/api/v1/pulse/metrics` | GET | Get scheduler metrics |
| `/api/v1/pulse/status` | GET | Get scheduler status |

### Example API Usage

```bash
# Get trending topics
curl http://localhost:8080/api/v1/pulse/topics

# Get HackerNews topics only
curl http://localhost:8080/api/v1/pulse/topics/hackernews

# Manually schedule a topic
curl -X POST http://localhost:8080/api/v1/pulse/schedule \
  -H "Content-Type: application/json" \
  -d '{"topic": "Should AI be open source?", "platform": "manual"}'

# Get scheduler status
curl http://localhost:8080/api/v1/pulse/status
```

## Circuit Breaker

Ingestors use circuit breakers to handle API failures gracefully.

```python
from aragora.resilience import CircuitBreaker

# Circuit breaker is built into each ingestor
# It automatically:
# - Opens after 5 consecutive failures
# - Stays open for 60 seconds
# - Half-opens to test recovery
# - Fully opens after successful test

# Check circuit state
print(f"Can proceed: {ingestor.circuit_breaker.can_proceed()}")
print(f"State: {ingestor.circuit_breaker.state}")
```

## Best Practices

1. **Start Conservative**: Begin with `max_debates_per_hour=2` and increase based on capacity.

2. **Category Filtering**: Always configure `blocked_categories` to avoid controversial non-technical topics.

3. **Deduplication Window**: Set `dedup_window_hours=24` or higher to avoid debate fatigue on trending topics.

4. **Monitor Metrics**: Track `topics_filtered` and `duplicates_skipped` to tune thresholds.

5. **Graceful Shutdown**: Always call `await scheduler.stop()` before exiting.

## Related Documentation

- [API Reference](API_REFERENCE.md) - Full REST API documentation
- [WebSocket Events](WEBSOCKET_EVENTS.md) - Real-time streaming
- [Integrations](INTEGRATIONS.md) - Discord/Slack notifications for scheduled debates

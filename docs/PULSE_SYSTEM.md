# Pulse System Documentation

The Pulse system provides trending topic ingestion and automated debate scheduling for Aragora.

## Overview

Pulse connects to external data sources (Hacker News, Reddit, Twitter/X) to identify trending topics suitable for AI debates. The system includes:

1. **Ingestors** - Fetch trending topics from various platforms
2. **Topic Scoring** - Evaluate topics for debate suitability
3. **Scheduler** - Automatically create debates from trending topics
4. **Analytics** - Track debate outcomes by topic/platform

## Architecture

```
                                    ┌─────────────────┐
                                    │  PulseManager   │
                                    │   (singleton)   │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
     ┌────────▼────────┐           ┌────────▼────────┐           ┌────────▼────────┐
     │ HackerNews      │           │  Reddit         │           │  Twitter        │
     │ Ingestor        │           │  Ingestor       │           │  Ingestor       │
     └─────────────────┘           └─────────────────┘           └─────────────────┘
              │                              │                              │
              └──────────────────────────────┼──────────────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ TopicSelector   │
                                    │   (scoring)     │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ PulseDebate     │
                                    │ Scheduler       │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ ScheduledDebate │
                                    │ Store (SQLite)  │
                                    └─────────────────┘
```

## Components

### PulseManager

Central coordinator for trending topic ingestion.

```python
from aragora.pulse.ingestor import PulseManager, HackerNewsIngestor, RedditIngestor

manager = PulseManager()
manager.add_ingestor("hackernews", HackerNewsIngestor())
manager.add_ingestor("reddit", RedditIngestor())

# Fetch trending topics
topics = await manager.get_trending_topics(limit_per_platform=10)
```

### Topic Scoring

Topics are scored based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Category match | 0.3 | Topic in allowed categories |
| Volume | 0.3 | Number of engagements/votes |
| Controversy | 0.4 | Keywords indicating debate potential |
| Boost keywords | 0.2 | AI, ethics, future, etc. |

```python
from aragora.pulse.scheduler import TopicSelector, SchedulerConfig

config = SchedulerConfig(
    allowed_categories=["tech", "ai", "science"],
    blocked_categories=["politics", "religion"],
    min_volume_threshold=100,
    min_controversy_score=0.3,
)

selector = TopicSelector(config)
scored = selector.score_topic(topic)
# scored.score (float), scored.reasons (list), scored.is_viable (bool)
```

### PulseDebateScheduler

Automated debate scheduling with rate limiting and deduplication.

```python
from aragora.pulse.scheduler import PulseDebateScheduler, SchedulerConfig
from aragora.pulse.store import ScheduledDebateStore

store = ScheduledDebateStore("data/scheduled_debates.db")
scheduler = PulseDebateScheduler(pulse_manager, store, config)

# Set up debate creation callback
async def create_debate(topic_text: str, rounds: int, threshold: float):
    arena = Arena(Environment(task=topic_text), agents, protocol)
    result = await arena.run()
    return {
        "debate_id": result.id,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
    }

scheduler.set_debate_creator(create_debate)

# Start scheduler
await scheduler.start()

# Control
await scheduler.pause()
await scheduler.resume()
await scheduler.stop(graceful=True)
```

## Configuration

### SchedulerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poll_interval_seconds` | 300 | How often to check for new topics |
| `platforms` | ["hackernews", "reddit"] | Sources to poll |
| `max_debates_per_hour` | 6 | Rate limit |
| `min_interval_between_debates` | 600 | Minimum seconds between debates |
| `min_volume_threshold` | 100 | Minimum engagement to consider |
| `min_controversy_score` | 0.3 | Minimum controversy score (0-1) |
| `allowed_categories` | ["tech", "ai", "science", "programming"] | Categories to include |
| `blocked_categories` | ["politics", "religion"] | Categories to exclude |
| `dedup_window_hours` | 24 | Skip topics debated within this window |
| `debate_rounds` | 3 | Rounds for created debates |
| `consensus_threshold` | 0.7 | Required confidence for consensus |

### Environment Variables

```bash
# Auto-start scheduler when server starts
PULSE_SCHEDULER_AUTOSTART=true

# Override poll interval (seconds)
PULSE_SCHEDULER_POLL_INTERVAL=300

# Maximum debates per hour
PULSE_SCHEDULER_MAX_PER_HOUR=6
```

## API Endpoints

### Trending Topics

```
GET /api/pulse/trending
GET /api/pulse/trending?limit=20

Response:
{
  "topics": [
    {
      "topic": "New AI model achieves breakthrough",
      "source": "hackernews",
      "score": 0.85,
      "volume": 450,
      "category": "ai"
    }
  ],
  "count": 10,
  "sources": ["hackernews", "reddit"]
}
```

### Suggest Topic

```
GET /api/pulse/suggest
GET /api/pulse/suggest?category=ai

Response:
{
  "topic": "Should AI models require safety testing?",
  "debate_prompt": "Debate: Should AI models require safety testing?",
  "source": "hackernews",
  "category": "ai",
  "volume": 320
}
```

### Scheduler Control

```
GET  /api/pulse/scheduler/status
POST /api/pulse/scheduler/start
POST /api/pulse/scheduler/stop     # Body: {"graceful": true}
POST /api/pulse/scheduler/pause
POST /api/pulse/scheduler/resume
PATCH /api/pulse/scheduler/config  # Body: {"poll_interval_seconds": 600}
GET  /api/pulse/scheduler/history?limit=50&offset=0&platform=hackernews
```

### Status Response

```json
{
  "state": "running",
  "run_id": "run-1736592000-abc123",
  "config": {
    "poll_interval_seconds": 300,
    "max_debates_per_hour": 6,
    ...
  },
  "metrics": {
    "polls_completed": 42,
    "topics_evaluated": 168,
    "topics_filtered": 150,
    "debates_created": 18,
    "debates_failed": 2,
    "duplicates_skipped": 12,
    "uptime_seconds": 3600
  },
  "store_analytics": {
    "total_debates": 156,
    "debates_today": 12,
    "consensus_rate": 0.72
  }
}
```

## Scheduler States

```
          start()
STOPPED ──────────► RUNNING
    ▲                 │  │
    │                 │  │ pause()
    │     stop()      │  ▼
    └─────────────────┼── PAUSED
                      │     │
                      │     │ resume()
                      └─────┘
```

## Deduplication

Topics are deduplicated using a hash of the normalized topic text:

```python
def hash_topic(topic_text: str) -> str:
    """Create a hash for deduplication."""
    normalized = topic_text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
```

The scheduler checks `is_duplicate(topic, dedup_window_hours)` before creating debates.

## Analytics

The scheduler tracks:

- **Polls completed** - Total polling cycles
- **Topics evaluated** - Total topics scored
- **Topics filtered** - Topics that didn't meet criteria
- **Debates created** - Successful debate creations
- **Debates failed** - Failed debate attempts
- **Duplicates skipped** - Topics skipped due to deduplication

The store provides additional analytics:

```python
analytics = store.get_analytics()
# {
#   "total_debates": 156,
#   "debates_today": 12,
#   "debates_this_week": 45,
#   "consensus_rate": 0.72,
#   "by_platform": {"hackernews": 80, "reddit": 76},
#   "by_category": {"ai": 50, "tech": 60, "science": 46}
# }
```

## Best Practices

1. **Start conservative** - Use lower `max_debates_per_hour` initially
2. **Monitor metrics** - Check scheduler status regularly
3. **Tune categories** - Adjust allowed/blocked based on debate quality
4. **Review duplicates** - High duplicate rate may indicate `dedup_window_hours` too short
5. **Watch for failures** - High failure rate may indicate agent issues

## Troubleshooting

### Scheduler won't start

```python
# Check if debate creator is set
if not scheduler._debate_creator:
    raise RuntimeError("No debate creator set")
```

### No topics found

- Check API keys for Twitter ingestor
- Verify network connectivity
- Lower `min_volume_threshold`

### High duplicate rate

- Increase `dedup_window_hours`
- Check if topics are truly different (semantic similarity)

### Debates failing

- Check agent availability
- Review agent error logs
- Verify API quotas

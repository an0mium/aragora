# Pulse System Guide

Pulse is Aragora's trending topic system that automatically ingests topics from social platforms and schedules debates on the most relevant and high-quality trends.

## Overview

The Pulse system provides:
- **Ingestors**: Fetch trending topics from Twitter, HackerNews, Reddit, GitHub
- **Weighting**: Score topics by source credibility
- **Quality**: Filter spam, clickbait, and low-value content
- **Freshness**: Apply time-based relevance decay
- **Consensus**: Detect cross-platform trends
- **Scheduling**: Automatically create debates from top trends

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Platform APIs  │────>│  Ingestors   │────>│  Weighting  │
│  (Twitter, HN)  │     │              │     │  System     │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                    │
                        ┌──────────────┐            │
                        │   Quality    │<───────────┘
                        │   Filter     │
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Freshness   │
                        │  Calculator  │
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Consensus   │
                        │  Detector    │
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Scheduler   │───> Debates
                        └──────────────┘
```

## Quick Start

\`\`\`python
from aragora.pulse import (
    PulseManager,
    SourceWeightingSystem,
    TopicQualityFilter,
    FreshnessCalculator,
    CrossSourceConsensus,
)

# Initialize components
manager = PulseManager()
weighting = SourceWeightingSystem()
quality = TopicQualityFilter()
freshness = FreshnessCalculator()
consensus = CrossSourceConsensus()

# Fetch trending topics
topics = manager.fetch_all_trending()

# Apply weighting
weighted = weighting.weight_topics(topics)

# Filter low quality
quality_scores = quality.filter_topics(topics)

# Check freshness
fresh_scores = freshness.filter_stale(topics, timestamps)

# Detect cross-platform trends
result = consensus.detect_consensus(topics)
print(f"Cross-platform trends: {result.cross_platform_count}")
\`\`\`

## Ingestors

### TrendingTopic

All ingestors return \`TrendingTopic\` objects:

\`\`\`python
@dataclass
class TrendingTopic:
    platform: str        # Source platform
    topic: str           # Topic text/title
    volume: int          # Engagement count
    category: str        # Topic category
    url: str | None      # Link to source
    raw_data: dict       # Platform-specific data
\`\`\`

### PulseManager

Coordinates multiple ingestors:

\`\`\`python
from aragora.pulse import PulseManager

manager = PulseManager(
    enable_twitter=True,
    enable_hackernews=True,
    enable_reddit=True,
    enable_github=False,
)

# Fetch from all enabled platforms
topics = manager.fetch_all_trending()

# Fetch from specific platform
hn_topics = manager.fetch_trending("hackernews")
\`\`\`

### Individual Ingestors

\`\`\`python
from aragora.pulse import (
    HackerNewsIngestor,
    TwitterIngestor,
    RedditIngestor,
    GitHubTrendingIngestor,
)

# HackerNews (no auth required)
hn = HackerNewsIngestor()
topics = hn.fetch_trending(limit=30)

# Twitter (requires API key)
twitter = TwitterIngestor(bearer_token="...")
topics = twitter.fetch_trending()

# Reddit (requires OAuth)
reddit = RedditIngestor(
    client_id="...",
    client_secret="...",
    subreddits=["technology", "programming"],
)
topics = reddit.fetch_trending()

# GitHub Trending
github = GitHubTrendingIngestor()
topics = github.fetch_trending(language="python")
\`\`\`

### Circuit Breaker

Ingestors include circuit breaker protection:

\`\`\`python
from aragora.pulse import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
)

# Automatically opens after failures
if breaker.is_open:
    print("Circuit open, using cached data")
\`\`\`

## Source Weighting

Score topics by platform credibility:

\`\`\`python
from aragora.pulse import (
    SourceWeightingSystem,
    DEFAULT_SOURCE_WEIGHTS,
    SourceWeight,
    WeightedTopic,
)

# View default weights
for platform, weight in DEFAULT_SOURCE_WEIGHTS.items():
    print(f"{platform}: credibility={weight.base_credibility}")

# Create weighting system
system = SourceWeightingSystem()

# Weight a single topic
topic = TrendingTopic("hackernews", "AI Safety Discussion", 500, "tech")
weighted: WeightedTopic = system.weight_topic(topic)
print(f"Weighted score: {weighted.weighted_score}")
print(f"Credibility: {weighted.credibility}")
print(f"Authority: {weighted.authority}")

# Weight multiple topics
topics = [...]
weighted_topics = system.weight_topics(topics)

# Rank by score
ranked = system.rank_by_weighted_score(weighted_topics)

# Filter by minimum credibility
high_quality = system.rank_by_weighted_score(
    weighted_topics,
    min_credibility=0.8
)
\`\`\`

### Custom Weights

\`\`\`python
# Update existing platform weight
system.update_source_weight("hackernews", credibility=0.90)

# Track performance and adapt
system.record_topic_performance("hackernews", outcome_score=0.85)
adaptive = system.get_adaptive_credibility("hackernews")
\`\`\`

### Default Platform Weights

| Platform | Credibility | Authority | Volume Multiplier |
|----------|-------------|-----------|-------------------|
| hackernews | 0.85 | 0.80 | 1.0 |
| github | 0.90 | 0.85 | 0.8 |
| reddit | 0.70 | 0.65 | 1.2 |
| twitter | 0.60 | 0.55 | 1.5 |

## Quality Filtering

Filter out spam, clickbait, and low-value content:

\`\`\`python
from aragora.pulse import (
    TopicQualityFilter,
    QualityScore,
    CLICKBAIT_PATTERNS,
    SPAM_INDICATORS,
)

# Create filter
filter = TopicQualityFilter(
    min_quality_threshold=0.5,
    min_text_length=10,
    max_emoji_ratio=0.1,
)

# Score single topic
score: QualityScore = filter.score_topic(topic)
print(f"Overall: {score.overall_score}")
print(f"Acceptable: {score.is_acceptable}")
print(f"Issues: {score.issues}")
print(f"Signals: {score.signals}")

# Signals include:
# - length: Text length score
# - substance: Meaningful content ratio
# - clickbait: Clickbait pattern detection
# - spam: Spam indicator detection
# - structure: Formatting quality
# - quality_indicators: Positive quality signals
# - blocklist: Custom blocklist matches

# Filter batch
filtered = filter.filter_topics(topics, min_quality=0.6)

# Add custom blocklist
filter.add_to_blocklist(["crypto", "nft", "giveaway"])
\`\`\`

### Quality Signals

\`\`\`python
# View patterns
print(CLICKBAIT_PATTERNS)  # ["won't believe", "doctors hate", ...]
print(SPAM_INDICATORS)     # ["free", "click here", "bit.ly", ...]
\`\`\`

## Freshness Decay

Apply time-based relevance scoring:

\`\`\`python
from aragora.pulse import (
    FreshnessCalculator,
    FreshnessConfig,
    FreshnessScore,
    DecayModel,
    DEFAULT_PLATFORM_HALF_LIVES,
)

# View platform half-lives
print(DEFAULT_PLATFORM_HALF_LIVES)
# {'twitter': 2.0, 'hackernews': 6.0, 'reddit': 12.0, 'github': 24.0}

# Create calculator with default config
calc = FreshnessCalculator()

# Custom configuration
config = FreshnessConfig(
    decay_model=DecayModel.EXPONENTIAL,  # or LINEAR, STEP, LOGARITHMIC
    half_life_hours=6.0,
    max_age_hours=48.0,
    min_freshness=0.05,
)
calc = FreshnessCalculator(config)

# Calculate freshness
import time
now = time.time()
created = now - (6 * 3600)  # 6 hours ago

score: FreshnessScore = calc.calculate_freshness(
    topic,
    created_at=created,
    reference_time=now,
)
print(f"Freshness: {score.freshness}")
print(f"Age (hours): {score.age_hours}")
print(f"Is stale: {score.is_stale}")

# Batch scoring
timestamps = {"topic1": now - 3600, "topic2": now - 7200}
scores = calc.score_topics(topics, timestamps)

# Filter stale topics
fresh_only = calc.filter_stale(topics, timestamps, min_freshness=0.3)
\`\`\`

### Decay Models

| Model | Behavior |
|-------|----------|
| \`EXPONENTIAL\` | Smooth decay, half-life based |
| \`LINEAR\` | Linear decrease to max age |
| \`STEP\` | Binary: fresh until threshold, then stale |
| \`LOGARITHMIC\` | Slow initial decay, accelerates over time |

### Custom Platform Half-Lives

\`\`\`python
calc.set_platform_half_life("hackernews", 4.0)  # Faster decay
calc.set_platform_half_life("github", 48.0)     # Slower decay
\`\`\`

## Cross-Source Consensus

Detect trends appearing across multiple platforms:

\`\`\`python
from aragora.pulse import (
    CrossSourceConsensus,
    ConsensusResult,
    TopicCluster,
)

# Create detector
consensus = CrossSourceConsensus(
    similarity_threshold=0.6,
    min_platforms_for_consensus=2,
    consensus_confidence_boost=0.2,
)

# Detect consensus
result: ConsensusResult = consensus.detect_consensus(topics)

print(f"Cross-platform trends: {result.cross_platform_count}")
print(f"Single-platform topics: {result.single_platform_count}")
print(f"Consensus topics: {len(result.consensus_topics)}")

# View clusters
for cluster in result.clusters:
    print(f"Cluster: {cluster.canonical_topic}")
    print(f"  Platforms: {cluster.platform_count}")
    print(f"  Total volume: {cluster.total_volume}")
    print(f"  Is cross-platform: {cluster.is_cross_platform}")

# Get confidence boosts for consensus topics
for topic in result.consensus_topics:
    boost = result.confidence_boosts.get(topic.topic, 0)
    print(f"{topic.topic}: +{boost:.2f} boost")

# Find related topics
target = topics[0]
related = consensus.find_related_topics(
    target,
    candidates=topics[1:],
    max_results=5,
)
for topic, similarity in related:
    print(f"  {topic.topic}: {similarity:.2f}")
\`\`\`

### Similarity Calculation

\`\`\`python
# Access similarity calculation (for debugging)
sim = consensus._calculate_similarity(
    "OpenAI releases GPT-5",
    "GPT-5 released by OpenAI today",
)
print(f"Similarity: {sim}")  # ~0.7

# Keyword extraction
keywords = consensus._extract_keywords("The quick brown fox jumps")
print(keywords)  # ['quick', 'brown', 'fox', 'jumps']
\`\`\`

## Scheduler

Automatically schedule debates from trending topics.

### Scheduler States

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

### SchedulerConfig Parameters

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

### Topic Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Category match | 0.3 | Topic in allowed categories |
| Volume | 0.3 | Number of engagements/votes |
| Controversy | 0.4 | Keywords indicating debate potential |
| Boost keywords | 0.2 | AI, ethics, future, etc. |

### Scheduler Usage

\`\`\`python
from aragora.pulse import (
    PulseDebateScheduler,
    SchedulerConfig,
    SchedulerState,
    SchedulerMetrics,
    TopicSelector,
    TopicScore,
)

# Configure scheduler
config = SchedulerConfig(
    min_topic_score=0.6,
    max_debates_per_hour=5,
    cooldown_minutes=30,
    diversity_weight=0.3,
)

# Create scheduler
scheduler = PulseDebateScheduler(config)

# Run scheduling cycle
selected = scheduler.select_topics(topics, max_count=3)
for topic_score in selected:
    print(f"Selected: {topic_score.topic.topic}")
    print(f"  Score: {topic_score.score}")
    print(f"  Reason: {topic_score.selection_reason}")

# Get scheduler state
state: SchedulerState = scheduler.get_state()
print(f"Running: {state.is_running}")
print(f"Last run: {state.last_run}")
print(f"Queued topics: {state.queued_count}")

# Get metrics
metrics: SchedulerMetrics = scheduler.get_metrics()
print(f"Total scheduled: {metrics.total_scheduled}")
print(f"Success rate: {metrics.success_rate}")
\`\`\`

### Topic Selection

Custom topic selection logic:

\`\`\`python
from aragora.pulse import TopicSelector

selector = TopicSelector(
    weighting_system=weighting,
    quality_filter=quality,
    freshness_calculator=freshness,
    consensus_detector=consensus,
)

scores = selector.score_topics(topics)
top_topics = selector.select_top(scores, count=5)
\`\`\`

## Persistence

Store scheduled debates:

\`\`\`python
from aragora.pulse import ScheduledDebateStore, ScheduledDebateRecord

store = ScheduledDebateStore(db_path="pulse.db")

# Record scheduled debate
record = ScheduledDebateRecord(
    topic="AI Safety Discussion",
    platform="hackernews",
    scheduled_at=datetime.now(),
    debate_id="debate-123",
    topic_score=0.85,
)
store.save(record)

# Query history
recent = store.get_recent(limit=10)
by_platform = store.get_by_platform("hackernews")
\`\`\`

## Deduplication

Topics are deduplicated using a hash of the normalized topic text:

\`\`\`python
def hash_topic(topic_text: str) -> str:
    """Create a hash for deduplication."""
    normalized = topic_text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
\`\`\`

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

\`\`\`python
analytics = store.get_analytics()
# {
#   "total_debates": 156,
#   "debates_today": 12,
#   "debates_this_week": 45,
#   "consensus_rate": 0.72,
#   "by_platform": {"hackernews": 80, "reddit": 76},
#   "by_category": {"ai": 50, "tech": 60, "science": 46}
# }
\`\`\`

## Best Practices

1. **Start conservative** - Use lower `max_debates_per_hour` initially
2. **Monitor metrics** - Check scheduler status regularly
3. **Tune categories** - Adjust allowed/blocked based on debate quality
4. **Review duplicates** - High duplicate rate may indicate `dedup_window_hours` too short
5. **Watch for failures** - High failure rate may indicate agent issues

## Troubleshooting

### Scheduler won't start

\`\`\`python
# Check if debate creator is set
if not scheduler._debate_creator:
    raise RuntimeError("No debate creator set")
\`\`\`

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

## Full Pipeline Example

\`\`\`python
from aragora.pulse import (
    PulseManager,
    SourceWeightingSystem,
    TopicQualityFilter,
    FreshnessCalculator,
    CrossSourceConsensus,
    PulseDebateScheduler,
    SchedulerConfig,
)
import time

# Initialize all components
manager = PulseManager()
weighting = SourceWeightingSystem()
quality = TopicQualityFilter(min_quality_threshold=0.5)
freshness = FreshnessCalculator()
consensus = CrossSourceConsensus()

config = SchedulerConfig(min_topic_score=0.6)
scheduler = PulseDebateScheduler(config)

# Pipeline
def process_trending():
    now = time.time()

    # 1. Fetch topics
    topics = manager.fetch_all_trending()
    print(f"Fetched {len(topics)} topics")

    # 2. Apply weighting
    weighted = weighting.weight_topics(topics)
    print(f"Weighted {len(weighted)} topics")

    # 3. Filter quality
    quality_filtered = quality.filter_topics(topics, min_quality=0.5)
    quality_topics = [qs.topic for qs in quality_filtered]
    print(f"Quality filter: {len(quality_topics)} passed")

    # 4. Check freshness
    timestamps = {t.topic: t.raw_data.get("created_at", now) for t in quality_topics}
    fresh_topics = freshness.filter_stale(quality_topics, timestamps)
    fresh_list = [fs.topic for fs in fresh_topics]
    print(f"Freshness filter: {len(fresh_list)} passed")

    # 5. Detect consensus
    result = consensus.detect_consensus(fresh_list)
    print(f"Cross-platform trends: {result.cross_platform_count}")

    # 6. Prioritize consensus topics
    final_topics = result.consensus_topics or fresh_list[:10]

    # 7. Schedule debates
    selected = scheduler.select_topics(final_topics, max_count=3)
    for ts in selected:
        print(f"Scheduling debate: {ts.topic.topic}")
        # Create debate...

    return selected

# Run
selected = process_trending()
\`\`\`

## Environment Variables

| Variable | Description |
|----------|-------------|
| \`TWITTER_BEARER_TOKEN\` | Twitter API bearer token |
| \`REDDIT_CLIENT_ID\` | Reddit OAuth client ID |
| \`REDDIT_CLIENT_SECRET\` | Reddit OAuth client secret |
| \`PULSE_DB_PATH\` | Path to Pulse SQLite database |
| \`PULSE_MIN_QUALITY\` | Minimum quality threshold (0.0-1.0) |

## API Endpoints

Pulse integrates with the Aragora API:

\`\`\`
GET  /api/pulse/trending           - Get current trending topics
GET  /api/pulse/scheduled          - Get scheduled debates
POST /api/pulse/schedule           - Manually schedule a topic
GET  /api/pulse/stats              - Get Pulse system statistics
\`\`\`

## Testing

\`\`\`python
from aragora.pulse import TrendingTopic

def test_quality_filter():
    filter = TopicQualityFilter()

    # High quality topic
    good = TrendingTopic(
        "hackernews",
        "New research on transformer architectures",
        500,
        "tech",
    )
    score = filter.score_topic(good)
    assert score.is_acceptable

    # Spam topic
    spam = TrendingTopic(
        "twitter",
        "FREE $1000 GIVEAWAY click here!!!",
        10000,
        "spam",
    )
    score = filter.score_topic(spam)
    assert not score.is_acceptable
\`\`\`

## See Also

- [Evidence System Guide](EVIDENCE.md) - External data integration
- [API Reference](API_REFERENCE.md) - REST API endpoints

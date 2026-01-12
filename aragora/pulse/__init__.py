"""Pulse ingestion module for trending topics and real-time feeds."""

from aragora.pulse.ingestor import (
    CircuitBreaker,
    TrendingTopic,
    TrendingTopicOutcome,
    PulseIngestor,
    TwitterIngestor,
    HackerNewsIngestor,
    RedditIngestor,
    GitHubTrendingIngestor,
    PulseManager,
)
from aragora.pulse.store import (
    ScheduledDebateRecord,
    ScheduledDebateStore,
)
from aragora.pulse.scheduler import (
    PulseDebateScheduler,
    SchedulerConfig,
    SchedulerState,
    SchedulerMetrics,
    TopicSelector,
    TopicScore,
)

__all__ = [
    # Ingestor classes
    "CircuitBreaker",
    "TrendingTopic",
    "TrendingTopicOutcome",
    "PulseIngestor",
    "TwitterIngestor",
    "HackerNewsIngestor",
    "RedditIngestor",
    "GitHubTrendingIngestor",
    "PulseManager",
    # Persistence
    "ScheduledDebateRecord",
    "ScheduledDebateStore",
    # Scheduler
    "PulseDebateScheduler",
    "SchedulerConfig",
    "SchedulerState",
    "SchedulerMetrics",
    "TopicSelector",
    "TopicScore",
]

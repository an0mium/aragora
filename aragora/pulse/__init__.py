"""Pulse ingestion module for trending topics and real-time feeds."""

from aragora.pulse.ingestor import (
    CircuitBreaker,
    GitHubTrendingIngestor,
    HackerNewsIngestor,
    PulseIngestor,
    PulseManager,
    RedditIngestor,
    TrendingTopic,
    TrendingTopicOutcome,
    TwitterIngestor,
)
from aragora.pulse.scheduler import (
    PulseDebateScheduler,
    SchedulerConfig,
    SchedulerMetrics,
    SchedulerState,
    TopicScore,
    TopicSelector,
)
from aragora.pulse.store import (
    ScheduledDebateRecord,
    ScheduledDebateStore,
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

"""
Pulse - Trending topic ingestion and processing for aragora.

Components:
- Ingestors: Fetch trending topics from social platforms
- Weighting: Source credibility scoring
- Quality: Content quality filtering
- Freshness: Time-based relevance decay
- Consensus: Cross-source trend detection
- Scheduler: Automated debate creation from trends
"""

from aragora.pulse.consensus import (
    ConsensusResult,
    CrossSourceConsensus,
    TopicCluster,
)
from aragora.pulse.freshness import (
    DecayModel,
    FreshnessCalculator,
    FreshnessConfig,
    FreshnessScore,
    DEFAULT_PLATFORM_HALF_LIVES,
)
from aragora.pulse.ingestor import (
    ArxivIngestor,
    CircuitBreaker,
    DevToIngestor,
    GitHubTrendingIngestor,
    GoogleTrendsIngestor,
    HackerNewsIngestor,
    LobstersIngestor,
    ProductHuntIngestor,
    PulseIngestor,
    PulseManager,
    RedditIngestor,
    SubstackIngestor,
    TrendingTopic,
    TrendingTopicOutcome,
    TwitterIngestor,
)
from aragora.pulse.quality import (
    QualityScore,
    TopicQualityFilter,
    CLICKBAIT_PATTERNS,
    SPAM_INDICATORS,
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
from aragora.pulse.weighting import (
    DEFAULT_SOURCE_WEIGHTS,
    SourceWeight,
    SourceWeightingSystem,
    WeightedTopic,
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
    "GoogleTrendsIngestor",
    "ArxivIngestor",
    "LobstersIngestor",
    "DevToIngestor",
    "ProductHuntIngestor",
    "SubstackIngestor",
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
    # Weighting
    "SourceWeight",
    "WeightedTopic",
    "SourceWeightingSystem",
    "DEFAULT_SOURCE_WEIGHTS",
    # Quality
    "QualityScore",
    "TopicQualityFilter",
    "CLICKBAIT_PATTERNS",
    "SPAM_INDICATORS",
    # Freshness
    "DecayModel",
    "FreshnessConfig",
    "FreshnessScore",
    "FreshnessCalculator",
    "DEFAULT_PLATFORM_HALF_LIVES",
    # Consensus
    "TopicCluster",
    "ConsensusResult",
    "CrossSourceConsensus",
]

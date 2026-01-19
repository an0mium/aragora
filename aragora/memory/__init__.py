"""
Memory and pattern storage module.

Provides:
- CritiqueStore: SQLite-based storage for debate results and patterns
- SemanticRetriever: Embedding-based similarity search
- Pattern: Dataclass for critique patterns
- ConsensusMemory: Persistent storage of debate outcomes
- DissentRetriever: Retrieval of historical dissenting views
- TierManager: Configurable memory tier management
- Surprise scoring: Unified surprise-based memorization

Tier System:
- MemoryTier (FAST/MEDIUM/SLOW/GLACIAL): Update frequency tiers
- AccessTier (HOT/WARM/COLD/ARCHIVE): Access recency tiers for debate context
"""

from aragora.memory.consensus import (
    ConsensusMemory,
    ConsensusRecord,
    ConsensusStrength,
    DissentRecord,
    DissentRetriever,
    DissentType,
    SimilarDebate,
)
from aragora.memory.cross_debate_rlm import AccessTier
from aragora.memory.embeddings import (
    GeminiEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    SemanticRetriever,
)
from aragora.memory.store import CritiqueStore, Pattern
from aragora.memory.surprise import (
    SurpriseScorer,
    calculate_base_rate,
    calculate_combined_surprise,
    calculate_surprise,
    calculate_surprise_from_db_row,
    update_surprise_ema,
)
from aragora.memory.tier_analytics import (
    MemoryAnalytics,
    MemoryUsageEvent,
    TierAnalyticsTracker,
    TierStats,
)
from aragora.memory.tier_manager import (
    MemoryTier,
    TierConfig,
    TierManager,
    TierTransitionMetrics,
    get_tier_manager,
)

__all__ = [
    "CritiqueStore",
    "Pattern",
    "SemanticRetriever",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    # Consensus Memory
    "ConsensusMemory",
    "ConsensusRecord",
    "ConsensusStrength",
    "DissentRecord",
    "DissentType",
    "DissentRetriever",
    "SimilarDebate",
    # Tier Management
    "TierManager",
    "TierConfig",
    "TierTransitionMetrics",
    "MemoryTier",
    "AccessTier",
    "get_tier_manager",
    # Tier Analytics
    "TierAnalyticsTracker",
    "TierStats",
    "MemoryUsageEvent",
    "MemoryAnalytics",
    # Surprise Scoring
    "SurpriseScorer",
    "calculate_surprise",
    "calculate_base_rate",
    "calculate_combined_surprise",
    "calculate_surprise_from_db_row",
    "update_surprise_ema",
]

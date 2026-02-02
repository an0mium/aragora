"""
Context gathering package for debate research and evidence collection.

This package provides modular components for gathering context from multiple sources
to support debate grounding. The main orchestrator is ContextGatherer, which
coordinates source fetching, content processing, caching, and ranking.

Components:
    - gatherer: Main ContextGatherer orchestrator class
    - sources: Source fetching from web, knowledge mound, threat intel, etc.
    - processors: Content processing, RLM compression, Aragora docs
    - rankers: Relevance ranking and confidence scoring utilities
    - cache: Task-keyed caching with isolation and size limits

Usage:
    from aragora.debate.context import ContextGatherer

    gatherer = ContextGatherer(evidence_store_callback=store_evidence)
    context = await gatherer.gather_all(task="Discuss AI safety")

For backwards compatibility, ContextGatherer can also be imported from:
    from aragora.debate.context_gatherer import ContextGatherer
"""

# Main orchestrator
from .gatherer import (
    ContextGatherer,
    CONTEXT_GATHER_TIMEOUT,
    HAS_KNOWLEDGE_MOUND,
    HAS_THREAT_INTEL,
    THREAT_INTEL_ENABLED,
)

# Cache utilities
from .cache import (
    ContextCache,
    MAX_EVIDENCE_CACHE_SIZE,
    MAX_CONTEXT_CACHE_SIZE,
    MAX_CONTINUUM_CACHE_SIZE,
    MAX_TRENDING_CACHE_SIZE,
)

# Source fetching
from .sources import (
    SourceFetcher,
    CLAUDE_SEARCH_TIMEOUT,
    EVIDENCE_TIMEOUT,
    TRENDING_TIMEOUT,
    KNOWLEDGE_MOUND_TIMEOUT,
    BELIEF_CRUX_TIMEOUT,
    THREAT_INTEL_TIMEOUT,
)

# Content processing
from .processors import (
    ContentProcessor,
    CODEBASE_CONTEXT_TIMEOUT,
    HAS_RLM,
    HAS_OFFICIAL_RLM,
    get_rlm,
    get_compressor,
)

# Ranking utilities
from .rankers import (
    CONFIDENCE_MAP,
    confidence_to_float,
    confidence_label,
    pattern_confidence_label,
    TopicRelevanceDetector,
    ContentRanker,
    KnowledgeItemCategorizer,
)

__all__ = [
    # Main class
    "ContextGatherer",
    # Cache
    "ContextCache",
    "MAX_EVIDENCE_CACHE_SIZE",
    "MAX_CONTEXT_CACHE_SIZE",
    "MAX_CONTINUUM_CACHE_SIZE",
    "MAX_TRENDING_CACHE_SIZE",
    # Source fetching
    "SourceFetcher",
    "CLAUDE_SEARCH_TIMEOUT",
    "EVIDENCE_TIMEOUT",
    "TRENDING_TIMEOUT",
    "KNOWLEDGE_MOUND_TIMEOUT",
    "BELIEF_CRUX_TIMEOUT",
    "THREAT_INTEL_TIMEOUT",
    # Content processing
    "ContentProcessor",
    "CODEBASE_CONTEXT_TIMEOUT",
    "HAS_RLM",
    "HAS_OFFICIAL_RLM",
    "get_rlm",
    "get_compressor",
    # Ranking
    "CONFIDENCE_MAP",
    "confidence_to_float",
    "confidence_label",
    "pattern_confidence_label",
    "TopicRelevanceDetector",
    "ContentRanker",
    "KnowledgeItemCategorizer",
    # Feature flags
    "CONTEXT_GATHER_TIMEOUT",
    "HAS_KNOWLEDGE_MOUND",
    "HAS_THREAT_INTEL",
    "THREAT_INTEL_ENABLED",
]

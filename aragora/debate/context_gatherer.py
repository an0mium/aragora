"""
Backwards compatibility shim for context gathering.

This module provides backwards compatibility for imports like:
    from aragora.debate.context_gatherer import ContextGatherer

The implementation has been decomposed into the aragora.debate.context package.
"""

# Re-export everything from the new context package for backwards compatibility
from aragora.debate.context import (
    # Main class
    ContextGatherer,
    # Cache
    ContextCache,
    MAX_EVIDENCE_CACHE_SIZE,
    MAX_CONTEXT_CACHE_SIZE,
    MAX_CONTINUUM_CACHE_SIZE,
    MAX_TRENDING_CACHE_SIZE,
    # Source fetching
    SourceFetcher,
    CLAUDE_SEARCH_TIMEOUT,
    EVIDENCE_TIMEOUT,
    TRENDING_TIMEOUT,
    KNOWLEDGE_MOUND_TIMEOUT,
    BELIEF_CRUX_TIMEOUT,
    THREAT_INTEL_TIMEOUT,
    # Content processing
    ContentProcessor,
    CODEBASE_CONTEXT_TIMEOUT,
    HAS_RLM,
    HAS_OFFICIAL_RLM,
    get_rlm,
    get_compressor,
    # Ranking
    CONFIDENCE_MAP,
    confidence_to_float,
    confidence_label,
    pattern_confidence_label,
    TopicRelevanceDetector,
    ContentRanker,
    KnowledgeItemCategorizer,
    # Feature flags
    CONTEXT_GATHER_TIMEOUT,
    HAS_KNOWLEDGE_MOUND,
    HAS_THREAT_INTEL,
    THREAT_INTEL_ENABLED,
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

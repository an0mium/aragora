"""
Context gathering for debate research and evidence collection.

Extracts research-related context gathering from the Arena class
to improve maintainability and allow independent testing.

Timeouts:
    CONTEXT_GATHER_TIMEOUT: Overall timeout for gather_all (default: 10s)
    EVIDENCE_TIMEOUT: Timeout for evidence collection (default: 5s)
    TRENDING_TIMEOUT: Timeout for trending topics (default: 3s)

RLM Integration:
    When enable_rlm_compression is True and a compressor is available,
    large documents are hierarchically compressed instead of truncated.
    This preserves semantic content while fitting within token budgets.

Package Structure:
    - constants.py: Configuration, timeouts, feature flags
    - sources.py: Context source gathering methods (web, evidence, trending, etc.)
    - compression.py: RLM compression logic
    - memory.py: ContinuumMemory context retrieval
    - gatherer.py: Main ContextGatherer class
"""

from .gatherer import ContextGatherer
from .constants import (
    # Timeouts
    CONTEXT_GATHER_TIMEOUT,
    CLAUDE_SEARCH_TIMEOUT,
    EVIDENCE_TIMEOUT,
    TRENDING_TIMEOUT,
    KNOWLEDGE_MOUND_TIMEOUT,
    BELIEF_CRUX_TIMEOUT,
    THREAT_INTEL_TIMEOUT,
    CODEBASE_CONTEXT_TIMEOUT,
    # Cache limits
    MAX_EVIDENCE_CACHE_SIZE,
    MAX_CONTEXT_CACHE_SIZE,
    MAX_CONTINUUM_CACHE_SIZE,
    MAX_TRENDING_CACHE_SIZE,
    # Feature flags
    HAS_RLM,
    HAS_OFFICIAL_RLM,
    HAS_KNOWLEDGE_MOUND,
    HAS_THREAT_INTEL,
    DISABLE_TRENDING,
    # Factory functions
    get_rlm,
    get_compressor,
    # Keywords
    ARAGORA_KEYWORDS,
)

__all__ = [
    # Main class
    "ContextGatherer",
    # Timeouts
    "CONTEXT_GATHER_TIMEOUT",
    "CLAUDE_SEARCH_TIMEOUT",
    "EVIDENCE_TIMEOUT",
    "TRENDING_TIMEOUT",
    "KNOWLEDGE_MOUND_TIMEOUT",
    "BELIEF_CRUX_TIMEOUT",
    "THREAT_INTEL_TIMEOUT",
    "CODEBASE_CONTEXT_TIMEOUT",
    # Cache limits
    "MAX_EVIDENCE_CACHE_SIZE",
    "MAX_CONTEXT_CACHE_SIZE",
    "MAX_CONTINUUM_CACHE_SIZE",
    "MAX_TRENDING_CACHE_SIZE",
    # Feature flags
    "HAS_RLM",
    "HAS_OFFICIAL_RLM",
    "HAS_KNOWLEDGE_MOUND",
    "HAS_THREAT_INTEL",
    "DISABLE_TRENDING",
    # Factory functions
    "get_rlm",
    "get_compressor",
    # Keywords
    "ARAGORA_KEYWORDS",
]

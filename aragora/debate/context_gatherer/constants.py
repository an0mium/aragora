"""
Constants and configuration for context gathering.

This module centralizes all configurable timeouts, cache limits,
and feature flags for the context gathering system.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# RLM Integration
# =============================================================================

# Define fallback values before imports to avoid redefinition errors
_get_rlm: Optional[Callable[[], Any]] = None
_get_compressor: Optional[Callable[[], Any]] = None
_KnowledgeMound: Optional[type] = None

# Check for RLM availability (use factory for consistent initialization)
try:
    from aragora.rlm import get_rlm as _imported_get_rlm
    from aragora.rlm import get_compressor as _imported_get_compressor
    from aragora.rlm import HAS_OFFICIAL_RLM

    HAS_RLM = True
    _get_rlm = _imported_get_rlm
    _get_compressor = _imported_get_compressor
except ImportError:
    HAS_RLM = False
    HAS_OFFICIAL_RLM = False

# Alias for cleaner usage
get_rlm = _get_rlm
get_compressor = _get_compressor

# =============================================================================
# Knowledge Mound Integration
# =============================================================================

# Check for Knowledge Mound availability
try:
    from aragora.knowledge.mound import KnowledgeMound as _ImportedKnowledgeMound

    HAS_KNOWLEDGE_MOUND = True
    _KnowledgeMound = _ImportedKnowledgeMound
except ImportError:
    HAS_KNOWLEDGE_MOUND = False

# Alias for cleaner usage
KnowledgeMound = _KnowledgeMound

# =============================================================================
# Threat Intelligence Integration
# =============================================================================

_ThreatIntelEnrichment: Optional[type] = None

try:
    from aragora.security.threat_intel_enrichment import (
        ThreatIntelEnrichment as _ImportedThreatIntelEnrichment,
        ENRICHMENT_ENABLED as THREAT_INTEL_ENABLED,
    )

    HAS_THREAT_INTEL = True
    _ThreatIntelEnrichment = _ImportedThreatIntelEnrichment
except ImportError:
    HAS_THREAT_INTEL = False
    THREAT_INTEL_ENABLED = False

# Alias for cleaner usage
ThreatIntelEnrichment = _ThreatIntelEnrichment

# =============================================================================
# Configurable Timeouts (in seconds)
# =============================================================================

# Overall timeout for gather_all
CONTEXT_GATHER_TIMEOUT = float(os.getenv("ARAGORA_CONTEXT_TIMEOUT", "150.0"))

# Claude web search timeout (increased to allow completion)
CLAUDE_SEARCH_TIMEOUT = float(os.getenv("ARAGORA_CLAUDE_SEARCH_TIMEOUT", "120.0"))

# Evidence collection timeout
EVIDENCE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_TIMEOUT", "30.0"))

# Trending topics timeout
TRENDING_TIMEOUT = float(os.getenv("ARAGORA_TRENDING_TIMEOUT", "5.0"))

# Knowledge Mound query timeout
KNOWLEDGE_MOUND_TIMEOUT = float(os.getenv("ARAGORA_KNOWLEDGE_MOUND_TIMEOUT", "10.0"))

# Belief crux analysis timeout
BELIEF_CRUX_TIMEOUT = float(os.getenv("ARAGORA_BELIEF_CRUX_TIMEOUT", "5.0"))

# Threat intelligence enrichment timeout
THREAT_INTEL_TIMEOUT = float(os.getenv("ARAGORA_THREAT_INTEL_TIMEOUT", "10.0"))

# Document store context timeout
DOCUMENT_STORE_TIMEOUT = float(os.getenv("ARAGORA_DOCUMENT_STORE_TIMEOUT", "5.0"))

# Evidence store context timeout
EVIDENCE_STORE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_STORE_TIMEOUT", "5.0"))

# Codebase context building timeout
CODEBASE_CONTEXT_TIMEOUT = float(os.getenv("ARAGORA_CODEBASE_CONTEXT_TIMEOUT", "60.0"))

# =============================================================================
# Cache Size Limits
# =============================================================================

# Maximum entries in evidence cache
MAX_EVIDENCE_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_EVIDENCE_CACHE", "100"))

# Maximum entries in context cache
MAX_CONTEXT_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTEXT_CACHE", "100"))

# Maximum entries in continuum memory cache
MAX_CONTINUUM_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTINUUM_CACHE", "100"))

# Maximum trending topics to cache
MAX_TRENDING_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_TRENDING_CACHE", "50"))

# =============================================================================
# Feature Flag Environment Variables
# =============================================================================

# Check if trending context is disabled via environment
DISABLE_TRENDING = os.getenv("ARAGORA_DISABLE_TRENDING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# Check if codebase context is enabled
def get_use_codebase() -> bool:
    """Determine if codebase context should be used."""
    use_env = os.getenv("ARAGORA_CONTEXT_USE_CODEBASE") or os.getenv("NOMIC_CONTEXT_USE_CODEBASE")
    if use_env is None:
        return True
    return use_env.strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# Aragora Topic Detection Keywords
# =============================================================================

ARAGORA_KEYWORDS = [
    "aragora",
    "multi-agent debate",
    "decision stress-test",
    "ai red team",
    "adversarial validation",
    "gauntlet",
    "nomic loop",
    "debate framework",
]

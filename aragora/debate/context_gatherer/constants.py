"""
Constants and configuration for context gathering.

Defines timeouts, cache sizes, and feature availability flags.
"""

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    pass

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

# Check for Knowledge Mound availability
try:
    from aragora.knowledge.mound import KnowledgeMound as _ImportedKnowledgeMound

    HAS_KNOWLEDGE_MOUND = True
    _KnowledgeMound = _ImportedKnowledgeMound
except ImportError:
    HAS_KNOWLEDGE_MOUND = False

# Alias for cleaner usage
KnowledgeMound = _KnowledgeMound

# Check for Threat Intelligence Enrichment availability
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

# Configurable timeouts (in seconds)
# Increased timeouts to allow Claude web search to complete
CONTEXT_GATHER_TIMEOUT = float(os.getenv("ARAGORA_CONTEXT_TIMEOUT", "150.0"))
CLAUDE_SEARCH_TIMEOUT = float(os.getenv("ARAGORA_CLAUDE_SEARCH_TIMEOUT", "120.0"))
EVIDENCE_TIMEOUT = float(os.getenv("ARAGORA_EVIDENCE_TIMEOUT", "30.0"))
TRENDING_TIMEOUT = float(os.getenv("ARAGORA_TRENDING_TIMEOUT", "5.0"))
KNOWLEDGE_MOUND_TIMEOUT = float(os.getenv("ARAGORA_KNOWLEDGE_MOUND_TIMEOUT", "10.0"))
BELIEF_CRUX_TIMEOUT = float(os.getenv("ARAGORA_BELIEF_CRUX_TIMEOUT", "5.0"))
THREAT_INTEL_TIMEOUT = float(os.getenv("ARAGORA_THREAT_INTEL_TIMEOUT", "10.0"))
CODEBASE_CONTEXT_TIMEOUT = float(os.getenv("ARAGORA_CODEBASE_CONTEXT_TIMEOUT", "60.0"))

# Cache size limits to prevent unbounded memory growth
# These can be configured via environment variables for different deployment scenarios
MAX_EVIDENCE_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_EVIDENCE_CACHE", "100"))
MAX_CONTEXT_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTEXT_CACHE", "100"))
MAX_CONTINUUM_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTINUUM_CACHE", "100"))
MAX_TRENDING_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_TRENDING_CACHE", "50"))

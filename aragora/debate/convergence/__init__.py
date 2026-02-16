"""
Semantic convergence detection for multi-agent debates.

Detects when agents' positions have converged, allowing early termination
of debates when further rounds would provide diminishing returns.

Uses a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)

Inspired by ai-counsel's convergence detection system.

Module Structure
----------------
This module has been split into submodules for maintainability:

- `aragora.debate.cache.embeddings_lru` - EmbeddingCache for text embeddings
- `aragora.debate.similarity.backends` - Similarity computation backends
- `aragora.debate.convergence.cache` - Pairwise similarity cache and management
- `aragora.debate.convergence.metrics` - Convergence result and metric dataclasses
- `aragora.debate.convergence.analyzer` - Advanced multi-metric convergence analyzer
- `aragora.debate.convergence.detector` - Main convergence detection

Performance optimizations:
- LRU cache for pairwise similarity results (256 pairs per backend)
- Session-scoped similarity cache for debate-specific computations
- Batch similarity computation for vectorizable backends
- Early termination in within-round convergence checks
"""

from __future__ import annotations

# Re-export cache utilities from embeddings_lru
from aragora.debate.cache.embeddings_lru import (
    EmbeddingCache,
    cleanup_embedding_cache,
    get_embedding_cache,
    get_scoped_embedding_cache,
    reset_embedding_cache,
)

# Re-export similarity backends
from aragora.debate.similarity.backends import (
    _ENV_CONVERGENCE_BACKEND,
    JaccardBackend,
    SentenceTransformerBackend,
    SimilarityBackend,
    TFIDFBackend,
    _normalize_backend_name,
    get_similarity_backend,
)

# Pairwise similarity cache and management
from aragora.debate.convergence.cache import (
    CACHE_MANAGER_TTL_SECONDS,
    MAX_SIMILARITY_CACHES,
    PERIODIC_CLEANUP_INTERVAL_SECONDS,
    CachedSimilarity,
    PairwiseSimilarityCache,
    _PeriodicCacheCleanup,  # noqa: F401 - exported for test access
    _periodic_cleanup,  # noqa: F401 - exported for test access
    _similarity_cache_lock,  # noqa: F401 - exported for test access
    _similarity_cache_manager,  # noqa: F401 - exported for test access
    _similarity_cache_timestamps,  # noqa: F401 - exported for test access
    cleanup_similarity_cache,
    cleanup_stale_caches,
    cleanup_stale_similarity_caches,
    evict_expired_cache_entries,
    get_cache_manager_stats,
    get_pairwise_similarity_cache,
    get_periodic_cleanup_stats,
    stop_periodic_cleanup,
)

# Convergence result and metrics
from aragora.debate.convergence.metrics import (
    AdvancedConvergenceMetrics,
    ArgumentDiversityMetric,
    ConvergenceResult,
    EvidenceConvergenceMetric,
    StanceVolatilityMetric,
)

# Advanced convergence analyzer
from aragora.debate.convergence.analyzer import AdvancedConvergenceAnalyzer

# Convergence detector
from aragora.debate.convergence.detector import ConvergenceDetector

# Convergence history store
from aragora.debate.convergence.history import (
    ConvergenceHistoryStore,
    get_convergence_history_store,
    init_convergence_history_store,
    set_convergence_history_store,
)

__all__ = [
    # Cache (re-exported)
    "EmbeddingCache",
    "cleanup_embedding_cache",
    "get_embedding_cache",
    "get_scoped_embedding_cache",
    "reset_embedding_cache",
    # Pairwise similarity cache
    "CachedSimilarity",
    "PairwiseSimilarityCache",
    "get_pairwise_similarity_cache",
    "cleanup_similarity_cache",
    "cleanup_stale_similarity_caches",
    "evict_expired_cache_entries",
    "cleanup_stale_caches",
    "get_cache_manager_stats",
    "MAX_SIMILARITY_CACHES",
    "CACHE_MANAGER_TTL_SECONDS",
    "PERIODIC_CLEANUP_INTERVAL_SECONDS",
    # Periodic cleanup management
    "get_periodic_cleanup_stats",
    "stop_periodic_cleanup",
    # Backends (re-exported)
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
    "_ENV_CONVERGENCE_BACKEND",
    "_normalize_backend_name",
    # Convergence
    "ConvergenceResult",
    "ArgumentDiversityMetric",
    "EvidenceConvergenceMetric",
    "StanceVolatilityMetric",
    "AdvancedConvergenceMetrics",
    "AdvancedConvergenceAnalyzer",
    "ConvergenceDetector",
    # Convergence history
    "ConvergenceHistoryStore",
    "get_convergence_history_store",
    "set_convergence_history_store",
    "init_convergence_history_store",
]

"""
Document indexing: vector storage and hybrid search.

Integrates with Weaviate for enterprise-grade vector DB
with BM25 + vector hybrid search.
"""

# Weaviate store requires weaviate (optional dependency)
from typing import Any

try:
    from aragora.documents.indexing.weaviate_store import (
        WeaviateStore,
        WeaviateConfig,
        SearchResult,
        get_weaviate_store,
        WEAVIATE_AVAILABLE,
    )
except ImportError:
    WeaviateStore: Any = None
    WeaviateConfig: Any = None
    SearchResult: Any = None
    get_weaviate_store: Any = None
    WEAVIATE_AVAILABLE = False
from aragora.documents.indexing.hybrid_search import (
    HybridSearcher,
    HybridSearchConfig,
    HybridResult,
    SimpleEmbedder,
    EmbeddingProvider,
    create_hybrid_searcher,
)

__all__ = [
    # Weaviate store
    "WeaviateStore",
    "WeaviateConfig",
    "SearchResult",
    "get_weaviate_store",
    "WEAVIATE_AVAILABLE",
    # Hybrid search
    "HybridSearcher",
    "HybridSearchConfig",
    "HybridResult",
    "SimpleEmbedder",
    "EmbeddingProvider",
    "create_hybrid_searcher",
]

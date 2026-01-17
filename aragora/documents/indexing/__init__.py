"""
Document indexing: vector storage and hybrid search.

Integrates with Weaviate for enterprise-grade vector DB
with BM25 + vector hybrid search.
"""

from aragora.documents.indexing.weaviate_store import (
    WeaviateStore,
    WeaviateConfig,
    SearchResult,
    get_weaviate_store,
    WEAVIATE_AVAILABLE,
)
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

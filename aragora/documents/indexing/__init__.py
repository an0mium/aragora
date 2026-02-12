"""
Document indexing: vector storage and hybrid search.

Integrates with Weaviate for enterprise-grade vector DB
with BM25 + vector hybrid search.
"""

# Weaviate store requires weaviate (optional dependency)
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

if TYPE_CHECKING:
    from aragora.documents.indexing.weaviate_store import (
        WeaviateStore as _WeaviateStore,
        WeaviateConfig as _WeaviateConfig,
        SearchResult as _SearchResult,
    )

WeaviateStore: type["_WeaviateStore"] | None
WeaviateConfig: type["_WeaviateConfig"] | None
SearchResult: type["_SearchResult"] | None
get_weaviate_store: Callable[..., Any] | None
WEAVIATE_AVAILABLE: bool

try:
    from aragora.documents.indexing.weaviate_store import (
        WeaviateStore,
        WeaviateConfig,
        SearchResult,
        get_weaviate_store,
        WEAVIATE_AVAILABLE,
    )
except ImportError:
    WeaviateStore = None
    WeaviateConfig = None
    SearchResult = None
    get_weaviate_store = None
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

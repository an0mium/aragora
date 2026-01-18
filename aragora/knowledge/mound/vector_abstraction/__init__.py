"""
Multi-backend vector store abstraction for Knowledge Mound.

Provides a unified interface across Weaviate, Qdrant, Chroma, and in-memory
vector stores. This enables the Knowledge Mound to work with different
backends based on deployment requirements.

Usage:
    from aragora.knowledge.mound.vector_abstraction import (
        VectorStoreFactory,
        BaseVectorStore,
        VectorSearchResult,
        VectorStoreConfig,
        VectorBackend,
    )

    # Create store from environment
    store = VectorStoreFactory.from_env()
    await store.connect()

    # Or create specific backend
    config = VectorStoreConfig(backend=VectorBackend.QDRANT, url="http://localhost:6333")
    store = VectorStoreFactory.create(config)
"""

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)
from aragora.knowledge.mound.vector_abstraction.factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "VectorBackend",
    "VectorSearchResult",
    "VectorStoreConfig",
    "VectorStoreFactory",
]

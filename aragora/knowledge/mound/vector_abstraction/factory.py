"""
Factory for creating vector store instances.

Provides a central point for vector store creation with automatic
backend selection based on configuration or environment variables.
"""

from __future__ import annotations

import logging
from typing import Type

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    Maintains a registry of backend implementations and provides
    factory methods for creating stores from config or environment.

    Usage:
        # Register a custom backend
        VectorStoreFactory.register(VectorBackend.CUSTOM, CustomVectorStore)

        # Create from config
        config = VectorStoreConfig(backend=VectorBackend.QDRANT)
        store = VectorStoreFactory.create(config)

        # Create from environment
        store = VectorStoreFactory.from_env()
    """

    _registry: dict[VectorBackend, Type[BaseVectorStore]] = {}

    @classmethod
    def register(
        cls,
        backend: VectorBackend,
        store_class: Type[BaseVectorStore],
    ) -> None:
        """
        Register a vector store implementation.

        Args:
            backend: Backend type identifier
            store_class: Store class implementing BaseVectorStore
        """
        cls._registry[backend] = store_class
        logger.debug(f"Registered vector store: {backend.value} -> {store_class.__name__}")

    @classmethod
    def unregister(cls, backend: VectorBackend) -> bool:
        """
        Unregister a vector store implementation.

        Args:
            backend: Backend type to unregister

        Returns:
            True if was registered, False otherwise
        """
        if backend in cls._registry:
            del cls._registry[backend]
            return True
        return False

    @classmethod
    def create(cls, config: VectorStoreConfig) -> BaseVectorStore:
        """
        Create a vector store instance from config.

        Args:
            config: Vector store configuration

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If backend is not registered
        """
        store_class = cls._registry.get(config.backend)
        if not store_class:
            available = [b.value for b in cls._registry.keys()]
            raise ValueError(
                f"Unknown backend: {config.backend.value}. "
                f"Available backends: {available}"
            )
        return store_class(config)

    @classmethod
    def from_env(cls) -> BaseVectorStore:
        """
        Create vector store from environment variables.

        Environment variables:
            VECTOR_BACKEND: Backend type (weaviate, qdrant, chroma, memory)
            VECTOR_STORE_URL: Backend URL
            VECTOR_STORE_API_KEY: API key for authentication
            VECTOR_COLLECTION: Collection name
            EMBEDDING_DIMENSIONS: Vector dimensions
            DISTANCE_METRIC: Distance metric (cosine, euclidean, dot_product)

        Returns:
            Configured vector store instance
        """
        config = VectorStoreConfig.from_env()
        return cls.create(config)

    @classmethod
    def list_backends(cls) -> list[VectorBackend]:
        """
        List registered backends.

        Returns:
            List of available backend types
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, backend: VectorBackend) -> bool:
        """
        Check if a backend is registered.

        Args:
            backend: Backend type to check

        Returns:
            True if registered
        """
        return backend in cls._registry

    @classmethod
    def get_store_class(cls, backend: VectorBackend) -> Type[BaseVectorStore] | None:
        """
        Get the store class for a backend.

        Args:
            backend: Backend type

        Returns:
            Store class or None if not registered
        """
        return cls._registry.get(backend)


def _register_default_backends() -> None:
    """Register default vector store implementations."""
    # Import and register backends
    # Each backend module registers itself on import

    # Always register in-memory (no dependencies)
    from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore

    VectorStoreFactory.register(VectorBackend.MEMORY, InMemoryVectorStore)

    # Try to register Weaviate
    try:
        from aragora.knowledge.mound.vector_abstraction.weaviate import (
            WeaviateVectorStore,
        )

        VectorStoreFactory.register(VectorBackend.WEAVIATE, WeaviateVectorStore)
    except ImportError:
        logger.debug("Weaviate backend not available (weaviate-client not installed)")

    # Try to register Qdrant
    try:
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        VectorStoreFactory.register(VectorBackend.QDRANT, QdrantVectorStore)
    except ImportError:
        logger.debug("Qdrant backend not available (qdrant-client not installed)")

    # Try to register Chroma
    try:
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        VectorStoreFactory.register(VectorBackend.CHROMA, ChromaVectorStore)
    except ImportError:
        logger.debug("Chroma backend not available (chromadb not installed)")


# Register defaults on module load
_register_default_backends()

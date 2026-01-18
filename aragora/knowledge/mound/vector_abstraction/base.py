"""
Base classes for vector store abstraction.

Defines the abstract interface that all vector store backends must implement,
enabling the Knowledge Mound to work with Weaviate, Qdrant, Chroma, or in-memory
stores interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence
import logging

logger = logging.getLogger(__name__)


class VectorBackend(str, Enum):
    """Supported vector store backends."""

    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    MEMORY = "memory"  # In-memory for testing


@dataclass
class VectorSearchResult:
    """
    Unified search result from any vector backend.

    Provides a consistent structure regardless of which backend
    performed the search.
    """

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store connection.

    Unified configuration that works across all backends.
    Backend-specific options go in the `extra` dict.
    """

    backend: VectorBackend = VectorBackend.MEMORY
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "knowledge_mound"
    embedding_dimensions: int = 1536
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    batch_size: int = 100
    timeout_seconds: int = 30
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> VectorStoreConfig:
        """Create configuration from environment variables."""
        import os

        backend_str = os.getenv("VECTOR_BACKEND", "memory").lower()
        try:
            backend = VectorBackend(backend_str)
        except ValueError:
            logger.warning(f"Unknown backend '{backend_str}', falling back to memory")
            backend = VectorBackend.MEMORY

        return cls(
            backend=backend,
            url=os.getenv("VECTOR_STORE_URL"),
            api_key=os.getenv("VECTOR_STORE_API_KEY"),
            collection_name=os.getenv("VECTOR_COLLECTION", "knowledge_mound"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
        )


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store backends.

    Provides a unified interface across Weaviate, Qdrant, Chroma, and in-memory.
    All methods are async to support non-blocking operations.

    Subclasses must implement all abstract methods to provide backend-specific
    functionality while maintaining a consistent API.
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store with configuration.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self._connected = False

    @property
    def backend(self) -> VectorBackend:
        """Get the backend type."""
        return self.config.backend

    @property
    def is_connected(self) -> bool:
        """Check if connected to the vector store."""
        return self._connected

    @property
    def collection_name(self) -> str:
        """Get the current collection name."""
        return self.config.collection_name

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to vector store.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to vector store."""
        ...

    async def __aenter__(self) -> BaseVectorStore:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Create a new collection/index.

        Args:
            name: Collection name
            schema: Optional schema definition for metadata fields
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection name to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            name: Collection name to check

        Returns:
            True if exists
        """
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        ...

    # -------------------------------------------------------------------------
    # Data Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Insert or update a vector with content and metadata.

        Args:
            id: Unique identifier for the vector
            embedding: Vector embedding
            content: Text content associated with the vector
            metadata: Optional metadata dictionary
            namespace: Optional namespace for multi-tenant isolation

        Returns:
            The ID of the upserted vector
        """
        ...

    @abstractmethod
    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> list[str]:
        """
        Batch upsert multiple vectors.

        Each item should have: id, embedding, content, and optionally metadata.

        Args:
            items: Sequence of items to upsert
            namespace: Optional namespace for multi-tenant isolation

        Returns:
            List of upserted IDs
        """
        ...

    @abstractmethod
    async def delete(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete vectors by ID.

        Args:
            ids: IDs to delete
            namespace: Optional namespace

        Returns:
            Number of deleted vectors
        """
        ...

    @abstractmethod
    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete vectors matching filter criteria.

        Args:
            filters: Filter criteria (backend-specific format)
            namespace: Optional namespace

        Returns:
            Number of deleted vectors
        """
        ...

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            embedding: Query embedding
            limit: Maximum results to return
            filters: Optional metadata filters
            namespace: Optional namespace
            min_score: Minimum similarity score threshold

        Returns:
            List of search results sorted by similarity
        """
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        alpha: float = 0.5,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search combining vector and keyword matching.

        Args:
            query: Text query for keyword search
            embedding: Query embedding for vector search
            limit: Maximum results to return
            alpha: Balance between vector (0) and keyword (1) search
            filters: Optional metadata filters
            namespace: Optional namespace

        Returns:
            List of search results with combined scoring
        """
        ...

    # -------------------------------------------------------------------------
    # Retrieval Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_by_id(
        self,
        id: str,
        namespace: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """
        Get a specific vector by ID.

        Args:
            id: Vector ID
            namespace: Optional namespace

        Returns:
            The vector if found, None otherwise
        """
        ...

    @abstractmethod
    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """
        Get multiple vectors by ID.

        Args:
            ids: Vector IDs
            namespace: Optional namespace

        Returns:
            List of found vectors (missing IDs are skipped)
        """
        ...

    @abstractmethod
    async def count(
        self,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Count vectors matching optional filters.

        Args:
            filters: Optional metadata filters
            namespace: Optional namespace

        Returns:
            Count of matching vectors
        """
        ...

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check connection health and return diagnostics.

        Returns:
            Dict with health status and metrics
        """
        ...

    async def ping(self) -> bool:
        """
        Quick connectivity check.

        Returns:
            True if reachable
        """
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False

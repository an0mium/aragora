"""
Memory Backend Protocols for Aragora.

Defines interfaces for memory storage backends, enabling pluggable
storage implementations (SQLite, PostgreSQL, Redis, etc.) while
maintaining type safety through structural subtyping.

Phase 8B: Memory backend protocol for flexible storage options.

Usage:
    from aragora.memory.protocols import MemoryBackend, MemoryEntry

    class RedisMemoryBackend(MemoryBackend):
        async def store(self, entry: MemoryEntry) -> str:
            # Redis-specific implementation
            ...

        async def get(self, entry_id: str) -> MemoryEntry | None:
            ...
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# Generic type for memory entry payloads
T = TypeVar("T")


@dataclass
class MemoryEntry:
    """
    Base class for memory entries across all backend types.

    Provides common fields needed by all memory storage systems.
    Specific memory systems (Continuum, Consensus, Critique) can
    extend this with additional fields.
    """

    id: str
    content: str
    tier: str = "fast"  # fast, medium, slow, glacial
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Embeddings for semantic search (optional)
    embedding: Optional[list[float]] = None

    # TTL support
    expires_at: Optional[datetime] = None


@dataclass
class MemoryQueryResult:
    """Result from a memory query with pagination support."""

    entries: list[MemoryEntry]
    total_count: int
    has_more: bool
    cursor: Optional[str] = None


@dataclass
class BackendHealth:
    """Health status for a memory backend."""

    healthy: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryBackend(Protocol):
    """
    Protocol for memory storage backends.

    Defines the interface that all memory backends must implement.
    Supports both sync and async operations, with async being preferred.

    Example:
        class PostgresMemoryBackend:
            async def store(self, entry: MemoryEntry) -> str:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO memories ...",
                        entry.id, entry.content, ...
                    )
                    return entry.id

    Backend Implementations:
    - SQLiteMemoryBackend: Local file-based storage (default)
    - PostgresMemoryBackend: Production PostgreSQL storage
    - RedisMemoryBackend: High-speed caching backend
    - InMemoryBackend: Testing/development backend
    """

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.

        Args:
            entry: The memory entry to store

        Returns:
            The entry ID (may be generated if not provided)

        Raises:
            MemoryBackendError: If storage fails
        """
        ...

    @abstractmethod
    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.

        Args:
            entry_id: The unique identifier

        Returns:
            The entry if found, None otherwise
        """
        ...

    @abstractmethod
    async def update(self, entry: MemoryEntry) -> bool:
        """
        Update an existing memory entry.

        Args:
            entry: The entry with updated fields

        Returns:
            True if entry was updated, False if not found
        """
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: The unique identifier

        Returns:
            True if entry was deleted, False if not found
        """
        ...

    # =========================================================================
    # Query Operations
    # =========================================================================

    @abstractmethod
    async def query(
        self,
        tier: Optional[str] = None,
        min_weight: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> MemoryQueryResult:
        """
        Query memory entries with filtering and pagination.

        Args:
            tier: Filter by memory tier (fast, medium, slow, glacial)
            min_weight: Minimum weight threshold
            limit: Maximum entries to return
            offset: Number of entries to skip
            order_by: Field to sort by
            descending: Sort direction

        Returns:
            Query result with entries and pagination info
        """
        ...

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        tier: Optional[str] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search for similar entries using vector similarity.

        Args:
            query_embedding: The query vector
            limit: Maximum results to return
            min_similarity: Minimum cosine similarity threshold
            tier: Optional tier filter

        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity
        """
        ...

    # =========================================================================
    # Batch Operations
    # =========================================================================

    @abstractmethod
    async def store_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """
        Store multiple entries in a single operation.

        Args:
            entries: List of entries to store

        Returns:
            List of stored entry IDs
        """
        ...

    @abstractmethod
    async def delete_batch(self, entry_ids: list[str]) -> int:
        """
        Delete multiple entries in a single operation.

        Args:
            entry_ids: List of entry IDs to delete

        Returns:
            Number of entries deleted
        """
        ...

    # =========================================================================
    # Tier Operations
    # =========================================================================

    @abstractmethod
    async def promote(self, entry_id: str, new_tier: str) -> bool:
        """
        Promote an entry to a higher (slower) memory tier.

        Args:
            entry_id: The entry to promote
            new_tier: Target tier (medium, slow, glacial)

        Returns:
            True if promotion succeeded
        """
        ...

    @abstractmethod
    async def count_by_tier(self) -> dict[str, int]:
        """
        Get entry counts per tier.

        Returns:
            Dictionary mapping tier names to entry counts
        """
        ...

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        ...

    @abstractmethod
    async def vacuum(self) -> None:
        """
        Optimize storage by reclaiming space.

        For SQLite, runs VACUUM. For PostgreSQL, runs VACUUM ANALYZE.
        """
        ...

    # =========================================================================
    # Health and Diagnostics
    # =========================================================================

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """
        Check backend health and connectivity.

        Returns:
            Health status with latency and error info
        """
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """
        Get backend statistics.

        Returns:
            Dictionary with backend-specific metrics
        """
        ...


@runtime_checkable
class MemoryBackendSync(Protocol):
    """
    Synchronous variant of MemoryBackend for compatibility.

    Use this when async is not available or for simpler use cases.
    """

    def store_sync(self, entry: MemoryEntry) -> str:
        """Store a memory entry synchronously."""
        ...

    def get_sync(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID synchronously."""
        ...

    def update_sync(self, entry: MemoryEntry) -> bool:
        """Update an existing memory entry synchronously."""
        ...

    def delete_sync(self, entry_id: str) -> bool:
        """Delete a memory entry synchronously."""
        ...

    def query_sync(
        self,
        tier: Optional[str] = None,
        min_weight: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> MemoryQueryResult:
        """Query memory entries synchronously."""
        ...


@runtime_checkable
class StreamingMemoryBackend(Protocol):
    """
    Protocol for backends that support streaming large result sets.

    Useful for exporting data or processing large memory collections
    without loading everything into memory at once.
    """

    async def stream_entries(
        self,
        tier: Optional[str] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[list[MemoryEntry]]:
        """
        Stream entries in batches.

        Args:
            tier: Optional tier filter
            batch_size: Number of entries per batch

        Yields:
            Batches of memory entries
        """
        ...


@runtime_checkable
class TransactionalMemoryBackend(Protocol):
    """
    Protocol for backends that support transactions.

    Enables atomic operations across multiple memory operations.
    """

    async def begin_transaction(self) -> Any:
        """Begin a new transaction."""
        ...

    async def commit_transaction(self, tx: Any) -> None:
        """Commit a transaction."""
        ...

    async def rollback_transaction(self, tx: Any) -> None:
        """Rollback a transaction."""
        ...

    async def store_in_transaction(self, tx: Any, entry: MemoryEntry) -> str:
        """Store an entry within a transaction."""
        ...

    async def delete_in_transaction(self, tx: Any, entry_id: str) -> bool:
        """Delete an entry within a transaction."""
        ...


class MemoryBackendError(Exception):
    """Base exception for memory backend errors."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        operation: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.backend = backend
        self.operation = operation
        self.recoverable = recoverable


class MemoryBackendConnectionError(MemoryBackendError):
    """Connection to backend failed."""

    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(message, backend=backend, operation="connect", recoverable=True)


class MemoryBackendTimeoutError(MemoryBackendError):
    """Operation timed out."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, backend=backend, operation=operation, recoverable=True)


class MemoryBackendCapacityError(MemoryBackendError):
    """Backend capacity exceeded."""

    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(message, backend=backend, operation="store", recoverable=False)


# Type aliases for convenience
MemoryBackendFactory = Callable[[], MemoryBackend]
MemoryEntryTransform = Callable[[MemoryEntry], MemoryEntry]


__all__ = [
    # Core types
    "MemoryEntry",
    "MemoryQueryResult",
    "BackendHealth",
    # Protocols
    "MemoryBackend",
    "MemoryBackendSync",
    "StreamingMemoryBackend",
    "TransactionalMemoryBackend",
    # Exceptions
    "MemoryBackendError",
    "MemoryBackendConnectionError",
    "MemoryBackendTimeoutError",
    "MemoryBackendCapacityError",
    # Type aliases
    "MemoryBackendFactory",
    "MemoryEntryTransform",
]

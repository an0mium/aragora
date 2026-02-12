"""Storage and infrastructure protocol definitions.

Provides Protocol classes for Redis clients, debate storage,
and user storage backends.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RedisClientProtocol(Protocol):
    """Protocol for Redis client implementations.

    Covers standard Redis operations used by RedisClusterManager.
    This allows typing Redis clients without requiring the redis library.

    Note: Return types use Any to accommodate both bytes (default)
    and str (when decode_responses=True) modes.
    """

    # Connection management
    def close(self) -> None:
        """Close the connection."""
        ...

    def ping(self) -> bool:
        """Ping the server to check connectivity."""
        ...

    def info(self, section: str | None = None) -> dict[str, Any]:
        """Get server information."""
        ...

    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        """Execute an arbitrary Redis command."""
        ...

    # Basic key-value operations
    def get(self, key: str) -> Any:
        """Get value for key. Returns bytes or str depending on decode_responses."""
        ...

    def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None:
        """Set key to value with optional expiration."""
        ...

    def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        ...

    def exists(self, *keys: str) -> int:
        """Check how many keys exist."""
        ...

    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on key."""
        ...

    def ttl(self, key: str) -> int:
        """Get TTL of key in seconds."""
        ...

    def incr(self, key: str) -> int:
        """Increment key by 1."""
        ...

    def decr(self, key: str) -> int:
        """Decrement key by 1."""
        ...

    # Hash operations
    def hget(self, name: str, key: str) -> Any:
        """Get field from hash. Returns bytes or str depending on decode_responses."""
        ...

    def hset(self, name: str, key: str, value: Any) -> int:
        """Set field in hash."""
        ...

    def hgetall(self, name: str) -> dict[str, Any]:
        """Get all fields from hash."""
        ...

    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash."""
        ...

    # Sorted set operations
    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        """Add members to sorted set."""
        ...

    def zrem(self, name: str, *members: str) -> int:
        """Remove members from sorted set."""
        ...

    def zcard(self, name: str) -> int:
        """Get sorted set cardinality."""
        ...

    def zrangebyscore(
        self,
        name: str,
        min: Any,
        max: Any,
        withscores: bool = False,
    ) -> list[Any]:
        """Get members by score range."""
        ...

    def zremrangebyscore(self, name: str, min: Any, max: Any) -> int:
        """Remove members by score range."""
        ...

    # Pipeline support
    def pipeline(self, transaction: bool = True) -> Any:
        """Get pipeline for batch operations."""
        ...


@runtime_checkable
class DebateStorageProtocol(Protocol):
    """Protocol for debate storage backends."""

    def save_debate(self, debate_id: str, data: dict[str, Any]) -> None:
        """Save debate data."""
        ...

    def load_debate(self, debate_id: str) -> dict[str, Any] | None:
        """Load debate data."""
        ...

    def list_debates(self, limit: int = 100, org_id: str | None = None) -> list[Any]:
        """List available debates. Returns list of debate metadata objects."""
        ...

    def delete_debate(self, debate_id: str) -> bool:
        """Delete a debate."""
        ...

    def get_debate(self, debate_id: str) -> dict[str, Any] | None:
        """Get debate by ID."""
        ...

    def get_debate_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Get debate by slug."""
        ...

    def get_by_id(self, debate_id: str) -> dict[str, Any] | None:
        """Get debate by ID (alias)."""
        ...

    def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Get debate by slug (alias)."""
        ...

    def list_recent(self, limit: int = 20, org_id: str | None = None) -> list[Any]:
        """List recent debates."""
        ...

    def search(
        self,
        query: str | None = None,
        agent: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
        org_id: str | None = None,
    ) -> list[Any]:
        """Search debates."""
        ...


@runtime_checkable
class UserStoreProtocol(Protocol):
    """Protocol for user storage backends."""

    def get_user_by_id(self, user_id: str) -> Any | None:
        """Get user by ID."""
        ...

    def get_user_by_email(self, email: str) -> Any | None:
        """Get user by email."""
        ...

    def create_user(
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        **kwargs: Any,
    ) -> Any:
        """Create a new user."""
        ...

    def update_user(self, user_id: str, **kwargs: Any) -> bool:
        """Update user attributes."""
        ...


__all__ = [
    "RedisClientProtocol",
    "DebateStorageProtocol",
    "UserStoreProtocol",
]

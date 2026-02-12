"""Memory protocol definitions.

Provides Protocol classes for memory backends including basic memory,
tiered memory, critique storage, and continuum memory.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory backends.

    Provides basic store/query interface for memory systems.
    """

    def store(self, content: str, **kwargs: Any) -> str:
        """Store content and return an identifier."""
        ...

    def query(self, **kwargs: Any) -> list[Any]:
        """Query stored content."""
        ...


@runtime_checkable
class TieredMemoryProtocol(MemoryProtocol, Protocol):
    """Protocol for tiered memory systems like ContinuumMemory."""

    def store(
        self,
        content: str,
        tier: Any = None,  # MemoryTier, optional for protocol compatibility
        importance: float = 0.5,
        **kwargs: Any,
    ) -> str:
        """Store content in a specific tier."""
        ...

    def query(
        self,
        tier: Any | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        **kwargs: Any,
    ) -> list[Any]:
        """Query content from specified tier."""
        ...

    def promote(self, entry_id: str, target_tier: Any) -> bool:
        """Promote entry to a faster tier."""
        ...

    def demote(self, entry_id: str, target_tier: Any) -> bool:
        """Demote entry to a slower tier."""
        ...

    def cleanup_expired_memories(self) -> int:
        """Clean up expired memories. Returns count of cleaned entries."""
        ...

    def enforce_tier_limits(self) -> None:
        """Enforce tier size limits by evicting excess entries."""
        ...


@runtime_checkable
class CritiqueStoreProtocol(Protocol):
    """Protocol for critique/pattern storage."""

    def store_pattern(self, critique: Any, resolution: str) -> str:
        """Store a critique pattern with its resolution."""
        ...

    def retrieve_patterns(
        self,
        issue_type: str | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Retrieve stored patterns."""
        ...

    def get_reputation(self, agent: str) -> dict[str, Any]:
        """Get reputation data for an agent."""
        ...


@runtime_checkable
class ContinuumMemoryProtocol(Protocol):
    """Protocol for cross-debate learning memory.

    ContinuumMemory provides multi-tier memory for long-term learning across debates.
    Used by Arena to provide historical context and cross-debate learning.
    """

    def store(
        self,
        key: str,
        value: Any,
        tier: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a value in the specified memory tier."""
        ...

    def retrieve(
        self,
        key: str,
        tier: str | None = None,
    ) -> Any | None:
        """Retrieve a value, searching tiers if not specified."""
        ...

    def search(
        self,
        query: str,
        limit: int = 10,
        tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant memories."""
        ...

    def get_context(
        self,
        task: str,
        limit: int = 5,
    ) -> str:
        """Get formatted context for a task from historical memories."""
        ...


__all__ = [
    "MemoryProtocol",
    "TieredMemoryProtocol",
    "CritiqueStoreProtocol",
    "ContinuumMemoryProtocol",
]

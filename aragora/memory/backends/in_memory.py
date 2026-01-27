"""
In-Memory Backend Implementation.

Fast, non-persistent memory backend for testing and development.
Implements the MemoryBackend protocol from aragora.memory.protocols.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aragora.memory.protocols import (
    BackendHealth,
    MemoryBackend,
    MemoryEntry,
    MemoryQueryResult,
)


@dataclass
class InMemoryBackend:
    """
    In-memory implementation of MemoryBackend.

    Uses dictionaries for O(1) lookups. Suitable for testing,
    development, and small-scale deployments.

    Thread-safe through asyncio locks.
    """

    _entries: dict[str, MemoryEntry] = field(default_factory=dict)
    _by_tier: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        async with self._lock:
            self._entries[entry.id] = entry
            self._by_tier[entry.tier].add(entry.id)
            return entry.id

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        return self._entries.get(entry_id)

    async def update(self, entry: MemoryEntry) -> bool:
        """Update an existing memory entry."""
        async with self._lock:
            if entry.id not in self._entries:
                return False

            old_entry = self._entries[entry.id]

            # Update tier index if tier changed
            if old_entry.tier != entry.tier:
                self._by_tier[old_entry.tier].discard(entry.id)
                self._by_tier[entry.tier].add(entry.id)

            entry.updated_at = datetime.utcnow()
            self._entries[entry.id] = entry
            return True

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        async with self._lock:
            if entry_id not in self._entries:
                return False

            entry = self._entries.pop(entry_id)
            self._by_tier[entry.tier].discard(entry_id)
            return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def query(
        self,
        tier: str | None = None,
        min_weight: float | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> MemoryQueryResult:
        """Query memory entries with filtering and pagination."""
        # Filter entries
        if tier is not None:
            entry_ids = self._by_tier.get(tier, set())
            entries = [self._entries[eid] for eid in entry_ids if eid in self._entries]
        else:
            entries = list(self._entries.values())

        if min_weight is not None:
            entries = [e for e in entries if e.weight >= min_weight]

        # Sort
        if order_by in ("created_at", "updated_at", "weight", "tier"):
            entries.sort(key=lambda e: getattr(e, order_by), reverse=descending)

        total_count = len(entries)

        # Paginate
        paginated = entries[offset : offset + limit]
        has_more = offset + limit < total_count

        return MemoryQueryResult(
            entries=paginated,
            total_count=total_count,
            has_more=has_more,
            cursor=str(offset + limit) if has_more else None,
        )

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        tier: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search for similar entries using vector similarity.

        Uses cosine similarity for comparison.
        """
        results: list[tuple[MemoryEntry, float]] = []

        for entry in self._entries.values():
            if tier is not None and entry.tier != tier:
                continue

            if entry.embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= min_similarity:
                results.append((entry, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def store_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Store multiple entries in a single operation."""
        ids = []
        async with self._lock:
            for entry in entries:
                self._entries[entry.id] = entry
                self._by_tier[entry.tier].add(entry.id)
                ids.append(entry.id)
        return ids

    async def delete_batch(self, entry_ids: list[str]) -> int:
        """Delete multiple entries in a single operation."""
        deleted = 0
        async with self._lock:
            for entry_id in entry_ids:
                if entry_id in self._entries:
                    entry = self._entries.pop(entry_id)
                    self._by_tier[entry.tier].discard(entry_id)
                    deleted += 1
        return deleted

    # =========================================================================
    # Tier Operations
    # =========================================================================

    async def promote(self, entry_id: str, new_tier: str) -> bool:
        """Promote an entry to a higher (slower) memory tier."""
        async with self._lock:
            if entry_id not in self._entries:
                return False

            entry = self._entries[entry_id]
            old_tier = entry.tier

            # Update tier
            entry.tier = new_tier
            entry.updated_at = datetime.utcnow()

            # Update indices
            self._by_tier[old_tier].discard(entry_id)
            self._by_tier[new_tier].add(entry_id)

            return True

    async def count_by_tier(self) -> dict[str, int]:
        """Get entry counts per tier."""
        return {tier: len(ids) for tier, ids in self._by_tier.items()}

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = datetime.utcnow()
        expired_ids = []

        for entry_id, entry in self._entries.items():
            if entry.expires_at is not None and entry.expires_at < now:
                expired_ids.append(entry_id)

        return await self.delete_batch(expired_ids)

    async def vacuum(self) -> None:
        """
        Optimize storage by reclaiming space.

        For in-memory backend, this is a no-op as Python handles memory.
        """
        pass

    # =========================================================================
    # Health and Diagnostics
    # =========================================================================

    async def health_check(self) -> BackendHealth:
        """Check backend health and connectivity."""
        return BackendHealth(
            healthy=True,
            latency_ms=0.0,
            details={"backend": "in_memory", "entry_count": len(self._entries)},
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        tier_counts = await self.count_by_tier()
        return {
            "total_entries": len(self._entries),
            "tier_counts": tier_counts,
            "memory_bytes": sum(len(e.content.encode("utf-8")) for e in self._entries.values()),
        }

    # =========================================================================
    # Additional Utility Methods
    # =========================================================================

    async def clear(self) -> int:
        """Clear all entries. Returns count of deleted entries."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._by_tier.clear()
            return count

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._entries)


# Verify protocol compliance
def _verify_protocol() -> None:
    """Verify InMemoryBackend implements MemoryBackend protocol."""
    backend: MemoryBackend = InMemoryBackend()  # Type check
    assert isinstance(backend, MemoryBackend)


__all__ = ["InMemoryBackend"]

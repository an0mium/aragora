"""
CRUD operations mixin for ContinuumMemory.

Provides add, get, update, delete operations for memory entries.
"""
# mypy: disable-error-code="misc"
# Mixin classes use self: "ContinuumMemory" type hints which mypy doesn't understand

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from aragora.memory.tier_manager import MemoryTier
from aragora.resilience.retry import PROVIDER_RETRY_POLICIES, with_retry
from aragora.utils.json_helpers import safe_json_loads

from .entry import ContinuumMemoryEntry

if TYPE_CHECKING:
    from .core import ContinuumMemory

logger = logging.getLogger(__name__)

# Retry configuration for memory operations
_MEMORY_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["memory"]


class CrudMixin:
    """Mixin providing CRUD operations for ContinuumMemory."""

    def add(
        self: "ContinuumMemory",
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """
        Add a new memory entry to the continuum.

        Args:
            id: Unique identifier for the memory
            content: The memory content
            tier: Initial memory tier
            importance: 0-1 importance score
            metadata: Optional additional data

        Returns:
            The created memory entry
        """
        now: str = datetime.now().isoformat()

        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO continuum_memory
                (id, tier, content, importance, surprise_score, consolidation_score,
                 update_count, success_count, failure_count, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, 0.0, 0.0, 1, 0, 0, ?, ?, ?)
                """,
                (id, tier.value, content, importance, now, now, json.dumps(metadata or {})),
            )
            conn.commit()

        entry: ContinuumMemoryEntry = ContinuumMemoryEntry(
            id=id,
            tier=tier,
            content=content,
            importance=importance,
            surprise_score=0.0,
            consolidation_score=0.0,
            update_count=1,
            success_count=0,
            failure_count=0,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        # Emit MEMORY_STORED event for cross-subsystem tracking
        if self.event_emitter:
            try:
                self.event_emitter.emit_sync(
                    event_type="memory_stored",
                    debate_id="",
                    memory_id=id,
                    tier=tier.value,
                    importance=importance,
                    content_length=len(content),
                )
            except (ImportError, AttributeError, TypeError):
                pass  # Events module not available

        # Sync to Knowledge Mound if adapter is configured and importance is high
        if self._km_adapter and importance >= 0.7:
            try:
                self._km_adapter.store_memory(entry)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Failed to sync memory to KM (network): {e}")
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to sync memory to KM (data): {e}")
            except (RuntimeError, AttributeError) as e:
                logger.warning(f"Unexpected error syncing memory to KM: {e}")

        return entry

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def add_async(
        self: "ContinuumMemory",
        id: str,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """Async wrapper for add() - offloads blocking I/O to executor."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.add(
                id=id,
                content=content,
                tier=tier,
                importance=importance,
                metadata=metadata,
            ),
        )

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def store(
        self: "ContinuumMemory",
        key: str,
        content: str,
        tier: str | MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuumMemoryEntry:
        """Async wrapper for add() - offloads blocking I/O to executor."""
        normalized_tier: MemoryTier = MemoryTier(tier) if isinstance(tier, str) else tier
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.add(
                id=key,
                content=content,
                tier=normalized_tier,
                importance=importance,
                metadata=metadata,
            ),
        )

    def get(self: "ContinuumMemory", id: str) -> ContinuumMemoryEntry | None:
        """Get a memory entry by ID."""
        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, tier, content, importance, surprise_score, consolidation_score,
                       update_count, success_count, failure_count, created_at, updated_at, metadata,
                       COALESCE(red_line, 0), COALESCE(red_line_reason, '')
                FROM continuum_memory
                WHERE id = ?
                """,
                (id,),
            )
            row: tuple[Any, ...] | None = cursor.fetchone()

        if not row:
            return None

        return ContinuumMemoryEntry(
            id=row[0],
            tier=MemoryTier(row[1]),
            content=row[2],
            importance=row[3],
            surprise_score=row[4],
            consolidation_score=row[5],
            update_count=row[6],
            success_count=row[7],
            failure_count=row[8],
            created_at=row[9],
            updated_at=row[10],
            metadata=safe_json_loads(row[11], {}),
            red_line=bool(row[12]),
            red_line_reason=row[13],
        )

    @with_retry(_MEMORY_RETRY_CONFIG)
    async def get_async(self: "ContinuumMemory", id: str) -> ContinuumMemoryEntry | None:
        """Async wrapper for get() - offloads blocking I/O to executor."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get, id)

    def get_entry(self: "ContinuumMemory", id: str) -> ContinuumMemoryEntry | None:
        """Alias for get() for interface compatibility with OutcomeMemoryBridge."""
        return self.get(id)

    def update_entry(self: "ContinuumMemory", entry: ContinuumMemoryEntry) -> bool:
        """Update an entry's success/failure counts.

        Interface compatibility method for OutcomeMemoryBridge.
        """
        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE continuum_memory
                SET success_count = ?, failure_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (entry.success_count, entry.failure_count, datetime.now().isoformat(), entry.id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update(
        self: "ContinuumMemory",
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: Optional[dict[str, Any]] = None,
        surprise_score: float | None = None,
        consolidation_score: float | None = None,
    ) -> bool:
        """Update specific fields of a memory entry.

        Flexible update method for modifying individual fields.
        Used by ContinuumAdapter for KM validation reverse flow.

        Args:
            memory_id: The ID of the memory entry to update
            content: New content (optional)
            importance: New importance score (optional)
            metadata: New metadata dict (optional, replaces existing)
            surprise_score: New surprise score (optional)
            consolidation_score: New consolidation score (optional)

        Returns:
            True if the entry was updated, False if not found
        """
        # Build update clauses dynamically
        updates: list[str] = []
        params: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        if surprise_score is not None:
            updates.append("surprise_score = ?")
            params.append(surprise_score)
        if consolidation_score is not None:
            updates.append("consolidation_score = ?")
            params.append(consolidation_score)

        if not updates:
            return False

        # Always update timestamp
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        # Add memory_id as final parameter
        params.append(memory_id)

        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE continuum_memory
                SET {", ".join(updates)}
                WHERE id = ?
                """,
                tuple(params),
            )
            conn.commit()
            return cursor.rowcount > 0

    def promote_entry(self: "ContinuumMemory", memory_id: str, new_tier: MemoryTier) -> bool:
        """Promote an entry to a specific tier.

        Interface compatibility method for OutcomeMemoryBridge.
        """
        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE continuum_memory
                SET tier = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_tier.value, datetime.now().isoformat(), memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def demote_entry(self: "ContinuumMemory", memory_id: str, new_tier: MemoryTier) -> bool:
        """Demote an entry to a specific tier.

        Interface compatibility method for OutcomeMemoryBridge.
        """
        with self.connection() as conn:
            cursor: sqlite3.Cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE continuum_memory
                SET tier = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_tier.value, datetime.now().isoformat(), memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0


__all__ = ["CrudMixin"]

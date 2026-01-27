"""
Hook Queue: GUPP Recovery Pattern.

Inspired by Gastown's GUPP (Guaranteed Unconditional Processing Priority) principle:
"If there is work on your Hook, YOU MUST RUN IT."

This module provides per-agent work queues that persist across restarts,
ensuring that assigned work is never lost and agents always resume
their incomplete work before accepting new tasks.

Key concepts:
- HookQueue: Per-agent work queue backed by beads
- HookEntry: An entry in an agent's hook (work queue)
- GUPP Principle: On startup, agents must process their hook before new work

Usage:
    # Create hook queue for an agent
    hook = HookQueue(agent_id="claude-001", bead_store=store)
    await hook.initialize()

    # Push work to agent's hook
    await hook.push(bead_id="task-123", priority=75)

    # On agent startup - GUPP recovery
    pending_work = await hook.recover_on_startup()
    for bead in pending_work:
        await process_bead(bead)

    # Normal processing
    while True:
        bead = await hook.pop()
        if bead:
            await process_bead(bead)
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.nomic.beads import Bead, BeadStatus, BeadStore

logger = logging.getLogger(__name__)


class HookEntryStatus(str, Enum):
    """Status of a hook entry."""

    QUEUED = "queued"  # Waiting to be processed
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Successfully processed
    FAILED = "failed"  # Processing failed
    SKIPPED = "skipped"  # Skipped (e.g., bead no longer exists)


@dataclass
class HookEntry:
    """
    An entry in an agent's work queue (hook).

    Hook entries are lightweight references to beads that track
    processing state for a specific agent.
    """

    id: str
    bead_id: str
    agent_id: str
    priority: int
    status: HookEntryStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        bead_id: str,
        agent_id: str,
        priority: int = 50,
        max_attempts: int = 3,
    ) -> "HookEntry":
        """Create a new hook entry."""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4())[:12],
            bead_id=bead_id,
            agent_id=agent_id,
            priority=priority,
            status=HookEntryStatus.QUEUED,
            created_at=now,
            updated_at=now,
            max_attempts=max_attempts,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "bead_id": self.bead_id,
            "agent_id": self.agent_id,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HookEntry":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            bead_id=data["bead_id"],
            agent_id=data["agent_id"],
            priority=data["priority"],
            status=HookEntryStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    def can_retry(self) -> bool:
        """Check if entry can be retried based on attempt count."""
        return self.attempt_count < self.max_attempts


class HookQueue:
    """
    Per-agent work queue implementing the GUPP principle.

    The GUPP (Guaranteed Unconditional Processing Priority) principle states:
    "If there is work on your Hook, YOU MUST RUN IT."

    This ensures that:
    1. Work is never lost across agent restarts
    2. Agents always resume incomplete work before new tasks
    3. Work is processed in priority order within the queue
    """

    def __init__(
        self,
        agent_id: str,
        bead_store: BeadStore,
        hooks_dir: Optional[Path] = None,
    ):
        """
        Initialize a hook queue for an agent.

        Args:
            agent_id: The agent this hook belongs to
            bead_store: The bead store for bead operations
            hooks_dir: Directory for hook storage (defaults to bead_store's dir)
        """
        self.agent_id = agent_id
        self.bead_store = bead_store
        self.hooks_dir = hooks_dir or bead_store.bead_dir / "hooks"
        self.hook_file = self.hooks_dir / f"{agent_id}.jsonl"
        self._lock = asyncio.Lock()
        self._entries: Dict[str, HookEntry] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the hook queue, loading existing entries."""
        if self._initialized:
            return

        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        await self._load_entries()
        self._initialized = True

        pending_count = len(
            [e for e in self._entries.values() if e.status == HookEntryStatus.QUEUED]
        )
        logger.info(
            f"HookQueue initialized for {self.agent_id}: "
            f"{len(self._entries)} entries, {pending_count} pending"
        )

    async def _load_entries(self) -> None:
        """Load hook entries from file."""
        if not self.hook_file.exists():
            return

        try:
            with open(self.hook_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = HookEntry.from_dict(data)
                        self._entries[entry.id] = entry
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid hook entry: {e}")
        except Exception as e:
            logger.error(f"Failed to load hook entries: {e}")

    async def _save_entries(self) -> None:
        """Save all entries to file."""
        temp_file = self.hook_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                for entry in self._entries.values():
                    f.write(json.dumps(entry.to_dict()) + "\n")
            temp_file.rename(self.hook_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    async def push(
        self,
        bead_id: str,
        priority: int = 50,
        max_attempts: int = 3,
    ) -> HookEntry:
        """
        Push a bead onto this agent's hook.

        Args:
            bead_id: The bead to add to the hook
            priority: Priority (higher = more urgent)
            max_attempts: Maximum processing attempts

        Returns:
            The created hook entry
        """
        async with self._lock:
            # Verify bead exists
            bead = await self.bead_store.get(bead_id)
            if not bead:
                raise ValueError(f"Bead {bead_id} not found")

            # Check if already in queue
            existing = [e for e in self._entries.values() if e.bead_id == bead_id]
            if existing:
                entry = existing[0]
                if entry.status in (HookEntryStatus.QUEUED, HookEntryStatus.PROCESSING):
                    logger.warning(f"Bead {bead_id} already in hook for {self.agent_id}")
                    return entry

            entry = HookEntry.create(
                bead_id=bead_id,
                agent_id=self.agent_id,
                priority=priority,
                max_attempts=max_attempts,
            )

            self._entries[entry.id] = entry
            await self._save_entries()

            logger.debug(f"Pushed bead {bead_id} to hook for {self.agent_id}")
            return entry

    async def pop(self) -> Optional[Bead]:
        """
        Pop the highest priority bead from the hook.

        Marks the entry as PROCESSING and claims the bead.

        Returns:
            The bead to process, or None if hook is empty
        """
        async with self._lock:
            # Get queued entries sorted by priority (highest first)
            queued = [e for e in self._entries.values() if e.status == HookEntryStatus.QUEUED]
            if not queued:
                return None

            queued.sort(key=lambda e: e.priority, reverse=True)
            entry = queued[0]

            # Get the bead
            bead = await self.bead_store.get(entry.bead_id)
            if not bead:
                # Bead no longer exists, skip this entry
                entry.status = HookEntryStatus.SKIPPED
                entry.updated_at = datetime.now(timezone.utc)
                await self._save_entries()
                return await self.pop()  # Try next entry

            # Mark entry as processing
            entry.status = HookEntryStatus.PROCESSING
            entry.started_at = datetime.now(timezone.utc)
            entry.updated_at = entry.started_at
            entry.attempt_count += 1
            await self._save_entries()

            # Claim the bead if not already claimed
            if bead.status == BeadStatus.PENDING:
                await self.bead_store.claim(bead.id, self.agent_id)
                await self.bead_store.update_status(bead.id, BeadStatus.RUNNING)
            elif bead.status == BeadStatus.CLAIMED and bead.claimed_by == self.agent_id:
                await self.bead_store.update_status(bead.id, BeadStatus.RUNNING)

            # Re-fetch to get updated bead
            bead = await self.bead_store.get(entry.bead_id)
            return bead

    async def peek(self) -> Optional[Bead]:
        """
        Peek at the next bead without removing it from the queue.

        Returns:
            The next bead, or None if hook is empty
        """
        queued = [e for e in self._entries.values() if e.status == HookEntryStatus.QUEUED]
        if not queued:
            return None

        queued.sort(key=lambda e: e.priority, reverse=True)
        entry = queued[0]

        return await self.bead_store.get(entry.bead_id)

    async def complete(self, bead_id: str) -> None:
        """
        Mark a bead as completed in the hook.

        Args:
            bead_id: The bead that was completed
        """
        async with self._lock:
            entries = [e for e in self._entries.values() if e.bead_id == bead_id]
            if not entries:
                return

            entry = entries[0]
            entry.status = HookEntryStatus.COMPLETED
            entry.completed_at = datetime.now(timezone.utc)
            entry.updated_at = entry.completed_at
            await self._save_entries()

            # Update bead status
            await self.bead_store.update_status(bead_id, BeadStatus.COMPLETED)

            logger.debug(f"Completed bead {bead_id} in hook for {self.agent_id}")

    async def fail(self, bead_id: str, error_message: str) -> bool:
        """
        Mark a bead as failed in the hook.

        Args:
            bead_id: The bead that failed
            error_message: Error message describing the failure

        Returns:
            True if can retry, False if max attempts reached
        """
        async with self._lock:
            entries = [e for e in self._entries.values() if e.bead_id == bead_id]
            if not entries:
                return False

            entry = entries[0]
            entry.error_message = error_message
            entry.updated_at = datetime.now(timezone.utc)

            if entry.can_retry():
                # Queue for retry
                entry.status = HookEntryStatus.QUEUED
                await self._save_entries()
                logger.info(
                    f"Bead {bead_id} failed (attempt {entry.attempt_count}/{entry.max_attempts}), "
                    f"will retry"
                )
                return True
            else:
                # Max attempts reached
                entry.status = HookEntryStatus.FAILED
                entry.completed_at = entry.updated_at
                await self._save_entries()

                # Update bead status
                await self.bead_store.update_status(bead_id, BeadStatus.FAILED, error_message)
                logger.warning(f"Bead {bead_id} failed after {entry.attempt_count} attempts")
                return False

    async def has_work(self) -> bool:
        """Check if there is work in the hook."""
        return any(e.status == HookEntryStatus.QUEUED for e in self._entries.values())

    async def recover_on_startup(self) -> List[Bead]:
        """
        GUPP: Return all incomplete work on this agent's hook.

        Called on agent startup to ensure any work that was in progress
        or queued is resumed before accepting new work.

        Returns:
            List of beads that need to be processed
        """
        await self.initialize()

        async with self._lock:
            # Reset any PROCESSING entries back to QUEUED
            for entry in self._entries.values():
                if entry.status == HookEntryStatus.PROCESSING:
                    entry.status = HookEntryStatus.QUEUED
                    entry.started_at = None
                    entry.updated_at = datetime.now(timezone.utc)

            await self._save_entries()

            # Collect all beads that need processing
            beads = []
            queued = [e for e in self._entries.values() if e.status == HookEntryStatus.QUEUED]
            queued.sort(key=lambda e: e.priority, reverse=True)

            for entry in queued:
                bead = await self.bead_store.get(entry.bead_id)
                if bead and not bead.is_terminal():
                    beads.append(bead)

            logger.info(f"GUPP recovery for {self.agent_id}: {len(beads)} beads to process")
            return beads

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this hook queue."""
        entries = list(self._entries.values())
        by_status = {}
        for entry in entries:
            by_status[entry.status.value] = by_status.get(entry.status.value, 0) + 1

        return {
            "agent_id": self.agent_id,
            "total_entries": len(entries),
            "by_status": by_status,
            "pending": by_status.get(HookEntryStatus.QUEUED.value, 0),
            "processing": by_status.get(HookEntryStatus.PROCESSING.value, 0),
        }

    async def clear_completed(self) -> int:
        """
        Clear completed entries from the hook.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            terminal_statuses = {
                HookEntryStatus.COMPLETED,
                HookEntryStatus.FAILED,
                HookEntryStatus.SKIPPED,
            }
            to_remove = [e.id for e in self._entries.values() if e.status in terminal_statuses]

            for entry_id in to_remove:
                del self._entries[entry_id]

            if to_remove:
                await self._save_entries()

            return len(to_remove)


class HookQueueRegistry:
    """
    Registry for managing multiple agent hook queues.

    Provides centralized access to all hook queues and
    coordinated recovery on system startup.
    """

    def __init__(self, bead_store: BeadStore, hooks_dir: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            bead_store: The bead store for bead operations
            hooks_dir: Directory for hook storage
        """
        self.bead_store = bead_store
        self.hooks_dir = hooks_dir or bead_store.bead_dir / "hooks"
        self._queues: Dict[str, HookQueue] = {}
        self._lock = asyncio.Lock()

    async def get_queue(self, agent_id: str) -> HookQueue:
        """
        Get or create a hook queue for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            The agent's hook queue
        """
        async with self._lock:
            if agent_id not in self._queues:
                queue = HookQueue(
                    agent_id=agent_id,
                    bead_store=self.bead_store,
                    hooks_dir=self.hooks_dir,
                )
                await queue.initialize()
                self._queues[agent_id] = queue

            return self._queues[agent_id]

    async def recover_all(self) -> Dict[str, List[Bead]]:
        """
        Perform GUPP recovery for all known agents.

        Returns:
            Dict mapping agent_id to list of beads needing processing
        """
        # Discover all hook files
        if not self.hooks_dir.exists():
            return {}

        results = {}
        for hook_file in self.hooks_dir.glob("*.jsonl"):
            agent_id = hook_file.stem
            queue = await self.get_queue(agent_id)
            beads = await queue.recover_on_startup()
            if beads:
                results[agent_id] = beads

        total_beads = sum(len(beads) for beads in results.values())
        logger.info(f"GUPP recovery complete: {len(results)} agents, {total_beads} beads")
        return results

    async def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all hook queues."""
        stats = {}
        for agent_id, queue in self._queues.items():
            stats[agent_id] = await queue.get_statistics()

        return {
            "total_agents": len(self._queues),
            "queues": stats,
        }


# Singleton registry
_default_registry: Optional[HookQueueRegistry] = None


async def get_hook_queue_registry(bead_store: BeadStore) -> HookQueueRegistry:
    """Get the default hook queue registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = HookQueueRegistry(bead_store)
    return _default_registry


def reset_hook_queue_registry() -> None:
    """Reset the default registry (for testing)."""
    global _default_registry
    _default_registry = None

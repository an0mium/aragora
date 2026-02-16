"""
Beads: Git-Backed Atomic Work Units.

.. deprecated::
    External consumers should import from the canonical stores package::

        from aragora.nomic.stores import BeadStore, Bead, BeadStatus, BeadType

    Direct imports from ``aragora.nomic.beads`` are reserved for internal
    nomic-package use only.

Inspired by Gastown's Beads pattern, this module provides persistent work tracking
that survives agent restarts. Beads are stored in JSONL format and can be backed
by git for durability and auditability.

Key concepts:
- Bead: An atomic work unit (task, issue, epic, or hook)
- BeadStore: JSONL-based persistence with optional git backing
- BeadType: Classification of work (ISSUE, TASK, EPIC, HOOK)
- BeadStatus: Lifecycle states (PENDING, CLAIMED, RUNNING, COMPLETED, FAILED)

Usage:
    store = await create_bead_store()

    # Create a bead
    bead = Bead.create(
        bead_type=BeadType.TASK,
        title="Implement feature X",
        description="Add the new feature...",
    )
    await store.create(bead)

    # Claim and run
    await store.claim(bead.id, agent_id="claude-001")
    await store.update_status(bead.id, BeadStatus.RUNNING)

    # Complete
    await store.update_status(bead.id, BeadStatus.COMPLETED)
    await store.commit_to_git("Completed task: Implement feature X")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Safe pattern for bead identifiers (alphanumeric, hyphens, underscores)
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class BeadType(str, Enum):
    """Type of work unit."""

    ISSUE = "issue"  # Bug report or feature request
    TASK = "task"  # Actionable work item
    EPIC = "epic"  # Large work item containing subtasks
    HOOK = "hook"  # Special: per-agent work queue entry
    DEBATE_DECISION = "debate_decision"  # Decision from a multi-agent debate


class BeadStatus(str, Enum):
    """Lifecycle status of a bead."""

    PENDING = "pending"  # Not yet started
    CLAIMED = "claimed"  # Assigned to an agent
    RUNNING = "running"  # Work in progress
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled before completion
    BLOCKED = "blocked"  # Waiting on dependencies


class BeadPriority(int, Enum):
    """Priority levels for beads."""

    LOW = 0
    NORMAL = 50
    HIGH = 75
    URGENT = 100


@dataclass
class Bead:
    """
    Git-backed atomic work unit.

    Beads are the fundamental unit of work tracking in the system.
    They can represent issues, tasks, epics, or hook entries.
    """

    id: str  # UUID
    bead_type: BeadType  # issue, task, epic, hook
    status: BeadStatus  # pending, claimed, running, completed, failed
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    claimed_by: str | None = None  # agent_id
    claimed_at: datetime | None = None
    completed_at: datetime | None = None
    parent_id: str | None = None  # For hierarchical beads
    dependencies: list[str] = field(default_factory=list)  # Bead IDs
    priority: BeadPriority = BeadPriority.NORMAL
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3

    @classmethod
    def create(
        cls,
        bead_type: BeadType,
        title: str,
        description: str = "",
        parent_id: str | None = None,
        dependencies: list[str] | None = None,
        priority: BeadPriority = BeadPriority.NORMAL,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bead:
        """Create a new bead with generated ID and timestamps."""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            bead_type=bead_type,
            status=BeadStatus.PENDING,
            title=title,
            description=description,
            created_at=now,
            updated_at=now,
            parent_id=parent_id,
            dependencies=dependencies or [],
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize bead to dictionary for JSON storage."""
        data = asdict(self)
        # Convert enums to strings
        data["bead_type"] = self.bead_type.value
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        # Convert datetimes to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.claimed_at:
            data["claimed_at"] = self.claimed_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bead:
        """Deserialize bead from dictionary."""
        return cls(
            id=data["id"],
            bead_type=BeadType(data["bead_type"]),
            status=BeadStatus(data["status"]),
            title=data["title"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            claimed_by=data.get("claimed_by"),
            claimed_at=(
                datetime.fromisoformat(data["claimed_at"]) if data.get("claimed_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            priority=BeadPriority(data.get("priority", BeadPriority.NORMAL.value)),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            error_message=data.get("error_message"),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
        )

    def can_start(self, completed_bead_ids: set) -> bool:
        """Check if this bead can start (all dependencies completed)."""
        return all(dep_id in completed_bead_ids for dep_id in self.dependencies)

    def is_terminal(self) -> bool:
        """Check if bead is in a terminal state."""
        return self.status in (
            BeadStatus.COMPLETED,
            BeadStatus.FAILED,
            BeadStatus.CANCELLED,
        )

    def can_retry(self) -> bool:
        """Check if bead can be retried."""
        return self.status == BeadStatus.FAILED and self.attempt_count < self.max_attempts

    # BeadRecord protocol properties (cross-layer compatibility)
    @property
    def bead_id(self) -> str:
        """Protocol: bead identifier."""
        return self.id

    @property
    def bead_convoy_id(self) -> str | None:
        """Protocol: convoy ID (beads don't track this; convoys track bead_ids)."""
        return None  # Inverse relationship - lookup convoy.bead_ids

    @property
    def bead_status_value(self) -> str:
        """Protocol: status enum value."""
        return self.status.value

    @property
    def bead_content(self) -> str:
        """Protocol: bead content (maps to description)."""
        return self.description

    @property
    def bead_created_at(self) -> datetime:
        """Protocol: creation timestamp."""
        return self.created_at

    @property
    def bead_metadata(self) -> dict[str, Any]:
        """Protocol: metadata dictionary."""
        return self.metadata


@dataclass
class BeadEvent:
    """Event recording a bead state change."""

    event_id: str
    bead_id: str
    event_type: str  # created, claimed, started, completed, failed, etc.
    timestamp: datetime
    agent_id: str | None = None
    old_status: BeadStatus | None = None
    new_status: BeadStatus | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "bead_id": self.bead_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "old_status": self.old_status.value if self.old_status else None,
            "new_status": self.new_status.value if self.new_status else None,
            "data": self.data,
        }


class BeadStore:
    """
    JSONL-based bead persistence with optional git backing.

    Stores beads in a JSONL file (one JSON object per line) for efficient
    append-only writes and streaming reads. Optionally commits changes to git.
    """

    def __init__(
        self,
        bead_dir: Path,
        git_enabled: bool = True,
        auto_commit: bool = False,
    ):
        """
        Initialize the bead store.

        Args:
            bead_dir: Directory for bead storage
            git_enabled: Whether to enable git operations
            auto_commit: Whether to auto-commit after each write
        """
        self.bead_dir = Path(bead_dir)
        self.bead_file = self.bead_dir / "beads.jsonl"
        self.events_file = self.bead_dir / "events.jsonl"
        self.index_file = self.bead_dir / "index.json"
        self.git_enabled = git_enabled
        self.auto_commit = auto_commit
        self._lock = asyncio.Lock()
        self._index: dict[str, int] = {}  # bead_id -> line number
        self._beads_cache: dict[str, Bead] = {}  # In-memory cache
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the store, creating directories and loading index."""
        if self._initialized:
            return

        # Create directory
        self.bead_dir.mkdir(parents=True, exist_ok=True)

        # Initialize git if enabled
        if self.git_enabled:
            await self._init_git()

        # Load existing beads into cache
        await self._load_beads()
        self._initialized = True
        logger.info(f"BeadStore initialized with {len(self._beads_cache)} beads")

    async def _init_git(self) -> None:
        """Initialize git repository if not exists."""
        git_dir = self.bead_dir / ".git"
        if not git_dir.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "init",
                    cwd=str(self.bead_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=15)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise
                logger.info(f"Initialized git repository in {self.bead_dir}")
            except OSError as e:
                logger.warning(f"Could not initialize git: {e}")
                self.git_enabled = False

    def _load_beads_sync(self) -> tuple[dict[str, Bead], dict[str, int]]:
        """Synchronous helper to load all beads from JSONL file."""
        cache: dict[str, Bead] = {}
        index: dict[str, int] = {}
        if not self.bead_file.exists():
            return cache, index

        try:
            with open(self.bead_file) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        bead = Bead.from_dict(data)
                        cache[bead.id] = bead
                        index[bead.id] = line_num
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid bead at line {line_num}: {e}")
        except OSError as e:
            logger.error(f"Failed to load beads: {e}")
        return cache, index

    async def _load_beads(self) -> None:
        """Load all beads from JSONL file into cache."""
        loop = asyncio.get_event_loop()
        cache, index = await loop.run_in_executor(None, self._load_beads_sync)
        self._beads_cache = cache
        self._index = index

    def _append_bead_sync(self, bead: Bead) -> None:
        """Synchronous helper to append a bead to the JSONL file."""
        with open(self.bead_file, "a") as f:
            f.write(json.dumps(bead.to_dict()) + "\n")

    async def _append_bead(self, bead: Bead) -> None:
        """Append a bead to the JSONL file."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._append_bead_sync, bead)

    def _rewrite_file_sync(self) -> None:
        """Synchronous helper to rewrite the entire JSONL file from cache."""
        temp_file = self.bead_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                for bead_id, bead in self._beads_cache.items():
                    f.write(json.dumps(bead.to_dict()) + "\n")
                    self._index[bead_id] = len(self._index)
            # Atomic rename
            temp_file.rename(self.bead_file)
        except OSError as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    async def _rewrite_file(self) -> None:
        """Rewrite the entire JSONL file from cache (for updates)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._rewrite_file_sync)

    def _record_event_sync(self, event: BeadEvent) -> None:
        """Synchronous helper to record a bead event to the events log."""
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    async def _record_event(self, event: BeadEvent) -> None:
        """Record a bead event to the events log."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._record_event_sync, event)

    async def create(self, bead: Bead) -> str:
        """
        Create a new bead.

        Args:
            bead: The bead to create

        Returns:
            The bead ID

        Raises:
            ValueError: If bead ID already exists
        """
        async with self._lock:
            if bead.id in self._beads_cache:
                raise ValueError(f"Bead {bead.id} already exists")

            self._beads_cache[bead.id] = bead
            await self._append_bead(bead)

            # Record event
            event = BeadEvent(
                event_id=str(uuid.uuid4())[:8],
                bead_id=bead.id,
                event_type="created",
                timestamp=datetime.now(timezone.utc),
                new_status=bead.status,
                data={"title": bead.title, "type": bead.bead_type.value},
            )
            await self._record_event(event)

            if self.auto_commit:
                await self.commit_to_git(f"Created bead: {bead.title}")

            logger.debug(f"Created bead: {bead.id} ({bead.title})")
            return bead.id

    async def get(self, bead_id: str) -> Bead | None:
        """Get a bead by ID."""
        return self._beads_cache.get(bead_id)

    async def update(self, bead: Bead) -> None:
        """
        Update an existing bead.

        Args:
            bead: The bead with updated values

        Raises:
            ValueError: If bead doesn't exist
        """
        async with self._lock:
            if bead.id not in self._beads_cache:
                raise ValueError(f"Bead {bead.id} not found")

            old_bead = self._beads_cache[bead.id]
            bead.updated_at = datetime.now(timezone.utc)
            self._beads_cache[bead.id] = bead

            # Rewrite file (could optimize with seek for large files)
            await self._rewrite_file()

            # Record event if status changed
            if old_bead.status != bead.status:
                event = BeadEvent(
                    event_id=str(uuid.uuid4())[:8],
                    bead_id=bead.id,
                    event_type="status_changed",
                    timestamp=datetime.now(timezone.utc),
                    old_status=old_bead.status,
                    new_status=bead.status,
                    agent_id=bead.claimed_by,
                )
                await self._record_event(event)

            if self.auto_commit:
                await self.commit_to_git(f"Updated bead: {bead.title}")

    async def claim(self, bead_id: str, agent_id: str) -> bool:
        """
        Claim a bead for an agent.

        Args:
            bead_id: The bead to claim
            agent_id: The agent claiming the bead

        Returns:
            True if claimed successfully, False if already claimed
        """
        async with self._lock:
            bead = self._beads_cache.get(bead_id)
            if not bead:
                raise ValueError(f"Bead {bead_id} not found")

            if bead.status != BeadStatus.PENDING:
                return False

            bead.status = BeadStatus.CLAIMED
            bead.claimed_by = agent_id
            bead.claimed_at = datetime.now(timezone.utc)
            bead.updated_at = bead.claimed_at

            await self._rewrite_file()

            event = BeadEvent(
                event_id=str(uuid.uuid4())[:8],
                bead_id=bead_id,
                event_type="claimed",
                timestamp=bead.claimed_at,
                agent_id=agent_id,
                old_status=BeadStatus.PENDING,
                new_status=BeadStatus.CLAIMED,
            )
            await self._record_event(event)

            logger.debug(f"Bead {bead_id} claimed by {agent_id}")
            return True

    async def update_status(
        self,
        bead_id: str,
        status: BeadStatus,
        error_message: str | None = None,
    ) -> None:
        """
        Update the status of a bead.

        Args:
            bead_id: The bead to update
            status: The new status
            error_message: Optional error message (for FAILED status)
        """
        async with self._lock:
            bead = self._beads_cache.get(bead_id)
            if not bead:
                raise ValueError(f"Bead {bead_id} not found")

            old_status = bead.status
            bead.status = status
            bead.updated_at = datetime.now(timezone.utc)

            if status == BeadStatus.COMPLETED:
                bead.completed_at = bead.updated_at
            elif status == BeadStatus.FAILED:
                bead.error_message = error_message
                bead.attempt_count += 1
            elif status == BeadStatus.RUNNING:
                bead.attempt_count += 1

            await self._rewrite_file()

            event = BeadEvent(
                event_id=str(uuid.uuid4())[:8],
                bead_id=bead_id,
                event_type="status_changed",
                timestamp=bead.updated_at,
                agent_id=bead.claimed_by,
                old_status=old_status,
                new_status=status,
                data={"error_message": error_message} if error_message else {},
            )
            await self._record_event(event)

            if self.auto_commit:
                await self.commit_to_git(f"Bead {bead_id}: {old_status.value} -> {status.value}")

    async def list_by_status(self, status: BeadStatus) -> list[Bead]:
        """List all beads with a given status."""
        return [b for b in self._beads_cache.values() if b.status == status]

    async def list_by_agent(self, agent_id: str) -> list[Bead]:
        """List all beads claimed by an agent."""
        return [b for b in self._beads_cache.values() if b.claimed_by == agent_id]

    async def list_by_type(self, bead_type: BeadType) -> list[Bead]:
        """List all beads of a given type."""
        return [b for b in self._beads_cache.values() if b.bead_type == bead_type]

    async def list_beads(
        self,
        *,
        status: BeadStatus | None = None,
        priority: BeadPriority | None = None,
        limit: int | None = None,
    ) -> list[Bead]:
        """List beads with optional status/priority filters and a limit.

        Compatibility shim for callers that previously relied on a
        BeadManager.list_beads() API.
        """
        beads = list(self._beads_cache.values())
        if status is not None:
            beads = [b for b in beads if b.status == status]
        if priority is not None:
            beads = [b for b in beads if b.priority == priority]
        if limit is not None:
            beads = beads[:limit]
        return beads

    async def list_pending_runnable(self) -> list[Bead]:
        """List pending beads that can be started (dependencies met)."""
        completed_ids = {
            b.id for b in self._beads_cache.values() if b.status == BeadStatus.COMPLETED
        }
        return [
            b
            for b in self._beads_cache.values()
            if b.status == BeadStatus.PENDING and b.can_start(completed_ids)
        ]

    async def list_retryable(self) -> list[Bead]:
        """List failed beads that can be retried."""
        return [b for b in self._beads_cache.values() if b.can_retry()]

    async def list_all(self) -> list[Bead]:
        """List all beads."""
        return list(self._beads_cache.values())

    async def get_children(self, parent_id: str) -> list[Bead]:
        """Get all child beads of a parent."""
        return [b for b in self._beads_cache.values() if b.parent_id == parent_id]

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the bead store."""
        beads = list(self._beads_cache.values())
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for bead in beads:
            by_status[bead.status.value] = by_status.get(bead.status.value, 0) + 1
            by_type[bead.bead_type.value] = by_type.get(bead.bead_type.value, 0) + 1

        return {
            "total": len(beads),
            "by_status": by_status,
            "by_type": by_type,
            "agents_active": len({b.claimed_by for b in beads if b.claimed_by}),
        }

    async def commit_to_git(self, message: str) -> str | None:
        """
        Commit current state to git.

        Args:
            message: Commit message

        Returns:
            Commit hash if successful, None otherwise
        """
        if not self.git_enabled:
            return None

        try:
            async def _git(*args: str) -> tuple[int, str]:
                """Run git command with timeout and zombie prevention."""
                p = await asyncio.create_subprocess_exec(
                    "git", *args,
                    cwd=str(self.bead_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out, _ = await asyncio.wait_for(p.communicate(), timeout=15)
                except asyncio.TimeoutError:
                    p.kill()
                    await p.wait()
                    raise
                return p.returncode or 0, out.decode().strip() if out else ""

            # Add files
            await _git("add", "-A")

            # Check if there are changes to commit
            rc, _ = await _git("diff", "--cached", "--quiet")
            if rc == 0:
                logger.debug("No changes to commit")
                return None

            # Commit
            await _git("commit", "-m", message)

            # Get commit hash
            _, out = await _git("rev-parse", "HEAD")
            commit_hash = out[:8]

            logger.info(f"Committed beads: {commit_hash} - {message}")
            return commit_hash

        except OSError as e:
            logger.warning(f"Git commit failed: {e}")
            return None


# Convenience functions
async def create_bead_store(
    bead_dir: str | None = None,
    git_enabled: bool = True,
    auto_commit: bool = False,
) -> BeadStore:
    """Create and initialize a bead store."""
    if bead_dir is None:
        from aragora.nomic.stores.paths import resolve_store_dir

        bead_dir = str(resolve_store_dir())
    store = BeadStore(
        bead_dir=Path(bead_dir),
        git_enabled=git_enabled,
        auto_commit=auto_commit,
    )
    await store.initialize()
    return store


# Singleton store instance
_default_store: BeadStore | None = None


async def get_bead_store(bead_dir: str | None = None) -> BeadStore:
    """Get the default bead store instance."""
    global _default_store
    if _default_store is None:
        _default_store = await create_bead_store(bead_dir)
    return _default_store


def reset_bead_store() -> None:
    """Reset the default store (for testing)."""
    global _default_store
    _default_store = None


# Backwards-compatible spec export for legacy imports.
BeadSpec: Any = None
try:
    from aragora.nomic.stores.specs import BeadSpec  # noqa: F401
except (ImportError, AttributeError):  # pragma: no cover - best-effort compatibility
    pass

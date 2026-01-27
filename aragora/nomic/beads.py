"""
Beads: Git-Backed Atomic Work Units.

Inspired by Gastown's Beads pattern, this module provides persistent work tracking
that survives agent restarts. Beads are stored in JSONL format and can be backed
by git for durability and auditability.

Key concepts:
- Bead: An atomic work unit (task, issue, epic, or hook)
- BeadStore: JSONL-based persistence with optional git backing
- BeadType: Classification of work (ISSUE, TASK, EPIC, HOOK)
- BeadStatus: Lifecycle states (PENDING, CLAIMED, RUNNING, COMPLETED, FAILED)

Usage:
    store = BeadStore(Path(".beads"))
    await store.initialize()

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

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Safe pattern for bead identifiers (alphanumeric, hyphens, underscores)
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class BeadType(str, Enum):
    """Type of work unit."""

    ISSUE = "issue"  # Bug report or feature request
    TASK = "task"  # Actionable work item
    EPIC = "epic"  # Large work item containing subtasks
    HOOK = "hook"  # Special: per-agent work queue entry


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
    claimed_by: Optional[str] = None  # agent_id
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parent_id: Optional[str] = None  # For hierarchical beads
    dependencies: List[str] = field(default_factory=list)  # Bead IDs
    priority: BeadPriority = BeadPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3

    @classmethod
    def create(
        cls,
        bead_type: BeadType,
        title: str,
        description: str = "",
        parent_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        priority: BeadPriority = BeadPriority.NORMAL,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Bead":
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Bead":
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


@dataclass
class BeadEvent:
    """Event recording a bead state change."""

    event_id: str
    bead_id: str
    event_type: str  # created, claimed, started, completed, failed, etc.
    timestamp: datetime
    agent_id: Optional[str] = None
    old_status: Optional[BeadStatus] = None
    new_status: Optional[BeadStatus] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
        self._index: Dict[str, int] = {}  # bead_id -> line number
        self._beads_cache: Dict[str, Bead] = {}  # In-memory cache
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
                await proc.wait()
                logger.info(f"Initialized git repository in {self.bead_dir}")
            except Exception as e:
                logger.warning(f"Could not initialize git: {e}")
                self.git_enabled = False

    async def _load_beads(self) -> None:
        """Load all beads from JSONL file into cache."""
        if not self.bead_file.exists():
            return

        try:
            with open(self.bead_file) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        bead = Bead.from_dict(data)
                        self._beads_cache[bead.id] = bead
                        self._index[bead.id] = line_num
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid bead at line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Failed to load beads: {e}")

    async def _append_bead(self, bead: Bead) -> None:
        """Append a bead to the JSONL file."""
        with open(self.bead_file, "a") as f:
            f.write(json.dumps(bead.to_dict()) + "\n")

    async def _rewrite_file(self) -> None:
        """Rewrite the entire JSONL file from cache (for updates)."""
        temp_file = self.bead_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                for bead_id, bead in self._beads_cache.items():
                    f.write(json.dumps(bead.to_dict()) + "\n")
                    self._index[bead_id] = len(self._index)
            # Atomic rename
            temp_file.rename(self.bead_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    async def _record_event(self, event: BeadEvent) -> None:
        """Record a bead event to the events log."""
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

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

    async def get(self, bead_id: str) -> Optional[Bead]:
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
        error_message: Optional[str] = None,
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

    async def list_by_status(self, status: BeadStatus) -> List[Bead]:
        """List all beads with a given status."""
        return [b for b in self._beads_cache.values() if b.status == status]

    async def list_by_agent(self, agent_id: str) -> List[Bead]:
        """List all beads claimed by an agent."""
        return [b for b in self._beads_cache.values() if b.claimed_by == agent_id]

    async def list_by_type(self, bead_type: BeadType) -> List[Bead]:
        """List all beads of a given type."""
        return [b for b in self._beads_cache.values() if b.bead_type == bead_type]

    async def list_pending_runnable(self) -> List[Bead]:
        """List pending beads that can be started (dependencies met)."""
        completed_ids = {
            b.id for b in self._beads_cache.values() if b.status == BeadStatus.COMPLETED
        }
        return [
            b
            for b in self._beads_cache.values()
            if b.status == BeadStatus.PENDING and b.can_start(completed_ids)
        ]

    async def list_retryable(self) -> List[Bead]:
        """List failed beads that can be retried."""
        return [b for b in self._beads_cache.values() if b.can_retry()]

    async def list_all(self) -> List[Bead]:
        """List all beads."""
        return list(self._beads_cache.values())

    async def get_children(self, parent_id: str) -> List[Bead]:
        """Get all child beads of a parent."""
        return [b for b in self._beads_cache.values() if b.parent_id == parent_id]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the bead store."""
        beads = list(self._beads_cache.values())
        by_status = {}
        by_type = {}
        for bead in beads:
            by_status[bead.status.value] = by_status.get(bead.status.value, 0) + 1
            by_type[bead.bead_type.value] = by_type.get(bead.bead_type.value, 0) + 1

        return {
            "total": len(beads),
            "by_status": by_status,
            "by_type": by_type,
            "agents_active": len({b.claimed_by for b in beads if b.claimed_by}),
        }

    async def commit_to_git(self, message: str) -> Optional[str]:
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
            # Add files
            proc = await asyncio.create_subprocess_exec(
                "git",
                "add",
                "-A",
                cwd=str(self.bead_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

            # Check if there are changes to commit
            proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--cached",
                "--quiet",
                cwd=str(self.bead_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            returncode = await proc.wait()
            if returncode == 0:
                logger.debug("No changes to commit")
                return None

            # Commit
            proc = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                message,
                cwd=str(self.bead_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()

            # Get commit hash
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=str(self.bead_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            commit_hash = stdout.decode().strip()[:8]

            logger.info(f"Committed beads: {commit_hash} - {message}")
            return commit_hash

        except Exception as e:
            logger.warning(f"Git commit failed: {e}")
            return None


# Convenience functions
async def create_bead_store(
    bead_dir: str = ".beads",
    git_enabled: bool = True,
    auto_commit: bool = False,
) -> BeadStore:
    """Create and initialize a bead store."""
    store = BeadStore(
        bead_dir=Path(bead_dir),
        git_enabled=git_enabled,
        auto_commit=auto_commit,
    )
    await store.initialize()
    return store


# Singleton store instance
_default_store: Optional[BeadStore] = None


async def get_bead_store(bead_dir: str = ".beads") -> BeadStore:
    """Get the default bead store instance."""
    global _default_store
    if _default_store is None:
        _default_store = await create_bead_store(bead_dir)
    return _default_store


def reset_bead_store() -> None:
    """Reset the default store (for testing)."""
    global _default_store
    _default_store = None

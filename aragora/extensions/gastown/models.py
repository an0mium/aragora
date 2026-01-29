"""
Gastown Extension Data Models.

Defines the core data structures for developer workspace orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class ConvoyStatus(Enum):
    """Convoy lifecycle status.

    Gastown uses a richer lifecycle than workspace (which has
    CREATED/ASSIGNING/EXECUTING/MERGING/DONE/FAILED/CANCELLED).
    Both delegate to ``aragora.nomic.convoys`` as the canonical backend.
    Use ``to_workspace_status()`` / ``from_workspace_status()`` for
    cross-layer interoperability.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

    def to_workspace_status(self) -> str:
        """Map gastown status to workspace ConvoyStatus value.

        Returns the *value* string so callers can construct the workspace
        enum without importing it directly, avoiding circular imports.
        """
        return {
            ConvoyStatus.PENDING: "created",
            ConvoyStatus.IN_PROGRESS: "executing",
            ConvoyStatus.BLOCKED: "failed",
            ConvoyStatus.REVIEW: "merging",
            ConvoyStatus.COMPLETED: "done",
            ConvoyStatus.CANCELLED: "cancelled",
        }[self]

    @classmethod
    def from_workspace_status(cls, value: str) -> "ConvoyStatus":
        """Map a workspace ConvoyStatus value string to gastown status."""
        return {
            "created": cls.PENDING,
            "assigning": cls.PENDING,
            "executing": cls.IN_PROGRESS,
            "merging": cls.REVIEW,
            "done": cls.COMPLETED,
            "failed": cls.BLOCKED,
            "cancelled": cls.CANCELLED,
        }.get(value, cls.PENDING)


class HookType(Enum):
    """Types of git hooks for persistence."""

    PRE_COMMIT = "pre-commit"
    POST_COMMIT = "post-commit"
    PRE_PUSH = "pre-push"
    POST_CHECKOUT = "post-checkout"
    POST_MERGE = "post-merge"
    PREPARE_COMMIT_MSG = "prepare-commit-msg"


@dataclass
class WorkspaceConfig:
    """Configuration for a workspace."""

    name: str
    root_path: str
    description: str = ""
    max_rigs: int = 10
    max_agents_per_rig: int = 5
    default_model: str = "claude-3-opus"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """
    Root container for projects and agents.

    A workspace manages multiple rigs (repository contexts) and coordinates
    agent work across them.
    """

    id: str
    config: WorkspaceConfig
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    owner_id: str = ""
    tenant_id: str | None = None
    status: Literal["active", "archived", "suspended"] = "active"
    rigs: list[str] = field(default_factory=list)  # Rig IDs
    active_convoys: int = 0


@dataclass
class RigConfig:
    """Configuration for a rig (repository context)."""

    name: str
    repo_path: str
    branch: str = "main"
    worktree_path: str | None = None
    description: str = ""
    max_agents: int = 5
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rig:
    """
    Per-repository container with agent context.

    A rig represents a specific repository or worktree where agents can
    operate. It maintains context about the codebase and active work.
    """

    id: str
    workspace_id: str
    config: RigConfig
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: Literal["active", "paused", "archived"] = "active"
    agents: list[str] = field(default_factory=list)  # Agent IDs
    active_convoys: list[str] = field(default_factory=list)  # Convoy IDs
    last_sync: datetime | None = None


@dataclass
class ConvoyArtifact:
    """An artifact produced by a convoy."""

    id: str
    convoy_id: str
    type: str  # "file", "diff", "log", "report", "receipt"
    path: str
    content_hash: str = ""
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Convoy:
    """
    Work tracking unit with artifacts and handoffs.

    A convoy represents a unit of work that flows through the system,
    collecting artifacts and tracking state across agent handoffs.
    """

    id: str
    rig_id: str
    title: str
    description: str = ""
    status: ConvoyStatus = ConvoyStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Task tracking
    issue_ref: str | None = None  # External issue reference
    parent_convoy: str | None = None  # For sub-tasks
    depends_on: list[str] = field(default_factory=list)

    # Agent assignment
    assigned_agents: list[str] = field(default_factory=list)
    current_agent: str | None = None
    handoff_count: int = 0

    # Artifacts and results
    artifacts: list[str] = field(default_factory=list)  # Artifact IDs
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # Metadata
    priority: int = 0  # Higher = more important
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hook:
    """
    Git hook for persistent storage.

    Hooks store agent state in git worktrees, enabling persistence
    across sessions and disaster recovery.
    """

    id: str
    rig_id: str
    type: HookType
    path: str
    content: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: datetime | None = None
    trigger_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEntry:
    """
    Entry in the work ledger (Beads-style issue tracking).

    Tracks issues, tasks, and decisions with full provenance.
    """

    id: str
    workspace_id: str
    type: Literal["issue", "task", "decision", "note"]
    title: str
    body: str = ""
    status: Literal["open", "in_progress", "resolved", "closed"] = "open"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None

    # Relationships
    convoy_id: str | None = None
    parent_id: str | None = None
    related_entries: list[str] = field(default_factory=list)

    # Attribution
    created_by: str = ""  # Agent or user ID
    assigned_to: list[str] = field(default_factory=list)

    # Labels and metadata
    labels: list[str] = field(default_factory=list)
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

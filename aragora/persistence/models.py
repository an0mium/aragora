"""
Data models for persistent storage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Optional

from aragora.serialization import SerializableMixin


@dataclass
class NomicCycle(SerializableMixin):
    """A single nomic loop cycle."""

    loop_id: str
    cycle_number: int
    phase: str  # debate, design, implement, verify, commit
    stage: str  # proposing, critiquing, voting, executing, etc.
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    git_commit: Optional[str] = None
    task_description: Optional[str] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    error_message: Optional[str] = None
    id: Optional[str] = None  # Set by database

    # to_dict() inherited from SerializableMixin (handles datetime serialization)


@dataclass
class DebateArtifact(SerializableMixin):
    """A debate transcript and result."""

    _exclude_fields: ClassVar[tuple[str, ...]] = ("id",)

    loop_id: str
    cycle_number: int
    phase: str
    task: str
    agents: list[str]
    transcript: list[dict]  # Full message history
    consensus_reached: bool
    confidence: float
    winning_proposal: Optional[str] = None
    vote_tally: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: Optional[str] = None

    # to_dict() inherited from SerializableMixin (excludes id, handles datetime)


@dataclass
class StreamEvent(SerializableMixin):
    """A real-time event from the nomic loop."""

    _exclude_fields: ClassVar[tuple[str, ...]] = ("id",)

    loop_id: str
    cycle: int
    event_type: str  # cycle_start, phase_start, task_complete, error, etc.
    event_data: dict
    agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: Optional[str] = None

    # to_dict() inherited from SerializableMixin (excludes id, handles datetime)


@dataclass
class AgentMetrics(SerializableMixin):
    """Performance metrics for an agent in a cycle."""

    _exclude_fields: ClassVar[tuple[str, ...]] = ("id",)

    loop_id: str
    cycle: int
    agent_name: str
    model: str
    phase: str
    messages_sent: int = 0
    proposals_made: int = 0
    critiques_given: int = 0
    votes_won: int = 0
    votes_received: int = 0
    consensus_contributions: int = 0
    avg_response_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: Optional[str] = None

    # to_dict() inherited from SerializableMixin (excludes id, handles datetime)


@dataclass
class NomicRollback(SerializableMixin):
    """
    Records a rollback event in the nomic loop.

    Tracks what was rolled back, why, and where the failed work was preserved.
    Enables learning from failures and surfacing historical evolution.
    """

    id: str
    loop_id: str
    cycle_number: int
    phase: str  # debate, design, implement, verify, commit
    reason: str  # verify_failure, manual_intervention, timeout, conflict
    severity: str  # low, medium, high, critical
    rolled_back_commit: Optional[str] = None  # Git commit that was reverted
    preserved_branch: Optional[str] = None  # Where failed work was saved
    files_affected: list[str] = field(default_factory=list)
    diff_summary: str = ""  # git diff --stat summary
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # to_dict() inherited from SerializableMixin (handles datetime)


@dataclass
class CycleEvolution(SerializableMixin):
    """
    Tracks the evolution of the codebase through nomic cycles.

    Links debates to their outcomes, tracks what files changed,
    and connects to any rollbacks that occurred.
    """

    id: str
    loop_id: str
    cycle_number: int
    debate_artifact_id: Optional[str] = None  # Link to DebateArtifact
    winning_proposal_summary: Optional[str] = None
    files_changed: list[str] = field(default_factory=list)
    git_commit: Optional[str] = None
    rollback_id: Optional[str] = None  # If this cycle was rolled back
    created_at: datetime = field(default_factory=datetime.utcnow)

    # to_dict() inherited from SerializableMixin (handles datetime)


@dataclass
class CycleFileChange(SerializableMixin):
    """
    Tracks individual file changes within a cycle.

    Enables querying which cycles touched a specific file.
    """

    loop_id: str
    cycle_number: int
    file_path: str
    change_type: str  # added, modified, deleted, renamed
    insertions: int = 0
    deletions: int = 0

    # to_dict() inherited from SerializableMixin

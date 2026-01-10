"""
Data models for persistent storage.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
import json


@dataclass
class NomicCycle:
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            d['completed_at'] = self.completed_at.isoformat()
        return d


@dataclass
class DebateArtifact:
    """A debate transcript and result."""
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

    def to_dict(self) -> dict:
        return {
            'loop_id': self.loop_id,
            'cycle_number': self.cycle_number,
            'phase': self.phase,
            'task': self.task,
            'agents': self.agents,
            'transcript': self.transcript,
            'consensus_reached': self.consensus_reached,
            'confidence': self.confidence,
            'winning_proposal': self.winning_proposal,
            'vote_tally': self.vote_tally,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class StreamEvent:
    """A real-time event from the nomic loop."""
    loop_id: str
    cycle: int
    event_type: str  # cycle_start, phase_start, task_complete, error, etc.
    event_data: dict
    agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'loop_id': self.loop_id,
            'cycle': self.cycle,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'agent': self.agent,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class AgentMetrics:
    """Performance metrics for an agent in a cycle."""
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

    def to_dict(self) -> dict:
        return {
            'loop_id': self.loop_id,
            'cycle': self.cycle,
            'agent_name': self.agent_name,
            'model': self.model,
            'phase': self.phase,
            'messages_sent': self.messages_sent,
            'proposals_made': self.proposals_made,
            'critiques_given': self.critiques_given,
            'votes_won': self.votes_won,
            'votes_received': self.votes_received,
            'consensus_contributions': self.consensus_contributions,
            'avg_response_time_ms': self.avg_response_time_ms,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class NomicRollback:
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

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'loop_id': self.loop_id,
            'cycle_number': self.cycle_number,
            'phase': self.phase,
            'reason': self.reason,
            'severity': self.severity,
            'rolled_back_commit': self.rolled_back_commit,
            'preserved_branch': self.preserved_branch,
            'files_affected': self.files_affected,
            'diff_summary': self.diff_summary,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class CycleEvolution:
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

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'loop_id': self.loop_id,
            'cycle_number': self.cycle_number,
            'debate_artifact_id': self.debate_artifact_id,
            'winning_proposal_summary': self.winning_proposal_summary,
            'files_changed': self.files_changed,
            'git_commit': self.git_commit,
            'rollback_id': self.rollback_id,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class CycleFileChange:
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

    def to_dict(self) -> dict:
        return {
            'loop_id': self.loop_id,
            'cycle_number': self.cycle_number,
            'file_path': self.file_path,
            'change_type': self.change_type,
            'insertions': self.insertions,
            'deletions': self.deletions,
        }

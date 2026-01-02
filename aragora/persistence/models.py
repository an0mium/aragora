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

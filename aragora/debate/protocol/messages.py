"""
Protocol Message Types for Aragora Debates.

Defines typed, structured messages for debate protocol events.
Each message type has a specific payload schema for queryability and auditability.

Inspired by gastown's protocol message pattern (protocol/messages.go).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ProtocolMessageType(Enum):
    """Enumeration of all protocol message types in the debate lifecycle."""

    # Proposal lifecycle
    PROPOSAL_SUBMITTED = "proposal_submitted"
    PROPOSAL_REVISED = "proposal_revised"

    # Critique lifecycle
    CRITIQUE_SUBMITTED = "critique_submitted"
    REBUTTAL_SUBMITTED = "rebuttal_submitted"

    # Revision lifecycle
    REVISION_SUBMITTED = "revision_submitted"

    # Voting
    VOTE_CAST = "vote_cast"
    VOTE_CHANGED = "vote_changed"

    # Consensus
    CONSENSUS_REACHED = "consensus_reached"
    CONSENSUS_FAILED = "consensus_failed"

    # Round lifecycle
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"

    # Agent lifecycle
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"
    AGENT_FAILED = "agent_failed"
    AGENT_REPLACED = "agent_replaced"

    # Debate lifecycle
    DEBATE_STARTED = "debate_started"
    DEBATE_COMPLETED = "debate_completed"
    DEBATE_CANCELLED = "debate_cancelled"

    # Error/Recovery
    DEADLOCK_DETECTED = "deadlock_detected"
    RECOVERY_INITIATED = "recovery_initiated"
    RECOVERY_COMPLETED = "recovery_completed"


@dataclass
class ProtocolPayload:
    """Base payload for protocol messages."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolPayload":
        """Create payload from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProposalPayload(ProtocolPayload):
    """Payload for proposal-related messages."""

    proposal_id: str
    content: str
    model: str
    round_number: int
    token_count: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiquePayload(ProtocolPayload):
    """Payload for critique-related messages."""

    critique_id: str
    proposal_id: str
    content: str
    model: str
    round_number: int
    critique_type: str = "standard"  # standard, rebuttal, counter
    severity: Optional[str] = None  # minor, moderate, major
    addressed_issues: List[str] = field(default_factory=list)
    token_count: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass
class VotePayload(ProtocolPayload):
    """Payload for vote-related messages."""

    vote_id: str
    proposal_id: str
    vote_type: str  # support, oppose, abstain
    confidence: float  # 0.0 to 1.0
    rationale: Optional[str] = None
    weighted_score: Optional[float] = None
    is_human: bool = False


@dataclass
class ConsensusPayload(ProtocolPayload):
    """Payload for consensus-related messages."""

    consensus_id: str
    winning_proposal_id: Optional[str]
    final_answer: str
    confidence: float
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    rounds_taken: int = 0
    convergence_score: Optional[float] = None
    dissent_summary: Optional[str] = None


@dataclass
class RoundPayload(ProtocolPayload):
    """Payload for round lifecycle messages."""

    round_number: int
    phase: str  # proposal, critique, revision, voting
    proposal_count: int = 0
    critique_count: int = 0
    vote_count: int = 0
    duration_ms: Optional[float] = None
    convergence_delta: Optional[float] = None


@dataclass
class AgentEventPayload(ProtocolPayload):
    """Payload for agent lifecycle messages."""

    agent_id: str
    agent_name: str
    model: str
    role: str  # proposer, critic, voter, judge
    reason: Optional[str] = None  # For failures/replacements
    replacement_id: Optional[str] = None


# Union type for all payloads
PayloadType = Union[
    ProposalPayload,
    CritiquePayload,
    VotePayload,
    ConsensusPayload,
    RoundPayload,
    AgentEventPayload,
    ProtocolPayload,
]


@dataclass
class ProtocolMessage:
    """
    A typed protocol message representing a debate event.

    Protocol messages are:
    - Typed with specific message_type enum
    - Structured with typed payloads
    - Queryable by debate_id, agent_id, message_type, round
    - Serializable for persistence and audit trails
    - Traceable with correlation IDs
    """

    message_type: ProtocolMessageType
    debate_id: str
    agent_id: Optional[str] = None
    round_number: Optional[int] = None
    payload: Optional[PayloadType] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    parent_message_id: Optional[str] = None  # For chaining (e.g., critique -> proposal)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "debate_id": self.debate_id,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.round_number is not None:
            result["round_number"] = self.round_number
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.parent_message_id:
            result["parent_message_id"] = self.parent_message_id
        if self.metadata:
            result["metadata"] = self.metadata
        if self.payload:
            result["payload"] = (
                self.payload.to_dict() if hasattr(self.payload, "to_dict") else self.payload
            )

        return result

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolMessage":
        """Create message from dictionary."""
        message_type = ProtocolMessageType(data["message_type"])
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=message_type,
            debate_id=data["debate_id"],
            agent_id=data.get("agent_id"),
            round_number=data.get("round_number"),
            timestamp=timestamp or datetime.now(timezone.utc),
            correlation_id=data.get("correlation_id"),
            parent_message_id=data.get("parent_message_id"),
            metadata=data.get("metadata", {}),
            payload=data.get("payload"),  # Raw dict, caller can parse
        )

    def __repr__(self) -> str:
        return (
            f"ProtocolMessage("
            f"type={self.message_type.value}, "
            f"debate={self.debate_id[:8]}..., "
            f"agent={self.agent_id or 'N/A'}, "
            f"round={self.round_number}"
            f")"
        )


# Factory functions for common message types


def proposal_message(
    debate_id: str,
    agent_id: str,
    proposal_id: str,
    content: str,
    model: str,
    round_number: int,
    **kwargs,
) -> ProtocolMessage:
    """Create a PROPOSAL_SUBMITTED message."""
    return ProtocolMessage(
        message_type=ProtocolMessageType.PROPOSAL_SUBMITTED,
        debate_id=debate_id,
        agent_id=agent_id,
        round_number=round_number,
        payload=ProposalPayload(
            proposal_id=proposal_id,
            content=content,
            model=model,
            round_number=round_number,
            **kwargs,
        ),
    )


def critique_message(
    debate_id: str,
    agent_id: str,
    critique_id: str,
    proposal_id: str,
    content: str,
    model: str,
    round_number: int,
    **kwargs,
) -> ProtocolMessage:
    """Create a CRITIQUE_SUBMITTED message."""
    return ProtocolMessage(
        message_type=ProtocolMessageType.CRITIQUE_SUBMITTED,
        debate_id=debate_id,
        agent_id=agent_id,
        round_number=round_number,
        parent_message_id=proposal_id,  # Links critique to proposal
        payload=CritiquePayload(
            critique_id=critique_id,
            proposal_id=proposal_id,
            content=content,
            model=model,
            round_number=round_number,
            **kwargs,
        ),
    )


def vote_message(
    debate_id: str,
    agent_id: str,
    vote_id: str,
    proposal_id: str,
    vote_type: str,
    confidence: float,
    **kwargs,
) -> ProtocolMessage:
    """Create a VOTE_CAST message."""
    return ProtocolMessage(
        message_type=ProtocolMessageType.VOTE_CAST,
        debate_id=debate_id,
        agent_id=agent_id,
        parent_message_id=proposal_id,
        payload=VotePayload(
            vote_id=vote_id,
            proposal_id=proposal_id,
            vote_type=vote_type,
            confidence=confidence,
            **kwargs,
        ),
    )


def consensus_message(
    debate_id: str,
    consensus_id: str,
    winning_proposal_id: Optional[str],
    final_answer: str,
    confidence: float,
    rounds_taken: int,
    **kwargs,
) -> ProtocolMessage:
    """Create a CONSENSUS_REACHED message."""
    return ProtocolMessage(
        message_type=ProtocolMessageType.CONSENSUS_REACHED,
        debate_id=debate_id,
        payload=ConsensusPayload(
            consensus_id=consensus_id,
            winning_proposal_id=winning_proposal_id,
            final_answer=final_answer,
            confidence=confidence,
            rounds_taken=rounds_taken,
            **kwargs,
        ),
    )


def round_message(
    debate_id: str,
    round_number: int,
    phase: str,
    started: bool = True,
    **kwargs,
) -> ProtocolMessage:
    """Create a ROUND_STARTED or ROUND_COMPLETED message."""
    return ProtocolMessage(
        message_type=(
            ProtocolMessageType.ROUND_STARTED if started else ProtocolMessageType.ROUND_COMPLETED
        ),
        debate_id=debate_id,
        round_number=round_number,
        payload=RoundPayload(
            round_number=round_number,
            phase=phase,
            **kwargs,
        ),
    )


def agent_event_message(
    debate_id: str,
    agent_id: str,
    agent_name: str,
    model: str,
    role: str,
    event_type: ProtocolMessageType,
    **kwargs,
) -> ProtocolMessage:
    """Create an agent lifecycle message (JOINED, LEFT, FAILED, REPLACED)."""
    return ProtocolMessage(
        message_type=event_type,
        debate_id=debate_id,
        agent_id=agent_id,
        payload=AgentEventPayload(
            agent_id=agent_id,
            agent_name=agent_name,
            model=model,
            role=role,
            **kwargs,
        ),
    )

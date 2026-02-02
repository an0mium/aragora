"""
Aragora Typed WebSocket Events

Typed dataclasses for WebSocket event payloads, enabling structured access
to event data instead of raw dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectedEvent:
    """Fired when the WebSocket connection is established."""

    server_version: str = ""


@dataclass
class DisconnectedEvent:
    """Fired when the WebSocket connection is lost."""

    code: int = 1000
    reason: str = ""


@dataclass
class ErrorEvent:
    """Fired when an error occurs."""

    error: str = ""
    code: str = ""


@dataclass
class DebateStartEvent:
    """Fired when a debate begins."""

    debate_id: str = ""
    task: str = ""
    agents: list[str] = field(default_factory=list)
    total_rounds: int = 0
    protocol: str = ""


@dataclass
class RoundStartEvent:
    """Fired at the beginning of each debate round."""

    debate_id: str = ""
    round_number: int = 0
    total_rounds: int = 0


@dataclass
class AgentMessageEvent:
    """Fired when an agent produces a message."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    content: str = ""
    confidence: float | None = None
    role: str = ""


@dataclass
class ProposeEvent:
    """Fired when an agent makes a proposal."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    content: str = ""
    confidence: float | None = None


@dataclass
class CritiqueEvent:
    """Fired when an agent critiques a proposal."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    target_agent: str = ""
    content: str = ""
    score: float | None = None


@dataclass
class RevisionEvent:
    """Fired when an agent revises their position."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    content: str = ""
    original_content: str = ""
    confidence: float | None = None


@dataclass
class SynthesisEvent:
    """Fired when a synthesis of positions is produced."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    content: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class VoteEvent:
    """Fired when an agent casts a vote."""

    debate_id: str = ""
    round_number: int = 0
    agent: str = ""
    choice: str = ""
    reasoning: str = ""
    confidence: float | None = None


@dataclass
class ConsensusEvent:
    """Fired when consensus status is updated."""

    debate_id: str = ""
    round_number: int = 0
    result: str = ""
    confidence: float | None = None
    method: str = ""


@dataclass
class ConsensusReachedEvent:
    """Fired when full consensus is achieved."""

    debate_id: str = ""
    result: str = ""
    confidence: float | None = None
    final_round: int = 0


@dataclass
class DebateEndEvent:
    """Fired when a debate concludes."""

    debate_id: str = ""
    result: str = ""
    total_rounds: int = 0
    consensus_reached: bool = False
    duration_seconds: float | None = None


@dataclass
class PhaseChangeEvent:
    """Fired when the debate transitions between phases."""

    debate_id: str = ""
    from_phase: str = ""
    to_phase: str = ""
    round_number: int = 0


@dataclass
class AudienceSuggestionEvent:
    """Fired when an audience member submits a suggestion."""

    debate_id: str = ""
    user_id: str = ""
    content: str = ""
    round_number: int = 0


@dataclass
class UserVoteEvent:
    """Fired when a user casts a vote on the debate."""

    debate_id: str = ""
    user_id: str = ""
    choice: str = ""
    round_number: int = 0


@dataclass
class WarningEvent:
    """Fired for non-fatal warnings during the debate."""

    debate_id: str = ""
    message: str = ""
    severity: str = "warning"


@dataclass
class MessageEvent:
    """Generic message event for untyped messages."""

    content: str = ""
    sender: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Mapping from event type string to typed dataclass
EVENT_CLASS_MAP: dict[str, type] = {
    "connected": ConnectedEvent,
    "disconnected": DisconnectedEvent,
    "error": ErrorEvent,
    "debate_start": DebateStartEvent,
    "round_start": RoundStartEvent,
    "agent_message": AgentMessageEvent,
    "propose": ProposeEvent,
    "critique": CritiqueEvent,
    "revision": RevisionEvent,
    "synthesis": SynthesisEvent,
    "vote": VoteEvent,
    "consensus": ConsensusEvent,
    "consensus_reached": ConsensusReachedEvent,
    "debate_end": DebateEndEvent,
    "phase_change": PhaseChangeEvent,
    "audience_suggestion": AudienceSuggestionEvent,
    "user_vote": UserVoteEvent,
    "warning": WarningEvent,
    "message": MessageEvent,
}

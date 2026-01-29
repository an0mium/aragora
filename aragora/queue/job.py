"""
Debate job definitions.

Provides specific job types for debate processing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from aragora.config import DEFAULT_AGENT_LIST, DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.queue.base import Job
from aragora.serialization import SerializableMixin

@dataclass
class DebateJobPayload(SerializableMixin):
    """
    Payload for a debate job.

    Contains all the information needed to run a debate.
    """

    question: str
    agents: list[str] = field(default_factory=lambda: list(DEFAULT_AGENT_LIST))
    rounds: int = DEFAULT_ROUNDS
    consensus: str = DEFAULT_CONSENSUS
    protocol: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional configuration
    timeout_seconds: int | None = None
    webhook_url: str | None = None
    user_id: str | None = None
    organization_id: str | None = None

    # to_dict() and from_dict() inherited from SerializableMixin

def create_debate_job(
    question: str,
    agents: Optional[list[str]] = None,
    rounds: int = DEFAULT_ROUNDS,
    consensus: str = DEFAULT_CONSENSUS,
    protocol: str = "standard",
    priority: int = 0,
    max_attempts: int = 3,
    timeout_seconds: int | None = None,
    webhook_url: str | None = None,
    user_id: str | None = None,
    organization_id: str | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Job:
    """
    Create a new debate job.

    Args:
        question: The debate question/topic
        agents: List of agent names to participate
        rounds: Number of debate rounds
        consensus: Consensus method (majority, unanimous, etc.)
        protocol: Debate protocol (standard, adversarial, etc.)
        priority: Job priority (higher = more important)
        max_attempts: Maximum retry attempts
        timeout_seconds: Debate timeout
        webhook_url: URL to call on completion
        user_id: User who submitted the debate
        organization_id: Organization for billing
        metadata: Additional metadata

    Returns:
        A Job instance ready to be enqueued
    """
    payload = DebateJobPayload(
        question=question,
        agents=agents or list(DEFAULT_AGENT_LIST),
        rounds=rounds,
        consensus=consensus,
        protocol=protocol,
        timeout_seconds=timeout_seconds,
        webhook_url=webhook_url,
        user_id=user_id,
        organization_id=organization_id,
        metadata=metadata or {},
    )

    return Job(
        id=str(uuid.uuid4()),
        payload=payload.to_dict(),
        priority=priority,
        max_attempts=max_attempts,
        metadata={
            "job_type": "debate",
            "question_preview": question[:100] if len(question) > 100 else question,
        },
    )

def get_debate_payload(job: Job) -> DebateJobPayload:
    """
    Extract the debate payload from a job.

    Args:
        job: The job to extract from

    Returns:
        The DebateJobPayload
    """
    return DebateJobPayload.from_dict(job.payload)

@dataclass
class DebateResult:
    """
    Result of a debate job.

    Contains the outcome and metrics from the debate.
    """

    debate_id: str
    consensus_reached: bool
    final_answer: str | None
    confidence: float
    rounds_used: int
    participants: list[str]
    duration_seconds: float
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_id": self.debate_id,
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "rounds_used": self.rounds_used,
            "participants": self.participants,
            "duration_seconds": self.duration_seconds,
            "token_usage": self.token_usage,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateResult":
        """Create from dictionary."""
        return cls(
            debate_id=data["debate_id"],
            consensus_reached=data["consensus_reached"],
            final_answer=data.get("final_answer"),
            confidence=data.get("confidence", 0.0),
            rounds_used=data.get("rounds_used", 0),
            participants=data.get("participants", []),
            duration_seconds=data.get("duration_seconds", 0.0),
            token_usage=data.get("token_usage", {}),
            error=data.get("error"),
        )

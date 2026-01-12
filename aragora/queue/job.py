"""
Debate job definitions.

Provides specific job types for debate processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from aragora.queue.base import Job, JobStatus


@dataclass
class DebateJobPayload:
    """
    Payload for a debate job.

    Contains all the information needed to run a debate.
    """

    question: str
    agents: List[str] = field(default_factory=list)
    rounds: int = 3
    consensus: str = "majority"
    protocol: str = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional configuration
    timeout_seconds: Optional[int] = None
    webhook_url: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "agents": self.agents,
            "rounds": self.rounds,
            "consensus": self.consensus,
            "protocol": self.protocol,
            "metadata": self.metadata,
            "timeout_seconds": self.timeout_seconds,
            "webhook_url": self.webhook_url,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebateJobPayload":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            agents=data.get("agents", []),
            rounds=data.get("rounds", 3),
            consensus=data.get("consensus", "majority"),
            protocol=data.get("protocol", "standard"),
            metadata=data.get("metadata", {}),
            timeout_seconds=data.get("timeout_seconds"),
            webhook_url=data.get("webhook_url"),
            user_id=data.get("user_id"),
            organization_id=data.get("organization_id"),
        )


def create_debate_job(
    question: str,
    agents: Optional[List[str]] = None,
    rounds: int = 3,
    consensus: str = "majority",
    protocol: str = "standard",
    priority: int = 0,
    max_attempts: int = 3,
    timeout_seconds: Optional[int] = None,
    webhook_url: Optional[str] = None,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
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
        agents=agents or [],
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
    final_answer: Optional[str]
    confidence: float
    rounds_used: int
    participants: List[str]
    duration_seconds: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "DebateResult":
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

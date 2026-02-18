"""
Data models for governance store records.

Defines the dataclasses used across both SQLite and PostgreSQL
governance store backends:
- ApprovalRecord: Human-in-the-loop approval requests
- VerificationRecord: Formal verification results
- DecisionRecord: Debate outcomes with provenance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ApprovalRecord:
    """Persistent approval request record."""

    approval_id: str
    title: str
    description: str
    risk_level: str  # low, medium, high, critical
    status: str  # pending, approved, rejected, expired, cancelled
    requested_by: str
    requested_at: datetime
    changes_json: str  # JSON serialized changes
    timeout_seconds: int = 3600
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str | None = None
    org_id: str | None = None
    workspace_id: str | None = None
    metadata_json: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "approval_id": self.approval_id,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level,
            "status": self.status,
            "requested_by": self.requested_by,
            "requested_at": (
                self.requested_at.isoformat()
                if isinstance(self.requested_at, datetime)
                else self.requested_at
            ),
            "changes": json.loads(self.changes_json) if self.changes_json else [],
            "timeout_seconds": self.timeout_seconds,
            "approved_by": self.approved_by,
            "approved_at": (
                self.approved_at.isoformat()
                if isinstance(self.approved_at, datetime)
                else self.approved_at
            ),
            "rejection_reason": self.rejection_reason,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


@dataclass
class VerificationRecord:
    """Persistent verification history entry."""

    verification_id: str
    claim: str
    claim_type: str | None
    context: str
    result_json: str  # JSON serialized result
    timestamp: datetime
    verified_by: str  # system, agent name, etc.
    confidence: float = 0.0
    proof_tree_json: str | None = None
    org_id: str | None = None
    workspace_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "verification_id": self.verification_id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "context": self.context,
            "result": json.loads(self.result_json) if self.result_json else {},
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "verified_by": self.verified_by,
            "confidence": self.confidence,
            "proof_tree": json.loads(self.proof_tree_json) if self.proof_tree_json else None,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
        }


@dataclass
class DecisionRecord:
    """Persistent decision outcome record."""

    decision_id: str
    debate_id: str
    conclusion: str
    consensus_reached: bool
    confidence: float
    timestamp: datetime
    evidence_chain_json: str  # JSON serialized
    vote_pivots_json: str  # JSON serialized
    belief_changes_json: str  # JSON serialized
    agents_involved_json: str  # JSON serialized list
    org_id: str | None = None
    workspace_id: str | None = None
    metadata_json: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "debate_id": self.debate_id,
            "conclusion": self.conclusion,
            "consensus_reached": self.consensus_reached,
            "confidence": self.confidence,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
            "evidence_chain": (
                json.loads(self.evidence_chain_json) if self.evidence_chain_json else []
            ),
            "vote_pivots": json.loads(self.vote_pivots_json) if self.vote_pivots_json else [],
            "belief_changes": (
                json.loads(self.belief_changes_json) if self.belief_changes_json else []
            ),
            "agents_involved": (
                json.loads(self.agents_involved_json) if self.agents_involved_json else []
            ),
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


@dataclass
class OutcomeRecord:
    """Persistent outcome tracking record for decision follow-up.

    Links a measured real-world outcome back to the decision that produced it,
    enabling closed-loop learning: decision -> action -> outcome -> next decision.
    """

    outcome_id: str
    decision_id: str
    debate_id: str
    outcome_type: str  # success, failure, partial, unknown
    outcome_description: str
    impact_score: float  # 0.0 - 1.0
    measured_at: datetime
    kpis_before_json: str  # JSON serialized dict[str, Any]
    kpis_after_json: str  # JSON serialized dict[str, Any]
    lessons_learned: str = ""
    tags_json: str = "[]"  # JSON serialized list[str]
    org_id: str | None = None
    workspace_id: str | None = None
    metadata_json: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "outcome_id": self.outcome_id,
            "decision_id": self.decision_id,
            "debate_id": self.debate_id,
            "outcome_type": self.outcome_type,
            "outcome_description": self.outcome_description,
            "impact_score": self.impact_score,
            "measured_at": (
                self.measured_at.isoformat()
                if isinstance(self.measured_at, datetime)
                else self.measured_at
            ),
            "kpis_before": json.loads(self.kpis_before_json) if self.kpis_before_json else {},
            "kpis_after": json.loads(self.kpis_after_json) if self.kpis_after_json else {},
            "lessons_learned": self.lessons_learned,
            "tags": json.loads(self.tags_json) if self.tags_json else [],
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


__all__ = [
    "ApprovalRecord",
    "VerificationRecord",
    "DecisionRecord",
    "OutcomeRecord",
]

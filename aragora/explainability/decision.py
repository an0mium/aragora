"""
Decision entity for explainability.

The Decision class aggregates all explainability data about a debate outcome,
providing a unified interface for understanding why and how a decision was made.

Features:
- Evidence chains: Links from conclusion to supporting evidence
- Vote pivots: Which votes were most influential
- Belief changes: How agent beliefs evolved during debate
- Confidence attribution: What factors contributed to confidence
- Counterfactual analysis: What would change the outcome
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InfluenceType(str, Enum):
    """Types of influence on a decision."""

    EVIDENCE = "evidence"
    VOTE = "vote"
    ARGUMENT = "argument"
    CALIBRATION = "calibration"
    ELO = "elo"
    CONSENSUS = "consensus"
    USER = "user"


@dataclass
class EvidenceLink:
    """A link in the evidence chain supporting a decision."""

    id: str
    content: str
    source: str  # Agent name, user, or external source
    relevance_score: float  # 0-1
    quality_scores: Dict[str, float] = field(default_factory=dict)
    cited_by: List[str] = field(default_factory=list)  # Agent names
    grounding_type: str = "claim"  # claim, fact, opinion, citation
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "quality_scores": self.quality_scores,
            "cited_by": self.cited_by,
            "grounding_type": self.grounding_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class VotePivot:
    """A vote that significantly influenced the outcome."""

    agent: str
    choice: str
    confidence: float
    weight: float  # Computed weight (ELO + calibration)
    reasoning_summary: str
    influence_score: float  # How much this vote affected the outcome
    calibration_adjustment: Optional[float] = None
    elo_rating: Optional[float] = None
    flip_detected: bool = False  # Did this agent flip position?
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "choice": self.choice,
            "confidence": self.confidence,
            "weight": self.weight,
            "reasoning_summary": self.reasoning_summary,
            "influence_score": self.influence_score,
            "calibration_adjustment": self.calibration_adjustment,
            "elo_rating": self.elo_rating,
            "flip_detected": self.flip_detected,
            "metadata": self.metadata,
        }


@dataclass
class BeliefChange:
    """A change in an agent's beliefs during debate."""

    agent: str
    round: int
    topic: str
    prior_belief: str
    posterior_belief: str
    prior_confidence: float
    posterior_confidence: float
    trigger: str  # What caused the change (argument, evidence, etc.)
    trigger_source: str  # Who/what provided the trigger
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_delta(self) -> float:
        return self.posterior_confidence - self.prior_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "round": self.round,
            "topic": self.topic,
            "prior_belief": self.prior_belief,
            "posterior_belief": self.posterior_belief,
            "prior_confidence": self.prior_confidence,
            "posterior_confidence": self.posterior_confidence,
            "confidence_delta": self.confidence_delta,
            "trigger": self.trigger,
            "trigger_source": self.trigger_source,
            "metadata": self.metadata,
        }


@dataclass
class ConfidenceAttribution:
    """Attribution of confidence to different factors."""

    factor: str  # consensus_strength, evidence_quality, agent_agreement, etc.
    contribution: float  # 0-1, contribution to final confidence
    explanation: str
    raw_value: Optional[float] = None  # Underlying metric value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "contribution": self.contribution,
            "explanation": self.explanation,
            "raw_value": self.raw_value,
            "metadata": self.metadata,
        }


@dataclass
class Counterfactual:
    """A counterfactual analysis of the decision."""

    condition: str  # What would need to change
    outcome_change: str  # What the new outcome would be
    likelihood: float  # How likely is this counterfactual (0-1)
    sensitivity: float  # How sensitive is outcome to this factor (0-1)
    affected_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "outcome_change": self.outcome_change,
            "likelihood": self.likelihood,
            "sensitivity": self.sensitivity,
            "affected_agents": self.affected_agents,
            "metadata": self.metadata,
        }


@dataclass
class Decision:
    """
    Unified decision entity for explainability.

    Aggregates all information needed to explain why and how
    a debate reached its conclusion.
    """

    # Identity
    decision_id: str
    debate_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Outcome
    conclusion: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    consensus_type: str = "majority"  # majority, supermajority, unanimous

    # Task context
    task: str = ""
    domain: str = "general"
    rounds_used: int = 0
    agents_participated: List[str] = field(default_factory=list)

    # Explainability components
    evidence_chain: List[EvidenceLink] = field(default_factory=list)
    vote_pivots: List[VotePivot] = field(default_factory=list)
    belief_changes: List[BeliefChange] = field(default_factory=list)
    confidence_attribution: List[ConfidenceAttribution] = field(default_factory=list)
    counterfactuals: List[Counterfactual] = field(default_factory=list)

    # Summary metrics
    evidence_quality_score: float = 0.0
    agent_agreement_score: float = 0.0
    belief_stability_score: float = 0.0  # How stable were beliefs (1 = no changes)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique decision ID."""
        content = f"{self.debate_id}:{self.timestamp}:{self.conclusion[:50]}"
        return f"dec-{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Export decision to dictionary."""
        return {
            "decision_id": self.decision_id,
            "debate_id": self.debate_id,
            "timestamp": self.timestamp,
            "conclusion": self.conclusion,
            "consensus_reached": self.consensus_reached,
            "confidence": self.confidence,
            "consensus_type": self.consensus_type,
            "task": self.task,
            "domain": self.domain,
            "rounds_used": self.rounds_used,
            "agents_participated": self.agents_participated,
            "evidence_chain": [e.to_dict() for e in self.evidence_chain],
            "vote_pivots": [v.to_dict() for v in self.vote_pivots],
            "belief_changes": [b.to_dict() for b in self.belief_changes],
            "confidence_attribution": [c.to_dict() for c in self.confidence_attribution],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "evidence_quality_score": self.evidence_quality_score,
            "agent_agreement_score": self.agent_agreement_score,
            "belief_stability_score": self.belief_stability_score,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export decision to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        """Create decision from dictionary."""
        return cls(
            decision_id=data.get("decision_id", ""),
            debate_id=data.get("debate_id", ""),
            timestamp=data.get("timestamp", ""),
            conclusion=data.get("conclusion", ""),
            consensus_reached=data.get("consensus_reached", False),
            confidence=data.get("confidence", 0.0),
            consensus_type=data.get("consensus_type", "majority"),
            task=data.get("task", ""),
            domain=data.get("domain", "general"),
            rounds_used=data.get("rounds_used", 0),
            agents_participated=data.get("agents_participated", []),
            evidence_chain=[
                EvidenceLink(**e) for e in data.get("evidence_chain", [])
            ],
            vote_pivots=[VotePivot(**v) for v in data.get("vote_pivots", [])],
            belief_changes=[
                BeliefChange(**b) for b in data.get("belief_changes", [])
            ],
            confidence_attribution=[
                ConfidenceAttribution(**c)
                for c in data.get("confidence_attribution", [])
            ],
            counterfactuals=[
                Counterfactual(**c) for c in data.get("counterfactuals", [])
            ],
            evidence_quality_score=data.get("evidence_quality_score", 0.0),
            agent_agreement_score=data.get("agent_agreement_score", 0.0),
            belief_stability_score=data.get("belief_stability_score", 0.0),
            metadata=data.get("metadata", {}),
        )

    # ==========================================================================
    # Query Methods
    # ==========================================================================

    def get_top_evidence(self, n: int = 5) -> List[EvidenceLink]:
        """Get top N evidence items by relevance."""
        return sorted(
            self.evidence_chain, key=lambda e: e.relevance_score, reverse=True
        )[:n]

    def get_pivotal_votes(self, threshold: float = 0.3) -> List[VotePivot]:
        """Get votes with influence above threshold."""
        return [v for v in self.vote_pivots if v.influence_score >= threshold]

    def get_significant_belief_changes(self, min_delta: float = 0.2) -> List[BeliefChange]:
        """Get belief changes with significant confidence delta."""
        return [
            b for b in self.belief_changes if abs(b.confidence_delta) >= min_delta
        ]

    def get_major_confidence_factors(self, threshold: float = 0.1) -> List[ConfidenceAttribution]:
        """Get confidence factors above contribution threshold."""
        return [
            c for c in self.confidence_attribution if c.contribution >= threshold
        ]

    def get_high_sensitivity_counterfactuals(
        self, threshold: float = 0.5
    ) -> List[Counterfactual]:
        """Get counterfactuals with high sensitivity."""
        return [c for c in self.counterfactuals if c.sensitivity >= threshold]


__all__ = [
    "Decision",
    "EvidenceLink",
    "VotePivot",
    "BeliefChange",
    "ConfidenceAttribution",
    "Counterfactual",
    "InfluenceType",
]

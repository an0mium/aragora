"""Core data types for adversarial multi-model debate.

All types use only the Python standard library — no external dependencies required.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    """Phases of a structured adversarial debate."""
    PROPOSE = "propose"
    CRITIQUE = "critique"
    REVISE = "revise"
    VOTE = "vote"
    CONSENSUS = "consensus"


class ConsensusMethod(str, Enum):
    """How consensus is determined."""
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    UNANIMOUS = "unanimous"
    JUDGE = "judge"
    WEIGHTED = "weighted"


class Verdict(str, Enum):
    """Canonical verdict for decision receipts."""
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Communication types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A message exchanged during a debate."""
    role: str
    agent: str
    content: str
    round: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "agent": self.agent,
            "content": self.content,
            "round": self.round,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Proposal:
    """A proposal submitted by an agent during the propose phase."""
    agent: str
    content: str
    round: int = 0
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Critique:
    """A critique of another agent's proposal."""
    agent: str
    target_agent: str
    target_content: str
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    severity: float = 5.0  # 0 = trivial, 10 = critical
    reasoning: str = ""

    @property
    def content(self) -> str:
        parts = []
        if self.issues:
            parts.append("Issues:\n" + "\n".join(f"- {i}" for i in self.issues))
        if self.suggestions:
            parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in self.suggestions))
        if self.reasoning:
            parts.append(f"Reasoning: {self.reasoning}")
        return "\n\n".join(parts) if parts else "(no critique)"


@dataclass
class Vote:
    """An agent's vote for a proposal during the voting phase."""
    agent: str
    choice: str  # which agent/proposal they endorse
    reasoning: str = ""
    confidence: float = 1.0  # 0-1, used as weight in weighted consensus


# ---------------------------------------------------------------------------
# Evidence & claims
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a claim."""
    source: str
    content: str
    evidence_type: str = "argument"  # argument, data, citation, tool_output
    supports_claim: bool = True
    strength: float = 0.5  # 0-1
    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    """A structured claim with supporting and refuting evidence."""
    statement: str
    author: str
    confidence: float = 0.5  # 0-1
    supporting_evidence: list[Evidence] = field(default_factory=list)
    refuting_evidence: list[Evidence] = field(default_factory=list)
    status: str = "active"  # active, revised, withdrawn, merged
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    round_introduced: int = 0

    @property
    def net_evidence_strength(self) -> float:
        support = sum(e.strength for e in self.supporting_evidence)
        refute = sum(e.strength for e in self.refuting_evidence)
        total = support + refute
        return (support / total) if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Consensus & dissent
# ---------------------------------------------------------------------------

@dataclass
class DissentRecord:
    """Record of an agent's dissent from the consensus."""
    agent: str
    reasons: list[str]
    alternative_view: str | None = None
    severity: float = 0.5  # 0-1
    claim_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "reasons": self.reasons,
            "alternative_view": self.alternative_view,
            "severity": self.severity,
        }


@dataclass
class Consensus:
    """The consensus state after voting."""
    reached: bool
    method: ConsensusMethod
    confidence: float  # 0-1
    supporting_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    dissents: list[DissentRecord] = field(default_factory=list)
    statement: str = ""

    @property
    def agreement_ratio(self) -> float:
        total = len(self.supporting_agents) + len(self.dissenting_agents)
        return len(self.supporting_agents) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Response returned from an agent's generate/critique/vote call."""
    content: str
    agent: str
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Abstract base class for debate agents.

    Subclass this and implement the three async methods to plug in any LLM
    provider (Anthropic, OpenAI, Mistral, local models, etc.).
    """

    def __init__(
        self,
        name: str,
        model: str = "",
        *,
        system_prompt: str = "",
        stance: Literal["affirmative", "negative", "neutral"] = "neutral",
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.stance = stance

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: list[Message] | None = None,
    ) -> str:
        """Generate a proposal or response to a prompt."""
        ...

    @abstractmethod
    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Critique another agent's proposal."""
        ...

    @abstractmethod
    async def vote(
        self,
        proposals: dict[str, str],
        task: str,
    ) -> Vote:
        """Vote on which proposal is strongest."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    """Configuration for a debate session."""
    rounds: int = 3
    consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY
    consensus_threshold: float = 0.6
    early_stopping: bool = True
    early_stop_threshold: float = 0.85
    min_rounds: int = 1
    timeout_seconds: int = 0  # 0 = no timeout
    require_reasoning: bool = True


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DecisionReceipt:
    """Cryptographic audit trail for a debate decision.

    See ``ReceiptBuilder`` to construct these from a ``DebateResult``.
    """
    receipt_id: str
    question: str
    verdict: Verdict
    confidence: float
    consensus: Consensus
    agents: list[str]
    rounds_used: int
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str | None = None
    signature_algorithm: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "question": self.question,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "consensus": {
                "reached": self.consensus.reached,
                "method": self.consensus.method.value,
                "confidence": self.consensus.confidence,
                "supporting_agents": self.consensus.supporting_agents,
                "dissenting_agents": self.consensus.dissenting_agents,
                "dissents": [d.to_dict() for d in self.consensus.dissents],
            },
            "agents": self.agents,
            "rounds_used": self.rounds_used,
            "claims": len(self.claims),
            "evidence_count": len(self.evidence),
            "timestamp": self.timestamp,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
        }

    def to_markdown(self) -> str:
        """Render the receipt as human-readable Markdown."""
        lines = [
            f"# Decision Receipt {self.receipt_id}",
            "",
            f"**Question:** {self.question}",
            f"**Verdict:** {self.verdict.value.replace('_', ' ').title()}",
            f"**Confidence:** {self.confidence:.0%}",
            f"**Consensus:** {'Reached' if self.consensus.reached else 'Not reached'}"
            f" ({self.consensus.method.value}, "
            f"{self.consensus.agreement_ratio:.0%} agreement)",
            f"**Agents:** {', '.join(self.agents)}",
            f"**Rounds:** {self.rounds_used}",
            "",
        ]
        if self.consensus.dissents:
            lines.append("## Dissenting Views")
            lines.append("")
            for d in self.consensus.dissents:
                lines.append(f"**{d.agent}:**")
                for r in d.reasons:
                    lines.append(f"- {r}")
                if d.alternative_view:
                    lines.append(f"  > {d.alternative_view}")
                lines.append("")
        if self.claims:
            lines.append(f"## Claims ({len(self.claims)})")
            lines.append("")
            for c in self.claims:
                status = f" [{c.status}]" if c.status != "active" else ""
                lines.append(
                    f"- **{c.author}** ({c.confidence:.0%}){status}: {c.statement}"
                )
            lines.append("")
        if self.signature:
            lines.append("---")
            lines.append(f"*Signature ({self.signature_algorithm}):* `{self.signature[:32]}...`")
        lines.append(f"\n*Generated {self.timestamp}*")
        return "\n".join(lines)


@dataclass
class DebateResult:
    """The complete result of a multi-agent debate."""
    # Identification
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task: str = ""

    # Outcome
    final_answer: str = ""
    confidence: float = 0.0
    consensus_reached: bool = False
    verdict: Verdict | None = None

    # Execution
    rounds_used: int = 0
    status: str = ""  # consensus_reached, completed, timeout, failed
    duration_seconds: float = 0.0

    # Participants
    participants: list[str] = field(default_factory=list)
    proposals: dict[str, str] = field(default_factory=dict)

    # History
    messages: list[Message] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)

    # Dissent
    dissenting_views: list[str] = field(default_factory=list)
    consensus: Consensus | None = None

    # Evidence
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    # Cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    per_agent_cost: dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    receipt: DecisionReceipt | None = None

    def summary(self) -> str:
        """One-paragraph summary of the debate outcome."""
        parts = [f"Debate on: {self.task}"]
        parts.append(f"Status: {self.status}")
        parts.append(f"Rounds: {self.rounds_used}")
        if self.consensus_reached:
            parts.append(f"Consensus reached ({self.confidence:.0%} confidence)")
        else:
            parts.append("No consensus")
        if self.dissenting_views:
            parts.append(f"Dissents: {len(self.dissenting_views)}")
        if self.final_answer:
            answer_preview = self.final_answer[:200]
            if len(self.final_answer) > 200:
                answer_preview += "..."
            parts.append(f"Answer: {answer_preview}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "consensus_reached": self.consensus_reached,
            "verdict": self.verdict.value if self.verdict else None,
            "rounds_used": self.rounds_used,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "participants": self.participants,
            "proposals": self.proposals,
            "messages": [m.to_dict() for m in self.messages],
            "dissenting_views": self.dissenting_views,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata,
        }

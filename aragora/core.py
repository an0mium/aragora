"""
Core abstractions for the Aragora adversarial validation engine (multi-agent debate framework).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal, Optional

# Type aliases for agent role and stance
AgentRole = Literal["proposer", "critic", "synthesizer", "judge"]
AgentStance = Literal["affirmative", "negative", "neutral"]


class TaskComplexity(Enum):
    """Classification of task complexity for timeout scaling.

    Used by AdaptiveComplexityGovernor to scale timeouts based on
    estimated task difficulty.
    """

    SIMPLE = "simple"  # Quick surveys, simple questions, definitions
    MODERATE = "moderate"  # Standard design/analysis tasks
    COMPLEX = "complex"  # Deep reasoning, multi-step problems, formal proofs
    UNKNOWN = "unknown"  # Fallback when classification is uncertain


@dataclass
class Message:
    """A message in a debate."""

    role: str  # "proposer", "critic", "synthesizer", etc.
    agent: str  # agent name
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    round: int = 0

    def __str__(self) -> str:
        return f"[{self.role}:{self.agent}] {self.content[:100]}..."


@dataclass
class Critique:
    """A critique of a proposal or response."""

    agent: str
    target_agent: str
    target_content: str
    issues: list[str]
    suggestions: list[str]
    severity: float  # 0-1, how serious are the issues
    reasoning: str

    @property
    def target(self) -> str:
        """Alias for target_agent for backward compatibility."""
        return self.target_agent

    @property
    def content(self) -> str:
        """Get the critique's content as formatted text."""
        return self.to_prompt()

    def to_prompt(self) -> str:
        """Format critique for inclusion in prompts."""
        issues_str = "\n".join(f"  - {i}" for i in self.issues)
        suggestions_str = "\n".join(f"  - {s}" for s in self.suggestions)
        return f"""Critique from {self.agent} (severity: {self.severity:.1f}):
Issues:
{issues_str}
Suggestions:
{suggestions_str}
Reasoning: {self.reasoning}"""


@dataclass
class Vote:
    """A vote for a proposal."""

    agent: str
    choice: str  # which proposal/agent they vote for
    reasoning: str
    confidence: float = 1.0  # 0-1, default to full confidence
    continue_debate: bool = True  # Whether agent thinks debate should continue


@dataclass
class DisagreementReport:
    """
    Surfaces explicit agreement/disagreement patterns from debates.

    Inspired by Heavy3.ai's insight: "When all models agree your argument is weak—fix it.
    When they disagree, you see the risk before you commit."
    """

    # Issues all agents unanimously agree on - high confidence problems
    unanimous_critiques: list[str] = field(default_factory=list)

    # Topics where agents split - each tuple is (topic, [agreeing_agents], [disagreeing_agents])
    split_opinions: list[tuple[str, list[str], list[str]]] = field(default_factory=list)

    # Risk areas identified from divergence patterns
    risk_areas: list[str] = field(default_factory=list)

    # Agreement score: 1.0 = complete unanimity, 0.0 = complete disagreement
    agreement_score: float = 0.0

    # Per-agent alignment with final consensus
    agent_alignment: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable disagreement summary."""
        lines = ["=== Disagreement Report ==="]

        if self.unanimous_critiques:
            lines.append(
                f"\n🚨 UNANIMOUS ISSUES ({len(self.unanimous_critiques)}) - Address these:"
            )
            for critique in self.unanimous_critiques[:5]:
                lines.append(f"  • {critique[:200]}")

        if self.split_opinions:
            lines.append(f"\n⚠️ SPLIT OPINIONS ({len(self.split_opinions)}) - Risks to consider:")
            for topic, agree, disagree in self.split_opinions[:5]:
                lines.append(f"  • {topic[:100]}")
                lines.append(f"    Agree: {', '.join(agree)} | Disagree: {', '.join(disagree)}")

        if self.risk_areas:
            lines.append(f"\n🔍 RISK AREAS ({len(self.risk_areas)}):")
            for risk in self.risk_areas[:5]:
                lines.append(f"  • {risk[:200]}")

        lines.append(f"\nAgreement Score: {self.agreement_score:.0%}")

        return "\n".join(lines)


@dataclass
class DebateResult:
    """The result of a multi-agent debate."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    debate_id: str = ""
    task: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    consensus_reached: bool = False
    rounds_used: int = 0
    rounds_completed: int = 0
    status: str = ""
    participants: list[str] = field(default_factory=list)
    proposals: dict[str, str] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    winning_patterns: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    winner: Optional[str] = None  # Winning agent name (set after consensus)
    # Convergence detection results
    convergence_status: str = ""  # "converged", "refining", "diverging", ""
    convergence_similarity: float = 0.0  # Average similarity at end
    per_agent_similarity: dict[str, float] = field(default_factory=dict)  # Agent -> similarity
    # Consensus strength: "strong" (var < 1), "medium" (var < 2), "weak" (var >= 2)
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    # Disagreement surfacing (Heavy3-inspired)
    disagreement_report: Optional["DisagreementReport"] = None
    # Evidence grounding (Heavy3-inspired)
    grounded_verdict: Optional[Any] = None  # GroundedVerdict from aragora.reasoning.citations
    # Belief network analysis - identifies key claims that drive disagreement
    debate_cruxes: list[dict[str, Any]] = field(
        default_factory=list
    )  # From BeliefPropagationAnalyzer
    evidence_suggestions: list[dict[str, Any]] = field(
        default_factory=list
    )  # Claims needing evidence
    # Verification results - claim verification during consensus
    verification_results: dict[str, int] = field(
        default_factory=dict
    )  # Agent -> verified_claim_count
    verification_bonuses: dict[str, float] = field(
        default_factory=dict
    )  # Agent -> vote_bonus_applied
    # Novelty tracking - semantic distance from prior proposals
    per_agent_novelty: dict[str, list[float]] = field(
        default_factory=dict
    )  # Agent -> novelty by round
    avg_novelty: float = 1.0  # Average novelty (1.0 = fresh ideas, 0.0 = repetitive)
    # Formal verification result (from Lean4/Z3)
    formal_verification: Optional[dict[str, Any]] = None  # FormalProofResult.to_dict()

    def __post_init__(self) -> None:
        if self.debate_id:
            self.id = self.debate_id
        else:
            self.debate_id = self.id

        if self.rounds_completed and not self.rounds_used:
            self.rounds_used = self.rounds_completed
        elif self.rounds_used and not self.rounds_completed:
            self.rounds_completed = self.rounds_used
        elif self.rounds_completed != self.rounds_used:
            self.rounds_used = self.rounds_completed

        if not self.status:
            self.status = "consensus_reached" if self.consensus_reached else "completed"

    @property
    def history(self) -> list[Message]:
        """Alias for messages (backward compatibility)."""
        return self.messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize core fields for JSON export."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "status": self.status,
            "final_answer": self.final_answer,
            "consensus_reached": self.consensus_reached,
            "confidence": self.confidence,
            "rounds_used": self.rounds_used,
            "rounds_completed": self.rounds_completed,
            "participants": list(self.participants),
            "duration_seconds": self.duration_seconds,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebateResult":
        """Deserialize from a dictionary."""
        return cls(
            id=data.get("id") or data.get("debate_id", ""),
            debate_id=data.get("debate_id", ""),
            task=data.get("task", ""),
            status=data.get("status", ""),
            final_answer=data.get("final_answer", ""),
            consensus_reached=data.get("consensus_reached", False),
            confidence=data.get("confidence", 0.0),
            rounds_used=data.get("rounds_used", data.get("rounds_completed", 0)),
            rounds_completed=data.get("rounds_completed", data.get("rounds_used", 0)),
            participants=data.get("participants", []),
            proposals=data.get("proposals", {}),
            duration_seconds=data.get("duration_seconds", 0.0),
            winner=data.get("winner"),
        )

    def summary(self) -> str:
        """Human-readable summary of the debate."""
        base = f"""Debate Result ({self.id[:8]}):
Task: {self.task[:100]}...
Rounds: {self.rounds_used}
Consensus: {"Yes" if self.consensus_reached else "No"} (confidence: {self.confidence:.1%})
Critiques: {len(self.critiques)}
Dissenting views: {len(self.dissenting_views)}
Duration: {self.duration_seconds:.1f}s

Final Answer:
{self.final_answer}"""

        if self.disagreement_report:
            base += f"\n\n{self.disagreement_report.summary()}"

        if self.grounded_verdict:
            base += f"\n\nGrounding Score: {self.grounded_verdict.grounding_score:.0%}"
            if hasattr(self.grounded_verdict, "all_citations"):
                base += f" ({len(self.grounded_verdict.all_citations)} citations)"

        return base


@dataclass
class Environment:
    """Defines a task environment for debate."""

    task: str
    context: str = ""  # additional context
    roles: list[str] = field(default_factory=lambda: ["proposer", "critic", "synthesizer"])
    success_fn: Optional[Callable[[str], float]] = None  # 0-1 score
    max_rounds: int = 3
    require_consensus: bool = False
    consensus_threshold: float = 0.7  # fraction of agents that must agree
    # Document IDs attached to this debate (Heavy3-inspired)
    documents: list[str] = field(default_factory=list)


class Agent(ABC):
    """Abstract base class for all agents.

    Attributes:
        name: Unique identifier for the agent
        model: The underlying model (e.g., "claude-3-opus", "gpt-4o")
        role: The agent's role in the debate (proposer, critic, synthesizer, judge)
        system_prompt: Custom system prompt for the agent
        agent_type: Agent type identifier for routing and role assignment
        stance: The agent's stance for asymmetric debate
    """

    name: str
    model: str
    role: AgentRole
    system_prompt: str
    agent_type: str
    stance: AgentStance

    def __init__(self, name: str, model: str, role: AgentRole = "proposer"):
        self.name = name
        self.model = model
        self.role = role
        self.system_prompt = ""
        # Agent type identifier for routing and role assignment
        self.agent_type = "unknown"
        # Stance for asymmetric debate: "affirmative", "negative", or "neutral"
        # - Affirmative: Defend/support proposals
        # - Negative: Challenge/critique proposals
        # - Neutral: Evaluate fairly without bias
        self.stance: AgentStance = "neutral"

    @abstractmethod
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response to a prompt."""
        pass

    @abstractmethod
    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        """Critique a proposal."""
        pass

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        """Vote on which proposal is best."""
        # Default implementation - can be overridden
        prompt = f"""Task: {task}

Proposals to evaluate:
{chr(10).join(f"{agent}: {prop[:500]}..." for agent, prop in proposals.items())}

Which proposal best addresses the task? Respond with:
CHOICE: <agent_name>
CONFIDENCE: <0.0-1.0>
CONTINUE: <yes/no> (whether more debate rounds would help improve the answer)
REASONING: <brief explanation>"""

        response = await self.generate(prompt)
        # Parse response (simple extraction)
        lines = response.strip().split("\n")
        choice = ""
        confidence = 0.5
        reasoning = ""
        continue_debate = True

        for line in lines:
            if line.startswith("CHOICE:"):
                choice = line.replace("CHOICE:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except (ValueError, TypeError):
                    confidence = 0.5  # Default confidence on parse error
            elif line.startswith("CONTINUE:"):
                cont_val = line.replace("CONTINUE:", "").strip().lower()
                continue_debate = cont_val not in ("no", "false", "0", "n")
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return Vote(
            agent=self.name,
            choice=choice,
            confidence=confidence,
            reasoning=reasoning,
            continue_debate=continue_debate,
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Update the agent's system prompt (for self-improvement)."""
        self.system_prompt = prompt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model}, role={self.role})"


def __getattr__(name: str) -> Any:
    if name == "DebateProtocol":
        from aragora.debate.protocol import DebateProtocol

        return DebateProtocol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["DebateProtocol"])

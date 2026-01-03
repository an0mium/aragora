"""
Core abstractions for the Agora multi-agent debate framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional
from datetime import datetime
import uuid


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
class DebateResult:
    """The result of a multi-agent debate."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    consensus_reached: bool = False
    rounds_used: int = 0
    messages: list[Message] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    winning_patterns: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    # Convergence detection results
    convergence_status: str = ""  # "converged", "refining", "diverging", ""
    convergence_similarity: float = 0.0  # Average similarity at end
    per_agent_similarity: dict = field(default_factory=dict)  # Agent -> similarity
    # Consensus strength: "strong" (var < 1), "medium" (var < 2), "weak" (var >= 2)
    consensus_strength: str = ""
    consensus_variance: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of the debate."""
        return f"""Debate Result ({self.id[:8]}):
Task: {self.task[:100]}...
Rounds: {self.rounds_used}
Consensus: {'Yes' if self.consensus_reached else 'No'} (confidence: {self.confidence:.1%})
Critiques: {len(self.critiques)}
Dissenting views: {len(self.dissenting_views)}
Duration: {self.duration_seconds:.1f}s

Final Answer:
{self.final_answer}"""


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


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, model: str, role: str = "proposer"):
        self.name = name
        self.model = model
        self.role = role
        self.system_prompt: str = ""
        # Stance for asymmetric debate: "affirmative", "negative", or "neutral"
        # - Affirmative: Defend/support proposals
        # - Negative: Challenge/critique proposals
        # - Neutral: Evaluate fairly without bias
        self.stance: str = "neutral"

    @abstractmethod
    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response to a prompt."""
        pass

    @abstractmethod
    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
        """Critique a proposal."""
        pass

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        """Vote on which proposal is best."""
        # Default implementation - can be overridden
        prompt = f"""Task: {task}

Proposals to evaluate:
{chr(10).join(f'{agent}: {prop[:500]}...' for agent, prop in proposals.items())}

Which proposal best addresses the task? Respond with:
CHOICE: <agent_name>
CONFIDENCE: <0.0-1.0>
CONTINUE: <yes/no> (whether more debate rounds would help improve the answer)
REASONING: <brief explanation>"""

        response = await self.generate(prompt)
        # Parse response (simple extraction)
        lines = response.strip().split('\n')
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
                except:
                    confidence = 0.5
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

    def set_system_prompt(self, prompt: str):
        """Update the agent's system prompt (for self-improvement)."""
        self.system_prompt = prompt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model}, role={self.role})"

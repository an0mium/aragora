"""
Debate context container for pipeline execution.

This module defines the shared state container used by all debate phases.
The DebateContext is passed between phases, allowing them to read and
modify debate state without tight coupling to the orchestrator.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent, Critique, DebateResult, Environment, Message


@dataclass
class DebateContext:
    """
    Shared state container for debate execution pipeline.

    This class holds all mutable and immutable state needed during debate execution.
    It is created at the start of _run_inner() and passed to each phase for
    reading and mutation.

    Phases should:
    - Read from context to get current state
    - Write to context to update state
    - Not store references to the orchestrator (use callbacks if needed)
    """

    # =========================================================================
    # Immutable Inputs (set once at debate start)
    # =========================================================================

    env: "Environment"
    """The debate environment containing task, context, and configuration."""

    agents: list["Agent"] = field(default_factory=list)
    """All agents participating in the debate."""

    start_time: float = 0.0
    """Unix timestamp when debate started."""

    debate_id: str = ""
    """Unique identifier for this debate."""

    domain: str = "general"
    """Extracted domain for metrics and specialization."""

    # =========================================================================
    # Agent Subsets (computed at phase boundaries)
    # =========================================================================

    proposers: list["Agent"] = field(default_factory=list)
    """Agents with proposer role (or fallback to first agent)."""

    critics: list["Agent"] = field(default_factory=list)
    """Agents that will provide critiques this round."""

    available_agents: list["Agent"] = field(default_factory=list)
    """Agents that passed circuit breaker filter."""

    # =========================================================================
    # Core Debate State (mutated during execution)
    # =========================================================================

    proposals: dict[str, str] = field(default_factory=dict)
    """Agent name -> proposal text mapping."""

    context_messages: list["Message"] = field(default_factory=list)
    """All messages in debate context (for prompt building)."""

    result: Optional["DebateResult"] = None
    """The DebateResult being built up during execution."""

    # =========================================================================
    # Round State (mutated each round)
    # =========================================================================

    current_round: int = 0
    """Current round number (0 = proposals, 1+ = critique/revision)."""

    previous_round_responses: dict[str, str] = field(default_factory=dict)
    """Agent name -> previous response (for revision prompts)."""

    round_critiques: list["Critique"] = field(default_factory=list)
    """Critiques collected in the current round."""

    # =========================================================================
    # Timeout Recovery (for partial results on timeout)
    # =========================================================================

    partial_messages: list["Message"] = field(default_factory=list)
    """Messages accumulated for timeout recovery."""

    partial_critiques: list["Critique"] = field(default_factory=list)
    """Critiques accumulated for timeout recovery."""

    partial_rounds: int = 0
    """Number of rounds completed before timeout."""

    # =========================================================================
    # Voting State (set during consensus phase)
    # =========================================================================

    vote_tally: dict[str, float] = field(default_factory=dict)
    """Final weighted vote counts per choice."""

    choice_mapping: dict[str, str] = field(default_factory=dict)
    """Variant -> canonical choice mapping from vote grouping."""

    vote_weight_cache: dict[str, float] = field(default_factory=dict)
    """Pre-computed vote weights per agent."""

    winner_agent: Optional[str] = None
    """Name of winning agent (set after consensus)."""

    # =========================================================================
    # Caches (populated once, read by multiple phases)
    # =========================================================================

    historical_context_cache: str = ""
    """Fetched historical debate context for institutional memory."""

    continuum_context_cache: str = ""
    """Context from ContinuumMemory retrieval."""

    research_context: Optional[str] = None
    """Pre-debate research results."""

    evidence_pack: Any = None
    """Collected evidence pack from EvidenceCollector."""

    ratings_cache: dict[str, Any] = field(default_factory=dict)
    """Batch-fetched AgentRating objects by agent name."""

    # =========================================================================
    # Convergence State
    # =========================================================================

    convergence_status: str = ""
    """Current convergence status: 'converged', 'refining', 'diverging', ''."""

    convergence_similarity: float = 0.0
    """Average semantic similarity between responses."""

    per_agent_similarity: dict[str, float] = field(default_factory=dict)
    """Per-agent similarity scores."""

    early_termination: bool = False
    """Flag set when debate should terminate early due to convergence."""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_agent_by_name(self, name: str) -> Optional["Agent"]:
        """Look up an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_proposal(self, agent_name: str) -> str:
        """Get proposal for an agent, or empty string if none."""
        return self.proposals.get(agent_name, "")

    def add_message(self, msg: "Message") -> None:
        """Add a message to both context and partial tracking."""
        self.context_messages.append(msg)
        self.partial_messages.append(msg)
        if self.result:
            self.result.messages.append(msg)

    def add_critique(self, critique: "Critique") -> None:
        """Add a critique to both result and partial tracking."""
        self.round_critiques.append(critique)
        self.partial_critiques.append(critique)
        if self.result:
            self.result.critiques.append(critique)

    def finalize_result(self) -> "DebateResult":
        """
        Finalize and return the debate result.

        Sets duration, rounds used, and other final fields.
        """
        import time

        if self.result:
            self.result.duration_seconds = time.time() - self.start_time
            self.result.rounds_used = self.current_round
            if self.winner_agent:
                self.result.winner = self.winner_agent
        return self.result

    def to_summary_dict(self) -> dict:
        """Return a summary dict for logging/debugging."""
        return {
            "debate_id": self.debate_id,
            "domain": self.domain,
            "agents": [a.name for a in self.agents],
            "proposers": [a.name for a in self.proposers],
            "current_round": self.current_round,
            "num_proposals": len(self.proposals),
            "num_messages": len(self.context_messages),
            "winner": self.winner_agent,
            "convergence_status": self.convergence_status,
        }

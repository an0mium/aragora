"""
Data types for Agent Introspection API.

Provides data classes for storing and formatting agent self-awareness data
that gets injected into debate prompts.
"""

from dataclasses import dataclass, field


@dataclass
class IntrospectionSnapshot:
    """
    Aggregated introspection data for an agent.

    Contains reputation metrics, performance history, and persona traits
    that enable agents to reason about their own track record during debates.
    """

    agent_name: str
    reputation_score: float = 0.0  # 0-1 from AgentReputation.score
    vote_weight: float = 1.0  # 0.4-1.6 from AgentReputation.vote_weight
    proposals_made: int = 0
    proposals_accepted: int = 0
    critiques_given: int = 0
    critiques_valuable: int = 0
    calibration_score: float = 0.5  # Titans/MIRAS prediction accuracy
    debate_count: int = 0
    top_expertise: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)

    @property
    def proposal_acceptance_rate(self) -> float:
        """Calculate proposal acceptance rate."""
        if self.proposals_made == 0:
            return 0.0
        return self.proposals_accepted / self.proposals_made

    @property
    def critique_effectiveness(self) -> float:
        """Calculate critique effectiveness rate."""
        if self.critiques_given == 0:
            return 0.0
        return self.critiques_valuable / self.critiques_given

    @property
    def calibration_label(self) -> str:
        """Human-readable calibration label."""
        if self.calibration_score >= 0.7:
            return "excellent"
        elif self.calibration_score >= 0.5:
            return "good"
        elif self.calibration_score >= 0.3:
            return "fair"
        else:
            return "developing"

    def to_prompt_section(self, max_chars: int = 600) -> str:
        """
        Format introspection data as a prompt section.

        Returns a markdown-formatted section under the character limit,
        matching the existing "## SUCCESSFUL PATTERNS" style.

        Args:
            max_chars: Maximum characters for the section (default 600)

        Returns:
            Formatted prompt section string
        """
        lines = ["## YOUR TRACK RECORD"]

        # Line 1: Reputation and vote weight
        rep_pct = int(self.reputation_score * 100)
        lines.append(f"Reputation: {rep_pct}% | Vote weight: {self.vote_weight:.1f}x")

        # Line 2: Proposals (only if agent has made proposals)
        if self.proposals_made > 0:
            acc_pct = int(self.proposal_acceptance_rate * 100)
            lines.append(
                f"Proposals: {self.proposals_accepted}/{self.proposals_made} "
                f"accepted ({acc_pct}%)"
            )

        # Line 3: Critiques (only if agent has given critiques)
        if self.critiques_given > 0:
            crit_pct = int(self.critique_effectiveness * 100)
            lines.append(
                f"Critiques: {crit_pct}% valuable | " f"Calibration: {self.calibration_label}"
            )

        # Line 4: Expertise (only if available)
        if self.top_expertise:
            expertise_str = ", ".join(self.top_expertise[:3])
            lines.append(f"Expertise: {expertise_str}")

        # Line 5: Traits (only if available and space permits)
        if self.traits:
            traits_str = ", ".join(self.traits[:3])
            lines.append(f"Style: {traits_str}")

        # Join and truncate if needed
        result = "\n".join(lines)

        # If over limit, progressively remove optional lines
        while len(result) > max_chars and len(lines) > 2:
            lines.pop()
            result = "\n".join(lines)

        return result

    def to_dict(self) -> dict:
        """Serialize snapshot to dictionary."""
        return {
            "agent_name": self.agent_name,
            "reputation_score": self.reputation_score,
            "vote_weight": self.vote_weight,
            "proposals_made": self.proposals_made,
            "proposals_accepted": self.proposals_accepted,
            "proposal_acceptance_rate": self.proposal_acceptance_rate,
            "critiques_given": self.critiques_given,
            "critiques_valuable": self.critiques_valuable,
            "critique_effectiveness": self.critique_effectiveness,
            "calibration_score": self.calibration_score,
            "calibration_label": self.calibration_label,
            "debate_count": self.debate_count,
            "top_expertise": self.top_expertise,
            "traits": self.traits,
        }

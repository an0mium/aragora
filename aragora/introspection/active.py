"""
Active Introspection for Agent Meta-Reasoning.

Provides real-time, per-round introspection that tracks agent performance
during a debate and generates reflective prompts to guide agent behavior.

Unlike the passive IntrospectionSnapshot (which aggregates historical data
at debate start), ActiveIntrospectionTracker updates every round with
live metrics: proposal acceptance, critique quality, agreement patterns,
and argument strength.

MetaReasoningEngine consumes these metrics to produce strategic guidance
that is injected into agent prompts each round.

Usage:
    from aragora.introspection.active import (
        ActiveIntrospectionTracker,
        MetaReasoningEngine,
        IntrospectionGoals,
        RoundMetrics,
    )

    # At debate start
    tracker = ActiveIntrospectionTracker()
    engine = MetaReasoningEngine()

    # After each round
    tracker.update_round("claude", round_num=1, metrics=RoundMetrics(...))
    guidance = engine.generate_guidance(tracker.get_summary("claude"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IntrospectionGoals:
    """Per-debate goals for an agent's introspection.

    Set at debate start to define what the agent should focus on.
    The MetaReasoningEngine uses these to tailor reflective prompts.
    """

    agent_name: str
    target_acceptance_rate: float = 0.6
    target_critique_quality: float = 0.5
    focus_expertise: list[str] = field(default_factory=list)
    collaboration_targets: list[str] = field(default_factory=list)
    strategic_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize goals to dictionary."""
        return {
            "agent_name": self.agent_name,
            "target_acceptance_rate": self.target_acceptance_rate,
            "target_critique_quality": self.target_critique_quality,
            "focus_expertise": self.focus_expertise,
            "collaboration_targets": self.collaboration_targets,
            "strategic_notes": self.strategic_notes,
        }


@dataclass
class RoundMetrics:
    """Metrics collected for a single round of debate.

    Each field captures one dimension of agent performance in a round.
    Fields default to None to distinguish 'not measured' from 'zero'.
    """

    round_number: int = 0
    proposals_made: int = 0
    proposals_accepted: int = 0
    critiques_given: int = 0
    critiques_led_to_changes: int = 0
    votes_received: int = 0
    votes_cast: int = 0
    agreements: dict[str, bool] = field(default_factory=dict)
    """Maps agent_name -> True if agreed, False if disagreed."""
    argument_influence: float = 0.0
    """0.0-1.0 score indicating how much this agent's arguments influenced the round outcome."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "round_number": self.round_number,
            "proposals_made": self.proposals_made,
            "proposals_accepted": self.proposals_accepted,
            "critiques_given": self.critiques_given,
            "critiques_led_to_changes": self.critiques_led_to_changes,
            "votes_received": self.votes_received,
            "votes_cast": self.votes_cast,
            "agreements": dict(self.agreements),
            "argument_influence": self.argument_influence,
        }


@dataclass
class AgentPerformanceSummary:
    """Aggregated performance summary for an agent across all rounds so far.

    Computed by ActiveIntrospectionTracker from per-round RoundMetrics.
    Consumed by MetaReasoningEngine to produce reflective prompts.
    """

    agent_name: str
    rounds_completed: int = 0
    total_proposals: int = 0
    total_accepted: int = 0
    total_critiques: int = 0
    total_critiques_effective: int = 0
    total_votes_received: int = 0
    total_argument_influence: float = 0.0
    agreement_patterns: dict[str, list[bool]] = field(default_factory=dict)
    """Maps other_agent_name -> list of agree/disagree booleans per round."""
    goals: IntrospectionGoals | None = None
    round_history: list[RoundMetrics] = field(default_factory=list)

    @property
    def proposal_acceptance_rate(self) -> float:
        """Acceptance rate for proposals in this debate."""
        if self.total_proposals == 0:
            return 0.0
        return self.total_accepted / self.total_proposals

    @property
    def critique_effectiveness(self) -> float:
        """How often this agent's critiques led to actual changes."""
        if self.total_critiques == 0:
            return 0.0
        return self.total_critiques_effective / self.total_critiques

    @property
    def average_influence(self) -> float:
        """Average argument influence across all rounds."""
        if self.rounds_completed == 0:
            return 0.0
        return self.total_argument_influence / self.rounds_completed

    def get_agreement_rate(self, other_agent: str) -> float | None:
        """Get agreement rate with a specific agent.

        Returns None if no interactions recorded.
        """
        history = self.agreement_patterns.get(other_agent)
        if not history:
            return None
        return sum(1 for a in history if a) / len(history)

    def get_top_disagreers(self, limit: int = 3) -> list[tuple[str, float]]:
        """Get agents who disagree most with this agent.

        Returns list of (agent_name, disagreement_rate) sorted by rate descending.
        """
        disagreers: list[tuple[str, float]] = []
        for agent, history in self.agreement_patterns.items():
            if not history:
                continue
            disagree_rate = sum(1 for a in history if not a) / len(history)
            disagreers.append((agent, disagree_rate))
        disagreers.sort(key=lambda x: x[1], reverse=True)
        return disagreers[:limit]

    def get_top_allies(self, limit: int = 3) -> list[tuple[str, float]]:
        """Get agents who agree most with this agent.

        Returns list of (agent_name, agreement_rate) sorted by rate descending.
        """
        allies: list[tuple[str, float]] = []
        for agent, history in self.agreement_patterns.items():
            if not history:
                continue
            agree_rate = sum(1 for a in history if a) / len(history)
            allies.append((agent, agree_rate))
        allies.sort(key=lambda x: x[1], reverse=True)
        return allies[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to dictionary."""
        return {
            "agent_name": self.agent_name,
            "rounds_completed": self.rounds_completed,
            "total_proposals": self.total_proposals,
            "total_accepted": self.total_accepted,
            "proposal_acceptance_rate": self.proposal_acceptance_rate,
            "total_critiques": self.total_critiques,
            "total_critiques_effective": self.total_critiques_effective,
            "critique_effectiveness": self.critique_effectiveness,
            "total_votes_received": self.total_votes_received,
            "average_influence": self.average_influence,
            "agreement_patterns": {
                k: list(v) for k, v in self.agreement_patterns.items()
            },
            "goals": self.goals.to_dict() if self.goals else None,
            "round_history": [r.to_dict() for r in self.round_history],
        }


class ActiveIntrospectionTracker:
    """Tracks per-round agent performance metrics during a debate.

    Unlike IntrospectionCache which loads static historical data once,
    this tracker updates every round with live performance data.

    Usage:
        tracker = ActiveIntrospectionTracker()
        tracker.set_goals("claude", IntrospectionGoals(agent_name="claude"))
        tracker.update_round("claude", round_num=1, metrics=RoundMetrics(...))
        summary = tracker.get_summary("claude")
    """

    def __init__(self) -> None:
        self._summaries: dict[str, AgentPerformanceSummary] = {}
        self._goals: dict[str, IntrospectionGoals] = {}

    def set_goals(self, agent_name: str, goals: IntrospectionGoals) -> None:
        """Set per-debate goals for an agent.

        Args:
            agent_name: Name of the agent
            goals: Goals for this debate
        """
        self._goals[agent_name] = goals
        # Ensure summary exists
        if agent_name not in self._summaries:
            self._summaries[agent_name] = AgentPerformanceSummary(
                agent_name=agent_name
            )
        self._summaries[agent_name].goals = goals

    def update_round(
        self,
        agent_name: str,
        round_num: int,
        metrics: RoundMetrics,
    ) -> None:
        """Record metrics for a completed round.

        Creates the agent summary if it does not exist yet.
        Metrics are accumulated into the running totals.

        Args:
            agent_name: Name of the agent
            round_num: Round number (1-indexed)
            metrics: Metrics for this round
        """
        if agent_name not in self._summaries:
            self._summaries[agent_name] = AgentPerformanceSummary(
                agent_name=agent_name
            )
            if agent_name in self._goals:
                self._summaries[agent_name].goals = self._goals[agent_name]

        summary = self._summaries[agent_name]

        # Set round_number on metrics if not already set
        if metrics.round_number == 0:
            metrics.round_number = round_num

        summary.rounds_completed = max(summary.rounds_completed, round_num)
        summary.total_proposals += metrics.proposals_made
        summary.total_accepted += metrics.proposals_accepted
        summary.total_critiques += metrics.critiques_given
        summary.total_critiques_effective += metrics.critiques_led_to_changes
        summary.total_votes_received += metrics.votes_received
        summary.total_argument_influence += metrics.argument_influence
        summary.round_history.append(metrics)

        # Update agreement patterns
        for other_agent, agreed in metrics.agreements.items():
            if other_agent not in summary.agreement_patterns:
                summary.agreement_patterns[other_agent] = []
            summary.agreement_patterns[other_agent].append(agreed)

        logger.debug(
            "[introspection] Updated round %d for %s: "
            "proposals=%d/%d, critiques=%d/%d, influence=%.2f",
            round_num,
            agent_name,
            metrics.proposals_accepted,
            metrics.proposals_made,
            metrics.critiques_led_to_changes,
            metrics.critiques_given,
            metrics.argument_influence,
        )

    def get_summary(self, agent_name: str) -> AgentPerformanceSummary | None:
        """Get the accumulated performance summary for an agent.

        Returns None if no data has been recorded for this agent.
        """
        return self._summaries.get(agent_name)

    def get_all_summaries(self) -> dict[str, AgentPerformanceSummary]:
        """Get summaries for all tracked agents."""
        return self._summaries.copy()

    @property
    def agent_count(self) -> int:
        """Number of agents being tracked."""
        return len(self._summaries)

    def reset(self) -> None:
        """Clear all tracked data. Call between debates."""
        self._summaries.clear()
        self._goals.clear()


class MetaReasoningEngine:
    """Generates reflective prompts from active introspection data.

    Consumes AgentPerformanceSummary to produce strategic guidance
    that helps agents reason about their own debate performance.

    Guidance categories:
    - Proposal effectiveness: what makes proposals succeed/fail
    - Critique quality: how effective critiques are at driving changes
    - Relationship dynamics: agreement/disagreement patterns
    - Influence assessment: argument strength signals
    - Goal alignment: progress toward per-debate goals

    Usage:
        engine = MetaReasoningEngine()
        summary = tracker.get_summary("claude")
        guidance = engine.generate_guidance(summary)
        prompt_section = engine.format_for_prompt(summary, max_chars=400)
    """

    def generate_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate reflective guidance lines from performance summary.

        Returns a list of guidance strings, each addressing one aspect
        of the agent's performance. Empty list if no meaningful guidance
        can be generated (e.g., no rounds completed).

        Args:
            summary: Accumulated performance data for the agent

        Returns:
            List of guidance strings (0 to ~5 items)
        """
        if summary.rounds_completed == 0:
            return []

        lines: list[str] = []

        # 1. Proposal effectiveness
        lines.extend(self._proposal_guidance(summary))

        # 2. Critique quality
        lines.extend(self._critique_guidance(summary))

        # 3. Relationship dynamics
        lines.extend(self._relationship_guidance(summary))

        # 4. Influence assessment
        lines.extend(self._influence_guidance(summary))

        # 5. Goal alignment
        lines.extend(self._goal_guidance(summary))

        return lines

    def format_for_prompt(
        self,
        summary: AgentPerformanceSummary,
        max_chars: int = 400,
    ) -> str:
        """Format introspection guidance as a prompt section.

        Produces a markdown section suitable for injection into agent prompts.
        Returns empty string if no guidance available.

        Args:
            summary: Performance summary for the agent
            max_chars: Maximum characters for the section

        Returns:
            Formatted prompt section string
        """
        guidance = self.generate_guidance(summary)
        if not guidance:
            return ""

        lines = ["## YOUR PERFORMANCE THIS DEBATE"]

        # Add round/acceptance header
        acc_pct = int(summary.proposal_acceptance_rate * 100)
        crit_pct = int(summary.critique_effectiveness * 100)
        lines.append(
            f"Rounds: {summary.rounds_completed} | "
            f"Proposal acceptance: {acc_pct}% | "
            f"Critique effectiveness: {crit_pct}%"
        )

        # Add guidance bullets
        for g in guidance:
            lines.append(f"- {g}")

        result = "\n".join(lines)

        # Truncate if needed, preserving header
        while len(result) > max_chars and len(lines) > 2:
            lines.pop()
            result = "\n".join(lines)

        return result

    def _proposal_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate guidance about proposal effectiveness."""
        lines: list[str] = []

        if summary.total_proposals == 0:
            return lines

        rate = summary.proposal_acceptance_rate
        if rate >= 0.8:
            lines.append(
                "Your proposals have a strong acceptance rate — "
                "maintain your current approach."
            )
        elif rate >= 0.5:
            lines.append(
                "Your proposals are moderately accepted — "
                "consider addressing critiques preemptively."
            )
        elif rate > 0.0:
            lines.append(
                "Your proposals have low acceptance — "
                "try incorporating other agents' perspectives before proposing."
            )
        else:
            lines.append(
                "None of your proposals have been accepted yet — "
                "consider a different approach or build on others' ideas."
            )

        return lines

    def _critique_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate guidance about critique quality."""
        lines: list[str] = []

        if summary.total_critiques == 0:
            return lines

        effectiveness = summary.critique_effectiveness
        if effectiveness >= 0.7:
            lines.append(
                "Your critiques frequently lead to changes — "
                "your analytical perspective is valued."
            )
        elif effectiveness >= 0.4:
            lines.append(
                "Some of your critiques drive changes — "
                "focus on actionable, specific feedback."
            )
        else:
            lines.append(
                "Few of your critiques lead to changes — "
                "try providing concrete alternatives alongside criticisms."
            )

        return lines

    def _relationship_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate guidance about agreement/disagreement patterns."""
        lines: list[str] = []

        top_disagreers = summary.get_top_disagreers(limit=2)
        for agent, disagree_rate in top_disagreers:
            if disagree_rate >= 0.6:
                pct = int(disagree_rate * 100)
                lines.append(
                    f"{agent} disagrees with you {pct}% of the time — "
                    f"consider their perspective to strengthen your arguments."
                )
                break  # Only one disagreer line

        top_allies = summary.get_top_allies(limit=2)
        for agent, agree_rate in top_allies:
            if agree_rate >= 0.7:
                pct = int(agree_rate * 100)
                lines.append(
                    f"{agent} agrees with you {pct}% of the time — "
                    f"you may be able to build coalition arguments together."
                )
                break  # Only one ally line

        return lines

    def _influence_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate guidance about argument influence."""
        lines: list[str] = []

        avg = summary.average_influence
        if avg >= 0.7:
            lines.append(
                "Your arguments have strong influence on outcomes — "
                "your expertise is highly relevant to this discussion."
            )
        elif avg >= 0.4:
            lines.append(
                "Your arguments have moderate influence — "
                "emphasize your strongest points to increase impact."
            )
        elif avg > 0.0 and summary.rounds_completed >= 2:
            lines.append(
                "Your arguments have limited influence so far — "
                "try aligning with the strongest evidence available."
            )

        return lines

    def _goal_guidance(
        self,
        summary: AgentPerformanceSummary,
    ) -> list[str]:
        """Generate guidance about progress toward debate goals."""
        lines: list[str] = []
        goals = summary.goals
        if goals is None:
            return lines

        # Check acceptance rate vs target
        if summary.total_proposals > 0:
            rate = summary.proposal_acceptance_rate
            target = goals.target_acceptance_rate
            if rate < target * 0.8:
                lines.append(
                    f"You are below your acceptance target ({int(rate * 100)}% vs "
                    f"{int(target * 100)}% goal) — adjust your strategy."
                )

        # Remind about focus expertise
        if goals.focus_expertise:
            expertise_str = ", ".join(goals.focus_expertise[:3])
            lines.append(
                f"Your expertise in {expertise_str} is most relevant "
                f"to this discussion."
            )

        return lines

"""
Debate Forking capability.

Inspired by UniversalBackrooms' branching conversations, this module provides:
- Forking debates when agents fundamentally disagree
- Running parallel branches to explore alternatives
- Comparing and merging branch outcomes
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from copy import deepcopy

logger = logging.getLogger(__name__)

from aragora.core import Agent, Message, DebateResult, Environment


@dataclass
class ForkPoint:
    """A point where the debate forked."""

    round: int
    reason: str
    disagreeing_agents: list[str]
    parent_debate_id: str
    branch_ids: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Branch:
    """A branch of a forked debate."""

    branch_id: str
    parent_debate_id: str
    fork_round: int
    hypothesis: str  # What this branch is exploring
    lead_agent: str  # Agent whose approach this branch follows
    messages: list[Message] = field(default_factory=list)
    result: Optional[DebateResult] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_complete(self) -> bool:
        return self.result is not None


@dataclass
class ForkDecision:
    """Decision about whether and how to fork."""

    should_fork: bool
    reason: str
    branches: list[dict]  # [{hypothesis: str, lead_agent: str}, ...]
    disagreement_score: float  # 0-1


@dataclass
class MergeResult:
    """Result of merging forked branches."""

    winning_branch_id: str
    winning_hypothesis: str
    comparison_summary: str
    all_branch_results: dict[str, DebateResult]
    merged_insights: list[str]


class ForkDetector:
    """
    Detects when a debate should fork.

    Analyzes disagreement patterns to identify fundamental divergences.
    """

    # Threshold for triggering a fork
    DISAGREEMENT_THRESHOLD = 0.7
    MIN_ROUNDS_BEFORE_FORK = 2

    def should_fork(
        self,
        messages: list[Message],
        round_num: int,
        agents: list[Agent],
    ) -> ForkDecision:
        """
        Determine if the debate should fork.

        Args:
            messages: Messages so far
            round_num: Current round number
            agents: Participating agents

        Returns:
            ForkDecision indicating whether and how to fork
        """
        if round_num < self.MIN_ROUNDS_BEFORE_FORK:
            return ForkDecision(
                should_fork=False,
                reason="Too early to fork",
                branches=[],
                disagreement_score=0.0,
            )

        # Get latest messages from each agent
        latest_by_agent = {}
        for msg in reversed(messages):
            if msg.agent not in latest_by_agent:
                latest_by_agent[msg.agent] = msg

        if len(latest_by_agent) < 2:
            return ForkDecision(
                should_fork=False,
                reason="Not enough agents",
                branches=[],
                disagreement_score=0.0,
            )

        # Detect fundamental disagreements
        disagreements = self._detect_disagreements(latest_by_agent)

        if not disagreements:
            return ForkDecision(
                should_fork=False,
                reason="No fundamental disagreements detected",
                branches=[],
                disagreement_score=0.0,
            )

        # Find the most significant disagreement
        top_disagreement = max(disagreements, key=lambda d: d["score"])

        if top_disagreement["score"] < self.DISAGREEMENT_THRESHOLD:
            return ForkDecision(
                should_fork=False,
                reason="Disagreement not severe enough",
                branches=[],
                disagreement_score=top_disagreement["score"],
            )

        # Create branch proposals
        branches = [
            {
                "hypothesis": f"{agent}'s approach: {self._extract_approach(msg)}",
                "lead_agent": agent,
            }
            for agent, msg in latest_by_agent.items()
            if agent in top_disagreement["agents"]
        ]

        return ForkDecision(
            should_fork=True,
            reason=top_disagreement["reason"],
            branches=branches[:3],  # Max 3 branches
            disagreement_score=top_disagreement["score"],
        )

    def _detect_disagreements(self, latest_by_agent: dict[str, Message]) -> list[dict]:
        """Detect fundamental disagreements between agents."""
        disagreements = []

        agents = list(latest_by_agent.keys())
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i + 1 :]:
                msg_a = latest_by_agent[agent_a]
                msg_b = latest_by_agent[agent_b]

                # Look for disagreement indicators
                score, reason = self._calculate_disagreement(msg_a, msg_b)

                if score > 0.3:
                    disagreements.append(
                        {
                            "agents": [agent_a, agent_b],
                            "score": score,
                            "reason": reason,
                        }
                    )

        return disagreements

    def _calculate_disagreement(self, msg_a: Message, msg_b: Message) -> tuple[float, str]:
        """Calculate disagreement score between two messages."""
        content_a = msg_a.content.lower()
        content_b = msg_b.content.lower()

        score = 0.0
        reasons = []

        # Check for explicit disagreement phrases
        disagreement_phrases = [
            "disagree",
            "don't agree",
            "incorrect",
            "wrong approach",
            "better alternative",
            "instead",
            "rather than",
            "however",
            "on the contrary",
            "fundamentally different",
        ]

        for phrase in disagreement_phrases:
            if phrase in content_a or phrase in content_b:
                score += 0.2
                reasons.append(f"Contains '{phrase}'")

        # Check for contradictory recommendations
        # (Simplified - would use NLP in production)
        if ("should" in content_a and "should not" in content_b) or (
            "should not" in content_a and "should" in content_b
        ):
            score += 0.3
            reasons.append("Contradictory should/should not")

        # Check for different technology choices
        tech_terms = [
            "microservice",
            "monolith",
            "sql",
            "nosql",
            "sync",
            "async",
            "rest",
            "graphql",
            "cache",
            "database",
            "queue",
            "polling",
        ]
        tech_a = set(t for t in tech_terms if t in content_a)
        tech_b = set(t for t in tech_terms if t in content_b)

        if tech_a and tech_b and not (tech_a & tech_b):
            score += 0.3
            reasons.append(f"Different tech choices: {tech_a} vs {tech_b}")

        return min(1.0, score), "; ".join(reasons) if reasons else "No clear disagreement"

    def _extract_approach(self, msg: Message) -> str:
        """Extract a brief description of the agent's approach."""
        content = msg.content

        # Look for key statements
        for marker in ["I propose", "My approach", "I suggest", "The solution is"]:
            if marker.lower() in content.lower():
                idx = content.lower().find(marker.lower())
                extract = content[idx : idx + 100]
                return extract.split(".")[0] + "..."

        # Fall back to first sentence
        return content[:100].split(".")[0] + "..."


class DebateForker:
    """
    Manages forking and merging of debates.

    Creates parallel branches to explore alternative solutions.
    """

    def __init__(self):
        self.detector = ForkDetector()
        self.branches: dict[str, list[Branch]] = {}  # parent_id -> branches
        self.fork_points: dict[str, list[ForkPoint]] = {}  # parent_id -> fork points

    def fork(
        self,
        parent_debate_id: str,
        fork_round: int,
        messages_so_far: list[Message],
        decision: ForkDecision,
    ) -> list[Branch]:
        """
        Create forked branches from a debate.

        Args:
            parent_debate_id: ID of the parent debate
            fork_round: Round at which the fork occurs
            messages_so_far: Messages up to the fork point
            decision: Fork decision with branch specifications

        Returns:
            List of created branches
        """
        branches = []

        for branch_spec in decision.branches:
            branch = Branch(
                branch_id=str(uuid.uuid4())[:8],
                parent_debate_id=parent_debate_id,
                fork_round=fork_round,
                hypothesis=branch_spec["hypothesis"],
                lead_agent=branch_spec["lead_agent"],
                messages=deepcopy(messages_so_far),  # Copy history
            )
            branches.append(branch)

        # Record fork point
        fork_point = ForkPoint(
            round=fork_round,
            reason=decision.reason,
            disagreeing_agents=[b["lead_agent"] for b in decision.branches],
            parent_debate_id=parent_debate_id,
            branch_ids=[b.branch_id for b in branches],
        )

        if parent_debate_id not in self.fork_points:
            self.fork_points[parent_debate_id] = []
        self.fork_points[parent_debate_id].append(fork_point)

        if parent_debate_id not in self.branches:
            self.branches[parent_debate_id] = []
        self.branches[parent_debate_id].extend(branches)

        return branches

    async def run_branches(
        self,
        branches: list[Branch],
        env: Environment,
        agents: list[Agent],
        run_debate_fn: Callable,
        max_rounds: int = 3,
    ) -> list[Branch]:
        """
        Run all branches in parallel.

        Args:
            branches: Branches to run
            env: Environment configuration
            agents: Available agents
            run_debate_fn: Function to run a debate
            max_rounds: Maximum additional rounds per branch

        Returns:
            Branches with results populated
        """

        async def run_branch(branch: Branch) -> Branch:
            # Create branch-specific environment
            branch_env = Environment(
                task=f"[Branch: {branch.hypothesis}]\n{env.task}",
                max_rounds=max_rounds,
            )

            # Run debate continuing from fork point
            result = await run_debate_fn(
                branch_env,
                agents,
                initial_messages=branch.messages,
            )

            branch.result = result
            return branch

        # Run all branches in parallel
        results = await asyncio.gather(*[run_branch(b) for b in branches], return_exceptions=True)
        # Filter out failed branches and log errors
        completed = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Branch execution failed: {type(result).__name__}: {result}")
            else:
                completed.append(result)
        return completed

    def merge(self, branches: list[Branch]) -> MergeResult:
        """
        Merge completed branches and determine the best outcome.

        Args:
            branches: Completed branches to merge

        Returns:
            MergeResult with comparison and winner
        """
        if not branches:
            raise ValueError("No branches to merge")

        completed = [b for b in branches if b.is_complete]
        if not completed:
            raise ValueError("No completed branches to merge")

        # Score each branch
        branch_scores = {}
        for branch in completed:
            score = self._score_branch(branch)
            branch_scores[branch.branch_id] = score

        # Find winner
        winner_id = max(branch_scores, key=branch_scores.get)
        winner = next((b for b in completed if b.branch_id == winner_id), None)
        if not winner:
            logger.error(f"branch_winner_not_found winner_id={winner_id}")
            winner = completed[0]  # Fallback to first branch

        # Generate comparison summary
        comparison = self._generate_comparison(completed, branch_scores)

        # Extract insights from all branches
        insights = self._extract_merged_insights(completed)

        return MergeResult(
            winning_branch_id=winner_id,
            winning_hypothesis=winner.hypothesis,
            comparison_summary=comparison,
            all_branch_results={b.branch_id: b.result for b in completed},
            merged_insights=insights,
        )

    def _score_branch(self, branch: Branch) -> float:
        """Score a completed branch."""
        if not branch.result:
            return 0.0

        result = branch.result
        score = 0.0

        # Consensus is good
        if result.consensus_reached:
            score += 0.3

        # Confidence matters
        score += 0.3 * result.confidence

        # Fewer rounds is more efficient
        efficiency = 1.0 - (result.rounds_used / 10)  # Assume max 10 rounds
        score += 0.2 * max(0, efficiency)

        # Less severe critiques is better (issues were resolved)
        if result.critiques:
            avg_severity = sum(c.severity for c in result.critiques) / len(result.critiques)
            score += 0.2 * (1 - avg_severity)

        return score

    def _generate_comparison(
        self,
        branches: list[Branch],
        scores: dict[str, float],
    ) -> str:
        """Generate a comparison summary of branches."""
        lines = ["## Branch Comparison\n"]

        for branch in sorted(branches, key=lambda b: scores.get(b.branch_id, 0), reverse=True):
            score = scores.get(branch.branch_id, 0)
            result = branch.result

            lines.append(f"### {branch.hypothesis}")
            lines.append(f"- Lead: {branch.lead_agent}")
            lines.append(f"- Score: {score:.2f}")
            if result:
                lines.append(f"- Consensus: {'Yes' if result.consensus_reached else 'No'}")
                lines.append(f"- Confidence: {result.confidence:.0%}")
                lines.append(f"- Rounds: {result.rounds_used}")
            lines.append("")

        return "\n".join(lines)

    def _extract_merged_insights(self, branches: list[Branch]) -> list[str]:
        """Extract insights from all branches."""
        insights = []

        for branch in branches:
            if not branch.result:
                continue

            # Extract key points from final answer
            answer = branch.result.final_answer
            if answer:
                # Take first few sentences as insight
                sentences = answer.split(".")[:2]
                insight = f"[{branch.lead_agent}]: {'.'.join(sentences)}"
                insights.append(insight)

        return insights

    def get_fork_history(self, parent_debate_id: str) -> list[ForkPoint]:
        """Get all fork points for a debate."""
        return self.fork_points.get(parent_debate_id, [])

    def get_branches(self, parent_debate_id: str) -> list[Branch]:
        """Get all branches for a debate."""
        return self.branches.get(parent_debate_id, [])

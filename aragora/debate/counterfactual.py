"""
Counterfactual Debate Branches.

When agents reach an impasse on a foundational assumption, this module
enables forking the debate into parallel counterfactual branches:
- "Assuming X is true..." → Branch A
- "Assuming X is false..." → Branch B

Each branch runs independently, then a meta-synthesis compares outcomes
to produce conditional conclusions like:
"If X holds, we recommend Y; otherwise Z"

Key features:
- Automatic impasse detection
- Parallel branch execution
- Conditional consensus synthesis
- Decision tree output
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from aragora.core import DebateResult, Message, Vote
from aragora.debate.graph import (
    Branch,
    BranchReason,
    DebateGraph,
    MergeResult,
    MergeStrategy,
)


class CounterfactualStatus(Enum):
    """Status of a counterfactual branch."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MERGED = "merged"


@dataclass
class PivotClaim:
    """
    A claim that serves as the pivot point for counterfactual branching.

    The debate forks on whether this claim is assumed true or false.
    """

    claim_id: str
    statement: str
    author: str

    # Branching context
    disagreement_score: float  # How much agents disagree (0-1)
    importance_score: float  # How central is this claim (0-1)
    blocking_agents: list[str]  # Agents who disagree on this

    # Decision
    branch_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def should_branch(self) -> bool:
        """Determine if this claim warrants branching."""
        # Branch if disagreement is high AND claim is important
        return self.disagreement_score > 0.5 and self.importance_score > 0.3


@dataclass
class CounterfactualBranch:
    """
    A branch exploring a counterfactual world.

    Contains the assumption (claim = True/False) and the debate
    that unfolds under that assumption.
    """

    branch_id: str
    parent_debate_id: str
    pivot_claim: PivotClaim
    assumption: bool  # True = claim assumed true, False = assumed false

    # State
    status: CounterfactualStatus = CounterfactualStatus.PENDING
    messages: list[Message] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)

    # Results
    conclusion: Optional[str] = None
    confidence: float = 0.0
    consensus_reached: bool = False
    key_insights: list[str] = field(default_factory=list)

    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Graph
    graph_branch_id: Optional[str] = None  # Link to DebateGraph branch

    @property
    def assumption_text(self) -> str:
        """Human-readable assumption statement."""
        if self.assumption:
            return f"Assuming '{self.pivot_claim.statement[:100]}...' is TRUE"
        else:
            return f"Assuming '{self.pivot_claim.statement[:100]}...' is FALSE"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of branch execution."""
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return (end - start).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "parent_debate_id": self.parent_debate_id,
            "pivot_claim": {
                "claim_id": self.pivot_claim.claim_id,
                "statement": self.pivot_claim.statement,
                "author": self.pivot_claim.author,
            },
            "assumption": self.assumption,
            "assumption_text": self.assumption_text,
            "status": self.status.value,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "consensus_reached": self.consensus_reached,
            "key_insights": self.key_insights,
            "message_count": len(self.messages),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class BranchComparison:
    """Comparison of two counterfactual branch outcomes."""

    branch_a_id: str
    branch_b_id: str

    # Outcomes
    branch_a_conclusion: str
    branch_b_conclusion: str
    branch_a_confidence: float
    branch_b_confidence: float

    # Analysis
    conclusions_differ: bool
    key_differences: list[str]
    shared_insights: list[str]

    # Recommendation
    recommended_branch: Optional[str] = None  # Which branch leads to better outcome
    recommendation_reason: str = ""


@dataclass
class ConditionalConsensus:
    """
    Consensus that depends on counterfactual assumptions.

    Output format:
    "If [condition], then [conclusion A]; otherwise [conclusion B]"
    """

    consensus_id: str
    pivot_claim: PivotClaim

    # Conditional conclusions
    if_true_conclusion: str
    if_true_confidence: float
    if_false_conclusion: str
    if_false_confidence: float

    # Decision tree
    decision_tree: dict[str, Any] = field(default_factory=dict)

    # Meta-analysis
    preferred_world: Optional[bool] = None  # True/False/None
    preference_reason: str = ""
    unresolved_uncertainties: list[str] = field(default_factory=list)

    # Source branches
    true_branch_id: str = ""
    false_branch_id: str = ""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_natural_language(self) -> str:
        """Generate natural language conditional statement."""
        return (
            f"**Conditional Consensus:**\n\n"
            f"IF {self.pivot_claim.statement[:200]}\n"
            f"THEN: {self.if_true_conclusion} "
            f"(confidence: {self.if_true_confidence:.0%})\n\n"
            f"ELSE: {self.if_false_conclusion} "
            f"(confidence: {self.if_false_confidence:.0%})"
        )

    def to_dict(self) -> dict:
        return {
            "consensus_id": self.consensus_id,
            "pivot_claim": self.pivot_claim.statement,
            "if_true_conclusion": self.if_true_conclusion,
            "if_true_confidence": self.if_true_confidence,
            "if_false_conclusion": self.if_false_conclusion,
            "if_false_confidence": self.if_false_confidence,
            "decision_tree": self.decision_tree,
            "preferred_world": self.preferred_world,
            "preference_reason": self.preference_reason,
            "unresolved_uncertainties": self.unresolved_uncertainties,
            "natural_language": self.to_natural_language(),
        }


class ImpactDetector:
    """Detects when a debate has reached an impasse that warrants branching."""

    # Phrases indicating fundamental disagreement
    DISAGREEMENT_PHRASES = [
        "fundamentally disagree",
        "core assumption",
        "premise is flawed",
        "if that were true",
        "depends on whether",
        "assuming that",
        "but if",
        "on the other hand",
        "opposite conclusion",
        "cannot accept",
        "reject the premise",
    ]

    def __init__(
        self,
        disagreement_threshold: float = 0.6,
        rounds_before_branch: int = 2,
    ):
        self.disagreement_threshold = disagreement_threshold
        self.rounds_before_branch = rounds_before_branch

    def detect_impasse(
        self,
        messages: list[Message],
        votes: list[Vote],
    ) -> Optional[PivotClaim]:
        """
        Detect if the debate has reached an impasse that should trigger branching.

        Returns the claim to pivot on, or None if no impasse detected.
        """
        if len(messages) < self.rounds_before_branch * 2:
            return None

        # Look for disagreement patterns in recent messages
        recent_messages = messages[-6:]  # Last ~2 rounds
        disagreements = self._find_disagreements(recent_messages)

        if not disagreements:
            return None

        # Score each potential pivot
        best_pivot = None
        best_score = 0.0

        for claim, agents in disagreements.items():
            score = len(agents) / max(1, len(set(m.agent for m in recent_messages)))

            if score > best_score and score > self.disagreement_threshold:
                best_score = score
                best_pivot = PivotClaim(
                    claim_id=f"pivot-{uuid.uuid4().hex[:8]}",
                    statement=claim,
                    author="multiple",
                    disagreement_score=score,
                    importance_score=self._estimate_importance(claim, messages),
                    blocking_agents=list(agents),
                )

        return best_pivot

    def _find_disagreements(
        self,
        messages: list[Message],
    ) -> dict[str, set[str]]:
        """Find claims that agents disagree on."""
        disagreements: dict[str, set[str]] = {}

        for msg in messages:
            content_lower = msg.content.lower()

            # Check for disagreement phrases
            for phrase in self.DISAGREEMENT_PHRASES:
                if phrase in content_lower:
                    # Extract the disputed claim (simplified)
                    # In production, use NLP to extract actual claims
                    start = content_lower.find(phrase)
                    end = min(start + 200, len(msg.content))
                    claim_text = msg.content[start:end].strip()

                    # Find sentence boundaries
                    for delim in ".!?":
                        if delim in claim_text:
                            claim_text = claim_text[: claim_text.index(delim) + 1]
                            break

                    if len(claim_text) > 20:
                        if claim_text not in disagreements:
                            disagreements[claim_text] = set()
                        disagreements[claim_text].add(msg.agent)

        return disagreements

    def _estimate_importance(self, claim: str, messages: list[Message]) -> float:
        """Estimate how important/central a claim is."""
        # Count how often similar concepts appear
        words = set(claim.lower().split())
        mentions = 0

        for msg in messages:
            msg_words = set(msg.content.lower().split())
            overlap = len(words & msg_words) / max(1, len(words))
            if overlap > 0.3:
                mentions += 1

        return min(1.0, mentions / len(messages)) if messages else 0.0


class CounterfactualOrchestrator:
    """
    Orchestrates counterfactual debate branches.

    When an impasse is detected, forks the debate into parallel
    branches exploring different assumptions.
    """

    def __init__(
        self,
        max_branches: int = 4,
        max_depth: int = 2,
        parallel_execution: bool = True,
    ):
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.parallel_execution = parallel_execution

        self.impasse_detector = ImpactDetector()
        self.branches: dict[str, CounterfactualBranch] = {}
        self.conditional_consensuses: list[ConditionalConsensus] = []
        self.max_history: int = 100  # Limit to prevent unbounded memory growth

        self._branch_counter = 0

    async def check_and_branch(
        self,
        debate_id: str,
        messages: list[Message],
        votes: list[Vote],
        run_branch_fn: Callable,
    ) -> Optional[list[CounterfactualBranch]]:
        """
        Check if branching is warranted and execute if so.

        Args:
            debate_id: Current debate ID
            messages: Messages so far
            votes: Votes so far
            run_branch_fn: Async function to run a branch debate

        Returns:
            List of completed branches if branching occurred
        """
        # Detect impasse
        pivot = self.impasse_detector.detect_impasse(messages, votes)

        if not pivot or not pivot.should_branch:
            return None

        # Check limits
        existing_branches = [b for b in self.branches.values() if b.parent_debate_id == debate_id]
        if len(existing_branches) >= self.max_branches:
            return None

        # Create branches for both assumptions
        branches = await self.create_and_run_branches(debate_id, pivot, messages, run_branch_fn)

        return branches

    async def create_and_run_branches(
        self,
        debate_id: str,
        pivot: PivotClaim,
        context_messages: list[Message],
        run_branch_fn: Callable,
    ) -> list[CounterfactualBranch]:
        """Create and run counterfactual branches."""
        branches = []

        for assumption in [True, False]:
            self._branch_counter += 1
            branch = CounterfactualBranch(
                branch_id=f"cf-{self._branch_counter:04d}",
                parent_debate_id=debate_id,
                pivot_claim=pivot,
                assumption=assumption,
            )
            self.branches[branch.branch_id] = branch
            branches.append(branch)

        # Run branches
        if self.parallel_execution:
            # Run in parallel
            tasks = [
                self._run_branch(branch, context_messages, run_branch_fn) for branch in branches
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any exceptions (branches are already updated in _run_branch)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Counterfactual branch {i} failed: {type(result).__name__}: {result}"
                    )
        else:
            # Run sequentially
            for branch in branches:
                await self._run_branch(branch, context_messages, run_branch_fn)

        return branches

    async def _run_branch(
        self,
        branch: CounterfactualBranch,
        context_messages: list[Message],
        run_branch_fn: Callable,
    ):
        """Run a single counterfactual branch."""
        branch.status = CounterfactualStatus.RUNNING
        branch.started_at = datetime.now().isoformat()

        try:
            # Prepare context with assumption
            assumption_msg = Message(
                role="system",
                agent="counterfactual",
                content=(
                    f"**COUNTERFACTUAL ASSUMPTION**\n\n"
                    f"For this branch of the debate, please assume the following is "
                    f"{'TRUE' if branch.assumption else 'FALSE'}:\n\n"
                    f"> {branch.pivot_claim.statement}\n\n"
                    f"Proceed with this assumption and explore its implications."
                ),
                round=0,
            )

            # Run the branch debate
            result = await run_branch_fn(
                task=f"Continue debate assuming: {branch.assumption_text}",
                context=[assumption_msg] + context_messages[-10:],  # Last 10 msgs
                branch_id=branch.branch_id,
            )

            # Extract results
            if isinstance(result, DebateResult):
                branch.conclusion = result.final_answer
                branch.confidence = result.confidence
                branch.consensus_reached = result.consensus_reached
                branch.messages = result.messages
                branch.votes = result.votes

            branch.status = CounterfactualStatus.COMPLETED

        except Exception as e:
            branch.status = CounterfactualStatus.FAILED
            branch.conclusion = f"Branch failed: {str(e)}"

        finally:
            branch.completed_at = datetime.now().isoformat()

    def synthesize_branches(
        self,
        branch_true: CounterfactualBranch,
        branch_false: CounterfactualBranch,
    ) -> ConditionalConsensus:
        """
        Synthesize results from true/false branches into conditional consensus.
        """
        # Compare outcomes
        comparison = self._compare_branches(branch_true, branch_false)

        # Build decision tree
        decision_tree = {
            "condition": branch_true.pivot_claim.statement,
            "if_true": {
                "conclusion": branch_true.conclusion,
                "confidence": branch_true.confidence,
                "consensus": branch_true.consensus_reached,
            },
            "if_false": {
                "conclusion": branch_false.conclusion,
                "confidence": branch_false.confidence,
                "consensus": branch_false.consensus_reached,
            },
        }

        # Determine preference
        preferred = None
        preference_reason = ""

        if comparison.recommended_branch:
            preferred = comparison.recommended_branch == branch_true.branch_id
            preference_reason = comparison.recommendation_reason
        elif branch_true.confidence > branch_false.confidence + 0.1:
            preferred = True
            preference_reason = "Higher confidence in true-assumption branch"
        elif branch_false.confidence > branch_true.confidence + 0.1:
            preferred = False
            preference_reason = "Higher confidence in false-assumption branch"

        consensus = ConditionalConsensus(
            consensus_id=f"cc-{uuid.uuid4().hex[:8]}",
            pivot_claim=branch_true.pivot_claim,
            if_true_conclusion=branch_true.conclusion or "No conclusion reached",
            if_true_confidence=branch_true.confidence,
            if_false_conclusion=branch_false.conclusion or "No conclusion reached",
            if_false_confidence=branch_false.confidence,
            decision_tree=decision_tree,
            preferred_world=preferred,
            preference_reason=preference_reason,
            unresolved_uncertainties=comparison.key_differences,
            true_branch_id=branch_true.branch_id,
            false_branch_id=branch_false.branch_id,
        )

        self.conditional_consensuses.append(consensus)
        return consensus

    def _compare_branches(
        self,
        branch_a: CounterfactualBranch,
        branch_b: CounterfactualBranch,
    ) -> BranchComparison:
        """Compare two branch outcomes."""
        # Check if conclusions differ meaningfully
        conclusions_differ = (
            branch_a.conclusion != branch_b.conclusion
            and branch_a.conclusion
            and branch_b.conclusion
        )

        # Find key differences
        key_differences = []
        if conclusions_differ:
            key_differences.append(f"Under assumption TRUE: {branch_a.conclusion[:100]}...")
            key_differences.append(f"Under assumption FALSE: {branch_b.conclusion[:100]}...")

        # Find shared insights
        shared_insights = list(set(branch_a.key_insights) & set(branch_b.key_insights))

        # Recommend based on confidence and consensus
        recommended = None
        reason = ""

        if branch_a.consensus_reached and not branch_b.consensus_reached:
            recommended = branch_a.branch_id
            reason = "Achieved consensus while alternative did not"
        elif branch_b.consensus_reached and not branch_a.consensus_reached:
            recommended = branch_b.branch_id
            reason = "Achieved consensus while alternative did not"
        elif branch_a.confidence > branch_b.confidence + 0.15:
            recommended = branch_a.branch_id
            reason = f"Higher confidence ({branch_a.confidence:.0%} vs {branch_b.confidence:.0%})"
        elif branch_b.confidence > branch_a.confidence + 0.15:
            recommended = branch_b.branch_id
            reason = f"Higher confidence ({branch_b.confidence:.0%} vs {branch_a.confidence:.0%})"

        return BranchComparison(
            branch_a_id=branch_a.branch_id,
            branch_b_id=branch_b.branch_id,
            branch_a_conclusion=branch_a.conclusion or "",
            branch_b_conclusion=branch_b.conclusion or "",
            branch_a_confidence=branch_a.confidence,
            branch_b_confidence=branch_b.confidence,
            conclusions_differ=bool(conclusions_differ),
            key_differences=key_differences,
            shared_insights=shared_insights,
            recommended_branch=recommended,
            recommendation_reason=reason,
        )

    def get_all_consensuses(self) -> list[ConditionalConsensus]:
        """Get all conditional consensuses generated."""
        return self.conditional_consensuses

    def generate_report(self) -> str:
        """Generate a report of all counterfactual explorations."""
        lines = [
            "# Counterfactual Exploration Report",
            "",
            f"**Total Branches:** {len(self.branches)}",
            f"**Conditional Consensuses:** {len(self.conditional_consensuses)}",
            "",
        ]

        # List branches by pivot
        pivots: dict[str, dict[str, Any]] = {}
        for branch in self.branches.values():
            pivot_id = branch.pivot_claim.claim_id
            if pivot_id not in pivots:
                pivots[pivot_id] = {
                    "claim": branch.pivot_claim,
                    "branches": [],
                }
            pivots[pivot_id]["branches"].append(branch)

        for pivot_id, data in pivots.items():
            claim = data["claim"]
            lines.append(f"## Pivot: {claim.statement[:100]}...")
            lines.append("")

            for branch in data["branches"]:
                status_icon = "✅" if branch.status == CounterfactualStatus.COMPLETED else "❌"
                lines.append(
                    f"### {status_icon} Branch: Assume {'TRUE' if branch.assumption else 'FALSE'}"
                )
                lines.append(f"- **Status:** {branch.status.value}")
                lines.append(f"- **Confidence:** {branch.confidence:.0%}")
                lines.append(f"- **Consensus:** {'Yes' if branch.consensus_reached else 'No'}")
                if branch.conclusion:
                    lines.append(f"- **Conclusion:** {branch.conclusion[:200]}...")
                lines.append("")

        # Conditional consensuses
        if self.conditional_consensuses:
            lines.append("## Conditional Consensuses")
            lines.append("")
            for cc in self.conditional_consensuses:
                lines.append(cc.to_natural_language())
                lines.append("")

        return "\n".join(lines)

    def cleanup_debate(self, debate_id: str) -> int:
        """
        Clean up branches for a completed debate to prevent memory leaks.

        Call this after a debate completes to free memory from old branches.

        Args:
            debate_id: The debate ID to clean up

        Returns:
            Number of branches removed
        """
        to_delete = [bid for bid, b in self.branches.items() if b.parent_debate_id == debate_id]
        for bid in to_delete:
            del self.branches[bid]

        # Also trim conditional_consensuses to max_history
        if len(self.conditional_consensuses) > self.max_history:
            self.conditional_consensuses = self.conditional_consensuses[-self.max_history :]

        return len(to_delete)

    def clear_all(self) -> None:
        """Clear all branches and consensuses. Use with caution."""
        self.branches.clear()
        self.conditional_consensuses.clear()
        self._branch_counter = 0


class CounterfactualIntegration:
    """
    Integration layer for counterfactual branching with DebateGraph.

    Bridges the CounterfactualOrchestrator with the existing graph-based
    debate infrastructure.
    """

    def __init__(self, graph: DebateGraph):
        self.graph = graph
        self.orchestrator = CounterfactualOrchestrator()

    def create_counterfactual_branch(
        self,
        from_node_id: str,
        pivot: PivotClaim,
        assumption: bool,
    ) -> tuple[Branch, CounterfactualBranch]:
        """
        Create a counterfactual branch in both the graph and orchestrator.
        """
        # Create graph branch
        graph_branch = self.graph.create_branch(
            from_node_id=from_node_id,
            reason=BranchReason.COUNTERFACTUAL_EXPLORATION,
            name=f"CF: {pivot.statement[:30]}... = {assumption}",
            hypothesis=pivot.statement if assumption else f"NOT: {pivot.statement}",
        )

        # Create counterfactual branch
        cf_branch = CounterfactualBranch(
            branch_id=f"cf-{graph_branch.id}",
            parent_debate_id=self.graph.debate_id,
            pivot_claim=pivot,
            assumption=assumption,
            graph_branch_id=graph_branch.id,
        )

        self.orchestrator.branches[cf_branch.branch_id] = cf_branch

        return graph_branch, cf_branch

    def merge_counterfactual_branches(
        self,
        true_branch: CounterfactualBranch,
        false_branch: CounterfactualBranch,
        synthesizer_agent_id: str,
    ) -> tuple[MergeResult, ConditionalConsensus]:
        """
        Merge counterfactual branches and create conditional consensus.
        """
        # Get conditional consensus
        consensus = self.orchestrator.synthesize_branches(true_branch, false_branch)

        # Merge in graph
        branch_ids = []
        if true_branch.graph_branch_id:
            branch_ids.append(true_branch.graph_branch_id)
        if false_branch.graph_branch_id:
            branch_ids.append(false_branch.graph_branch_id)

        merge_result = None
        if len(branch_ids) >= 2:
            merge_result = self.graph.merge_branches(
                branch_ids=branch_ids,
                strategy=MergeStrategy.SYNTHESIS,
                synthesizer_agent_id=synthesizer_agent_id,
                synthesis_content=consensus.to_natural_language(),
            )

        return merge_result, consensus


# Convenience function for standalone use
async def explore_counterfactual(
    debate_id: str,
    pivot_statement: str,
    context_messages: list[Message],
    run_branch_fn: Callable,
) -> ConditionalConsensus:
    """
    Explore a counterfactual by running both true/false branches.

    Args:
        debate_id: Parent debate ID
        pivot_statement: The claim to branch on
        context_messages: Context from the parent debate
        run_branch_fn: Async function to run a branch

    Returns:
        Conditional consensus from both branches
    """
    orchestrator = CounterfactualOrchestrator()

    pivot = PivotClaim(
        claim_id=f"pivot-{uuid.uuid4().hex[:8]}",
        statement=pivot_statement,
        author="user",
        disagreement_score=1.0,
        importance_score=1.0,
        blocking_agents=[],
    )

    branches = await orchestrator.create_and_run_branches(
        debate_id=debate_id,
        pivot=pivot,
        context_messages=context_messages,
        run_branch_fn=run_branch_fn,
    )

    true_branch = next((b for b in branches if b.assumption), None)
    false_branch = next((b for b in branches if not b.assumption), None)

    if not true_branch or not false_branch:
        logger.error(
            f"counterfactual_missing_branch true={true_branch is not None} false={false_branch is not None}"
        )
        raise ValueError("Counterfactual analysis requires both true and false branches")

    return orchestrator.synthesize_branches(true_branch, false_branch)

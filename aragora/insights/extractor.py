"""
Insight Extractor - Extract actionable insights from debate results.

Analyzes completed debates to extract:
- Winning argument patterns
- Failure modes
- Agent performance metrics
- Convergence dynamics
- Decision-making patterns
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from aragora.serialization import SerializableMixin


class InsightType(Enum):
    """Types of insights that can be extracted."""

    CONSENSUS = "consensus"  # The final decision and how it was reached
    DISSENT = "dissent"  # Minority views and their reasoning
    PATTERN = "pattern"  # Recurring argument patterns
    CONVERGENCE = "convergence"  # How views converged/diverged over time
    AGENT_PERFORMANCE = "agent_perf"  # Individual agent contributions
    FAILURE_MODE = "failure_mode"  # Why consensus wasn't reached
    DECISION_PROCESS = "decision"  # How the final decision was made


@dataclass
class Insight(SerializableMixin):
    """A structured insight extracted from a debate."""

    id: str
    type: InsightType
    title: str
    description: str
    confidence: float  # 0-1 confidence in this insight

    # Context
    debate_id: str
    agents_involved: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # Message IDs or content snippets

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    # to_dict() inherited from SerializableMixin


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent in a debate."""

    agent_name: str
    proposals_made: int = 0
    critiques_given: int = 0
    critiques_received: int = 0
    proposal_accepted: bool = False
    vote_aligned_with_consensus: bool = False
    average_critique_severity: float = 0.0
    contribution_score: float = 0.5  # 0-1 overall contribution


@dataclass
class DebateInsights:
    """Collection of all insights from a single debate."""

    debate_id: str
    task: str
    consensus_reached: bool
    duration_seconds: float

    # Extracted insights
    consensus_insight: Optional[Insight] = None
    dissent_insights: list[Insight] = field(default_factory=list)
    pattern_insights: list[Insight] = field(default_factory=list)
    convergence_insight: Optional[Insight] = None
    failure_mode_insight: Optional[Insight] = None
    decision_insight: Optional[Insight] = None

    # Agent performance
    agent_performances: list[AgentPerformance] = field(default_factory=list)

    # Summary
    total_insights: int = 0
    key_takeaway: str = ""

    def all_insights(self) -> list[Insight]:
        """Get all insights as a flat list."""
        insights = []
        if self.consensus_insight:
            insights.append(self.consensus_insight)
        insights.extend(self.dissent_insights)
        insights.extend(self.pattern_insights)
        if self.convergence_insight:
            insights.append(self.convergence_insight)
        if self.failure_mode_insight:
            insights.append(self.failure_mode_insight)
        if self.decision_insight:
            insights.append(self.decision_insight)
        return insights


class InsightExtractor:
    """
    Extracts structured insights from debate results.

    Usage:
        extractor = InsightExtractor()
        insights = await extractor.extract(debate_result)

        for insight in insights.all_insights():
            print(f"{insight.type.value}: {insight.title}")
    """

    # Issue categories (from CritiqueStore)
    ISSUE_CATEGORIES = {
        "performance": ["slow", "performance", "efficient", "optimize", "speed", "latency"],
        "security": ["security", "vulnerab", "injection", "auth", "permission", "xss", "csrf"],
        "correctness": ["bug", "error", "incorrect", "wrong", "fail", "break", "crash"],
        "clarity": ["unclear", "confusing", "readab", "document", "comment", "naming"],
        "architecture": ["design", "structure", "pattern", "modular", "coupling", "cohesion"],
        "completeness": ["missing", "incomplete", "todo", "edge case", "handle"],
        "testing": ["test", "coverage", "assert", "mock", "unit", "integration"],
    }

    def __init__(self):
        pass

    async def extract(self, result) -> DebateInsights:
        """
        Extract all insights from a debate result.

        Args:
            result: DebateResult object

        Returns:
            DebateInsights containing all extracted insights
        """
        debate_id = getattr(result, "id", hashlib.sha256(str(result).encode()).hexdigest()[:16])
        task = getattr(result, "task", "Unknown task")

        insights = DebateInsights(
            debate_id=debate_id,
            task=task[:200] if task else "",
            consensus_reached=getattr(result, "consensus_reached", False),
            duration_seconds=getattr(result, "duration_seconds", 0),
        )

        # Extract each type of insight
        insights.consensus_insight = self._extract_consensus_insight(result, debate_id)
        insights.dissent_insights = self._extract_dissent_insights(result, debate_id)
        insights.pattern_insights = self._extract_pattern_insights(result, debate_id)
        insights.convergence_insight = self._extract_convergence_insight(result, debate_id)
        insights.decision_insight = self._extract_decision_insight(result, debate_id)

        # If consensus not reached, extract failure mode
        if not insights.consensus_reached:
            insights.failure_mode_insight = self._extract_failure_mode(result, debate_id)

        # Extract agent performances
        insights.agent_performances = self._extract_agent_performances(result)

        # Calculate summary
        insights.total_insights = len(insights.all_insights())
        insights.key_takeaway = self._generate_key_takeaway(insights)

        return insights

    def _extract_consensus_insight(self, result, debate_id: str) -> Optional[Insight]:
        """Extract the consensus decision insight."""
        if not getattr(result, "consensus_reached", False):
            return None

        final_answer = getattr(result, "final_answer", "")
        confidence = getattr(result, "confidence", 0.5)
        strength = getattr(result, "consensus_strength", "unknown")

        # Summarize the decision
        summary = final_answer[:500] if final_answer else "No answer recorded"

        return Insight(
            id=f"{debate_id}_consensus",
            type=InsightType.CONSENSUS,
            title=f"Consensus Reached ({strength})",
            description=f"The debate reached consensus with {confidence:.0%} confidence. "
            f"Decision: {summary}...",
            confidence=confidence,
            debate_id=debate_id,
            agents_involved=self._get_agent_names(result),
            evidence=[final_answer[:200]] if final_answer else [],
            metadata={
                "consensus_strength": strength,
                "confidence": confidence,
                "rounds_used": getattr(result, "rounds_used", 0),
            },
        )

    def _extract_dissent_insights(self, result, debate_id: str) -> list[Insight]:
        """Extract insights from dissenting views."""
        dissenting_views = getattr(result, "dissenting_views", [])
        insights = []

        for i, view in enumerate(dissenting_views):
            # Parse agent name from view format: "[agent]: content"
            agent_match = re.match(r"\[([^\]]+)\]:\s*(.+)", view, re.DOTALL)
            if agent_match:
                agent_name = agent_match.group(1)
                content = agent_match.group(2)
            else:
                agent_name = f"agent_{i}"
                content = view

            insights.append(
                Insight(
                    id=f"{debate_id}_dissent_{i}",
                    type=InsightType.DISSENT,
                    title=f"Dissent from {agent_name}",
                    description=f"Alternative view: {content[:300]}...",
                    confidence=0.6,  # Lower confidence for dissenting views
                    debate_id=debate_id,
                    agents_involved=[agent_name],
                    evidence=[content[:200]],
                    metadata={"dissent_index": i},
                )
            )

        return insights

    def _extract_pattern_insights(self, result, debate_id: str) -> list[Insight]:
        """Extract recurring patterns from critiques."""
        critiques = getattr(result, "critiques", [])
        insights = []

        # Group critiques by issue category
        category_counts: dict[str, list[dict]] = {cat: [] for cat in self.ISSUE_CATEGORIES}

        for critique in critiques:
            issues = getattr(critique, "issues", [])
            for issue in issues:
                category = self._categorize_issue(issue)
                if category:
                    category_counts[category].append(
                        {
                            "issue": issue,
                            "severity": getattr(critique, "severity", 0.5),
                            "agent": getattr(critique, "agent", "unknown"),
                        }
                    )

        # Create insight for significant patterns (2+ occurrences)
        for category, occurrences in category_counts.items():
            if len(occurrences) >= 2:
                avg_severity = sum(o["severity"] for o in occurrences) / len(occurrences)
                agents = list(set(o["agent"] for o in occurrences))

                insights.append(
                    Insight(
                        id=f"{debate_id}_pattern_{category}",
                        type=InsightType.PATTERN,
                        title=f"Recurring {category.title()} Issues",
                        description=f"Multiple agents raised {category} concerns "
                        f"(avg severity: {avg_severity:.1f}). "
                        f"Issues: {', '.join(o['issue'][:50] for o in occurrences[:3])}",
                        confidence=min(0.9, 0.5 + len(occurrences) * 0.1),
                        debate_id=debate_id,
                        agents_involved=agents,
                        evidence=[o["issue"] for o in occurrences[:5]],
                        metadata={
                            "category": category,
                            "occurrence_count": len(occurrences),
                            "avg_severity": avg_severity,
                        },
                    )
                )

        return insights

    def _extract_convergence_insight(self, result, debate_id: str) -> Optional[Insight]:
        """Extract insight about how views converged or diverged."""
        messages = getattr(result, "messages", [])
        if len(messages) < 3:
            return None

        # Analyze message length trends (proxy for convergence)
        early_lengths = [len(str(m)) for m in messages[: len(messages) // 2]]
        late_lengths = [len(str(m)) for m in messages[len(messages) // 2 :]]

        avg_early = sum(early_lengths) / len(early_lengths) if early_lengths else 0
        avg_late = sum(late_lengths) / len(late_lengths) if late_lengths else 0

        if avg_late < avg_early * 0.7:
            convergence_type = "strong_convergence"
            description = "Agents significantly reduced their response lengths, indicating convergence toward consensus."
        elif avg_late < avg_early * 0.9:
            convergence_type = "mild_convergence"
            description = "Some convergence observed as responses became more focused."
        elif avg_late > avg_early * 1.3:
            convergence_type = "divergence"
            description = "Responses grew longer over time, suggesting increasing disagreement."
        else:
            convergence_type = "stable"
            description = "Response patterns remained stable throughout the debate."

        variance = getattr(result, "consensus_variance", 0)

        return Insight(
            id=f"{debate_id}_convergence",
            type=InsightType.CONVERGENCE,
            title=f"Debate Dynamics: {convergence_type.replace('_', ' ').title()}",
            description=description + f" Consensus variance: {variance:.2f}",
            confidence=0.7,
            debate_id=debate_id,
            agents_involved=self._get_agent_names(result),
            metadata={
                "convergence_type": convergence_type,
                "early_avg_length": avg_early,
                "late_avg_length": avg_late,
                "variance": variance,
            },
        )

    def _extract_decision_insight(self, result, debate_id: str) -> Optional[Insight]:
        """Extract insight about how the decision was made."""
        votes = getattr(result, "votes", [])
        consensus_mode = "unknown"

        # Try to detect consensus mode from result
        if hasattr(result, "consensus_strength"):
            strength = result.consensus_strength
            if strength == "unanimous":
                consensus_mode = "unanimous"
            elif len(votes) > 0:
                consensus_mode = "majority"
            else:
                consensus_mode = "judge"

        vote_summary = ""
        if votes:
            vote_choices: dict[str, int] = {}
            for v in votes:
                choice = getattr(v, "choice", "unknown")
                vote_choices[choice] = vote_choices.get(choice, 0) + 1
            vote_summary = ", ".join(f"{c}: {n}" for c, n in vote_choices.items())

        return Insight(
            id=f"{debate_id}_decision",
            type=InsightType.DECISION_PROCESS,
            title=f"Decision via {consensus_mode.title()}",
            description=f"Final decision reached through {consensus_mode} mechanism. "
            f"Votes: {vote_summary or 'N/A'}. "
            f"Rounds used: {getattr(result, 'rounds_used', 'unknown')}.",
            confidence=0.9,
            debate_id=debate_id,
            agents_involved=self._get_agent_names(result),
            metadata={
                "consensus_mode": consensus_mode,
                "vote_count": len(votes),
                "rounds_used": getattr(result, "rounds_used", 0),
            },
        )

    def _extract_failure_mode(self, result, debate_id: str) -> Optional[Insight]:
        """Extract insight about why consensus wasn't reached."""
        votes = getattr(result, "votes", [])
        critiques = getattr(result, "critiques", [])

        # Analyze failure reasons
        failure_reasons = []

        # Check for vote fragmentation
        if votes:
            unique_choices = set(getattr(v, "choice", "") for v in votes)
            if len(unique_choices) > 2:
                failure_reasons.append("vote fragmentation (many different choices)")

        # Check for high-severity critiques
        high_severity = [c for c in critiques if getattr(c, "severity", 0) > 0.7]
        if len(high_severity) > len(critiques) / 2:
            failure_reasons.append("persistent high-severity issues")

        # Check for unresolved dissent
        dissenting = getattr(result, "dissenting_views", [])
        if len(dissenting) >= 2:
            failure_reasons.append(f"{len(dissenting)} unresolved dissenting views")

        reason_str = "; ".join(failure_reasons) if failure_reasons else "unknown factors"

        return Insight(
            id=f"{debate_id}_failure",
            type=InsightType.FAILURE_MODE,
            title="Consensus Not Reached",
            description=f"The debate failed to reach consensus due to: {reason_str}. "
            f"Final confidence: {getattr(result, 'confidence', 0):.0%}.",
            confidence=0.8,
            debate_id=debate_id,
            agents_involved=self._get_agent_names(result),
            metadata={
                "failure_reasons": failure_reasons,
                "vote_count": len(votes),
                "critique_count": len(critiques),
                "dissent_count": len(dissenting),
            },
        )

    def _extract_agent_performances(self, result) -> list[AgentPerformance]:
        """Extract performance metrics for each agent."""
        performances = {}
        final_answer = getattr(result, "final_answer", "")

        # Count proposals (from messages)
        messages = getattr(result, "messages", [])
        for msg in messages:
            agent = (
                getattr(msg, "agent", None) or msg.get("agent", "unknown")
                if isinstance(msg, dict)
                else "unknown"
            )
            if agent not in performances:
                performances[agent] = AgentPerformance(agent_name=agent)
            performances[agent].proposals_made += 1

        # Count critiques
        critiques = getattr(result, "critiques", [])
        for critique in critiques:
            agent = getattr(critique, "agent", "unknown")
            target = getattr(critique, "target_agent", "unknown")
            severity = getattr(critique, "severity", 0.5)

            if agent not in performances:
                performances[agent] = AgentPerformance(agent_name=agent)
            performances[agent].critiques_given += 1

            if target in performances:
                performances[target].critiques_received += 1
                # Update average severity
                perf = performances[target]
                n = perf.critiques_received
                perf.average_critique_severity = (
                    perf.average_critique_severity * (n - 1) + severity
                ) / n

        # Check vote alignment
        votes = getattr(result, "votes", [])
        winning_choice = None
        if votes and final_answer:
            # Try to identify winning choice
            vote_counts: dict[str, int] = {}
            for v in votes:
                choice = getattr(v, "choice", "")
                vote_counts[choice] = vote_counts.get(choice, 0) + 1
            if vote_counts:
                winning_choice = max(vote_counts, key=lambda x: vote_counts.get(x, 0))

        for v in votes:
            agent = getattr(v, "agent", "unknown")
            choice = getattr(v, "choice", "")
            if agent in performances:
                performances[agent].vote_aligned_with_consensus = choice == winning_choice

        # Check if proposal was accepted (agent name in final answer)
        for agent, perf in performances.items():
            if agent.lower() in final_answer.lower():
                perf.proposal_accepted = True

        # Calculate contribution scores
        for perf in performances.values():
            score = 0.5  # Base score
            if perf.proposal_accepted:
                score += 0.3
            if perf.vote_aligned_with_consensus:
                score += 0.1
            if perf.critiques_given > 0:
                score += 0.1 * min(perf.critiques_given, 3) / 3
            if perf.average_critique_severity > 0.5:
                score -= 0.05  # Received harsh critiques
            perf.contribution_score = min(1.0, max(0.0, score))

        return list(performances.values())

    def _get_agent_names(self, result) -> list[str]:
        """Extract all agent names from a result."""
        agents = set()

        for msg in getattr(result, "messages", []):
            agent = (
                getattr(msg, "agent", None) or msg.get("agent") if isinstance(msg, dict) else None
            )
            if agent:
                agents.add(agent)

        for critique in getattr(result, "critiques", []):
            agent = getattr(critique, "agent", None)
            if agent:
                agents.add(agent)

        for vote in getattr(result, "votes", []):
            agent = getattr(vote, "agent", None)
            if agent:
                agents.add(agent)

        return list(agents)

    def _categorize_issue(self, issue: str) -> Optional[str]:
        """Categorize an issue string."""
        issue_lower = issue.lower()

        for category, keywords in self.ISSUE_CATEGORIES.items():
            if any(kw in issue_lower for kw in keywords):
                return category

        return None

    def _generate_key_takeaway(self, insights: DebateInsights) -> str:
        """Generate a one-line key takeaway from the insights."""
        if insights.consensus_reached:
            strength = (
                insights.consensus_insight.metadata.get("consensus_strength", "unknown")
                if insights.consensus_insight
                else "unknown"
            )
            return f"Reached {strength} consensus after {insights.duration_seconds:.0f}s"
        else:
            if insights.failure_mode_insight:
                reasons = insights.failure_mode_insight.metadata.get("failure_reasons", [])
                if reasons:
                    return f"No consensus: {reasons[0]}"
            return "Failed to reach consensus"

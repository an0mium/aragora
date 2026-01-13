"""
Context formatters for nomic loop prompt injection.

Provides formatted context from various memory and tracking systems
for injection into agent prompts. Each formatter handles graceful
degradation when its data source is unavailable.

Systems integrated:
- CritiqueStore: Pattern success/failure tracking
- ContinuumMemory: Multi-timescale strategic patterns
- ConsensusMemory: Historical debate conclusions
- DissentRetriever: Unaddressed concerns and contrarian views
- EloSystem: Agent calibration and track records
- RelationshipTracker: Agent dynamics and influence
- PositionTracker: Agent position history
- IntrospectionAPI: Agent self-awareness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FormatterDependencies:
    """Container for formatter dependencies.

    All dependencies are optional - formatters return empty strings
    when their dependencies are unavailable.
    """

    critique_store: Any = None
    continuum: Any = None
    consensus_memory: Any = None
    dissent_retriever: Any = None
    elo_system: Any = None
    relationship_tracker: Any = None
    position_tracker: Any = None
    memory_stream: Any = None
    introspection_api: Any = None

    # Feature flags
    continuum_available: bool = False
    consensus_memory_available: bool = False
    elo_available: bool = False
    grounded_personas_available: bool = False
    introspection_available: bool = False


class ContextFormatter:
    """
    Formats context from various systems for prompt injection.

    All methods return empty strings when their dependencies are
    unavailable, allowing graceful degradation.

    Usage:
        deps = FormatterDependencies(
            critique_store=loop.critique_store,
            continuum=loop.continuum,
            ...
        )
        formatter = ContextFormatter(deps, log_fn=loop._log)

        # Get successful patterns
        patterns = formatter.format_successful_patterns(limit=5)

        # Get full learning context
        context = formatter.build_learning_context(topic)
    """

    def __init__(
        self,
        deps: FormatterDependencies,
        log_fn: Optional[callable] = None,
    ):
        """
        Initialize context formatter.

        Args:
            deps: Container with all optional dependencies
            log_fn: Optional logging function
        """
        self.deps = deps
        self._log = log_fn or (lambda msg: logger.info(msg))

    def format_successful_patterns(self, limit: int = 5) -> str:
        """Format successful critique patterns for prompt injection.

        Retrieves patterns from the CritiqueStore that have led to
        successful fixes in previous debates.
        """
        if not self.deps.critique_store:
            return ""

        try:
            patterns = self.deps.critique_store.retrieve_patterns(min_success=2, limit=limit)
            if not patterns:
                return ""

            lines = ["## SUCCESSFUL PATTERNS (from past debates)"]
            lines.append("These critique patterns have worked well before:\n")

            for p in patterns:
                lines.append(f"- **{p.issue_type}**: {p.issue_text}")
                if p.suggestion_text:
                    lines.append(f"  → Fix: {p.suggestion_text}")
                lines.append(f"  ({p.success_count} successes)")

            return "\n".join(lines)
        except Exception:
            return ""

    def format_failure_patterns(self, limit: int = 5) -> str:
        """Format failure patterns to avoid repeating mistakes.

        Uses failure tracking to show patterns that have NOT worked well,
        so agents can avoid repeating them.
        """
        if not self.deps.critique_store:
            return ""

        try:
            # Query patterns with high failure rates
            conn = getattr(self.deps.critique_store, "conn", None)
            if not conn:
                import sqlite3

                conn = sqlite3.connect(self.deps.critique_store.db_path)

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT issue_type, issue_text, failure_count, success_count
                FROM patterns
                WHERE failure_count > 0
                ORDER BY failure_count DESC
                LIMIT ?
                """,
                (limit,),
            )
            failures = cursor.fetchall()

            if not failures:
                return ""

            lines = ["## PATTERNS TO AVOID (learned from past failures)"]
            lines.append("These approaches have NOT worked well:\n")

            for issue_type, issue_text, fail_count, success_count in failures:
                total = success_count + fail_count
                success_rate = success_count / total if total > 0 else 0
                if success_rate < 0.5:  # Only show patterns with <50% success
                    lines.append(f"- **{issue_type}**: {issue_text}")
                    lines.append(f"  ({fail_count} failures, {success_rate:.0%} success rate)")

            return "\n".join(lines) if len(lines) > 2 else ""
        except Exception:
            return ""

    def format_continuum_patterns(self, limit: int = 5) -> str:
        """Format patterns from ContinuumMemory for prompt injection.

        Retrieves strategic patterns from the SLOW tier that capture
        successful cycle outcomes and learnings across time.
        """
        if not self.deps.continuum or not self.deps.continuum_available:
            return ""

        try:
            # Import MemoryTier locally to avoid import issues
            from aragora.memory.continuum import MemoryTier

            # Get recent successful patterns from SLOW tier
            memories = self.deps.continuum.export_for_tier(MemoryTier.SLOW)
            if not memories:
                return ""

            # Filter to successful patterns and sort by importance
            successful = [m for m in memories if m.get("metadata", {}).get("success", False)]
            successful = sorted(successful, key=lambda x: x.get("importance", 0), reverse=True)[
                :limit
            ]

            if not successful:
                return ""

            lines = ["## STRATEGIC PATTERNS (from ContinuumMemory)"]
            lines.append("Successful patterns learned across cycles:\n")

            for m in successful:
                content = m.get("content", "")
                cycle = m.get("metadata", {}).get("cycle", "?")
                lines.append(f"- Cycle {cycle}: {content}")

            return "\n".join(lines)
        except Exception:
            return ""

    def format_consensus_history(self, topic: str, limit: int = 3) -> str:
        """Format prior consensus decisions for prompt injection.

        Retrieves similar past debates and their conclusions to avoid
        rehashing settled topics and to surface unaddressed dissents.
        """
        if not self.deps.consensus_memory or not self.deps.consensus_memory_available:
            return ""

        try:
            # Find similar past debates
            similar = self.deps.consensus_memory.find_similar_debates(topic, limit=limit)
            if not similar:
                return ""

            lines = ["## HISTORICAL CONSENSUS (from past debates)"]
            lines.append("Previous debates on similar topics:\n")

            for s in similar:
                strength = s.consensus.strength.value if s.consensus.strength else "unknown"
                lines.append(
                    f"- **{s.consensus.topic}** ({strength}, {s.similarity_score:.0%} similar)"
                )
                lines.append(f"  Decision: {s.consensus.conclusion}")
                if s.dissents:
                    lines.append(f"  ⚠️ {len(s.dissents)} dissenting view(s) - consider addressing")

            # Add unaddressed dissents if retriever available
            if self.deps.dissent_retriever:
                context = self.deps.dissent_retriever.retrieve_for_new_debate(topic)
                if context.get("unacknowledged_dissents"):
                    lines.append("\n### Unaddressed Historical Concerns")
                    for d in context["unacknowledged_dissents"][:3]:
                        lines.append(f"- [{d['dissent_type']}] {d['content']}")

                # Add contrarian views
                contrarian = self.deps.dissent_retriever.find_contrarian_views(topic, limit=3)
                if contrarian:
                    lines.append("\n### Contrarian Perspectives (Devil's Advocate)")
                    for c in contrarian:
                        lines.append(f"- {c.content} (from {c.agent_id})")

            return "\n".join(lines)
        except Exception:
            return ""

    def format_agent_reputations(self) -> str:
        """Format agent reputations for prompt injection.

        Shows which agents have been most successful so agents can
        weight their collaboration accordingly.
        """
        if not self.deps.critique_store:
            return ""

        try:
            reputations = self.deps.critique_store.get_all_reputations()
            if not reputations:
                return ""

            lines = ["## AGENT TRACK RECORDS"]
            for rep in sorted(reputations, key=lambda r: r.score, reverse=True):
                if rep.proposals_made > 0:
                    acceptance = rep.proposals_accepted / rep.proposals_made
                    lines.append(
                        f"- {rep.agent_name}: {acceptance:.0%} proposal acceptance "
                        f"({rep.proposals_accepted}/{rep.proposals_made})"
                    )

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def format_relationship_network(self, limit: int = 3) -> str:
        """Format agent relationship dynamics for debate context."""
        if not self.deps.relationship_tracker or not self.deps.grounded_personas_available:
            return ""

        try:
            lines = ["## Inter-Agent Dynamics"]

            # Get influence network per agent
            agents = ["gemini", "claude", "codex", "grok"]
            if hasattr(self.deps.relationship_tracker, "get_influence_network"):
                lines.append("\n### Influence Patterns:")
                influence_scores = []
                for agent in agents:
                    try:
                        network = self.deps.relationship_tracker.get_influence_network(agent)
                        if network and network.get("influences"):
                            total_influence = sum(score for _, score in network["influences"])
                            influence_scores.append((agent, total_influence))
                    except Exception:
                        continue

                influence_scores.sort(key=lambda x: x[1], reverse=True)
                for agent, score in influence_scores[:limit]:
                    lines.append(f"- {agent}: influence score {score:.2f}")

            # Get rivals and allies
            dynamics_found = False
            for agent in agents:
                if hasattr(self.deps.relationship_tracker, "get_rivals"):
                    rivals = self.deps.relationship_tracker.get_rivals(agent, limit=2)
                    allies = (
                        self.deps.relationship_tracker.get_allies(agent, limit=2)
                        if hasattr(self.deps.relationship_tracker, "get_allies")
                        else []
                    )
                    if rivals or allies:
                        dynamics_found = True
                        rival_names = [r[0] for r in rivals] if rivals else []
                        ally_names = [a[0] for a in allies] if allies else []
                        lines.append(f"- {agent}: rivals={rival_names}, allies={ally_names}")

            return "\n".join(lines) if len(lines) > 1 and dynamics_found else ""
        except Exception as e:
            self._log(f"  [relationships] Formatting error: {e}")
            return ""

    def audit_agent_calibration(self) -> str:
        """Audit agent calibration and flag poorly calibrated agents."""
        if not self.deps.elo_system or not self.deps.elo_available:
            return ""

        try:
            lines = ["## Calibration Health Check"]
            flagged = []

            for agent_name in ["gemini", "claude", "codex", "grok"]:
                if hasattr(self.deps.elo_system, "get_expected_calibration_error"):
                    ece = self.deps.elo_system.get_expected_calibration_error(agent_name)
                    if ece and ece > 0.2:  # Poorly calibrated
                        flagged.append((agent_name, ece))
                        lines.append(
                            f"- WARNING: {agent_name} has high calibration error ({ece:.2f})"
                        )
                        lines.append("  Consider weighing their opinions lower on uncertain topics")

            if flagged:
                self._log(f"  [calibration] Flagged {len(flagged)} poorly calibrated agents")
                return "\n".join(lines)
            return ""
        except Exception as e:
            self._log(f"  [calibration] Audit error: {e}")
            return ""

    def format_agent_memories(self, agent_name: str, task: str, limit: int = 3) -> str:
        """Format agent-specific memories relevant to current task."""
        if not self.deps.memory_stream:
            return ""

        try:
            memories = self.deps.memory_stream.retrieve(
                agent_name=agent_name,
                query=task,
                limit=limit,
            )
            if not memories:
                return ""

            lines = [f"## Your Recent Memories ({agent_name})"]
            for m in memories:
                lines.append(f"- {m.content}")

            return "\n".join(lines)
        except Exception:
            return ""

    def format_position_history(self, agent_name: str, topic: str, limit: int = 5) -> str:
        """Format agent's position history on similar topics."""
        if not self.deps.position_tracker:
            return ""

        try:
            positions = self.deps.position_tracker.get_positions(
                agent_name=agent_name,
                topic=topic,
                limit=limit,
            )
            if not positions:
                return ""

            lines = [f"## Your Position History ({agent_name})"]
            for p in positions:
                lines.append(f"- {p.topic}: {p.stance} (confidence: {p.confidence:.0%})")

            return "\n".join(lines)
        except Exception:
            return ""

    def build_learning_context(
        self,
        topic: str = "",
        include_patterns: bool = True,
        include_consensus: bool = True,
        include_reputations: bool = True,
        include_calibration: bool = True,
        include_relationships: bool = True,
    ) -> str:
        """Build comprehensive learning context for prompt injection.

        Combines all available context sources into a single formatted string.

        Args:
            topic: Current debate topic for relevant lookups
            include_*: Flags to include/exclude specific context types

        Returns:
            Combined context string, or empty string if nothing available
        """
        sections = []

        if include_patterns:
            if patterns := self.format_successful_patterns():
                sections.append(patterns)
            if failures := self.format_failure_patterns():
                sections.append(failures)
            if continuum := self.format_continuum_patterns():
                sections.append(continuum)

        if include_consensus and topic:
            if consensus := self.format_consensus_history(topic):
                sections.append(consensus)

        if include_reputations:
            if reps := self.format_agent_reputations():
                sections.append(reps)

        if include_calibration:
            if calibration := self.audit_agent_calibration():
                sections.append(calibration)

        if include_relationships:
            if rels := self.format_relationship_network():
                sections.append(rels)

        if not sections:
            return ""

        return "\n\n".join(sections)


__all__ = [
    "FormatterDependencies",
    "ContextFormatter",
]

"""
Arena helper functions and utilities.

Extracted from orchestrator.py to reduce file size while maintaining
reusable logic for debate orchestration.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from aragora.core import Agent, Vote
from aragora.utils.cache_registry import register_lru_cache

if TYPE_CHECKING:
    from aragora.reasoning.citations import CitationExtractor

logger = logging.getLogger(__name__)


@register_lru_cache
@lru_cache(maxsize=1024)
def compute_domain_from_task(task_lower: str) -> str:
    """Compute domain from lowercased task string.

    Module-level cached helper to avoid O(n) string matching
    for repeated task strings across debate instances.
    """
    if any(w in task_lower for w in ("security", "hack", "vulnerability", "auth", "encrypt")):
        return "security"
    if any(w in task_lower for w in ("performance", "speed", "optimize", "cache", "latency")):
        return "performance"
    if any(w in task_lower for w in ("test", "testing", "coverage", "regression")):
        return "testing"
    if any(w in task_lower for w in ("design", "architecture", "pattern", "structure")):
        return "architecture"
    if any(w in task_lower for w in ("bug", "error", "fix", "crash", "exception")):
        return "debugging"
    if any(w in task_lower for w in ("api", "endpoint", "rest", "graphql")):
        return "api"
    if any(w in task_lower for w in ("database", "sql", "query", "schema")):
        return "database"
    if any(w in task_lower for w in ("ui", "frontend", "react", "css", "layout")):
        return "frontend"
    return "general"


class CitationHelper:
    """Helper for citation extraction and logging."""

    def __init__(self, citation_extractor: Optional["CitationExtractor"] = None):
        self._citation_extractor = citation_extractor

    @staticmethod
    def has_high_priority_needs(needs: list[dict]) -> list[dict]:
        """Filter citation needs to high-priority items only."""
        return [n for n in needs if n.get("priority") == "high"]

    @staticmethod
    def log_citation_needs(agent_name: str, needs: list[dict]) -> None:
        """Log high-priority citation needs for an agent if any exist."""
        high_priority = CitationHelper.has_high_priority_needs(needs)
        if high_priority:
            logger.debug(f"citations_needed agent={agent_name} count={len(high_priority)}")

    def extract_citation_needs(self, proposals: dict[str, str]) -> dict[str, list[dict]]:
        """Extract claims that need citations from all proposals.

        Heavy3-inspired: Identifies statements that should be backed by evidence.
        """
        if not self._citation_extractor:
            return {}

        citation_needs = {}
        for agent_name, proposal in proposals.items():
            needs = self._citation_extractor.identify_citation_needs(proposal)
            if needs:
                citation_needs[agent_name] = needs
                self.log_citation_needs(agent_name, needs)

        return citation_needs


class QualityFilterHelper:
    """Helper for ML-based quality filtering."""

    def __init__(
        self,
        enable_quality_gates: bool = False,
        quality_gate: Any = None,
    ):
        self._enable_quality_gates = enable_quality_gates
        self._quality_gate = quality_gate

    def filter_responses_by_quality(
        self, responses: list[tuple[str, str]], context: str = ""
    ) -> list[tuple[str, str]]:
        """Filter responses using ML quality gate if enabled.

        Args:
            responses: List of (agent_name, response_text) tuples
            context: Optional task context for quality assessment

        Returns:
            Filtered list containing only high-quality responses
        """
        if not self._enable_quality_gates or not self._quality_gate:
            return responses

        try:
            filtered = self._quality_gate.filter_responses(responses, context=context)
            removed = len(responses) - len(filtered)
            if removed > 0:
                logger.debug(f"[ml] Quality gate filtered {removed} low-quality responses")
            return filtered
        except Exception as e:
            logger.warning(f"[ml] Quality gate failed, keeping all responses: {e}")
            return responses


class ConsensusEstimationHelper:
    """Helper for ML-based consensus estimation."""

    def __init__(
        self,
        enable_consensus_estimation: bool = False,
        consensus_estimator: Any = None,
    ):
        self._enable_consensus_estimation = enable_consensus_estimation
        self._consensus_estimator = consensus_estimator

    def should_terminate_early(
        self,
        responses: list[tuple[str, str]],
        current_round: int,
        total_rounds: int,
        context: str,
    ) -> bool:
        """Check if debate should terminate early based on consensus estimation.

        Args:
            responses: List of (agent_name, response_text) tuples
            current_round: Current debate round number
            total_rounds: Total number of rounds
            context: Task context

        Returns:
            True if consensus is highly likely and safe to terminate early
        """
        if not self._enable_consensus_estimation or not self._consensus_estimator:
            return False

        try:
            should_stop = self._consensus_estimator.should_terminate_early(
                responses=responses,
                current_round=current_round,
                total_rounds=total_rounds,
                context=context,
            )
            if should_stop:
                logger.info(
                    f"[ml] Consensus estimator recommends early termination at round "
                    f"{current_round}/{total_rounds}"
                )
            return should_stop
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"[ml] Consensus estimation failed with data error: {e}")
            return False
        except Exception as e:
            logger.exception(f"[ml] Unexpected consensus estimation error: {e}")
            return False


class TeamSelectionHelper:
    """Helper for ML-based team selection."""

    def __init__(
        self,
        enable_ml_delegation: bool = False,
        ml_delegation_strategy: Any = None,
        use_performance_selection: bool = False,
        agent_pool: Any = None,
    ):
        self._enable_ml_delegation = enable_ml_delegation
        self._ml_delegation_strategy = ml_delegation_strategy
        self._use_performance_selection = use_performance_selection
        self._agent_pool = agent_pool

    def select_debate_team(
        self,
        requested_agents: list[Agent],
        task: str,
        domain: str,
        protocol: Any,
    ) -> list[Agent]:
        """Select debate team using ML delegation or AgentPool.

        Priority:
        1. ML delegation (if enable_ml_delegation=True)
        2. Performance selection via AgentPool (if use_performance_selection=True)
        3. Original requested agents
        """
        # ML-based agent selection takes priority
        if self._enable_ml_delegation and self._ml_delegation_strategy:
            try:
                selected = self._ml_delegation_strategy.select_agents(
                    task=task,
                    agents=requested_agents,
                    context={
                        "domain": domain,
                        "protocol": protocol,
                    },
                    max_agents=len(requested_agents),
                )
                logger.debug(
                    f"[ml] Selected {len(selected)} agents via ML delegation: "
                    f"{[a.name for a in selected]}"
                )
                return selected
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"[ml] ML delegation failed with data error, falling back: {e}")
            except Exception as e:
                logger.exception(f"[ml] Unexpected ML delegation error, falling back: {e}")

        # Fall back to performance-based selection
        if self._use_performance_selection and self._agent_pool:
            return self._agent_pool.select_team(
                domain=domain,
                team_size=len(requested_agents),
            )

        return requested_agents


class VoteGroupHelper:
    """Helper for vote grouping and analysis."""

    @staticmethod
    def group_similar_votes(votes: list[Vote]) -> dict[str, list[str]]:
        """Group votes by which agent they voted for.

        Returns:
            Dict mapping chosen agent names to list of voter names
        """
        groups: dict[str, list[str]] = {}
        for vote in votes:
            # Vote uses 'choice' for who was voted for and 'agent' for voter
            voted_for = vote.choice
            voter = vote.agent
            if voted_for not in groups:
                groups[voted_for] = []
            groups[voted_for].append(voter)
        return groups


def format_conclusion(final_answer: str) -> str:
    """Format the debate conclusion for display.

    Args:
        final_answer: The final answer from the debate

    Returns:
        Formatted conclusion string
    """
    if not final_answer:
        return "No conclusion reached."
    return final_answer.strip()


def format_patterns_for_prompt(patterns: list[dict]) -> str:
    """Format successful patterns for inclusion in prompts.

    Args:
        patterns: List of pattern dictionaries with 'pattern' and 'context' keys

    Returns:
        Formatted string of patterns
    """
    if not patterns:
        return ""

    lines = ["Consider these successful patterns from similar debates:"]
    for i, p in enumerate(patterns[:5], 1):
        pattern = p.get("pattern", "")
        context = p.get("context", "")
        if pattern:
            lines.append(f"{i}. {pattern}")
            if context:
                lines.append(f"   Context: {context}")

    return "\n".join(lines)


__all__ = [
    "compute_domain_from_task",
    "CitationHelper",
    "QualityFilterHelper",
    "ConsensusEstimationHelper",
    "TeamSelectionHelper",
    "VoteGroupHelper",
    "format_conclusion",
    "format_patterns_for_prompt",
]

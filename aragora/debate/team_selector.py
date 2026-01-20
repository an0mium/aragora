"""Team selection for debate participation.

Extracted from orchestrator.py to reduce complexity and improve testability.
Handles agent scoring based on ELO, calibration, and circuit breaker filtering.

Enhanced with DelegationStrategy integration for intelligent task routing
and domain/capability-based agent filtering for optimal team composition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.debate.delegation import DelegationStrategy
    from aragora.debate.protocol import CircuitBreaker

logger = logging.getLogger(__name__)

# Domain-to-capability mapping for intelligent agent routing
# Maps task domains to agent name patterns that excel in those areas
DOMAIN_CAPABILITY_MAP: dict[str, list[str]] = {
    # Code-related tasks - prefer coding specialists
    "code": ["claude", "codex", "codestral", "deepseek", "gpt"],
    "programming": ["claude", "codex", "codestral", "deepseek", "gpt"],
    "technical": ["claude", "codex", "codestral", "deepseek", "gpt", "gemini"],
    # Research and analysis tasks
    "research": ["claude", "gemini", "gpt", "deepseek-r1"],
    "analysis": ["claude", "gemini", "gpt", "deepseek-r1"],
    "science": ["claude", "gemini", "gpt", "deepseek-r1"],
    # Creative tasks
    "creative": ["claude", "gpt", "gemini", "llama"],
    "writing": ["claude", "gpt", "gemini"],
    "storytelling": ["claude", "gpt", "gemini", "llama"],
    # Reasoning-heavy tasks
    "reasoning": ["claude", "deepseek-r1", "gpt", "gemini"],
    "logic": ["claude", "deepseek-r1", "gpt"],
    "math": ["claude", "deepseek-r1", "gpt", "gemini"],
    # General/default - no filtering
    "general": [],
}


class AgentScorer(Protocol):
    """Protocol for agent scoring systems."""

    def get_rating(self, agent_name: str) -> float:
        """Get agent's rating score."""
        ...


class CalibrationScorer(Protocol):
    """Protocol for calibration scoring systems."""

    def get_brier_score(self, agent_name: str, domain: Optional[str] = None) -> float:
        """Get agent's Brier score (lower is better).

        Args:
            agent_name: Name of the agent
            domain: Optional domain for domain-specific calibration
        """
        ...


@dataclass
class TeamSelectionConfig:
    """Configuration for team selection behavior."""

    elo_weight: float = 0.3
    calibration_weight: float = 0.2
    delegation_weight: float = 0.2  # Weight for delegation strategy scoring
    domain_capability_weight: float = 0.25  # Weight for domain expertise matching
    culture_weight: float = 0.15  # Weight for culture-based agent recommendations
    base_score: float = 1.0
    elo_baseline: int = 1000
    enable_domain_filtering: bool = True  # Enable domain-based agent filtering
    domain_filter_fallback: bool = True  # Fall back to all agents if no match
    enable_culture_selection: bool = False  # Enable culture-based agent scoring
    custom_domain_map: dict[str, list[str]] = field(default_factory=dict)


class TeamSelector:
    """Selects and scores agents for debate participation.

    Uses ELO ratings, calibration scores, delegation strategies, and
    circuit breaker status to prioritize high-performing, reliable agents.

    Example:
        selector = TeamSelector(
            elo_system=elo,
            calibration_tracker=tracker,
            circuit_breaker=breaker,
            delegation_strategy=ContentBasedDelegation(),
        )
        team = selector.select(agents, domain="technical", task="Review security")
    """

    def __init__(
        self,
        elo_system: Optional[AgentScorer] = None,
        calibration_tracker: Optional[CalibrationScorer] = None,
        circuit_breaker: Optional["CircuitBreaker"] = None,
        delegation_strategy: Optional["DelegationStrategy"] = None,
        knowledge_mound: Optional[Any] = None,
        config: Optional[TeamSelectionConfig] = None,
    ):
        self.elo_system = elo_system
        self.calibration_tracker = calibration_tracker
        self.circuit_breaker = circuit_breaker
        self.delegation_strategy = delegation_strategy
        self.knowledge_mound = knowledge_mound
        self.config = config or TeamSelectionConfig()
        self._culture_recommendations_cache: dict[str, list[str]] = {}

    def select(
        self,
        agents: list["Agent"],
        domain: str = "general",
        task: str = "",
        context: Optional["DebateContext"] = None,
    ) -> list["Agent"]:
        """Select and rank agents for debate participation.

        Args:
            agents: List of candidate agents
            domain: Task domain for context-aware selection
            task: Task description for delegation-based routing
            context: Optional debate context for state-aware selection

        Returns:
            Agents sorted by performance score (highest first)
        """
        # 1. Filter by domain capability first (before circuit breaker)
        domain_filtered = self._filter_by_domain_capability(agents, domain)

        # 2. Filter unavailable agents via circuit breaker
        available_names = self._filter_available(domain_filtered)

        # 3. Score remaining agents (using ELO, calibration, delegation, and domain)
        scored: list[tuple["Agent", float]] = []
        for agent in domain_filtered:
            if agent.name not in available_names:
                logger.info(f"agent_filtered_by_circuit_breaker agent={agent.name}")
                continue

            score = self._compute_score(agent, domain=domain, task=task, context=context)
            scored.append((agent, score))

        if not scored:
            logger.warning("No agents available after performance filtering")
            return agents  # Fall back to original list

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [agent for agent, _ in scored]
        logger.info(
            f"performance_selection domain={domain} "
            f"selected={[a.name for a in selected]} "
            f"scores={[f'{s:.2f}' for _, s in scored]}"
        )

        return selected

    def _filter_available(self, agents: list["Agent"]) -> set[str]:
        """Filter agents through circuit breaker."""
        available_names = {a.name for a in agents}

        if self.circuit_breaker:
            try:
                available_names = set(
                    self.circuit_breaker.filter_available_agents([a.name for a in agents])
                )
            except (AttributeError, TypeError) as e:
                logger.debug(f"circuit_breaker filter error: {e}")

        return available_names

    def _filter_by_domain_capability(
        self,
        agents: list["Agent"],
        domain: str,
    ) -> list["Agent"]:
        """Filter agents by domain expertise/capability.

        Uses DOMAIN_CAPABILITY_MAP to identify agents that excel in specific domains.
        Falls back to all agents if no matches found (configurable).

        Args:
            agents: List of candidate agents
            domain: Task domain (e.g., "code", "research", "creative")

        Returns:
            Filtered list of agents suited for the domain
        """
        if not self.config.enable_domain_filtering:
            return agents

        # Check custom domain map first, then default
        domain_lower = domain.lower()
        preferred_patterns = self.config.custom_domain_map.get(
            domain_lower,
            DOMAIN_CAPABILITY_MAP.get(domain_lower, []),
        )

        if not preferred_patterns:
            logger.debug(f"No domain mapping for '{domain}', using all agents")
            return agents

        # Filter agents whose name or agent_type matches preferred patterns
        matching_agents: list["Agent"] = []
        for agent in agents:
            if self._agent_matches_capability(agent, preferred_patterns):
                matching_agents.append(agent)

        if not matching_agents:
            if self.config.domain_filter_fallback:
                logger.info(
                    f"No agents match domain '{domain}' patterns {preferred_patterns}, "
                    f"falling back to all {len(agents)} agents"
                )
                return agents
            else:
                logger.warning(f"No agents match domain '{domain}', returning empty list")
                return []

        logger.info(
            f"domain_capability_filter domain={domain} "
            f"matched={[a.name for a in matching_agents]} "
            f"from={[a.name for a in agents]}"
        )
        return matching_agents

    def _agent_matches_capability(
        self,
        agent: "Agent",
        patterns: list[str],
    ) -> bool:
        """Check if an agent matches any of the capability patterns.

        Args:
            agent: Agent to check
            patterns: List of name/type patterns to match against

        Returns:
            True if agent matches any pattern
        """
        agent_identifiers = [
            agent.name.lower(),
            getattr(agent, "agent_type", "").lower(),
            getattr(agent, "model", "").lower(),
        ]

        for pattern in patterns:
            pattern_lower = pattern.lower()
            for identifier in agent_identifiers:
                if pattern_lower in identifier:
                    return True
        return False

    def _compute_domain_score(
        self,
        agent: "Agent",
        domain: str,
    ) -> float:
        """Compute a bonus score for domain expertise.

        Args:
            agent: Agent to score
            domain: Task domain

        Returns:
            Score bonus (0.0 to 1.0) based on domain match quality
        """
        domain_lower = domain.lower()
        preferred_patterns = self.config.custom_domain_map.get(
            domain_lower,
            DOMAIN_CAPABILITY_MAP.get(domain_lower, []),
        )

        if not preferred_patterns:
            return 0.0

        # Score based on position in preference list (earlier = better)
        for idx, pattern in enumerate(preferred_patterns):
            if self._agent_matches_capability(agent, [pattern]):
                # First in list gets 1.0, decreasing for later positions
                position_score = 1.0 - (idx * 0.15)
                return max(0.0, position_score)

        return 0.0

    def _compute_culture_score(
        self,
        agent: "Agent",
        task_type: str,
    ) -> float:
        """Compute a bonus score based on organizational culture patterns.

        Uses the Knowledge Mound's culture accumulator to get agent recommendations
        based on historical success patterns for the given task type.

        Args:
            agent: Agent to score
            task_type: Type of task (e.g., "code_review", "analysis", "creative")

        Returns:
            Score bonus (0.0 to 1.0) based on culture-based ranking
        """
        if not self.knowledge_mound or not self.config.enable_culture_selection:
            return 0.0

        # Check cache first
        cache_key = task_type.lower()
        if cache_key not in self._culture_recommendations_cache:
            try:
                # Get culture-based recommendations (sync wrapper for async call)
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context, can't block
                    # Use cached recommendations or skip
                    return 0.0
                else:
                    recommendations = loop.run_until_complete(
                        self.knowledge_mound.recommend_agents(task_type)
                    )
                    self._culture_recommendations_cache[cache_key] = recommendations or []
            except Exception as e:
                logger.debug(f"Culture recommendation failed for {task_type}: {e}")
                self._culture_recommendations_cache[cache_key] = []

        recommendations = self._culture_recommendations_cache.get(cache_key, [])
        if not recommendations:
            return 0.0

        # Score based on position in recommendation list
        agent_name_lower = agent.name.lower()
        for idx, rec_name in enumerate(recommendations):
            if rec_name.lower() in agent_name_lower or agent_name_lower in rec_name.lower():
                # First recommended gets 1.0, decreasing for later positions
                position_score = 1.0 - (idx * 0.2)
                return max(0.0, position_score)

        return 0.0

    async def compute_culture_score_async(
        self,
        agent: "Agent",
        task_type: str,
    ) -> float:
        """Async version of culture score computation.

        Call this from async contexts to avoid event loop issues.
        """
        if not self.knowledge_mound or not self.config.enable_culture_selection:
            return 0.0

        cache_key = task_type.lower()
        if cache_key not in self._culture_recommendations_cache:
            try:
                recommendations = await self.knowledge_mound.recommend_agents(task_type)
                self._culture_recommendations_cache[cache_key] = recommendations or []
            except Exception as e:
                logger.debug(f"Culture recommendation failed for {task_type}: {e}")
                self._culture_recommendations_cache[cache_key] = []

        recommendations = self._culture_recommendations_cache.get(cache_key, [])
        if not recommendations:
            return 0.0

        agent_name_lower = agent.name.lower()
        for idx, rec_name in enumerate(recommendations):
            if rec_name.lower() in agent_name_lower or agent_name_lower in rec_name.lower():
                position_score = 1.0 - (idx * 0.2)
                return max(0.0, position_score)

        return 0.0

    def _compute_score(
        self,
        agent: "Agent",
        domain: Optional[str] = None,
        task: str = "",
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Compute composite score for an agent.

        Args:
            agent: Agent to score
            domain: Optional domain for domain-specific calibration lookup
            task: Task description for delegation-based scoring
            context: Optional debate context for state-aware scoring
        """
        score = self.config.base_score

        # ELO contribution
        if self.elo_system:
            try:
                elo = self.elo_system.get_rating(agent.name)
                # Normalize: baseline is average, each 100 points = weight bonus
                score += (elo - self.config.elo_baseline) / 1000 * self.config.elo_weight
            except (KeyError, AttributeError) as e:
                logger.debug(f"ELO rating not found for {agent.name}: {e}")

        # Calibration contribution (well-calibrated agents get a bonus)
        # Uses domain-specific calibration when available
        if self.calibration_tracker:
            try:
                brier = self.calibration_tracker.get_brier_score(agent.name, domain=domain)
                # Lower Brier = better calibration = higher score
                score += (1 - brier) * self.config.calibration_weight
            except (KeyError, AttributeError, TypeError) as e:
                logger.debug(f"Calibration score not found for {agent.name}: {e}")

        # Delegation strategy contribution
        if self.delegation_strategy and task:
            try:
                delegation_score = self.delegation_strategy.score_agent(agent, task, context)
                # Normalize delegation score (assuming 0-5 range typical)
                normalized = min(delegation_score / 5.0, 1.0)
                score += normalized * self.config.delegation_weight
            except (AttributeError, TypeError) as e:
                logger.debug(f"Delegation score failed for {agent.name}: {e}")

        # Domain capability contribution (agents matching domain get bonus)
        if domain and self.config.enable_domain_filtering:
            domain_score = self._compute_domain_score(agent, domain)
            score += domain_score * self.config.domain_capability_weight

        # Culture-based contribution (agents recommended by org culture patterns)
        if self.knowledge_mound and self.config.enable_culture_selection and domain:
            culture_score = self._compute_culture_score(agent, domain)
            score += culture_score * self.config.culture_weight

        return score

    def score_agent(
        self,
        agent: "Agent",
        domain: Optional[str] = None,
        task: str = "",
        context: Optional["DebateContext"] = None,
    ) -> float:
        """Get score for a single agent (for external use).

        Args:
            agent: Agent to score
            domain: Optional domain for domain-specific calibration
            task: Optional task for delegation-based scoring
            context: Optional debate context for state-aware scoring
        """
        return self._compute_score(agent, domain=domain, task=task, context=context)

    def set_delegation_strategy(self, strategy: "DelegationStrategy") -> None:
        """Set or update the delegation strategy.

        Args:
            strategy: New delegation strategy to use
        """
        self.delegation_strategy = strategy

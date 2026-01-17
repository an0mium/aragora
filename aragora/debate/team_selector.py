"""Team selection for debate participation.

Extracted from orchestrator.py to reduce complexity and improve testability.
Handles agent scoring based on ELO, calibration, and circuit breaker filtering.

Enhanced with DelegationStrategy integration for intelligent task routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.debate.delegation import DelegationStrategy
    from aragora.debate.protocol import CircuitBreaker

logger = logging.getLogger(__name__)


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
    base_score: float = 1.0
    elo_baseline: int = 1000


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
        config: Optional[TeamSelectionConfig] = None,
    ):
        self.elo_system = elo_system
        self.calibration_tracker = calibration_tracker
        self.circuit_breaker = circuit_breaker
        self.delegation_strategy = delegation_strategy
        self.config = config or TeamSelectionConfig()

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
        # Filter unavailable agents via circuit breaker
        available_names = self._filter_available(agents)

        # Score remaining agents (using domain-specific calibration and delegation)
        scored: list[tuple["Agent", float]] = []
        for agent in agents:
            if agent.name not in available_names:
                logger.info(f"agent_filtered_by_circuit_breaker agent={agent.name}")
                continue

            score = self._compute_score(agent, domain=domain, task=task, context=context)
            scored.append((agent, score))

        if not scored:
            logger.warning("No agents available after performance filtering")
            return agents

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [agent for agent, _ in scored]
        logger.info(
            f"performance_selection domain={domain} "
            f"selected={[a.name for a in selected]} "
            f"scores={[f'{s:.2f}' for _, s in scored]}"
        )

        return selected

    def _filter_available(self, agents: list[Agent]) -> set[str]:
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

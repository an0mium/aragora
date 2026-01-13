"""Team selection for debate participation.

Extracted from orchestrator.py to reduce complexity and improve testability.
Handles agent scoring based on ELO, calibration, and circuit breaker filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.protocol import CircuitBreaker

logger = logging.getLogger(__name__)


class AgentScorer(Protocol):
    """Protocol for agent scoring systems."""

    def get_rating(self, agent_name: str) -> float:
        """Get agent's rating score."""
        ...


class CalibrationScorer(Protocol):
    """Protocol for calibration scoring systems."""

    def get_brier_score(self, agent_name: str) -> float:
        """Get agent's Brier score (lower is better)."""
        ...


@dataclass
class TeamSelectionConfig:
    """Configuration for team selection behavior."""

    elo_weight: float = 0.3
    calibration_weight: float = 0.2
    base_score: float = 1.0
    elo_baseline: int = 1000


class TeamSelector:
    """Selects and scores agents for debate participation.

    Uses ELO ratings, calibration scores, and circuit breaker status
    to prioritize high-performing, reliable agents.

    Example:
        selector = TeamSelector(
            elo_system=elo,
            calibration_tracker=tracker,
            circuit_breaker=breaker,
        )
        team = selector.select(agents, domain="technical")
    """

    def __init__(
        self,
        elo_system: Optional[AgentScorer] = None,
        calibration_tracker: Optional[CalibrationScorer] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        config: Optional[TeamSelectionConfig] = None,
    ):
        self.elo_system = elo_system
        self.calibration_tracker = calibration_tracker
        self.circuit_breaker = circuit_breaker
        self.config = config or TeamSelectionConfig()

    def select(
        self,
        agents: list[Agent],
        domain: str = "general",
    ) -> list[Agent]:
        """Select and rank agents for debate participation.

        Args:
            agents: List of candidate agents
            domain: Task domain for context-aware selection

        Returns:
            Agents sorted by performance score (highest first)
        """
        # Filter unavailable agents via circuit breaker
        available_names = self._filter_available(agents)

        # Score remaining agents
        scored: list[tuple[Agent, float]] = []
        for agent in agents:
            if agent.name not in available_names:
                logger.info(f"agent_filtered_by_circuit_breaker agent={agent.name}")
                continue

            score = self._compute_score(agent)
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

    def _compute_score(self, agent: Agent) -> float:
        """Compute composite score for an agent."""
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
        if self.calibration_tracker:
            try:
                brier = self.calibration_tracker.get_brier_score(agent.name)
                # Lower Brier = better calibration = higher score
                score += (1 - brier) * self.config.calibration_weight
            except (KeyError, AttributeError) as e:
                logger.debug(f"Calibration score not found for {agent.name}: {e}")

        return score

    def score_agent(self, agent: Agent) -> float:
        """Get score for a single agent (for external use)."""
        return self._compute_score(agent)

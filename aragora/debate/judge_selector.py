"""
Judge Selection for Multi-Agent Debates.

Provides strategies for selecting judges to evaluate debate outcomes.
Extracted from Arena orchestrator for cleaner separation of concerns.

Strategies:
- last: Use synthesizer or last agent (legacy)
- random: Random selection
- voted: Agents vote for judge
- elo_ranked: Highest ELO agent judges
- calibrated: Best composite score (ELO + calibration)

Usage:
    selector = JudgeSelector(agents, elo_system, protocol)
    judge = await selector.select_judge(proposals, context)
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from aragora.core import Agent, Message
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


class JudgeProtocol(Protocol):
    """Protocol for judge selection configuration."""

    @property
    def judge_selection(self) -> str:
        """Judge selection strategy: last, random, voted, elo_ranked, calibrated."""
        ...

    @property
    def judge_termination(self) -> bool:
        """Whether to allow judge-based early termination."""
        ...

    @property
    def min_rounds_before_judge_check(self) -> int:
        """Minimum rounds before judge can terminate."""
        ...


@dataclass
class JudgeScore:
    """Composite score for judge selection."""

    agent_name: str
    elo_score: float
    calibration_score: float
    composite_score: float


class JudgeScoringMixin:
    """Mixin providing scoring utilities for judge selection."""

    def __init__(self, elo_system: Optional["EloSystem"] = None):
        self._elo_system = elo_system

    def get_calibration_weight(self, agent_name: str) -> float:
        """Get agent weight based on calibration score (0.5-1.5 range).

        Uses calibration_score from ELO system to weight agent contributions.
        Agents with better calibration have higher weight.

        Returns:
            Weight between 0.5 (uncalibrated/poor) and 1.5 (perfect calibration)
        """
        if not self._elo_system:
            return 1.0

        try:
            rating = self._elo_system.get_rating(agent_name)
            cal_score = rating.calibration_score
            return 0.5 + cal_score
        except Exception as e:
            logger.debug(f"Calibration weight lookup failed for {agent_name}: {e}")
            return 1.0

    def compute_composite_score(self, agent_name: str) -> float:
        """Compute composite score for judge selection (ELO + calibration).

        Combines ELO ranking with calibration score for nuanced judge selection.

        Returns:
            Composite score (higher is better)
        """
        if not self._elo_system:
            return 0.0

        try:
            rating = self._elo_system.get_rating(agent_name)
            # Normalize ELO: 1000 is baseline, 500 is typical deviation
            elo_normalized = (rating.elo - 1000) / 500
            elo_normalized = max(0, elo_normalized)

            cal_score = rating.calibration_score

            # Weighted combination: 70% ELO, 30% calibration
            return (elo_normalized * 0.7) + (cal_score * 0.3)
        except Exception as e:
            logger.debug(f"Composite score calculation failed for {agent_name}: {e}")
            return 0.0

    def get_all_scores(self, agents: Sequence["Agent"]) -> list[JudgeScore]:
        """Get scores for all agents.

        Args:
            agents: Agents to score

        Returns:
            List of JudgeScore objects sorted by composite score descending
        """
        # Batch fetch all ratings in single query
        ratings = {}
        if self._elo_system:
            agent_names = [a.name for a in agents]
            ratings = self._elo_system.get_ratings_batch(agent_names)

        scores = []
        for agent in agents:
            elo_score = 0.0
            cal_score = 0.0

            rating = ratings.get(agent.name)
            if rating:
                elo_score = (rating.elo - 1000) / 500
                elo_score = max(0, elo_score)
                cal_score = rating.calibration_score

            composite = (elo_score * 0.7) + (cal_score * 0.3)
            scores.append(JudgeScore(
                agent_name=agent.name,
                elo_score=elo_score,
                calibration_score=cal_score,
                composite_score=composite,
            ))

        scores.sort(key=lambda x: x.composite_score, reverse=True)
        return scores


class JudgeSelectionStrategy(ABC):
    """Base class for judge selection strategies."""

    @abstractmethod
    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Select a judge from available agents.

        Args:
            agents: Available agents to select from
            proposals: Current proposals by agent name
            context: Debate context messages

        Returns:
            Selected judge agent
        """
        ...


class LastAgentStrategy(JudgeSelectionStrategy):
    """Legacy strategy: use synthesizer or last agent."""

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Select synthesizer if available, else last agent."""
        synthesizers = [a for a in agents if getattr(a, "role", None) == "synthesizer"]
        if synthesizers:
            return synthesizers[0]
        return agents[-1] if agents else None


class RandomStrategy(JudgeSelectionStrategy):
    """Random selection from all agents."""

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Select a random agent as judge."""
        return random.choice(list(agents)) if agents else None


class EloRankedStrategy(JudgeSelectionStrategy, JudgeScoringMixin):
    """Select highest ELO-rated agent as judge."""

    def __init__(self, elo_system: Optional["EloSystem"] = None):
        JudgeScoringMixin.__init__(self, elo_system)

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Select agent with highest ELO rating."""
        if not self._elo_system or not agents:
            return random.choice(list(agents)) if agents else None

        agent_names = [a.name for a in agents]

        try:
            leaderboard = self._elo_system.get_leaderboard(limit=len(agent_names))
            for entry in leaderboard:
                agent_name = getattr(entry, 'agent', None) or entry.get("agent") if hasattr(entry, 'get') else None  # type: ignore[union-attr]
                if agent_name in agent_names:
                    top_name = agent_name
                    top_elo = getattr(entry, 'elo', None) or (entry.get("elo", 1500) if hasattr(entry, 'get') else 1500)  # type: ignore[union-attr]
                    judge = next((a for a in agents if a.name == top_name), None)
                    if judge:
                        logger.debug(f"Selected {top_name} (ELO: {top_elo}) as judge")
                        return judge
        except Exception as e:
            logger.warning(f"ELO query failed: {e}; falling back to random")

        return random.choice(list(agents)) if agents else None


class CalibratedStrategy(JudgeSelectionStrategy, JudgeScoringMixin):
    """Select based on composite score (ELO + calibration)."""

    def __init__(self, elo_system: Optional["EloSystem"] = None):
        JudgeScoringMixin.__init__(self, elo_system)

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Select agent with best composite score."""
        if not self._elo_system or not agents:
            return random.choice(list(agents)) if agents else None

        scores = self.get_all_scores(agents)
        if scores:
            best = scores[0]
            judge = next((a for a in agents if a.name == best.agent_name), None)
            if judge:
                logger.debug(f"Selected {best.agent_name} (composite: {best.composite_score:.3f}) as judge")
                return judge

        return random.choice(list(agents)) if agents else None


class VotedStrategy(JudgeSelectionStrategy):
    """Agents vote on who should judge."""

    def __init__(
        self,
        generate_fn: Callable[["Agent", str, list["Message"]], str],
        build_vote_prompt_fn: Callable[[list["Agent"], dict[str, str]], str],
        sanitize_fn: Optional[Callable[[str, str], str]] = None,
    ):
        """
        Initialize voted strategy.

        Args:
            generate_fn: Async function to generate agent response
            build_vote_prompt_fn: Function to build vote prompt
            sanitize_fn: Optional function to sanitize output
        """
        self._generate = generate_fn
        self._build_prompt = build_vote_prompt_fn
        self._sanitize = sanitize_fn or (lambda x, _: x)

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """Have agents vote on who should judge."""
        if not agents:
            return None

        vote_counts: dict[str, int] = {}

        for agent in agents:
            other_agents = [a for a in agents if a.name != agent.name]
            if not other_agents:
                continue

            prompt = self._build_prompt(other_agents, proposals)

            try:
                raw_response = await self._generate(agent, prompt, context)
                response = self._sanitize(raw_response, agent.name)

                # Parse vote - look for agent names in response
                for other in other_agents:
                    if other.name.lower() in response.lower():
                        vote_counts[other.name] = vote_counts.get(other.name, 0) + 1
                        break
            except Exception as e:
                logger.warning(f"Judge vote error for {agent.name}: {e}")

        # Select agent with most votes, random tiebreaker
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [name for name, count in vote_counts.items() if count == max_votes]
            winner_name = random.choice(candidates)
            winner = next((a for a in agents if a.name == winner_name), None)
            if winner:
                return winner

        return random.choice(list(agents)) if agents else None


class JudgeSelector(JudgeScoringMixin):
    """
    Main judge selection coordinator.

    Provides unified interface for all judge selection strategies.

    Usage:
        selector = JudgeSelector(
            agents=agents,
            elo_system=elo_system,
            judge_selection="calibrated",
        )
        judge = await selector.select_judge(proposals, context)
    """

    def __init__(
        self,
        agents: Sequence["Agent"],
        elo_system: Optional["EloSystem"] = None,
        judge_selection: str = "random",
        generate_fn: Optional[Callable] = None,
        build_vote_prompt_fn: Optional[Callable] = None,
        sanitize_fn: Optional[Callable] = None,
    ):
        """
        Initialize the judge selector.

        Args:
            agents: Available agents
            elo_system: Optional ELO system for ranked selection
            judge_selection: Strategy name (last, random, voted, elo_ranked, calibrated)
            generate_fn: Agent generation function (required for voted strategy)
            build_vote_prompt_fn: Vote prompt builder (required for voted strategy)
            sanitize_fn: Output sanitizer function
        """
        JudgeScoringMixin.__init__(self, elo_system)
        self._agents = list(agents)
        self._judge_selection = judge_selection
        self._generate_fn = generate_fn
        self._build_vote_prompt_fn = build_vote_prompt_fn
        self._sanitize_fn = sanitize_fn

        # Initialize strategies
        self._strategies: dict[str, JudgeSelectionStrategy] = {
            "last": LastAgentStrategy(),
            "random": RandomStrategy(),
            "elo_ranked": EloRankedStrategy(elo_system),
            "calibrated": CalibratedStrategy(elo_system),
        }

        # Add voted strategy if dependencies provided
        if generate_fn and build_vote_prompt_fn:
            self._strategies["voted"] = VotedStrategy(
                generate_fn=generate_fn,
                build_vote_prompt_fn=build_vote_prompt_fn,
                sanitize_fn=sanitize_fn,
            )

    async def select_judge(
        self,
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """
        Select a judge using the configured strategy.

        Args:
            proposals: Current proposals by agent name
            context: Debate context messages

        Returns:
            Selected judge agent
        """
        strategy = self._strategies.get(self._judge_selection)

        if not strategy:
            logger.warning(f"Unknown judge selection '{self._judge_selection}', using random")
            strategy = self._strategies["random"]

        judge = await strategy.select(self._agents, proposals, context)

        if judge is None and self._agents:
            logger.warning("Judge selection returned None, falling back to random")
            judge = random.choice(self._agents)

        return judge

    @classmethod
    def from_protocol(
        cls,
        protocol: JudgeProtocol,
        agents: Sequence["Agent"],
        elo_system: Optional["EloSystem"] = None,
        generate_fn: Optional[Callable] = None,
        build_vote_prompt_fn: Optional[Callable] = None,
        sanitize_fn: Optional[Callable] = None,
    ) -> "JudgeSelector":
        """
        Create JudgeSelector from a debate protocol.

        Args:
            protocol: Protocol with judge_selection setting
            agents: Available agents
            elo_system: Optional ELO system
            generate_fn: Agent generation function
            build_vote_prompt_fn: Vote prompt builder
            sanitize_fn: Output sanitizer

        Returns:
            Configured JudgeSelector
        """
        return cls(
            agents=agents,
            elo_system=elo_system,
            judge_selection=protocol.judge_selection,
            generate_fn=generate_fn,
            build_vote_prompt_fn=build_vote_prompt_fn,
            sanitize_fn=sanitize_fn,
        )

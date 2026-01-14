"""
Judge Selection for Multi-Agent Debates.

Provides strategies for selecting judges to evaluate debate outcomes.
Extracted from Arena orchestrator for cleaner separation of concerns.

Single Judge Strategies:
- last: Use synthesizer or last agent (legacy)
- random: Random selection
- voted: Agents vote for judge
- elo_ranked: Highest ELO agent judges
- calibrated: Best composite score (ELO + calibration)
- crux_aware: Prefer historical dissenters on similar topics

Multi-Judge Panel:
- JudgePanel: Coordinates multiple judges with voting strategies
- JudgingStrategy: MAJORITY, SUPERMAJORITY, UNANIMOUS, WEIGHTED

Features:
- Circuit breaker awareness: filters unavailable agents before selection
- Fallback selection: provides ordered list of candidates for retry
- Panel voting: aggregate multiple judge opinions

Usage:
    # Single judge
    selector = JudgeSelector(agents, elo_system, protocol)
    judge = await selector.select_judge(proposals, context)

    # Multi-judge panel
    panel = JudgePanel(judges=judges, strategy=JudgingStrategy.MAJORITY)
    panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Well-reasoned")
    result = panel.get_result()
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from aragora.core import Agent, Message
    from aragora.ranking.elo import EloSystem
    from aragora.resilience import CircuitBreaker

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
            scores.append(
                JudgeScore(
                    agent_name=agent.name,
                    elo_score=elo_score,
                    calibration_score=cal_score,
                    composite_score=composite,
                )
            )

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
                agent_name = (
                    getattr(entry, "agent", None) or entry.get("agent")
                    if hasattr(entry, "get")
                    else None
                )  # type: ignore[union-attr]
                if agent_name in agent_names:
                    top_name = agent_name
                    top_elo = getattr(entry, "elo", None) or (
                        entry.get("elo", 1500) if hasattr(entry, "get") else 1500
                    )  # type: ignore[union-attr]
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
                logger.debug(
                    f"Selected {best.agent_name} (composite: {best.composite_score:.3f}) as judge"
                )
                return judge

        return random.choice(list(agents)) if agents else None


class CruxAwareStrategy(JudgeSelectionStrategy, JudgeScoringMixin):
    """Select judges who historically dissented on debate cruxes.

    This strategy aims to improve debate quality by selecting judges
    who have previously shown contrarian perspectives on similar topics.
    Dissenters often catch blind spots the majority misses.

    Falls back to CalibratedStrategy if no relevant historical data.
    """

    def __init__(
        self,
        elo_system: Optional["EloSystem"] = None,
        consensus_memory: Optional[object] = None,  # ConsensusMemory type
    ):
        JudgeScoringMixin.__init__(self, elo_system)
        self._consensus_memory = consensus_memory

    async def select(
        self,
        agents: Sequence["Agent"],
        proposals: dict[str, str],
        context: list["Message"],
        cruxes: Optional[list[dict]] = None,
    ) -> "Agent":
        """Select agent who historically dissented on similar cruxes.

        Args:
            agents: Available agents
            proposals: Current proposals
            context: Debate messages
            cruxes: Optional list of identified cruxes for this debate

        Returns:
            Selected judge, preferring historical dissenters
        """
        if not agents:
            return None

        # Try to find historical dissenters if we have consensus memory and cruxes
        if self._consensus_memory and cruxes:
            dissenters = self._find_historical_dissenters(cruxes, agents)
            if dissenters:
                # Rank dissenters by ELO and pick best one
                ranked = self._rank_by_elo(dissenters)
                if ranked:
                    logger.info(
                        f"crux_aware_judge selected={ranked[0].name} reason=historical_dissenter"
                    )
                    return ranked[0]

        # Fall back to calibrated strategy
        if self._elo_system:
            scores = self.get_all_scores(agents)
            if scores:
                best = scores[0]
                judge = next((a for a in agents if a.name == best.agent_name), None)
                if judge:
                    logger.debug(
                        f"crux_aware_judge fallback={best.agent_name} "
                        f"composite={best.composite_score:.3f}"
                    )
                    return judge

        # Ultimate fallback: random
        import random

        return random.choice(list(agents))

    def _find_historical_dissenters(
        self,
        cruxes: list[dict],
        agents: Sequence["Agent"],
    ) -> list["Agent"]:
        """Find agents who historically dissented on similar topics.

        Args:
            cruxes: Cruxes identified in this debate
            agents: Available agents to filter

        Returns:
            Agents who have dissented on similar topics
        """
        if not self._consensus_memory:
            return []

        agent_names = {a.name for a in agents}
        dissenters = set()

        try:
            # Extract topic from first crux if available
            for crux in cruxes[:2]:  # Check first 2 cruxes
                claim = crux.get("claim", "") or str(crux)

                # Query for similar debates
                similar = self._consensus_memory.find_similar_debates(claim, limit=5)

                for debate in similar:
                    # Check if this agent dissented in similar debate
                    dissenting = getattr(debate, "dissenting_agents", [])
                    if hasattr(debate, "consensus") and hasattr(
                        debate.consensus, "dissenting_agents"
                    ):
                        dissenting = debate.consensus.dissenting_agents

                    for agent_name in dissenting:
                        if agent_name in agent_names:
                            dissenters.add(agent_name)

        except Exception as e:
            logger.debug(f"Historical dissent query failed: {e}")

        # Convert names back to agent objects
        return [a for a in agents if a.name in dissenters]

    def _rank_by_elo(self, agents: Sequence["Agent"]) -> list["Agent"]:
        """Rank agents by ELO score descending.

        Args:
            agents: Agents to rank

        Returns:
            Agents sorted by ELO (highest first)
        """
        if not self._elo_system or not agents:
            return list(agents)

        try:
            agent_names = [a.name for a in agents]
            ratings = self._elo_system.get_ratings_batch(agent_names)

            ranked = sorted(
                agents,
                key=lambda a: ratings.get(a.name, type("", (), {"elo": 1000})()).elo,
                reverse=True,
            )
            return ranked
        except Exception as e:
            logger.debug(f"ELO ranking failed: {e}")
            return list(agents)


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
                raw_response = await self._generate(agent, prompt, context)  # type: ignore[misc]
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
    Supports circuit breaker integration to filter unavailable agents
    and provides fallback candidate lists for retry scenarios.

    Usage:
        selector = JudgeSelector(
            agents=agents,
            elo_system=elo_system,
            judge_selection="calibrated",
        )
        judge = await selector.select_judge(proposals, context)

        # With circuit breaker (filters unavailable agents)
        selector = JudgeSelector(
            agents=agents,
            elo_system=elo_system,
            circuit_breaker=breaker,
        )

        # Get ordered fallback candidates
        candidates = await selector.get_judge_candidates(proposals, context)
    """

    def __init__(
        self,
        agents: Sequence["Agent"],
        elo_system: Optional["EloSystem"] = None,
        judge_selection: str = "random",
        generate_fn: Optional[Callable] = None,
        build_vote_prompt_fn: Optional[Callable] = None,
        sanitize_fn: Optional[Callable] = None,
        circuit_breaker: Optional["CircuitBreaker"] = None,
        consensus_memory: Optional[object] = None,
    ):
        """
        Initialize the judge selector.

        Args:
            agents: Available agents
            elo_system: Optional ELO system for ranked selection
            judge_selection: Strategy name (last, random, voted, elo_ranked, calibrated, crux_aware)
            generate_fn: Agent generation function (required for voted strategy)
            build_vote_prompt_fn: Vote prompt builder (required for voted strategy)
            sanitize_fn: Output sanitizer function
            circuit_breaker: Optional circuit breaker for filtering unavailable agents
            consensus_memory: Optional ConsensusMemory for crux-aware selection
        """
        JudgeScoringMixin.__init__(self, elo_system)
        self._agents = list(agents)
        self._judge_selection = judge_selection
        self._generate_fn = generate_fn
        self._build_vote_prompt_fn = build_vote_prompt_fn
        self._sanitize_fn = sanitize_fn
        self._circuit_breaker = circuit_breaker
        self._consensus_memory = consensus_memory

        # Initialize strategies
        self._strategies: dict[str, JudgeSelectionStrategy] = {
            "last": LastAgentStrategy(),
            "random": RandomStrategy(),
            "elo_ranked": EloRankedStrategy(elo_system),
            "calibrated": CalibratedStrategy(elo_system),
            "crux_aware": CruxAwareStrategy(elo_system, consensus_memory),
        }

        # Add voted strategy if dependencies provided
        if generate_fn and build_vote_prompt_fn:
            self._strategies["voted"] = VotedStrategy(
                generate_fn=generate_fn,
                build_vote_prompt_fn=build_vote_prompt_fn,
                sanitize_fn=sanitize_fn,
            )

    def _filter_available_agents(self, agents: Sequence["Agent"]) -> list["Agent"]:
        """
        Filter agents by circuit breaker availability.

        Args:
            agents: Agents to filter

        Returns:
            List of agents with circuit breaker in closed/half-open state.
            Returns all agents if no circuit breaker configured.
        """
        if not self._circuit_breaker:
            return list(agents)

        available = []
        for agent in agents:
            if self._circuit_breaker.is_available(agent.name):
                available.append(agent)
            else:
                logger.debug(f"judge_filter_unavailable agent={agent.name}")

        if not available:
            # All agents unavailable - fall back to all agents with warning
            logger.warning(f"judge_selection_all_unavailable count={len(agents)}, using all")
            return list(agents)

        logger.debug(f"judge_filter_result available={len(available)}/{len(agents)}")
        return available

    async def select_judge(
        self,
        proposals: dict[str, str],
        context: list["Message"],
    ) -> "Agent":
        """
        Select a judge using the configured strategy.

        Filters unavailable agents (via circuit breaker) before selection.

        Args:
            proposals: Current proposals by agent name
            context: Debate context messages

        Returns:
            Selected judge agent
        """
        # Filter available agents first
        available_agents = self._filter_available_agents(self._agents)

        strategy = self._strategies.get(self._judge_selection)

        if not strategy:
            logger.warning(f"Unknown judge selection '{self._judge_selection}', using random")
            strategy = self._strategies["random"]

        judge = await strategy.select(available_agents, proposals, context)

        if judge is None and available_agents:
            logger.warning("Judge selection returned None, falling back to random")
            judge = random.choice(available_agents)

        return judge

    async def get_judge_candidates(
        self,
        proposals: dict[str, str],
        context: list["Message"],
        max_candidates: int = 3,
    ) -> list["Agent"]:
        """
        Get an ordered list of judge candidates for fallback selection.

        Returns candidates sorted by composite score (ELO + calibration),
        filtered by circuit breaker availability.

        Args:
            proposals: Current proposals by agent name
            context: Debate context messages
            max_candidates: Maximum number of candidates to return

        Returns:
            Ordered list of agent candidates (best first)
        """
        available_agents = self._filter_available_agents(self._agents)

        if not available_agents:
            return []

        # Get scores for all available agents
        scores = self.get_all_scores(available_agents)

        # Return agents in score order
        candidates = []
        for score in scores[:max_candidates]:
            agent = next((a for a in available_agents if a.name == score.agent_name), None)
            if agent:
                candidates.append(agent)

        # If we got no scores (no ELO system), use random shuffling
        if not candidates:
            candidates = list(available_agents)[:max_candidates]
            random.shuffle(candidates)

        logger.debug(
            f"judge_candidates count={len(candidates)} agents={[a.name for a in candidates]}"
        )
        return candidates

    @classmethod
    def from_protocol(
        cls,
        protocol: JudgeProtocol,
        agents: Sequence["Agent"],
        elo_system: Optional["EloSystem"] = None,
        generate_fn: Optional[Callable] = None,
        build_vote_prompt_fn: Optional[Callable] = None,
        sanitize_fn: Optional[Callable] = None,
        circuit_breaker: Optional["CircuitBreaker"] = None,
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
            circuit_breaker: Optional circuit breaker for filtering

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
            circuit_breaker=circuit_breaker,
        )


# =============================================================================
# Multi-Judge Panel System
# =============================================================================


class JudgingStrategy(Enum):
    """Strategy for aggregating multiple judge votes."""

    MAJORITY = "majority"  # Simple majority (>50%)
    SUPERMAJORITY = "supermajority"  # 2/3+ agreement
    UNANIMOUS = "unanimous"  # All judges must agree
    WEIGHTED = "weighted"  # Votes weighted by expertise/calibration


class JudgeVote(Enum):
    """A judge's vote on consensus validity."""

    APPROVE = "approve"  # Consensus is valid
    REJECT = "reject"  # Consensus should be rejected
    ABSTAIN = "abstain"  # Cannot evaluate


@dataclass
class JudgeVoteRecord:
    """Record of a judge's vote on consensus."""

    judge_name: str
    vote: JudgeVote
    confidence: float  # 0-1
    reasoning: str
    weight: float = 1.0  # For weighted voting
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class JudgingResult:
    """Result of judge panel evaluation."""

    approved: bool
    strategy: JudgingStrategy
    votes: list[JudgeVoteRecord]
    approval_ratio: float
    weighted_approval: float
    confidence: float
    reasoning: str
    dissenting_judges: list[str] = field(default_factory=list)
    abstaining_judges: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "strategy": self.strategy.value,
            "votes": [
                {
                    "judge": v.judge_name,
                    "vote": v.vote.value,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "weight": v.weight,
                }
                for v in self.votes
            ],
            "approval_ratio": self.approval_ratio,
            "weighted_approval": self.weighted_approval,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "dissenting_judges": self.dissenting_judges,
            "abstaining_judges": self.abstaining_judges,
        }


class JudgePanel:
    """
    Panel of judges that evaluates debate consensus.

    Coordinates judge voting and applies the selected judging strategy.
    Use this when you want multiple independent judges to evaluate
    debate outcomes for higher confidence in consensus validity.

    Example:
        # Create panel with 3 judges using weighted voting
        panel = JudgePanel(
            judges=selected_judges,
            strategy=JudgingStrategy.WEIGHTED,
            judge_weights={"claude": 1.2, "gpt4": 1.0, "gemini": 0.9}
        )

        # Record votes (typically from async judge evaluation)
        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Well-reasoned argument")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Sound logic")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Missing edge case")

        # Get aggregated result
        result = panel.get_result()
        if result.approved:
            print(f"Consensus approved with {result.confidence:.0%} confidence")
    """

    def __init__(
        self,
        judges: list["Agent"],
        strategy: JudgingStrategy = JudgingStrategy.MAJORITY,
        judge_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize judge panel.

        Args:
            judges: List of judge agents
            strategy: Voting strategy for consensus
            judge_weights: Optional per-judge weights (for weighted voting)
        """
        self.judges = judges
        self.strategy = strategy
        self.judge_weights = judge_weights or {}
        self.votes: list[JudgeVoteRecord] = []

    def record_vote(
        self,
        judge_name: str,
        vote: JudgeVote,
        confidence: float,
        reasoning: str,
        metadata: Optional[dict] = None,
    ) -> JudgeVoteRecord:
        """
        Record a judge's vote.

        Args:
            judge_name: Name of the voting judge
            vote: The vote (approve/reject/abstain)
            confidence: Confidence level (0-1)
            reasoning: Explanation for the vote
            metadata: Optional additional data

        Returns:
            The recorded vote
        """
        weight = self.judge_weights.get(judge_name, 1.0)
        record = JudgeVoteRecord(
            judge_name=judge_name,
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
            weight=weight,
            metadata=metadata or {},
        )
        self.votes.append(record)
        logger.debug(
            f"judge_vote recorded judge={judge_name} vote={vote.value} "
            f"confidence={confidence:.2f} weight={weight:.2f}"
        )
        return record

    def get_result(self) -> JudgingResult:
        """
        Compute judging result based on recorded votes.

        Returns:
            JudgingResult with approval status and reasoning
        """
        if not self.votes:
            return JudgingResult(
                approved=False,
                strategy=self.strategy,
                votes=[],
                approval_ratio=0.0,
                weighted_approval=0.0,
                confidence=0.0,
                reasoning="No votes recorded",
            )

        # Categorize votes
        approvals = [v for v in self.votes if v.vote == JudgeVote.APPROVE]
        rejections = [v for v in self.votes if v.vote == JudgeVote.REJECT]
        abstentions = [v for v in self.votes if v.vote == JudgeVote.ABSTAIN]

        # Compute ratios (excluding abstentions)
        non_abstain = len(approvals) + len(rejections)
        approval_ratio = len(approvals) / non_abstain if non_abstain > 0 else 0.0

        # Compute weighted approval
        total_weight = sum(v.weight for v in self.votes if v.vote != JudgeVote.ABSTAIN)
        approve_weight = sum(v.weight for v in approvals)
        weighted_approval = approve_weight / total_weight if total_weight > 0 else 0.0

        # Apply strategy
        approved = self._apply_strategy(approval_ratio, weighted_approval, approvals, rejections)

        # Compute aggregate confidence
        if approvals:
            confidence = sum(v.confidence for v in approvals) / len(approvals)
        elif rejections:
            confidence = sum(v.confidence for v in rejections) / len(rejections)
        else:
            confidence = 0.0

        # Generate reasoning
        reasoning = self._generate_reasoning(approved, approvals, rejections, abstentions)

        result = JudgingResult(
            approved=approved,
            strategy=self.strategy,
            votes=self.votes.copy(),
            approval_ratio=approval_ratio,
            weighted_approval=weighted_approval,
            confidence=confidence,
            reasoning=reasoning,
            dissenting_judges=[v.judge_name for v in rejections],
            abstaining_judges=[v.judge_name for v in abstentions],
        )

        logger.info(
            f"judge_panel_result approved={approved} strategy={self.strategy.value} "
            f"ratio={approval_ratio:.2f} weighted={weighted_approval:.2f} "
            f"confidence={confidence:.2f}"
        )

        return result

    def _apply_strategy(
        self,
        approval_ratio: float,
        weighted_approval: float,
        approvals: list[JudgeVoteRecord],
        rejections: list[JudgeVoteRecord],
    ) -> bool:
        """Apply judging strategy to determine approval."""
        if self.strategy == JudgingStrategy.MAJORITY:
            return approval_ratio > 0.5

        elif self.strategy == JudgingStrategy.SUPERMAJORITY:
            return approval_ratio >= 2 / 3

        elif self.strategy == JudgingStrategy.UNANIMOUS:
            return len(rejections) == 0 and len(approvals) > 0

        elif self.strategy == JudgingStrategy.WEIGHTED:
            return weighted_approval > 0.5

        else:
            # Default to majority
            return approval_ratio > 0.5

    def _generate_reasoning(
        self,
        approved: bool,
        approvals: list[JudgeVoteRecord],
        rejections: list[JudgeVoteRecord],
        abstentions: list[JudgeVoteRecord],
    ) -> str:
        """Generate human-readable reasoning for the result."""
        parts = []

        if approved:
            parts.append(f"Consensus APPROVED by judge panel ({self.strategy.value} strategy).")
        else:
            parts.append(f"Consensus REJECTED by judge panel ({self.strategy.value} strategy).")

        parts.append(
            f"Votes: {len(approvals)} approve, {len(rejections)} reject, "
            f"{len(abstentions)} abstain."
        )

        if approvals:
            parts.append(f"Key approval reason: {approvals[0].reasoning[:100]}")

        if rejections:
            parts.append(f"Key rejection reason: {rejections[0].reasoning[:100]}")

        return " ".join(parts)

    def reset(self) -> None:
        """Clear recorded votes for reuse."""
        self.votes.clear()


def create_judge_panel(
    candidates: list["Agent"],
    participants: Optional[list["Agent"]] = None,
    domain: str = "general",
    strategy: JudgingStrategy = JudgingStrategy.MAJORITY,
    count: int = 3,
    elo_system: Optional["EloSystem"] = None,
    exclude_participants: bool = True,
) -> JudgePanel:
    """
    Convenience function to create a judge panel from candidate agents.

    Selects judges based on ELO and calibration scores, excluding
    debate participants if specified.

    Args:
        candidates: Pool of potential judges
        participants: Debate participants to exclude
        domain: Debate domain (for logging)
        strategy: Judging strategy
        count: Number of judges to select
        elo_system: Optional ELO system for scoring
        exclude_participants: Whether to exclude participants from judging

    Returns:
        Configured JudgePanel ready for evaluation
    """
    # Filter out participants
    participant_names = set()
    if participants and exclude_participants:
        participant_names = {p.name for p in participants}

    available = [a for a in candidates if a.name not in participant_names]

    if not available:
        logger.warning("No eligible judges after filtering participants")
        available = candidates  # Fall back to all candidates

    # Score and rank by composite score if ELO system available
    if elo_system:
        scorer = JudgeScoringMixin(elo_system)
        scores = scorer.get_all_scores(available)

        # Select top candidates
        judge_names = [s.agent_name for s in scores[:count]]
        judges = [a for a in available if a.name in judge_names]

        # Build weights from calibration scores
        weights = {s.agent_name: 0.5 + s.calibration_score for s in scores[:count]}
    else:
        # Random selection without scoring
        judges = available[:count] if len(available) >= count else available
        weights = {}

    logger.info(
        f"create_judge_panel domain={domain} count={len(judges)} "
        f"excluded={len(participant_names)} participants"
    )

    return JudgePanel(judges=judges, strategy=strategy, judge_weights=weights)

"""
Consensus phase for debate orchestration.

This module extracts the consensus/voting logic (Phase 3) from the
Arena._run_inner() method, handling:
- None mode: No consensus, combine all proposals
- Majority mode: Weighted voting with reputation/reliability/consistency/calibration
- Unanimous mode: All agents must agree
- Judge mode: Single judge synthesizes

"""

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Agent, Vote, DebateResult
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


@dataclass
class ConsensusDependencies:
    """Core dependencies for consensus phase execution.

    Groups the system components needed for consensus resolution,
    making dependency injection cleaner and more explicit.
    """

    protocol: Any = None
    elo_system: Any = None
    memory: Any = None
    agent_weights: dict[str, float] = field(default_factory=dict)
    flip_detector: Any = None
    position_tracker: Any = None
    calibration_tracker: Any = None
    recorder: Any = None
    hooks: dict = field(default_factory=dict)
    user_votes: list = field(default_factory=list)


@dataclass
class ConsensusCallbacks:
    """Callback functions for consensus phase operations.

    Separates the callback dependencies from core dependencies,
    making the interface cleaner and more testable.
    """

    vote_with_agent: Optional[Callable] = None
    with_timeout: Optional[Callable] = None
    select_judge: Optional[Callable] = None
    build_judge_prompt: Optional[Callable] = None
    generate_with_agent: Optional[Callable] = None
    group_similar_votes: Optional[Callable] = None
    get_calibration_weight: Optional[Callable] = None
    notify_spectator: Optional[Callable] = None
    drain_user_events: Optional[Callable] = None
    extract_debate_domain: Optional[Callable] = None
    get_belief_analyzer: Optional[Callable] = None
    user_vote_multiplier: Optional[Callable] = None


class ConsensusPhase:
    """
    Executes the consensus resolution phase.

    This class encapsulates the voting and consensus logic that was
    previously in Arena._run_inner() after the debate rounds.

    Usage (new style with dataclasses):
        deps = ConsensusDependencies(
            protocol=arena.protocol,
            elo_system=arena.elo_system,
            memory=arena.memory,
        )
        callbacks = ConsensusCallbacks(
            vote_with_agent=arena._vote_with_agent,
        )
        consensus_phase = ConsensusPhase(deps, callbacks)
        await consensus_phase.execute(ctx)

    Usage (legacy style - backward compatible):
        consensus_phase = ConsensusPhase(
            protocol=arena.protocol,
            elo_system=arena.elo_system,
        )
        await consensus_phase.execute(ctx)
    """

    def __init__(
        self,
        deps: ConsensusDependencies | Any = None,
        callbacks: ConsensusCallbacks | None = None,
        # Legacy parameters for backward compatibility
        protocol: Any = None,
        elo_system: Any = None,
        memory: Any = None,
        agent_weights: Optional[dict[str, float]] = None,
        flip_detector: Any = None,
        position_tracker: Any = None,
        calibration_tracker: Any = None,
        recorder: Any = None,
        hooks: Optional[dict] = None,
        user_votes: Optional[list] = None,
        # Legacy callbacks
        vote_with_agent: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        select_judge: Optional[Callable] = None,
        build_judge_prompt: Optional[Callable] = None,
        generate_with_agent: Optional[Callable] = None,
        group_similar_votes: Optional[Callable] = None,
        get_calibration_weight: Optional[Callable] = None,
        notify_spectator: Optional[Callable] = None,
        drain_user_events: Optional[Callable] = None,
        extract_debate_domain: Optional[Callable] = None,
        get_belief_analyzer: Optional[Callable] = None,
        user_vote_multiplier: Optional[Callable] = None,
    ):
        """
        Initialize the consensus phase.

        Args:
            deps: ConsensusDependencies dataclass (new style)
            callbacks: ConsensusCallbacks dataclass (new style)

            Legacy args (for backward compatibility):
            protocol, elo_system, memory, agent_weights, flip_detector,
            position_tracker, calibration_tracker, recorder, hooks, user_votes,
            and all callback parameters.
        """
        # Support both new dataclass style and legacy parameter style
        if isinstance(deps, ConsensusDependencies):
            # New style: use dataclasses
            self.protocol = deps.protocol
            self.elo_system = deps.elo_system
            self.memory = deps.memory
            self.agent_weights = deps.agent_weights
            self.flip_detector = deps.flip_detector
            self.position_tracker = deps.position_tracker
            self.calibration_tracker = deps.calibration_tracker
            self.recorder = deps.recorder
            self.hooks = deps.hooks
            self.user_votes = deps.user_votes
        else:
            # Legacy style: use individual parameters
            self.protocol = deps if deps is not None else protocol
            self.elo_system = elo_system
            self.memory = memory
            self.agent_weights = agent_weights or {}
            self.flip_detector = flip_detector
            self.position_tracker = position_tracker
            self.calibration_tracker = calibration_tracker
            self.recorder = recorder
            self.hooks = hooks or {}
            self.user_votes = user_votes or []

        # Callbacks: prefer dataclass, fall back to legacy parameters
        if callbacks is not None:
            self._vote_with_agent = callbacks.vote_with_agent
            self._with_timeout = callbacks.with_timeout
            self._select_judge = callbacks.select_judge
            self._build_judge_prompt = callbacks.build_judge_prompt
            self._generate_with_agent = callbacks.generate_with_agent
            self._group_similar_votes = callbacks.group_similar_votes
            self._get_calibration_weight = callbacks.get_calibration_weight
            self._notify_spectator = callbacks.notify_spectator
            self._drain_user_events = callbacks.drain_user_events
            self._extract_debate_domain = callbacks.extract_debate_domain
            self._get_belief_analyzer = callbacks.get_belief_analyzer
            self._user_vote_multiplier = callbacks.user_vote_multiplier
        else:
            self._vote_with_agent = vote_with_agent
            self._with_timeout = with_timeout
            self._select_judge = select_judge
            self._build_judge_prompt = build_judge_prompt
            self._generate_with_agent = generate_with_agent
            self._group_similar_votes = group_similar_votes
            self._get_calibration_weight = get_calibration_weight
            self._notify_spectator = notify_spectator
            self._drain_user_events = drain_user_events
            self._extract_debate_domain = extract_debate_domain
            self._get_belief_analyzer = get_belief_analyzer
            self._user_vote_multiplier = user_vote_multiplier

    # Default timeout for consensus phase (can be overridden via protocol)
    DEFAULT_CONSENSUS_TIMEOUT = 120  # 2 minutes

    async def execute(self, ctx: "DebateContext") -> None:
        """
        Execute the consensus phase with fallback mechanisms.

        This method wraps consensus execution with:
        - Timeout protection (default 120s)
        - Exception handling with fallback to 'none' mode
        - Graceful degradation when agents fail

        Args:
            ctx: The DebateContext with proposals and result
        """
        consensus_mode = self.protocol.consensus if self.protocol else "none"
        logger.info(f"consensus_phase_start mode={consensus_mode}")

        # Get timeout from protocol or use default
        timeout = getattr(self.protocol, 'consensus_timeout', self.DEFAULT_CONSENSUS_TIMEOUT)

        try:
            await asyncio.wait_for(
                self._execute_consensus(ctx, consensus_mode),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"consensus_timeout mode={consensus_mode} timeout={timeout}s, falling back to none"
            )
            await self._handle_fallback_consensus(ctx, reason="timeout")
        except Exception as e:
            logger.error(
                f"consensus_error mode={consensus_mode} error={type(e).__name__}: {e}",
                exc_info=True
            )
            await self._handle_fallback_consensus(ctx, reason=f"error: {type(e).__name__}")

    async def _execute_consensus(self, ctx: "DebateContext", consensus_mode: str) -> None:
        """Execute the consensus logic for the given mode."""
        if consensus_mode == "none":
            await self._handle_none_consensus(ctx)
        elif consensus_mode == "majority":
            await self._handle_majority_consensus(ctx)
        elif consensus_mode == "unanimous":
            await self._handle_unanimous_consensus(ctx)
        elif consensus_mode == "judge":
            await self._handle_judge_consensus(ctx)
        else:
            logger.warning(f"Unknown consensus mode: {consensus_mode}, using none")
            await self._handle_none_consensus(ctx)

    async def _handle_fallback_consensus(self, ctx: "DebateContext", reason: str) -> None:
        """
        Handle consensus fallback when the primary mechanism fails.

        This provides graceful degradation by combining all proposals
        and marking the result as a fallback consensus.

        Args:
            ctx: The DebateContext with proposals and result
            reason: Description of why fallback was triggered
        """
        result = ctx.result
        proposals = ctx.proposals

        logger.info(f"consensus_fallback reason={reason} proposals={len(proposals)}")

        if proposals:
            result.final_answer = (
                f"[Consensus fallback ({reason})]\n\n"
                + "\n\n---\n\n".join(
                    f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
                )
            )
        else:
            result.final_answer = f"[No proposals available - consensus fallback ({reason})]"

        result.consensus_reached = False
        result.confidence = 0.0
        result.consensus_strength = "fallback"

        # Log fallback reason (DebateResult doesn't have metadata field)
        logger.info(f"consensus_fallback reason={reason}")

    async def _handle_none_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'none' consensus mode - combine all proposals."""
        result = ctx.result
        proposals = ctx.proposals

        result.final_answer = "\n\n---\n\n".join(
            f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
        )
        result.consensus_reached = False
        result.confidence = 0.5

    async def _handle_majority_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'majority' consensus mode - weighted voting."""
        result = ctx.result
        proposals = ctx.proposals

        # Cast votes from all agents
        votes = await self._collect_votes(ctx)
        result.votes.extend(votes)

        # Group similar votes
        vote_groups, choice_mapping = self._compute_vote_groups(votes)

        # Pre-compute vote weights
        vote_weight_cache = self._compute_vote_weights(ctx)

        # Count weighted votes
        vote_counts, total_weighted = self._count_weighted_votes(
            votes, choice_mapping, vote_weight_cache
        )

        # Include user votes
        vote_counts, total_weighted = self._add_user_votes(
            vote_counts, total_weighted, choice_mapping
        )

        ctx.vote_tally = dict(vote_counts)

        # Determine winner
        self._determine_majority_winner(
            ctx, vote_counts, total_weighted, choice_mapping
        )

        # Analyze belief network for cruxes
        self._analyze_belief_network(ctx)

    async def _handle_unanimous_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'unanimous' consensus mode - all must agree."""
        result = ctx.result
        proposals = ctx.proposals

        # Cast votes
        votes, voting_errors = await self._collect_votes_with_errors(ctx)
        result.votes.extend(votes)

        # Group similar votes
        vote_groups, choice_mapping = self._compute_vote_groups(votes)

        # Count votes (no weighting for unanimous)
        vote_counts: Counter[str] = Counter()
        for v in votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                vote_counts[canonical] += 1

        # Drain user events and include user votes if configured
        if self._drain_user_events:
            self._drain_user_events()

        user_vote_weight = getattr(self.protocol, 'user_vote_weight', 0.0)
        user_vote_count = 0
        if user_vote_weight > 0:
            for user_vote in self.user_votes:
                choice = user_vote.get("choice", "")
                if choice:
                    canonical = choice_mapping.get(choice, choice)
                    vote_counts[canonical] += 1
                    user_vote_count += 1
                    logger.debug(
                        f"user_vote_unanimous user={user_vote.get('user_id', 'anonymous')} "
                        f"choice={choice}"
                    )

        ctx.vote_tally = dict(vote_counts)

        # Check for unanimity
        total_voters = len(votes) + voting_errors + user_vote_count

        most_common = vote_counts.most_common(1) if vote_counts else []
        if most_common and total_voters > 0:
            winner, count = most_common[0]
            unanimity_ratio = count / total_voters

            if unanimity_ratio >= 1.0:
                self._set_unanimous_winner(ctx, winner, unanimity_ratio, total_voters, count)
            else:
                self._set_no_unanimity(ctx, winner, unanimity_ratio, total_voters, count, choice_mapping)
        else:
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            result.confidence = 0.0

    async def _handle_judge_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'judge' consensus mode - single judge synthesis."""
        result = ctx.result
        proposals = ctx.proposals

        if not self._select_judge or not self._generate_with_agent:
            logger.error("Judge consensus requires select_judge and generate_with_agent")
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            return

        # Select judge
        judge = await self._select_judge(proposals, ctx.context_messages)
        judge_method = self.protocol.judge_selection if self.protocol else "random"
        logger.info(f"judge_selected judge={judge.name} method={judge_method}")

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "judge",
                agent=judge.name,
                details=f"Selected as judge via {judge_method}",
            )

        # Emit judge selection hook
        if "on_judge_selected" in self.hooks:
            self.hooks["on_judge_selected"](judge.name, judge_method)

        # Build judge prompt and generate synthesis
        task = ctx.env.task if ctx.env else ""
        judge_prompt = (
            self._build_judge_prompt(proposals, task, result.critiques)
            if self._build_judge_prompt
            else f"Synthesize these proposals: {proposals}"
        )

        try:
            synthesis = await self._generate_with_agent(
                judge, judge_prompt, ctx.context_messages
            )
            result.final_answer = synthesis
            result.consensus_reached = True
            result.confidence = 0.8
            # Set winner to judge for ELO tracking
            ctx.winner_agent = judge.name
            result.winner = judge.name
            logger.info(f"judge_synthesis judge={judge.name} length={len(synthesis)}")

            # Notify spectator
            if self._notify_spectator:
                self._notify_spectator(
                    "consensus",
                    agent=judge.name,
                    details=f"Judge synthesis ({len(synthesis)} chars)",
                    metric=0.8,
                )

            # Emit message for activity feed
            if "on_message" in self.hooks:
                rounds = self.protocol.rounds if self.protocol else 0
                self.hooks["on_message"](
                    agent=judge.name,
                    content=synthesis,
                    role="judge",
                    round_num=rounds + 1,
                )

        except Exception as e:
            logger.error(f"judge_error error={e}")
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False

    async def _collect_votes(self, ctx: "DebateContext") -> list["Vote"]:
        """Collect votes from all agents."""
        if not self._vote_with_agent:
            logger.warning("No vote_with_agent callback, skipping votes")
            return []

        votes = []
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent):
            logger.debug(f"agent_voting agent={agent.name}")
            try:
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=90.0,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                return (agent, e)

        vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in ctx.agents]

        for completed_task in asyncio.as_completed(vote_tasks):
            try:
                agent, vote_result = await completed_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"task_exception phase=vote error={e}")
                continue

            if vote_result is None or isinstance(vote_result, Exception):
                if isinstance(vote_result, Exception):
                    logger.error(f"vote_error agent={agent.name} error={vote_result}")
                else:
                    logger.error(f"vote_error agent={agent.name} error=vote returned None")
            else:
                votes.append(vote_result)
                self._handle_vote_success(ctx, agent, vote_result)

        return votes

    async def _collect_votes_with_errors(
        self, ctx: "DebateContext"
    ) -> tuple[list["Vote"], int]:
        """Collect votes, tracking error count for unanimity calculation."""
        if not self._vote_with_agent:
            return [], 0

        votes = []
        voting_errors = 0
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent):
            logger.debug(f"agent_voting_unanimous agent={agent.name}")
            try:
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=90.0,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                return (agent, e)

        vote_tasks = [asyncio.create_task(cast_vote(agent)) for agent in ctx.agents]

        for completed_task in asyncio.as_completed(vote_tasks):
            try:
                agent, vote_result = await completed_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"task_exception phase=unanimous_vote error={e}")
                voting_errors += 1
                continue

            if vote_result is None or isinstance(vote_result, Exception):
                if isinstance(vote_result, Exception):
                    logger.error(f"vote_error_unanimous agent={agent.name} error={vote_result}")
                else:
                    logger.error(f"vote_error_unanimous agent={agent.name} error=vote returned None")
                voting_errors += 1
            else:
                votes.append(vote_result)
                self._handle_vote_success(ctx, agent, vote_result, unanimous=True)

        return votes, voting_errors

    def _handle_vote_success(
        self,
        ctx: "DebateContext",
        agent: "Agent",
        vote: "Vote",
        unanimous: bool = False,
    ) -> None:
        """Handle successful vote: notifications, hooks, recording."""
        result = ctx.result

        logger.debug(
            f"vote_cast{'_unanimous' if unanimous else ''} agent={agent.name} "
            f"choice={vote.choice} confidence={vote.confidence:.0%}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "vote",
                agent=agent.name,
                details=f"Voted for {vote.choice}",
                metric=vote.confidence,
            )

        # Emit vote hook
        if "on_vote" in self.hooks:
            self.hooks["on_vote"](agent.name, vote.choice, vote.confidence)

        # Record vote
        if self.recorder:
            try:
                self.recorder.record_vote(agent.name, vote.choice, vote.reasoning)
            except Exception as e:
                logger.debug(f"Recorder error for vote: {e}")

        # Record position for truth-grounded personas
        if self.position_tracker:
            try:
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                self.position_tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent.name,
                    position_type="vote",
                    position_text=vote.choice,
                    round_num=result.rounds_used,
                    confidence=vote.confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracking error for vote: {e}")

    def _compute_vote_groups(
        self, votes: list["Vote"]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Group similar votes and create choice mapping."""
        if not self._group_similar_votes:
            # No grouping, identity mapping
            choices = set(v.choice for v in votes if not isinstance(v, Exception))
            return {c: [c] for c in choices}, {c: c for c in choices}

        vote_groups = self._group_similar_votes(votes)

        choice_mapping: dict[str, str] = {}
        for canonical, variants in vote_groups.items():
            for variant in variants:
                choice_mapping[variant] = canonical

        if vote_groups:
            logger.debug(f"vote_grouping_merged groups={vote_groups}")

        return vote_groups, choice_mapping

    def _compute_vote_weights(self, ctx: "DebateContext") -> dict[str, float]:
        """Pre-compute vote weights for all agents."""
        vote_weight_cache: dict[str, float] = {}

        # Batch fetch all agent ratings
        ratings_cache: dict[str, Any] = {}
        if self.elo_system:
            try:
                agent_names = [agent.name for agent in ctx.agents]
                ratings_cache = self.elo_system.get_ratings_batch(agent_names)
            except Exception as e:
                logger.debug(f"Batch ratings fetch failed: {e}")

        for agent in ctx.agents:
            agent_weight = 1.0

            # Reputation weight (0.5-1.5)
            if self.memory and hasattr(self.memory, 'get_vote_weight'):
                agent_weight = self.memory.get_vote_weight(agent.name)

            # Reliability weight from capability probing (0.0-1.0 multiplier)
            if self.agent_weights and agent.name in self.agent_weights:
                agent_weight *= self.agent_weights[agent.name]

            # Consistency weight from FlipDetector (0.5-1.0 multiplier)
            if self.flip_detector:
                try:
                    consistency = self.flip_detector.get_agent_consistency(agent.name)
                    consistency_weight = 0.5 + (consistency.consistency_score * 0.5)
                    agent_weight *= consistency_weight
                except Exception as e:
                    logger.debug(f"FlipDetector consistency error: {e}")

            # Calibration weight (0.5-1.5 multiplier)
            if agent.name in ratings_cache:
                cal_score = ratings_cache[agent.name].calibration_score
                calibration_weight = 0.5 + cal_score
            elif self._get_calibration_weight:
                calibration_weight = self._get_calibration_weight(agent.name)
            else:
                calibration_weight = 1.0
            agent_weight *= calibration_weight

            vote_weight_cache[agent.name] = agent_weight

        return vote_weight_cache

    def _count_weighted_votes(
        self,
        votes: list["Vote"],
        choice_mapping: dict[str, str],
        vote_weight_cache: dict[str, float],
    ) -> tuple[Counter[str], float]:
        """Count weighted votes."""
        vote_counts: Counter[str] = Counter()
        total_weighted = 0.0

        for v in votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                weight = vote_weight_cache.get(v.agent, 1.0)
                vote_counts[canonical] += weight  # type: ignore[assignment]
                total_weighted += weight

        return vote_counts, total_weighted

    def _add_user_votes(
        self,
        vote_counts: Counter,
        total_weighted: float,
        choice_mapping: dict[str, str],
    ) -> tuple[Counter, float]:
        """Add user votes to counts."""
        if self._drain_user_events:
            self._drain_user_events()

        base_user_weight = getattr(self.protocol, 'user_vote_weight', 0.5)

        for user_vote in self.user_votes:
            choice = user_vote.get("choice", "")
            if choice:
                canonical = choice_mapping.get(choice, choice)
                intensity = user_vote.get("intensity", 5)

                if self._user_vote_multiplier:
                    intensity_multiplier = self._user_vote_multiplier(intensity, self.protocol)
                else:
                    intensity_multiplier = 1.0

                final_weight = base_user_weight * intensity_multiplier
                vote_counts[canonical] += final_weight  # type: ignore[assignment]
                total_weighted += final_weight

                logger.debug(
                    f"user_vote user={user_vote.get('user_id', 'anonymous')} "
                    f"choice={choice} intensity={intensity} weight={final_weight:.2f}"
                )

        return vote_counts, total_weighted

    def _determine_majority_winner(
        self,
        ctx: "DebateContext",
        vote_counts: Counter,
        total_votes: float,
        choice_mapping: dict[str, str],
    ) -> None:
        """Determine winner for majority consensus."""
        result = ctx.result
        proposals = ctx.proposals

        most_common = vote_counts.most_common(1) if vote_counts else []
        if not most_common:
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            result.confidence = 0.0
            return

        winner, count = most_common[0]
        threshold = self.protocol.consensus_threshold if self.protocol else 0.5

        result.final_answer = proposals.get(
            winner, list(proposals.values())[0] if proposals else ""
        )
        result.consensus_reached = count / total_votes >= threshold
        result.confidence = count / total_votes
        ctx.winner_agent = winner
        result.winner = winner  # Set winner for ELO tracking

        # Calculate consensus variance and strength
        if len(vote_counts) > 1:
            counts = list(vote_counts.values())
            mean = sum(counts) / len(counts)
            variance = sum((c - mean) ** 2 for c in counts) / len(counts)
            result.consensus_variance = variance

            if variance < 1:
                result.consensus_strength = "strong"
            elif variance < 2:
                result.consensus_strength = "medium"
            else:
                result.consensus_strength = "weak"

            logger.info(f"consensus_strength strength={result.consensus_strength} variance={variance:.2f}")
        else:
            result.consensus_strength = "unanimous"
            result.consensus_variance = 0.0

        # Track dissenting views
        for agent, prop in proposals.items():
            if agent != winner:
                result.dissenting_views.append(f"[{agent}]: {prop}")

        logger.info(f"consensus_winner winner={winner} votes={count}/{len(ctx.agents)}")

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"Majority vote: {winner}",
                metric=result.confidence,
            )

        # Record consensus
        if self.recorder:
            try:
                self.recorder.record_phase_change(f"consensus_reached: {winner}")
            except Exception as e:
                logger.debug(f"Recorder error for consensus: {e}")

        # Finalize for truth-grounded personas
        if self.position_tracker:
            try:
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                self.position_tracker.finalize_debate(
                    debate_id=debate_id,
                    winning_agent=winner,
                    winning_position=result.final_answer[:1000],
                    consensus_confidence=result.confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracker finalize error: {e}")

        # Record calibration predictions
        if self.calibration_tracker:
            try:
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                domain = self._extract_debate_domain() if self._extract_debate_domain else "general"
                for v in result.votes:
                    if not isinstance(v, Exception):
                        canonical = choice_mapping.get(v.choice, v.choice)
                        correct = canonical == winner
                        self.calibration_tracker.record_prediction(
                            agent=v.agent,
                            confidence=v.confidence,
                            correct=correct,
                            domain=domain,
                            debate_id=debate_id,
                        )
                logger.debug(f"calibration_recorded predictions={len(result.votes)}")
            except Exception as e:
                logger.warning(f"calibration_error error={e}")

    def _set_unanimous_winner(
        self,
        ctx: "DebateContext",
        winner: str,
        unanimity_ratio: float,
        total_voters: int,
        count: int,
    ) -> None:
        """Set result for unanimous consensus."""
        result = ctx.result
        proposals = ctx.proposals

        result.final_answer = proposals.get(
            winner, list(proposals.values())[0] if proposals else ""
        )
        result.consensus_reached = True
        result.confidence = unanimity_ratio
        result.consensus_strength = "unanimous"
        result.consensus_variance = 0.0
        ctx.winner_agent = winner
        result.winner = winner  # Set winner for ELO tracking

        logger.info(
            f"consensus_unanimous winner={winner} votes={count}/{total_voters} "
            f"ratio={unanimity_ratio:.0%}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"Unanimous: {winner}",
                metric=result.confidence,
            )

        # Record consensus
        if self.recorder:
            try:
                self.recorder.record_phase_change(f"consensus_reached: {winner}")
            except Exception as e:
                logger.debug(f"Recorder error for unanimous consensus: {e}")

        # Record calibration predictions
        if self.calibration_tracker:
            try:
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                domain = self._extract_debate_domain() if self._extract_debate_domain else "general"
                for v in result.votes:
                    if not isinstance(v, Exception):
                        correct = v.choice == winner
                        self.calibration_tracker.record_prediction(
                            agent=v.agent,
                            confidence=v.confidence,
                            correct=correct,
                            domain=domain,
                            debate_id=debate_id,
                        )
                logger.debug(f"calibration_recorded_unanimous predictions={len(result.votes)}")
            except Exception as e:
                logger.warning(f"calibration_error_unanimous error={e}")

    def _set_no_unanimity(
        self,
        ctx: "DebateContext",
        winner: str,
        unanimity_ratio: float,
        total_voters: int,
        count: int,
        choice_mapping: dict[str, str],
    ) -> None:
        """Set result when unanimity not reached."""
        result = ctx.result
        proposals = ctx.proposals
        vote_counts: Counter[str] = Counter()
        for v in result.votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                vote_counts[canonical] += 1

        result.final_answer = (
            f"[No unanimous consensus reached]\n\nProposals:\n"
            + "\n\n---\n\n".join(
                f"[{agent}] ({vote_counts.get(choice_mapping.get(agent, agent), 0)} votes):\n{prop}"
                for agent, prop in proposals.items()
            )
        )
        result.consensus_reached = False
        result.confidence = unanimity_ratio
        result.consensus_strength = "none"

        # Track all views as dissenting
        for agent, prop in proposals.items():
            result.dissenting_views.append(f"[{agent}]: {prop}")

        logger.info(
            f"consensus_not_unanimous best={winner} ratio={unanimity_ratio:.0%} "
            f"votes={count}/{total_voters}"
        )

        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"No unanimity: {winner} got {unanimity_ratio:.0%}",
                metric=unanimity_ratio,
            )

    def _analyze_belief_network(self, ctx: "DebateContext") -> None:
        """Analyze belief network to identify debate cruxes."""
        if not self._get_belief_analyzer:
            return

        result = ctx.result
        if not result.messages:
            return

        BN, BPA = self._get_belief_analyzer()
        if not BN or not BPA:
            return

        try:
            network = BN()
            for msg in result.messages:
                if msg.role in ("proposer", "critic"):
                    network.add_claim(
                        claim_id=f"{msg.agent}_{hash(msg.content[:100])}",
                        statement=msg.content[:500],
                        author=msg.agent,
                        initial_confidence=0.5,
                    )

            if network.nodes:
                network.propagate(iterations=3)
                analyzer = BPA(network)
                result.debate_cruxes = analyzer.identify_debate_cruxes(top_k=3)
                result.evidence_suggestions = analyzer.suggest_evidence_targets()[:3]
                if result.debate_cruxes:
                    logger.debug(f"belief_cruxes count={len(result.debate_cruxes)}")
        except Exception as e:
            logger.warning(f"belief_analysis_error error={e}")

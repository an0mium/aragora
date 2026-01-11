"""
Consensus phase for debate orchestration.

This module extracts the consensus/voting logic (Phase 3) from the
Arena._run_inner() method, handling:
- None mode: No consensus, combine all proposals
- Majority mode: Weighted voting with reputation/reliability/consistency/calibration
- Unanimous mode: All agents must agree
- Judge mode: Single judge synthesizes

Weight calculation and vote aggregation logic is extracted to:
- weight_calculator.py: WeightCalculator class
- vote_aggregator.py: VoteAggregator class, calculate_consensus_strength()
"""

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from aragora.agents.errors import _build_error_action
from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.debate.phases.weight_calculator import WeightCalculator
from aragora.debate.phases.vote_aggregator import (
    VoteAggregator,
    calculate_consensus_strength,
)

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
    verify_claims: Optional[Callable] = None  # Optional verification callback


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
        verify_claims: Optional[Callable] = None,
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
            self._verify_claims = callbacks.verify_claims
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
            self._verify_claims = verify_claims

    # Default timeout for consensus phase (can be overridden via protocol)
    # Judge mode needs more time due to LLM generation latency
    DEFAULT_CONSENSUS_TIMEOUT = AGENT_TIMEOUT_SECONDS + 60  # Agent timeout + margin

    # Per-judge timeout for fallback retries
    JUDGE_TIMEOUT_PER_ATTEMPT = AGENT_TIMEOUT_SECONDS - 60  # Slightly less than full agent timeout

    # Outer timeout for collecting ALL votes
    # This is a hard cap to prevent N*agent_timeout runaway. Votes are collected
    # in parallel, so this should be sufficient for most cases. With sequential
    # voting (rare), partial votes are returned when this expires.
    VOTE_COLLECTION_TIMEOUT = AGENT_TIMEOUT_SECONDS + 60  # Same as consensus timeout

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

            # Attempt formal verification if enabled and consensus reached
            if ctx.result.consensus_reached:
                await self._verify_consensus_formally(ctx)

        except asyncio.TimeoutError:
            logger.warning(
                f"consensus_timeout mode={consensus_mode} timeout={timeout}s, falling back to none"
            )
            await self._handle_fallback_consensus(ctx, reason="timeout")
        except Exception as e:
            category, msg, _ = _build_error_action(e, "consensus")
            logger.error(
                f"consensus_error mode={consensus_mode} category={category} error={msg}",
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

        This provides graceful degradation by:
        1. Trying to determine a winner from any collected votes
        2. Falling back to vote_tally if available
        3. Finally combining all proposals if no votes

        Args:
            ctx: The DebateContext with proposals and result
            reason: Description of why fallback was triggered
        """
        result = ctx.result
        proposals = ctx.proposals

        logger.info(f"consensus_fallback reason={reason} proposals={len(proposals)}")

        # Try to determine winner from votes or vote_tally
        winner_agent = None
        winner_confidence = 0.0

        # Check if we have votes in result
        if result.votes:
            vote_counts: dict[str, int] = {}
            for vote in result.votes:
                if hasattr(vote, 'choice') and vote.choice:
                    vote_counts[vote.choice] = vote_counts.get(vote.choice, 0) + 1
            if vote_counts:
                winner_agent = max(vote_counts.items(), key=lambda x: x[1])[0]
                total_votes = sum(vote_counts.values())
                winner_confidence = vote_counts[winner_agent] / total_votes if total_votes > 0 else 0.0
                logger.info(f"consensus_fallback_winner_from_votes winner={winner_agent} confidence={winner_confidence:.2f}")

        # Fallback to vote_tally if available
        if not winner_agent and ctx.vote_tally:
            winner_agent = max(ctx.vote_tally.items(), key=lambda x: x[1])[0]
            total_votes = sum(ctx.vote_tally.values())
            winner_confidence = ctx.vote_tally[winner_agent] / total_votes if total_votes > 0 else 0.5
            logger.info(f"consensus_fallback_winner_from_tally winner={winner_agent} confidence={winner_confidence:.2f}")

        # Set winner if determined
        if winner_agent:
            ctx.winner_agent = winner_agent
            result.winner = winner_agent
            result.confidence = winner_confidence
            # Use winner's proposal as final answer if available
            if winner_agent in proposals:
                result.final_answer = proposals[winner_agent]
            else:
                result.final_answer = (
                    f"[Consensus fallback ({reason}) - Winner: {winner_agent}]\n\n"
                    + "\n\n---\n\n".join(
                        f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
                    )
                )
            result.consensus_reached = True  # Partial consensus achieved via votes
            result.consensus_strength = "fallback"
        else:
            # No votes available - just combine proposals
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
            result.confidence = 0.5  # Neutral default, not failure
            result.consensus_strength = "fallback"

        logger.info(f"consensus_fallback reason={reason} winner={winner_agent}")

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

        # Apply calibration adjustments to vote confidences (E3)
        votes = self._apply_calibration_to_votes(votes, ctx)

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

        # Apply verification bonuses if enabled
        vote_counts = await self._apply_verification_bonuses(
            ctx, vote_counts, proposals, choice_mapping
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

        # Apply calibration adjustments to vote confidences (E3)
        votes = self._apply_calibration_to_votes(votes, ctx)

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
        # Note: voting_errors are excluded from total_voters because agents that
        # failed to vote shouldn't count against unanimity. If 5 agents vote and
        # 3 timeout, unanimity is 5/5=100%, not 5/8=62.5%.
        total_voters = len(votes) + user_vote_count
        if voting_errors > 0:
            logger.info(f"unanimous_vote_errors excluded={voting_errors} from total")

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
            result.confidence = 0.5  # Neutral default when no votes (not 0.0 failure)

    async def _handle_judge_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'judge' consensus mode - single judge synthesis with fallback.

        Tries the primary judge first, then falls back to alternative judges
        if the primary times out or fails. This provides resilience against
        individual agent failures during consensus.
        """
        result = ctx.result
        proposals = ctx.proposals

        if not self._select_judge or not self._generate_with_agent:
            logger.error("Judge consensus requires select_judge and generate_with_agent")
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            return

        judge_method = self.protocol.judge_selection if self.protocol else "random"
        task = ctx.env.task if ctx.env else ""

        # Build judge prompt (same for all judges)
        judge_prompt = (
            self._build_judge_prompt(proposals, task, result.critiques)
            if self._build_judge_prompt
            else f"Synthesize these proposals: {proposals}"
        )

        # Get judge candidates for fallback (if selector supports it)
        judge_candidates = []
        if hasattr(self._select_judge, '__self__') and hasattr(self._select_judge.__self__, 'get_judge_candidates'):
            # Using JudgeSelector instance - get ordered candidates
            try:
                judge_candidates = await self._select_judge.__self__.get_judge_candidates(
                    proposals, ctx.context_messages, max_candidates=3
                )
            except Exception as e:
                logger.debug(f"Failed to get judge candidates: {e}")

        # If no candidates from selector, use single judge selection
        if not judge_candidates:
            judge = await self._select_judge(proposals, ctx.context_messages)
            judge_candidates = [judge] if judge else []

        # Try each judge candidate until one succeeds
        tried_judges = []
        for judge in judge_candidates:
            if judge is None:
                continue

            tried_judges.append(judge.name)
            logger.info(f"judge_attempt judge={judge.name} method={judge_method} attempt={len(tried_judges)}")

            # Notify spectator
            if self._notify_spectator:
                self._notify_spectator(
                    "judge",
                    agent=judge.name,
                    details=f"Selected as judge via {judge_method}" +
                            (f" (attempt {len(tried_judges)})" if len(tried_judges) > 1 else ""),
                )

            # Emit judge selection hook
            if "on_judge_selected" in self.hooks:
                self.hooks["on_judge_selected"](judge.name, judge_method)

            try:
                # Use per-attempt timeout with overall timeout managed by execute()
                synthesis = await asyncio.wait_for(
                    self._generate_with_agent(judge, judge_prompt, ctx.context_messages),
                    timeout=self.JUDGE_TIMEOUT_PER_ATTEMPT
                )

                result.final_answer = synthesis
                result.consensus_reached = True
                result.confidence = 0.8
                # Set winner to judge for ELO tracking
                ctx.winner_agent = judge.name
                result.winner = judge.name

                logger.info(
                    f"judge_synthesis judge={judge.name} length={len(synthesis)} "
                    f"attempts={len(tried_judges)}"
                )

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

                # Success - return early
                return

            except asyncio.TimeoutError:
                logger.warning(
                    f"judge_timeout judge={judge.name} timeout={self.JUDGE_TIMEOUT_PER_ATTEMPT}s"
                )
                # Continue to next candidate
            except Exception as e:
                # Catch all exceptions to allow fallback to next judge candidate
                logger.error(f"judge_error judge={judge.name} error={type(e).__name__}: {e}")
                # Continue to next candidate

        # All judges failed - fall back to majority voting
        logger.warning(
            f"judge_all_failed tried={tried_judges} falling back to majority voting"
        )

        # Try majority consensus as fallback
        try:
            await self._handle_majority_consensus(ctx)
            if result.consensus_reached:
                logger.info("judge_fallback_majority_success")
                return
        except Exception as e:
            logger.warning(f"judge_fallback_majority_failed error={e}")

        # Majority also failed - use generic fallback
        await self._handle_fallback_consensus(ctx, reason="judge_and_majority_failed")

    async def _collect_votes(self, ctx: "DebateContext") -> list["Vote"]:
        """Collect votes from all agents with outer timeout protection.

        Uses VOTE_COLLECTION_TIMEOUT to prevent total vote collection time from
        exceeding reasonable bounds (N agents * per-agent timeout could be very long).
        If timeout is reached, returns partial votes collected so far.
        """
        if not self._vote_with_agent:
            logger.warning("No vote_with_agent callback, skipping votes")
            return []

        votes: list["Vote"] = []
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent):
            logger.debug(f"agent_voting agent={agent.name}")
            try:
                # Use complexity-scaled timeout from governor
                timeout = get_complexity_governor().get_scaled_timeout(
                    float(AGENT_TIMEOUT_SECONDS)
                )
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                # Catch all exceptions to prevent voting failures from crashing the phase
                logger.warning(f"vote_exception agent={agent.name} error={type(e).__name__}: {e}")
                return (agent, e)

        async def collect_all_votes():
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

        # Apply outer timeout to prevent N*agent_timeout runaway
        try:
            await asyncio.wait_for(
                collect_all_votes(),
                timeout=self.VOTE_COLLECTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"vote_collection_timeout collected={len(votes)} "
                f"expected={len(ctx.agents)} timeout={self.VOTE_COLLECTION_TIMEOUT}s"
            )
            # Return partial votes - better than nothing

        return votes

    async def _collect_votes_with_errors(
        self, ctx: "DebateContext"
    ) -> tuple[list["Vote"], int]:
        """Collect votes with error tracking and outer timeout protection.

        Used for unanimity mode where we need to track errors.
        Uses VOTE_COLLECTION_TIMEOUT to prevent runaway collection time.
        """
        if not self._vote_with_agent:
            return [], 0

        votes: list["Vote"] = []
        voting_errors = 0
        task = ctx.env.task if ctx.env else ""

        async def cast_vote(agent):
            logger.debug(f"agent_voting_unanimous agent={agent.name}")
            try:
                # Use complexity-scaled timeout from governor
                timeout = get_complexity_governor().get_scaled_timeout(
                    float(AGENT_TIMEOUT_SECONDS)
                )
                if self._with_timeout:
                    vote_result = await self._with_timeout(
                        self._vote_with_agent(agent, ctx.proposals, task),
                        agent.name,
                        timeout_seconds=timeout,
                    )
                else:
                    vote_result = await self._vote_with_agent(agent, ctx.proposals, task)
                return (agent, vote_result)
            except Exception as e:
                # Catch all exceptions to prevent voting failures from crashing the phase
                logger.warning(f"vote_exception_unanimous agent={agent.name} error={type(e).__name__}: {e}")
                return (agent, e)

        async def collect_all_votes():
            nonlocal voting_errors
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

        # Apply outer timeout to prevent N*agent_timeout runaway
        try:
            await asyncio.wait_for(
                collect_all_votes(),
                timeout=self.VOTE_COLLECTION_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Treat timeout as errors for missing votes
            missing = len(ctx.agents) - len(votes) - voting_errors
            voting_errors += missing
            logger.warning(
                f"vote_collection_timeout_unanimous collected={len(votes)} "
                f"errors={voting_errors} expected={len(ctx.agents)} "
                f"timeout={self.VOTE_COLLECTION_TIMEOUT}s"
            )

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
        """Pre-compute vote weights for all agents.

        Uses the extracted WeightCalculator class for cleaner code.
        """
        calculator = WeightCalculator(
            memory=self.memory,
            elo_system=self.elo_system,
            flip_detector=self.flip_detector,
            agent_weights=self.agent_weights,
            calibration_tracker=self.calibration_tracker,
            get_calibration_weight=self._get_calibration_weight,
        )
        return calculator.compute_weights(ctx.agents)

    def _apply_calibration_to_votes(
        self,
        votes: list["Vote"],
        ctx: "DebateContext",
    ) -> list["Vote"]:
        """Apply calibration adjustments to vote confidences.

        E3: Calibration-driven agent adaptation.

        Adjusts each vote's confidence based on the agent's historical
        calibration performance:
        - Overconfident agents have their confidence scaled down
        - Underconfident agents have their confidence scaled up
        - Well-calibrated agents are unchanged

        Args:
            votes: List of votes to adjust
            ctx: Debate context

        Returns:
            List of votes with adjusted confidences
        """
        if not self.calibration_tracker:
            return votes

        # Import here to avoid circular imports
        from aragora.agents.calibration import adjust_agent_confidence

        adjusted_votes: list[Any] = []
        for vote in votes:
            if isinstance(vote, Exception):
                adjusted_votes.append(vote)
                continue

            try:
                summary = self.calibration_tracker.get_calibration_summary(vote.agent)
                original_conf = vote.confidence
                adjusted_conf = adjust_agent_confidence(original_conf, summary)

                # Create new vote with adjusted confidence
                if adjusted_conf != original_conf:
                    from aragora.core import Vote
                    adjusted_vote = Vote(
                        agent=vote.agent,
                        choice=vote.choice,
                        reasoning=vote.reasoning,
                        confidence=adjusted_conf,
                        continue_debate=vote.continue_debate,
                    )
                    adjusted_votes.append(adjusted_vote)
                    logger.debug(
                        "calibration_confidence_adjustment agent=%s "
                        "original=%.2f adjusted=%.2f bias=%s",
                        vote.agent, original_conf, adjusted_conf, summary.bias_direction
                    )
                else:
                    adjusted_votes.append(vote)
            except Exception as e:
                logger.debug(f"Calibration adjustment failed for {vote.agent}: {e}")
                adjusted_votes.append(vote)

        return adjusted_votes

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

    async def _apply_verification_bonuses(
        self,
        ctx: "DebateContext",
        vote_counts: Counter,
        proposals: dict[str, str],
        choice_mapping: dict[str, str],
    ) -> Counter:
        """Apply verification bonuses to vote counts for verified proposals.

        When verify_claims_during_consensus is enabled in the protocol,
        proposals with verified claims get a weight bonus. Results are
        stored in ctx.result.verification_results for feedback loop.

        Args:
            ctx: DebateContext to store verification results
            vote_counts: Current vote counts by choice
            proposals: Dict of agent_name -> proposal_text
            choice_mapping: Mapping from vote choice to canonical form

        Returns:
            Updated vote counts with verification bonuses applied
        """
        if not self.protocol or not getattr(self.protocol, 'verify_claims_during_consensus', False):
            return vote_counts

        if not self._verify_claims:
            return vote_counts

        verification_bonus = getattr(self.protocol, 'verification_weight_bonus', 0.2)
        verification_timeout = getattr(self.protocol, 'verification_timeout_seconds', 5.0)
        result = ctx.result

        for agent_name, proposal_text in proposals.items():
            # Map agent name to canonical choice
            canonical = choice_mapping.get(agent_name, agent_name)
            if canonical not in vote_counts:
                continue

            try:
                # Verify top claims in the proposal (async with timeout)
                verification_result = await asyncio.wait_for(
                    self._verify_claims(proposal_text, limit=2),
                    timeout=verification_timeout
                )

                # Phase 11A: Handle both dict and int return types for backward compat
                if isinstance(verification_result, dict):
                    verified_count = verification_result.get("verified", 0)
                    disproven_count = verification_result.get("disproven", 0)
                else:
                    # Legacy: callback returns int
                    verified_count = verification_result or 0
                    disproven_count = 0

                # Store verification counts for feedback loop (Phase 11A: now includes disproven)
                if hasattr(result, 'verification_results'):
                    result.verification_results[agent_name] = {
                        "verified": verified_count,
                        "disproven": disproven_count,
                    }

                if verified_count > 0:
                    # Apply bonus: boost votes for this proposal
                    current_count = vote_counts[canonical]
                    bonus = current_count * verification_bonus * verified_count
                    vote_counts[canonical] = current_count + bonus

                    # Store bonus for feedback loop
                    if hasattr(result, 'verification_bonuses'):
                        result.verification_bonuses[agent_name] = bonus

                    logger.info(
                        f"verification_bonus agent={agent_name} "
                        f"verified={verified_count} bonus={bonus:.2f}"
                    )

                # Emit verification result event
                self._emit_verification_event(
                    ctx, agent_name, verified_count or 0, bonus if verified_count else 0.0
                )
            except asyncio.TimeoutError:
                logger.debug(f"verification_timeout agent={agent_name}")
                if hasattr(result, 'verification_results'):
                    result.verification_results[agent_name] = -1  # Timeout indicator
                self._emit_verification_event(ctx, agent_name, -1, 0.0, timeout=True)
            except Exception as e:
                logger.debug(f"verification_error agent={agent_name} error={e}")

        # Phase 10E: Update ELO based on verification results
        await self._update_elo_from_verification(ctx)

        return vote_counts

    async def _update_elo_from_verification(self, ctx: "DebateContext") -> None:
        """
        Update agent ELO ratings based on verification results.

        Phase 10E: Verification-to-ELO Integration.

        When claims are formally verified, the authoring agent's ELO is adjusted:
        - Verified claims: ELO boost (quality reasoning)
        - Disproven claims: ELO penalty (flawed reasoning)
        - Timeouts/errors: No change

        Args:
            ctx: DebateContext with verification_results
        """
        if not self.elo_system:
            return

        result = ctx.result
        if not hasattr(result, 'verification_results') or not result.verification_results:
            return

        # Extract domain from context
        domain = "general"
        if self._extract_debate_domain:
            try:
                domain = self._extract_debate_domain()
            except Exception as e:
                logger.debug(f"Failed to extract debate domain: {e}")

        # Process verification results for each agent
        for agent_name, verification_data in result.verification_results.items():
            # Phase 11A: Handle both dict and int formats for backward compatibility
            if isinstance(verification_data, dict):
                verified_count = verification_data.get("verified", 0)
                disproven_count = verification_data.get("disproven", 0)
            else:
                # Legacy format: int value
                # Skip timeouts (indicated by -1) and errors
                if verification_data < 0:
                    continue
                verified_count = verification_data
                disproven_count = 0

            # Skip if nothing to report
            if verified_count == 0 and disproven_count == 0:
                continue

            try:
                # Phase 11A: Now properly tracks both verified and disproven claims
                change = self.elo_system.update_from_verification(
                    agent_name=agent_name,
                    domain=domain,
                    verified_count=verified_count,
                    disproven_count=disproven_count,
                )

                if change != 0:
                    logger.debug(
                        f"verification_elo_applied agent={agent_name} "
                        f"verified={verified_count} disproven={disproven_count} "
                        f"change={change:.1f}"
                    )
            except Exception as e:
                logger.debug(f"verification_elo_error agent={agent_name} error={e}")

    def _emit_verification_event(
        self,
        ctx: "DebateContext",
        agent_name: str,
        verified_count: int,
        bonus: float,
        timeout: bool = False,
    ) -> None:
        """Emit CLAIM_VERIFICATION_RESULT event to WebSocket.

        Args:
            ctx: DebateContext with event_emitter
            agent_name: Name of agent whose proposal was verified
            verified_count: Number of verified claims (-1 if timeout)
            bonus: Vote bonus applied
            timeout: Whether verification timed out
        """
        if not ctx.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            ctx.event_emitter.emit(StreamEvent(
                type=StreamEventType.CLAIM_VERIFICATION_RESULT,
                loop_id=ctx.loop_id,
                agent=agent_name,
                data={
                    "agent": agent_name,
                    "verified_count": verified_count,
                    "bonus_applied": bonus,
                    "timeout": timeout,
                    "debate_id": ctx.debate_id,
                }
            ))
        except Exception as e:
            logger.debug(f"verification_event_error: {e}")

    def _normalize_choice_to_agent(
        self,
        choice: str,
        agents: list,
        proposals: dict[str, str],
    ) -> str:
        """Normalize a vote choice to an agent name.

        Handles common mismatches:
        - Case differences: "Claude" vs "claude"
        - Partial names: "claude" vs "claude-visionary"
        - Proposal keys vs agent names

        Args:
            choice: The raw vote choice string
            agents: List of Agent objects with .name attribute
            proposals: Dict of agent_name -> proposal text

        Returns:
            The matching agent name, or the original choice if no match
        """
        if not choice:
            return choice

        choice_lower = choice.lower().strip()

        # Direct match in proposals keys
        if choice in proposals:
            return choice

        # Case-insensitive match in proposals keys
        for agent_name in proposals:
            if agent_name.lower() == choice_lower:
                return agent_name

        # Try to match agent names (handles partial matches)
        for agent in agents:
            agent_name = agent.name
            agent_lower = agent_name.lower()

            # Exact match (case-insensitive)
            if agent_lower == choice_lower:
                return agent_name

            # Choice is a prefix of agent name (e.g., "claude" matches "claude-visionary")
            if agent_lower.startswith(choice_lower):
                return agent_name

            # Agent name is a prefix of choice (e.g., "claude" matches "claude's proposal")
            if choice_lower.startswith(agent_lower):
                return agent_name

            # Contains match for hyphenated names
            if "-" in agent_name:
                base_name = agent_name.split("-")[0].lower()
                if base_name == choice_lower or choice_lower.startswith(base_name):
                    return agent_name

        # No match found - return original (logging for debugging)
        logger.debug(f"vote_choice_no_match choice={choice} agents={[a.name for a in agents]}")
        return choice

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
            result.confidence = 0.5  # Neutral default when no winner (not 0.0 failure)
            return

        winner_choice, count = most_common[0]
        threshold = self.protocol.consensus_threshold if self.protocol else 0.5

        # Normalize vote choice to agent name
        # Vote choices might be agent names with different casing/format
        winner_agent = self._normalize_choice_to_agent(winner_choice, ctx.agents, proposals)

        result.final_answer = proposals.get(
            winner_agent, list(proposals.values())[0] if proposals else ""
        )
        result.consensus_reached = (count / total_votes >= threshold) if total_votes > 0 else False
        result.confidence = count / total_votes if total_votes > 0 else 0.5
        ctx.winner_agent = winner_agent
        result.winner = winner_agent  # Set winner for ELO tracking (agent name)

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
            if agent != winner_agent:
                result.dissenting_views.append(f"[{agent}]: {prop}")

        logger.info(f"consensus_winner winner={winner_agent} votes={count}/{len(ctx.agents)}")

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "consensus",
                details=f"Majority vote: {winner_agent}",
                metric=result.confidence,
            )

        # Record consensus
        if self.recorder:
            try:
                self.recorder.record_phase_change(f"consensus_reached: {winner_agent}")
            except Exception as e:
                logger.debug(f"Recorder error for consensus: {e}")

        # Finalize for truth-grounded personas
        if self.position_tracker:
            try:
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                self.position_tracker.finalize_debate(
                    debate_id=debate_id,
                    winning_agent=winner_agent,
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
                        correct = canonical == winner_agent
                        self.calibration_tracker.record_prediction(
                            agent=v.agent,
                            confidence=v.confidence,
                            correct=correct,
                            domain=domain,
                            debate_id=debate_id,
                        )
                logger.debug(f"calibration_recorded predictions={len(result.votes)}")
            except Exception as e:
                category, msg, exc_info = _build_error_action(e, "calibration")
                logger.warning(f"calibration_error category={category} error={msg}", exc_info=exc_info)

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
                category, msg, exc_info = _build_error_action(e, "calibration")
                logger.warning(f"calibration_error_unanimous category={category} error={msg}", exc_info=exc_info)

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

    async def _verify_consensus_formally(self, ctx: "DebateContext") -> None:
        """
        Attempt formal verification of consensus claims using Lean4/Z3.

        This method is called after consensus is reached, if formal_verification_enabled
        is set in the protocol. It attempts to translate the consensus to a formal
        language and verify it using available proof backends.

        The verification result is stored in ctx.result.formal_verification as a dict
        containing status, proof details, and any error messages.

        Args:
            ctx: The DebateContext with result containing final_answer
        """
        if not self.protocol:
            return

        # Check if formal verification is enabled
        formal_enabled = getattr(self.protocol, 'formal_verification_enabled', False)
        if not formal_enabled:
            return

        result = ctx.result
        if not result.final_answer:
            return

        # Get verification timeout from protocol
        timeout = getattr(self.protocol, 'formal_verification_timeout', 30.0)
        languages = getattr(self.protocol, 'formal_verification_languages', ['z3_smt'])

        logger.info(f"formal_verification_start languages={languages} timeout={timeout}")

        try:
            # Import verification manager (deferred to avoid circular imports)
            from aragora.verification.formal import get_formal_verification_manager

            manager = get_formal_verification_manager()

            # Check if any backend is available
            status = manager.status_report()
            if not status.get('any_available', False):
                logger.debug("formal_verification_skip no backends available")
                result.formal_verification = {
                    "status": "skipped",
                    "reason": "No formal verification backends available",
                    "backends_checked": languages,
                }
                return

            # Attempt verification with timeout
            verification_result = await asyncio.wait_for(
                manager.attempt_formal_verification(
                    claim=result.final_answer,
                    claim_type="DEBATE_CONSENSUS",
                    context=ctx.env.task if ctx.env else "",
                    timeout_seconds=timeout,
                ),
                timeout=timeout + 5.0,  # Buffer for manager overhead
            )

            # Store result
            result.formal_verification = verification_result.to_dict()

            logger.info(
                f"formal_verification_complete status={verification_result.status.value} "
                f"language={verification_result.language.value if verification_result.language else 'none'} "
                f"verified={verification_result.is_verified}"
            )

            # Emit event if emitter is available
            if ctx.event_emitter:
                try:
                    from aragora.server.stream import StreamEvent, StreamEventType

                    ctx.event_emitter.emit(StreamEvent(
                        type=StreamEventType.FORMAL_VERIFICATION_RESULT,
                        loop_id=ctx.loop_id,
                        data={
                            "debate_id": ctx.debate_id,
                            "status": verification_result.status.value,
                            "is_verified": verification_result.is_verified,
                            "language": verification_result.language.value if verification_result.language else None,
                            "formal_statement": verification_result.formal_statement[:500] if verification_result.formal_statement else None,
                        }
                    ))
                except Exception as e:
                    logger.debug(f"formal_verification_event_error: {e}")

        except asyncio.TimeoutError:
            logger.warning(f"formal_verification_timeout timeout={timeout}s")
            result.formal_verification = {
                "status": "timeout",
                "timeout_seconds": timeout,
                "is_verified": False,
            }
        except ImportError as e:
            logger.debug(f"formal_verification_import_error: {e}")
            result.formal_verification = {
                "status": "unavailable",
                "reason": "Formal verification module not available",
                "is_verified": False,
            }
        except Exception as e:
            logger.warning(f"formal_verification_error: {e}")
            result.formal_verification = {
                "status": "error",
                "error": str(e),
                "is_verified": False,
            }

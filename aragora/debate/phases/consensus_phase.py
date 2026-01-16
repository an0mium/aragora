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
- vote_collector.py: VoteCollector class for parallel vote collection
- winner_selector.py: WinnerSelector class for consensus determination
- consensus_verification.py: ConsensusVerifier class for claim verification
- synthesis_generator.py: SynthesisGenerator class for final synthesis
"""

__all__ = [
    "ConsensusDependencies",
    "ConsensusCallbacks",
    "ConsensusPhase",
]

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.agents.errors import _build_error_action
from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.debate.phases.consensus_verification import ConsensusVerifier
from aragora.debate.phases.synthesis_generator import SynthesisGenerator
from aragora.debate.phases.vote_collector import VoteCollector, VoteCollectorConfig
from aragora.debate.phases.weight_calculator import WeightCalculator
from aragora.debate.phases.winner_selector import WinnerSelector
from aragora.server.stream.arena_hooks import streaming_task_context

if TYPE_CHECKING:
    from aragora.core import Agent, Vote
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

        # Initialize helper classes
        self._winner_selector = WinnerSelector(
            protocol=self.protocol,
            position_tracker=self.position_tracker,
            calibration_tracker=self.calibration_tracker,
            recorder=self.recorder,
            notify_spectator=self._notify_spectator,
            extract_debate_domain=self._extract_debate_domain,
            get_belief_analyzer=self._get_belief_analyzer,
        )

        self._consensus_verifier = ConsensusVerifier(
            protocol=self.protocol,
            elo_system=self.elo_system,
            verify_claims=self._verify_claims,
            extract_debate_domain=self._extract_debate_domain,
        )

        self._synthesis_generator = SynthesisGenerator(
            protocol=self.protocol,
            hooks=self.hooks,
            notify_spectator=self._notify_spectator,
        )

    # Default timeout for consensus phase (can be overridden via protocol)
    # Judge mode needs more time due to LLM generation latency
    DEFAULT_CONSENSUS_TIMEOUT = AGENT_TIMEOUT_SECONDS + 60  # Agent timeout + margin

    # Per-judge timeout for fallback retries
    JUDGE_TIMEOUT_PER_ATTEMPT = AGENT_TIMEOUT_SECONDS - 60

    # Outer timeout for collecting ALL votes
    VOTE_COLLECTION_TIMEOUT = AGENT_TIMEOUT_SECONDS + 60

    @property
    def _vote_collector(self) -> VoteCollector:
        """Lazy-initialized VoteCollector instance."""
        if not hasattr(self, "_vote_collector_instance"):
            config = VoteCollectorConfig(
                vote_with_agent=self._vote_with_agent,
                with_timeout=self._with_timeout,
                notify_spectator=self._notify_spectator,
                hooks=self.hooks,
                recorder=self.recorder,
                position_tracker=self.position_tracker,
                group_similar_votes=self._group_similar_votes,
                vote_collection_timeout=self.VOTE_COLLECTION_TIMEOUT,
                agent_timeout=AGENT_TIMEOUT_SECONDS,
            )
            self._vote_collector_instance = VoteCollector(config)
        return self._vote_collector_instance

    async def execute(self, ctx: "DebateContext") -> None:
        """
        Execute the consensus phase with fallback mechanisms.

        This method wraps consensus execution with:
        - Timeout protection (default 120s)
        - Exception handling with fallback to 'none' mode
        - Graceful degradation when agents fail
        - GUARANTEED synthesis generation (never fails silently)

        Args:
            ctx: The DebateContext with proposals and result
        """
        consensus_mode = self.protocol.consensus if self.protocol else "none"
        logger.info(f"consensus_phase_start mode={consensus_mode}")

        # Get timeout from protocol or use default
        timeout = getattr(self.protocol, "consensus_timeout", self.DEFAULT_CONSENSUS_TIMEOUT)

        try:
            await asyncio.wait_for(self._execute_consensus(ctx, consensus_mode), timeout=timeout)

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
                exc_info=True,
            )
            await self._handle_fallback_consensus(ctx, reason=f"error: {type(e).__name__}")

        # Always generate final synthesis regardless of consensus mode
        try:
            synthesis_generated = await self._synthesis_generator.generate_mandatory_synthesis(ctx)

            if not synthesis_generated:
                logger.error("synthesis_failed_all_fallbacks - this should not happen")
                if ctx.proposals:
                    fallback_synthesis = (
                        f"## Debate Summary\n\n{list(ctx.proposals.values())[0][:1000]}"
                    )
                    ctx.result.synthesis = fallback_synthesis
                    ctx.result.final_answer = fallback_synthesis
                    try:
                        if self.hooks and "on_message" in self.hooks:
                            self.hooks["on_message"](
                                agent="synthesis-agent",
                                content=fallback_synthesis,
                                role="synthesis",
                                round_num=(self.protocol.rounds if self.protocol else 3) + 1,
                            )
                    except Exception as hook_err:
                        logger.warning(f"on_message hook failed in fallback: {hook_err}")
        except Exception as e:
            logger.error(f"synthesis_or_hooks_failed: {e}", exc_info=True)
        finally:
            logger.info("consensus_phase_emitting_guaranteed_events")
            self._emit_guaranteed_events(ctx)

    def _emit_guaranteed_events(self, ctx: "DebateContext") -> None:
        """Emit consensus and debate_end events with guaranteed delivery."""
        if not ctx.result:
            return

        if self.hooks and "on_consensus" in self.hooks:
            try:
                self.hooks["on_consensus"](
                    reached=ctx.result.consensus_reached,
                    confidence=ctx.result.confidence,
                    answer=ctx.result.final_answer,
                    synthesis=ctx.result.synthesis or "",
                )
                logger.debug("consensus_event_emitted reached=%s", ctx.result.consensus_reached)
            except Exception as e:
                logger.warning(f"Failed to emit consensus event: {e}")

        if self.hooks and "on_debate_end" in self.hooks:
            try:
                duration = time.time() - ctx.start_time if hasattr(ctx, "start_time") else 0.0
                self.hooks["on_debate_end"](
                    duration=duration,
                    rounds=ctx.result.rounds_used,
                )
                logger.debug("debate_end_event_emitted duration=%.1fs", duration)
            except Exception as e:
                logger.warning(f"Failed to emit debate_end event: {e}")

    async def _execute_consensus(self, ctx: "DebateContext", consensus_mode: str) -> None:
        """Execute the consensus logic for the given mode."""
        normalized = consensus_mode
        threshold_override: float | None = None

        if consensus_mode == "weighted":
            normalized = "majority"
        elif consensus_mode == "supermajority":
            normalized = "majority"
            threshold_override = max(getattr(self.protocol, "consensus_threshold", 0.6), 2 / 3)
        elif consensus_mode == "any":
            normalized = "majority"
            threshold_override = 0.0

        if normalized == "none":
            await self._handle_none_consensus(ctx)
        elif normalized == "majority":
            await self._handle_majority_consensus(ctx, threshold_override=threshold_override)
        elif normalized == "unanimous":
            await self._handle_unanimous_consensus(ctx)
        elif normalized == "judge":
            await self._handle_judge_consensus(ctx)
        else:
            logger.warning(f"Unknown consensus mode: {consensus_mode}, using none")
            await self._handle_none_consensus(ctx)

    async def _handle_fallback_consensus(self, ctx: "DebateContext", reason: str) -> None:
        """Handle consensus fallback when the primary mechanism fails."""
        result = ctx.result
        proposals = ctx.proposals

        logger.info(f"consensus_fallback reason={reason} proposals={len(proposals)}")

        winner_agent = None
        winner_confidence = 0.0

        if result.votes:
            vote_counts: dict[str, int] = {}
            for vote in result.votes:
                if hasattr(vote, "choice") and vote.choice:
                    vote_counts[vote.choice] = vote_counts.get(vote.choice, 0) + 1
            if vote_counts:
                winner_agent = max(vote_counts.items(), key=lambda x: x[1])[0]
                total_votes = sum(vote_counts.values())
                winner_confidence = (
                    vote_counts[winner_agent] / total_votes if total_votes > 0 else 0.0
                )
                logger.info(
                    f"consensus_fallback_winner_from_votes winner={winner_agent} "
                    f"confidence={winner_confidence:.2f}"
                )

        if not winner_agent and ctx.vote_tally:
            winner_agent = max(ctx.vote_tally.items(), key=lambda x: x[1])[0]
            total_votes = sum(ctx.vote_tally.values())
            winner_confidence = (
                ctx.vote_tally[winner_agent] / total_votes if total_votes > 0 else 0.5
            )
            logger.info(
                f"consensus_fallback_winner_from_tally winner={winner_agent} "
                f"confidence={winner_confidence:.2f}"
            )

        if winner_agent:
            ctx.winner_agent = winner_agent
            result.winner = winner_agent
            result.confidence = winner_confidence
            if winner_agent in proposals:
                result.final_answer = proposals[winner_agent]
            else:
                result.final_answer = (
                    f"[Consensus fallback ({reason}) - Winner: {winner_agent}]\n\n"
                    + "\n\n---\n\n".join(f"[{agent}]:\n{prop}" for agent, prop in proposals.items())
                )
            result.consensus_reached = True
            result.consensus_strength = "fallback"
        else:
            if proposals:
                result.final_answer = f"[Consensus fallback ({reason})]\n\n" + "\n\n---\n\n".join(
                    f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
                )
            else:
                result.final_answer = f"[No proposals available - consensus fallback ({reason})]"
            result.consensus_reached = False
            result.confidence = 0.5
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

    async def _handle_majority_consensus(
        self,
        ctx: "DebateContext",
        threshold_override: float | None = None,
    ) -> None:
        """Handle 'majority' consensus mode - weighted voting."""
        result = ctx.result
        proposals = ctx.proposals

        # Cast votes from all agents
        votes = await self._collect_votes(ctx)

        # Apply calibration adjustments to vote confidences
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
        vote_counts = await self._consensus_verifier.apply_verification_bonuses(
            ctx, vote_counts, proposals, choice_mapping
        )

        # Apply evidence citation bonuses if enabled
        vote_counts = self._apply_evidence_citation_bonuses(ctx, votes, vote_counts, choice_mapping)

        ctx.vote_tally = dict(vote_counts)

        # Determine winner using WinnerSelector
        self._winner_selector.determine_majority_winner(
            ctx,
            vote_counts,
            total_weighted,
            choice_mapping,
            normalize_choice=self._normalize_choice_to_agent,
            threshold_override=threshold_override,
        )

        # Analyze belief network for cruxes
        self._winner_selector.analyze_belief_network(ctx)

    async def _handle_unanimous_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'unanimous' consensus mode - all must agree."""
        result = ctx.result
        proposals = ctx.proposals

        votes, voting_errors = await self._collect_votes_with_errors(ctx)
        votes = self._apply_calibration_to_votes(votes, ctx)
        result.votes.extend(votes)

        vote_groups, choice_mapping = self._compute_vote_groups(votes)

        vote_counts: Counter[str] = Counter()
        for v in votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                vote_counts[canonical] += 1

        if self._drain_user_events:
            self._drain_user_events()

        user_vote_weight = getattr(self.protocol, "user_vote_weight", 0.0)
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

        total_voters = len(votes) + user_vote_count
        if voting_errors > 0:
            logger.info(f"unanimous_vote_errors excluded={voting_errors} from total")

        most_common = vote_counts.most_common(1) if vote_counts else []
        if most_common and total_voters > 0:
            winner, count = most_common[0]
            unanimity_ratio = count / total_voters

            if unanimity_ratio >= 1.0:
                self._winner_selector.set_unanimous_winner(
                    ctx, winner, unanimity_ratio, total_voters, count
                )
            else:
                self._winner_selector.set_no_unanimity(
                    ctx, winner, unanimity_ratio, total_voters, count, choice_mapping
                )
        else:
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            result.confidence = 0.5

    async def _handle_judge_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'judge' consensus mode - single judge synthesis with fallback."""
        result = ctx.result
        proposals = ctx.proposals

        if not self._select_judge or not self._generate_with_agent:
            logger.error("Judge consensus requires select_judge and generate_with_agent")
            result.final_answer = list(proposals.values())[0] if proposals else ""
            result.consensus_reached = False
            return

        judge_method = self.protocol.judge_selection if self.protocol else "random"
        task = ctx.env.task if ctx.env else ""

        judge_prompt = (
            self._build_judge_prompt(proposals, task, result.critiques)
            if self._build_judge_prompt
            else f"Synthesize these proposals: {proposals}"
        )

        judge_candidates = []
        if hasattr(self._select_judge, "__self__") and hasattr(
            self._select_judge.__self__, "get_judge_candidates"
        ):
            try:
                judge_candidates = await self._select_judge.__self__.get_judge_candidates(
                    proposals, ctx.context_messages, max_candidates=3
                )
            except Exception as e:
                logger.debug(f"Failed to get judge candidates: {e}")

        if not judge_candidates:
            judge = await self._select_judge(proposals, ctx.context_messages)
            judge_candidates = [judge] if judge else []

        tried_judges = []
        for judge in judge_candidates:
            if judge is None:
                continue

            tried_judges.append(judge.name)
            logger.info(
                f"judge_attempt judge={judge.name} method={judge_method} attempt={len(tried_judges)}"
            )

            if self._notify_spectator:
                self._notify_spectator(
                    "judge",
                    agent=judge.name,
                    details=f"Selected as judge via {judge_method}"
                    + (f" (attempt {len(tried_judges)})" if len(tried_judges) > 1 else ""),
                )

            if "on_judge_selected" in self.hooks:
                self.hooks["on_judge_selected"](judge.name, judge_method)

            try:
                task_id = f"{judge.name}:judge_synthesis"
                with streaming_task_context(task_id):
                    synthesis = await asyncio.wait_for(
                        self._generate_with_agent(judge, judge_prompt, ctx.context_messages),
                        timeout=self.JUDGE_TIMEOUT_PER_ATTEMPT,
                    )

                result.final_answer = synthesis
                result.consensus_reached = True
                result.confidence = 0.8
                ctx.winner_agent = judge.name
                result.winner = judge.name

                logger.info(
                    f"judge_synthesis judge={judge.name} length={len(synthesis)} "
                    f"attempts={len(tried_judges)}"
                )

                if self._notify_spectator:
                    self._notify_spectator(
                        "consensus",
                        agent=judge.name,
                        details=f"Judge synthesis ({len(synthesis)} chars)",
                        metric=0.8,
                    )

                if "on_message" in self.hooks:
                    rounds = self.protocol.rounds if self.protocol else 0
                    self.hooks["on_message"](
                        agent=judge.name,
                        content=synthesis,
                        role="judge",
                        round_num=rounds + 1,
                    )

                return

            except asyncio.TimeoutError:
                logger.warning(
                    f"judge_timeout judge={judge.name} timeout={self.JUDGE_TIMEOUT_PER_ATTEMPT}s"
                )
            except Exception as e:
                logger.error(f"judge_error judge={judge.name} error={type(e).__name__}: {e}")

        logger.warning(f"judge_all_failed tried={tried_judges} falling back to majority voting")

        try:
            await self._handle_majority_consensus(ctx)
            if result.consensus_reached:
                logger.info("judge_fallback_majority_success")
                return
        except Exception as e:
            logger.warning(f"judge_fallback_majority_failed error={e}")

        await self._handle_fallback_consensus(ctx, reason="judge_and_majority_failed")

    async def _collect_votes(self, ctx: "DebateContext") -> list["Vote"]:
        """Collect votes from all agents with outer timeout protection."""
        return await self._vote_collector.collect_votes(ctx)

    async def _collect_votes_with_errors(self, ctx: "DebateContext") -> tuple[list["Vote"], int]:
        """Collect votes with error tracking and outer timeout protection."""
        return await self._vote_collector.collect_votes_with_errors(ctx)

    def _compute_vote_groups(
        self, votes: list["Vote"]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Group similar votes and create choice mapping."""
        return self._vote_collector.compute_vote_groups(votes)

    def _compute_vote_weights(self, ctx: "DebateContext") -> dict[str, float]:
        """Pre-compute vote weights for all agents."""
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
        """Apply calibration adjustments to vote confidences."""
        if not self.calibration_tracker:
            return votes

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
                        vote.agent,
                        original_conf,
                        adjusted_conf,
                        summary.bias_direction,
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
                vote_counts[canonical] += weight
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

        base_user_weight = getattr(self.protocol, "user_vote_weight", 0.5)

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
                vote_counts[canonical] += final_weight
                total_weighted += final_weight

                logger.debug(
                    f"user_vote user={user_vote.get('user_id', 'anonymous')} "
                    f"choice={choice} intensity={intensity} weight={final_weight:.2f}"
                )

        return vote_counts, total_weighted

    def _apply_evidence_citation_bonuses(
        self,
        ctx: "DebateContext",
        votes: list["Vote"],
        vote_counts: Counter,
        choice_mapping: dict[str, str],
    ) -> Counter:
        """Apply evidence citation bonuses to vote counts."""
        if not self.protocol or not getattr(self.protocol, "enable_evidence_weighting", False):
            return vote_counts

        evidence_pack = getattr(ctx, "evidence_pack", None)
        if not evidence_pack or not hasattr(evidence_pack, "snippets"):
            return vote_counts

        evidence_ids = {s.id for s in evidence_pack.snippets}
        if not evidence_ids:
            return vote_counts

        evidence_bonus = getattr(self.protocol, "evidence_citation_bonus", 0.15)
        evidence_citations: dict[str, int] = {}

        for vote in votes:
            if isinstance(vote, Exception):
                continue

            cited_ids = set(re.findall(r"EVID-([a-zA-Z0-9]+)", vote.reasoning))
            valid_citations = len(cited_ids & evidence_ids)

            if valid_citations > 0:
                canonical = choice_mapping.get(vote.choice, vote.choice)
                if canonical in vote_counts:
                    current_count = vote_counts[canonical]
                    bonus = evidence_bonus * valid_citations
                    vote_counts[canonical] = current_count + bonus

                    evidence_citations[vote.agent] = valid_citations
                    logger.debug(
                        f"evidence_citation_bonus agent={vote.agent} "
                        f"citations={valid_citations} bonus={bonus:.2f}"
                    )

        result = ctx.result
        if evidence_citations and hasattr(result, "metadata"):
            if result.metadata is None:
                result.metadata = {}
            result.metadata["evidence_citations"] = evidence_citations

        if evidence_citations:
            logger.info(
                f"evidence_weighting applied: {len(evidence_citations)} agents cited evidence, "
                f"total citations={sum(evidence_citations.values())}"
            )

        return vote_counts

    def _normalize_choice_to_agent(
        self,
        choice: str,
        agents: list,
        proposals: dict[str, str],
    ) -> str:
        """Normalize a vote choice to an agent name."""
        if not choice:
            return choice

        choice_lower = choice.lower().strip()

        if choice in proposals:
            return choice

        for agent_name in proposals:
            if agent_name.lower() == choice_lower:
                return agent_name

        for agent in agents:
            agent_name = agent.name
            agent_lower = agent_name.lower()

            if agent_lower == choice_lower:
                return agent_name

            if agent_lower.startswith(choice_lower):
                return agent_name

            if choice_lower.startswith(agent_lower):
                return agent_name

            if "-" in agent_name:
                base_name = agent_name.split("-")[0].lower()
                if base_name == choice_lower or choice_lower.startswith(base_name):
                    return agent_name

        logger.debug(f"vote_choice_no_match choice={choice} agents={[a.name for a in agents]}")
        return choice

    async def _verify_consensus_formally(self, ctx: "DebateContext") -> None:
        """Attempt formal verification of consensus claims using Lean4/Z3."""
        if not self.protocol:
            return

        formal_enabled = getattr(self.protocol, "formal_verification_enabled", False)
        if not formal_enabled:
            return

        result = ctx.result
        if not result.final_answer:
            return

        timeout = getattr(self.protocol, "formal_verification_timeout", 30.0)
        languages = getattr(self.protocol, "formal_verification_languages", ["z3_smt"])

        logger.info(f"formal_verification_start languages={languages} timeout={timeout}")

        try:
            from aragora.verification.formal import get_formal_verification_manager

            manager = get_formal_verification_manager()

            status = manager.status_report()
            if not status.get("any_available", False):
                logger.debug("formal_verification_skip no backends available")
                result.formal_verification = {
                    "status": "skipped",
                    "reason": "No formal verification backends available",
                    "backends_checked": languages,
                }
                return

            verification_result = await asyncio.wait_for(
                manager.attempt_formal_verification(
                    claim=result.final_answer,
                    claim_type="DEBATE_CONSENSUS",
                    context=ctx.env.task if ctx.env else "",
                    timeout_seconds=timeout,
                ),
                timeout=timeout + 5.0,
            )

            result.formal_verification = verification_result.to_dict()

            logger.info(
                f"formal_verification_complete status={verification_result.status.value} "
                f"language={verification_result.language.value if verification_result.language else 'none'} "
                f"verified={verification_result.is_verified}"
            )

            if ctx.event_emitter:
                try:
                    from aragora.server.stream import StreamEvent, StreamEventType

                    ctx.event_emitter.emit(
                        StreamEvent(
                            type=StreamEventType.FORMAL_VERIFICATION_RESULT,
                            loop_id=ctx.loop_id,
                            data={
                                "debate_id": ctx.debate_id,
                                "status": verification_result.status.value,
                                "is_verified": verification_result.is_verified,
                                "language": (
                                    verification_result.language.value
                                    if verification_result.language
                                    else None
                                ),
                                "formal_statement": (
                                    verification_result.formal_statement[:500]
                                    if verification_result.formal_statement
                                    else None
                                ),
                            },
                        )
                    )
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

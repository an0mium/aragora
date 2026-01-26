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
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.agents.errors import _build_error_action
from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.debate.phases.consensus_verification import ConsensusVerifier
from aragora.debate.phases.synthesis_generator import SynthesisGenerator
from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator
from aragora.debate.phases.vote_collector import VoteCollector, VoteCollectorConfig
from aragora.debate.phases.weight_calculator import WeightCalculator
from aragora.debate.phases.winner_selector import WinnerSelector
from aragora.server.stream.arena_hooks import streaming_task_context

if TYPE_CHECKING:
    from aragora.core import Vote
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

        self._vote_bonus_calculator = VoteBonusCalculator(protocol=self.protocol)

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
            # Get position shuffling config from protocol if available
            enable_position_shuffling = getattr(self.protocol, "enable_position_shuffling", False)
            position_shuffling_permutations = getattr(
                self.protocol, "position_shuffling_permutations", 3
            )

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
                # Agent-as-a-Judge position bias mitigation
                enable_position_shuffling=enable_position_shuffling,
                position_shuffling_permutations=position_shuffling_permutations,
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
        # Check for cancellation before starting
        if ctx.cancellation_token and ctx.cancellation_token.is_cancelled:
            from aragora.debate.cancellation import DebateCancelled

            raise DebateCancelled(ctx.cancellation_token.reason)

        # Trigger PRE_CONSENSUS hook if hook_manager is available
        if ctx.hook_manager:
            try:
                await ctx.hook_manager.trigger("pre_consensus", ctx=ctx, proposals=ctx.proposals)
            except Exception as e:
                logger.debug(f"PRE_CONSENSUS hook failed: {e}")

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

        # Trigger POST_CONSENSUS hook if hook_manager is available
        if ctx.hook_manager:
            try:
                # Use asyncio.create_task for async hook in sync method
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        ctx.hook_manager.trigger(
                            "post_consensus",
                            ctx=ctx,
                            result=ctx.result,
                            consensus_reached=ctx.result.consensus_reached,
                        )
                    )
            except Exception as e:
                logger.debug(f"POST_CONSENSUS hook failed: {e}")

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
        elif normalized == "byzantine":
            await self._handle_byzantine_consensus(ctx)
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
            tally_total = sum(ctx.vote_tally.values())
            winner_confidence = (
                ctx.vote_tally[winner_agent] / tally_total if tally_total > 0 else 0.5
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

        if proposals:
            result.final_answer = "\n\n---\n\n".join(
                f"[{agent}]:\n{prop}" for agent, prop in proposals.items()
            )
        else:
            result.final_answer = "[No proposals available - consensus mode 'none']"
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

        # Pre-compute vote weights (pass votes for bias mitigation)
        vote_weight_cache = self._compute_vote_weights(ctx, votes=votes)

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

        # Adjust individual vote confidences based on verification results
        # This cross-pollinates formal verification with vote confidence
        if hasattr(result, "verification_results") and result.verification_results:
            self._consensus_verifier.adjust_vote_confidence_from_verification(
                votes, result.verification_results, proposals
            )

        # Apply evidence citation bonuses if enabled
        vote_counts = self._vote_bonus_calculator.apply_evidence_citation_bonuses(
            ctx, votes, vote_counts, choice_mapping
        )

        # Apply process-based evaluation bonuses if enabled (Agent-as-a-Judge)
        vote_counts = await self._vote_bonus_calculator.apply_process_evaluation_bonuses(
            ctx, vote_counts, choice_mapping
        )

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
            result.confidence = 0.5
            return

        # Check for judge deliberation mode (Agent-as-a-Judge enhancement)
        enable_deliberation = getattr(self.protocol, "enable_judge_deliberation", False)
        if enable_deliberation:
            await self._handle_judge_deliberation(ctx)
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

    async def _handle_judge_deliberation(self, ctx: "DebateContext") -> None:
        """Handle judge consensus with deliberation (Agent-as-a-Judge).

        Multiple judges deliberate on proposals before rendering verdict.
        This reduces individual biases by exposing judges to diverse perspectives.
        """
        from aragora.debate.judge_selector import (
            JudgingStrategy,
            JudgeVote,
            create_judge_panel,
        )

        result = ctx.result
        proposals = ctx.proposals
        task = ctx.env.task if ctx.env else ""

        deliberation_rounds = getattr(self.protocol, "judge_deliberation_rounds", 2)

        logger.info(
            f"judge_deliberation_start proposals={len(proposals)} rounds={deliberation_rounds}"
        )

        # Get judge candidates (use 3 judges for deliberation)
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

        if not judge_candidates or len(judge_candidates) < 2:
            # Not enough judges for deliberation, fall back to single judge
            logger.warning("judge_deliberation_insufficient_judges falling_back_to_single")
            # Continue with regular judge consensus (without deliberation)
            judge = judge_candidates[0] if judge_candidates else None
            if judge:
                await self._run_single_judge_synthesis(ctx, judge)
            else:
                await self._handle_fallback_consensus(ctx, reason="no_judges_available")
            return

        # Create judge panel
        panel = create_judge_panel(
            candidates=judge_candidates,
            participants=ctx.agents,
            domain="debate_deliberation",
            strategy=JudgingStrategy.MAJORITY,
            count=min(3, len(judge_candidates)),
            elo_system=self.elo_system,
            exclude_participants=True,
        )

        if self._notify_spectator:
            self._notify_spectator(
                "judge_deliberation",
                details=f"Starting deliberation with {len(panel.judges)} judges",
                agent="system",
            )

        try:
            # Run deliberation
            deliberation_result = await panel.deliberate_and_vote(
                proposals=proposals,
                task=task,
                context=ctx.context_messages,
                generate_fn=self._generate_with_agent,
                deliberation_rounds=deliberation_rounds,
            )

            logger.info(
                f"judge_deliberation_result approved={deliberation_result.approved} "
                f"confidence={deliberation_result.confidence:.2f} "
                f"approval_ratio={deliberation_result.approval_ratio:.2f}"
            )

            # If judges approve, use the best proposal as synthesis
            if deliberation_result.approved and proposals:
                # Pick proposal with highest approval from judges
                best_proposal_name = max(
                    proposals.keys(),
                    key=lambda k: sum(
                        1 for v in deliberation_result.votes if v.vote == JudgeVote.APPROVE
                    ),
                )
                result.final_answer = proposals[best_proposal_name]
                result.consensus_reached = True
                result.confidence = deliberation_result.confidence
                ctx.winner_agent = best_proposal_name
                result.winner = best_proposal_name

                if self._notify_spectator:
                    self._notify_spectator(
                        "consensus",
                        agent="judge_panel",
                        details=f"Deliberation approved: {best_proposal_name}",
                        metric=deliberation_result.confidence,
                    )
            else:
                # Judges rejected or need more debate
                logger.info("judge_deliberation_rejected continuing to synthesis")
                # Fall back to single judge synthesis
                judge = panel.judges[0] if panel.judges else None
                if judge:
                    await self._run_single_judge_synthesis(ctx, judge)
                else:
                    await self._handle_fallback_consensus(ctx, reason="deliberation_rejected")

        except Exception as e:
            logger.error(f"judge_deliberation_error: {e}")
            await self._handle_fallback_consensus(ctx, reason="deliberation_error")

    async def _run_single_judge_synthesis(self, ctx: "DebateContext", judge) -> None:
        """Run single judge synthesis (helper for deliberation fallback)."""
        result = ctx.result
        proposals = ctx.proposals
        task = ctx.env.task if ctx.env else ""

        judge_prompt = (
            self._build_judge_prompt(proposals, task, result.critiques)
            if self._build_judge_prompt
            else f"Synthesize these proposals: {proposals}"
        )

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

            if self._notify_spectator:
                self._notify_spectator(
                    "consensus",
                    agent=judge.name,
                    details=f"Judge synthesis ({len(synthesis)} chars)",
                    metric=0.8,
                )

        except Exception as e:
            logger.error(f"single_judge_synthesis_error judge={judge.name}: {e}")
            await self._handle_fallback_consensus(ctx, reason="synthesis_error")

    async def _handle_byzantine_consensus(self, ctx: "DebateContext") -> None:
        """Handle 'byzantine' consensus mode - PBFT-style fault-tolerant consensus.

        Uses Byzantine Fault-Tolerant consensus protocol adapted from claude-flow.
        Tolerates up to f faulty (adversarial/hallucinating) agents where n >= 3f+1.

        PBFT Phases:
        1. PRE_PREPARE: Leader proposes a synthesis
        2. PREPARE: Agents validate and signal readiness
        3. COMMIT: Agents commit if 2f+1 prepare messages received
        """
        from aragora.debate.byzantine import (
            ByzantineConsensus,
            ByzantineConsensusConfig,
        )

        result = ctx.result
        proposals = ctx.proposals
        agents = ctx.agents

        if len(agents) < 4:
            logger.warning(
                f"Byzantine consensus requires at least 4 agents, got {len(agents)}. "
                "Falling back to majority voting."
            )
            await self._handle_majority_consensus(ctx)
            return

        # Build configuration from protocol settings
        config = ByzantineConsensusConfig(
            max_faulty_fraction=getattr(self.protocol, "byzantine_fault_tolerance", 0.33),
            phase_timeout_seconds=getattr(self.protocol, "byzantine_phase_timeout", 30.0),
            max_view_changes=getattr(self.protocol, "byzantine_max_view_changes", 3),
            min_agents=4,
        )

        # Create Byzantine consensus protocol
        protocol = ByzantineConsensus(agents=agents, config=config)

        # Build proposal from best proposal or synthesis
        if proposals:
            # Use the first proposal as the base for consensus
            # In a full implementation, we might synthesize or select the best
            proposal_agent = list(proposals.keys())[0]
            proposal_text = proposals[proposal_agent]
        else:
            logger.warning("No proposals available for Byzantine consensus")
            await self._handle_fallback_consensus(ctx, reason="no_proposals")
            return

        task = ctx.env.task if ctx.env else ""

        logger.info(
            f"byzantine_consensus_start agents={len(agents)} "
            f"quorum={protocol.quorum_size} f={protocol.f}"
        )

        if self._notify_spectator:
            self._notify_spectator(
                "byzantine_consensus",
                details=f"Starting PBFT with {len(agents)} agents (f={protocol.f})",
            )

        try:
            # Run Byzantine consensus
            byz_result = await protocol.propose(proposal_text, task=task)

            if byz_result.success:
                result.final_answer = byz_result.value or proposal_text
                result.consensus_reached = True
                result.confidence = byz_result.confidence
                result.consensus_strength = "byzantine"

                # Store Byzantine-specific metadata in formal_verification field
                # (reusing this Optional[dict[str, Any]] field for consensus metadata)
                if result.formal_verification is None:
                    result.formal_verification = {}
                result.formal_verification["byzantine_consensus"] = {
                    "view": byz_result.view,
                    "sequence": byz_result.sequence,
                    "commit_count": byz_result.commit_count,
                    "total_agents": byz_result.total_agents,
                    "agreement_ratio": byz_result.agreement_ratio,
                    "duration_seconds": byz_result.duration_seconds,
                }

                logger.info(
                    f"byzantine_consensus_success view={byz_result.view} "
                    f"commits={byz_result.commit_count}/{byz_result.total_agents} "
                    f"confidence={byz_result.confidence:.2f}"
                )

                if self._notify_spectator:
                    self._notify_spectator(
                        "consensus",
                        details=f"Byzantine consensus reached ({byz_result.commit_count}/{byz_result.total_agents} commits)",
                        metric=byz_result.confidence,
                    )
            else:
                logger.warning(f"byzantine_consensus_failed reason={byz_result.failure_reason}")
                # Fall back to majority voting
                await self._handle_majority_consensus(ctx)

        except Exception as e:
            logger.error(f"byzantine_consensus_error: {e}", exc_info=True)
            # Fall back to majority voting
            await self._handle_majority_consensus(ctx)

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

    def _compute_vote_weights(
        self,
        ctx: "DebateContext",
        votes: Optional[list["Vote"]] = None,
    ) -> dict[str, float]:
        """Pre-compute vote weights for all agents.

        Args:
            ctx: Debate context with agents and proposals
            votes: Optional list of votes for bias mitigation

        Returns:
            Dict mapping agent names to their weights
        """
        from aragora.debate.phases.weight_calculator import WeightCalculatorConfig

        # Get bias mitigation config from protocol
        enable_self_vote = getattr(self.protocol, "enable_self_vote_mitigation", False)
        enable_verbosity = getattr(self.protocol, "enable_verbosity_normalization", False)

        config = WeightCalculatorConfig(
            # Agent-as-a-Judge bias mitigation
            enable_self_vote_mitigation=enable_self_vote,
            self_vote_mode=getattr(self.protocol, "self_vote_mode", "downweight"),
            self_vote_downweight=getattr(self.protocol, "self_vote_downweight", 0.5),
            enable_verbosity_normalization=enable_verbosity,
            verbosity_target_length=getattr(self.protocol, "verbosity_target_length", 1000),
            verbosity_penalty_threshold=getattr(self.protocol, "verbosity_penalty_threshold", 3.0),
            verbosity_max_penalty=getattr(self.protocol, "verbosity_max_penalty", 0.3),
        )

        # Get domain from context for domain-specific ELO weighting
        domain = getattr(ctx, "domain", None) or "general"

        calculator = WeightCalculator(
            memory=self.memory,
            elo_system=self.elo_system,
            flip_detector=self.flip_detector,
            agent_weights=self.agent_weights,
            calibration_tracker=self.calibration_tracker,
            get_calibration_weight=self._get_calibration_weight,
            config=config,
            domain=domain,
        )

        # Use bias-aware computation if votes and proposals available
        if votes and ctx.proposals and (enable_self_vote or enable_verbosity):
            return calculator.compute_weights_with_context(ctx.agents, votes, ctx.proposals)

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
    ) -> tuple[dict[str, float], float]:
        """Count weighted votes."""
        vote_counts: dict[str, float] = {}
        total_weighted = 0.0

        for v in votes:
            if not isinstance(v, Exception):
                canonical = choice_mapping.get(v.choice, v.choice)
                weight = vote_weight_cache.get(v.agent, 1.0)
                vote_counts[canonical] = vote_counts.get(canonical, 0.0) + weight
                total_weighted += weight

        return vote_counts, total_weighted

    def _add_user_votes(
        self,
        vote_counts: dict[str, float],
        total_weighted: float,
        choice_mapping: dict[str, str],
    ) -> tuple[dict[str, float], float]:
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
                vote_counts[canonical] = vote_counts.get(canonical, 0.0) + final_weight
                total_weighted += final_weight

                logger.debug(
                    f"user_vote user={user_vote.get('user_id', 'anonymous')} "
                    f"choice={choice} intensity={intensity} weight={final_weight:.2f}"
                )

        return vote_counts, total_weighted

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

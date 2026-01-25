"""
Debate rounds phase for debate orchestration.

This module extracts the debate round loop (Phase 2) from the
Arena._run_inner() method, handling:
- Role assignment updates per round
- Stance rotation for asymmetric debates
- Critique phase (parallel generation)
- Revision phase (parallel generation)
- Convergence detection
- Termination checks (judge-based, early stopping)
- RLM "ready signal" pattern for agent self-termination
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.config import AGENT_TIMEOUT_SECONDS, MAX_CONCURRENT_CRITIQUES, MAX_CONCURRENT_REVISIONS
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.debate.performance_monitor import get_debate_monitor
from aragora.debate.phases.convergence_tracker import (
    DebateConvergenceTracker,
)
from aragora.server.stream.arena_hooks import streaming_task_context

# Timeout for async callbacks that can hang (evidence refresh, judge termination, etc.)
DEFAULT_CALLBACK_TIMEOUT = 30.0

# Base timeout for the entire revision phase gather (prevents indefinite stalls)
# Actual timeout is calculated dynamically based on agent count
REVISION_PHASE_BASE_TIMEOUT = 120.0


def _calculate_phase_timeout(num_agents: int, agent_timeout: float) -> float:
    """Calculate dynamic phase timeout based on agent count.

    Ensures phase timeout exceeds (agents / max_concurrent) * agent_timeout.
    This prevents the phase from timing out before all agents can complete.

    Args:
        num_agents: Number of agents in the phase
        agent_timeout: Per-agent timeout in seconds

    Returns:
        Phase timeout in seconds
    """
    # With bounded concurrency, worst case is sequential execution
    # Add 60s buffer for gather overhead and safety margin
    calculated = (num_agents / MAX_CONCURRENT_REVISIONS) * agent_timeout + 60.0
    return max(calculated, REVISION_PHASE_BASE_TIMEOUT)


async def _with_callback_timeout(coro, timeout: float = DEFAULT_CALLBACK_TIMEOUT, default=None):
    """Execute coroutine with timeout, returning default on timeout.

    This prevents debates from stalling indefinitely when callbacks
    like evidence refresh or judge termination hang.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Callback timed out after {timeout}s, using default: {default}")
        return default


def _record_adaptive_round(direction: str) -> None:
    """Record adaptive round change metric with lazy import."""
    try:
        from aragora.observability.metrics import record_adaptive_round_change

        record_adaptive_round_change(direction)
    except ImportError:
        pass


if TYPE_CHECKING:
    from aragora.core import Agent, Critique, Message
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class DebateRoundsPhase:
    """
    Executes the debate rounds phase.

    This class encapsulates the critique -> revision -> convergence loop
    that was previously in Arena._run_inner().

    Usage:
        debate_rounds = DebateRoundsPhase(
            protocol=arena.protocol,
            circuit_breaker=arena.circuit_breaker,
            convergence_detector=arena.convergence_detector,
            hooks=arena.hooks,
        )
        await debate_rounds.execute(ctx)
    """

    def __init__(
        self,
        protocol: Any = None,
        circuit_breaker: Any = None,
        convergence_detector: Any = None,
        recorder: Any = None,
        hooks: Optional[dict] = None,
        trickster: Any = None,  # EvidencePoweredTrickster for hollow consensus detection
        rhetorical_observer: Any = None,  # RhetoricalAnalysisObserver for pattern detection
        event_emitter: Any = None,  # EventEmitter for broadcasting observations
        novelty_tracker: Any = None,  # NoveltyTracker for semantic novelty detection
        # Callbacks
        update_role_assignments: Optional[Callable] = None,
        assign_stances: Optional[Callable] = None,
        select_critics_for_proposal: Optional[Callable] = None,
        critique_with_agent: Optional[Callable] = None,
        build_revision_prompt: Optional[Callable] = None,
        generate_with_agent: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        notify_spectator: Optional[Callable] = None,
        record_grounded_position: Optional[Callable] = None,
        check_judge_termination: Optional[Callable] = None,
        check_early_stopping: Optional[Callable] = None,
        inject_challenge: Optional[Callable] = None,  # Callback to inject trickster challenges
        refresh_evidence: Optional[Callable] = None,  # Callback to refresh evidence during rounds
        checkpoint_callback: Optional[
            Callable
        ] = None,  # Async callback to save checkpoint after each round
        context_initializer: Any = None,  # ContextInitializer for background task awaiting
        compress_context: Optional[Callable] = None,  # Async callback to compress debate messages
        rlm_compression_round_threshold: int = 3,  # Start compression after this many rounds
        debate_strategy: Any = None,  # DebateStrategy for adaptive round estimation
    ):
        """
        Initialize the debate rounds phase.

        Args:
            protocol: DebateProtocol with rounds, asymmetric settings
            circuit_breaker: CircuitBreaker for agent availability
            convergence_detector: ConvergenceDetector for semantic similarity
            recorder: ReplayRecorder
            hooks: Hook callbacks dict
            trickster: EvidencePoweredTrickster for hollow consensus detection
            rhetorical_observer: RhetoricalAnalysisObserver for pattern detection
            event_emitter: EventEmitter for broadcasting observations
            novelty_tracker: NoveltyTracker for detecting proposal staleness
            update_role_assignments: Callback to update role assignments
            assign_stances: Callback to assign stances for asymmetric debates
            select_critics_for_proposal: Callback to select critics
            critique_with_agent: Async callback for critique generation
            build_revision_prompt: Callback to build revision prompt
            generate_with_agent: Async callback to generate with agent
            with_timeout: Async callback for timeout wrapper
            notify_spectator: Callback for spectator notifications
            record_grounded_position: Callback to record grounded position
            check_judge_termination: Async callback for judge termination
            check_early_stopping: Async callback for early stopping
            inject_challenge: Callback to inject trickster challenge into context
            refresh_evidence: Async callback to refresh evidence based on round claims
            checkpoint_callback: Async callback to save checkpoint after each round
            context_initializer: ContextInitializer for awaiting background research/evidence
            compress_context: Async callback to compress debate messages using RLM
            rlm_compression_round_threshold: Start compression after this many rounds (default 3)
            debate_strategy: Optional DebateStrategy for memory-based round estimation
        """
        self.protocol = protocol
        self.debate_strategy = debate_strategy
        self.circuit_breaker = circuit_breaker
        self.convergence_detector = convergence_detector
        self.recorder = recorder
        self.hooks = hooks or {}
        self.trickster = trickster
        self.rhetorical_observer = rhetorical_observer
        self.event_emitter = event_emitter
        self.novelty_tracker = novelty_tracker

        # Callbacks
        self._update_role_assignments = update_role_assignments
        self._assign_stances = assign_stances
        self._select_critics_for_proposal = select_critics_for_proposal
        self._critique_with_agent = critique_with_agent
        self._build_revision_prompt = build_revision_prompt
        self._generate_with_agent = generate_with_agent
        self._with_timeout = with_timeout
        self._notify_spectator = notify_spectator
        self._record_grounded_position = record_grounded_position
        self._check_judge_termination = check_judge_termination
        self._check_early_stopping = check_early_stopping
        self._inject_challenge = inject_challenge
        self._refresh_evidence = refresh_evidence
        self._checkpoint_callback = checkpoint_callback
        self._context_initializer = context_initializer
        self._compress_context = compress_context
        self._rlm_compression_round_threshold = rlm_compression_round_threshold

        # Internal state
        self._partial_messages: list["Message"] = []
        self._partial_critiques: list["Critique"] = []

        # Convergence tracker handles convergence, novelty, and RLM ready signals
        self._convergence_tracker = DebateConvergenceTracker(
            convergence_detector=convergence_detector,
            novelty_tracker=novelty_tracker,
            trickster=trickster,
            hooks=self.hooks,
            event_emitter=event_emitter,
            notify_spectator=notify_spectator,
            inject_challenge=inject_challenge,
        )

    def _emit_heartbeat(self, phase: str, status: str = "alive") -> None:
        """Emit heartbeat to indicate debate is still running.

        Prevents frontend timeouts during long-running operations.
        """
        if "on_heartbeat" in self.hooks:
            try:
                self.hooks["on_heartbeat"](phase=phase, status=status)
            except Exception as e:
                logger.debug(f"Heartbeat emission failed: {e}")

    def _observe_rhetorical_patterns(
        self,
        agent: str,
        content: str,
        round_num: int,
        loop_id: str = "",
    ) -> None:
        """Observe content for rhetorical patterns and emit events."""
        if not self.rhetorical_observer:
            return

        try:
            observations = self.rhetorical_observer.observe(
                agent=agent,
                content=content,
                round_num=round_num,
            )

            if not observations:
                return

            # Extract pattern names and build observation data
            patterns = [obs.pattern.value for obs in observations]
            observation_data = [o.to_dict() for o in observations]
            # Use first observation's commentary as analysis if available
            analysis = observations[0].audience_commentary if observations else ""

            # Emit events for rhetorical observations via EventEmitter
            # Use emit_sync since this is not an async context
            if self.event_emitter:
                self.event_emitter.emit_sync(
                    event_type="rhetorical_observation",
                    debate_id=loop_id,
                    agent=agent,
                    round=round_num,
                    patterns=patterns,
                    observations=observation_data,
                    analysis=analysis,
                )

            # Also call hook for arena_hooks-based WebSocket broadcast
            if "on_rhetorical_observation" in self.hooks:
                self.hooks["on_rhetorical_observation"](
                    agent=agent,
                    patterns=patterns,
                    round_num=round_num,
                    analysis=analysis,
                )

            # Log for debugging
            for obs in observations:
                logger.debug(
                    f"rhetorical_pattern agent={agent} pattern={obs.pattern.value} "
                    f"confidence={obs.confidence:.2f}"
                )

        except Exception as e:
            logger.debug(f"Rhetorical observation failed: {e}")

    async def execute(self, ctx: "DebateContext") -> None:
        """
        Execute the debate rounds phase.

        Args:
            ctx: The DebateContext with proposals and result
        """

        result = ctx.result
        proposals = ctx.proposals

        # Determine rounds: use strategy if available, otherwise protocol
        rounds = self.protocol.rounds if self.protocol else 1
        if self.debate_strategy and ctx.env:
            try:
                # Use async version if available
                strategy_rec = await self.debate_strategy.estimate_rounds_async(
                    task=ctx.env.task,
                    default_rounds=rounds,
                )
                if strategy_rec.estimated_rounds != rounds:
                    direction = "increase" if strategy_rec.estimated_rounds > rounds else "decrease"
                    logger.info(
                        f"[strategy] Adaptive rounds: {rounds} -> {strategy_rec.estimated_rounds} "
                        f"(confidence={strategy_rec.confidence:.2f}, reason={strategy_rec.reasoning[:50]})"
                    )
                    _record_adaptive_round(direction)
                    rounds = strategy_rec.estimated_rounds
                    # Store strategy recommendation in result metadata
                    if hasattr(result, "metadata") and result.metadata is not None:
                        result.metadata["strategy_recommendation"] = {
                            "estimated_rounds": strategy_rec.estimated_rounds,
                            "confidence": strategy_rec.confidence,
                            "reasoning": strategy_rec.reasoning,
                            "relevant_memories": strategy_rec.relevant_memories[:3],
                        }
            except Exception as e:
                logger.debug(f"[strategy] Round estimation failed, using protocol default: {e}")

        # Track novelty for initial proposals (round 0 baseline)
        if self.novelty_tracker and proposals:
            self._convergence_tracker.track_novelty(ctx, round_num=0)

        # Get performance monitor for round tracking
        perf_monitor = get_debate_monitor()

        for round_num in range(1, rounds + 1):
            # Check for cancellation before each round
            if ctx.cancellation_token and ctx.cancellation_token.is_cancelled:
                from aragora.debate.cancellation import DebateCancelled

                raise DebateCancelled(ctx.cancellation_token.reason)

            logger.info(f"round_start round={round_num}")

            # Track round with performance monitor for detailed phase metrics
            with perf_monitor.track_round(ctx.debate_id, round_num):
                should_continue = await self._execute_round(ctx, perf_monitor, round_num, rounds)
                if not should_continue:
                    logger.info(f"early_exit_convergence round={round_num}")
                    break

    async def _execute_round(
        self,
        ctx: "DebateContext",
        perf_monitor,
        round_num: int,
        total_rounds: int,
    ) -> bool:
        """Execute a single debate round with performance tracking.

        Returns:
            True if debate should continue to next round, False if converged/should stop.
        """
        result = ctx.result

        # Track round start time for slow debate detection
        _round_start_time = time.time()

        # Trigger PRE_ROUND hook if hook_manager is available
        if ctx.hook_manager:
            try:
                await ctx.hook_manager.trigger("pre_round", ctx=ctx, round_num=round_num)
            except Exception as e:
                logger.debug(f"PRE_ROUND hook failed: {e}")

        # Emit heartbeat at round start
        self._emit_heartbeat(f"round_{round_num}", "starting")

        # Update role assignments
        if self._update_role_assignments:
            self._update_role_assignments(round_num=round_num)

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "round",
                details=f"Starting Round {round_num}",
                agent="system",
            )

        # Rotate stances if asymmetric debate
        if self.protocol:
            if self.protocol.asymmetric_stances and self.protocol.rotate_stances:
                if self._assign_stances:
                    self._assign_stances(round_num)
                    stances_str = ", ".join(f"{a.name}:{a.stance}" for a in ctx.agents)
                    logger.debug(f"stances_rotated stances={stances_str}")

        # Emit round start event
        if "on_round_start" in self.hooks:
            self.hooks["on_round_start"](round_num)

        # Record round start
        if self.recorder:
            try:
                self.recorder.record_phase_change(f"round_{round_num}_start")
            except Exception as e:
                logger.debug(f"Recorder error for round start: {e}")

        # Await background research/evidence before round 1 critiques
        # This ensures research context is available for critique prompts
        if round_num == 1 and self._context_initializer:
            await self._context_initializer.await_background_context(ctx)

        # Compress context messages using RLM after threshold rounds
        # This keeps context manageable for long debates
        if self._compress_context and round_num >= self._rlm_compression_round_threshold:
            await self._compress_debate_context(ctx, round_num)

        # Round 7 special handling: Final Synthesis
        # Each agent synthesizes the discussion and revises their proposal to final form
        # This skips the normal critique/revision cycle
        if self.protocol and self.protocol.use_structured_phases and round_num == 7:
            round_phase = self.protocol.get_round_phase(round_num)
            if round_phase and "Final Synthesis" in round_phase.name:
                logger.info(f"round_7_final_synthesis agents={len(ctx.proposers)}")
                await self._execute_final_synthesis_round(ctx, round_num)
                result.rounds_used = round_num
                return True  # Skip normal critique/revision, move to Round 8 (continue debate)

        # Get and filter critics
        critics = self._get_critics(ctx)

        # Critique phase with performance tracking
        with perf_monitor.track_phase(ctx.debate_id, "critique"):
            await self._critique_phase(ctx, critics, round_num)

        # Refresh evidence based on claims made in critiques and proposals
        with perf_monitor.track_phase(ctx.debate_id, "evidence_refresh"):
            await self._refresh_evidence_for_round(ctx, round_num)

        # Revision phase with performance tracking
        with perf_monitor.track_phase(ctx.debate_id, "revision"):
            await self._revision_phase(ctx, critics, round_num)

        # Track novelty of revised proposals
        self._convergence_tracker.track_novelty(ctx, round_num)

        result.rounds_used = round_num

        # Create checkpoint after each round
        if self._checkpoint_callback:
            try:
                await self._checkpoint_callback(ctx, round_num)
            except Exception as e:
                logger.debug(f"Checkpoint failed for round {round_num}: {e}")

        # Trigger POST_ROUND hook if hook_manager is available
        if ctx.hook_manager:
            try:
                await ctx.hook_manager.trigger(
                    "post_round",
                    ctx=ctx,
                    round_num=round_num,
                    proposals=ctx.proposals,
                )
            except Exception as e:
                logger.debug(f"POST_ROUND hook failed: {e}")

        # Emit heartbeat before convergence check
        self._emit_heartbeat(f"round_{round_num}", "checking_convergence")

        # Convergence detection
        convergence_result = self._convergence_tracker.check_convergence(ctx, round_num)
        should_break = convergence_result.converged and not convergence_result.blocked_by_trickster

        # Record round duration for slow debate detection
        _round_duration = time.time() - _round_start_time
        _slow_threshold = perf_monitor.slow_round_threshold
        if _round_duration > _slow_threshold:
            logger.warning(
                f"slow_round_detected debate_id={ctx.debate_id} round={round_num} "
                f"duration={_round_duration:.2f}s threshold={_slow_threshold:.2f}s"
            )
            try:
                from aragora.observability.metrics import (
                    record_slow_round,
                    record_round_latency,
                )

                record_slow_round(debate_outcome="in_progress")
                record_round_latency(_round_duration)
            except ImportError:
                pass
        else:
            try:
                from aragora.observability.metrics import record_round_latency

                record_round_latency(_round_duration)
            except ImportError:
                pass

        if should_break:
            return False  # Converged - exit round execution early, stop debate loop

        # Termination checks (only if not last round)
        if round_num < total_rounds:
            if await self._should_terminate(ctx, round_num):
                # Signal early termination by setting a flag
                ctx.result.metadata = ctx.result.metadata or {}  # type: ignore[attr-defined]
                ctx.result.metadata["early_termination"] = True  # type: ignore[attr-defined]
                return False  # Stop debate loop

        return True  # Continue to next round

    def _get_critics(self, ctx: "DebateContext") -> list["Agent"]:
        """Get and filter critics for the round."""
        # Get critics - when all agents are proposers, they all critique each other
        critics = [a for a in ctx.agents if a.role in ("critic", "synthesizer")]
        if not critics:
            critics = list(ctx.agents)

        # Filter through circuit breaker
        if self.circuit_breaker:
            try:
                available = self.circuit_breaker.filter_available_agents(critics)
                if len(available) < len(critics):
                    skipped = [c.name for c in critics if c not in available]
                    logger.info(f"circuit_breaker_skip_critics skipped={skipped}")
                critics = available
            except Exception as e:
                logger.error(f"Circuit breaker filter error for critics: {e}")

        return critics

    async def _critique_phase(
        self,
        ctx: "DebateContext",
        critics: list["Agent"],
        round_num: int,
    ) -> None:
        """Execute critique phase with parallel generation."""
        from aragora.core import Message

        result = ctx.result
        proposals = ctx.proposals

        if not self._critique_with_agent:
            logger.warning("No critique_with_agent callback, skipping critiques")
            return

        async def generate_critique(critic, proposal_agent, proposal):
            """Generate critique and return (critic, proposal_agent, result_or_error)."""
            logger.debug(f"critique_generating critic={critic.name} target={proposal_agent}")
            # Use complexity-scaled timeout from governor
            timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))
            # Use task context to distinguish concurrent streaming from same agent
            task_id = f"{critic.name}:critique:{proposal_agent}"
            try:
                with streaming_task_context(task_id):
                    if self._with_timeout:
                        crit_result = await self._with_timeout(
                            self._critique_with_agent(
                                critic,
                                proposal,
                                ctx.env.task if ctx.env else "",
                                ctx.context_messages,
                                target_agent=proposal_agent,
                            ),
                            critic.name,
                            timeout_seconds=timeout,
                        )
                    else:
                        crit_result = await self._critique_with_agent(
                            critic,
                            proposal,
                            ctx.env.task if ctx.env else "",
                            ctx.context_messages,
                            target_agent=proposal_agent,
                        )
                return (critic, proposal_agent, crit_result)
            except Exception as e:
                return (critic, proposal_agent, e)

        # Create critique tasks based on topology with bounded concurrency
        # Semaphore prevents exhausting API rate limits with too many parallel requests
        critique_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRITIQUES)

        async def generate_critique_bounded(critic, proposal_agent, proposal):
            """Wrap critique generation with semaphore for bounded concurrency."""
            async with critique_semaphore:
                return await generate_critique(critic, proposal_agent, proposal)

        critique_tasks = []
        # Filter out empty/placeholder proposals to avoid wasting critic resources
        valid_proposals = {
            agent: content
            for agent, content in proposals.items()
            if content and "(Agent produced empty output)" not in content
        }
        if len(valid_proposals) < len(proposals):
            skipped = [a for a in proposals if a not in valid_proposals]
            logger.warning(f"critique_skip_empty_proposals skipped={skipped}")

        for proposal_agent, proposal in valid_proposals.items():
            if self._select_critics_for_proposal:
                selected_critics = self._select_critics_for_proposal(proposal_agent, critics)
            else:
                # Default: all critics except self
                selected_critics = [c for c in critics if c.name != proposal_agent]

            for critic in selected_critics:
                critique_tasks.append(
                    asyncio.create_task(generate_critique_bounded(critic, proposal_agent, proposal))
                )

        # Emit heartbeat before critique phase
        self._emit_heartbeat(f"critique_round_{round_num}", "generating_critiques")

        # Stream output as each critique completes
        critique_count = 0
        total_critiques = len(critique_tasks)
        for completed_task in asyncio.as_completed(critique_tasks):
            try:
                critic, proposal_agent, crit_result = await completed_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"task_exception phase=critique error={e}")
                continue

            critique_count += 1
            # Emit heartbeat every 3 critiques to signal progress
            if critique_count % 3 == 0 or critique_count == total_critiques:
                self._emit_heartbeat(
                    f"critique_round_{round_num}",
                    f"completed_{critique_count}_of_{total_critiques}",
                )

            if isinstance(crit_result, Exception):
                logger.error(
                    f"critique_error critic={critic.name} target={proposal_agent} error={crit_result}"
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(critic.name)
            elif crit_result is None:
                # Handle timeout/error case where autonomic_executor returned None
                logger.warning(
                    f"critique_returned_none critic={critic.name} target={proposal_agent}"
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(critic.name)

                # Create placeholder critique so the debate can continue
                from aragora.core import Critique

                placeholder_critique = Critique(  # type: ignore[call-arg]
                    agent=critic.name,
                    target_agent=proposal_agent,
                    issues=["[Critique unavailable - agent timed out or encountered an error]"],
                    suggestions=[],
                    severity=0.0,
                    reasoning="Critique generation failed due to timeout or agent error.",
                )
                result.critiques.append(placeholder_critique)
                self._partial_critiques.append(placeholder_critique)

                # Emit placeholder critique event
                if "on_critique" in self.hooks:
                    self.hooks["on_critique"](
                        agent=critic.name,
                        target=proposal_agent,
                        issues=placeholder_critique.issues,
                        severity=placeholder_critique.severity,
                        round_num=round_num,
                        full_content=placeholder_critique.to_prompt(),
                    )
            else:
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(critic.name)
                result.critiques.append(crit_result)
                self._partial_critiques.append(crit_result)

                logger.debug(
                    f"critique_complete critic={critic.name} target={proposal_agent} "
                    f"issues={len(crit_result.issues)} severity={crit_result.severity:.1f}"
                )

                # Notify spectator
                if self._notify_spectator:
                    self._notify_spectator(
                        "critique",
                        agent=critic.name,
                        details=f"Critiqued {proposal_agent}: {len(crit_result.issues)} issues",
                        metric=crit_result.severity,
                    )

                # Get full critique content
                critique_content = crit_result.to_prompt()

                # Emit critique event (includes full_content for activity feeds)
                # NOTE: Previously also emitted on_message which caused duplicate display.
                # on_critique now includes full_content, so on_message is not needed.
                if "on_critique" in self.hooks:
                    self.hooks["on_critique"](
                        agent=critic.name,
                        target=proposal_agent,
                        issues=crit_result.issues,
                        severity=crit_result.severity,
                        round_num=round_num,
                        full_content=critique_content,
                    )

                # Record critique
                if self.recorder:
                    try:
                        self.recorder.record_turn(critic.name, critique_content, round_num)
                    except Exception as e:
                        logger.debug(f"Recorder error for critique: {e}")

                # Add to context
                msg = Message(
                    role="critic",
                    agent=critic.name,
                    content=critique_content,
                    round=round_num,
                )
                ctx.add_message(msg)
                result.messages.append(msg)
                self._partial_messages.append(msg)

    async def _revision_phase(
        self,
        ctx: "DebateContext",
        critics: list["Agent"],
        round_num: int,
    ) -> None:
        """Execute revision phase with parallel generation."""
        from aragora.core import Message

        result = ctx.result
        proposals = ctx.proposals

        if not self._generate_with_agent or not self._build_revision_prompt:
            logger.warning("Missing callbacks for revision phase")
            return

        # Get all critiques from this round for revision
        # NOTE: Critiques have target_agent set to the actual agent name (e.g., "alice", "bob"),
        # not "proposal". We filter per-agent below in the loop.
        all_critiques = list(result.critiques)

        if not all_critiques:
            return

        # Build revision tasks for all proposers with bounded concurrency
        # Use complexity-scaled timeout from governor
        timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))

        # Semaphore prevents exhausting API rate limits with too many parallel requests
        revision_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REVISIONS)

        async def generate_revision_bounded(agent, revision_prompt):
            """Wrap revision generation with semaphore for bounded concurrency."""
            # Use task context to distinguish concurrent streaming from same agent
            task_id = f"{agent.name}:revision:{round_num}"
            async with revision_semaphore:
                with streaming_task_context(task_id):
                    if self._with_timeout:
                        return await self._with_timeout(
                            self._generate_with_agent(agent, revision_prompt, ctx.context_messages),
                            agent.name,
                            timeout_seconds=timeout,
                        )
                    else:
                        return await self._generate_with_agent(
                            agent, revision_prompt, ctx.context_messages
                        )

        revision_tasks = []
        revision_agents = []
        for agent in ctx.proposers:
            # Filter critiques specifically targeting this agent
            # This ensures each agent only sees critiques directed at their proposal
            agent_critiques = [c for c in all_critiques if c.target_agent == agent.name]

            # Skip revision if no critiques for this agent
            if not agent_critiques:
                logger.debug(f"No critiques targeting {agent.name}, skipping revision")
                continue

            revision_prompt = self._build_revision_prompt(
                agent, proposals.get(agent.name, ""), agent_critiques, round_num
            )
            revision_tasks.append(generate_revision_bounded(agent, revision_prompt))
            revision_agents.append(agent)

        # Calculate dynamic phase timeout based on number of agents
        phase_timeout = _calculate_phase_timeout(len(revision_agents), timeout)

        # Emit heartbeat before revision phase
        self._emit_heartbeat(
            f"revision_round_{round_num}", f"starting_{len(revision_agents)}_agents"
        )

        # Periodic heartbeat task during long-running revisions
        async def heartbeat_during_revisions():
            """Emit heartbeat during revisions to keep connection alive."""
            from aragora.config import HEARTBEAT_INTERVAL_SECONDS

            heartbeat_count = 0
            interval = HEARTBEAT_INTERVAL_SECONDS
            try:
                while True:
                    await asyncio.sleep(interval)
                    heartbeat_count += 1
                    self._emit_heartbeat(
                        f"revision_round_{round_num}",
                        f"in_progress_{heartbeat_count * interval}s",
                    )
            except asyncio.CancelledError:
                logger.debug("Heartbeat task cancelled during revision round %d", round_num)

        # Execute all revisions with bounded concurrency and phase-level timeout
        heartbeat_task = asyncio.create_task(heartbeat_during_revisions())
        try:
            revision_results = await asyncio.wait_for(
                asyncio.gather(*revision_tasks, return_exceptions=True),
                timeout=phase_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"revision_phase_timeout: phase exceeded {phase_timeout:.0f}s limit, "
                f"agents={[a.name for a in revision_agents]}"
            )
            # Return timeout errors for all pending agents
            revision_results = [asyncio.TimeoutError()] * len(revision_tasks)
        finally:
            # Cancel the heartbeat task now that revisions are done
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                logger.debug("Heartbeat task cleanup completed for round %d", round_num)

        # Process results
        revision_count = 0
        total_revisions = len(revision_agents)
        for agent, revised in zip(revision_agents, revision_results):
            revision_count += 1
            # Emit heartbeat for each completed revision
            self._emit_heartbeat(
                f"revision_round_{round_num}",
                f"processed_{revision_count}_of_{total_revisions}",
            )
            if isinstance(revised, BaseException):
                logger.error(f"revision_error agent={agent.name} error={revised}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(agent.name)
                continue

            # At this point, revised is confirmed to be str
            revised_str: str = revised
            if self.circuit_breaker:
                self.circuit_breaker.record_success(agent.name)

            proposals[agent.name] = revised_str
            logger.debug(f"revision_complete agent={agent.name} length={len(revised_str)}")

            # Notify spectator
            if self._notify_spectator:
                self._notify_spectator(
                    "propose",
                    agent=agent.name,
                    details=f"Revised proposal ({len(revised_str)} chars)",
                    metric=len(revised_str),
                )

            # Create message
            msg = Message(
                role="proposer",
                agent=agent.name,
                content=revised_str,
                round=round_num,
            )
            ctx.add_message(msg)
            result.messages.append(msg)
            self._partial_messages.append(msg)

            # Emit message event
            if "on_message" in self.hooks:
                self.hooks["on_message"](
                    agent=agent.name,
                    content=revised_str,
                    role="proposer",
                    round_num=round_num,
                )

            # Record revision
            if self.recorder:
                try:
                    self.recorder.record_turn(agent.name, revised_str, round_num)
                except Exception as e:
                    logger.debug(f"Recorder error for revision: {e}")

            # Record position for grounded personas
            if self._record_grounded_position:
                debate_id = (
                    result.id if hasattr(result, "id") else (ctx.env.task[:50] if ctx.env else "")
                )
                self._record_grounded_position(agent.name, revised_str, debate_id, round_num, 0.75)

            # Observe rhetorical patterns for audience engagement
            loop_id = ctx.loop_id if hasattr(ctx, "loop_id") else ""
            self._observe_rhetorical_patterns(agent.name, revised_str, round_num, loop_id)

    async def _should_terminate(self, ctx: "DebateContext", round_num: int) -> bool:
        """Check if debate should terminate early.

        Uses timeout protection on callbacks to prevent indefinite hangs.
        Includes RLM ready signal quorum check for agent self-termination.
        """
        # RLM ready signal check (agents self-signal readiness)
        # This is the most responsive - agents explicitly say "I'm done"
        if self._convergence_tracker.check_rlm_ready_quorum(ctx, round_num):
            logger.info(f"debate_terminate_rlm_ready round={round_num}")
            return True

        # Judge-based termination (with timeout protection)
        if self._check_judge_termination:
            result = await _with_callback_timeout(
                self._check_judge_termination(round_num, ctx.proposals, ctx.context_messages),
                timeout=DEFAULT_CALLBACK_TIMEOUT,
                default=(True, "Judge check timed out"),  # Continue on timeout
            )
            should_continue, reason = result
            if not should_continue:
                return True

        # Early stopping (agent votes) with timeout protection
        if self._check_early_stopping:
            should_continue = await _with_callback_timeout(
                self._check_early_stopping(round_num, ctx.proposals, ctx.context_messages),
                timeout=DEFAULT_CALLBACK_TIMEOUT,
                default=True,  # Continue on timeout
            )
            if not should_continue:
                return True

        return False

    async def _refresh_evidence_for_round(self, ctx: "DebateContext", round_num: int) -> None:
        """Refresh evidence based on claims made in the current round.

        Extracts factual claims from proposals and critiques, then
        searches for new evidence to support or refute those claims.
        The fresh evidence is injected into the context for the revision phase.

        Args:
            ctx: The DebateContext with proposals and critiques
            round_num: Current round number
        """
        if not self._refresh_evidence:
            return

        # Only refresh evidence every other round to avoid API overload
        if round_num % 2 == 0:
            return

        try:
            # Collect text from proposals and recent critiques
            texts_to_analyze = []

            # Add proposal content
            for agent_name, proposal in ctx.proposals.items():
                if proposal:
                    texts_to_analyze.append(proposal[:2000])  # Limit per proposal

            # Add recent critique content
            for critique in self._partial_critiques[-5:]:  # Last 5 critiques
                critique_text = (
                    critique.to_prompt() if hasattr(critique, "to_prompt") else str(critique)
                )
                texts_to_analyze.append(critique_text[:1000])

            if not texts_to_analyze:
                return

            combined_text = "\n".join(texts_to_analyze)

            # Call the refresh callback with timeout protection
            refreshed = await _with_callback_timeout(
                self._refresh_evidence(combined_text, ctx, round_num),
                timeout=DEFAULT_CALLBACK_TIMEOUT,
                default=0,  # Return 0 snippets on timeout
            )

            if refreshed:
                logger.info(f"evidence_refreshed round={round_num} new_snippets={refreshed}")

                # Notify spectator
                if self._notify_spectator:
                    self._notify_spectator(
                        "evidence",
                        details=f"Refreshed evidence: {refreshed} new sources",
                        metric=refreshed,
                        agent="system",
                    )

                # Emit evidence refresh event
                if "on_evidence_refresh" in self.hooks:
                    self.hooks["on_evidence_refresh"](
                        round_num=round_num,
                        new_snippets=refreshed,
                    )

        except Exception as e:
            logger.warning(f"Evidence refresh failed for round {round_num}: {e}")

    def get_partial_messages(self) -> list["Message"]:
        """Get partial messages for timeout recovery."""
        return self._partial_messages

    def get_partial_critiques(self) -> list["Critique"]:
        """Get partial critiques for timeout recovery."""
        return self._partial_critiques

    async def _compress_debate_context(
        self,
        ctx: "DebateContext",
        round_num: int,
    ) -> None:
        """Compress debate context using RLM cognitive load limiter.

        Called at the start of each round after the threshold to keep
        context manageable for long debates. Old messages are summarized
        while recent messages are kept at full detail.

        Args:
            ctx: The DebateContext with messages to compress
            round_num: Current round number
        """
        if not self._compress_context:
            return

        # Only compress if there are enough messages to warrant it
        if len(ctx.context_messages) < 10:
            return

        try:
            # Emit heartbeat to signal compression is happening
            self._emit_heartbeat(f"round_{round_num}", "compressing_context")

            # Call Arena's compress_debate_messages method
            compressed_msgs, compressed_crits = await _with_callback_timeout(
                self._compress_context(
                    messages=ctx.context_messages,
                    critiques=self._partial_critiques,
                ),
                timeout=DEFAULT_CALLBACK_TIMEOUT,
                default=(ctx.context_messages, self._partial_critiques),
            )

            # Update context with compressed messages
            if compressed_msgs is not ctx.context_messages:
                original_count = len(ctx.context_messages)
                ctx.context_messages = list(compressed_msgs)
                logger.info(
                    f"[rlm] Compressed context: {original_count} → {len(ctx.context_messages)} messages"
                )

                # Notify spectator about compression
                if self._notify_spectator:
                    self._notify_spectator(
                        "context_compression",
                        details=f"Compressed {original_count} → {len(ctx.context_messages)} messages",
                        agent="system",
                    )

                # Emit hook for WebSocket clients
                if "on_context_compression" in self.hooks:
                    self.hooks["on_context_compression"](
                        round_num=round_num,
                        original_count=original_count,
                        compressed_count=len(ctx.context_messages),
                    )

        except Exception as e:
            logger.warning(f"[rlm] Context compression failed: {e}")
            # Continue without compression - don't break the debate

    async def _execute_final_synthesis_round(
        self,
        ctx: "DebateContext",
        round_num: int,
    ) -> None:
        """Execute Round 7: Final Synthesis.

        Each agent synthesizes the discussion and revises their proposal to final form.
        This is different from normal rounds - agents write their polished final position
        incorporating insights from the entire debate.

        Args:
            ctx: The DebateContext with proposals and critiques
            round_num: Round number (should be 7)
        """
        from aragora.core import Message

        result = ctx.result
        proposals = ctx.proposals

        # Get all proposers
        proposers = ctx.proposers if ctx.proposers else ctx.agents

        # Filter through circuit breaker
        if self.circuit_breaker:
            try:
                available = self.circuit_breaker.filter_available_agents(list(proposers))
                if len(available) < len(proposers):
                    skipped = [p.name for p in proposers if p not in available]
                    logger.info(f"circuit_breaker_skip_synthesis skipped={skipped}")
                proposers = available
            except Exception as e:
                logger.error(f"Circuit breaker filter error for synthesis: {e}")

        # Each proposer writes their final synthesis
        for agent in proposers:
            try:
                prompt = self._build_final_synthesis_prompt(
                    agent=agent,
                    current_proposal=proposals.get(agent.name, ""),
                    all_proposals=proposals,
                    critiques=ctx.critiques,  # type: ignore[attr-defined]
                    round_num=round_num,
                )

                # Generate final synthesis with timeout
                final_proposal = await asyncio.wait_for(
                    self._generate_revision(agent, prompt, ctx.context_messages),  # type: ignore[attr-defined]
                    timeout=self.agent_timeout,  # type: ignore[attr-defined]
                )

                if final_proposal:
                    proposals[agent.name] = final_proposal

                    # Record as message
                    msg = Message(
                        role="final_synthesis",
                        agent=agent.name,
                        content=final_proposal,
                        round=round_num,
                    )
                    ctx.add_message(msg)
                    result.messages.append(msg)
                    self._partial_messages.append(msg)

                    # Emit event
                    if "on_message" in self.hooks:
                        self.hooks["on_message"](
                            agent=agent.name,
                            content=final_proposal,
                            role="final_synthesis",
                            round_num=round_num,
                            full_content=final_proposal,
                        )

                    logger.info(f"final_synthesis_complete agent={agent.name}")

            except asyncio.TimeoutError:
                logger.warning(f"Final synthesis timeout for agent {agent.name}")
            except Exception as e:
                logger.error(f"Final synthesis error for agent {agent.name}: {e}")

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "final_synthesis",
                details="All agents have submitted final syntheses",
                agent="system",
            )

    def _build_final_synthesis_prompt(
        self,
        agent: "Agent",
        current_proposal: str,
        all_proposals: dict,
        critiques: list,
        round_num: int,
    ) -> str:
        """Build prompt for Round 7 final synthesis.

        Args:
            agent: The agent writing the synthesis
            current_proposal: Agent's current proposal
            all_proposals: All proposals from all agents
            critiques: List of critiques from the debate
            round_num: Round number (7)

        Returns:
            Formatted prompt for final synthesis
        """
        # Get other agents' proposals
        other_proposals = "\n\n".join(
            f"**{name}:** {prop[:1200]}..."
            for name, prop in all_proposals.items()
            if name != agent.name and prop
        )

        # Get recent critique summaries
        critique_summary = "\n".join(
            f"- {getattr(c, 'critic', 'Unknown')} on {getattr(c, 'target', 'Unknown')}: {getattr(c, 'summary', str(c)[:200])}"
            for c in critiques[-15:]  # Last 15 critiques
        )

        return f"""## ROUND 7: FINAL SYNTHESIS

You are {agent.name}. This is your FINAL opportunity to revise your proposal.

After 6 rounds of debate, critique, and refinement, you must now present your
polished, definitive position that incorporates the strongest insights from
the entire discussion.

### Your Current Proposal
{current_proposal[:2000] if current_proposal else "(No previous proposal)"}

### Other Agents' Current Positions
{other_proposals if other_proposals else "(No other proposals available)"}

### Key Critiques from the Debate
{critique_summary if critique_summary else "(No critiques recorded)"}

### Your Task
Write your FINAL, POLISHED proposal that:

1. **Incorporates the strongest points** raised by other agents during the debate
2. **Addresses the most compelling critiques** of your position
3. **Presents your clearest, most defensible position** with supporting reasoning
4. **Acknowledges remaining uncertainties** honestly and explicitly
5. **Provides actionable conclusions** where applicable

This is your final word. Make it count. Be thorough but focused.
Write in a clear, confident voice while acknowledging genuine complexity."""

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
"""

import asyncio
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

from aragora.config import AGENT_TIMEOUT_SECONDS, MAX_CONCURRENT_CRITIQUES, MAX_CONCURRENT_REVISIONS
from aragora.debate.complexity_governor import get_complexity_governor

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
        """
        self.protocol = protocol
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

        # Internal state
        self._partial_messages: list["Message"] = []
        self._partial_critiques: list["Critique"] = []
        self._previous_round_responses: dict[str, str] = {}

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

            # Emit events for each observation
            if self.event_emitter:
                from aragora.server.stream.events import StreamEvent, StreamEventType

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.RHETORICAL_OBSERVATION,
                        loop_id=loop_id,
                        data={
                            "agent": agent,
                            "round_num": round_num,
                            "observations": [o.to_dict() for o in observations],
                        },
                    )
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
        from aragora.core import Message

        result = ctx.result
        proposals = ctx.proposals
        rounds = self.protocol.rounds if self.protocol else 1

        # Track novelty for initial proposals (round 0 baseline)
        if self.novelty_tracker and proposals:
            self._track_novelty(ctx, round_num=0)

        for round_num in range(1, rounds + 1):
            logger.info(f"round_start round={round_num}")

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

            # Get and filter critics
            critics = self._get_critics(ctx)

            # Critique phase
            await self._critique_phase(ctx, critics, round_num)

            # Refresh evidence based on claims made in critiques and proposals
            await self._refresh_evidence_for_round(ctx, round_num)

            # Revision phase
            await self._revision_phase(ctx, critics, round_num)

            # Track novelty of revised proposals
            self._track_novelty(ctx, round_num)

            result.rounds_used = round_num

            # Create checkpoint after each round
            if self._checkpoint_callback:
                try:
                    await self._checkpoint_callback(ctx, round_num)
                except Exception as e:
                    logger.debug(f"Checkpoint failed for round {round_num}: {e}")

            # Convergence detection
            should_break = self._check_convergence(ctx, round_num)
            if should_break:
                break

            # Termination checks (only if not last round)
            if round_num < rounds:
                if await self._should_terminate(ctx, round_num):
                    break

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
            try:
                if self._with_timeout:
                    crit_result = await self._with_timeout(
                        self._critique_with_agent(
                            critic, proposal, ctx.env.task if ctx.env else "", ctx.context_messages
                        ),
                        critic.name,
                        timeout_seconds=timeout,
                    )
                else:
                    crit_result = await self._critique_with_agent(
                        critic, proposal, ctx.env.task if ctx.env else "", ctx.context_messages
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
        for proposal_agent, proposal in proposals.items():
            if self._select_critics_for_proposal:
                selected_critics = self._select_critics_for_proposal(proposal_agent, critics)
            else:
                # Default: all critics except self
                selected_critics = [c for c in critics if c.name != proposal_agent]

            for critic in selected_critics:
                critique_tasks.append(
                    asyncio.create_task(generate_critique_bounded(critic, proposal_agent, proposal))
                )

        # Stream output as each critique completes
        for completed_task in asyncio.as_completed(critique_tasks):
            try:
                critic, proposal_agent, crit_result = await completed_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"task_exception phase=critique error={e}")
                continue

            if isinstance(crit_result, Exception):
                logger.error(
                    f"critique_error critic={critic.name} target={proposal_agent} error={crit_result}"
                )
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(critic.name)
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

                # Emit critique event
                if "on_critique" in self.hooks:
                    self.hooks["on_critique"](
                        agent=critic.name,
                        target=proposal_agent,
                        issues=crit_result.issues,
                        severity=crit_result.severity,
                        round_num=round_num,
                        full_content=critique_content,
                    )

                # Emit as message for activity feed
                if "on_message" in self.hooks:
                    self.hooks["on_message"](
                        agent=critic.name,
                        content=critique_content,
                        role="critic",
                        round_num=round_num,
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

        # Get critiques for revision
        agent_critiques = [c for c in result.critiques if c.target_agent == "proposal"]

        if not agent_critiques:
            return

        # Build revision tasks for all proposers with bounded concurrency
        # Use complexity-scaled timeout from governor
        timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))

        # Semaphore prevents exhausting API rate limits with too many parallel requests
        revision_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REVISIONS)

        async def generate_revision_bounded(agent, revision_prompt):
            """Wrap revision generation with semaphore for bounded concurrency."""
            async with revision_semaphore:
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
            revision_prompt = self._build_revision_prompt(
                agent, proposals.get(agent.name, ""), agent_critiques[-len(critics) :]
            )
            revision_tasks.append(generate_revision_bounded(agent, revision_prompt))
            revision_agents.append(agent)

        # Execute all revisions with bounded concurrency
        revision_results = await asyncio.gather(*revision_tasks, return_exceptions=True)

        # Process results
        for agent, revised in zip(revision_agents, revision_results):
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

    def _check_convergence(self, ctx: "DebateContext", round_num: int) -> bool:
        """Check for convergence and return True if should break."""
        if not self.convergence_detector:
            return False

        current_responses = dict(ctx.proposals)

        if not self._previous_round_responses:
            self._previous_round_responses = current_responses
            return False

        convergence = self.convergence_detector.check_convergence(
            current_responses, self._previous_round_responses, round_num
        )

        self._previous_round_responses = current_responses

        if not convergence:
            return False

        result = ctx.result
        result.convergence_status = convergence.status
        result.convergence_similarity = convergence.avg_similarity
        result.per_agent_similarity = convergence.per_agent_similarity

        logger.info(
            f"convergence_check status={convergence.status} "
            f"similarity={convergence.avg_similarity:.0%}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "convergence",
                details=f"{convergence.status}",
                metric=convergence.avg_similarity,
            )

        # Emit convergence event
        if "on_convergence_check" in self.hooks:
            self.hooks["on_convergence_check"](
                status=convergence.status,
                similarity=convergence.avg_similarity,
                per_agent=convergence.per_agent_similarity,
                round_num=round_num,
            )

        # Check for hollow consensus using trickster
        if self.trickster and convergence.avg_similarity > 0.5:
            intervention = self.trickster.check_and_intervene(
                responses=current_responses,
                convergence_similarity=convergence.avg_similarity,
                round_num=round_num,
            )
            if intervention:
                logger.info(
                    f"trickster_intervention round={round_num} "
                    f"type={intervention.intervention_type.value} "
                    f"targets={intervention.target_agents}"
                )
                # Notify spectator about hollow consensus
                if self._notify_spectator:
                    self._notify_spectator(
                        "hollow_consensus",
                        details=f"Evidence quality challenge triggered",
                        metric=intervention.priority,
                        agent="trickster",
                    )
                # Emit trickster event
                if "on_trickster_intervention" in self.hooks:
                    self.hooks["on_trickster_intervention"](
                        intervention_type=intervention.intervention_type.value,
                        targets=intervention.target_agents,
                        challenge=intervention.challenge_text,
                        round_num=round_num,
                    )
                # Inject challenge into context for next round
                if self._inject_challenge:
                    self._inject_challenge(intervention.challenge_text, ctx)
                # Don't declare convergence if hollow - continue debate
                if intervention.priority > 0.5:
                    logger.info(f"hollow_consensus_blocked round={round_num}")
                    return False

        if convergence.converged:
            logger.info(f"debate_converged round={round_num}")
            return True

        return False

    def _track_novelty(self, ctx: "DebateContext", round_num: int) -> None:
        """
        Track novelty of current proposals compared to prior proposals.

        Updates context with novelty scores and triggers trickster intervention
        if proposals are too similar to previous rounds.
        """
        if not self.novelty_tracker:
            return

        current_proposals = dict(ctx.proposals)
        if not current_proposals:
            return

        # Compute novelty against prior proposals
        novelty_result = self.novelty_tracker.compute_novelty(current_proposals, round_num)

        # Update context with novelty scores
        for agent, novelty in novelty_result.per_agent_novelty.items():
            if agent not in ctx.per_agent_novelty:
                ctx.per_agent_novelty[agent] = []
            ctx.per_agent_novelty[agent].append(novelty)

        ctx.avg_novelty = novelty_result.avg_novelty
        ctx.low_novelty_agents = novelty_result.low_novelty_agents

        # Add to history for future comparisons
        self.novelty_tracker.add_to_history(current_proposals)

        # Log novelty status
        logger.info(
            f"novelty_check round={round_num} avg={novelty_result.avg_novelty:.2f} "
            f"min={novelty_result.min_novelty:.2f} low_novelty={novelty_result.low_novelty_agents}"
        )

        # Notify spectator about novelty
        if self._notify_spectator:
            self._notify_spectator(
                "novelty",
                details=f"Avg novelty: {novelty_result.avg_novelty:.0%}",
                metric=novelty_result.avg_novelty,
            )

        # Emit novelty event
        if "on_novelty_check" in self.hooks:
            self.hooks["on_novelty_check"](
                avg_novelty=novelty_result.avg_novelty,
                per_agent=novelty_result.per_agent_novelty,
                low_novelty_agents=novelty_result.low_novelty_agents,
                round_num=round_num,
            )

        # Check for low novelty and trigger trickster intervention
        if novelty_result.has_low_novelty() and self.trickster:
            # Use trickster to generate novelty challenge
            from aragora.debate.trickster import InterventionType

            if hasattr(self.trickster, "create_novelty_challenge"):
                intervention = self.trickster.create_novelty_challenge(
                    low_novelty_agents=novelty_result.low_novelty_agents,
                    novelty_scores=novelty_result.per_agent_novelty,
                    round_num=round_num,
                )
                if intervention:
                    logger.info(
                        f"novelty_challenge round={round_num} "
                        f"targets={intervention.target_agents}"
                    )
                    # Notify spectator
                    if self._notify_spectator:
                        self._notify_spectator(
                            "low_novelty",
                            details="Proposals too similar to prior rounds",
                            metric=novelty_result.min_novelty,
                            agent="trickster",
                        )
                    # Emit trickster event
                    if "on_trickster_intervention" in self.hooks:
                        self.hooks["on_trickster_intervention"](
                            intervention_type=intervention.intervention_type.value,
                            targets=intervention.target_agents,
                            challenge=intervention.challenge_text,
                            round_num=round_num,
                        )
                    # Inject challenge into context for next round
                    if self._inject_challenge:
                        self._inject_challenge(intervention.challenge_text, ctx)

    async def _should_terminate(self, ctx: "DebateContext", round_num: int) -> bool:
        """Check if debate should terminate early."""
        # Judge-based termination
        if self._check_judge_termination:
            should_continue, reason = await self._check_judge_termination(
                round_num, ctx.proposals, ctx.context_messages
            )
            if not should_continue:
                return True

        # Early stopping (agent votes)
        if self._check_early_stopping:
            should_continue = await self._check_early_stopping(
                round_num, ctx.proposals, ctx.context_messages
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

            # Call the refresh callback
            refreshed = await self._refresh_evidence(combined_text, ctx, round_num)

            if refreshed:
                logger.info(f"evidence_refreshed round={round_num} " f"new_snippets={refreshed}")

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

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

from aragora.config import AGENT_TIMEOUT_SECONDS

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
    ):
        """
        Initialize the debate rounds phase.

        Args:
            protocol: DebateProtocol with rounds, asymmetric settings
            circuit_breaker: CircuitBreaker for agent availability
            convergence_detector: ConvergenceDetector for semantic similarity
            recorder: ReplayRecorder
            hooks: Hook callbacks dict
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
        """
        self.protocol = protocol
        self.circuit_breaker = circuit_breaker
        self.convergence_detector = convergence_detector
        self.recorder = recorder
        self.hooks = hooks or {}

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

        # Internal state
        self._partial_messages: list["Message"] = []
        self._partial_critiques: list["Critique"] = []
        self._previous_round_responses: dict[str, str] = {}

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
                        stances_str = ", ".join(
                            f"{a.name}:{a.stance}" for a in ctx.agents
                        )
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

            # Revision phase
            await self._revision_phase(ctx, critics, round_num)

            result.rounds_used = round_num

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
            try:
                if self._with_timeout:
                    crit_result = await self._with_timeout(
                        self._critique_with_agent(
                            critic, proposal, ctx.env.task if ctx.env else "", ctx.context_messages
                        ),
                        critic.name,
                        timeout_seconds=float(AGENT_TIMEOUT_SECONDS),
                    )
                else:
                    crit_result = await self._critique_with_agent(
                        critic, proposal, ctx.env.task if ctx.env else "", ctx.context_messages
                    )
                return (critic, proposal_agent, crit_result)
            except Exception as e:
                return (critic, proposal_agent, e)

        # Create critique tasks based on topology
        critique_tasks = []
        for proposal_agent, proposal in proposals.items():
            if self._select_critics_for_proposal:
                selected_critics = self._select_critics_for_proposal(proposal_agent, critics)
            else:
                # Default: all critics except self
                selected_critics = [c for c in critics if c.name != proposal_agent]

            for critic in selected_critics:
                critique_tasks.append(
                    asyncio.create_task(generate_critique(critic, proposal_agent, proposal))
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
        agent_critiques = [
            c for c in result.critiques if c.target_agent == "proposal"
        ]

        if not agent_critiques:
            return

        # Build revision tasks for all proposers
        revision_tasks = []
        revision_agents = []
        for agent in ctx.proposers:
            revision_prompt = self._build_revision_prompt(
                agent, proposals.get(agent.name, ""), agent_critiques[-len(critics):]
            )

            if self._with_timeout:
                task = self._with_timeout(
                    self._generate_with_agent(agent, revision_prompt, ctx.context_messages),
                    agent.name,
                    timeout_seconds=90.0,
                )
            else:
                task = self._generate_with_agent(agent, revision_prompt, ctx.context_messages)

            revision_tasks.append(task)
            revision_agents.append(agent)

        # Execute all revisions in parallel
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
                debate_id = result.id if hasattr(result, 'id') else (ctx.env.task[:50] if ctx.env else "")
                self._record_grounded_position(agent.name, revised_str, debate_id, round_num, 0.75)

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

        if convergence.converged:
            logger.info(f"debate_converged round={round_num}")
            return True

        return False

    async def _should_terminate(
        self, ctx: "DebateContext", round_num: int
    ) -> bool:
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

    def get_partial_messages(self) -> list["Message"]:
        """Get partial messages for timeout recovery."""
        return self._partial_messages

    def get_partial_critiques(self) -> list["Critique"]:
        """Get partial critiques for timeout recovery."""
        return self._partial_critiques

"""
Proposal phase for debate orchestration.

This module extracts the initial proposal generation logic (Phase 1) from the
Arena._run_inner() method, handling:
- Proposer selection and circuit breaker filtering
- Parallel proposal generation with streaming
- Position tracking for grounded personas
- Message and event emission
- Citation need extraction
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.debate.types import AgentType, DebateContextType

logger = logging.getLogger(__name__)


class ProposalPhase:
    """
    Generates initial proposals from proposer agents.

    This class encapsulates the parallel proposal generation logic that was
    previously in Arena._run_inner() for round 0.

    Usage:
        proposal_phase = ProposalPhase(
            circuit_breaker=arena.circuit_breaker,
            position_tracker=arena.position_tracker,
            recorder=arena.recorder,
            hooks=arena.hooks,
        )
        await proposal_phase.execute(ctx)
    """

    def __init__(
        self,
        circuit_breaker: Any = None,
        position_tracker: Any = None,
        position_ledger: Any = None,
        recorder: Any = None,
        hooks: Optional[dict] = None,
        # Callbacks for orchestrator methods
        build_proposal_prompt: Optional[Callable] = None,
        generate_with_agent: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        notify_spectator: Optional[Callable] = None,
        update_role_assignments: Optional[Callable] = None,
        record_grounded_position: Optional[Callable] = None,
        extract_citation_needs: Optional[Callable] = None,
    ):
        """
        Initialize the proposal phase.

        Args:
            circuit_breaker: CircuitBreaker for agent availability
            position_tracker: Optional PositionTracker for personas
            position_ledger: Optional PositionLedger for grounded personas
            recorder: Optional ReplayRecorder
            hooks: Optional hooks dict for events
            build_proposal_prompt: Callback to build proposal prompt
            generate_with_agent: Async callback to generate with agent
            with_timeout: Async callback for timeout wrapper
            notify_spectator: Callback for spectator notifications
            update_role_assignments: Callback to update role assignments
            record_grounded_position: Callback to record grounded position
            extract_citation_needs: Callback to extract citation needs
        """
        self.circuit_breaker = circuit_breaker
        self.position_tracker = position_tracker
        self.position_ledger = position_ledger
        self.recorder = recorder
        self.hooks = hooks or {}

        # Callbacks
        self._build_proposal_prompt = build_proposal_prompt
        self._generate_with_agent = generate_with_agent
        self._with_timeout = with_timeout
        self._notify_spectator = notify_spectator
        self._update_role_assignments = update_role_assignments
        self._record_grounded_position = record_grounded_position
        self._extract_citation_needs = extract_citation_needs

    async def execute(self, ctx: "DebateContextType") -> None:
        """
        Execute the proposal phase.

        Args:
            ctx: The DebateContextType to update with proposals
        """

        # 1. Update role assignments for round 0
        if self._update_role_assignments:
            self._update_role_assignments(round_num=0)

        # 2. Log debate start
        agent_names = [a.name for a in ctx.agents]
        logger.info(f"debate_start task={ctx.env.task[:80]} agents={agent_names}")

        # 3. Emit debate start event
        self._emit_debate_start(ctx)

        # 4. Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "debate_start",
                details=f"Task: {ctx.env.task[:50]}...",
                agent="system",
            )

        # 5. Filter proposers through circuit breaker
        logger.info("round_start round=0 phase=proposals")
        available_proposers = self._filter_proposers(ctx)

        # 6. Generate proposals in parallel
        await self._generate_proposals_parallel(ctx, available_proposers)

        # 7. Extract citation needs
        if self._extract_citation_needs:
            self._extract_citation_needs(ctx.proposals)

    def _emit_debate_start(self, ctx: "DebateContextType") -> None:
        """Emit debate start hook event."""
        if "on_debate_start" not in self.hooks:
            return

        self.hooks["on_debate_start"](
            ctx.env.task,
            [a.name for a in ctx.agents],
        )

    def _filter_proposers(self, ctx: "DebateContextType") -> list["AgentType"]:
        """Filter proposers through circuit breaker."""
        proposers = ctx.proposers

        if not self.circuit_breaker:
            return proposers

        try:
            available = self.circuit_breaker.filter_available_agents(proposers)
        except Exception as e:
            logger.error(f"Circuit breaker filter error: {e}")
            return proposers  # Fall back to all proposers

        if len(available) < len(proposers):
            skipped = [a.name for a in proposers if a not in available]
            logger.info(f"circuit_breaker_skip agents={skipped}")

        return available

    async def _generate_proposals_parallel(
        self, ctx: "DebateContextType", proposers: list["AgentType"]
    ) -> None:
        """Generate proposals in parallel with streaming output."""

        if not proposers:
            logger.warning("No proposers available for proposal phase")
            return

        # Create tasks for parallel execution
        tasks = []
        for agent in proposers:
            task = asyncio.create_task(self._generate_single_proposal(ctx, agent))
            tasks.append(task)

        # Stream output as each agent finishes
        for completed_task in asyncio.as_completed(tasks):
            try:
                agent, result_or_error = await completed_task
            except asyncio.CancelledError:
                raise  # Propagate cancellation
            except Exception as e:
                logger.error(f"task_exception phase=proposal error={e}")
                continue

            # Process the result
            self._process_proposal_result(ctx, agent, result_or_error)

    async def _generate_single_proposal(
        self, ctx: "DebateContextType", agent: "AgentType"
    ) -> tuple["AgentType", Any]:
        """Generate a single proposal from an agent."""
        if not self._build_proposal_prompt or not self._generate_with_agent:
            return (agent, Exception("Missing callbacks"))

        prompt = self._build_proposal_prompt(agent)
        logger.debug(f"agent_generating agent={agent.name} phase=proposal")

        try:
            # Use complexity-scaled timeout from governor
            timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))
            if self._with_timeout:
                result = await self._with_timeout(
                    self._generate_with_agent(agent, prompt, ctx.context_messages),
                    agent.name,
                    timeout_seconds=timeout,
                )
            else:
                result = await self._generate_with_agent(agent, prompt, ctx.context_messages)
            return (agent, result)
        except Exception as e:
            return (agent, e)

    def _process_proposal_result(
        self, ctx: "DebateContextType", agent: "AgentType", result_or_error: Any
    ) -> None:
        """Process a proposal result from an agent."""
        from aragora.core import Message

        is_error = isinstance(result_or_error, Exception)

        if is_error:
            logger.error(f"agent_error agent={agent.name} phase=proposal error={result_or_error}")
            ctx.proposals[agent.name] = f"[Error generating proposal: {result_or_error}]"
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(agent.name)
        else:
            ctx.proposals[agent.name] = result_or_error
            logger.info(
                f"agent_complete agent={agent.name} phase=proposal chars={len(result_or_error)}"
            )
            if self.circuit_breaker:
                self.circuit_breaker.record_success(agent.name)

            # Notify spectator
            if self._notify_spectator:
                self._notify_spectator(
                    "propose",
                    agent=agent.name,
                    details=f"Initial proposal ({len(result_or_error)} chars)",
                    metric=len(result_or_error),
                )

            # Record positions
            self._record_positions(ctx, agent, result_or_error)

        # Create and add message
        msg = Message(
            role="proposer",
            agent=agent.name,
            content=ctx.proposals[agent.name],
            round=0,
        )
        ctx.add_message(msg)

        # Emit message event
        self._emit_message_event(agent, ctx.proposals[agent.name])

        # Record to replay recorder
        if self.recorder and not is_error:
            try:
                self.recorder.record_turn(agent.name, ctx.proposals[agent.name], 0)
            except Exception as e:
                logger.debug(f"Recorder error for proposal: {e}")

    def _record_positions(self, ctx: "DebateContextType", agent: "AgentType", proposal: str) -> None:
        """Record positions for truth-grounded personas."""
        debate_id = ctx.debate_id or ctx.env.task[:50]

        # Legacy position tracker
        if self.position_tracker:
            try:
                self.position_tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent.name,
                    position_type="proposal",
                    position_text=proposal[:1000],
                    round_num=0,
                    confidence=0.7,
                )
            except Exception as e:
                logger.debug(f"Position tracking error: {e}")

        # New grounded position system
        if self._record_grounded_position:
            self._record_grounded_position(agent.name, proposal, debate_id, 0, 0.7)

    def _emit_message_event(self, agent: "AgentType", content: str) -> None:
        """Emit on_message hook event."""
        if "on_message" not in self.hooks:
            return

        self.hooks["on_message"](
            agent=agent.name,
            content=content,
            role="proposer",
            round_num=0,
        )

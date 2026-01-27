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

__all__ = [
    "ProposalPhase",
]

import asyncio
import logging
from typing import Any, Callable, Optional

from aragora.config import (
    AGENT_TIMEOUT_SECONDS,
    MAX_CONCURRENT_PROPOSALS,
    PROPOSAL_STAGGER_SECONDS,
)
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.debate.types import AgentType, DebateContextType
from aragora.server.stream.arena_hooks import streaming_task_context

logger = logging.getLogger(__name__)


def _record_calibration_adjustment(agent: str) -> None:
    """Record calibration adjustment metric with lazy import."""
    try:
        from aragora.observability.metrics import record_calibration_adjustment

        record_calibration_adjustment(agent)
    except ImportError:
        pass


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
        # Calibration for proposal confidence scaling
        calibration_tracker: Any = None,
        # Callbacks for orchestrator methods
        build_proposal_prompt: Optional[Callable] = None,
        generate_with_agent: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        notify_spectator: Optional[Callable] = None,
        update_role_assignments: Optional[Callable] = None,
        record_grounded_position: Optional[Callable] = None,
        extract_citation_needs: Optional[Callable] = None,
        # Propulsion engine for push-based work assignment (Gastown pattern)
        propulsion_engine: Any = None,
        enable_propulsion: bool = False,
    ):
        """
        Initialize the proposal phase.

        Args:
            circuit_breaker: CircuitBreaker for agent availability
            position_tracker: Optional PositionTracker for personas
            position_ledger: Optional PositionLedger for grounded personas
            recorder: Optional ReplayRecorder
            hooks: Optional hooks dict for events
            calibration_tracker: Optional CalibrationTracker for confidence scaling
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
        self.calibration_tracker = calibration_tracker

        # Callbacks
        self._build_proposal_prompt = build_proposal_prompt
        self._generate_with_agent = generate_with_agent
        self._with_timeout = with_timeout
        self._notify_spectator = notify_spectator
        self._update_role_assignments = update_role_assignments
        self._record_grounded_position = record_grounded_position
        self._extract_citation_needs = extract_citation_needs

        # Propulsion engine for push-based work assignment
        self._propulsion_engine = propulsion_engine
        self._enable_propulsion = enable_propulsion

    async def execute(self, ctx: "DebateContextType") -> None:
        """
        Execute the proposal phase.

        Args:
            ctx: The DebateContextType to update with proposals
        """
        # Check for cancellation before starting
        if hasattr(ctx, "cancellation_token") and ctx.cancellation_token:
            if ctx.cancellation_token.is_cancelled:
                from aragora.debate.cancellation import DebateCancelled

                raise DebateCancelled(ctx.cancellation_token.reason)

        # Trigger PRE_DEBATE hook if hook_manager is available
        if hasattr(ctx, "hook_manager") and ctx.hook_manager:
            try:
                await ctx.hook_manager.trigger(
                    "pre_debate", ctx=ctx, agents=ctx.agents, task=ctx.env.task
                )
            except Exception as e:
                logger.debug(f"PRE_DEBATE hook failed: {e}")

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

        # 8. Fire propulsion event (proposals_ready) for push-based flow
        await self._fire_propulsion_event(ctx)

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
        """Generate proposals in parallel with bounded concurrency."""

        if not proposers:
            logger.warning("No proposers available for proposal phase")
            return

        # Use semaphore for bounded concurrency (matching critique/revision phases)
        # Legacy stagger mode available via PROPOSAL_STAGGER_SECONDS > 0
        proposal_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROPOSALS)

        async def generate_proposal_bounded(
            idx: int, agent: "AgentType"
        ) -> tuple["AgentType", Any]:
            """Generate proposal with semaphore-bounded concurrency."""
            # Optional stagger delay for backward compatibility
            if PROPOSAL_STAGGER_SECONDS > 0 and idx > 0:
                await asyncio.sleep(PROPOSAL_STAGGER_SECONDS * idx)
            async with proposal_semaphore:
                logger.info(f"proposal_started agent={agent.name} idx={idx}")
                return await self._generate_single_proposal(ctx, agent)

        # Create all tasks immediately (semaphore controls actual concurrency)
        tasks = [
            asyncio.create_task(
                generate_proposal_bounded(idx, agent), name=f"proposal_{agent.name}"
            )
            for idx, agent in enumerate(proposers)
        ]

        # Wait for all proposals and process as they complete
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
            # Use unique task_id to prevent token interleaving between concurrent agents
            task_id = f"{agent.name}:proposal"
            with streaming_task_context(task_id):
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

        def _is_effectively_empty_response(content: Any) -> bool:
            if not isinstance(content, str):
                return False
            normalized = content.strip().lower()
            return not normalized or normalized in (
                "agent response was empty",
                "(agent produced empty output)",
                "agent produced empty output",
            )

        is_error = isinstance(result_or_error, Exception)

        if not is_error and _is_effectively_empty_response(result_or_error):
            provider = getattr(agent, "provider", None) or getattr(agent, "model_type", "unknown")
            logger.warning(
                f"agent_empty_response agent={agent.name} provider={provider} phase=proposal"
            )
            result_or_error = ValueError("Agent response was empty")
            is_error = True

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

    def _record_positions(
        self, ctx: "DebateContextType", agent: "AgentType", proposal: str
    ) -> None:
        """Record positions for truth-grounded personas."""
        debate_id = ctx.debate_id or ctx.env.task[:50]

        # Base proposal confidence (default heuristic)
        raw_confidence = 0.7

        # Apply calibration scaling if available
        calibrated_confidence = self._get_calibrated_confidence(agent.name, raw_confidence, ctx)

        # Legacy position tracker
        if self.position_tracker:
            try:
                self.position_tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent.name,
                    position_type="proposal",
                    position_text=proposal[:1000],
                    round_num=0,
                    confidence=calibrated_confidence,
                )
            except Exception as e:
                logger.debug(f"Position tracking error: {e}")

        # New grounded position system
        if self._record_grounded_position:
            self._record_grounded_position(
                agent.name, proposal, debate_id, 0, calibrated_confidence
            )

    def _get_calibrated_confidence(
        self, agent_name: str, raw_confidence: float, ctx: "DebateContextType"
    ) -> float:
        """Apply calibration scaling to confidence value.

        Uses temperature scaling from CalibrationTracker if available and agent
        has sufficient prediction history.

        Args:
            agent_name: Name of the agent
            raw_confidence: Raw confidence value (0-1)
            ctx: Debate context for domain information

        Returns:
            Calibrated confidence, or raw confidence if calibration unavailable
        """
        if not self.calibration_tracker:
            return raw_confidence

        try:
            domain = getattr(ctx, "domain", None) or "general"
            summary = self.calibration_tracker.get_calibration_summary(agent_name)

            if summary and summary.total_predictions >= 10:
                calibrated = summary.adjust_confidence(raw_confidence, domain=domain)
                if calibrated != raw_confidence:
                    logger.debug(
                        f"[calibration] {agent_name} proposal confidence: "
                        f"{raw_confidence:.2f} -> {calibrated:.2f}"
                    )
                    _record_calibration_adjustment(agent_name)
                return calibrated
        except Exception as e:
            logger.debug(f"[calibration] Failed for {agent_name}: {e}")

        return raw_confidence

    def _emit_message_event(self, agent: "AgentType", content: str) -> None:
        """Emit on_message hook event and agent_message for TTS integration."""
        # Legacy hook for backwards compatibility
        if "on_message" in self.hooks:
            self.hooks["on_message"](
                agent=agent.name,
                content=content,
                role="proposer",
                round_num=0,
            )

        # Emit via spectator notification for EventBus/TTS integration
        if self._notify_spectator:
            try:
                self._notify_spectator(
                    "agent_message",
                    agent=agent.name,
                    content=content,
                    role="proposer",
                    round_num=0,
                    enable_tts=True,
                )
            except Exception as e:
                logger.debug(f"Spectator notification failed: {e}")

    async def _fire_propulsion_event(self, ctx: "DebateContextType") -> None:
        """Fire propulsion event to push work to the next stage.

        Triggers "proposals_ready" event with all generated proposals,
        enabling reactive debate flow via the Gastown pattern.

        Args:
            ctx: The DebateContext with proposals
        """
        if not self._enable_propulsion or not self._propulsion_engine:
            return

        try:
            from aragora.debate.propulsion import PropulsionPayload, PropulsionPriority

            # Create payload with proposal data
            payload = PropulsionPayload(
                data={
                    "proposals": dict(ctx.proposals),
                    "agent_count": len(ctx.proposals),
                    "debate_id": getattr(ctx, "debate_id", None),
                    "task": ctx.env.task[:200] if ctx.env else None,
                },
                priority=PropulsionPriority.NORMAL,
                source_stage="proposal_phase",
                source_molecule_id=getattr(ctx, "debate_id", None),
            )

            # Fire the propulsion event
            results = await self._propulsion_engine.propel("proposals_ready", payload)

            if results:
                success_count = sum(1 for r in results if r.success)
                logger.info(
                    f"[propulsion] proposals_ready fired "
                    f"handlers={len(results)} success={success_count}"
                )
        except ImportError:
            logger.debug("[propulsion] PropulsionEngine imports unavailable")
        except Exception as e:
            logger.warning(f"[propulsion] Failed to fire proposals_ready: {e}")

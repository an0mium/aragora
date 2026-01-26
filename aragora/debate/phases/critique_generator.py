"""
Critique generation module for debate rounds.

Handles parallel critique generation with bounded concurrency.
This module is extracted from debate_rounds.py for better modularity.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from aragora.config import AGENT_TIMEOUT_SECONDS, MAX_CONCURRENT_CRITIQUES
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.server.stream.arena_hooks import streaming_task_context

if TYPE_CHECKING:
    from aragora.core import Agent, Critique, Message
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of a single critique generation attempt."""

    critic: "Agent"
    target_agent: str
    critique: Optional["Critique"]
    error: Optional[Exception]

    @property
    def success(self) -> bool:
        """True if critique was generated successfully."""
        return self.critique is not None and self.error is None


class CritiqueGenerator:
    """
    Generates critiques in parallel with bounded concurrency.

    This class handles the parallel generation of critiques from multiple
    critics for multiple proposals, with semaphore-bounded concurrency
    to prevent API rate limit exhaustion.

    Usage:
        generator = CritiqueGenerator(
            critique_with_agent=arena._critique_with_agent,
            with_timeout=arena._with_timeout,
            circuit_breaker=arena.circuit_breaker,
            hooks=arena.hooks,
            recorder=arena.recorder,
            select_critics_for_proposal=arena._select_critics_for_proposal,
            notify_spectator=arena._notify_spectator,
        )
        await generator.execute_critique_phase(ctx, critics, round_num, result)
    """

    def __init__(
        self,
        critique_with_agent: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        circuit_breaker: Optional[Any] = None,
        hooks: Optional[dict] = None,
        recorder: Optional[Any] = None,
        select_critics_for_proposal: Optional[Callable] = None,
        notify_spectator: Optional[Callable] = None,
        heartbeat_callback: Optional[Callable] = None,
        max_concurrent: int = MAX_CONCURRENT_CRITIQUES,
    ):
        """
        Initialize the critique generator.

        Args:
            critique_with_agent: Async callback for generating critique
            with_timeout: Async timeout wrapper callback
            circuit_breaker: Circuit breaker for agent failure tracking
            hooks: Dictionary of event hooks
            recorder: Debate recorder for logging turns
            select_critics_for_proposal: Callback to select critics for a proposal
            notify_spectator: Callback for spectator notifications
            heartbeat_callback: Callback for emitting heartbeats
            max_concurrent: Maximum concurrent critique generations
        """
        self._critique_with_agent = critique_with_agent
        self._with_timeout = with_timeout
        self.circuit_breaker = circuit_breaker
        self.hooks = hooks or {}
        self.recorder = recorder
        self._select_critics_for_proposal = select_critics_for_proposal
        self._notify_spectator = notify_spectator
        self._emit_heartbeat = heartbeat_callback
        self._max_concurrent = max_concurrent

    async def execute_critique_phase(
        self,
        ctx: "DebateContext",
        critics: List["Agent"],
        round_num: int,
        partial_messages: List["Message"],
        partial_critiques: List["Critique"],
    ) -> Tuple[List["Message"], List["Critique"]]:
        """
        Execute critique phase with parallel generation.

        Args:
            ctx: The DebateContext with proposals
            critics: List of agents who can critique
            round_num: Current round number
            partial_messages: List to append new messages to
            partial_critiques: List to append new critiques to

        Returns:
            Tuple of (new_messages, new_critiques) generated
        """
        from aragora.core import Critique, Message

        result = ctx.result
        proposals = ctx.proposals
        new_messages: List[Message] = []
        new_critiques: List[Critique] = []

        if not self._critique_with_agent:
            logger.warning("No critique_with_agent callback, skipping critiques")
            return (new_messages, new_critiques)

        # Create critique tasks based on topology with bounded concurrency
        critique_semaphore = asyncio.Semaphore(self._max_concurrent)

        async def generate_critique(critic: "Agent", proposal_agent: str, proposal: str):
            """Generate critique and return CritiqueResult."""
            logger.debug(f"critique_generating critic={critic.name} target={proposal_agent}")
            timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))
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
                return CritiqueResult(
                    critic=critic,
                    target_agent=proposal_agent,
                    critique=crit_result,
                    error=None,
                )
            except Exception as e:
                return CritiqueResult(
                    critic=critic,
                    target_agent=proposal_agent,
                    critique=None,
                    error=e,
                )

        async def generate_critique_bounded(critic: "Agent", proposal_agent: str, proposal: str):
            """Wrap critique generation with semaphore for bounded concurrency."""
            async with critique_semaphore:
                return await generate_critique(critic, proposal_agent, proposal)

        # Filter out empty/placeholder proposals
        valid_proposals = {
            agent: content
            for agent, content in proposals.items()
            if content and "(Agent produced empty output)" not in content
        }
        if len(valid_proposals) < len(proposals):
            skipped = [a for a in proposals if a not in valid_proposals]
            logger.warning(f"critique_skip_empty_proposals skipped={skipped}")

        # Create tasks
        critique_tasks = []
        for proposal_agent, proposal in valid_proposals.items():
            if self._select_critics_for_proposal:
                selected_critics = self._select_critics_for_proposal(proposal_agent, critics)
            else:
                selected_critics = [c for c in critics if c.name != proposal_agent]

            for critic in selected_critics:
                critique_tasks.append(
                    asyncio.create_task(generate_critique_bounded(critic, proposal_agent, proposal))
                )

        # Emit heartbeat before critique phase
        if self._emit_heartbeat:
            self._emit_heartbeat(f"critique_round_{round_num}", "generating_critiques")

        # Process critiques as they complete
        critique_count = 0
        total_critiques = len(critique_tasks)

        for completed_task in asyncio.as_completed(critique_tasks):
            try:
                crit_result: CritiqueResult = await completed_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"task_exception phase=critique error={e}")
                continue

            critique_count += 1

            # Emit heartbeat every 3 critiques
            if (
                critique_count % 3 == 0 or critique_count == total_critiques
            ) and self._emit_heartbeat:
                self._emit_heartbeat(
                    f"critique_round_{round_num}",
                    f"completed_{critique_count}_of_{total_critiques}",
                )

            # Process the result (modifies new_messages and new_critiques in place)
            self._process_critique_result(
                crit_result,
                ctx,
                round_num,
                result,
                new_messages,
                new_critiques,
                partial_messages,
                partial_critiques,
            )

        return (new_messages, new_critiques)

    def _process_critique_result(
        self,
        crit_result: CritiqueResult,
        ctx: "DebateContext",
        round_num: int,
        result: Any,
        new_messages: List["Message"],
        new_critiques: List["Critique"],
        partial_messages: List["Message"],
        partial_critiques: List["Critique"],
    ) -> Optional["Critique"]:
        """Process a single critique result."""
        from aragora.core import Critique, Message

        critic = crit_result.critic
        proposal_agent = crit_result.target_agent

        if crit_result.error:
            logger.error(
                f"critique_error critic={critic.name} target={proposal_agent} "
                f"error={crit_result.error}"
            )
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(critic.name)
            # Create placeholder critique so the UI shows a failure instead of a silent drop
            placeholder = Critique(  # type: ignore[call-arg]
                agent=critic.name,
                target_agent=proposal_agent,
                issues=[f"[Critique failed: {crit_result.error}]"],
                suggestions=[],
                severity=0.0,
                reasoning="Critique generation failed due to an exception.",
            )
            result.critiques.append(placeholder)
            partial_critiques.append(placeholder)
            new_critiques.append(placeholder)

            if "on_critique" in self.hooks:
                self.hooks["on_critique"](
                    agent=critic.name,
                    target=proposal_agent,
                    issues=placeholder.issues,
                    severity=placeholder.severity,
                    round_num=round_num,
                    full_content=placeholder.to_prompt(),
                    error=str(crit_result.error),
                )
            return placeholder

        if crit_result.critique is None:
            # Handle timeout/error case
            logger.warning(f"critique_returned_none critic={critic.name} target={proposal_agent}")
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(critic.name)

            # Create placeholder critique
            placeholder = Critique(  # type: ignore[call-arg]
                agent=critic.name,
                target_agent=proposal_agent,
                issues=["[Critique unavailable - agent timed out or encountered an error]"],
                suggestions=[],
                severity=0.0,
                reasoning="Critique generation failed due to timeout or agent error.",
            )
            result.critiques.append(placeholder)
            partial_critiques.append(placeholder)
            new_critiques.append(placeholder)

            # Emit placeholder event
            if "on_critique" in self.hooks:
                self.hooks["on_critique"](
                    agent=critic.name,
                    target=proposal_agent,
                    issues=placeholder.issues,
                    severity=placeholder.severity,
                    round_num=round_num,
                    full_content=placeholder.to_prompt(),
                )
            return placeholder

        # Successful critique
        critique = crit_result.critique
        if self.circuit_breaker:
            self.circuit_breaker.record_success(critic.name)

        result.critiques.append(critique)
        partial_critiques.append(critique)
        new_critiques.append(critique)

        logger.debug(
            f"critique_complete critic={critic.name} target={proposal_agent} "
            f"issues={len(critique.issues)} severity={critique.severity:.1f}"
        )

        # Notify spectator
        if self._notify_spectator:
            self._notify_spectator(
                "critique",
                agent=critic.name,
                details=f"Critiqued {proposal_agent}: {len(critique.issues)} issues",
                metric=critique.severity,
            )

        # Get full critique content
        critique_content = critique.to_prompt()

        # Emit critique event
        if "on_critique" in self.hooks:
            self.hooks["on_critique"](
                agent=critic.name,
                target=proposal_agent,
                issues=critique.issues,
                severity=critique.severity,
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
        partial_messages.append(msg)
        new_messages.append(msg)

        return critique

"""
Revision phase module for debate rounds.

Handles parallel revision generation with bounded concurrency.
This module is extracted from debate_rounds.py for better modularity.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from aragora.config import (
    AGENT_TIMEOUT_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_CONCURRENT_REVISIONS,
)
from aragora.debate.complexity_governor import get_complexity_governor
from aragora.server.stream.arena_hooks import streaming_task_context

if TYPE_CHECKING:
    from aragora.core import Agent, Critique, Message
    from aragora.debate.context import DebateContext
    from aragora.debate.molecules import MoleculeTracker

logger = logging.getLogger(__name__)

# Base timeout for the entire revision phase gather (prevents indefinite stalls)
REVISION_PHASE_BASE_TIMEOUT = 120.0


def calculate_phase_timeout(num_agents: int, agent_timeout: float) -> float:
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


class RevisionGenerator:
    """
    Generates revisions in parallel with bounded concurrency.

    This class handles the parallel generation of revised proposals
    from agents based on critiques received, with semaphore-bounded
    concurrency to prevent API rate limit exhaustion.

    Usage:
        generator = RevisionGenerator(
            generate_with_agent=arena._generate_with_agent,
            build_revision_prompt=arena._build_revision_prompt,
            with_timeout=arena._with_timeout,
            circuit_breaker=arena.circuit_breaker,
            hooks=arena.hooks,
            recorder=arena.recorder,
            notify_spectator=arena._notify_spectator,
            heartbeat_callback=arena._emit_heartbeat,
            record_grounded_position=arena._record_grounded_position,
            rhetorical_observer=arena._rhetorical_observer,
        )
        await generator.execute_revision_phase(ctx, round_num, all_critiques, result)
    """

    def __init__(
        self,
        generate_with_agent: Optional[Callable] = None,
        build_revision_prompt: Optional[Callable] = None,
        with_timeout: Optional[Callable] = None,
        circuit_breaker: Optional[Any] = None,
        hooks: Optional[dict] = None,
        recorder: Optional[Any] = None,
        notify_spectator: Optional[Callable] = None,
        heartbeat_callback: Optional[Callable] = None,
        record_grounded_position: Optional[Callable] = None,
        rhetorical_observer: Optional[Any] = None,
        max_concurrent: int = MAX_CONCURRENT_REVISIONS,
        # Molecule tracking for work unit management (Gastown pattern)
        molecule_tracker: Optional["MoleculeTracker"] = None,
    ):
        """
        Initialize the revision generator.

        Args:
            generate_with_agent: Async callback for generating content
            build_revision_prompt: Callback to build revision prompt
            with_timeout: Async timeout wrapper callback
            circuit_breaker: Circuit breaker for agent failure tracking
            hooks: Dictionary of event hooks
            recorder: Debate recorder for logging turns
            notify_spectator: Callback for spectator notifications
            heartbeat_callback: Callback for emitting heartbeats
            record_grounded_position: Callback for grounded persona tracking
            rhetorical_observer: Rhetorical observer for pattern detection
            max_concurrent: Maximum concurrent revision generations
            molecule_tracker: Optional MoleculeTracker for work unit tracking
        """
        self._generate_with_agent = generate_with_agent
        self._build_revision_prompt = build_revision_prompt
        self._with_timeout = with_timeout
        self.circuit_breaker = circuit_breaker
        self.hooks = hooks or {}
        self.recorder = recorder
        self._notify_spectator = notify_spectator
        self._emit_heartbeat = heartbeat_callback
        self._record_grounded_position = record_grounded_position
        self._rhetorical_observer = rhetorical_observer
        self._max_concurrent = max_concurrent

        # Molecule tracking for work unit management
        self._molecule_tracker = molecule_tracker
        self._active_molecules: Dict[str, str] = {}  # agent_name -> molecule_id

    async def execute_revision_phase(
        self,
        ctx: "DebateContext",
        round_num: int,
        all_critiques: List["Critique"],
        partial_messages: List["Message"],
    ) -> Dict[str, str]:
        """
        Execute revision phase with parallel generation.

        Args:
            ctx: The DebateContext with proposals and proposers
            round_num: Current round number
            all_critiques: List of all critiques from this round
            partial_messages: List to append new messages to

        Returns:
            Dict mapping agent name to revised proposal
        """
        from aragora.core import Message

        result = ctx.result
        proposals = ctx.proposals
        updated_proposals: Dict[str, str] = {}

        if not self._generate_with_agent or not self._build_revision_prompt:
            logger.warning("Missing callbacks for revision phase")
            return updated_proposals

        if not all_critiques:
            return updated_proposals

        # Get complexity-scaled timeout
        timeout = get_complexity_governor().get_scaled_timeout(float(AGENT_TIMEOUT_SECONDS))

        # Semaphore for bounded concurrency
        revision_semaphore = asyncio.Semaphore(self._max_concurrent)

        async def generate_revision_bounded(agent: "Agent", revision_prompt: str):
            """Wrap revision generation with semaphore for bounded concurrency."""
            task_id = f"{agent.name}:revision:{round_num}"
            async with revision_semaphore:
                # Mark molecule as in_progress
                self._start_molecule(agent.name)
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

        # Build revision tasks for all proposers
        revision_tasks = []
        revision_agents = []

        for agent in ctx.proposers:
            # Filter critiques specifically targeting this agent
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

        # Create molecules for tracking if tracker is available
        debate_id = getattr(ctx, "debate_id", None) or (ctx.env.task[:50] if ctx.env else "unknown")
        self._create_revision_molecules(debate_id, round_num, revision_agents)

        # Calculate dynamic phase timeout
        phase_timeout = calculate_phase_timeout(len(revision_agents), timeout)

        # Emit heartbeat before revision phase
        if self._emit_heartbeat:
            self._emit_heartbeat(
                f"revision_round_{round_num}", f"starting_{len(revision_agents)}_agents"
            )

        # Periodic heartbeat task during long-running revisions
        async def heartbeat_during_revisions():
            """Emit heartbeat during revisions to keep connection alive."""
            heartbeat_count = 0
            interval = HEARTBEAT_INTERVAL_SECONDS
            try:
                while True:
                    await asyncio.sleep(interval)
                    heartbeat_count += 1
                    if self._emit_heartbeat:
                        self._emit_heartbeat(
                            f"revision_round_{round_num}",
                            f"in_progress_{heartbeat_count * interval}s",
                        )
            except asyncio.CancelledError:
                pass

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
            revision_results = [asyncio.TimeoutError()] * len(revision_tasks)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Process results
        revision_count = 0
        total_revisions = len(revision_agents)

        for agent, revised in zip(revision_agents, revision_results):
            revision_count += 1

            # Emit heartbeat for each completed revision
            if self._emit_heartbeat:
                self._emit_heartbeat(
                    f"revision_round_{round_num}",
                    f"processed_{revision_count}_of_{total_revisions}",
                )

            if isinstance(revised, BaseException):
                logger.error(f"revision_error agent={agent.name} error={revised}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(agent.name)
                # Mark molecule as failed
                self._fail_molecule(agent.name, str(revised))
                continue

            # Process successful revision
            revised_str: str = revised
            if self.circuit_breaker:
                self.circuit_breaker.record_success(agent.name)

            proposals[agent.name] = revised_str
            updated_proposals[agent.name] = revised_str
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
            partial_messages.append(msg)

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

            # Mark molecule as completed
            self._complete_molecule(
                agent.name,
                {
                    "chars": len(revised_str),
                    "round": round_num,
                },
            )

        return updated_proposals

    def _observe_rhetorical_patterns(
        self,
        agent: str,
        content: str,
        round_num: int,
        loop_id: str,
    ) -> None:
        """Observe content for rhetorical patterns.

        Args:
            agent: Name of the agent
            content: Content to analyze
            round_num: Current round number
            loop_id: Optional loop ID for tracking
        """
        if not self._rhetorical_observer:
            return

        try:
            # Analyze content for rhetorical patterns
            self._rhetorical_observer.observe(
                agent=agent,
                content=content,
                round=round_num,
                loop_id=loop_id,
            )
        except Exception as e:
            logger.debug(f"Rhetorical observation error: {e}")

    # Molecule tracking methods (Gastown pattern)

    def _create_revision_molecules(
        self,
        debate_id: str,
        round_num: int,
        agents: List["Agent"],
    ) -> None:
        """Create revision molecules for all agents.

        Args:
            debate_id: ID of the current debate
            round_num: Current round number
            agents: List of agents generating revisions
        """
        if not self._molecule_tracker:
            return

        try:
            from aragora.debate.molecules import MoleculeType

            for agent in agents:
                molecule = self._molecule_tracker.create_molecule(
                    debate_id=debate_id,
                    molecule_type=MoleculeType.REVISION,
                    round_number=round_num,
                    input_data={"agent": agent.name, "round": round_num},
                )
                self._active_molecules[agent.name] = molecule.molecule_id
                logger.debug(
                    f"[molecule] Created revision molecule {molecule.molecule_id} "
                    f"for agent={agent.name} round={round_num}"
                )
        except ImportError:
            logger.debug("[molecule] Molecule imports unavailable")
        except Exception as e:
            logger.debug(f"[molecule] Failed to create revision molecules: {e}")

    def _start_molecule(self, agent_name: str) -> None:
        """Mark a revision molecule as in_progress.

        Args:
            agent_name: Name of the agent whose molecule to start
        """
        if not self._molecule_tracker:
            return

        molecule_id = self._active_molecules.get(agent_name)
        if molecule_id:
            try:
                self._molecule_tracker.start_molecule(molecule_id)
                logger.debug(f"[molecule] Started revision molecule {molecule_id}")
            except Exception as e:
                logger.debug(f"[molecule] Failed to start molecule: {e}")

    def _complete_molecule(self, agent_name: str, output: Dict[str, Any]) -> None:
        """Mark a revision molecule as completed.

        Args:
            agent_name: Name of the agent whose molecule to complete
            output: Output data from the revision
        """
        if not self._molecule_tracker:
            return

        molecule_id = self._active_molecules.get(agent_name)
        if molecule_id:
            try:
                self._molecule_tracker.complete_molecule(molecule_id, output)
                logger.debug(f"[molecule] Completed revision molecule {molecule_id}")
            except Exception as e:
                logger.debug(f"[molecule] Failed to complete molecule: {e}")

    def _fail_molecule(self, agent_name: str, error: str) -> None:
        """Mark a revision molecule as failed.

        Args:
            agent_name: Name of the agent whose molecule failed
            error: Error message
        """
        if not self._molecule_tracker:
            return

        molecule_id = self._active_molecules.get(agent_name)
        if molecule_id:
            try:
                self._molecule_tracker.fail_molecule(molecule_id, error)
                logger.debug(f"[molecule] Failed revision molecule {molecule_id}: {error}")
            except Exception as e:
                logger.debug(f"[molecule] Failed to record molecule failure: {e}")

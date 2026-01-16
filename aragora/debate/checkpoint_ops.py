"""
Checkpoint and memory operations for Arena.

Extracted from Arena to reduce orchestrator size. Handles:
- Checkpoint creation during debate rounds
- Memory outcome storage and updates
- Evidence storage in memory
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import DebateResult
    from aragora.debate.context import DebateContext
    from aragora.debate.memory_manager import MemoryManager
    from aragora.debate.state_cache import DebateStateCache

logger = logging.getLogger(__name__)


class CheckpointOperations:
    """Manages checkpoint and memory operations for Arena debates.

    Extracted from Arena to centralize checkpoint and memory operations:
    - Creating checkpoints during debate rounds
    - Storing debate outcomes in memory
    - Storing evidence snippets
    - Updating memory based on outcomes

    Usage:
        ops = CheckpointOperations(
            checkpoint_manager=checkpoint_mgr,
            memory_manager=mem_mgr,
            cache=cache,
        )
        await ops.create_checkpoint(ctx, round_num, env, agents, protocol)
    """

    def __init__(
        self,
        checkpoint_manager: Any = None,
        memory_manager: Optional["MemoryManager"] = None,
        cache: Optional["DebateStateCache"] = None,
    ) -> None:
        """Initialize checkpoint operations.

        Args:
            checkpoint_manager: Optional CheckpointManager for debate resume
            memory_manager: Optional MemoryManager for outcome storage
            cache: Optional state cache for tracking retrieved IDs
        """
        self.checkpoint_manager = checkpoint_manager
        self.memory_manager = memory_manager
        self._cache = cache

    async def create_checkpoint(
        self,
        ctx: "DebateContext",
        round_num: int,
        env: Any,
        agents: list,
        protocol: Any,
    ) -> None:
        """Create a checkpoint after a debate round.

        Called by DebateRoundsPhase after each round completes.
        Only checkpoints if should_checkpoint returns True.

        Args:
            ctx: DebateContext with current debate state
            round_num: The round number that just completed
            env: Environment with task info
            agents: List of agents
            protocol: Debate protocol with rounds info
        """
        if not self.checkpoint_manager:
            return

        if not self.checkpoint_manager.should_checkpoint(ctx.debate_id, round_num):
            return

        try:
            await self.checkpoint_manager.create_checkpoint(
                debate_id=ctx.debate_id,
                task=env.task,
                current_round=round_num,
                total_rounds=protocol.rounds,
                phase="revision",
                messages=ctx.result.messages,
                critiques=ctx.result.critiques,
                votes=ctx.result.votes,
                agents=agents,
                current_consensus=getattr(ctx.result, "final_answer", None),
            )
            logger.debug(f"[checkpoint] Saved checkpoint after round {round_num}")
        except (IOError, OSError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"[checkpoint] Failed to create checkpoint: {e}")

    def store_debate_outcome(
        self,
        result: "DebateResult",
        task: str,
        belief_cruxes: Optional[list[str]] = None,
    ) -> None:
        """Store debate outcome in ContinuumMemory for future retrieval.

        Args:
            result: The debate result to store
            task: The debate task description
            belief_cruxes: Optional list of belief cruxes from result
        """
        if not self.memory_manager:
            return
        self.memory_manager.store_debate_outcome(result, task, belief_cruxes=belief_cruxes)

    def store_evidence(self, evidence_snippets: list, task: str) -> None:
        """Store collected evidence snippets in ContinuumMemory.

        Args:
            evidence_snippets: List of evidence snippets to store
            task: The debate task description
        """
        if not self.memory_manager:
            return
        self.memory_manager.store_evidence(evidence_snippets, task)

    def update_memory_outcomes(self, result: "DebateResult") -> None:
        """Update retrieved memories based on debate outcome.

        Args:
            result: The debate result for outcome evaluation
        """
        if not self.memory_manager or not self._cache:
            return

        # Sync tracked IDs and tier info to memory manager
        self.memory_manager.track_retrieved_ids(
            self._cache.continuum_retrieved_ids,
            tiers=self._cache.continuum_retrieved_tiers,
        )
        self.memory_manager.update_memory_outcomes(result)
        # Clear local tracking
        self._cache.clear_continuum_tracking()


__all__ = ["CheckpointOperations"]

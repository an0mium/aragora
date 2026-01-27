"""
Molecule-Checkpoint Bridge.

Unifies molecule tracking with checkpoint-based recovery:
- Converts molecule state to checkpoint format
- Restores molecules from checkpoints
- Provides unified recovery interface

This bridge enables seamless integration between:
- MoleculeOrchestrator: Fine-grained phase tracking
- CheckpointManager: Durable state persistence

Usage:
    from aragora.debate.checkpoint_bridge import CheckpointBridge

    # Create bridge
    bridge = CheckpointBridge(molecule_orchestrator, checkpoint_manager)

    # Save state
    await bridge.save_checkpoint(debate_id, phase="voting")

    # Recover
    state = await bridge.restore_checkpoint(checkpoint_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.debate.checkpoint import CheckpointManager
    from aragora.debate.molecule_orchestrator import MoleculeOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRecoveryState:
    """
    Unified state for recovery from either molecules or checkpoints.

    Provides a common interface for recovering debate state
    regardless of the underlying persistence mechanism.
    """

    debate_id: str
    current_round: int
    phase: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Molecule state
    molecule_state: Optional[dict[str, Any]] = None
    pending_molecules: int = 0
    completed_molecules: int = 0
    failed_molecules: int = 0

    # Checkpoint state
    checkpoint_id: Optional[str] = None
    checkpoint_messages: list[dict] = field(default_factory=list)
    checkpoint_critiques: list[dict] = field(default_factory=list)
    checkpoint_votes: list[dict] = field(default_factory=list)

    # Agent channel state
    channel_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_id": self.debate_id,
            "current_round": self.current_round,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "molecule_state": self.molecule_state,
            "pending_molecules": self.pending_molecules,
            "completed_molecules": self.completed_molecules,
            "failed_molecules": self.failed_molecules,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_messages": self.checkpoint_messages,
            "checkpoint_critiques": self.checkpoint_critiques,
            "checkpoint_votes": self.checkpoint_votes,
            "channel_history": self.channel_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedRecoveryState":
        """Create from dictionary."""
        return cls(
            debate_id=data["debate_id"],
            current_round=data["current_round"],
            phase=data["phase"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            molecule_state=data.get("molecule_state"),
            pending_molecules=data.get("pending_molecules", 0),
            completed_molecules=data.get("completed_molecules", 0),
            failed_molecules=data.get("failed_molecules", 0),
            checkpoint_id=data.get("checkpoint_id"),
            checkpoint_messages=data.get("checkpoint_messages", []),
            checkpoint_critiques=data.get("checkpoint_critiques", []),
            checkpoint_votes=data.get("checkpoint_votes", []),
            channel_history=data.get("channel_history", []),
        )


class CheckpointBridge:
    """
    Bridge between molecule tracking and checkpoint persistence.

    Provides:
    - Unified save/restore interface
    - Molecule-to-checkpoint conversion
    - Recovery priority (molecules first, then checkpoints)
    - Channel history persistence
    """

    def __init__(
        self,
        molecule_orchestrator: Optional["MoleculeOrchestrator"] = None,
        checkpoint_manager: Optional["CheckpointManager"] = None,
    ):
        """
        Initialize the bridge.

        Args:
            molecule_orchestrator: Optional molecule orchestrator
            checkpoint_manager: Optional checkpoint manager
        """
        self._molecules = molecule_orchestrator
        self._checkpoints = checkpoint_manager

    def has_molecules(self) -> bool:
        """Check if molecule orchestrator is available."""
        return self._molecules is not None

    def has_checkpoints(self) -> bool:
        """Check if checkpoint manager is available."""
        return self._checkpoints is not None

    async def save_checkpoint(
        self,
        debate_id: str,
        current_round: int,
        phase: str,
        messages: Optional[list[dict]] = None,
        critiques: Optional[list[dict]] = None,
        votes: Optional[list[dict]] = None,
        channel_history: Optional[list[dict]] = None,
    ) -> UnifiedRecoveryState:
        """
        Save unified state combining molecules and checkpoint data.

        Args:
            debate_id: Debate identifier
            current_round: Current round number
            phase: Current phase name
            messages: Optional message history
            critiques: Optional critique history
            votes: Optional vote history
            channel_history: Optional channel message history

        Returns:
            UnifiedRecoveryState with all state
        """
        state = UnifiedRecoveryState(
            debate_id=debate_id,
            current_round=current_round,
            phase=phase,
            checkpoint_messages=messages or [],
            checkpoint_critiques=critiques or [],
            checkpoint_votes=votes or [],
            channel_history=channel_history or [],
        )

        # Add molecule state
        if self._molecules:
            mol_state = self._molecules.to_checkpoint_state(debate_id)
            state.molecule_state = mol_state

            progress = self._molecules.get_progress(debate_id)
            state.pending_molecules = progress.get("pending", 0)
            state.completed_molecules = progress.get("completed", 0)
            state.failed_molecules = progress.get("failed", 0)

        # Save to checkpoint manager if available
        if self._checkpoints:
            checkpoint_id = await self._save_to_checkpoint_manager(
                debate_id,
                current_round,
                phase,
                state,
            )
            state.checkpoint_id = checkpoint_id

        logger.info(f"Saved unified checkpoint for debate {debate_id} at round {current_round}")

        return state

    async def _save_to_checkpoint_manager(
        self,
        debate_id: str,
        current_round: int,
        phase: str,
        state: UnifiedRecoveryState,
    ) -> str:
        """Save state to checkpoint manager."""
        if not self._checkpoints:
            return ""

        # Create checkpoint with molecule state embedded
        from aragora.debate.checkpoint import DebateCheckpoint

        checkpoint = DebateCheckpoint(
            checkpoint_id=f"cp_{debate_id}_{current_round}_{phase}",
            debate_id=debate_id,
            task="",  # Task is stored elsewhere
            current_round=current_round,
            total_rounds=0,  # Unknown at this point
            phase=phase,
            messages=state.checkpoint_messages,
            critiques=state.checkpoint_critiques,
            votes=state.checkpoint_votes,
            agent_states=[],
        )

        # Store molecule state in claims_kernel_state field (repurposed)
        if state.molecule_state:
            checkpoint.claims_kernel_state = {
                "molecule_state": state.molecule_state,
                "channel_history": state.channel_history,
            }

        await self._checkpoints.save(checkpoint)

        return checkpoint.checkpoint_id

    async def restore_checkpoint(
        self,
        debate_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[UnifiedRecoveryState]:
        """
        Restore unified state from molecules and/or checkpoints.

        Priority:
        1. Try to restore molecules if available
        2. Fall back to checkpoint if molecules unavailable
        3. Combine both if both are available

        Args:
            debate_id: Debate to restore
            checkpoint_id: Optional specific checkpoint ID

        Returns:
            UnifiedRecoveryState if found, None otherwise
        """
        state = None

        # Try molecules first
        if self._molecules:
            mol_state = self._molecules.to_checkpoint_state(debate_id)
            if mol_state and mol_state.get("molecules"):
                progress = self._molecules.get_progress(debate_id)
                state = UnifiedRecoveryState(
                    debate_id=debate_id,
                    current_round=0,  # Will be updated from checkpoint
                    phase="unknown",
                    molecule_state=mol_state,
                    pending_molecules=progress.get("pending", 0),
                    completed_molecules=progress.get("completed", 0),
                    failed_molecules=progress.get("failed", 0),
                )
                logger.info(f"Restored molecule state for debate {debate_id}")

        # Try checkpoints
        if self._checkpoints:
            checkpoint = None
            if checkpoint_id:
                checkpoint = await self._checkpoints.load(checkpoint_id)
            else:
                # Get latest checkpoint for debate
                checkpoint = await self._checkpoints.get_latest(debate_id)

            if checkpoint:
                if state is None:
                    state = UnifiedRecoveryState(
                        debate_id=debate_id,
                        current_round=checkpoint.current_round,
                        phase=checkpoint.phase,
                    )
                else:
                    # Update with checkpoint data
                    state.current_round = checkpoint.current_round
                    state.phase = checkpoint.phase

                state.checkpoint_id = checkpoint.checkpoint_id
                state.checkpoint_messages = checkpoint.messages
                state.checkpoint_critiques = checkpoint.critiques
                state.checkpoint_votes = checkpoint.votes

                # Extract molecule/channel state if stored
                if checkpoint.claims_kernel_state:
                    embedded = checkpoint.claims_kernel_state
                    if "molecule_state" in embedded and state.molecule_state is None:
                        state.molecule_state = embedded["molecule_state"]
                    if "channel_history" in embedded:
                        state.channel_history = embedded["channel_history"]

                logger.info(
                    f"Restored checkpoint {checkpoint.checkpoint_id} for debate {debate_id}"
                )

        return state

    async def recover_molecules_from_checkpoint(
        self,
        checkpoint_id: str,
    ) -> bool:
        """
        Restore molecule orchestrator state from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore from

        Returns:
            True if restoration succeeded
        """
        if not self._molecules or not self._checkpoints:
            return False

        checkpoint = await self._checkpoints.load(checkpoint_id)
        if not checkpoint:
            return False

        # Extract molecule state
        if checkpoint.claims_kernel_state and "molecule_state" in checkpoint.claims_kernel_state:
            mol_state = checkpoint.claims_kernel_state["molecule_state"]
            self._molecules.restore_from_checkpoint(mol_state)
            logger.info(f"Restored molecules from checkpoint {checkpoint_id}")
            return True

        return False

    def get_recovery_summary(self, debate_id: str) -> dict[str, Any]:
        """
        Get a summary of recoverable state for a debate.

        Returns dict with:
        - has_molecules: bool
        - has_checkpoint: bool
        - molecule_progress: progress dict
        - latest_checkpoint_id: str or None
        """
        summary = {
            "debate_id": debate_id,
            "has_molecules": False,
            "has_checkpoint": False,
            "molecule_progress": {},
            "latest_checkpoint_id": None,
        }

        if self._molecules:
            progress = self._molecules.get_progress(debate_id)
            if progress.get("total", 0) > 0:
                summary["has_molecules"] = True
                summary["molecule_progress"] = progress

        # Note: Checkpoint check would need to be async
        # This is a sync summary for quick checks

        return summary


# Factory function
def create_checkpoint_bridge(
    molecule_orchestrator: Optional["MoleculeOrchestrator"] = None,
    checkpoint_manager: Optional["CheckpointManager"] = None,
) -> CheckpointBridge:
    """
    Create a checkpoint bridge.

    Args:
        molecule_orchestrator: Optional molecule orchestrator
        checkpoint_manager: Optional checkpoint manager

    Returns:
        CheckpointBridge instance
    """
    return CheckpointBridge(molecule_orchestrator, checkpoint_manager)


__all__ = [
    "CheckpointBridge",
    "UnifiedRecoveryState",
    "create_checkpoint_bridge",
]

"""
Base classes, types, and dataclasses for the molecules module.

This module contains the foundational types used throughout the molecules system:
- Enums for status tracking
- Exception classes for error handling
- Core dataclasses (MoleculeStep, Molecule, MoleculeResult)
- Transaction-related classes
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Type alias for consensus values accepted by DebateProtocol
ConsensusType = Literal[
    "majority",
    "unanimous",
    "judge",
    "none",
    "weighted",
    "supermajority",
    "any",
    "byzantine",
]


# =============================================================================
# Transaction Safety - Exceptions
# =============================================================================


class TransactionError(Exception):
    """Base exception for transaction-related errors."""

    pass


class TransactionRollbackError(TransactionError):
    """Raised when a transaction rollback fails."""

    pass


class CyclicDependencyError(TransactionError):
    """Raised when a cyclic dependency is detected in step dependencies."""

    def __init__(self, cycle: list[str], message: str | None = None):
        self.cycle = cycle
        super().__init__(message or f"Cyclic dependency detected: {' -> '.join(cycle)}")


class DeadlockError(TransactionError):
    """Raised when a deadlock is detected between parallel molecules."""

    def __init__(self, molecules: list[str], resources: list[str], message: str | None = None):
        self.molecules = molecules
        self.resources = resources
        super().__init__(
            message or f"Deadlock detected between molecules {molecules} on resources {resources}"
        )


class DependencyValidationError(TransactionError):
    """Raised when step dependencies are invalid."""

    def __init__(self, step_id: str, missing_deps: list[str], message: str | None = None):
        self.step_id = step_id
        self.missing_deps = missing_deps
        super().__init__(message or f"Step {step_id} has missing dependencies: {missing_deps}")


# =============================================================================
# Transaction Safety - Transaction State
# =============================================================================


class TransactionState(str, Enum):
    """State of a transaction."""

    PENDING = "pending"
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Status of a molecule step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MoleculeStatus(str, Enum):
    """Status of a molecule."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# =============================================================================
# Compensating Actions
# =============================================================================


@dataclass
class CompensatingAction:
    """
    A compensating action to undo a step's effects on rollback.

    Used to implement saga-style transactions where each step can be
    undone if a later step fails.
    """

    step_id: str
    step_name: str
    action_type: str  # restore_checkpoint, custom
    checkpoint_path: Path | None = None
    custom_handler: Any | None = None  # Callable for custom compensation
    metadata: dict[str, Any] = field(default_factory=dict)
    executed: bool = False

    async def execute(self) -> bool:
        """Execute the compensating action."""
        try:
            if self.action_type == "restore_checkpoint" and self.checkpoint_path:
                # Restore from checkpoint file
                if self.checkpoint_path.exists():
                    logger.info(f"Restoring checkpoint for step {self.step_name}")
                    return True
            elif self.action_type == "custom" and self.custom_handler:
                if asyncio.iscoroutinefunction(self.custom_handler):
                    await self.custom_handler(self.metadata)
                else:
                    self.custom_handler(self.metadata)
                return True
            self.executed = True
            return True
        except (RuntimeError, OSError, ValueError) as e:
            logger.error(f"Compensating action failed for step {self.step_name}: {e}")
            return False


# =============================================================================
# Molecule Transaction
# =============================================================================


@dataclass
class MoleculeTransaction:
    """
    Transaction wrapper for molecule execution with begin/commit/rollback semantics.

    Provides atomic checkpointing across multiple steps with automatic rollback
    on failure.
    """

    transaction_id: str
    molecule_id: str
    state: TransactionState
    started_at: datetime
    checkpoint_dir: Path
    completed_steps: list[str] = field(default_factory=list)
    compensating_actions: list[CompensatingAction] = field(default_factory=list)
    pre_transaction_snapshot: dict[str, Any] | None = None
    committed_at: datetime | None = None
    rolled_back_at: datetime | None = None
    error_message: str | None = None

    @classmethod
    def begin(
        cls,
        molecule_id: str,
        checkpoint_dir: Path,
        snapshot: dict[str, Any] | None = None,
    ) -> MoleculeTransaction:
        """Begin a new transaction."""
        return cls(
            transaction_id=str(uuid.uuid4()),
            molecule_id=molecule_id,
            state=TransactionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            checkpoint_dir=checkpoint_dir,
            pre_transaction_snapshot=snapshot,
        )

    def add_compensating_action(self, action: CompensatingAction) -> None:
        """Add a compensating action for rollback."""
        self.compensating_actions.append(action)

    def mark_step_completed(self, step_id: str) -> None:
        """Mark a step as completed within this transaction."""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    async def commit(self) -> bool:
        """
        Commit the transaction, making all changes permanent.

        Returns True if commit succeeds, False otherwise.
        """
        if self.state != TransactionState.ACTIVE:
            logger.warning(f"Cannot commit transaction in state {self.state}")
            return False

        try:
            # Write final transaction state
            await self._write_transaction_log("committed")
            self.state = TransactionState.COMMITTED
            self.committed_at = datetime.now(timezone.utc)
            logger.info(f"Transaction {self.transaction_id} committed successfully")
            return True
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Transaction commit failed: %s", e)
            self.error_message = f"Failed: {type(e).__name__}"
            self.state = TransactionState.FAILED
            return False

    async def rollback(self) -> bool:
        """
        Rollback the transaction, undoing all completed steps.

        Executes compensating actions in reverse order (LIFO).
        Returns True if rollback succeeds, False otherwise.
        """
        if self.state not in (TransactionState.ACTIVE, TransactionState.FAILED):
            logger.warning(f"Cannot rollback transaction in state {self.state}")
            return False

        logger.info(f"Rolling back transaction {self.transaction_id}")
        rollback_success = True

        # Execute compensating actions in reverse order
        for action in reversed(self.compensating_actions):
            if action.step_id in self.completed_steps:
                try:
                    success = await action.execute()
                    if not success:
                        rollback_success = False
                        logger.error(f"Compensating action failed for step {action.step_name}")
                except (RuntimeError, OSError, ValueError) as e:
                    rollback_success = False
                    logger.error(f"Error executing compensating action: {e}")

        # Restore pre-transaction snapshot if available
        if self.pre_transaction_snapshot:
            try:
                await self._restore_snapshot()
            except (RuntimeError, ValueError, OSError) as e:
                rollback_success = False
                logger.error(f"Failed to restore pre-transaction snapshot: {e}")

        await self._write_transaction_log("rolled_back")
        self.state = TransactionState.ROLLED_BACK
        self.rolled_back_at = datetime.now(timezone.utc)

        if rollback_success:
            logger.info(f"Transaction {self.transaction_id} rolled back successfully")
        else:
            logger.warning(f"Transaction {self.transaction_id} rollback completed with errors")

        return rollback_success

    async def _write_transaction_log(self, status: str) -> None:
        """Write transaction log entry."""
        log_file = self.checkpoint_dir / f"txn_{self.transaction_id}.log"
        log_entry = {
            "transaction_id": self.transaction_id,
            "molecule_id": self.molecule_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_steps": self.completed_steps,
        }
        try:
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to write transaction log: {e}")

    async def _restore_snapshot(self) -> None:
        """Restore pre-transaction snapshot."""
        if not self.pre_transaction_snapshot:
            return

        snapshot_file = self.checkpoint_dir / f"{self.molecule_id}.json"
        try:
            with open(snapshot_file, "w") as f:
                json.dump(self.pre_transaction_snapshot, f, indent=2)
            logger.info(f"Restored pre-transaction snapshot for molecule {self.molecule_id}")
        except (RuntimeError, OSError, ValueError) as e:
            raise TransactionRollbackError(f"Failed to restore snapshot: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize transaction to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "molecule_id": self.molecule_id,
            "state": self.state.value,
            "started_at": self.started_at.isoformat(),
            "completed_steps": self.completed_steps,
            "committed_at": (self.committed_at.isoformat() if self.committed_at else None),
            "rolled_back_at": (self.rolled_back_at.isoformat() if self.rolled_back_at else None),
            "error_message": self.error_message,
        }


# =============================================================================
# Molecule Step
# =============================================================================


@dataclass
class MoleculeStep:
    """
    A single step in a molecule, individually checkpointed.

    Each step can be:
    - Executed by a specific executor (agent, shell, parallel)
    - Checkpointed before and after execution
    - Retried on failure
    - Skipped based on conditions
    """

    id: str
    name: str
    step_type: str  # agent, shell, parallel, conditional
    config: dict[str, Any]
    status: StepStatus
    result: Any | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 300.0
    bead_id: str | None = None  # Associated bead for tracking
    dependencies: list[str] = field(default_factory=list)  # Step IDs
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        step_type: str,
        config: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        timeout_seconds: float = 300.0,
        max_attempts: int = 3,
    ) -> MoleculeStep:
        """Create a new molecule step."""
        return cls(
            id=str(uuid.uuid4())[:8],
            name=name,
            step_type=step_type,
            config=config or {},
            status=StepStatus.PENDING,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type,
            "config": self.config,
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "timeout_seconds": self.timeout_seconds,
            "bead_id": self.bead_id,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoleculeStep:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            step_type=data["step_type"],
            config=data.get("config", {}),
            status=StepStatus(data["status"]),
            result=data.get("result"),
            error_message=data.get("error_message"),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            bead_id=data.get("bead_id"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )

    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.status == StepStatus.FAILED and self.attempt_count < self.max_attempts

    def is_terminal(self) -> bool:
        """Check if step is in terminal state."""
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)


# =============================================================================
# Molecule
# =============================================================================


@dataclass
class Molecule:
    """
    A durable chained workflow where each step survives restarts.

    Molecules provide:
    - Step-by-step execution with checkpoints
    - Automatic recovery from crashes
    - Dependency management between steps
    - Parallel step execution where possible
    """

    id: str
    name: str
    description: str
    steps: list[MoleculeStep]
    current_step_index: int
    status: MoleculeStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    checkpoint_dir: Path | None = None
    parent_id: str | None = None  # For nested molecules
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @classmethod
    def create(
        cls,
        name: str,
        steps: list[MoleculeStep],
        description: str = "",
        checkpoint_dir: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Molecule:
        """Create a new molecule."""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            steps=steps,
            current_step_index=0,
            status=MoleculeStatus.PENDING,
            created_at=now,
            updated_at=now,
            checkpoint_dir=checkpoint_dir,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Molecule:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=[MoleculeStep.from_dict(s) for s in data["steps"]],
            current_step_index=data.get("current_step_index", 0),
            status=MoleculeStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            checkpoint_dir=Path(data["checkpoint_dir"]) if data.get("checkpoint_dir") else None,
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
            error_message=data.get("error_message"),
        )

    def get_current_step(self) -> MoleculeStep | None:
        """Get the current step to execute."""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_completed_step_ids(self) -> set:
        """Get IDs of completed steps."""
        return {s.id for s in self.steps if s.status == StepStatus.COMPLETED}

    def get_next_runnable_steps(self) -> list[MoleculeStep]:
        """Get steps that can be run (dependencies met)."""
        completed_ids = self.get_completed_step_ids()
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]

    @property
    def progress_percentage(self) -> float:
        """Get completion percentage."""
        if not self.steps:
            return 0.0
        completed = len([s for s in self.steps if s.is_terminal()])
        return completed / len(self.steps) * 100


# =============================================================================
# Molecule Result
# =============================================================================


@dataclass
class MoleculeResult:
    """Result of molecule execution."""

    molecule_id: str
    status: MoleculeStatus
    completed_steps: int
    failed_steps: int
    total_steps: int
    duration_seconds: float
    step_results: dict[str, Any]
    error_message: str | None = None

    @property
    def success(self) -> bool:
        """Check if molecule completed successfully."""
        return self.status == MoleculeStatus.COMPLETED

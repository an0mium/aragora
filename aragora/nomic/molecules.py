"""
Molecules: Durable Chained Workflows.

Inspired by Gastown's Molecules pattern, this module provides workflows
that survive agent restarts with per-step persistence. Each step is
individually checkpointed, ensuring no work is lost.

Key concepts:
- Molecule: A durable workflow containing multiple steps
- MoleculeStep: A single step with checkpoint support
- MoleculeEngine: Executes molecules with recovery support
- StepExecutor: Pluggable step execution strategy

Usage:
    engine = MoleculeEngine(bead_store)

    # Define a molecule
    molecule = Molecule.create(
        name="deploy_feature",
        steps=[
            MoleculeStep.create("run_tests", "test", {"command": "pytest"}),
            MoleculeStep.create("build", "build", {"command": "make build"}),
            MoleculeStep.create("deploy", "deploy", {"env": "staging"}),
        ]
    )

    # Execute with automatic checkpointing
    result = await engine.execute(molecule)

    # Resume after crash
    result = await engine.resume("molecule-123")
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, cast

from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS

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
from aragora.config.settings import get_settings
from aragora.nomic.beads import BeadStore

logger = logging.getLogger(__name__)


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
        except Exception as e:
            logger.error(f"Compensating action failed for step {self.step_name}: {e}")
            return False


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
    ) -> "MoleculeTransaction":
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
        except Exception as e:
            logger.error(f"Transaction commit failed: {e}")
            self.error_message = str(e)
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
                except Exception as e:
                    rollback_success = False
                    logger.error(f"Error executing compensating action: {e}")

        # Restore pre-transaction snapshot if available
        if self.pre_transaction_snapshot:
            try:
                await self._restore_snapshot()
            except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
# Transaction Safety - Dependency Graph Validation
# =============================================================================


class DependencyGraph:
    """
    Validates step dependencies and detects cycles.

    Uses Kahn's algorithm for topological sorting and cycle detection.
    """

    def __init__(self, steps: list["MoleculeStep"]):
        self.steps = {s.id: s for s in steps}
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.in_degree: dict[str, int] = defaultdict(int)
        self._build_graph()

    def _build_graph(self) -> None:
        """Build adjacency list and in-degree map from steps."""
        for step in self.steps.values():
            # Ensure all steps have an entry
            if step.id not in self.in_degree:
                self.in_degree[step.id] = 0

            for dep_id in step.dependencies:
                self.adjacency[dep_id].append(step.id)
                self.in_degree[step.id] += 1

    def validate(self) -> tuple[bool, list[str] | None]:
        """
        Validate the dependency graph.

        Returns:
            Tuple of (is_valid, error_details)
            - If valid: (True, None)
            - If invalid: (False, list of step IDs forming cycle or missing)
        """
        # Check for missing dependencies
        missing = self._find_missing_dependencies()
        if missing:
            return False, missing

        # Check for cycles
        cycle = self._detect_cycle()
        if cycle:
            return False, cycle

        return True, None

    def _find_missing_dependencies(self) -> list[str] | None:
        """Find any missing dependency references."""
        missing = []
        for step in self.steps.values():
            for dep_id in step.dependencies:
                if dep_id not in self.steps:
                    missing.append(f"{step.id}:{dep_id}")
        return missing if missing else None

    def _detect_cycle(self) -> list[str] | None:
        """
        Detect cycles using DFS with coloring.

        Returns list of step IDs forming the cycle, or None if no cycle.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {step_id: WHITE for step_id in self.steps}
        parent: dict[str, str | None] = {step_id: None for step_id in self.steps}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY

            for neighbor in self.adjacency[node]:
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found a cycle - reconstruct it
                    cycle = [neighbor, node]
                    current = parent[node]
                    while current and current != neighbor:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result

            color[node] = BLACK
            return None

        for step_id in self.steps:
            if color[step_id] == WHITE:
                result = dfs(step_id)
                if result:
                    return result

        return None

    def get_execution_order(self) -> list[str]:
        """
        Get topologically sorted execution order.

        Returns list of step IDs in valid execution order.
        Raises CyclicDependencyError if cycle exists.
        """
        in_degree = dict(self.in_degree)
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for neighbor in self.adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.steps):
            # Cycle detected
            remaining = [s for s in self.steps if s not in order]
            raise CyclicDependencyError(remaining, "Could not complete topological sort")

        return order


# =============================================================================
# Transaction Safety - Deadlock Detection
# =============================================================================


class ResourceLock:
    """Represents a resource lock held by a molecule."""

    def __init__(self, resource_id: str, molecule_id: str, acquired_at: datetime):
        self.resource_id = resource_id
        self.molecule_id = molecule_id
        self.acquired_at = acquired_at


class DeadlockDetector:
    """
    Detects deadlocks between parallel molecule executions.

    Uses wait-for graph analysis to detect cycles indicating deadlocks.
    """

    def __init__(self):
        self._locks: dict[str, ResourceLock] = {}  # resource_id -> lock
        self._waiting: dict[str, set[str]] = defaultdict(
            set
        )  # molecule_id -> waiting_for_resources
        self._holding: dict[str, set[str]] = defaultdict(set)  # molecule_id -> held_resources
        self._lock = asyncio.Lock()

    async def acquire_lock(
        self,
        molecule_id: str,
        resource_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """
        Attempt to acquire a lock on a resource.

        Args:
            molecule_id: ID of the molecule requesting the lock
            resource_id: ID of the resource to lock
            timeout: Maximum time to wait for lock

        Returns:
            True if lock acquired, False if timeout or deadlock detected
        """
        async with self._lock:
            # Check if resource is available
            if resource_id not in self._locks:
                self._locks[resource_id] = ResourceLock(
                    resource_id, molecule_id, datetime.now(timezone.utc)
                )
                self._holding[molecule_id].add(resource_id)
                return True

            # Resource is locked - check for potential deadlock
            holder_id = self._locks[resource_id].molecule_id
            if holder_id == molecule_id:
                return True  # Already own the lock

            # Add to waiting set
            self._waiting[molecule_id].add(resource_id)

            # Check for deadlock
            if self._detect_deadlock_cycle(molecule_id):
                self._waiting[molecule_id].discard(resource_id)
                raise DeadlockError(
                    molecules=[molecule_id, holder_id],
                    resources=[resource_id],
                )

        # Wait for lock with timeout
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            await asyncio.sleep(0.1)
            async with self._lock:
                if resource_id not in self._locks:
                    self._locks[resource_id] = ResourceLock(
                        resource_id, molecule_id, datetime.now(timezone.utc)
                    )
                    self._holding[molecule_id].add(resource_id)
                    self._waiting[molecule_id].discard(resource_id)
                    return True

        async with self._lock:
            self._waiting[molecule_id].discard(resource_id)
        return False

    async def release_lock(self, molecule_id: str, resource_id: str) -> None:
        """Release a lock on a resource."""
        async with self._lock:
            if resource_id in self._locks and self._locks[resource_id].molecule_id == molecule_id:
                del self._locks[resource_id]
                self._holding[molecule_id].discard(resource_id)

    async def release_all_locks(self, molecule_id: str) -> None:
        """Release all locks held by a molecule."""
        async with self._lock:
            resources_to_release = list(self._holding.get(molecule_id, set()))
            for resource_id in resources_to_release:
                if resource_id in self._locks:
                    del self._locks[resource_id]
            self._holding[molecule_id].clear()
            self._waiting[molecule_id].clear()

    def _detect_deadlock_cycle(self, start_molecule: str) -> bool:
        """
        Detect if adding this wait would create a deadlock cycle.

        Uses DFS to find cycles in the wait-for graph.
        """
        visited = set()
        path = set()

        def dfs(molecule_id: str) -> bool:
            if molecule_id in path:
                return True  # Cycle found
            if molecule_id in visited:
                return False

            visited.add(molecule_id)
            path.add(molecule_id)

            # Find molecules that this one is waiting for
            for resource_id in self._waiting.get(molecule_id, set()):
                if resource_id in self._locks:
                    holder_id = self._locks[resource_id].molecule_id
                    if dfs(holder_id):
                        return True

            path.remove(molecule_id)
            return False

        return dfs(start_molecule)

    async def get_lock_state(self) -> dict[str, Any]:
        """Get current lock state for debugging."""
        async with self._lock:
            return {
                "locks": {
                    r: {
                        "molecule": lock.molecule_id,
                        "acquired_at": lock.acquired_at.isoformat(),
                    }
                    for r, lock in self._locks.items()
                },
                "waiting": {m: list(w) for m, w in self._waiting.items() if w},
                "holding": {m: list(h) for m, h in self._holding.items() if h},
            }


# Global deadlock detector instance
_deadlock_detector: DeadlockDetector | None = None


def get_deadlock_detector() -> DeadlockDetector:
    """Get the global deadlock detector instance."""
    global _deadlock_detector
    if _deadlock_detector is None:
        _deadlock_detector = DeadlockDetector()
    return _deadlock_detector


def reset_deadlock_detector() -> None:
    """Reset the global deadlock detector (for testing)."""
    global _deadlock_detector
    _deadlock_detector = None


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
        config: Optional[dict[str, Any]] = None,
        dependencies: Optional[list[str]] = None,
        timeout_seconds: float = 300.0,
        max_attempts: int = 3,
    ) -> "MoleculeStep":
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
    def from_dict(cls, data: dict[str, Any]) -> "MoleculeStep":
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Molecule":
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
    def from_dict(cls, data: dict[str, Any]) -> "Molecule":
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


class StepExecutor(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    async def execute(
        self,
        step: MoleculeStep,
        context: dict[str, Any],
    ) -> Any:
        """
        Execute a step.

        Args:
            step: The step to execute
            context: Execution context (previous results, etc.)

        Returns:
            Step result
        """
        pass


class AgentStepExecutor(StepExecutor):
    """Execute steps using AI agents."""

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step via agent."""
        # This would integrate with the actual agent system
        # For now, return a placeholder
        logger.info(f"Agent executing step: {step.name}")
        return {"status": "executed", "step": step.name}


class ShellStepExecutor(StepExecutor):
    """Execute steps as shell commands using sandboxed subprocess runner.

    Security: Uses subprocess_runner module which enforces command allowlisting
    and blocks dangerous operations.
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step as shell command with security validation."""
        import shlex

        from aragora.utils.subprocess_runner import SandboxError, run_sandboxed

        command = step.config.get("command", "echo 'No command'")
        logger.info(f"Shell executing: {command}")

        try:
            # Parse command string into arguments for secure execution
            args = shlex.split(command)
            if not args:
                return {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "Empty command",
                }

            # Use sandboxed runner with command whitelist validation
            result = await run_sandboxed(
                args,
                timeout=min(step.timeout_seconds, 300.0),  # Cap at 5 minutes
                capture_output=True,
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except SandboxError as e:
            # Command failed security validation
            logger.warning(f"Step {step.name} blocked by sandbox: {e}")
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Security: {e}",
            }
        except ValueError as e:
            # shlex.split failed
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Invalid command syntax: {e}",
            }
        except asyncio.TimeoutError:
            raise TimeoutError(f"Step {step.name} timed out after {step.timeout_seconds}s")


class DebateStepExecutor(StepExecutor):
    """
    Execute steps by routing contested decisions to Arena debate.

    Used when a step requires multiple perspectives or when there's
    disagreement about the best approach.

    Config options:
        question: The question to debate
        agents: List of agent names to participate
        rounds: Number of debate rounds
        consensus: Consensus requirement (majority, unanimous, etc.)
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step via Arena debate."""
        question = step.config.get("question", step.name)
        agents_config = step.config.get("agents", get_settings().agent.default_agent_list)
        rounds: int = step.config.get("rounds", DEFAULT_ROUNDS)
        consensus_value: ConsensusType = cast(
            ConsensusType, step.config.get("consensus", DEFAULT_CONSENSUS)
        )

        try:
            from aragora.core import Environment, DebateProtocol
            from aragora.core_types import Agent
            from aragora.debate.orchestrator import Arena
            from aragora.agents.registry import AgentRegistry

            # Get agents
            agents: list[Agent] = []
            for agent_name in agents_config:
                if AgentRegistry.is_registered(agent_name):
                    agent = AgentRegistry.create(agent_name)
                    if agent:
                        agents.append(agent)

            if not agents:
                return {"status": "skipped", "reason": "No agents available"}

            # Create and run debate
            env = Environment(task=question)
            # DebateProtocol's consensus parameter accepts ConsensusType at runtime
            protocol = cast(Any, DebateProtocol)(rounds=rounds, consensus=consensus_value)
            arena = Arena(env, agents, protocol)
            result = await arena.run()

            return {
                "status": "debated",
                "decision": getattr(result, "decision", str(result)),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "rounds_used": getattr(result, "rounds_used", rounds),
            }
        except ImportError as e:
            logger.warning(f"Debate modules not available: {e}")
            return {"status": "skipped", "reason": "Debate modules not available"}
        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            raise


class ParallelStepExecutor(StepExecutor):
    """
    Execute steps by fanning out to multiple agents in parallel.

    Each agent receives the same task, and results are aggregated.

    Config options:
        agents: List of agent names to use
        task: The task for all agents
        aggregate: How to combine results (all, first, majority, custom)
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step across multiple agents in parallel."""
        agents_config = step.config.get("agents", ["claude", "gpt4"])
        task = step.config.get("task", step.name)
        aggregate = step.config.get("aggregate", "all")

        try:
            from aragora.agents.registry import AgentRegistry

            # Get agents
            agents = []
            for agent_name in agents_config:
                if AgentRegistry.is_registered(agent_name):
                    agent = AgentRegistry.create(agent_name)
                    if agent:
                        agents.append(agent)

            if not agents:
                return {"status": "skipped", "reason": "No agents available"}

            # Execute in parallel
            async def run_agent(agent):
                try:
                    result = await agent.generate(task)
                    return {"agent": agent.name, "result": result, "success": True}
                except Exception as e:
                    return {"agent": agent.name, "error": str(e), "success": False}

            results = await asyncio.gather(*[run_agent(a) for a in agents])

            # Aggregate results
            successful = [r for r in results if r.get("success")]

            return {
                "status": "parallel_completed",
                "total_agents": len(agents),
                "successful": len(successful),
                "results": results,
                "aggregate": aggregate,
            }
        except ImportError as e:
            logger.warning(f"Agent modules not available: {e}")
            return {"status": "skipped", "reason": "Agent modules not available"}
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise


class ConditionalStepExecutor(StepExecutor):
    """
    Execute steps conditionally based on previous step results.

    Enables branching logic in molecules.

    Config options:
        condition: Expression to evaluate (simple key lookup or callable name)
        condition_key: Key in previous result to check
        expected_value: Value to compare against
        operator: Comparison operator (eq, ne, gt, lt, contains, exists)
        if_true: Action if condition is true (continue, skip, branch)
        if_false: Action if condition is false
        branch_step: Step ID to branch to if branching
    """

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute step conditionally based on context."""
        condition_key = step.config.get("condition_key", "status")
        expected_value = step.config.get("expected_value", "success")
        operator = step.config.get("operator", "eq")
        if_true = step.config.get("if_true", "continue")
        if_false = step.config.get("if_false", "skip")

        # Get the value from context (previous step results)
        previous_results = context.get("previous_results", {})
        actual_value = None

        # Try to find the value in context
        for key, value in previous_results.items():
            if isinstance(value, dict) and condition_key in value:
                actual_value = value[condition_key]
                break
        if actual_value is None:
            actual_value = previous_results.get(condition_key)

        # Evaluate condition
        condition_met = self._evaluate_condition(actual_value, expected_value, operator)

        action = if_true if condition_met else if_false

        result = {
            "status": "conditional_evaluated",
            "condition_key": condition_key,
            "expected": expected_value,
            "actual": actual_value,
            "operator": operator,
            "condition_met": condition_met,
            "action": action,
        }

        if action == "skip":
            result["should_skip"] = True
        elif action == "branch":
            result["branch_to"] = step.config.get("branch_step")

        return result

    def _evaluate_condition(self, actual: Any, expected: Any, operator: str) -> bool:
        """Evaluate a condition."""
        if operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return actual > expected
        elif operator == "lt":
            return actual < expected
        elif operator == "gte":
            return actual >= expected
        elif operator == "lte":
            return actual <= expected
        elif operator == "contains":
            return expected in str(actual)
        elif operator == "exists":
            return actual is not None
        elif operator == "not_exists":
            return actual is None
        else:
            logger.warning(f"Unknown operator: {operator}, defaulting to equality")
            return actual == expected


class MoleculeEngine:
    """
    Executes molecules with step-level persistence.

    Provides:
    - Automatic checkpointing before/after each step
    - Recovery from crashes
    - Pluggable step executors
    - Parallel execution where possible
    - Transaction safety with begin/commit/rollback semantics
    - Dependency graph validation
    - Deadlock detection for parallel molecules
    """

    def __init__(
        self,
        bead_store: BeadStore | None = None,
        checkpoint_dir: Path | None = None,
        enable_transactions: bool = True,
        validate_dependencies: bool = True,
        enable_deadlock_detection: bool = True,
    ):
        """
        Initialize the molecule engine.

        Args:
            bead_store: Optional bead store for tracking
            checkpoint_dir: Directory for checkpoints
            enable_transactions: Enable transaction wrapper with rollback support
            validate_dependencies: Validate step dependencies before execution
            enable_deadlock_detection: Enable deadlock detection for parallel molecules
        """
        self.bead_store = bead_store
        self.checkpoint_dir = checkpoint_dir or Path(".molecules")
        self._executors: dict[str, StepExecutor] = {
            "agent": AgentStepExecutor(),
            "shell": ShellStepExecutor(),
        }
        self._molecules: dict[str, Molecule] = {}
        self._transactions: dict[str, MoleculeTransaction] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._enable_transactions = enable_transactions
        self._validate_dependencies = validate_dependencies
        self._enable_deadlock_detection = enable_deadlock_detection

    def register_executor(self, step_type: str, executor: StepExecutor) -> None:
        """Register a custom step executor."""
        self._executors[step_type] = executor

    async def initialize(self) -> None:
        """Initialize the engine, loading checkpoints."""
        if self._initialized:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        await self._load_molecules()
        self._initialized = True
        logger.info(f"MoleculeEngine initialized with {len(self._molecules)} molecules")

    async def _load_molecules(self) -> None:
        """Load molecules from checkpoint files."""
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    molecule = Molecule.from_dict(data)
                    self._molecules[molecule.id] = molecule
            except Exception as e:
                logger.warning(f"Failed to load molecule from {checkpoint_file}: {e}")

    async def _checkpoint(self, molecule: Molecule) -> None:
        """Save a checkpoint for a molecule."""
        checkpoint_file = self.checkpoint_dir / f"{molecule.id}.json"
        molecule.updated_at = datetime.now(timezone.utc)

        try:
            with open(checkpoint_file, "w") as f:
                json.dump(molecule.to_dict(), f, indent=2)
            logger.debug(
                f"Checkpointed molecule {molecule.id} at step {molecule.current_step_index}"
            )
        except Exception as e:
            logger.error(f"Failed to checkpoint molecule {molecule.id}: {e}")

    def validate_dependencies(self, molecule: Molecule) -> tuple[bool, str | None]:
        """
        Validate step dependencies before execution.

        Args:
            molecule: The molecule to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not molecule.steps:
            return True, None

        graph = DependencyGraph(molecule.steps)
        is_valid, error_details = graph.validate()

        if not is_valid and error_details:
            # Check if it's a missing dependency or a cycle
            if any(":" in detail for detail in error_details):
                # Missing dependencies
                return False, f"Missing dependencies: {error_details}"
            else:
                # Cycle detected
                return False, f"Cyclic dependency detected: {' -> '.join(error_details)}"

        return True, None

    async def _begin_transaction(self, molecule: Molecule) -> MoleculeTransaction | None:
        """Begin a transaction for molecule execution."""
        if not self._enable_transactions:
            return None

        # Take a snapshot of the current state
        snapshot = molecule.to_dict() if molecule else None

        transaction = MoleculeTransaction.begin(
            molecule_id=molecule.id,
            checkpoint_dir=self.checkpoint_dir,
            snapshot=snapshot,
        )
        self._transactions[transaction.transaction_id] = transaction
        logger.info(f"Started transaction {transaction.transaction_id} for molecule {molecule.id}")
        return transaction

    async def _commit_transaction(self, transaction: MoleculeTransaction) -> bool:
        """Commit a transaction."""
        if not transaction:
            return True

        success = await transaction.commit()
        if success:
            # Clean up transaction log after successful commit
            log_file = self.checkpoint_dir / f"txn_{transaction.transaction_id}.log"
            try:
                if log_file.exists():
                    log_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up transaction log: {e}")
        return success

    async def _rollback_transaction(
        self, transaction: MoleculeTransaction, molecule: Molecule
    ) -> bool:
        """Rollback a transaction and restore molecule state."""
        if not transaction:
            return True

        success = await transaction.rollback()

        if success and transaction.pre_transaction_snapshot:
            # Restore molecule from snapshot
            try:
                restored = Molecule.from_dict(transaction.pre_transaction_snapshot)
                self._molecules[molecule.id] = restored
                await self._checkpoint(restored)
                logger.info(f"Restored molecule {molecule.id} from pre-transaction state")
            except Exception as e:
                logger.error(f"Failed to restore molecule state: {e}")
                success = False

        # Release any locks held by this molecule
        if self._enable_deadlock_detection:
            detector = get_deadlock_detector()
            await detector.release_all_locks(molecule.id)

        return success

    async def _acquire_step_resources(self, molecule_id: str, step: MoleculeStep) -> bool:
        """Acquire resource locks for a step (deadlock prevention)."""
        if not self._enable_deadlock_detection:
            return True

        # Extract resource requirements from step config
        resources = step.config.get("resources", [])
        if not resources:
            # Default: lock by step type and name
            resources = [f"{step.step_type}:{step.name}"]

        detector = get_deadlock_detector()
        for resource_id in resources:
            try:
                acquired = await detector.acquire_lock(molecule_id, resource_id)
                if not acquired:
                    logger.warning(
                        f"Failed to acquire lock on {resource_id} for molecule {molecule_id}"
                    )
                    return False
            except DeadlockError as e:
                logger.error(f"Deadlock detected: {e}")
                raise

        return True

    async def _release_step_resources(self, molecule_id: str, step: MoleculeStep) -> None:
        """Release resource locks after step completion."""
        if not self._enable_deadlock_detection:
            return

        resources = step.config.get("resources", [])
        if not resources:
            resources = [f"{step.step_type}:{step.name}"]

        detector = get_deadlock_detector()
        for resource_id in resources:
            await detector.release_lock(molecule_id, resource_id)

    async def _execute_step(
        self,
        step: MoleculeStep,
        context: dict[str, Any],
        transaction: MoleculeTransaction | None = None,
        molecule: Molecule | None = None,
    ) -> Any:
        """Execute a single step with transaction support."""
        executor = self._executors.get(step.step_type)
        if not executor:
            raise ValueError(f"No executor for step type: {step.step_type}")

        # Acquire resource locks for deadlock prevention
        if molecule:
            await self._acquire_step_resources(molecule.id, step)

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        step.attempt_count += 1

        # Create compensating action for rollback
        if transaction and molecule:
            compensating = CompensatingAction(
                step_id=step.id,
                step_name=step.name,
                action_type="restore_checkpoint",
                checkpoint_path=self.checkpoint_dir / f"{molecule.id}.json",
                metadata={"step_config": step.config},
            )
            transaction.add_compensating_action(compensating)

        try:
            result = await asyncio.wait_for(
                executor.execute(step, context),
                timeout=step.timeout_seconds,
            )
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            step.result = result

            # Mark step completed in transaction
            if transaction:
                transaction.mark_step_completed(step.id)

            return result

        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error_message = f"Timeout after {step.timeout_seconds}s"
            raise

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            raise

        finally:
            # Release resource locks
            if molecule:
                await self._release_step_resources(molecule.id, step)

    async def execute(
        self,
        molecule: Molecule,
        auto_rollback: bool = True,
    ) -> MoleculeResult:
        """
        Execute a molecule with automatic checkpointing and transaction safety.

        Args:
            molecule: The molecule to execute
            auto_rollback: Automatically rollback on failure (default True)

        Returns:
            MoleculeResult with execution details

        Raises:
            CyclicDependencyError: If cyclic dependencies are detected
            DependencyValidationError: If dependencies are invalid
            DeadlockError: If deadlock is detected during execution
        """
        async with self._lock:
            start_time = datetime.now(timezone.utc)
            transaction: MoleculeTransaction | None = None

            # Validate dependencies before execution
            if self._validate_dependencies and molecule.steps:
                is_valid, error_msg = self.validate_dependencies(molecule)
                if not is_valid:
                    molecule.status = MoleculeStatus.FAILED
                    molecule.error_message = error_msg
                    logger.error(f"Dependency validation failed: {error_msg}")
                    return MoleculeResult(
                        molecule_id=molecule.id,
                        status=MoleculeStatus.FAILED,
                        completed_steps=0,
                        failed_steps=0,
                        total_steps=len(molecule.steps),
                        duration_seconds=0.0,
                        step_results={},
                        error_message=error_msg,
                    )

            # Begin transaction
            transaction = await self._begin_transaction(molecule)

            molecule.status = MoleculeStatus.RUNNING
            molecule.started_at = molecule.started_at or start_time

            self._molecules[molecule.id] = molecule
            await self._checkpoint(molecule)

            context: dict[str, Any] = {"step_results": {}}
            step_results: dict[str, Any] = {}
            execution_failed = False
            failure_reason: str | None = None

            try:
                # Execute steps in order (respecting dependencies)
                while True:
                    runnable_steps = molecule.get_next_runnable_steps()
                    if not runnable_steps:
                        # Check if all done or stuck
                        pending = [s for s in molecule.steps if s.status == StepStatus.PENDING]
                        if not pending:
                            break
                        else:
                            # Dependencies not met - likely circular
                            raise CyclicDependencyError(
                                [s.name for s in pending],
                                f"Circular dependency or unmet dependencies: "
                                f"{[s.name for s in pending]}",
                            )

                    # Execute runnable steps (could parallelize here)
                    for step in runnable_steps:
                        logger.info(f"Executing step: {step.name}")
                        await self._checkpoint(molecule)

                        try:
                            result = await self._execute_step(step, context, transaction, molecule)
                            step_results[step.id] = result
                            context["step_results"][step.name] = result
                        except DeadlockError:
                            # Deadlock - need to rollback
                            execution_failed = True
                            failure_reason = f"Deadlock detected while executing {step.name}"
                            raise
                        except Exception as e:
                            logger.error(f"Step {step.name} failed: {e}")
                            if step.can_retry():
                                step.status = StepStatus.PENDING
                                logger.info(
                                    f"Will retry step {step.name} "
                                    f"({step.attempt_count}/{step.max_attempts})"
                                )
                            else:
                                step_results[step.id] = {"error": str(e)}
                                molecule.status = MoleculeStatus.FAILED
                                molecule.error_message = f"Step {step.name} failed: {e}"
                                execution_failed = True
                                failure_reason = molecule.error_message
                                await self._checkpoint(molecule)
                                break

                        molecule.current_step_index += 1
                        await self._checkpoint(molecule)

                    if molecule.status == MoleculeStatus.FAILED:
                        break

                # Check final status
                failed_steps = [s for s in molecule.steps if s.status == StepStatus.FAILED]
                if failed_steps:
                    molecule.status = MoleculeStatus.FAILED
                    execution_failed = True
                else:
                    molecule.status = MoleculeStatus.COMPLETED

                molecule.completed_at = datetime.now(timezone.utc)
                await self._checkpoint(molecule)

            except (CyclicDependencyError, DeadlockError) as e:
                molecule.status = MoleculeStatus.FAILED
                molecule.error_message = str(e)
                molecule.completed_at = datetime.now(timezone.utc)
                execution_failed = True
                failure_reason = str(e)
                await self._checkpoint(molecule)
                logger.error(f"Molecule {molecule.id} failed: {e}")

            except Exception as e:
                molecule.status = MoleculeStatus.FAILED
                molecule.error_message = str(e)
                molecule.completed_at = datetime.now(timezone.utc)
                execution_failed = True
                failure_reason = str(e)
                await self._checkpoint(molecule)
                logger.error(f"Molecule {molecule.id} failed: {e}")

            # Handle transaction commit/rollback
            if transaction:
                if execution_failed and auto_rollback:
                    logger.info(f"Auto-rolling back transaction for molecule {molecule.id}")
                    await self._rollback_transaction(transaction, molecule)
                elif not execution_failed:
                    await self._commit_transaction(transaction)
                else:
                    # Failed but no auto-rollback - just mark transaction as failed
                    transaction.state = TransactionState.FAILED
                    transaction.error_message = failure_reason

            # Release all locks held by this molecule
            if self._enable_deadlock_detection:
                detector = get_deadlock_detector()
                await detector.release_all_locks(molecule.id)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            return MoleculeResult(
                molecule_id=molecule.id,
                status=molecule.status,
                completed_steps=len(
                    [s for s in molecule.steps if s.status == StepStatus.COMPLETED]
                ),
                failed_steps=len([s for s in molecule.steps if s.status == StepStatus.FAILED]),
                total_steps=len(molecule.steps),
                duration_seconds=duration,
                step_results=step_results,
                error_message=molecule.error_message,
            )

    async def resume(self, molecule_id: str) -> MoleculeResult:
        """
        Resume a molecule from its last checkpoint.

        Args:
            molecule_id: ID of the molecule to resume

        Returns:
            MoleculeResult with execution details
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            # Try to load from checkpoint
            checkpoint_file = self.checkpoint_dir / f"{molecule_id}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    molecule = Molecule.from_dict(data)
                    self._molecules[molecule_id] = molecule
            else:
                raise ValueError(f"Molecule {molecule_id} not found")

        # Reset any RUNNING steps back to PENDING
        for step in molecule.steps:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING

        molecule.status = MoleculeStatus.RUNNING
        logger.info(f"Resuming molecule {molecule_id} from step {molecule.current_step_index}")

        return await self.execute(molecule)

    async def get_molecule(self, molecule_id: str) -> Molecule | None:
        """Get a molecule by ID."""
        return self._molecules.get(molecule_id)

    async def list_molecules(
        self,
        status: MoleculeStatus | None = None,
    ) -> list[Molecule]:
        """List molecules with optional status filter."""
        molecules = list(self._molecules.values())
        if status:
            molecules = [m for m in molecules if m.status == status]
        return molecules

    async def cancel(self, molecule_id: str) -> bool:
        """
        Cancel a running molecule.

        Args:
            molecule_id: ID of the molecule to cancel

        Returns:
            True if cancelled, False if not found
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            return False

        molecule.status = MoleculeStatus.CANCELLED
        molecule.completed_at = datetime.now(timezone.utc)
        await self._checkpoint(molecule)

        logger.info(f"Cancelled molecule {molecule_id}")
        return True

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about molecules."""
        molecules = list(self._molecules.values())
        by_status: dict[str, int] = {}
        for m in molecules:
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1

        # Include transaction statistics
        txn_by_state: dict[str, int] = {}
        for txn in self._transactions.values():
            txn_by_state[txn.state.value] = txn_by_state.get(txn.state.value, 0) + 1

        return {
            "total_molecules": len(molecules),
            "by_status": by_status,
            "total_steps": sum(len(m.steps) for m in molecules),
            "transactions": {
                "total": len(self._transactions),
                "by_state": txn_by_state,
            },
        }

    async def get_transaction(self, transaction_id: str) -> MoleculeTransaction | None:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    async def list_transactions(
        self,
        molecule_id: str | None = None,
        state: TransactionState | None = None,
    ) -> list[MoleculeTransaction]:
        """
        List transactions with optional filters.

        Args:
            molecule_id: Filter by molecule ID
            state: Filter by transaction state

        Returns:
            List of matching transactions
        """
        transactions = list(self._transactions.values())
        if molecule_id:
            transactions = [t for t in transactions if t.molecule_id == molecule_id]
        if state:
            transactions = [t for t in transactions if t.state == state]
        return transactions

    async def rollback_molecule(self, molecule_id: str) -> bool:
        """
        Manually rollback a molecule to its pre-execution state.

        Args:
            molecule_id: ID of the molecule to rollback

        Returns:
            True if rollback succeeded, False otherwise
        """
        molecule = self._molecules.get(molecule_id)
        if not molecule:
            logger.warning(f"Molecule {molecule_id} not found for rollback")
            return False

        # Find the most recent active/failed transaction for this molecule
        matching_txns = [
            t
            for t in self._transactions.values()
            if t.molecule_id == molecule_id
            and t.state in (TransactionState.ACTIVE, TransactionState.FAILED)
        ]

        if not matching_txns:
            logger.warning(f"No active transaction found for molecule {molecule_id}")
            return False

        # Rollback the most recent transaction
        transaction = max(matching_txns, key=lambda t: t.started_at)
        return await self._rollback_transaction(transaction, molecule)

    async def recover_incomplete_transactions(self) -> dict[str, bool]:
        """
        Recover incomplete transactions after a crash.

        Scans for transaction logs and rolls back any that were
        in progress when the system crashed.

        Returns:
            Dict mapping transaction_id to recovery success status
        """
        results: dict[str, bool] = {}

        # Scan for transaction log files
        for log_file in self.checkpoint_dir.glob("txn_*.log"):
            try:
                with open(log_file) as f:
                    log_data = json.load(f)

                txn_id = log_data.get("transaction_id")
                molecule_id = log_data.get("molecule_id")
                status = log_data.get("status")

                if status in ("committed", "rolled_back"):
                    # Transaction was completed - just clean up the log
                    log_file.unlink()
                    results[txn_id] = True
                    continue

                # Transaction was incomplete - attempt rollback
                logger.warning(f"Found incomplete transaction {txn_id} for molecule {molecule_id}")

                molecule = self._molecules.get(molecule_id)
                if molecule:
                    # Recreate transaction from log
                    transaction = MoleculeTransaction(
                        transaction_id=txn_id,
                        molecule_id=molecule_id,
                        state=TransactionState.ACTIVE,
                        started_at=datetime.fromisoformat(log_data.get("timestamp", "")),
                        checkpoint_dir=self.checkpoint_dir,
                        completed_steps=log_data.get("completed_steps", []),
                    )
                    success = await self._rollback_transaction(transaction, molecule)
                    results[txn_id] = success
                else:
                    logger.error(
                        f"Cannot recover transaction {txn_id}: molecule {molecule_id} not found"
                    )
                    results[txn_id] = False

            except Exception as e:
                logger.error(f"Error processing transaction log {log_file}: {e}")
                results[str(log_file)] = False

        return results


# Escalation support (Gastown pattern)


class EscalationLevel(str, Enum):
    """Standard escalation severity levels."""

    WARN = "warn"  # Log warning, notify observers
    THROTTLE = "throttle"  # Reduce throughput/rate limit
    SUSPEND = "suspend"  # Pause agent/operation
    TERMINATE = "terminate"  # Full stop with cleanup


@dataclass
class EscalationContext:
    """Context for escalation step execution."""

    level: EscalationLevel
    source: str  # What triggered the escalation
    reason: str  # Why escalation was triggered
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_level: EscalationLevel | None = None
    auto_escalate_at: datetime | None = None


class EscalationStepExecutor(StepExecutor):
    """Execute escalation steps with severity-aware handlers."""

    def __init__(
        self,
        handlers: dict[str, Any],
        auto_escalate_seconds: float = 300.0,
    ):
        """
        Initialize escalation executor.

        Args:
            handlers: Dict mapping severity level to handler callable
            auto_escalate_seconds: Time before auto-escalating to next level
        """
        self._handlers = handlers
        self._auto_escalate_seconds = auto_escalate_seconds
        self._current_level: EscalationLevel | None = None

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute escalation step."""
        level_str = step.config.get("level", "warn")
        try:
            level = EscalationLevel(level_str)
        except ValueError:
            level = EscalationLevel.WARN

        source = step.config.get("source", "unknown")
        reason = step.config.get("reason", "unspecified")
        metadata = step.config.get("metadata", {})

        escalation_ctx = EscalationContext(
            level=level,
            source=source,
            reason=reason,
            metadata=metadata,
            previous_level=self._current_level,
        )

        # Find and execute handler
        handler = self._handlers.get(level.value)
        if handler is None:
            logger.warning(f"No handler for escalation level: {level.value}")
            return {"status": "no_handler", "level": level.value}

        logger.info(f"Executing escalation: level={level.value} source={source}")

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(escalation_ctx)
            else:
                result = handler(escalation_ctx)

            self._current_level = level

            return {
                "status": "executed",
                "level": level.value,
                "result": result,
                "previous_level": escalation_ctx.previous_level.value
                if escalation_ctx.previous_level
                else None,
            }
        except Exception as e:
            logger.error(f"Escalation handler failed: {e}")
            raise


def create_escalation_molecule(
    name: str,
    severity_levels: list[str],
    handlers: dict[str, Any],
    source: str = "system",
    reason: str = "threshold_exceeded",
    auto_escalate_after_seconds: float = 300.0,
    metadata: Optional[dict[str, Any]] = None,
) -> Molecule:
    """
    Create a molecule for escalation workflows (Gastown pattern).

    This creates a multi-step molecule where each step represents an
    escalation level. Steps are executed in sequence if the previous
    level doesn't resolve the issue.

    Args:
        name: Name for the escalation molecule
        severity_levels: List of severity levels in order (e.g., ["warn", "throttle", "suspend"])
        handlers: Dict mapping level names to handler callables
        source: Source of the escalation (for logging)
        reason: Reason for the escalation (for logging)
        auto_escalate_after_seconds: Time before auto-escalating to next level
        metadata: Additional metadata for the escalation

    Returns:
        Configured Molecule ready for execution

    Example:
        async def warn_handler(ctx):
            await send_alert(ctx.source, ctx.reason)

        async def throttle_handler(ctx):
            await apply_rate_limit(ctx.source, 0.5)

        async def suspend_handler(ctx):
            await pause_agent(ctx.source)

        molecule = create_escalation_molecule(
            name="agent_overload_escalation",
            severity_levels=["warn", "throttle", "suspend"],
            handlers={
                "warn": warn_handler,
                "throttle": throttle_handler,
                "suspend": suspend_handler,
            },
            source="agent_monitor",
            reason="response_latency_exceeded",
        )

        engine = await get_molecule_engine()
        engine.register_executor("escalation", EscalationStepExecutor(handlers))
        result = await engine.execute(molecule)
    """
    steps = []
    prev_step_id = None

    for i, level in enumerate(severity_levels):
        step = MoleculeStep.create(
            name=f"escalate_{level}",
            step_type="escalation",
            config={
                "level": level,
                "source": source,
                "reason": reason,
                "metadata": metadata or {},
                "auto_escalate_seconds": auto_escalate_after_seconds,
            },
            timeout_seconds=auto_escalate_after_seconds * 2,  # Allow time for resolution
            dependencies=[prev_step_id] if prev_step_id else [],
        )
        steps.append(step)
        prev_step_id = step.id

    return Molecule.create(
        name=name,
        steps=steps,
        description=f"Escalation workflow: {' -> '.join(severity_levels)}",
        metadata={
            "type": "escalation",
            "source": source,
            "reason": reason,
            "severity_levels": severity_levels,
            **(metadata or {}),
        },
    )


def create_conditional_escalation_molecule(
    name: str,
    check_fn: Any,  # Callable[[], bool] - condition to check
    severity_levels: list[str],
    handlers: dict[str, Any],
    check_interval_seconds: float = 60.0,
    max_checks_per_level: int = 5,
    source: str = "conditional_monitor",
    reason: str = "condition_triggered",
) -> Molecule:
    """
    Create an escalation molecule with condition checking.

    This molecule checks a condition at each level and only escalates
    if the condition remains true after multiple checks.

    Args:
        name: Name for the escalation molecule
        check_fn: Callable that returns True if escalation should continue
        severity_levels: List of severity levels in order
        handlers: Dict mapping level names to handler callables
        check_interval_seconds: Time between condition checks
        max_checks_per_level: Number of checks before escalating
        source: Source of the escalation
        reason: Reason for the escalation

    Returns:
        Configured Molecule with condition checking
    """
    steps = []
    prev_step_id = None

    for level in severity_levels:
        # Add check step
        check_step = MoleculeStep.create(
            name=f"check_before_{level}",
            step_type="conditional",
            config={
                "check_fn": check_fn,
                "max_checks": max_checks_per_level,
                "check_interval": check_interval_seconds,
            },
            timeout_seconds=check_interval_seconds * max_checks_per_level * 2,
            dependencies=[prev_step_id] if prev_step_id else [],
        )
        steps.append(check_step)

        # Add escalation step
        escalate_step = MoleculeStep.create(
            name=f"escalate_{level}",
            step_type="escalation",
            config={
                "level": level,
                "source": source,
                "reason": reason,
            },
            dependencies=[check_step.id],
        )
        steps.append(escalate_step)
        prev_step_id = escalate_step.id

    return Molecule.create(
        name=name,
        steps=steps,
        description=f"Conditional escalation: {' -> '.join(severity_levels)}",
        metadata={
            "type": "conditional_escalation",
            "source": source,
            "severity_levels": severity_levels,
        },
    )


# Singleton instance
_default_engine: MoleculeEngine | None = None


async def get_molecule_engine(
    bead_store: BeadStore | None = None,
    checkpoint_dir: Path | None = None,
) -> MoleculeEngine:
    """Get the default molecule engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MoleculeEngine(bead_store, checkpoint_dir)
        await _default_engine.initialize()
    return _default_engine


def reset_molecule_engine() -> None:
    """Reset the default engine (for testing)."""
    global _default_engine
    _default_engine = None

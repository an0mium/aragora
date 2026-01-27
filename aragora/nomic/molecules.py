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

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.nomic.beads import BeadStore

logger = logging.getLogger(__name__)


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
    config: Dict[str, Any]
    status: StepStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    max_attempts: int = 3
    timeout_seconds: float = 300.0
    bead_id: Optional[str] = None  # Associated bead for tracking
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        step_type: str,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "MoleculeStep":
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
    steps: List[MoleculeStep]
    current_step_index: int
    status: MoleculeStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    checkpoint_dir: Optional[Path] = None
    parent_id: Optional[str] = None  # For nested molecules
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        name: str,
        steps: List[MoleculeStep],
        description: str = "",
        checkpoint_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Molecule":
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

    def get_current_step(self) -> Optional[MoleculeStep]:
        """Get the current step to execute."""
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_completed_step_ids(self) -> set:
        """Get IDs of completed steps."""
        return {s.id for s in self.steps if s.status == StepStatus.COMPLETED}

    def get_next_runnable_steps(self) -> List[MoleculeStep]:
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
    step_results: Dict[str, Any]
    error_message: Optional[str] = None

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
        context: Dict[str, Any],
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

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
        """Execute step via agent."""
        # This would integrate with the actual agent system
        # For now, return a placeholder
        logger.info(f"Agent executing step: {step.name}")
        return {"status": "executed", "step": step.name}


class ShellStepExecutor(StepExecutor):
    """Execute steps as shell commands."""

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
        """Execute step as shell command."""
        command = step.config.get("command", "echo 'No command'")
        logger.info(f"Shell executing: {command}")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=step.timeout_seconds,
            )
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
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

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
        """Execute step via Arena debate."""
        question = step.config.get("question", step.name)
        agents_config = step.config.get("agents", ["claude", "gpt4"])
        rounds = step.config.get("rounds", 3)
        consensus = step.config.get("consensus", "majority")

        try:
            from aragora.core import Environment, DebateProtocol
            from aragora.debate.orchestrator import Arena
            from aragora.agents.cli_agents import get_agent

            # Get agents
            agents = []
            for agent_name in agents_config:
                agent = get_agent(agent_name)
                if agent:
                    agents.append(agent)

            if not agents:
                return {"status": "skipped", "reason": "No agents available"}

            # Create and run debate
            env = Environment(task=question)
            protocol = DebateProtocol(rounds=rounds, consensus=consensus)
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

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
        """Execute step across multiple agents in parallel."""
        agents_config = step.config.get("agents", ["claude", "gpt4"])
        task = step.config.get("task", step.name)
        aggregate = step.config.get("aggregate", "all")

        try:
            from aragora.agents.cli_agents import get_agent

            # Get agents
            agents = []
            for agent_name in agents_config:
                agent = get_agent(agent_name)
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

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
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
    """

    def __init__(
        self,
        bead_store: Optional[BeadStore] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the molecule engine.

        Args:
            bead_store: Optional bead store for tracking
            checkpoint_dir: Directory for checkpoints
        """
        self.bead_store = bead_store
        self.checkpoint_dir = checkpoint_dir or Path(".molecules")
        self._executors: Dict[str, StepExecutor] = {
            "agent": AgentStepExecutor(),
            "shell": ShellStepExecutor(),
        }
        self._molecules: Dict[str, Molecule] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

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

    async def _execute_step(
        self,
        step: MoleculeStep,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single step."""
        executor = self._executors.get(step.step_type)
        if not executor:
            raise ValueError(f"No executor for step type: {step.step_type}")

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        step.attempt_count += 1

        try:
            result = await asyncio.wait_for(
                executor.execute(step, context),
                timeout=step.timeout_seconds,
            )
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            step.result = result
            return result

        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error_message = f"Timeout after {step.timeout_seconds}s"
            raise

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            raise

    async def execute(self, molecule: Molecule) -> MoleculeResult:
        """
        Execute a molecule with automatic checkpointing.

        Args:
            molecule: The molecule to execute

        Returns:
            MoleculeResult with execution details
        """
        async with self._lock:
            start_time = datetime.now(timezone.utc)
            molecule.status = MoleculeStatus.RUNNING
            molecule.started_at = molecule.started_at or start_time

            self._molecules[molecule.id] = molecule
            await self._checkpoint(molecule)

            context: Dict[str, Any] = {"step_results": {}}
            step_results: Dict[str, Any] = {}

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
                            raise ValueError(
                                f"Circular dependency or unmet dependencies: "
                                f"{[s.name for s in pending]}"
                            )

                    # Execute runnable steps (could parallelize here)
                    for step in runnable_steps:
                        logger.info(f"Executing step: {step.name}")
                        await self._checkpoint(molecule)

                        try:
                            result = await self._execute_step(step, context)
                            step_results[step.id] = result
                            context["step_results"][step.name] = result
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
                else:
                    molecule.status = MoleculeStatus.COMPLETED

                molecule.completed_at = datetime.now(timezone.utc)
                await self._checkpoint(molecule)

            except Exception as e:
                molecule.status = MoleculeStatus.FAILED
                molecule.error_message = str(e)
                molecule.completed_at = datetime.now(timezone.utc)
                await self._checkpoint(molecule)
                logger.error(f"Molecule {molecule.id} failed: {e}")

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

    async def get_molecule(self, molecule_id: str) -> Optional[Molecule]:
        """Get a molecule by ID."""
        return self._molecules.get(molecule_id)

    async def list_molecules(
        self,
        status: Optional[MoleculeStatus] = None,
    ) -> List[Molecule]:
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

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about molecules."""
        molecules = list(self._molecules.values())
        by_status = {}
        for m in molecules:
            by_status[m.status.value] = by_status.get(m.status.value, 0) + 1

        return {
            "total_molecules": len(molecules),
            "by_status": by_status,
            "total_steps": sum(len(m.steps) for m in molecules),
        }


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_level: Optional[EscalationLevel] = None
    auto_escalate_at: Optional[datetime] = None


class EscalationStepExecutor(StepExecutor):
    """Execute escalation steps with severity-aware handlers."""

    def __init__(
        self,
        handlers: Dict[str, Any],
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
        self._current_level: Optional[EscalationLevel] = None

    async def execute(self, step: MoleculeStep, context: Dict[str, Any]) -> Any:
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
    severity_levels: List[str],
    handlers: Dict[str, Any],
    source: str = "system",
    reason: str = "threshold_exceeded",
    auto_escalate_after_seconds: float = 300.0,
    metadata: Optional[Dict[str, Any]] = None,
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
    severity_levels: List[str],
    handlers: Dict[str, Any],
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
_default_engine: Optional[MoleculeEngine] = None


async def get_molecule_engine(
    bead_store: Optional[BeadStore] = None,
    checkpoint_dir: Optional[Path] = None,
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

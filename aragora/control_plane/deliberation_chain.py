"""
Deliberation Chaining for the Aragora Control Plane.

Provides multi-step deliberation workflows where the output of one
deliberation can feed into subsequent deliberations. This enables
complex review pipelines like:
- Code review → Security audit → Architecture review
- Draft → Critique → Revise → Final approval
- Research → Synthesis → Validation

Example:
    >>> from aragora.control_plane.deliberation_chain import (
    ...     DeliberationChain,
    ...     DeliberationStage,
    ...     ChainExecutor,
    ... )
    >>>
    >>> # Define a code review chain
    >>> chain = DeliberationChain(
    ...     name="Code Review Pipeline",
    ...     stages=[
    ...         DeliberationStage(
    ...             id="initial_review",
    ...             topic_template="Review this code for correctness: {context.code}",
    ...             agents=["claude", "gpt-4"],
    ...             required_consensus=0.7,
    ...             timeout_seconds=120,
    ...             next_on_success="security_audit",
    ...         ),
    ...         DeliberationStage(
    ...             id="security_audit",
    ...             topic_template="Security audit of code. Previous review: {previous.output}",
    ...             agents=["claude-security", "gpt-4-security"],
    ...             required_consensus=0.8,
    ...             timeout_seconds=180,
    ...             next_on_success="architecture_review",
    ...             next_on_failure="revise",
    ...         ),
    ...         ...
    ...     ],
    ...     initial_context={"code": "def example(): ..."},
    ... )
    >>>
    >>> executor = ChainExecutor(coordinator)
    >>> result = await executor.execute(chain)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.control_plane.coordinator import ControlPlaneCoordinator
    from aragora.control_plane.deliberation import DeliberationManager

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ChainStatus(Enum):
    """Status of a deliberation chain execution."""

    PENDING = "pending"  # Chain created but not started
    RUNNING = "running"  # Chain is executing
    PAUSED = "paused"  # Chain execution paused
    COMPLETED = "completed"  # All stages completed successfully
    FAILED = "failed"  # A stage failed and chain stopped
    CANCELLED = "cancelled"  # Chain was cancelled
    TIMEOUT = "timeout"  # Chain exceeded overall timeout


class StageStatus(Enum):
    """Status of an individual stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # Stage was skipped due to routing
    TIMEOUT = "timeout"


class StageTransition(Enum):
    """Types of stage transitions."""

    SUCCESS = "success"  # Move to next_on_success
    FAILURE = "failure"  # Move to next_on_failure
    TIMEOUT = "timeout"  # Stage timed out
    ERROR = "error"  # Unexpected error


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DeliberationStage:
    """
    A single stage in a deliberation chain.

    Stages define what deliberation to run and how to route to the next stage
    based on the outcome.

    Attributes:
        id: Unique identifier for this stage within the chain
        topic_template: Template for the deliberation topic, supports placeholders:
            - {context.key}: Values from initial_context
            - {previous.output}: Output from previous stage
            - {previous.confidence}: Confidence from previous stage
            - {stage_id.output}: Output from a specific stage
        agents: List of agents to participate in this stage
        required_consensus: Minimum consensus confidence for success (0.0-1.0)
        timeout_seconds: Maximum time for this stage
        next_on_success: Stage ID to execute on success, None means end chain
        next_on_failure: Stage ID to execute on failure, None means end chain
        max_rounds: Maximum debate rounds for this stage
        min_agents: Minimum agents required
        metadata: Additional stage metadata
        retry_count: Number of retries on failure before moving to next_on_failure
        retry_delay_seconds: Delay between retries
    """

    id: str
    topic_template: str
    agents: list[str] = field(default_factory=list)
    required_consensus: float = 0.7
    timeout_seconds: int = 300
    next_on_success: str | None = None
    next_on_failure: str | None = None
    max_rounds: int = 5
    min_agents: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    retry_delay_seconds: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Serialize stage to dictionary."""
        return {
            "id": self.id,
            "topic_template": self.topic_template,
            "agents": self.agents,
            "required_consensus": self.required_consensus,
            "timeout_seconds": self.timeout_seconds,
            "next_on_success": self.next_on_success,
            "next_on_failure": self.next_on_failure,
            "max_rounds": self.max_rounds,
            "min_agents": self.min_agents,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeliberationStage:
        """Deserialize stage from dictionary."""
        return cls(
            id=data["id"],
            topic_template=data["topic_template"],
            agents=data.get("agents", []),
            required_consensus=data.get("required_consensus", 0.7),
            timeout_seconds=data.get("timeout_seconds", 300),
            next_on_success=data.get("next_on_success"),
            next_on_failure=data.get("next_on_failure"),
            max_rounds=data.get("max_rounds", 5),
            min_agents=data.get("min_agents", 2),
            metadata=data.get("metadata", {}),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 5),
        )


@dataclass
class StageResult:
    """Result of executing a single stage."""

    stage_id: str
    status: StageStatus
    output: str | None = None
    confidence: float | None = None
    consensus_reached: bool = False
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    retries: int = 0
    transition: StageTransition | None = None
    task_id: str | None = None  # Control plane task ID

    @property
    def duration_seconds(self) -> float | None:
        """Get duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "stage_id": self.stage_id,
            "status": self.status.value,
            "output": self.output,
            "confidence": self.confidence,
            "consensus_reached": self.consensus_reached,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "retries": self.retries,
            "transition": self.transition.value if self.transition else None,
            "task_id": self.task_id,
        }


@dataclass
class DeliberationChain:
    """
    A multi-stage deliberation workflow.

    Chains define a series of deliberations where outputs can flow
    from one stage to the next. Conditional routing allows different
    paths based on success or failure of each stage.

    Attributes:
        id: Unique identifier for the chain
        name: Human-readable name
        description: Description of what this chain does
        stages: List of stages in execution order
        initial_context: Initial context values available to all stages
        entry_stage_id: ID of the first stage to execute (default: first in list)
        overall_timeout_seconds: Maximum time for entire chain
        metadata: Additional chain metadata
        created_at: When the chain was created
        created_by: Who created the chain
        tags: Tags for organization
    """

    name: str
    stages: list[DeliberationStage]
    initial_context: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    entry_stage_id: str | None = None
    overall_timeout_seconds: int = 1800  # 30 minutes default
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate chain configuration."""
        if not self.stages:
            raise ValueError("Chain must have at least one stage")

        # Set entry stage if not specified
        if self.entry_stage_id is None:
            self.entry_stage_id = self.stages[0].id

        # Validate stage references
        stage_ids = {s.id for s in self.stages}
        for stage in self.stages:
            if stage.next_on_success and stage.next_on_success not in stage_ids:
                raise ValueError(
                    f"Stage {stage.id}: next_on_success '{stage.next_on_success}' not found"
                )
            if stage.next_on_failure and stage.next_on_failure not in stage_ids:
                raise ValueError(
                    f"Stage {stage.id}: next_on_failure '{stage.next_on_failure}' not found"
                )

    def get_stage(self, stage_id: str) -> DeliberationStage | None:
        """Get a stage by ID."""
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize chain to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
            "initial_context": self.initial_context,
            "entry_stage_id": self.entry_stage_id,
            "overall_timeout_seconds": self.overall_timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeliberationChain:
        """Deserialize chain from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            stages=[DeliberationStage.from_dict(s) for s in data["stages"]],
            initial_context=data.get("initial_context", {}),
            entry_stage_id=data.get("entry_stage_id"),
            overall_timeout_seconds=data.get("overall_timeout_seconds", 1800),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            created_by=data.get("created_by"),
            tags=data.get("tags", []),
        )


@dataclass
class ChainExecution:
    """
    Execution state of a deliberation chain.

    Tracks the progress of a chain execution including all stage results.
    """

    chain: DeliberationChain
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ChainStatus = ChainStatus.PENDING
    current_stage_id: str | None = None
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)  # Accumulated context

    @property
    def duration_seconds(self) -> float | None:
        """Get total duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def get_last_result(self) -> StageResult | None:
        """Get the most recent stage result."""
        if not self.stage_results:
            return None
        # Return the result with the latest completed_at
        results = list(self.stage_results.values())
        completed = [r for r in results if r.completed_at]
        if not completed:
            return results[-1] if results else None
        return max(completed, key=lambda r: r.completed_at or 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution to dictionary."""
        return {
            "execution_id": self.execution_id,
            "chain_id": self.chain.id,
            "chain_name": self.chain.name,
            "status": self.status.value,
            "current_stage_id": self.current_stage_id,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "stages_completed": len(
                [r for r in self.stage_results.values() if r.status == StageStatus.SUCCESS]
            ),
            "stages_total": len(self.chain.stages),
        }


# =============================================================================
# Template Engine
# =============================================================================


class TemplateEngine:
    """
    Simple template engine for stage topic templates.

    Supports placeholders like:
    - {context.key}: Values from initial context
    - {previous.output}: Output from previous stage
    - {previous.confidence}: Confidence from previous stage
    - {stage_id.output}: Output from a specific stage
    - {stage_id.confidence}: Confidence from a specific stage
    """

    PLACEHOLDER_PATTERN = re.compile(r"\{([^}]+)\}")

    def __init__(
        self,
        initial_context: dict[str, Any],
        stage_results: dict[str, StageResult],
        previous_result: StageResult | None = None,
    ):
        self.initial_context = initial_context
        self.stage_results = stage_results
        self.previous_result = previous_result

    def render(self, template: str) -> str:
        """Render a template with the current context."""

        def replace(match: re.Match[str]) -> str:
            path = match.group(1)
            value = self._resolve_path(path)
            if value is None:
                return match.group(0)  # Keep original if not found
            return str(value)

        return self.PLACEHOLDER_PATTERN.sub(replace, template)

    def _resolve_path(self, path: str) -> Any:
        """Resolve a dotted path to a value."""
        parts = path.split(".")
        if not parts:
            return None

        source = parts[0]
        rest = parts[1:] if len(parts) > 1 else []

        # context.key
        if source == "context":
            return self._get_nested(self.initial_context, rest)

        # previous.output or previous.confidence
        if source == "previous" and self.previous_result:
            if not rest:
                return None
            attr = rest[0]
            if attr == "output":
                return self.previous_result.output
            if attr == "confidence":
                return self.previous_result.confidence
            return None

        # stage_id.output or stage_id.confidence
        if source in self.stage_results:
            result = self.stage_results[source]
            if not rest:
                return result.output
            attr = rest[0]
            if attr == "output":
                return result.output
            if attr == "confidence":
                return result.confidence
            return None

        return None

    def _get_nested(self, data: dict[str, Any], keys: list[str]) -> Any:
        """Get a nested value from a dictionary."""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
            if current is None:
                return None
        return current


# =============================================================================
# Chain Executor
# =============================================================================


class ChainExecutor:
    """
    Executes deliberation chains through the control plane.

    The executor manages the lifecycle of chain execution, including:
    - Stage transitions based on outcomes
    - Template rendering for stage topics
    - Timeout handling
    - Progress callbacks
    - Pause/resume support
    """

    def __init__(
        self,
        coordinator: Optional["ControlPlaneCoordinator"] = None,
        deliberation_manager: Optional["DeliberationManager"] = None,
        progress_callback: Callable[[ChainExecution], None] | None = None,
    ):
        """
        Initialize the chain executor.

        Args:
            coordinator: Control plane coordinator
            deliberation_manager: Deliberation manager for executing stages
            progress_callback: Optional callback called after each stage
        """
        self._coordinator = coordinator
        self._deliberation_manager = deliberation_manager
        self._progress_callback = progress_callback
        self._active_executions: dict[str, ChainExecution] = {}
        self._pause_flags: dict[str, asyncio.Event] = {}

    def set_coordinator(self, coordinator: "ControlPlaneCoordinator") -> None:
        """Set the coordinator after initialization."""
        self._coordinator = coordinator

    def set_deliberation_manager(self, manager: "DeliberationManager") -> None:
        """Set the deliberation manager after initialization."""
        self._deliberation_manager = manager

    async def execute(
        self,
        chain: DeliberationChain,
        context_overrides: dict[str, Any] | None = None,
    ) -> ChainExecution:
        """
        Execute a deliberation chain.

        Args:
            chain: The chain to execute
            context_overrides: Additional context values to merge

        Returns:
            ChainExecution with results
        """
        if not self._deliberation_manager:
            raise RuntimeError("Deliberation manager not set")

        # Create execution
        execution = ChainExecution(
            chain=chain,
            context={**chain.initial_context, **(context_overrides or {})},
        )
        execution.started_at = time.time()
        execution.status = ChainStatus.RUNNING
        execution.current_stage_id = chain.entry_stage_id

        # Track execution
        self._active_executions[execution.execution_id] = execution
        self._pause_flags[execution.execution_id] = asyncio.Event()
        self._pause_flags[execution.execution_id].set()  # Start unpaused

        logger.info(
            "chain_started",
            extra={
                "execution_id": execution.execution_id,
                "chain_id": chain.id,
                "chain_name": chain.name,
                "entry_stage": chain.entry_stage_id,
            },
        )

        try:
            # Execute stages until completion or failure
            await self._execute_chain_loop(execution)
        except asyncio.CancelledError:
            execution.status = ChainStatus.CANCELLED
            execution.error = "Chain execution was cancelled"
        except asyncio.TimeoutError:
            execution.status = ChainStatus.TIMEOUT
            execution.error = f"Chain exceeded overall timeout of {chain.overall_timeout_seconds}s"
        except Exception as e:
            execution.status = ChainStatus.FAILED
            execution.error = str(e)
            logger.exception(
                "chain_failed",
                extra={"execution_id": execution.execution_id, "error": str(e)},
            )
        finally:
            execution.completed_at = time.time()
            self._pause_flags.pop(execution.execution_id, None)

            logger.info(
                "chain_completed",
                extra={
                    "execution_id": execution.execution_id,
                    "status": execution.status.value,
                    "duration_seconds": execution.duration_seconds,
                    "stages_completed": len(
                        [
                            r
                            for r in execution.stage_results.values()
                            if r.status == StageStatus.SUCCESS
                        ]
                    ),
                },
            )

        return execution

    async def _execute_chain_loop(self, execution: ChainExecution) -> None:
        """Main execution loop for a chain."""
        chain = execution.chain
        timeout = chain.overall_timeout_seconds

        async with asyncio.timeout(timeout):
            while execution.current_stage_id and execution.status == ChainStatus.RUNNING:
                # Check for pause
                await self._pause_flags[execution.execution_id].wait()

                # Get current stage
                stage = chain.get_stage(execution.current_stage_id)
                if not stage:
                    execution.status = ChainStatus.FAILED
                    execution.error = f"Stage not found: {execution.current_stage_id}"
                    break

                # Execute stage
                result = await self._execute_stage(execution, stage)
                execution.stage_results[stage.id] = result

                # Call progress callback
                if self._progress_callback:
                    try:
                        self._progress_callback(execution)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

                # Determine next stage based on result
                next_stage_id = self._determine_next_stage(stage, result)

                if next_stage_id is None:
                    # Chain complete
                    if result.status == StageStatus.SUCCESS:
                        execution.status = ChainStatus.COMPLETED
                    else:
                        execution.status = ChainStatus.FAILED
                        execution.error = result.error or f"Stage {stage.id} failed"
                    break

                execution.current_stage_id = next_stage_id

    async def _execute_stage(
        self,
        execution: ChainExecution,
        stage: DeliberationStage,
    ) -> StageResult:
        """Execute a single stage."""
        result = StageResult(
            stage_id=stage.id,
            status=StageStatus.RUNNING,
            started_at=time.time(),
        )

        # Get previous result for template rendering
        previous_result = execution.get_last_result()

        # Render topic template
        template_engine = TemplateEngine(
            initial_context=execution.context,
            stage_results=execution.stage_results,
            previous_result=previous_result,
        )
        topic = template_engine.render(stage.topic_template)

        logger.info(
            "stage_started",
            extra={
                "execution_id": execution.execution_id,
                "stage_id": stage.id,
                "topic_preview": topic[:100],
            },
        )

        # Execute with retries
        retries = 0
        while retries <= stage.retry_count:
            try:
                # Submit deliberation through the manager
                task_id = await self._deliberation_manager.submit_deliberation(
                    question=topic,
                    context=None,  # Context is in the topic
                    agents=stage.agents if stage.agents else None,
                    priority="high" if retries > 0 else "normal",
                    timeout_seconds=float(stage.timeout_seconds),
                    max_rounds=stage.max_rounds,
                    min_agents=stage.min_agents,
                    consensus_required=True,
                    metadata={
                        "chain_id": execution.chain.id,
                        "execution_id": execution.execution_id,
                        "stage_id": stage.id,
                        "retry": retries,
                        **stage.metadata,
                    },
                )

                result.task_id = task_id

                # Wait for outcome
                outcome = await self._deliberation_manager.wait_for_outcome(
                    task_id,
                    timeout=float(stage.timeout_seconds) + 30,  # Buffer
                )

                if outcome is None:
                    raise TimeoutError("Deliberation did not complete")

                # Process outcome
                result.output = outcome.winning_position
                result.confidence = outcome.consensus_confidence
                result.consensus_reached = outcome.consensus_reached

                # Check if consensus meets requirement
                if (
                    outcome.success
                    and (outcome.consensus_confidence or 0) >= stage.required_consensus
                ):
                    result.status = StageStatus.SUCCESS
                    result.transition = StageTransition.SUCCESS
                else:
                    result.status = StageStatus.FAILED
                    result.transition = StageTransition.FAILURE
                    result.error = (
                        f"Consensus confidence {outcome.consensus_confidence:.2f} "
                        f"below required {stage.required_consensus:.2f}"
                    )

                # Update execution context with this stage's result
                execution.context[f"{stage.id}_output"] = result.output
                execution.context[f"{stage.id}_confidence"] = result.confidence

                break  # Success, exit retry loop

            except asyncio.TimeoutError:
                result.status = StageStatus.TIMEOUT
                result.transition = StageTransition.TIMEOUT
                result.error = f"Stage timed out after {stage.timeout_seconds}s"
                if retries < stage.retry_count:
                    retries += 1
                    result.retries = retries
                    await asyncio.sleep(stage.retry_delay_seconds)
                else:
                    break

            except Exception as e:
                result.status = StageStatus.FAILED
                result.transition = StageTransition.ERROR
                result.error = str(e)
                if retries < stage.retry_count:
                    retries += 1
                    result.retries = retries
                    await asyncio.sleep(stage.retry_delay_seconds)
                else:
                    break

        result.completed_at = time.time()

        logger.info(
            "stage_completed",
            extra={
                "execution_id": execution.execution_id,
                "stage_id": stage.id,
                "status": result.status.value,
                "confidence": result.confidence,
                "duration_seconds": result.duration_seconds,
            },
        )

        return result

    def _determine_next_stage(
        self,
        stage: DeliberationStage,
        result: StageResult,
    ) -> str | None:
        """Determine the next stage based on the result."""
        if result.status == StageStatus.SUCCESS:
            return stage.next_on_success
        else:
            return stage.next_on_failure

    def pause(self, execution_id: str) -> bool:
        """Pause a running chain execution."""
        if execution_id not in self._active_executions:
            return False

        execution = self._active_executions[execution_id]
        if execution.status != ChainStatus.RUNNING:
            return False

        self._pause_flags[execution_id].clear()
        execution.status = ChainStatus.PAUSED
        logger.info("chain_paused", extra={"execution_id": execution_id})
        return True

    def resume(self, execution_id: str) -> bool:
        """Resume a paused chain execution."""
        if execution_id not in self._active_executions:
            return False

        execution = self._active_executions[execution_id]
        if execution.status != ChainStatus.PAUSED:
            return False

        execution.status = ChainStatus.RUNNING
        self._pause_flags[execution_id].set()
        logger.info("chain_resumed", extra={"execution_id": execution_id})
        return True

    def cancel(self, execution_id: str) -> bool:
        """Cancel a chain execution."""
        if execution_id not in self._active_executions:
            return False

        execution = self._active_executions[execution_id]
        if execution.status in (ChainStatus.COMPLETED, ChainStatus.CANCELLED):
            return False

        execution.status = ChainStatus.CANCELLED
        # Clear pause flag to unblock if paused
        if execution_id in self._pause_flags:
            self._pause_flags[execution_id].set()

        logger.info("chain_cancelled", extra={"execution_id": execution_id})
        return True

    def get_execution(self, execution_id: str) -> ChainExecution | None:
        """Get an active execution by ID."""
        return self._active_executions.get(execution_id)

    def list_active_executions(self) -> list[ChainExecution]:
        """List all active chain executions."""
        return list(self._active_executions.values())


# =============================================================================
# Chain Store
# =============================================================================


class ChainStore:
    """
    Storage for deliberation chain definitions.

    Provides CRUD operations for chains and chain templates.
    """

    def __init__(self) -> None:
        self._chains: dict[str, DeliberationChain] = {}

    def save(self, chain: DeliberationChain) -> None:
        """Save a chain definition."""
        self._chains[chain.id] = chain

    def get(self, chain_id: str) -> DeliberationChain | None:
        """Get a chain by ID."""
        return self._chains.get(chain_id)

    def delete(self, chain_id: str) -> bool:
        """Delete a chain by ID."""
        if chain_id in self._chains:
            del self._chains[chain_id]
            return True
        return False

    def list_all(
        self,
        tags: list[str] | None = None,
    ) -> list[DeliberationChain]:
        """List all chains, optionally filtered by tags."""
        chains = list(self._chains.values())
        if tags:
            chains = [c for c in chains if any(t in c.tags for t in tags)]
        return chains


# =============================================================================
# Predefined Chain Templates
# =============================================================================


def create_code_review_chain(code: str, context: str = "") -> DeliberationChain:
    """
    Create a code review pipeline chain.

    Stages:
    1. Initial code review for correctness
    2. Security audit
    3. Performance review
    4. Final approval
    """
    return DeliberationChain(
        name="Code Review Pipeline",
        description="Multi-stage code review with security and performance audits",
        stages=[
            DeliberationStage(
                id="initial_review",
                topic_template=(
                    "Review this code for correctness, readability, and best practices:\n\n"
                    "{context.code}\n\n"
                    "Additional context: {context.context}"
                ),
                required_consensus=0.7,
                timeout_seconds=180,
                next_on_success="security_audit",
                next_on_failure=None,  # End on failure
            ),
            DeliberationStage(
                id="security_audit",
                topic_template=(
                    "Perform a security audit on this code. "
                    "Initial review findings: {previous.output}\n\n"
                    "Code:\n{context.code}"
                ),
                required_consensus=0.8,
                timeout_seconds=240,
                next_on_success="performance_review",
                next_on_failure="revision_needed",
            ),
            DeliberationStage(
                id="revision_needed",
                topic_template=(
                    "The code failed security audit. Provide specific revision requirements.\n"
                    "Security findings: {previous.output}\n"
                    "Original code:\n{context.code}"
                ),
                required_consensus=0.6,
                timeout_seconds=120,
                next_on_success=None,
                next_on_failure=None,
            ),
            DeliberationStage(
                id="performance_review",
                topic_template=(
                    "Review code for performance issues and optimization opportunities.\n"
                    "Security audit passed: {security_audit.output}\n"
                    "Code:\n{context.code}"
                ),
                required_consensus=0.7,
                timeout_seconds=180,
                next_on_success="final_approval",
                next_on_failure="final_approval",  # Continue even with perf issues
            ),
            DeliberationStage(
                id="final_approval",
                topic_template=(
                    "Make final approval decision for this code.\n"
                    "Initial review: {initial_review.output}\n"
                    "Security: {security_audit.output}\n"
                    "Performance: {performance_review.output}\n"
                    "Code:\n{context.code}"
                ),
                required_consensus=0.8,
                timeout_seconds=120,
                next_on_success=None,
                next_on_failure=None,
            ),
        ],
        initial_context={"code": code, "context": context},
        tags=["code-review", "security", "performance"],
    )


def create_draft_review_chain(draft: str, requirements: str = "") -> DeliberationChain:
    """
    Create a draft review chain.

    Stages:
    1. Draft analysis
    2. Critique
    3. Revision suggestions
    4. Final review
    """
    return DeliberationChain(
        name="Draft Review Pipeline",
        description="Multi-stage draft review: Draft → Critique → Revise → Approve",
        stages=[
            DeliberationStage(
                id="analysis",
                topic_template=(
                    "Analyze this draft for clarity, completeness, and accuracy:\n\n"
                    "{context.draft}\n\n"
                    "Requirements: {context.requirements}"
                ),
                required_consensus=0.6,
                timeout_seconds=180,
                next_on_success="critique",
                next_on_failure="critique",  # Continue to critique even if analysis differs
            ),
            DeliberationStage(
                id="critique",
                topic_template=(
                    "Provide constructive critique of this draft.\n"
                    "Analysis: {previous.output}\n\n"
                    "Draft:\n{context.draft}"
                ),
                required_consensus=0.7,
                timeout_seconds=240,
                next_on_success="revision",
                next_on_failure=None,
            ),
            DeliberationStage(
                id="revision",
                topic_template=(
                    "Based on the critique, suggest specific revisions.\n"
                    "Critique: {previous.output}\n\n"
                    "Original draft:\n{context.draft}"
                ),
                required_consensus=0.7,
                timeout_seconds=180,
                next_on_success="final_review",
                next_on_failure="final_review",
            ),
            DeliberationStage(
                id="final_review",
                topic_template=(
                    "Make final recommendation: approve, revise, or reject.\n"
                    "Analysis: {analysis.output}\n"
                    "Critique: {critique.output}\n"
                    "Suggested revisions: {revision.output}\n\n"
                    "Original draft:\n{context.draft}"
                ),
                required_consensus=0.8,
                timeout_seconds=120,
                next_on_success=None,
                next_on_failure=None,
            ),
        ],
        initial_context={"draft": draft, "requirements": requirements},
        tags=["draft", "review", "content"],
    )


def create_research_synthesis_chain(
    topic: str, sources: list[str] | None = None
) -> DeliberationChain:
    """
    Create a research synthesis chain.

    Stages:
    1. Research gathering
    2. Analysis and synthesis
    3. Validation
    4. Final summary
    """
    return DeliberationChain(
        name="Research Synthesis Pipeline",
        description="Research → Synthesize → Validate → Summarize",
        stages=[
            DeliberationStage(
                id="research",
                topic_template=(
                    "Research the following topic and gather key findings:\n"
                    "Topic: {context.topic}\n"
                    "Known sources: {context.sources}"
                ),
                required_consensus=0.6,
                timeout_seconds=300,
                next_on_success="synthesis",
                next_on_failure=None,
            ),
            DeliberationStage(
                id="synthesis",
                topic_template=(
                    "Synthesize the research findings into coherent themes.\n"
                    "Research findings: {previous.output}\n"
                    "Topic: {context.topic}"
                ),
                required_consensus=0.7,
                timeout_seconds=240,
                next_on_success="validation",
                next_on_failure="validation",
            ),
            DeliberationStage(
                id="validation",
                topic_template=(
                    "Validate the synthesized findings for accuracy and completeness.\n"
                    "Synthesis: {previous.output}\n"
                    "Original research: {research.output}\n"
                    "Topic: {context.topic}"
                ),
                required_consensus=0.8,
                timeout_seconds=180,
                next_on_success="summary",
                next_on_failure="revision",
            ),
            DeliberationStage(
                id="revision",
                topic_template=(
                    "Revise the synthesis based on validation feedback.\n"
                    "Validation feedback: {previous.output}\n"
                    "Original synthesis: {synthesis.output}"
                ),
                required_consensus=0.7,
                timeout_seconds=180,
                next_on_success="summary",
                next_on_failure=None,
            ),
            DeliberationStage(
                id="summary",
                topic_template=(
                    "Create a final summary of the validated research.\n"
                    "Validated synthesis: {validation.output}\n"
                    "Topic: {context.topic}"
                ),
                required_consensus=0.7,
                timeout_seconds=120,
                next_on_success=None,
                next_on_failure=None,
            ),
        ],
        initial_context={"topic": topic, "sources": sources or []},
        tags=["research", "synthesis", "validation"],
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ChainStatus",
    "StageStatus",
    "StageTransition",
    # Data Classes
    "DeliberationStage",
    "StageResult",
    "DeliberationChain",
    "ChainExecution",
    # Classes
    "TemplateEngine",
    "ChainExecutor",
    "ChainStore",
    # Template Functions
    "create_code_review_chain",
    "create_draft_review_chain",
    "create_research_synthesis_chain",
]

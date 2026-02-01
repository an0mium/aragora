"""
Workflow Executor Protocol - Unified interface for all workflow execution systems.

Provides a common Protocol that WorkflowEngine, EnhancedWorkflowEngine, and
TaskQueue (via adapter) can implement, enabling polymorphic usage.

Usage:
    from aragora.workflow import get_workflow_executor, WorkflowExecutor

    # Get the appropriate executor
    executor = get_workflow_executor(mode="default")

    # All executors share the same interface
    result = await executor.execute(definition, inputs)
    metrics = executor.get_metrics()
    executor.request_termination("User cancelled")
"""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from aragora.workflow.types import (
        WorkflowCheckpoint,
        WorkflowDefinition,
        WorkflowResult,
    )


@runtime_checkable
class WorkflowExecutor(Protocol):
    """
    Base protocol for all workflow execution systems.

    This is the minimal interface that all workflow executors must implement,
    enabling polymorphic usage across different execution backends.

    Implementations:
    - WorkflowEngine: DAG-based execution with checkpointing
    - EnhancedWorkflowEngine: Resource-aware execution with cost tracking
    - TaskQueueExecutorAdapter: Queue-based execution with priority scheduling
    """

    async def execute(
        self,
        definition: "WorkflowDefinition",
        inputs: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> "WorkflowResult":
        """
        Execute a workflow from the beginning.

        Args:
            definition: Workflow definition to execute
            inputs: Input parameters for the workflow
            workflow_id: Optional ID (generated if not provided)

        Returns:
            WorkflowResult with step results and final output
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """
        Get execution metrics.

        Returns:
            Dictionary containing execution statistics such as:
            - total_steps: Number of steps executed
            - completed_steps: Successfully completed steps
            - failed_steps: Steps that failed
            - total_duration_ms: Total execution time
        """
        ...

    def request_termination(self, reason: str = "Requested") -> None:
        """
        Request early termination of workflow execution.

        Args:
            reason: Human-readable reason for termination
        """
        ...


@runtime_checkable
class ResumableExecutor(WorkflowExecutor, Protocol):
    """
    Extended protocol for executors that support checkpoint-based resume.

    Adds the ability to resume workflows from a saved checkpoint,
    enabling long-running workflow recovery and continuation.
    """

    async def resume(
        self,
        workflow_id: str,
        checkpoint: "WorkflowCheckpoint",
        definition: "WorkflowDefinition",
    ) -> "WorkflowResult":
        """
        Resume a workflow from a checkpoint.

        Args:
            workflow_id: ID of the workflow to resume
            checkpoint: Checkpoint to resume from
            definition: Workflow definition

        Returns:
            WorkflowResult from resumed execution
        """
        ...

    async def get_checkpoint(self, checkpoint_id: str) -> "WorkflowCheckpoint | None":
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: ID of the checkpoint to retrieve

        Returns:
            WorkflowCheckpoint if found, None otherwise
        """
        ...

    async def get_latest_checkpoint(self, workflow_id: str) -> "WorkflowCheckpoint | None":
        """
        Get the most recent checkpoint for a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Most recent WorkflowCheckpoint if available
        """
        ...


@runtime_checkable
class ResourceAwareExecutor(WorkflowExecutor, Protocol):
    """
    Extended protocol for executors that track and limit resource usage.

    Adds cost estimation and resource tracking capabilities for
    budget-conscious workflow execution.
    """

    def estimate_cost(self, definition: "WorkflowDefinition") -> dict[str, float]:
        """
        Estimate workflow cost before execution.

        Args:
            definition: Workflow definition to estimate

        Returns:
            Dictionary with estimated costs:
            - total: Total estimated cost in USD
            - Per-agent breakdown (e.g., {"claude": 0.05, "gpt4": 0.12})
        """
        ...

    @property
    def usage(self) -> Any:
        """
        Get current resource usage.

        Returns:
            ResourceUsage object with token counts, costs, etc.
        """
        ...

    @property
    def limits(self) -> Any:
        """
        Get current resource limits.

        Returns:
            ResourceLimits object with max tokens, max cost, etc.
        """
        ...

    def set_limits(self, limits: Any) -> None:
        """
        Update resource limits.

        Args:
            limits: New ResourceLimits to apply
        """
        ...


# Type aliases for convenience
ExecutorType = WorkflowExecutor | ResumableExecutor | ResourceAwareExecutor


__all__ = [
    "WorkflowExecutor",
    "ResumableExecutor",
    "ResourceAwareExecutor",
    "ExecutorType",
]

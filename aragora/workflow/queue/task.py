"""
Workflow Task Definition.

Defines the WorkflowTask dataclass that represents a unit of work
to be scheduled and executed by the task queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional
import uuid


class TaskStatus(str, Enum):
    """Status of a workflow task."""

    PENDING = "pending"  # Waiting for dependencies
    READY = "ready"  # Dependencies resolved, waiting for execution
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Cancelled before execution
    TIMEOUT = "timeout"  # Execution timed out
    RETRY = "retry"  # Scheduled for retry


class TaskPriority(IntEnum):
    """Task execution priority (lower = higher priority)."""

    CRITICAL = 0  # Execute immediately
    HIGH = 10
    NORMAL = 50
    LOW = 100
    BACKGROUND = 200


@dataclass
class TaskResult:
    """Result of task execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    retries_used: int = 0


@dataclass
class WorkflowTask:
    """
    A task to be scheduled and executed as part of a workflow.

    Tasks track their dependencies and can only execute when all
    dependencies have completed successfully.

    Attributes:
        id: Unique task identifier
        workflow_id: ID of the parent workflow
        step_id: ID of the step this task executes
        step_config: Configuration for the step
        depends_on: List of task IDs that must complete first
        priority: Execution priority (lower = higher)
        timeout_seconds: Maximum execution time
        max_retries: Maximum retry attempts
        metadata: Additional task metadata
    """

    id: str
    workflow_id: str
    step_id: str
    step_config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    # Scheduling options
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: float = 300.0
    max_retries: int = 2

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Optional[TaskResult] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Execution context
    executor_id: Optional[str] = None
    tenant_id: str = "default"

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        workflow_id: str,
        step_id: str,
        step_config: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> WorkflowTask:
        """Create a new workflow task with auto-generated ID."""
        return cls(
            id=f"task_{uuid.uuid4().hex[:12]}",
            workflow_id=workflow_id,
            step_id=step_id,
            step_config=step_config or {},
            depends_on=depends_on or [],
            priority=priority,
            **kwargs,
        )

    @property
    def is_runnable(self) -> bool:
        """Check if task can be executed (status is READY)."""
        return self.status == TaskStatus.READY

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    @property
    def has_dependencies(self) -> bool:
        """Check if task has unresolved dependencies."""
        return len(self.depends_on) > 0

    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT)
            and self.retry_count < self.max_retries
        )

    @property
    def wait_time_ms(self) -> float:
        """Time spent waiting (from creation to start)."""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds() * 1000
        return 0

    @property
    def execution_time_ms(self) -> float:
        """Time spent executing (from start to completion)."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0

    def mark_ready(self) -> None:
        """Mark task as ready for execution."""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.READY
            self.queued_at = datetime.now()

    def mark_running(self, executor_id: str) -> None:
        """Mark task as running."""
        if self.status in (TaskStatus.READY, TaskStatus.RETRY):
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.now()
            self.executor_id = executor_id

    def mark_completed(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.result = TaskResult(success=False, error=error)
        self.completed_at = datetime.now()

    def mark_timeout(self) -> None:
        """Mark task as timed out."""
        self.status = TaskStatus.TIMEOUT
        self.result = TaskResult(success=False, error="Task execution timeout")
        self.completed_at = datetime.now()

    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        if not self.is_terminal:
            self.status = TaskStatus.CANCELLED
            self.completed_at = datetime.now()

    def schedule_retry(self) -> bool:
        """Schedule task for retry if possible."""
        if self.can_retry:
            self.status = TaskStatus.RETRY
            self.retry_count += 1
            self.started_at = None
            self.completed_at = None
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "step_config": self.step_config,
            "depends_on": self.depends_on,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "result": self.result.__dict__ if self.result else None,
            "created_at": self.created_at.isoformat(),
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "executor_id": self.executor_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowTask:
        """Create from dictionary."""
        result_data = data.get("result")
        result = TaskResult(**result_data) if result_data else None

        return cls(
            id=data["id"],
            workflow_id=data["workflow_id"],
            step_id=data["step_id"],
            step_config=data.get("step_config", {}),
            depends_on=data.get("depends_on", []),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL)),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            max_retries=data.get("max_retries", 2),
            status=TaskStatus(data.get("status", TaskStatus.PENDING)),
            retry_count=data.get("retry_count", 0),
            result=result,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            queued_at=datetime.fromisoformat(data["queued_at"]) if data.get("queued_at") else None,
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            executor_id=data.get("executor_id"),
            tenant_id=data.get("tenant_id", "default"),
            metadata=data.get("metadata", {}),
        )

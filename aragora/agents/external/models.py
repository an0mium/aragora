"""Data models for external agent framework integration.

Defines the core data structures for task submission, tracking, and results
when interacting with external AI agent frameworks like OpenHands, AutoGPT,
and CrewAI through Aragora's gateway layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """External agent task lifecycle status."""

    PENDING = "pending"  # Task submitted, waiting for execution
    INITIALIZING = "initializing"  # Framework is setting up
    RUNNING = "running"  # Task is executing
    PAUSED = "paused"  # Task paused (awaiting approval or resource)
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task failed
    CANCELLED = "cancelled"  # Task was cancelled
    TIMEOUT = "timeout"  # Task exceeded timeout


class ToolPermission(Enum):
    """Permissions for external agent tool access.

    Maps to Aragora RBAC permissions using existing COMPUTER_USE resource type.
    This reuses the computer_use.* permission namespace for consistency.
    """

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL_EXECUTE = "shell_execute"
    NETWORK_ACCESS = "network_access"
    BROWSER_USE = "browser_use"
    CODE_EXECUTE = "code_execute"
    API_CALL = "api_call"
    SCREENSHOT = "screenshot"

    # Mapping to existing RBAC permission keys (computer_use.*)
    _PERMISSION_MAP = {
        "file_read": "computer_use.file_read",
        "file_write": "computer_use.file_write",
        "shell_execute": "computer_use.shell",
        "network_access": "computer_use.network",
        "browser_use": "computer_use.browser",
        "code_execute": "computer_use.execute",
        "api_call": "computer_use.network",  # Reuse network permission
        "screenshot": "computer_use.screenshot",
    }

    def to_permission_key(self) -> str:
        """Convert to Aragora RBAC permission key.

        Uses existing computer_use.* permissions for consistency with
        the broader RBAC system.
        """
        return self._PERMISSION_MAP.get(self.value, f"computer_use.{self.value}")


@dataclass
class TaskRequest:
    """Request to submit a task to an external agent framework.

    Attributes:
        task_type: Type of task (e.g., 'code', 'research', 'analysis').
        prompt: Task description/prompt for the agent.
        context: Additional context dict (workspace info, conversation history).
        tool_permissions: Set of tools the task is allowed to use.
        timeout_seconds: Maximum execution time.
        max_steps: Maximum agent steps/iterations.
        workspace_id: Optional workspace identifier for isolation.
        user_id: User who initiated the task.
        tenant_id: Tenant for multi-tenancy isolation.
        metadata: Additional metadata for tracking/billing.
        id: Unique task identifier (auto-generated).
        created_at: Timestamp when request was created.
    """

    task_type: str
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    tool_permissions: set[ToolPermission] = field(default_factory=set)
    timeout_seconds: float = 3600.0  # 1 hour default
    max_steps: int = 100
    workspace_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "context": self.context,
            "tool_permissions": [p.value for p in self.tool_permissions],
            "timeout_seconds": self.timeout_seconds,
            "max_steps": self.max_steps,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TaskResult:
    """Result from an external agent task.

    Attributes:
        task_id: ID of the task this result belongs to.
        status: Final task status.
        output: Primary output text/response from the agent.
        artifacts: List of artifacts produced (files, code, etc.).
        steps_executed: Number of steps/iterations the agent performed.
        tokens_used: Total tokens consumed (input + output).
        cost_usd: Estimated cost in USD.
        started_at: When execution started.
        completed_at: When execution completed.
        error: Error message if task failed.
        logs: Execution logs for debugging/audit.
    """

    task_id: str
    status: TaskStatus
    output: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    steps_executed: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_terminal(self) -> bool:
        """Check if status is terminal (no more updates expected)."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "artifacts": self.artifacts,
            "steps_executed": self.steps_executed,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "logs": self.logs,
        }


@dataclass
class TaskProgress:
    """Progress update for a running task.

    Used for streaming progress updates to clients via WebSocket.
    """

    task_id: str
    status: TaskStatus
    current_step: int
    total_steps: int | None
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def progress_percent(self) -> float | None:
        """Calculate progress percentage if total_steps is known."""
        if self.total_steps and self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthStatus:
    """Health status of an external agent adapter.

    Used for monitoring adapter availability and performance.
    """

    adapter_name: str
    healthy: bool
    last_check: datetime
    response_time_ms: float
    error: str | None = None
    framework_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "adapter_name": self.adapter_name,
            "healthy": self.healthy,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "framework_version": self.framework_version,
            "metadata": self.metadata,
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation for audit/analysis.

    Tracks every tool call made by an external agent for security
    auditing and compliance.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool = True
    blocked: bool = False
    block_reason: str | None = None
    approval_required: bool = False
    approved_by: str | None = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }

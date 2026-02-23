"""Base class for external agent framework adapters.

Provides an abstract base class that defines the interface for integrating
external AI agent frameworks (OpenHands, AutoGPT, CrewAI, etc.) with
Aragora's enterprise controls.

Adapters inherit from ExternalAgentAdapter and implement the required
methods for task submission, status tracking, and result retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import AsyncIterator, Callable

from aragora.resilience.circuit_breaker_v2 import (
    BaseCircuitBreaker,
    CircuitBreakerConfig as CBConfig,
    get_circuit_breaker,
)

from .config import ExternalAgentConfig
from .models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# Type alias for event callbacks (WebSocket notifications)
EventCallback = Callable[[str, dict[str, Any]], None]


class ExternalAgentError(Exception):
    """Base exception for external agent adapter errors."""

    def __init__(
        self,
        message: str,
        adapter_name: str | None = None,
        task_id: str | None = None,
    ):
        super().__init__(message)
        self.adapter_name = adapter_name
        self.task_id = task_id


class TaskNotFoundError(ExternalAgentError):
    """Raised when a task is not found."""

    pass


class TaskNotCompleteError(ExternalAgentError):
    """Raised when trying to get result of incomplete task."""

    pass


class ExternalAgentAdapter(ABC):
    """Abstract base class for external AI agent framework adapters.

    Provides a unified interface for interacting with external agent
    frameworks like OpenHands, AutoGPT, and CrewAI with Aragora's
    enterprise controls.

    Subclasses must implement:
        - submit_task: Submit a task for async execution
        - get_task_status: Poll for task status
        - get_task_result: Retrieve completed task result
        - cancel_task: Cancel a running task
        - health_check: Check adapter/framework health

    Optional overrides:
        - stream_progress: Stream progress updates
        - list_tools: List available tools
        - validate_task: Pre-validate a task

    Features:
        - Circuit breaker protection for external service calls
        - Event callbacks for WebSocket notifications
        - Token usage and cost tracking

    Example:
        class MyAdapter(ExternalAgentAdapter):
            adapter_name = "my_framework"

            async def submit_task(self, request: TaskRequest) -> str:
                # Validate and submit to external framework
                task_id = await self._framework.create_task(request.prompt)
                self._tasks_submitted += 1
                return task_id

            async def get_task_status(self, task_id: str) -> TaskStatus:
                status = await self._framework.get_status(task_id)
                return self._map_status(status)

            # ... implement other abstract methods
    """

    adapter_name: str = "base"

    def __init__(
        self,
        config: ExternalAgentConfig,
        event_callback: EventCallback | None = None,
        enable_circuit_breaker: bool = True,
    ):
        """Initialize the adapter.

        Args:
            config: Configuration for the adapter.
            event_callback: Optional callback for emitting events.
            enable_circuit_breaker: Enable circuit breaker protection.
        """
        self._config = config
        self._event_callback = event_callback
        self._enable_circuit_breaker = enable_circuit_breaker

        # Circuit breaker setup
        self._circuit_breaker: BaseCircuitBreaker | None = None
        if enable_circuit_breaker:
            cb_config = CBConfig(
                failure_threshold=config.circuit_breaker.failure_threshold,
                success_threshold=config.circuit_breaker.success_threshold,
                cooldown_seconds=config.circuit_breaker.cooldown_seconds,
                half_open_max_requests=config.circuit_breaker.half_open_max_requests,
            )
            self._circuit_breaker = get_circuit_breaker(
                f"external_agent_{config.adapter_name}",
                config=cb_config,
            )

        # Metrics tracking
        self._tasks_submitted: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0

    @property
    def config(self) -> ExternalAgentConfig:
        """Get adapter configuration."""
        return self._config

    @property
    def is_available(self) -> bool:
        """Check if adapter is available (circuit breaker not open)."""
        if self._circuit_breaker is None:
            return True
        return self._circuit_breaker.can_execute()

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def submit_task(self, request: TaskRequest) -> str:
        """Submit a task for asynchronous execution.

        Args:
            request: Task request with prompt, context, and permissions.

        Returns:
            Task ID for tracking.

        Raises:
            ExternalAgentError: If submission fails.
        """
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current status of a task.

        Args:
            task_id: ID of task to check.

        Returns:
            Current task status.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        pass

    @abstractmethod
    async def get_task_result(self, task_id: str) -> TaskResult:
        """Get result of a completed task.

        Args:
            task_id: ID of completed task.

        Returns:
            Task result with output, artifacts, and metrics.

        Raises:
            TaskNotFoundError: If task doesn't exist.
            TaskNotCompleteError: If task is still running.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or pending task.

        Args:
            task_id: ID of task to cancel.

        Returns:
            True if cancelled, False if not cancellable.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check health of the external framework.

        Returns:
            Health status with latency and version info.
        """
        pass

    # =========================================================================
    # Optional Methods - Can be overridden by subclasses
    # =========================================================================

    async def stream_progress(self, task_id: str) -> AsyncIterator[TaskProgress]:
        """Stream progress updates for a running task.

        Default implementation polls get_task_status.
        Override for frameworks with native streaming.

        Args:
            task_id: ID of task to stream.

        Yields:
            TaskProgress updates.
        """
        import asyncio

        step = 0
        while True:
            status = await self.get_task_status(task_id)
            step += 1
            yield TaskProgress(
                task_id=task_id,
                status=status,
                current_step=step,
                total_steps=None,
                message=f"Task status: {status.value}",
            )

            if status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ):
                break

            await asyncio.sleep(2.0)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools in the framework.

        Returns:
            List of tool definitions with name, description, parameters.
        """
        return []

    async def validate_task(self, request: TaskRequest) -> tuple[bool, str | None]:
        """Pre-validate a task request before submission.

        Args:
            request: Task request to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Default validation checks
        if not request.prompt:
            return False, "Task prompt cannot be empty"

        if request.timeout_seconds > self._config.timeout_seconds:
            return (
                False,
                f"Timeout exceeds maximum: {self._config.timeout_seconds}s",
            )

        if request.max_steps <= 0:
            return False, "max_steps must be positive"

        return True, None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit event via callback if configured.

        Args:
            event_type: Type of event (e.g., 'task_submitted', 'task_completed').
            data: Event data dictionary.
        """
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning("[%s] Event emission failed: %s", self.adapter_name, e)

    def _record_success(self) -> None:
        """Record successful operation for circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def _record_failure(self, error: Exception | None = None) -> None:
        """Record failed operation for circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_failure(error)

    def _update_metrics(self, result: TaskResult) -> None:
        """Update internal metrics from a task result.

        Args:
            result: Task result with token and cost info.
        """
        if result.status == TaskStatus.COMPLETED:
            self._tasks_completed += 1
        elif result.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
            self._tasks_failed += 1

        self._total_tokens += result.tokens_used
        self._total_cost_usd += result.cost_usd

    def get_metrics(self) -> dict[str, Any]:
        """Get adapter metrics for monitoring.

        Returns:
            Dictionary with task counts, token usage, and circuit breaker status.
        """
        metrics = {
            "adapter_name": self.adapter_name,
            "tasks_submitted": self._tasks_submitted,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost_usd,
            "is_available": self.is_available,
        }

        if self._circuit_breaker:
            stats = self._circuit_breaker.get_stats()
            metrics["circuit_breaker"] = {
                "state": stats.state.value,
                "failure_count": stats.failure_count,
                "success_count": stats.success_count,
                "last_failure_time": stats.last_failure_time,
            }

        return metrics

    def reset_metrics(self) -> None:
        """Reset internal metrics. Useful for testing."""
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"adapter_name={self.adapter_name!r}, "
            f"is_available={self.is_available})"
        )

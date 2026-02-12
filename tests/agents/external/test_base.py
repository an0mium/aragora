"""Tests for external agent adapter base class."""

from datetime import datetime, timezone
from collections.abc import AsyncIterator

import pytest

from aragora.agents.external.base import (
    ExternalAgentAdapter,
    ExternalAgentError,
    TaskNotCompleteError,
    TaskNotFoundError,
)
from aragora.agents.external.config import ExternalAgentConfig
from aragora.agents.external.models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


class ConcreteAdapter(ExternalAgentAdapter):
    """Concrete implementation for testing."""

    adapter_name = "concrete"

    def __init__(self, config: ExternalAgentConfig | None = None):
        super().__init__(config or ExternalAgentConfig(adapter_name="concrete"))
        self._tasks: dict[str, TaskStatus] = {}
        self._healthy = True

    async def submit_task(self, request: TaskRequest) -> str:
        task_id = f"task-{request.id}"
        self._tasks[task_id] = TaskStatus.PENDING
        self._tasks_submitted += 1
        return task_id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        return self._tasks[task_id]

    async def get_task_result(self, task_id: str) -> TaskResult:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        status = self._tasks[task_id]
        if status == TaskStatus.RUNNING:
            raise TaskNotCompleteError(f"Task {task_id} not complete")
        return TaskResult(
            task_id=task_id,
            status=status,
            output="Completed",
            tokens_used=50,
            cost_usd=0.005,
        )

    async def cancel_task(self, task_id: str) -> bool:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        if self._tasks[task_id] in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return False
        self._tasks[task_id] = TaskStatus.CANCELLED
        return True

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            adapter_name=self.adapter_name,
            healthy=self._healthy,
            last_check=datetime.now(timezone.utc),
            response_time_ms=5.0,
        )

    def set_status(self, task_id: str, status: TaskStatus) -> None:
        self._tasks[task_id] = status


class TestExternalAgentError:
    """Tests for exception classes."""

    def test_error_with_context(self) -> None:
        """Test error creation with adapter and task context."""
        error = ExternalAgentError(
            "Connection failed",
            adapter_name="openhands",
            task_id="task-123",
        )
        assert "Connection failed" in str(error)
        assert error.adapter_name == "openhands"
        assert error.task_id == "task-123"

    def test_task_not_found_error(self) -> None:
        """Test TaskNotFoundError."""
        error = TaskNotFoundError("Task abc not found")
        assert isinstance(error, ExternalAgentError)

    def test_task_not_complete_error(self) -> None:
        """Test TaskNotCompleteError."""
        error = TaskNotCompleteError("Task still running")
        assert isinstance(error, ExternalAgentError)


class TestExternalAgentAdapter:
    """Tests for ExternalAgentAdapter base class."""

    def test_adapter_creation(self) -> None:
        """Test adapter creation with config."""
        config = ExternalAgentConfig(adapter_name="test")
        adapter = ConcreteAdapter(config)

        assert adapter.adapter_name == "concrete"
        assert adapter.config is config
        assert adapter.is_available is True

    def test_adapter_metrics_initial(self) -> None:
        """Test initial metrics are zero."""
        adapter = ConcreteAdapter()
        metrics = adapter.get_metrics()

        assert metrics["tasks_submitted"] == 0
        assert metrics["tasks_completed"] == 0
        assert metrics["tasks_failed"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["total_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_submit_task(self) -> None:
        """Test task submission."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Hello")

        task_id = await adapter.submit_task(request)

        assert task_id.startswith("task-")
        metrics = adapter.get_metrics()
        assert metrics["tasks_submitted"] == 1

    @pytest.mark.asyncio
    async def test_get_task_status(self) -> None:
        """Test getting task status."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Test")
        task_id = await adapter.submit_task(request)

        status = await adapter.get_task_status(task_id)
        assert status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self) -> None:
        """Test status check for non-existent task."""
        adapter = ConcreteAdapter()

        with pytest.raises(TaskNotFoundError):
            await adapter.get_task_status("nonexistent")

    @pytest.mark.asyncio
    async def test_get_task_result(self) -> None:
        """Test getting task result."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Test")
        task_id = await adapter.submit_task(request)
        adapter.set_status(task_id, TaskStatus.COMPLETED)

        result = await adapter.get_task_result(task_id)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "Completed"
        assert result.tokens_used == 50

    @pytest.mark.asyncio
    async def test_get_result_not_complete(self) -> None:
        """Test getting result for incomplete task."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Test")
        task_id = await adapter.submit_task(request)
        adapter.set_status(task_id, TaskStatus.RUNNING)

        with pytest.raises(TaskNotCompleteError):
            await adapter.get_task_result(task_id)

    @pytest.mark.asyncio
    async def test_cancel_task(self) -> None:
        """Test task cancellation."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Test")
        task_id = await adapter.submit_task(request)

        cancelled = await adapter.cancel_task(task_id)

        assert cancelled is True
        status = await adapter.get_task_status(task_id)
        assert status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self) -> None:
        """Test cancelling already completed task."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Test")
        task_id = await adapter.submit_task(request)
        adapter.set_status(task_id, TaskStatus.COMPLETED)

        cancelled = await adapter.cancel_task(task_id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check."""
        adapter = ConcreteAdapter()

        health = await adapter.health_check()

        assert health.healthy is True
        assert health.adapter_name == "concrete"
        assert health.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_validate_task_valid(self) -> None:
        """Test validation of valid request."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="Hello")

        valid, error = await adapter.validate_task(request)

        assert valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_task_empty_prompt(self) -> None:
        """Test validation rejects empty prompt."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="")

        valid, error = await adapter.validate_task(request)

        assert valid is False
        assert "empty" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_task_negative_steps(self) -> None:
        """Test validation rejects negative max_steps."""
        adapter = ConcreteAdapter()
        request = TaskRequest(task_type="code", prompt="test", max_steps=-1)

        valid, error = await adapter.validate_task(request)

        assert valid is False
        assert "positive" in error.lower()

    @pytest.mark.asyncio
    async def test_list_tools_default_empty(self) -> None:
        """Test default list_tools returns empty."""
        adapter = ConcreteAdapter()

        tools = await adapter.list_tools()

        assert tools == []

    def test_update_metrics(self) -> None:
        """Test metrics update from result."""
        adapter = ConcreteAdapter()
        result = TaskResult(
            task_id="test",
            status=TaskStatus.COMPLETED,
            tokens_used=100,
            cost_usd=0.01,
        )

        adapter._update_metrics(result)

        metrics = adapter.get_metrics()
        assert metrics["tasks_completed"] == 1
        assert metrics["total_tokens"] == 100
        assert metrics["total_cost_usd"] == 0.01

    def test_update_metrics_failed(self) -> None:
        """Test metrics update for failed task."""
        adapter = ConcreteAdapter()
        result = TaskResult(
            task_id="test",
            status=TaskStatus.FAILED,
            tokens_used=50,
            cost_usd=0.005,
        )

        adapter._update_metrics(result)

        metrics = adapter.get_metrics()
        assert metrics["tasks_failed"] == 1

    def test_reset_metrics(self) -> None:
        """Test metrics reset."""
        adapter = ConcreteAdapter()
        result = TaskResult(
            task_id="test",
            status=TaskStatus.COMPLETED,
            tokens_used=100,
            cost_usd=0.01,
        )
        adapter._update_metrics(result)

        adapter.reset_metrics()

        metrics = adapter.get_metrics()
        assert metrics["tasks_completed"] == 0
        assert metrics["total_tokens"] == 0

    def test_repr(self) -> None:
        """Test string representation."""
        adapter = ConcreteAdapter()
        repr_str = repr(adapter)

        assert "ConcreteAdapter" in repr_str
        assert "concrete" in repr_str


class TestEventCallback:
    """Tests for event callback functionality."""

    @pytest.mark.asyncio
    async def test_emit_event_calls_callback(self) -> None:
        """Test event emission calls callback."""
        events: list[tuple[str, dict]] = []

        def capture_event(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        config = ExternalAgentConfig(adapter_name="test")
        adapter = ConcreteAdapter(config)
        adapter._event_callback = capture_event

        adapter._emit_event("test_event", {"key": "value"})

        assert len(events) == 1
        assert events[0][0] == "test_event"
        assert events[0][1] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_emit_event_handles_callback_error(self) -> None:
        """Test event emission handles callback errors gracefully."""

        def failing_callback(event_type: str, data: dict) -> None:
            raise RuntimeError("Callback failed")

        adapter = ConcreteAdapter()
        adapter._event_callback = failing_callback

        # Should not raise
        adapter._emit_event("test", {})


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_enabled_by_default(self) -> None:
        """Test circuit breaker is enabled by default."""
        adapter = ConcreteAdapter()
        assert adapter._circuit_breaker is not None

    def test_circuit_breaker_can_be_disabled(self) -> None:
        """Test circuit breaker can be disabled."""
        config = ExternalAgentConfig(adapter_name="test")
        adapter = ConcreteAdapter.__new__(ConcreteAdapter)
        ExternalAgentAdapter.__init__(
            adapter,
            config,
            enable_circuit_breaker=False,
        )
        assert adapter._circuit_breaker is None

    def test_record_success(self) -> None:
        """Test recording success."""
        adapter = ConcreteAdapter()
        # Should not raise
        adapter._record_success()

    def test_record_failure(self) -> None:
        """Test recording failure."""
        adapter = ConcreteAdapter()
        # Should not raise
        adapter._record_failure(RuntimeError("test"))

    def test_metrics_include_circuit_breaker(self) -> None:
        """Test metrics include circuit breaker state."""
        adapter = ConcreteAdapter()
        metrics = adapter.get_metrics()

        assert "circuit_breaker" in metrics
        assert "state" in metrics["circuit_breaker"]
        assert "failure_count" in metrics["circuit_breaker"]

"""Tests for external agent proxy wrapper."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import AsyncIterator

import pytest

from aragora.agents.external.base import ExternalAgentAdapter
from aragora.agents.external.config import ExternalAgentConfig
from aragora.agents.external.models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
    ToolPermission,
)
from aragora.agents.external.proxy import (
    ExternalAgentProxy,
    PolicyDeniedError,
    ProxyConfig,
)
from aragora.agents.external.security import ExternalAgentSecurityPolicy


@dataclass
class MockAuthContext:
    """Mock authorization context for testing."""

    user_id: str = "test-user"
    permissions: set[str] | None = None

    def has_permission(self, permission: str) -> bool:
        if self.permissions is None:
            return True
        return permission in self.permissions


class MockAdapter(ExternalAgentAdapter):
    """Mock adapter for testing proxy."""

    adapter_name = "mock"

    def __init__(self, config: ExternalAgentConfig | None = None):
        super().__init__(config or ExternalAgentConfig(adapter_name="mock"))
        self._tasks: dict[str, TaskStatus] = {}
        self._results: dict[str, TaskResult] = {}
        self._should_fail = False
        self._should_timeout = False
        self._submit_delay = 0.0

    async def submit_task(self, request: TaskRequest) -> str:
        if self._should_timeout:
            await asyncio.sleep(10)  # Longer than default timeout
        if self._submit_delay > 0:
            await asyncio.sleep(self._submit_delay)
        if self._should_fail:
            raise RuntimeError("Mock submission failure")
        task_id = f"mock-{request.id}"
        self._tasks[task_id] = TaskStatus.PENDING
        self._tasks_submitted += 1
        return task_id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        return self._tasks[task_id]

    async def get_task_result(self, task_id: str) -> TaskResult:
        if task_id in self._results:
            return self._results[task_id]
        return TaskResult(
            task_id=task_id,
            status=self._tasks.get(task_id, TaskStatus.PENDING),
            output="Mock result",
            tokens_used=100,
            cost_usd=0.01,
        )

    async def cancel_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            self._tasks[task_id] = TaskStatus.CANCELLED
            return True
        return False

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            adapter_name=self.adapter_name,
            healthy=True,
            last_check=datetime.now(timezone.utc),
            response_time_ms=10.0,
        )

    async def stream_progress(self, task_id: str) -> AsyncIterator[TaskProgress]:
        for i in range(3):
            yield TaskProgress(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                current_step=i + 1,
                total_steps=3,
                message=f"Step {i + 1}",
            )

    def set_task_status(self, task_id: str, status: TaskStatus) -> None:
        self._tasks[task_id] = status

    def set_task_result(self, task_id: str, result: TaskResult) -> None:
        self._results[task_id] = result


class TestProxyConfig:
    """Tests for ProxyConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProxyConfig()
        assert config.submit_timeout == 60.0
        assert config.status_timeout == 30.0
        assert config.result_timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 2.0
        assert config.retry_backoff == 2.0
        assert config.enable_policy_checks is True
        assert config.redact_output_secrets is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ProxyConfig(
            submit_timeout=120.0,
            max_retries=5,
            enable_policy_checks=False,
        )
        assert config.submit_timeout == 120.0
        assert config.max_retries == 5
        assert config.enable_policy_checks is False


class TestPolicyDeniedError:
    """Tests for PolicyDeniedError."""

    def test_error_creation(self) -> None:
        """Test error creation with reason."""
        error = PolicyDeniedError("Missing permission: gateway.execute")
        assert "Missing permission" in str(error)
        assert error.reason == "Missing permission: gateway.execute"

    def test_error_with_result(self) -> None:
        """Test error creation with policy result."""
        from aragora.agents.external.security import PolicyCheckResult

        result = PolicyCheckResult(
            allowed=False,
            reason="Blocked",
            blocked_tools={"shell_execute"},
        )
        error = PolicyDeniedError("Policy denied", result)
        assert error.result is not None
        assert error.result.blocked_tools == {"shell_execute"}


class TestExternalAgentProxy:
    """Tests for ExternalAgentProxy."""

    def test_proxy_creation(self) -> None:
        """Test proxy creation."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        assert proxy.wrapped_adapter is adapter
        assert proxy.metrics["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_submit_task_success(self) -> None:
        """Test successful task submission."""
        adapter = MockAdapter()
        context = MockAuthContext(permissions={"gateway.execute"})
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="Hello world")
        task_id = await proxy.submit_task(request)

        assert task_id.startswith("mock-")
        assert proxy.metrics["total_calls"] == 1
        assert proxy.metrics["successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_submit_task_policy_denied(self) -> None:
        """Test task submission denied by policy."""
        adapter = MockAdapter()
        context = MockAuthContext(permissions=set())  # No permissions
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="Hello world")

        with pytest.raises(PolicyDeniedError) as exc_info:
            await proxy.submit_task(request)

        assert "gateway.execute" in str(exc_info.value)
        assert proxy.metrics["policy_denials"] == 1

    @pytest.mark.asyncio
    async def test_submit_task_policy_disabled(self) -> None:
        """Test submission when policy checks disabled."""
        adapter = MockAdapter()
        context = MockAuthContext(permissions=set())
        config = ProxyConfig(enable_policy_checks=False)
        proxy = ExternalAgentProxy(adapter, context, config)

        request = TaskRequest(task_type="code", prompt="Hello world")
        task_id = await proxy.submit_task(request)

        assert task_id is not None  # Should succeed despite no permissions

    @pytest.mark.asyncio
    async def test_get_task_status(self) -> None:
        """Test getting task status."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="test")
        task_id = await proxy.submit_task(request)
        adapter.set_task_status(task_id, TaskStatus.RUNNING)

        status = await proxy.get_task_status(task_id)
        assert status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_task_result(self) -> None:
        """Test getting task result."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="test")
        task_id = await proxy.submit_task(request)
        adapter.set_task_status(task_id, TaskStatus.COMPLETED)

        result = await proxy.get_task_result(task_id)
        assert result.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_task_result_sanitizes_secrets(self) -> None:
        """Test that result output is sanitized."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="test")
        task_id = await proxy.submit_task(request)
        adapter.set_task_result(
            task_id,
            TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output="API_KEY=sk-ant-abc123 password=secret",
            ),
        )

        result = await proxy.get_task_result(task_id)
        assert "sk-ant-abc123" not in (result.output or "")
        assert "[REDACTED]" in (result.output or "")

    @pytest.mark.asyncio
    async def test_cancel_task(self) -> None:
        """Test task cancellation."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="test")
        task_id = await proxy.submit_task(request)

        cancelled = await proxy.cancel_task(task_id)
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_stream_progress(self) -> None:
        """Test streaming progress updates."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(task_type="code", prompt="test")
        task_id = await proxy.submit_task(request)

        updates = []
        async for progress in proxy.stream_progress(task_id):
            updates.append(progress)

        assert len(updates) == 3
        assert updates[0].current_step == 1
        assert updates[2].current_step == 3

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health check passthrough."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        health = await proxy.health_check()
        assert health.healthy is True
        assert health.adapter_name == "mock"

    def test_metrics(self) -> None:
        """Test metrics tracking."""
        adapter = MockAdapter()
        context = MockAuthContext()
        proxy = ExternalAgentProxy(adapter, context)

        metrics = proxy.metrics
        assert "total_calls" in metrics
        assert "successful_calls" in metrics
        assert "timeout_errors" in metrics
        assert "policy_denials" in metrics
        assert "retry_count" in metrics
        assert "success_rate" in metrics

    @pytest.mark.asyncio
    async def test_tool_permission_denied(self) -> None:
        """Test submission denied for missing tool permission."""
        adapter = MockAdapter()
        context = MockAuthContext(permissions={"gateway.execute"})  # No tool perms
        proxy = ExternalAgentProxy(adapter, context)

        request = TaskRequest(
            task_type="code",
            prompt="Execute shell",
            tool_permissions={ToolPermission.SHELL_EXECUTE},
        )

        with pytest.raises(PolicyDeniedError) as exc_info:
            await proxy.submit_task(request)

        assert "shell_execute" in str(exc_info.value).lower()


class TestProxyRetry:
    """Tests for proxy retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry logic on transient failures."""
        adapter = MockAdapter()
        adapter._should_fail = True
        context = MockAuthContext()
        config = ProxyConfig(
            max_retries=2,
            retry_delay=0.01,
            enable_policy_checks=False,
        )
        proxy = ExternalAgentProxy(adapter, context, config)

        request = TaskRequest(task_type="code", prompt="test")

        with pytest.raises(RuntimeError, match="Mock submission failure"):
            await proxy.submit_task(request)

        # Should have retried
        assert proxy.metrics["retry_count"] > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test timeout on slow operations."""
        adapter = MockAdapter()
        context = MockAuthContext()
        config = ProxyConfig(
            submit_timeout=0.01,  # Very short timeout
            max_retries=1,
            enable_policy_checks=False,
        )
        proxy = ExternalAgentProxy(adapter, context, config)
        adapter._submit_delay = 1.0  # Longer than timeout

        request = TaskRequest(task_type="code", prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            await proxy.submit_task(request)

        assert proxy.metrics["timeout_errors"] > 0

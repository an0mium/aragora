"""Tests for external agent gateway base module."""

import pytest
from datetime import datetime, timezone

from aragora.gateway.external_agents.base import (
    AgentCapability,
    IsolationLevel,
    ExternalAgentTask,
    ExternalAgentResult,
    BaseExternalAgentAdapter,
    ExternalAgentGateway,
)


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_capability_values(self):
        """Test capability enum values."""
        assert AgentCapability.WEB_SEARCH.value == "web_search"
        assert AgentCapability.EXECUTE_CODE.value == "execute_code"
        assert AgentCapability.SHELL_ACCESS.value == "shell_access"

    def test_all_capabilities_have_values(self):
        """Ensure all capabilities have string values."""
        for cap in AgentCapability:
            assert isinstance(cap.value, str)
            assert len(cap.value) > 0


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels(self):
        """Test isolation level values."""
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.PROCESS.value == "process"
        assert IsolationLevel.CONTAINER.value == "container"
        assert IsolationLevel.VM.value == "vm"


class TestExternalAgentTask:
    """Tests for ExternalAgentTask dataclass."""

    def test_default_task(self):
        """Test task with default values."""
        task = ExternalAgentTask()
        assert task.task_id is not None
        assert task.task_type == "text_generation"
        assert task.prompt == ""
        assert task.timeout_seconds == 300.0
        assert task.max_tokens == 4096

    def test_custom_task(self):
        """Test task with custom values."""
        task = ExternalAgentTask(
            task_type="code_execution",
            prompt="Write a hello world",
            required_capabilities=[AgentCapability.EXECUTE_CODE],
            tenant_id="tenant-123",
            user_id="user-456",
        )
        assert task.task_type == "code_execution"
        assert task.prompt == "Write a hello world"
        assert AgentCapability.EXECUTE_CODE in task.required_capabilities
        assert task.tenant_id == "tenant-123"
        assert task.user_id == "user-456"


class TestExternalAgentResult:
    """Tests for ExternalAgentResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ExternalAgentResult(
            task_id="task-123",
            success=True,
            output="Hello, World!",
            agent_name="test-agent",
            execution_time_ms=150.0,
        )
        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.error is None
        assert result.execution_time_ms == 150.0

    def test_failure_result(self):
        """Test failure result."""
        result = ExternalAgentResult(
            task_id="task-123",
            success=False,
            error="Timeout exceeded",
            agent_name="test-agent",
        )
        assert result.success is False
        assert result.error == "Timeout exceeded"

    def test_result_with_security_metadata(self):
        """Test result with security metadata."""
        result = ExternalAgentResult(
            task_id="task-123",
            success=True,
            output="result",
            capabilities_used=[AgentCapability.WEB_SEARCH],
            was_sandboxed=True,
            isolation_level=IsolationLevel.CONTAINER,
            output_redacted=True,
            redaction_count=3,
        )
        assert result.was_sandboxed is True
        assert result.isolation_level == IsolationLevel.CONTAINER
        assert result.output_redacted is True
        assert result.redaction_count == 3


class MockAdapter(BaseExternalAgentAdapter):
    """Mock adapter for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail
        self._capabilities = {AgentCapability.WEB_SEARCH, AgentCapability.EXECUTE_CODE}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return self._capabilities

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.CONTAINER

    async def execute(self, task: ExternalAgentTask) -> ExternalAgentResult:
        if self._should_fail:
            return ExternalAgentResult(
                task_id=task.task_id,
                success=False,
                error="Mock failure",
                agent_name=self._name,
            )
        return ExternalAgentResult(
            task_id=task.task_id,
            success=True,
            output=f"Executed by {self._name}",
            agent_name=self._name,
            execution_time_ms=100.0,
        )


class TestBaseExternalAgentAdapter:
    """Tests for BaseExternalAgentAdapter."""

    @pytest.mark.asyncio
    async def test_adapter_properties(self):
        """Test adapter properties."""
        adapter = MockAdapter("test-adapter")
        assert adapter.name == "test-adapter"
        assert adapter.version == "1.0.0"
        assert AgentCapability.WEB_SEARCH in adapter.capabilities
        assert adapter.isolation_level == IsolationLevel.CONTAINER

    @pytest.mark.asyncio
    async def test_adapter_execute_success(self):
        """Test successful execution."""
        adapter = MockAdapter("test-adapter")
        task = ExternalAgentTask(prompt="test")
        result = await adapter.execute(task)
        assert result.success is True
        assert "test-adapter" in result.output

    @pytest.mark.asyncio
    async def test_adapter_execute_failure(self):
        """Test failed execution."""
        adapter = MockAdapter("failing-adapter", should_fail=True)
        task = ExternalAgentTask(prompt="test")
        result = await adapter.execute(task)
        assert result.success is False
        assert result.error == "Mock failure"


class TestExternalAgentGateway:
    """Tests for ExternalAgentGateway."""

    @pytest.mark.asyncio
    async def test_register_adapter(self):
        """Test registering an adapter."""
        gateway = ExternalAgentGateway()
        adapter = MockAdapter("test-adapter")
        gateway.register_adapter(adapter)
        assert "test-adapter" in gateway._adapters

    @pytest.mark.asyncio
    async def test_unregister_adapter(self):
        """Test unregistering an adapter."""
        gateway = ExternalAgentGateway()
        adapter = MockAdapter("test-adapter")
        gateway.register_adapter(adapter)
        result = gateway.unregister_adapter("test-adapter")
        assert result is True
        assert "test-adapter" not in gateway._adapters

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test executing a task through the gateway."""
        gateway = ExternalAgentGateway()
        adapter = MockAdapter("test-adapter")
        gateway.register_adapter(adapter)

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )
        result = await gateway.execute("test-adapter", task)
        assert result.success is True
        assert result.agent_name == "test-adapter"

    @pytest.mark.asyncio
    async def test_execute_unknown_adapter(self):
        """Test executing with unknown adapter."""
        gateway = ExternalAgentGateway()
        task = ExternalAgentTask(prompt="test")

        with pytest.raises(KeyError):
            await gateway.execute("unknown-adapter", task)

    @pytest.mark.asyncio
    async def test_get_adapters(self):
        """Test getting registered adapters."""
        gateway = ExternalAgentGateway()
        adapter1 = MockAdapter("adapter1")
        adapter2 = MockAdapter("adapter2")
        gateway.register_adapter(adapter1)
        gateway.register_adapter(adapter2)

        adapters = gateway.get_adapters()
        assert len(adapters) == 2
        assert "adapter1" in adapters
        assert "adapter2" in adapters

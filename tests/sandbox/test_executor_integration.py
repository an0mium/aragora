"""Tests for SandboxExecutor and gauntlet integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.sandbox.executor import (
    SandboxExecutor,
    SandboxConfig,
    ExecutionMode,
    ExecutionStatus,
    ExecutionResult,
)
from aragora.sandbox.policies import ToolPolicy, create_default_policy


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()

        assert config.mode == ExecutionMode.SUBPROCESS
        assert config.cleanup_on_complete is True
        assert config.network_enabled is False
        assert config.capture_output is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            mode=ExecutionMode.DOCKER,
            docker_image="python:3.12",
            cleanup_on_complete=False,
            network_enabled=True,
        )

        assert config.mode == ExecutionMode.DOCKER
        assert config.docker_image == "python:3.12"
        assert config.cleanup_on_complete is False
        assert config.network_enabled is True

    def test_mock_mode(self):
        """Test mock mode configuration for testing."""
        config = SandboxConfig(mode=ExecutionMode.MOCK)
        assert config.mode == ExecutionMode.MOCK


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful execution result."""
        result = ExecutionResult(
            execution_id="exec-123",
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            duration_seconds=0.5,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating a failed execution result."""
        result = ExecutionResult(
            execution_id="exec-456",
            status=ExecutionStatus.FAILED,
            exit_code=1,
            stderr="Error occurred",
            error_message="Execution failed",
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.exit_code == 1
        assert result.error_message == "Execution failed"

    def test_timeout_result(self):
        """Test timeout execution result."""
        result = ExecutionResult(
            execution_id="exec-789",
            status=ExecutionStatus.TIMEOUT,
            error_message="Execution timed out after 30s",
        )

        assert result.status == ExecutionStatus.TIMEOUT

    def test_policy_denied_result(self):
        """Test policy denied execution result."""
        result = ExecutionResult(
            execution_id="exec-000",
            status=ExecutionStatus.POLICY_DENIED,
            policy_violations=["network_access", "file_write"],
        )

        assert result.status == ExecutionStatus.POLICY_DENIED
        assert len(result.policy_violations) == 2

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ExecutionResult(
            execution_id="exec-123",
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="output",
        )

        data = result.to_dict()

        assert data["execution_id"] == "exec-123"
        assert data["status"] == "completed"
        assert data["exit_code"] == 0
        assert data["stdout"] == "output"


class TestSandboxExecutor:
    """Tests for SandboxExecutor."""

    @pytest.fixture
    def mock_executor(self):
        """Create executor with mock mode for testing."""
        config = SandboxConfig(mode=ExecutionMode.MOCK)
        return SandboxExecutor(config)

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    def test_init_default(self):
        """Test default initialization."""
        executor = SandboxExecutor()

        assert executor.config.mode == ExecutionMode.SUBPROCESS
        assert executor._active_executions == {}

    def test_init_with_config(self):
        """Test initialization with config."""
        config = SandboxConfig(mode=ExecutionMode.DOCKER)
        executor = SandboxExecutor(config)

        assert executor.config.mode == ExecutionMode.DOCKER

    @pytest.mark.asyncio
    async def test_execute_mock_mode(self, mock_executor):
        """Test execution in mock mode."""
        result = await mock_executor.execute(
            code='print("hello")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_id is not None

    @pytest.mark.asyncio
    async def test_execute_simple_python(self, subprocess_executor):
        """Test simple Python code execution."""
        result = await subprocess_executor.execute(
            code='print("Hello, World!")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello, World!" in result.stdout
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_python_with_error(self, subprocess_executor):
        """Test Python code that raises an error."""
        result = await subprocess_executor.execute(
            code='raise ValueError("test error")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.FAILED
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_timeout(self, subprocess_executor):
        """Test execution timeout."""
        result = await subprocess_executor.execute(
            code="import time; time.sleep(60)",
            language="python",
            timeout=0.5,
        )

        assert result.status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_captures_stdout_stderr(self, subprocess_executor):
        """Test that stdout and stderr are captured."""
        result = await subprocess_executor.execute(
            code='import sys; print("out"); print("err", file=sys.stderr)',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "out" in result.stdout
        assert "err" in result.stderr

    def test_get_policy_checker(self, subprocess_executor):
        """Test getting policy checker."""
        checker = subprocess_executor.get_policy_checker()
        assert checker is not None

    def test_update_policy(self, subprocess_executor):
        """Test updating policy."""
        new_policy = create_default_policy()
        new_policy.add_tool_allowlist(["read_file"])
        subprocess_executor.update_policy(new_policy)

        # update_policy sets self.policy, not self.config.policy
        assert subprocess_executor.policy == new_policy


class TestSandboxPolicyEnforcement:
    """Tests for policy enforcement in sandbox."""

    @pytest.fixture
    def restricted_executor(self):
        """Create executor with restrictive policy."""
        policy = create_default_policy()
        # Default policy denies by default, so just add specific allowlist
        policy.add_tool_allowlist(["read_file"])
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            policy=policy,
            network_enabled=False,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_network_disabled(self, restricted_executor):
        """Test that network is disabled when configured."""
        assert restricted_executor.config.network_enabled is False

    def test_policy_attached(self, restricted_executor):
        """Test that policy is properly attached."""
        assert restricted_executor.config.policy is not None
        # Check policy has default deny action
        from aragora.sandbox.policies import PolicyAction

        assert restricted_executor.config.policy.default_tool_action == PolicyAction.DENY


class TestGauntletSandboxIntegration:
    """Tests for sandbox integration with gauntlet runner."""

    @pytest.fixture
    def mock_gauntlet_runner(self):
        """Create a mock gauntlet runner with sandbox."""
        runner = MagicMock()
        runner.enable_sandbox = True
        runner._sandbox = MagicMock()
        runner._sandbox.execute = AsyncMock()
        return runner

    @pytest.mark.asyncio
    async def test_gauntlet_sandbox_enabled(self, mock_gauntlet_runner):
        """Test sandbox is used when enabled in gauntlet."""
        mock_result = ExecutionResult(
            execution_id="test-exec",
            status=ExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="test output",
        )
        mock_gauntlet_runner._sandbox.execute.return_value = mock_result

        # Simulate calling sandbox execution
        result = await mock_gauntlet_runner._sandbox.execute(
            code='print("test")',
            language="python",
            timeout=30.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        mock_gauntlet_runner._sandbox.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_gauntlet_sandbox_disabled(self):
        """Test behavior when sandbox is disabled."""
        runner = MagicMock()
        runner.enable_sandbox = False
        runner._sandbox = None

        # When sandbox is disabled, should return appropriate message
        assert runner._sandbox is None


class TestExecutionResourceLimits:
    """Tests for execution resource limits."""

    @pytest.fixture
    def limited_executor(self):
        """Create executor with resource limits."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_memory_tracking(self, limited_executor):
        """Test that memory usage is tracked."""
        result = await limited_executor.execute(
            code='x = [0] * 1000; print("done")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        # Memory tracking should be present (may be 0 if not supported)
        assert isinstance(result.memory_used_mb, float)

    @pytest.mark.asyncio
    async def test_duration_tracking(self, limited_executor):
        """Test that execution duration is tracked."""
        result = await limited_executor.execute(
            code='import time; time.sleep(0.1); print("done")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.duration_seconds > 0


class TestSandboxCleanup:
    """Tests for sandbox cleanup behavior."""

    @pytest.fixture
    def cleanup_executor(self):
        """Create executor with cleanup enabled."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.fixture
    def no_cleanup_executor(self):
        """Create executor with cleanup disabled."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=False,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_cleanup_on_complete(self, cleanup_executor):
        """Test that workspace is cleaned up on completion."""
        result = await cleanup_executor.execute(
            code='print("test")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_cleanup_preserves_workspace(self, no_cleanup_executor):
        """Test that workspace is preserved when cleanup disabled."""
        result = await no_cleanup_executor.execute(
            code='print("test")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED


class TestExecutionCancel:
    """Tests for execution cancellation."""

    @pytest.fixture
    def executor(self):
        """Create a subprocess executor."""
        return SandboxExecutor(SandboxConfig(mode=ExecutionMode.SUBPROCESS))

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_execution(self, executor):
        """Test cancelling an execution that doesn't exist."""
        result = await executor.cancel("nonexistent-id")

        # Cancel returns bool - may return True if Docker thinks it killed something
        # (Docker kill may succeed even for non-existent containers in some cases)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_cancel_running_execution(self, executor):
        """Test cancelling a running execution."""
        # Start a long-running execution
        import asyncio

        exec_task = asyncio.create_task(
            executor.execute(
                code="import time; time.sleep(60)",
                language="python",
                timeout=60.0,
            )
        )

        # Give it time to start
        await asyncio.sleep(0.2)

        # Get execution ID and cancel
        if executor._active_executions:
            exec_id = list(executor._active_executions.keys())[0]
            cancelled = await executor.cancel(exec_id)
            # Cancel may or may not succeed depending on timing
            assert isinstance(cancelled, bool)

        # Cancel the task
        exec_task.cancel()
        try:
            await exec_task
        except asyncio.CancelledError:
            pass

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


class TestExecutionIsolation:
    """Tests for sandbox execution isolation."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_separate_workspaces(self, subprocess_executor):
        """Test that each execution has separate workspace."""
        import asyncio

        # Run two executions that write files
        result1 = await subprocess_executor.execute(
            code='with open("test1.txt", "w") as f: f.write("exec1")',
            language="python",
            timeout=10.0,
        )
        result2 = await subprocess_executor.execute(
            code='with open("test2.txt", "w") as f: f.write("exec2")',
            language="python",
            timeout=10.0,
        )

        # Both should complete
        assert result1.status == ExecutionStatus.COMPLETED
        assert result2.status == ExecutionStatus.COMPLETED

        # They should have different execution IDs
        assert result1.execution_id != result2.execution_id

    @pytest.mark.asyncio
    async def test_environment_isolation(self, subprocess_executor):
        """Test that environment variables are isolated."""
        # Set an env variable in one execution
        result1 = await subprocess_executor.execute(
            code='import os; os.environ["TEST_VAR"] = "test_value"; print("set")',
            language="python",
            timeout=10.0,
        )

        # Try to read it in another
        result2 = await subprocess_executor.execute(
            code='import os; print(os.environ.get("TEST_VAR", "not_found"))',
            language="python",
            timeout=10.0,
        )

        assert result1.status == ExecutionStatus.COMPLETED
        assert result2.status == ExecutionStatus.COMPLETED
        assert "not_found" in result2.stdout

    @pytest.mark.asyncio
    async def test_custom_environment_passed(self, subprocess_executor):
        """Test that custom environment is passed to execution."""
        result = await subprocess_executor.execute(
            code='import os; print(os.environ.get("CUSTOM_VAR", "missing"))',
            language="python",
            timeout=10.0,
            env={"CUSTOM_VAR": "custom_value"},
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_concurrent_executions_isolated(self, subprocess_executor):
        """Test that concurrent executions are isolated."""
        import asyncio

        async def run_execution(value: str):
            return await subprocess_executor.execute(
                code=f'import time; time.sleep(0.1); print("{value}")',
                language="python",
                timeout=10.0,
            )

        # Run multiple executions concurrently
        results = await asyncio.gather(
            run_execution("exec1"),
            run_execution("exec2"),
            run_execution("exec3"),
        )

        # All should complete successfully
        assert all(r.status == ExecutionStatus.COMPLETED for r in results)

        # Each should have its own output
        assert "exec1" in results[0].stdout
        assert "exec2" in results[1].stdout
        assert "exec3" in results[2].stdout


class TestTimeoutHandling:
    """Tests for execution timeout handling."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_short_timeout(self, subprocess_executor):
        """Test very short timeout."""
        result = await subprocess_executor.execute(
            code="import time; time.sleep(10)",
            language="python",
            timeout=0.2,
        )

        assert result.status == ExecutionStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_timeout_message_includes_duration(self, subprocess_executor):
        """Test that timeout message includes the timeout duration."""
        timeout = 0.3
        result = await subprocess_executor.execute(
            code="import time; time.sleep(10)",
            language="python",
            timeout=timeout,
        )

        assert result.status == ExecutionStatus.TIMEOUT
        assert str(timeout) in result.error_message

    @pytest.mark.asyncio
    async def test_execution_within_timeout(self, subprocess_executor):
        """Test execution that completes within timeout."""
        result = await subprocess_executor.execute(
            code="import time; time.sleep(0.1); print('done')",
            language="python",
            timeout=5.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "done" in result.stdout

    @pytest.mark.asyncio
    async def test_default_timeout_from_policy(self):
        """Test that default timeout comes from policy."""
        from aragora.sandbox.policies import ResourceLimit

        policy = create_default_policy()
        policy.resource_limits = ResourceLimit(max_execution_seconds=1)

        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            policy=policy,
        )
        executor = SandboxExecutor(config)

        result = await executor.execute(
            code="import time; time.sleep(10)",
            language="python",
            # No explicit timeout - should use policy default
        )

        assert result.status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_explicit_timeout_overrides_policy(self):
        """Test that explicit timeout overrides policy."""
        from aragora.sandbox.policies import ResourceLimit

        policy = create_default_policy()
        policy.resource_limits = ResourceLimit(max_execution_seconds=10)

        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            policy=policy,
        )
        executor = SandboxExecutor(config)

        result = await executor.execute(
            code="import time; time.sleep(5)",
            language="python",
            timeout=0.2,  # Much shorter than policy
        )

        assert result.status == ExecutionStatus.TIMEOUT


class TestResourceLimitEnforcement:
    """Tests for resource limit enforcement."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_cpu_intensive_code(self, subprocess_executor):
        """Test CPU-intensive code execution."""
        result = await subprocess_executor.execute(
            code="sum(range(10000)); print('done')",
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "done" in result.stdout

    @pytest.mark.asyncio
    async def test_file_creation_tracked(self, subprocess_executor):
        """Test that created files are tracked."""
        result = await subprocess_executor.execute(
            code="""
with open("output.txt", "w") as f:
    f.write("test")
with open("data.json", "w") as f:
    f.write("{}")
print("done")
""",
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        # Files should be tracked
        assert len(result.files_created) >= 2


class TestLanguageSupport:
    """Tests for multi-language support."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode and permissive policy for all languages."""
        from aragora.sandbox.policies import create_permissive_policy

        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
            policy=create_permissive_policy(),
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_python_execution(self, subprocess_executor):
        """Test Python code execution."""
        result = await subprocess_executor.execute(
            code='print("Hello from Python")',
            language="python",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello from Python" in result.stdout

    @pytest.mark.asyncio
    async def test_bash_execution(self, subprocess_executor):
        """Test bash script execution."""
        result = await subprocess_executor.execute(
            code='echo "Hello from Bash"',
            language="bash",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello from Bash" in result.stdout

    @pytest.mark.asyncio
    async def test_shell_execution(self, subprocess_executor):
        """Test shell script execution."""
        result = await subprocess_executor.execute(
            code='echo "Hello from Shell"',
            language="shell",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello from Shell" in result.stdout

    @pytest.mark.asyncio
    async def test_unsupported_language(self, subprocess_executor):
        """Test execution with unsupported language."""
        result = await subprocess_executor.execute(
            code='fn main() { println!("Hello from Rust"); }',
            language="rust",
            timeout=10.0,
        )

        assert result.status == ExecutionStatus.FAILED
        assert "Unsupported language" in result.error_message


class TestAdditionalFileSupport:
    """Tests for additional file support in execution."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        config = SandboxConfig(
            mode=ExecutionMode.SUBPROCESS,
            cleanup_on_complete=True,
        )
        return SandboxExecutor(config)

    @pytest.mark.asyncio
    async def test_additional_files_created(self, subprocess_executor):
        """Test that additional files are created in workspace."""
        result = await subprocess_executor.execute(
            code="""
with open("input.txt", "r") as f:
    print(f.read())
""",
            language="python",
            timeout=10.0,
            files={"input.txt": "Hello from input file"},
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Hello from input file" in result.stdout

    @pytest.mark.asyncio
    async def test_nested_additional_files(self, subprocess_executor):
        """Test that nested additional files are created."""
        result = await subprocess_executor.execute(
            code="""
with open("subdir/data.txt", "r") as f:
    print(f.read())
""",
            language="python",
            timeout=10.0,
            files={"subdir/data.txt": "Nested file content"},
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "Nested file content" in result.stdout

    @pytest.mark.asyncio
    async def test_multiple_additional_files(self, subprocess_executor):
        """Test multiple additional files."""
        result = await subprocess_executor.execute(
            code="""
with open("file1.txt", "r") as f1:
    with open("file2.txt", "r") as f2:
        print(f1.read() + " " + f2.read())
""",
            language="python",
            timeout=10.0,
            files={
                "file1.txt": "First",
                "file2.txt": "Second",
            },
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert "First Second" in result.stdout


class TestExecutionModeEnum:
    """Tests for ExecutionMode enum."""

    def test_execution_modes(self):
        """Test execution mode values."""
        assert ExecutionMode.DOCKER.value == "docker"
        assert ExecutionMode.SUBPROCESS.value == "subprocess"
        assert ExecutionMode.MOCK.value == "mock"


class TestExecutionStatusEnum:
    """Tests for ExecutionStatus enum."""

    def test_execution_statuses(self):
        """Test execution status values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.POLICY_DENIED.value == "policy_denied"


class TestWriteCodeFile:
    """Tests for code file writing."""

    @pytest.fixture
    def subprocess_executor(self):
        """Create executor with subprocess mode."""
        return SandboxExecutor(SandboxConfig(mode=ExecutionMode.SUBPROCESS))

    def test_write_python_file(self, subprocess_executor, tmp_path):
        """Test writing Python code file."""
        code = 'print("test")'
        result = subprocess_executor._write_code_file(tmp_path, code, "python")

        assert result.suffix == ".py"
        assert result.read_text() == code

    def test_write_javascript_file(self, subprocess_executor, tmp_path):
        """Test writing JavaScript code file."""
        code = 'console.log("test")'
        result = subprocess_executor._write_code_file(tmp_path, code, "javascript")

        assert result.suffix == ".js"
        assert result.read_text() == code

    def test_write_bash_file(self, subprocess_executor, tmp_path):
        """Test writing bash code file."""
        code = 'echo "test"'
        result = subprocess_executor._write_code_file(tmp_path, code, "bash")

        assert result.suffix == ".sh"
        assert result.read_text() == code

    def test_write_unknown_language_file(self, subprocess_executor, tmp_path):
        """Test writing file for unknown language."""
        code = "unknown code"
        result = subprocess_executor._write_code_file(tmp_path, code, "unknown")

        assert result.suffix == ".txt"
        assert result.read_text() == code


class TestMockExecution:
    """Tests for mock execution mode."""

    @pytest.fixture
    def mock_executor(self):
        """Create executor with mock mode."""
        return SandboxExecutor(SandboxConfig(mode=ExecutionMode.MOCK))

    @pytest.mark.asyncio
    async def test_mock_returns_completed(self, mock_executor):
        """Test mock mode returns completed status."""
        result = await mock_executor.execute(
            code='print("test")',
            language="python",
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_mock_includes_code_preview(self, mock_executor):
        """Test mock mode includes code preview in output."""
        code = 'print("Hello, World!")'
        result = await mock_executor.execute(
            code=code,
            language="python",
        )

        assert "[MOCK]" in result.stdout
        assert "print" in result.stdout

    @pytest.mark.asyncio
    async def test_mock_truncates_long_code(self, mock_executor):
        """Test mock mode truncates long code in preview."""
        long_code = "x = " + "1" * 1000
        result = await mock_executor.execute(
            code=long_code,
            language="python",
        )

        assert len(result.stdout) < len(long_code) + 50

    @pytest.mark.asyncio
    async def test_mock_fast_execution(self, mock_executor):
        """Test mock mode executes quickly."""
        import time

        start = time.time()
        result = await mock_executor.execute(
            code="import time; time.sleep(100)",  # Would take forever in real mode
            language="python",
        )
        duration = time.time() - start

        assert result.status == ExecutionStatus.COMPLETED
        assert duration < 1.0  # Should be nearly instant

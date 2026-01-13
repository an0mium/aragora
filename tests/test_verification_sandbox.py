"""
Tests for ProofSandbox - isolated subprocess execution for formal verification.

Tests cover:
- Resource limits (timeout, memory, output truncation)
- Temporary directory cleanup
- Subprocess isolation (process groups, restricted environment)
- Error handling for malformed proofs
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.verification.sandbox import (
    ProofSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    run_sandboxed,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sandbox():
    """Create a sandbox with test configuration."""
    return ProofSandbox(timeout=5.0, memory_mb=64, max_output_bytes=1024)


@pytest.fixture
def sandbox_short_timeout():
    """Create a sandbox with very short timeout for timeout tests."""
    return ProofSandbox(timeout=0.1, memory_mb=64, max_output_bytes=1024)


@pytest.fixture
def temp_dir_tracker():
    """Track temp directories for cleanup verification."""
    created_dirs: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def tracked(*args, **kwargs):
        path = original_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(path))
        return path

    with patch("tempfile.mkdtemp", tracked):
        yield created_dirs


# ============================================================================
# Basic Configuration Tests
# ============================================================================


class TestSandboxConfiguration:
    """Tests for sandbox configuration."""

    def test_sandbox_default_config(self):
        """Sandbox has sensible defaults."""
        sandbox = ProofSandbox()
        assert sandbox.config.timeout_seconds == 30.0
        assert sandbox.config.memory_mb == 512
        assert sandbox.config.max_output_bytes == 1024 * 1024

    def test_sandbox_custom_config(self, sandbox):
        """Sandbox accepts custom configuration."""
        assert sandbox.config.timeout_seconds == 5.0
        assert sandbox.config.memory_mb == 64
        assert sandbox.config.max_output_bytes == 1024

    def test_sandbox_config_dataclass(self):
        """SandboxConfig dataclass is properly structured."""
        config = SandboxConfig(
            timeout_seconds=10.0,
            memory_mb=256,
            max_output_bytes=2048,
            cleanup_on_exit=False,
            allow_network=True,
        )
        assert config.timeout_seconds == 10.0
        assert config.memory_mb == 256
        assert config.max_output_bytes == 2048
        assert config.cleanup_on_exit is False
        assert config.allow_network is True


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_result_success_property(self):
        """is_success returns True only on SUCCESS with exit_code 0."""
        result = SandboxResult(status=SandboxStatus.SUCCESS, exit_code=0)
        assert result.is_success is True

    def test_result_success_property_nonzero_exit(self):
        """is_success returns False with non-zero exit code."""
        result = SandboxResult(status=SandboxStatus.SUCCESS, exit_code=1)
        assert result.is_success is False

    def test_result_success_property_timeout(self):
        """is_success returns False for TIMEOUT status."""
        result = SandboxResult(status=SandboxStatus.TIMEOUT, exit_code=0)
        assert result.is_success is False

    def test_result_default_values(self):
        """SandboxResult has correct default values."""
        result = SandboxResult(status=SandboxStatus.SUCCESS)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == -1
        assert result.execution_time_ms == 0.0
        assert result.memory_used_mb == 0.0
        assert result.error_message == ""


class TestSandboxStatus:
    """Tests for SandboxStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses exist."""
        assert hasattr(SandboxStatus, "SUCCESS")
        assert hasattr(SandboxStatus, "TIMEOUT")
        assert hasattr(SandboxStatus, "MEMORY_LIMIT")
        assert hasattr(SandboxStatus, "EXECUTION_ERROR")
        assert hasattr(SandboxStatus, "SETUP_FAILED")
        assert hasattr(SandboxStatus, "KILLED")

    def test_status_values(self):
        """Status values are strings."""
        assert SandboxStatus.SUCCESS.value == "success"
        assert SandboxStatus.TIMEOUT.value == "timeout"
        assert SandboxStatus.SETUP_FAILED.value == "setup_failed"


# ============================================================================
# Temporary Directory Cleanup Tests
# ============================================================================


class TestTempDirectoryCleanup:
    """Tests for temporary directory cleanup behavior."""

    def test_context_manager_cleanup(self, temp_dir_tracker):
        """Context manager ensures cleanup."""
        with ProofSandbox(timeout=1.0) as sandbox:
            temp_dir = sandbox._create_temp_dir()
            assert temp_dir.exists()
            temp_path = temp_dir

        # After context exit, dir should be cleaned
        assert not temp_path.exists()

    def test_explicit_cleanup(self, temp_dir_tracker):
        """Explicit cleanup() clears temp directories."""
        sandbox = ProofSandbox(timeout=1.0)
        temp_dir = sandbox._create_temp_dir()
        assert temp_dir.exists()

        sandbox.cleanup()
        assert not temp_dir.exists()
        assert sandbox._closed is True

    def test_cleanup_idempotent(self, sandbox):
        """cleanup() can be called multiple times safely."""
        temp_dir = sandbox._create_temp_dir()
        sandbox.cleanup()
        sandbox.cleanup()  # Should not raise
        assert sandbox._closed is True

    def test_cleanup_handles_missing_dir(self, sandbox):
        """Cleanup handles already-deleted directories gracefully."""
        temp_dir = sandbox._create_temp_dir()
        # Manually remove it
        temp_dir.rmdir()

        # Should not raise
        sandbox.cleanup()

    def test_multiple_temp_dirs_cleaned(self, sandbox):
        """All temporary directories are cleaned up."""
        dirs = [sandbox._create_temp_dir() for _ in range(3)]
        for d in dirs:
            assert d.exists()

        sandbox.cleanup()

        for d in dirs:
            assert not d.exists()


# ============================================================================
# Subprocess Isolation Tests
# ============================================================================


class TestSubprocessIsolation:
    """Tests for subprocess isolation security features."""

    @pytest.mark.asyncio
    async def test_new_session_process_group(self, sandbox):
        """Sandbox creates new process group (start_new_session=True)."""
        # We can verify this by checking that os.killpg is used in timeout handling
        # The sandbox uses start_new_session=True in create_subprocess_exec

        # Mock to verify the flag is passed
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await sandbox._run_subprocess(["echo", "test"])

            # Verify start_new_session=True was passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs.get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_restricted_path(self, sandbox):
        """PATH environment is restricted to essential directories."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await sandbox._run_subprocess(["echo", "test"])

            # Verify PATH was restricted
            call_kwargs = mock_exec.call_args.kwargs
            env = call_kwargs.get("env", {})
            assert env.get("PATH") == "/usr/local/bin:/usr/bin:/bin"

    @pytest.mark.asyncio
    async def test_network_disabled_by_default(self, sandbox):
        """Network access disabled when allow_network=False (default)."""
        assert sandbox.config.allow_network is False

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await sandbox._run_subprocess(["echo", "test"])

            call_kwargs = mock_exec.call_args.kwargs
            env = call_kwargs.get("env", {})
            assert env.get("no_proxy") == "*"
            assert env.get("NO_PROXY") == "*"

    @pytest.mark.asyncio
    async def test_preexec_fn_sets_limits(self, sandbox):
        """preexec_fn is provided to set resource limits."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await sandbox._run_subprocess(["echo", "test"])

            call_kwargs = mock_exec.call_args.kwargs
            assert "preexec_fn" in call_kwargs
            assert call_kwargs["preexec_fn"] is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in sandbox execution."""

    @pytest.mark.asyncio
    async def test_command_not_found(self, sandbox):
        """Missing command returns SETUP_FAILED."""
        result = await sandbox._run_subprocess(["/nonexistent/command"])
        assert result.status == SandboxStatus.SETUP_FAILED
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_permission_denied(self, sandbox):
        """Permission denied returns SETUP_FAILED."""
        # Create a file without execute permission
        temp_dir = sandbox._create_temp_dir()
        script = temp_dir / "script.sh"
        script.write_text("#!/bin/bash\necho test")
        script.chmod(0o644)  # No execute permission

        result = await sandbox._run_subprocess([str(script)])
        assert result.status == SandboxStatus.SETUP_FAILED
        assert "permission" in result.error_message.lower()

    def test_empty_code_input_z3(self, sandbox):
        """Empty code input handled gracefully for Z3."""
        # Z3 with empty input should not crash
        # It will fail at the solver level but not crash the sandbox

    @pytest.mark.asyncio
    async def test_invalid_z3_syntax(self, sandbox):
        """Invalid Z3 syntax returns execution result with error."""
        with patch("shutil.which", return_value="/usr/bin/z3"):
            with patch.object(sandbox, "_run_subprocess") as mock_run:
                mock_run.return_value = SandboxResult(
                    status=SandboxStatus.SUCCESS,
                    exit_code=1,
                    stderr='(error "line 1 column 1: unknown command")',
                )

                result = await sandbox.execute_z3("invalid smt code")
                assert result.exit_code != 0 or "error" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_z3_not_installed(self, sandbox):
        """Z3 not installed returns SETUP_FAILED."""
        with patch("shutil.which", return_value=None):
            result = await sandbox.execute_z3("(check-sat)")
            assert result.status == SandboxStatus.SETUP_FAILED
            assert "not installed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_lean_not_installed(self, sandbox):
        """Lean not installed returns SETUP_FAILED."""
        with patch("shutil.which", return_value=None):
            result = await sandbox.execute_lean("theorem test : True := trivial")
            assert result.status == SandboxStatus.SETUP_FAILED
            assert "not installed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_unknown_language(self, sandbox):
        """Unknown language returns SETUP_FAILED."""
        result = await sandbox.execute("code", language="unknown_lang")
        assert result.status == SandboxStatus.SETUP_FAILED
        assert "unknown language" in result.error_message.lower()


# ============================================================================
# Timeout Behavior Tests
# ============================================================================


class TestTimeoutBehavior:
    """Tests for timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_status(self, sandbox_short_timeout):
        """Timeout returns TIMEOUT status."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.pid = 12345

            async def slow_communicate(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout

            mock_process.communicate = slow_communicate
            mock_exec.return_value = mock_process

            with patch("os.killpg"):
                with patch("os.getpgid", return_value=12345):
                    result = await sandbox_short_timeout._run_subprocess(["sleep", "10"])

            assert result.status == SandboxStatus.TIMEOUT
            assert result.error_message != ""

    @pytest.mark.asyncio
    async def test_timeout_execution_time_recorded(self, sandbox):
        """Execution time is recorded in result."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"output", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            result = await sandbox._run_subprocess(["echo", "test"])

            assert result.execution_time_ms >= 0


# ============================================================================
# Output Handling Tests
# ============================================================================


class TestOutputHandling:
    """Tests for output truncation and handling."""

    @pytest.mark.asyncio
    async def test_output_truncation(self, sandbox):
        """Output is truncated at max_output_bytes."""
        large_output = b"x" * 2000  # Larger than max_output_bytes (1024)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(large_output, b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            result = await sandbox._run_subprocess(["echo", "test"])

            assert len(result.stdout) <= sandbox.config.max_output_bytes + 20  # +truncation msg
            assert "truncated" in result.stdout

    @pytest.mark.asyncio
    async def test_stderr_truncation(self, sandbox):
        """stderr is also truncated at max_output_bytes."""
        large_error = b"e" * 2000

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", large_error))
            mock_process.returncode = 1
            mock_exec.return_value = mock_process

            result = await sandbox._run_subprocess(["echo", "test"])

            assert len(result.stderr) <= sandbox.config.max_output_bytes + 20
            assert "truncated" in result.stderr

    @pytest.mark.asyncio
    async def test_unicode_handling(self, sandbox):
        """Non-UTF8 output is handled with replacement."""
        # Invalid UTF-8 bytes
        invalid_output = b"valid \xff\xfe invalid"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(invalid_output, b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            result = await sandbox._run_subprocess(["echo", "test"])

            # Should not raise, uses errors="replace"
            assert isinstance(result.stdout, str)


# ============================================================================
# Language Dispatcher Tests
# ============================================================================


class TestLanguageDispatcher:
    """Tests for the execute() language dispatcher."""

    @pytest.mark.asyncio
    async def test_z3_aliases(self, sandbox):
        """Z3 language aliases are recognized."""
        with patch.object(sandbox, "execute_z3", new_callable=AsyncMock) as mock_z3:
            mock_z3.return_value = SandboxResult(status=SandboxStatus.SUCCESS)

            for alias in ["z3", "smt", "smtlib", "smt2", "Z3", "SMT"]:
                await sandbox.execute("code", language=alias)
                mock_z3.assert_called()
                mock_z3.reset_mock()

    @pytest.mark.asyncio
    async def test_lean_aliases(self, sandbox):
        """Lean language aliases are recognized."""
        with patch.object(sandbox, "execute_lean", new_callable=AsyncMock) as mock_lean:
            mock_lean.return_value = SandboxResult(status=SandboxStatus.SUCCESS)

            for alias in ["lean", "lean4", "LEAN", "LEAN4"]:
                await sandbox.execute("code", language=alias)
                mock_lean.assert_called()
                mock_lean.reset_mock()


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunction:
    """Tests for run_sandboxed convenience function."""

    @pytest.mark.asyncio
    async def test_run_sandboxed_z3(self):
        """run_sandboxed works for Z3."""
        with patch("shutil.which", return_value=None):
            result = await run_sandboxed("(check-sat)", language="z3")
            assert result.status == SandboxStatus.SETUP_FAILED

    @pytest.mark.asyncio
    async def test_run_sandboxed_custom_params(self):
        """run_sandboxed accepts custom parameters."""
        with patch("shutil.which", return_value=None):
            result = await run_sandboxed(
                "code",
                language="z3",
                timeout=10.0,
                memory_mb=256,
            )
            assert result.status == SandboxStatus.SETUP_FAILED


# ============================================================================
# Resource Limits Tests
# ============================================================================


class TestResourceLimits:
    """Tests for resource limit configuration."""

    def test_set_resource_limits_function_exists(self, sandbox):
        """_set_resource_limits method exists."""
        assert hasattr(sandbox, "_set_resource_limits")
        assert callable(sandbox._set_resource_limits)

    def test_resource_limits_with_mock(self, sandbox):
        """Resource limits are set via setrlimit."""
        with patch("resource.setrlimit") as mock_setrlimit:
            # Call the function directly
            sandbox._set_resource_limits()

            # Verify setrlimit was called for various resources
            assert mock_setrlimit.call_count >= 1

    def test_resource_limit_errors_handled(self, sandbox):
        """Errors in setrlimit are handled gracefully."""
        with patch("resource.setrlimit", side_effect=ValueError("not permitted")):
            # Should not raise
            sandbox._set_resource_limits()


# ============================================================================
# Integration-style Tests (with mocks)
# ============================================================================


class TestIntegrationWithMocks:
    """Integration-style tests using mocks for external dependencies."""

    @pytest.mark.asyncio
    async def test_full_z3_execution_flow(self, sandbox):
        """Full Z3 execution flow with mocked subprocess."""
        z3_code = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"

        with patch("shutil.which", return_value="/usr/bin/z3"):
            with patch.object(sandbox, "_run_subprocess") as mock_run:
                mock_run.return_value = SandboxResult(
                    status=SandboxStatus.SUCCESS,
                    stdout="sat\n",
                    stderr="",
                    exit_code=0,
                    execution_time_ms=50.0,
                )

                result = await sandbox.execute_z3(z3_code)

                assert result.status == SandboxStatus.SUCCESS
                assert result.stdout == "sat\n"
                assert result.is_success is True

    @pytest.mark.asyncio
    async def test_full_lean_execution_flow(self, sandbox):
        """Full Lean execution flow with mocked subprocess."""
        lean_code = "theorem test : True := trivial"

        with patch("shutil.which", return_value="/usr/bin/lean"):
            with patch.object(sandbox, "_run_subprocess") as mock_run:
                mock_run.return_value = SandboxResult(
                    status=SandboxStatus.SUCCESS,
                    stdout="",
                    stderr="",
                    exit_code=0,
                    execution_time_ms=100.0,
                )

                result = await sandbox.execute_lean(lean_code)

                assert result.status == SandboxStatus.SUCCESS
                assert result.is_success is True

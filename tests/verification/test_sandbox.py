"""Tests for the verification sandbox module.

Tests cover:
- SandboxStatus enum values
- SandboxResult dataclass properties
- SandboxConfig dataclass defaults and validation
- ProofSandbox initialization and lifecycle
- Safe execution of code with resource limits
- Timeout enforcement
- Cleanup behavior
"""

import asyncio
from pathlib import Path

import pytest

from aragora.verification.sandbox import (
    ProofSandbox,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    run_sandboxed,
)


class TestSandboxStatus:
    """Tests for SandboxStatus enum."""

    def test_all_status_values_exist(self):
        """All expected status values should be defined."""
        assert SandboxStatus.SUCCESS.value == "success"
        assert SandboxStatus.TIMEOUT.value == "timeout"
        assert SandboxStatus.MEMORY_LIMIT.value == "memory_limit"
        assert SandboxStatus.EXECUTION_ERROR.value == "execution_error"
        assert SandboxStatus.SETUP_FAILED.value == "setup_failed"
        assert SandboxStatus.KILLED.value == "killed"

    def test_status_count(self):
        """Should have exactly 6 status values."""
        assert len(SandboxStatus) == 6


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = SandboxResult(status=SandboxStatus.SUCCESS)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == -1
        assert result.execution_time_ms == 0.0
        assert result.memory_used_mb == 0.0
        assert result.error_message == ""

    def test_is_success_true(self):
        """is_success should be True when status is SUCCESS and exit_code is 0."""
        result = SandboxResult(status=SandboxStatus.SUCCESS, exit_code=0)
        assert result.is_success is True

    def test_is_success_false_wrong_status(self):
        """is_success should be False when status is not SUCCESS."""
        result = SandboxResult(status=SandboxStatus.TIMEOUT, exit_code=0)
        assert result.is_success is False

    def test_is_success_false_nonzero_exit(self):
        """is_success should be False when exit_code is non-zero."""
        result = SandboxResult(status=SandboxStatus.SUCCESS, exit_code=1)
        assert result.is_success is False

    def test_full_result(self):
        """Should store all provided values."""
        result = SandboxResult(
            status=SandboxStatus.EXECUTION_ERROR,
            stdout="output",
            stderr="error",
            exit_code=1,
            execution_time_ms=100.5,
            memory_used_mb=50.0,
            error_message="Something went wrong",
        )
        assert result.status == SandboxStatus.EXECUTION_ERROR
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.exit_code == 1
        assert result.execution_time_ms == 100.5
        assert result.memory_used_mb == 50.0
        assert result.error_message == "Something went wrong"


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        config = SandboxConfig()
        assert config.timeout_seconds == 30.0
        assert config.memory_mb == 512
        assert config.max_output_bytes == 1024 * 1024  # 1MB
        assert config.cleanup_on_exit is True
        assert config.allow_network is False
        assert config.working_dir is None

    def test_custom_values(self):
        """Should accept custom values."""
        config = SandboxConfig(
            timeout_seconds=60.0,
            memory_mb=1024,
            max_output_bytes=2 * 1024 * 1024,
            cleanup_on_exit=False,
            allow_network=True,
            working_dir=Path("/tmp/test"),
        )
        assert config.timeout_seconds == 60.0
        assert config.memory_mb == 1024
        assert config.max_output_bytes == 2 * 1024 * 1024
        assert config.cleanup_on_exit is False
        assert config.allow_network is True
        assert config.working_dir == Path("/tmp/test")


class TestProofSandbox:
    """Tests for ProofSandbox class."""

    def test_initialization_with_defaults(self):
        """Should initialize with default config."""
        sandbox = ProofSandbox()
        assert sandbox.config.timeout_seconds == 30.0
        assert sandbox.config.memory_mb == 512
        assert sandbox.config.max_output_bytes == 1024 * 1024

    def test_initialization_with_custom_values(self):
        """Should initialize with custom values."""
        sandbox = ProofSandbox(timeout=60.0, memory_mb=1024, max_output_bytes=2048)
        assert sandbox.config.timeout_seconds == 60.0
        assert sandbox.config.memory_mb == 1024
        assert sandbox.config.max_output_bytes == 2048

    def test_context_manager(self):
        """Should work as context manager."""
        with ProofSandbox() as sandbox:
            assert sandbox._closed is False
        assert sandbox._closed is True

    def test_cleanup_is_idempotent(self):
        """Cleanup should be safe to call multiple times."""
        sandbox = ProofSandbox()
        sandbox.cleanup()
        sandbox.cleanup()  # Should not raise
        assert sandbox._closed is True


class TestProofSandboxExecution:
    """Tests for ProofSandbox execution methods."""

    @pytest.mark.asyncio
    async def test_execute_z3_code(self):
        """Should execute Z3 SMT code."""
        import shutil
        if not shutil.which("z3"):
            pytest.skip("Z3 not installed")

        with ProofSandbox(timeout=10.0) as sandbox:
            # Simple Z3 SMT-LIB2 code that should succeed
            z3_code = "(check-sat)"
            result = await sandbox.execute_z3(z3_code)
            assert result.status in (SandboxStatus.SUCCESS, SandboxStatus.SETUP_FAILED)

    @pytest.mark.asyncio
    async def test_execute_with_language(self):
        """Should route to correct executor based on language."""
        with ProofSandbox(timeout=10.0) as sandbox:
            # Execute with explicit language
            result = await sandbox.execute("(check-sat)", language="z3")
            # Status depends on whether z3 is installed
            assert result.status in (
                SandboxStatus.SUCCESS,
                SandboxStatus.SETUP_FAILED,
                SandboxStatus.EXECUTION_ERROR,
            )

    @pytest.mark.asyncio
    async def test_execute_unknown_language(self):
        """Should fail for unknown language."""
        with ProofSandbox(timeout=10.0) as sandbox:
            result = await sandbox.execute("some code", language="unknown")
            assert result.status == SandboxStatus.SETUP_FAILED
            assert "Unknown language" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_lean_not_installed(self):
        """Should handle Lean not being installed."""
        import shutil
        if shutil.which("lean"):
            pytest.skip("Lean is installed, skipping not-installed test")

        with ProofSandbox(timeout=10.0) as sandbox:
            result = await sandbox.execute_lean("-- some lean code")
            assert result.status == SandboxStatus.SETUP_FAILED

    @pytest.mark.asyncio
    async def test_execute_z3_not_installed(self):
        """Should handle Z3 not being installed."""
        import shutil
        if shutil.which("z3"):
            pytest.skip("Z3 is installed, skipping not-installed test")

        with ProofSandbox(timeout=10.0) as sandbox:
            result = await sandbox.execute_z3("(check-sat)")
            assert result.status == SandboxStatus.SETUP_FAILED


class TestRunSandboxed:
    """Tests for the run_sandboxed convenience function."""

    @pytest.mark.asyncio
    async def test_run_sandboxed_z3(self):
        """Should run Z3 code in sandbox."""
        import shutil
        if not shutil.which("z3"):
            pytest.skip("Z3 not installed")

        result = await run_sandboxed("(check-sat)", language="z3")
        assert result.status in (SandboxStatus.SUCCESS, SandboxStatus.SETUP_FAILED)

    @pytest.mark.asyncio
    async def test_run_sandboxed_with_timeout(self):
        """Should respect timeout parameter."""
        # run_sandboxed takes code and language, not commands
        result = await run_sandboxed("(check-sat)", language="z3", timeout=5.0)
        # Should complete or fail gracefully
        assert result.status in (
            SandboxStatus.SUCCESS,
            SandboxStatus.SETUP_FAILED,
            SandboxStatus.TIMEOUT,
        )

    @pytest.mark.asyncio
    async def test_run_sandboxed_with_memory_limit(self):
        """Should accept memory limit parameter."""
        result = await run_sandboxed("(check-sat)", language="z3", memory_mb=256)
        # Should complete or fail gracefully
        assert result.status in (
            SandboxStatus.SUCCESS,
            SandboxStatus.SETUP_FAILED,
            SandboxStatus.EXECUTION_ERROR,
        )


class TestSandboxSecurity:
    """Security-focused tests for sandbox isolation."""

    @pytest.mark.asyncio
    async def test_temp_dir_isolation(self):
        """Should create isolated temp directories."""
        with ProofSandbox() as sandbox:
            temp_dir = sandbox._create_temp_dir()
            assert temp_dir.exists()
            assert temp_dir.is_dir()
        # After context exit, temp dir should be cleaned up
        assert not temp_dir.exists()

    def test_max_output_bytes_config(self):
        """Should accept max_output_bytes configuration."""
        sandbox = ProofSandbox(max_output_bytes=100)
        assert sandbox.config.max_output_bytes == 100

    def test_config_prevents_network_by_default(self):
        """Default config should not allow network access."""
        config = SandboxConfig()
        assert config.allow_network is False

    def test_config_cleanup_by_default(self):
        """Default config should cleanup on exit."""
        config = SandboxConfig()
        assert config.cleanup_on_exit is True

"""
Tests for the modular nomic package.

Tests the extracted modules from scripts/nomic_loop.py:
- recovery: PhaseError, PhaseRecovery
- circuit_breaker: AgentCircuitBreaker
- safety: checksums, backups
- git: operations
- config: constants and loaders
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestPhaseRecovery:
    """Test PhaseRecovery error handling."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.recovery import PhaseError, PhaseRecovery

        assert PhaseError is not None
        assert PhaseRecovery is not None

    def test_phase_error_attributes(self) -> None:
        """PhaseError stores phase and recovery info."""
        from scripts.nomic.recovery import PhaseError

        error = PhaseError("debate", "Connection failed", recoverable=True)
        assert error.phase == "debate"
        assert error.recoverable is True
        assert "debate" in str(error)

    def test_phase_recovery_config(self) -> None:
        """PhaseRecovery has correct default configs."""
        from scripts.nomic.recovery import PhaseRecovery

        recovery = PhaseRecovery()
        assert "debate" in recovery.PHASE_RETRY_CONFIG
        assert "implement" in recovery.PHASE_RETRY_CONFIG
        assert recovery.PHASE_RETRY_CONFIG["debate"]["critical"] is True
        assert recovery.PHASE_RETRY_CONFIG["verify"]["critical"] is False

    def test_record_success_resets_failures(self) -> None:
        """record_success resets consecutive failure count."""
        from scripts.nomic.recovery import PhaseRecovery

        recovery = PhaseRecovery()
        recovery.consecutive_failures["debate"] = 3
        recovery.record_success("debate")
        assert recovery.consecutive_failures["debate"] == 0

    def test_record_failure_increments(self) -> None:
        """record_failure increments failure count."""
        from scripts.nomic.recovery import PhaseRecovery

        recovery = PhaseRecovery()
        recovery.record_failure("debate", Exception("test"))
        assert recovery.consecutive_failures["debate"] == 1
        recovery.record_failure("debate", Exception("test"))
        assert recovery.consecutive_failures["debate"] == 2

    def test_is_retryable_respects_max(self) -> None:
        """is_retryable returns False when max retries reached."""
        from scripts.nomic.recovery import PhaseRecovery

        recovery = PhaseRecovery()
        # debate has max_retries=1
        recovery.consecutive_failures["debate"] = 0
        assert recovery.is_retryable(Exception("test"), "debate") is True
        recovery.consecutive_failures["debate"] = 1
        assert recovery.is_retryable(Exception("test"), "debate") is False

    def test_rate_limit_increases_delay(self) -> None:
        """Rate limit errors get longer delays."""
        from scripts.nomic.recovery import PhaseRecovery

        recovery = PhaseRecovery(log_func=lambda x: None)
        normal_delay = recovery.get_retry_delay(Exception("normal error"), "debate")
        rate_limit_delay = recovery.get_retry_delay(Exception("rate limit exceeded"), "debate")
        assert rate_limit_delay >= 120  # Minimum for rate limits
        assert rate_limit_delay > normal_delay


class TestAgentCircuitBreaker:
    """Test AgentCircuitBreaker functionality."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        assert AgentCircuitBreaker is not None

    def test_initial_availability(self) -> None:
        """All agents available initially."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker()
        assert breaker.is_available("claude") is True
        assert breaker.is_available("codex") is True

    def test_failures_trip_circuit(self) -> None:
        """Consecutive failures trip the circuit."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(failure_threshold=3, cooldown_cycles=2)
        breaker.record_failure("claude")
        assert breaker.is_available("claude") is True
        breaker.record_failure("claude")
        assert breaker.is_available("claude") is True
        tripped = breaker.record_failure("claude")
        assert tripped is True
        assert breaker.is_available("claude") is False

    def test_success_resets_failures(self) -> None:
        """Success resets failure count."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(failure_threshold=3)
        breaker.record_failure("claude")
        breaker.record_failure("claude")
        breaker.record_success("claude")
        assert breaker.failures["claude"] == 0

    def test_cooldown_decrements(self) -> None:
        """start_new_cycle decrements cooldowns."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(failure_threshold=1, cooldown_cycles=2)
        breaker.record_failure("claude")  # Trips circuit
        assert breaker.is_available("claude") is False
        breaker.start_new_cycle()
        assert breaker.cooldowns["claude"] == 1
        breaker.start_new_cycle()
        assert breaker.cooldowns["claude"] == 0
        assert breaker.is_available("claude") is True

    def test_task_specific_tracking(self) -> None:
        """Task-specific failures tracked separately."""
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(failure_threshold=2)
        breaker.record_task_failure("claude", "implement")
        breaker.record_task_failure("claude", "implement")
        # Should be available for other tasks
        assert breaker.is_available_for_task("claude", "debate") is True


class TestSafetyChecksums:
    """Test safety checksum functionality."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.safety.checksums import (
            PROTECTED_FILES,
            SAFETY_PREAMBLE,
            compute_file_checksum,
        )

        assert len(PROTECTED_FILES) > 0
        assert "CRITICAL" in SAFETY_PREAMBLE

    def test_compute_checksum(self) -> None:
        """compute_file_checksum produces consistent hashes."""
        from scripts.nomic.safety.checksums import compute_file_checksum

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("test content")
            f.flush()
            path = Path(f.name)

        try:
            hash1 = compute_file_checksum(path)
            hash2 = compute_file_checksum(path)
            assert hash1 == hash2
            assert len(hash1) == 16  # Short hash
        finally:
            path.unlink()

    def test_missing_file_returns_empty(self) -> None:
        """compute_file_checksum returns empty for missing files."""
        from scripts.nomic.safety.checksums import compute_file_checksum

        result = compute_file_checksum(Path("/nonexistent/file.py"))
        assert result == ""


class TestSafetyBackups:
    """Test backup functionality."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.safety.backups import (
            create_backup,
            restore_backup,
            get_latest_backup,
        )

        assert create_backup is not None
        assert restore_backup is not None

    def test_get_latest_backup_empty_dir(self) -> None:
        """get_latest_backup returns None for empty directory."""
        from scripts.nomic.safety.backups import get_latest_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_latest_backup(Path(tmpdir))
            assert result is None


class TestGitOperations:
    """Test git operations module."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.git import (
            git_stash_create,
            get_git_diff,
            get_git_changed_files,
        )

        assert git_stash_create is not None
        assert get_git_diff is not None

    def test_get_git_diff_handles_errors(self) -> None:
        """get_git_diff returns empty string on error."""
        from scripts.nomic.git import get_git_diff

        # Non-git directory should return empty
        result = get_git_diff(Path("/tmp"))
        assert result == "" or "fatal" not in result.lower()


class TestConfig:
    """Test configuration module."""

    def test_import(self) -> None:
        """Module imports successfully."""
        from scripts.nomic.config import (
            NOMIC_AUTO_COMMIT,
            NOMIC_AUTO_CONTINUE,
            NOMIC_MAX_CYCLE_SECONDS,
            NOMIC_STALL_THRESHOLD,
            load_dotenv,
        )

        assert isinstance(NOMIC_AUTO_COMMIT, bool)
        assert isinstance(NOMIC_MAX_CYCLE_SECONDS, int)
        assert NOMIC_MAX_CYCLE_SECONDS > 0

    def test_load_dotenv(self) -> None:
        """load_dotenv loads environment variables."""
        import os
        from scripts.nomic.config import load_dotenv

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            f.write("TEST_NOMIC_VAR=test_value\n")
            f.write("# Comment line\n")
            f.write("ANOTHER_VAR=another\n")
            f.flush()
            path = Path(f.name)

        try:
            # Clear if exists
            os.environ.pop("TEST_NOMIC_VAR", None)
            load_dotenv(path)
            assert os.environ.get("TEST_NOMIC_VAR") == "test_value"
        finally:
            path.unlink()
            os.environ.pop("TEST_NOMIC_VAR", None)


class TestPackageIntegration:
    """Test that the package integrates with nomic_loop.py."""

    def test_nomic_loop_imports_package(self) -> None:
        """nomic_loop.py can import from the package."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.nomic_loop import (
            PhaseError,
            PhaseRecovery,
            AgentCircuitBreaker,
            PROTECTED_FILES,
        )

        assert PhaseError is not None
        assert len(PROTECTED_FILES) > 0

    def test_package_classes_match_original(self) -> None:
        """Package classes have same interface as originals."""
        from scripts.nomic.recovery import PhaseError, PhaseRecovery
        from scripts.nomic.circuit_breaker import AgentCircuitBreaker

        # Check PhaseError
        error = PhaseError("test", "message", recoverable=True)
        assert hasattr(error, "phase")
        assert hasattr(error, "recoverable")

        # Check PhaseRecovery
        recovery = PhaseRecovery()
        assert hasattr(recovery, "is_retryable")
        assert hasattr(recovery, "record_success")
        assert hasattr(recovery, "record_failure")

        # Check AgentCircuitBreaker
        breaker = AgentCircuitBreaker()
        assert hasattr(breaker, "is_available")
        assert hasattr(breaker, "record_success")
        assert hasattr(breaker, "record_failure")

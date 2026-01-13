"""Tests for PhaseRecovery class in nomic_loop.py."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

import sys
from pathlib import Path

# Add scripts directory to path for nomic_loop import
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from nomic_loop import PhaseRecovery


class TestPhaseRecoveryRetryLogic:
    """Tests for retry mechanism."""

    def test_is_retryable_returns_true_initially(self):
        """First failure should be retryable for phases with retries."""
        recovery = PhaseRecovery()
        error = ValueError("some error")

        # Context has max_retries=2, should be retryable initially
        assert recovery.is_retryable(error, "context") is True

    def test_is_retryable_false_after_max_retries(self):
        """Should not retry after max retries exceeded."""
        recovery = PhaseRecovery()
        error = ValueError("some error")

        # Context has max_retries=2
        recovery.record_failure("context", error)
        recovery.record_failure("context", error)

        assert recovery.is_retryable(error, "context") is False

    def test_non_retryable_errors_return_false(self):
        """KeyboardInterrupt, SystemExit, MemoryError should not be retried."""
        recovery = PhaseRecovery()

        assert recovery.is_retryable(KeyboardInterrupt(), "context") is False
        assert recovery.is_retryable(SystemExit(), "context") is False
        assert recovery.is_retryable(MemoryError(), "context") is False

    def test_unknown_phase_has_default_retry(self):
        """Unknown phases should have default max_retries=1."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # First attempt retryable
        assert recovery.is_retryable(error, "unknown_phase") is True

        # After one failure, should not retry
        recovery.record_failure("unknown_phase", error)
        assert recovery.is_retryable(error, "unknown_phase") is False


class TestPhaseRecoveryExponentialBackoff:
    """Tests for exponential backoff calculation."""

    def test_base_delay_for_each_phase(self):
        """Each phase has correct base delay."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Context has base_delay=5
        assert recovery.get_retry_delay(error, "context") == 5

        # Debate has base_delay=10
        assert recovery.get_retry_delay(error, "debate") == 10

        # Implement has base_delay=15
        assert recovery.get_retry_delay(error, "implement") == 15

    def test_exponential_increase_with_failures(self):
        """Delay doubles with each failure."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Context: base=5
        # 0 failures: 5 * 2^0 = 5
        assert recovery.get_retry_delay(error, "context") == 5

        recovery.record_failure("context", error)
        # 1 failure: 5 * 2^1 = 10
        assert recovery.get_retry_delay(error, "context") == 10

        recovery.record_failure("context", error)
        # 2 failures: 5 * 2^2 = 20
        assert recovery.get_retry_delay(error, "context") == 20

    def test_delay_caps_at_300_seconds(self):
        """Delay should not exceed 5 minutes."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Simulate many failures to exceed cap
        for _ in range(10):
            recovery.record_failure("context", error)

        delay = recovery.get_retry_delay(error, "context")
        assert delay == 300  # 5 minutes max

    def test_rate_limit_triggers_60s_minimum(self):
        """Rate limit errors should wait at least 60 seconds."""
        recovery = PhaseRecovery()

        # Various rate limit error messages
        rate_limit_errors = [
            ValueError("Rate limit exceeded"),
            ValueError("Error 429: Too many requests"),
            ValueError("quota exceeded for today"),
            ValueError("Resource exhausted"),
        ]

        for error in rate_limit_errors:
            delay = recovery.get_retry_delay(error, "context")
            assert delay >= 60, f"Rate limit error '{error}' should have delay >= 60"


class TestPhaseRecoveryRollbackTrigger:
    """Tests for rollback trigger logic."""

    def test_critical_phase_triggers_rollback_at_2_failures(self):
        """Critical phases (debate, implement, commit) trigger rollback at 2 failures."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Debate is critical
        assert recovery.should_trigger_rollback("debate") is False  # 0 failures
        recovery.record_failure("debate", error)
        assert recovery.should_trigger_rollback("debate") is False  # 1 failure
        recovery.record_failure("debate", error)
        assert recovery.should_trigger_rollback("debate") is True  # 2 failures

    def test_non_critical_phase_never_triggers_rollback(self):
        """Non-critical phases (context, design, verify) never trigger rollback."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Context is non-critical
        for _ in range(10):
            recovery.record_failure("context", error)

        assert recovery.should_trigger_rollback("context") is False

        # Design is non-critical
        for _ in range(10):
            recovery.record_failure("design", error)

        assert recovery.should_trigger_rollback("design") is False

        # Verify is non-critical
        for _ in range(10):
            recovery.record_failure("verify", error)

        assert recovery.should_trigger_rollback("verify") is False

    def test_implement_and_commit_are_critical(self):
        """Implement and commit phases are critical."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Implement
        recovery.record_failure("implement", error)
        recovery.record_failure("implement", error)
        assert recovery.should_trigger_rollback("implement") is True

        # Commit
        recovery.record_failure("commit", error)
        recovery.record_failure("commit", error)
        assert recovery.should_trigger_rollback("commit") is True

    def test_success_resets_failure_counter(self):
        """Success should reset consecutive failure count."""
        recovery = PhaseRecovery()
        error = ValueError("error")

        # Build up failures
        recovery.record_failure("debate", error)
        assert recovery.consecutive_failures["debate"] == 1

        # Success resets
        recovery.record_success("debate")
        assert recovery.consecutive_failures["debate"] == 0

        # Rollback should not trigger
        recovery.record_failure("debate", error)
        assert recovery.should_trigger_rollback("debate") is False


class TestPhaseRecoveryHealthReport:
    """Tests for health report tracking."""

    def test_health_report_structure(self):
        """Health report should have expected structure."""
        recovery = PhaseRecovery()
        report = recovery.get_health_report()

        assert "phase_health" in report
        assert "consecutive_failures" in report
        assert isinstance(report["phase_health"], dict)
        assert isinstance(report["consecutive_failures"], dict)

    def test_health_report_tracks_successes(self):
        """Health report should track successes."""
        recovery = PhaseRecovery()
        recovery.record_success("context")
        recovery.record_success("context")
        recovery.record_success("debate")

        report = recovery.get_health_report()

        assert report["phase_health"]["context"]["successes"] == 2
        assert report["phase_health"]["debate"]["successes"] == 1

    def test_health_report_tracks_failures(self):
        """Health report should track failures and last error."""
        recovery = PhaseRecovery()
        error1 = ValueError("first error")
        error2 = ValueError("second error")

        recovery.record_failure("context", error1)
        recovery.record_failure("context", error2)

        report = recovery.get_health_report()

        assert report["phase_health"]["context"]["failures"] == 2
        assert "second error" in report["phase_health"]["context"]["last_error"]

    def test_error_message_truncated_to_200_chars(self):
        """Long error messages should be truncated."""
        recovery = PhaseRecovery()
        long_error = ValueError("x" * 500)

        recovery.record_failure("context", long_error)

        report = recovery.get_health_report()
        assert len(report["phase_health"]["context"]["last_error"]) <= 200


class TestPhaseRecoveryRunWithRecovery:
    """Tests for the async run_with_recovery method."""

    @pytest.mark.asyncio
    async def test_success_path_returns_result(self):
        """Successful execution returns (True, result)."""
        recovery = PhaseRecovery(log_func=Mock())

        async def success_func():
            return "success_value"

        success, result = await recovery.run_with_recovery("context", success_func)

        assert success is True
        assert result == "success_value"
        assert recovery.phase_health["context"]["successes"] == 1

    @pytest.mark.asyncio
    async def test_failure_after_retries_returns_error(self):
        """Failure after retries returns (False, error_message)."""
        recovery = PhaseRecovery(log_func=Mock())

        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("test error")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            success, result = await recovery.run_with_recovery("context", failing_func)

        assert success is False
        assert "test error" in result
        # Context has max_retries=2, attempts continue while attempts <= max_retries
        # After each failure, consecutive_failures increments
        # is_retryable checks failures >= max_retries
        # So: attempt 1 -> fail -> failures=1, 1>=2=False, retryable
        #     attempt 2 -> fail -> failures=2, 2>=2=True, not retryable
        # Total: 2 attempts
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_count_matches_config(self):
        """Number of retries matches phase config."""
        recovery = PhaseRecovery(log_func=Mock())

        # Debate has max_retries=1
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await recovery.run_with_recovery("debate", failing_func)

        # Debate has max_retries=1
        # After first failure, failures=1, 1>=1=True, not retryable
        # Total: 1 attempt
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Function succeeds after initial failure."""
        recovery = PhaseRecovery(log_func=Mock())

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient error")
            return "recovered"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            success, result = await recovery.run_with_recovery("context", flaky_func)

        assert success is True
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        """Non-retryable errors (KeyboardInterrupt) are re-raised."""
        recovery = PhaseRecovery(log_func=Mock())

        async def interrupt_func():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            await recovery.run_with_recovery("context", interrupt_func)

    @pytest.mark.asyncio
    async def test_rollback_message_logged_for_critical_phase(self):
        """Critical phase failure logs rollback message after 2 consecutive failures."""
        log_calls = []
        recovery = PhaseRecovery(log_func=lambda msg: log_calls.append(msg))

        async def failing_func():
            raise ValueError("critical error")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Debate is critical with max_retries=1
            # Need 2 consecutive failures to trigger rollback
            # First call: 1 failure
            await recovery.run_with_recovery("debate", failing_func)
            # Second call: 2 failures total
            await recovery.run_with_recovery("debate", failing_func)

        # Should log rollback message after 2nd failure
        rollback_logged = any("rollback" in msg.lower() for msg in log_calls)
        assert rollback_logged


class TestPhaseRecoveryPhaseTimeouts:
    """Tests for phase timeout configuration."""

    def test_all_phases_have_timeouts(self):
        """All phases should have defined timeouts."""
        expected_phases = ["context", "debate", "design", "implement", "verify", "commit"]

        for phase in expected_phases:
            assert phase in PhaseRecovery.PHASE_TIMEOUTS, f"Missing timeout for {phase}"

    def test_timeout_values_reasonable(self):
        """Timeout values should be within reasonable bounds."""
        for phase, timeout in PhaseRecovery.PHASE_TIMEOUTS.items():
            assert 60 <= timeout <= 3600, f"Timeout for {phase} ({timeout}s) outside bounds"

    def test_heavy_phases_have_longest_timeouts(self):
        """Debate and implement phases should have the longest timeouts."""
        debate_timeout = PhaseRecovery.PHASE_TIMEOUTS["debate"]
        implement_timeout = PhaseRecovery.PHASE_TIMEOUTS["implement"]

        # Debate and implement are the heaviest phases
        heavy_phases = {"debate", "implement"}
        for phase, timeout in PhaseRecovery.PHASE_TIMEOUTS.items():
            if phase not in heavy_phases:
                assert (
                    debate_timeout >= timeout or implement_timeout >= timeout
                ), f"{phase} timeout ({timeout}s) exceeds both debate ({debate_timeout}s) and implement ({implement_timeout}s)"


class TestPhaseRecoveryPhaseConfig:
    """Tests for phase retry configuration."""

    def test_all_phases_have_config(self):
        """All phases should have retry config."""
        expected_phases = ["context", "debate", "design", "implement", "verify", "commit"]

        for phase in expected_phases:
            config = PhaseRecovery.PHASE_RETRY_CONFIG.get(phase)
            assert config is not None, f"Missing config for {phase}"
            assert "max_retries" in config
            assert "base_delay" in config
            assert "critical" in config

    def test_critical_phases_identified(self):
        """Debate, implement, and commit should be marked critical."""
        critical_phases = ["debate", "implement", "commit"]
        non_critical_phases = ["context", "design", "verify"]

        for phase in critical_phases:
            assert PhaseRecovery.PHASE_RETRY_CONFIG[phase]["critical"] is True

        for phase in non_critical_phases:
            assert PhaseRecovery.PHASE_RETRY_CONFIG[phase]["critical"] is False

    def test_verify_has_most_retries(self):
        """Verify phase should have most retries (tests can be flaky)."""
        verify_retries = PhaseRecovery.PHASE_RETRY_CONFIG["verify"]["max_retries"]
        assert verify_retries == 3

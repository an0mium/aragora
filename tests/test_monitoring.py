"""
Tests for the monitoring module.

Covers SimpleObserver patterns: attempt tracking, failure rates, and reporting.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aragora.monitoring.simple_observer import SimpleObserver


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def observer(temp_log_file):
    """Create a SimpleObserver with a temporary log file."""
    return SimpleObserver(log_file=temp_log_file)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSimpleObserverInit:
    """Tests for SimpleObserver initialization."""

    def test_creates_with_default_log_file(self):
        """Should initialize with default log file path."""
        obs = SimpleObserver()
        assert obs.log_file == "system_health.log"

    def test_creates_with_custom_log_file(self, temp_log_file):
        """Should accept custom log file path."""
        obs = SimpleObserver(log_file=temp_log_file)
        assert obs.log_file == temp_log_file

    def test_initializes_empty_metrics(self, observer):
        """Should start with empty metrics dictionary."""
        assert observer.metrics == {}

    def test_creates_logger(self, observer):
        """Should create a logger instance."""
        assert observer.logger is not None
        assert observer.logger.name == "aragora_observer"

    def test_logger_has_file_handler(self, temp_log_file):
        """Should add file handler to logger."""
        obs = SimpleObserver(log_file=temp_log_file)
        # Check that at least one handler exists
        assert len(obs.logger.handlers) >= 1

    def test_logger_level_is_info(self, observer):
        """Should set logger level to INFO."""
        assert observer.logger.level == logging.INFO


# =============================================================================
# Agent Attempt Recording Tests
# =============================================================================


class TestRecordAgentAttempt:
    """Tests for record_agent_attempt method."""

    def test_returns_unique_attempt_id(self, observer):
        """Should return a unique UUID for each attempt."""
        id1 = observer.record_agent_attempt("agent1", 30.0)
        id2 = observer.record_agent_attempt("agent2", 30.0)
        assert id1 != id2

    def test_stores_agent_name(self, observer):
        """Should store the agent name in metrics."""
        attempt_id = observer.record_agent_attempt("test-agent", 30.0)
        assert observer.metrics[attempt_id]["agent"] == "test-agent"

    def test_stores_timeout(self, observer):
        """Should store the timeout value in metrics."""
        attempt_id = observer.record_agent_attempt("agent", 45.0)
        assert observer.metrics[attempt_id]["timeout"] == 45.0

    def test_stores_start_time(self, observer):
        """Should record start time close to current time."""
        before = time.time()
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        after = time.time()

        start_time = observer.metrics[attempt_id]["start_time"]
        assert before <= start_time <= after

    def test_sets_status_to_in_progress(self, observer):
        """Should set initial status to 'in_progress'."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        assert observer.metrics[attempt_id]["status"] == "in_progress"

    def test_logs_attempt(self, observer):
        """Should log the agent attempt."""
        with patch.object(observer.logger, "info") as mock_log:
            observer.record_agent_attempt("test-agent", 30.0)
            mock_log.assert_called_once()
            assert "test-agent" in mock_log.call_args[0][0]


# =============================================================================
# Agent Completion Recording Tests
# =============================================================================


class TestRecordAgentCompletion:
    """Tests for record_agent_completion method."""

    def test_ignores_unknown_attempt_id(self, observer):
        """Should silently ignore unknown attempt IDs."""
        # Should not raise
        observer.record_agent_completion("unknown-id", "output")
        assert "unknown-id" not in observer.metrics

    def test_records_end_time(self, observer):
        """Should record end time."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output")

        assert "end_time" in observer.metrics[attempt_id]
        assert observer.metrics[attempt_id]["end_time"] > observer.metrics[attempt_id]["start_time"]

    def test_calculates_duration(self, observer):
        """Should calculate duration from start to end."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        time.sleep(0.01)  # Small delay
        observer.record_agent_completion(attempt_id, "output")

        duration = observer.metrics[attempt_id]["duration"]
        assert duration >= 0.01

    def test_sets_success_status_without_error(self, observer):
        """Should set status to 'success' when no error."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output")

        assert observer.metrics[attempt_id]["status"] == "success"

    def test_sets_failed_status_with_error(self, observer):
        """Should set status to 'failed' when error provided."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output", ValueError("test error"))

        assert observer.metrics[attempt_id]["status"] == "failed"

    def test_records_error_message(self, observer):
        """Should record error message string."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output", ValueError("test error"))

        assert "test error" in observer.metrics[attempt_id]["error"]

    def test_records_output_length(self, observer):
        """Should record output length."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "hello world")

        assert observer.metrics[attempt_id]["output_length"] == 11

    def test_handles_none_output(self, observer):
        """Should handle None output gracefully."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, None)

        assert observer.metrics[attempt_id]["output_length"] == 0

    def test_detects_null_bytes_in_output(self, observer):
        """Should detect null bytes in output."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output\x00with\x00nulls")

        assert observer.metrics[attempt_id]["has_null_bytes"] is True

    def test_no_null_bytes_in_clean_output(self, observer):
        """Should report no null bytes in clean output."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "clean output")

        assert observer.metrics[attempt_id]["has_null_bytes"] is False

    def test_logs_error_on_failure(self, observer):
        """Should log error message on failure."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        with patch.object(observer.logger, "error") as mock_log:
            observer.record_agent_completion(attempt_id, "output", ValueError("test error"))
            mock_log.assert_called_once()
            log_msg = mock_log.call_args[0][0].lower()
            assert "failed" in log_msg or "error" in log_msg

    def test_logs_warning_on_null_bytes(self, observer):
        """Should log warning when null bytes detected."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        with patch.object(observer.logger, "warning") as mock_log:
            observer.record_agent_completion(attempt_id, "output\x00")
            mock_log.assert_called_once()
            assert "null" in mock_log.call_args[0][0].lower()


# =============================================================================
# Loop ID Issue Recording Tests
# =============================================================================


class TestRecordLoopIdIssue:
    """Tests for record_loop_id_issue method."""

    def test_logs_present_loop_id(self, observer):
        """Should log when loop_id is present."""
        with patch.object(observer.logger, "info") as mock_log:
            observer.record_loop_id_issue("ws-123", True, "client")
            mock_log.assert_called_once()
            log_msg = mock_log.call_args[0][0]
            assert "present" in log_msg
            assert "ws-123" in log_msg

    def test_logs_missing_loop_id(self, observer):
        """Should log when loop_id is missing."""
        with patch.object(observer.logger, "info") as mock_log:
            observer.record_loop_id_issue("ws-456", False, "server")
            mock_log.assert_called_once()
            log_msg = mock_log.call_args[0][0]
            assert "missing" in log_msg
            assert "ws-456" in log_msg

    def test_logs_source(self, observer):
        """Should log the source of the issue."""
        with patch.object(observer.logger, "info") as mock_log:
            observer.record_loop_id_issue("ws-789", True, "middleware")
            mock_log.assert_called_once()
            assert "middleware" in mock_log.call_args[0][0]


# =============================================================================
# Failure Rate Calculation Tests
# =============================================================================


class TestGetFailureRate:
    """Tests for get_failure_rate method."""

    def test_returns_zero_with_no_data(self, observer):
        """Should return 0.0 when no attempts recorded."""
        assert observer.get_failure_rate() == 0.0

    def test_returns_zero_with_only_in_progress(self, observer):
        """Should return 0.0 when only in-progress attempts exist."""
        observer.record_agent_attempt("agent", 30.0)
        assert observer.get_failure_rate() == 0.0

    def test_returns_zero_with_all_success(self, observer):
        """Should return 0.0 when all attempts succeed."""
        for i in range(5):
            attempt_id = observer.record_agent_attempt(f"agent-{i}", 30.0)
            observer.record_agent_completion(attempt_id, "output")

        assert observer.get_failure_rate() == 0.0

    def test_returns_one_with_all_failures(self, observer):
        """Should return 1.0 when all attempts fail."""
        for i in range(5):
            attempt_id = observer.record_agent_attempt(f"agent-{i}", 30.0)
            observer.record_agent_completion(attempt_id, "output", ValueError("error"))

        assert observer.get_failure_rate() == 1.0

    def test_calculates_correct_rate(self, observer):
        """Should calculate correct failure rate for mixed results."""
        # 3 successes
        for i in range(3):
            attempt_id = observer.record_agent_attempt(f"success-{i}", 30.0)
            observer.record_agent_completion(attempt_id, "output")

        # 2 failures
        for i in range(2):
            attempt_id = observer.record_agent_attempt(f"fail-{i}", 30.0)
            observer.record_agent_completion(attempt_id, "output", ValueError("error"))

        # 2/5 = 0.4
        assert observer.get_failure_rate() == 0.4

    def test_ignores_in_progress_in_calculation(self, observer):
        """Should ignore in-progress attempts when calculating rate."""
        # 1 success
        attempt_id = observer.record_agent_attempt("success", 30.0)
        observer.record_agent_completion(attempt_id, "output")

        # 1 in-progress (should be ignored)
        observer.record_agent_attempt("in-progress", 30.0)

        assert observer.get_failure_rate() == 0.0


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestGetReport:
    """Tests for get_report method."""

    def test_returns_error_with_no_data(self, observer):
        """Should return error message when no data available."""
        report = observer.get_report()
        assert "error" in report
        assert "No data" in report["error"]

    def test_returns_error_with_only_in_progress(self, observer):
        """Should return error when only in-progress attempts exist."""
        observer.record_agent_attempt("agent", 30.0)
        report = observer.get_report()
        assert "error" in report

    def test_includes_total_attempts(self, observer):
        """Should include total completed attempts count."""
        for i in range(3):
            attempt_id = observer.record_agent_attempt(f"agent-{i}", 30.0)
            observer.record_agent_completion(attempt_id, "output")

        report = observer.get_report()
        assert report["total_attempts"] == 3

    def test_includes_failure_rate(self, observer):
        """Should include failure rate in report."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output")

        report = observer.get_report()
        assert "failure_rate" in report
        assert report["failure_rate"] == 0.0

    def test_includes_null_byte_incidents(self, observer):
        """Should count null byte incidents."""
        # 2 with null bytes
        for i in range(2):
            attempt_id = observer.record_agent_attempt(f"null-{i}", 30.0)
            observer.record_agent_completion(attempt_id, f"output\x00{i}")

        # 1 clean
        attempt_id = observer.record_agent_attempt("clean", 30.0)
        observer.record_agent_completion(attempt_id, "clean output")

        report = observer.get_report()
        assert report["null_byte_incidents"] == 2

    def test_includes_timeout_incidents(self, observer):
        """Should count timeout incidents."""
        # Create an attempt with short timeout
        attempt_id = observer.record_agent_attempt("agent", 0.001)  # 1ms timeout
        time.sleep(0.01)  # Sleep 10ms to exceed timeout
        observer.record_agent_completion(attempt_id, "output")

        report = observer.get_report()
        assert report["timeout_incidents"] >= 1

    def test_no_timeouts_under_limit(self, observer):
        """Should report zero timeouts when all under limit."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)  # 30s timeout
        observer.record_agent_completion(attempt_id, "output")  # Immediate completion

        report = observer.get_report()
        assert report["timeout_incidents"] == 0

    def test_report_structure(self, observer):
        """Should return report with expected structure."""
        attempt_id = observer.record_agent_attempt("agent", 30.0)
        observer.record_agent_completion(attempt_id, "output")

        report = observer.get_report()

        assert isinstance(report, dict)
        assert "total_attempts" in report
        assert "failure_rate" in report
        assert "null_byte_incidents" in report
        assert "timeout_incidents" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestSimpleObserverIntegration:
    """Integration tests for SimpleObserver."""

    def test_full_lifecycle(self, observer):
        """Test complete lifecycle of agent monitoring."""
        # Start multiple attempts
        attempts = []
        for i in range(5):
            attempt_id = observer.record_agent_attempt(f"agent-{i}", 30.0)
            attempts.append(attempt_id)

        # Complete with mixed results
        observer.record_agent_completion(attempts[0], "success")
        observer.record_agent_completion(attempts[1], "success")
        observer.record_agent_completion(attempts[2], "output\x00", ValueError("error"))
        observer.record_agent_completion(attempts[3], "output", ValueError("error"))
        observer.record_agent_completion(attempts[4], "success")

        # Check metrics
        report = observer.get_report()
        assert report["total_attempts"] == 5
        assert report["failure_rate"] == 0.4  # 2/5
        assert report["null_byte_incidents"] == 1

    def test_concurrent_attempts(self, observer):
        """Test tracking multiple concurrent attempts."""
        # Start 10 concurrent attempts
        attempts = [observer.record_agent_attempt(f"agent-{i}", 30.0) for i in range(10)]

        # All should be in_progress
        for attempt_id in attempts:
            assert observer.metrics[attempt_id]["status"] == "in_progress"

        # Complete all
        for attempt_id in attempts:
            observer.record_agent_completion(attempt_id, "output")

        assert observer.get_failure_rate() == 0.0
        assert observer.get_report()["total_attempts"] == 10

    def test_agent_type_tracking(self, observer):
        """Test tracking different agent types."""
        agents = ["claude", "gpt-4", "gemini", "codex"]

        for agent in agents:
            attempt_id = observer.record_agent_attempt(agent, 30.0)
            observer.record_agent_completion(attempt_id, f"output from {agent}")

        # Verify all agents tracked
        agent_names = [m["agent"] for m in observer.metrics.values()]
        for agent in agents:
            assert agent in agent_names

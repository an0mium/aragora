"""Tests for aragora.monitoring.simple_observer.SimpleObserver."""

from __future__ import annotations

import uuid

import pytest

from aragora.monitoring.simple_observer import SimpleObserver


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


def test_import_simple_observer():
    """SimpleObserver is importable from its module."""
    assert SimpleObserver is not None


def test_module_all_export():
    """SimpleObserver is listed in __all__."""
    from aragora.monitoring import simple_observer

    assert "SimpleObserver" in simple_observer.__all__


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_default_instantiation(tmp_path):
    """SimpleObserver can be created with default log_file."""
    log_file = str(tmp_path / "default.log")
    observer = SimpleObserver(log_file=log_file)
    assert observer.log_file == log_file
    assert observer.metrics == {}


def test_custom_log_file(tmp_path):
    """log_file property reflects the value passed at init."""
    custom = str(tmp_path / "custom_health.log")
    observer = SimpleObserver(log_file=custom)
    assert observer.log_file == custom


# ---------------------------------------------------------------------------
# record_agent_attempt
# ---------------------------------------------------------------------------


def test_record_agent_attempt_returns_string(tmp_path):
    """record_agent_attempt returns a valid UUID string."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("claude", timeout=30.0)
    # Should be a valid UUID
    uuid.UUID(attempt_id)


def test_record_agent_attempt_stores_metrics(tmp_path):
    """Attempt is recorded in the metrics dict with expected fields."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("gemini", timeout=15.0)

    assert attempt_id in observer.metrics
    record = observer.metrics[attempt_id]
    assert record["agent"] == "gemini"
    assert record["timeout"] == 15.0
    assert record["status"] == "in_progress"
    assert "start_time" in record


# ---------------------------------------------------------------------------
# record_agent_completion -- success
# ---------------------------------------------------------------------------


def test_record_completion_success(tmp_path):
    """Successful completion sets status and duration."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("claude", timeout=30.0)
    observer.record_agent_completion(attempt_id, output="Hello world")

    record = observer.metrics[attempt_id]
    assert record["status"] == "success"
    assert record["output_length"] == len("Hello world")
    assert record["has_null_bytes"] is False
    assert "duration" in record


def test_record_completion_detects_null_bytes(tmp_path):
    """Null bytes in output are flagged."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("gpt", timeout=10.0)
    observer.record_agent_completion(attempt_id, output="bad\x00data")

    assert observer.metrics[attempt_id]["has_null_bytes"] is True


def test_record_completion_none_output(tmp_path):
    """None output is handled gracefully."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("gpt", timeout=10.0)
    observer.record_agent_completion(attempt_id, output=None)

    record = observer.metrics[attempt_id]
    assert record["output_length"] == 0
    assert record["has_null_bytes"] is False
    assert record["status"] == "success"


# ---------------------------------------------------------------------------
# record_agent_completion -- error
# ---------------------------------------------------------------------------


def test_record_completion_error(tmp_path):
    """Error completion sets status to failed and records error string."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    attempt_id = observer.record_agent_attempt("mistral", timeout=20.0)
    observer.record_agent_completion(
        attempt_id, output=None, error=RuntimeError("timeout exceeded")
    )

    record = observer.metrics[attempt_id]
    assert record["status"] == "failed"
    assert "timeout exceeded" in record["error"]


def test_record_completion_unknown_attempt_id(tmp_path):
    """Completing an unknown attempt_id is silently ignored."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    # Should not raise
    observer.record_agent_completion("nonexistent-id", output="data")
    assert "nonexistent-id" not in observer.metrics


# ---------------------------------------------------------------------------
# record_loop_id_issue
# ---------------------------------------------------------------------------


def test_record_loop_id_issue_present(tmp_path):
    """Recording a loop_id present event does not raise."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    observer.record_loop_id_issue(ws_id="ws-123", present=True, source="client")


def test_record_loop_id_issue_missing(tmp_path):
    """Recording a loop_id missing event does not raise."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    observer.record_loop_id_issue(ws_id="ws-456", present=False, source="server")


# ---------------------------------------------------------------------------
# get_failure_rate
# ---------------------------------------------------------------------------


def test_failure_rate_no_data(tmp_path):
    """Failure rate is 0.0 when there are no completed attempts."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    assert observer.get_failure_rate() == 0.0


def test_failure_rate_all_success(tmp_path):
    """Failure rate is 0.0 when all attempts succeed."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    for _ in range(3):
        aid = observer.record_agent_attempt("claude", timeout=10.0)
        observer.record_agent_completion(aid, output="ok")
    assert observer.get_failure_rate() == 0.0


def test_failure_rate_all_failed(tmp_path):
    """Failure rate is 1.0 when all attempts fail."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    for _ in range(3):
        aid = observer.record_agent_attempt("claude", timeout=10.0)
        observer.record_agent_completion(aid, output=None, error=Exception("err"))
    assert observer.get_failure_rate() == 1.0


def test_failure_rate_mixed(tmp_path):
    """Failure rate reflects the ratio of failures to completed attempts."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))

    # 1 success
    aid = observer.record_agent_attempt("claude", timeout=10.0)
    observer.record_agent_completion(aid, output="ok")

    # 1 failure
    aid = observer.record_agent_attempt("gpt", timeout=10.0)
    observer.record_agent_completion(aid, output=None, error=Exception("err"))

    assert observer.get_failure_rate() == pytest.approx(0.5)


def test_failure_rate_ignores_in_progress(tmp_path):
    """In-progress attempts are excluded from failure rate calculation."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))

    # 1 success
    aid = observer.record_agent_attempt("claude", timeout=10.0)
    observer.record_agent_completion(aid, output="ok")

    # 1 still in progress (not completed)
    observer.record_agent_attempt("gpt", timeout=10.0)

    assert observer.get_failure_rate() == 0.0


# ---------------------------------------------------------------------------
# get_report
# ---------------------------------------------------------------------------


def test_report_no_data(tmp_path):
    """Report returns error dict when no completed attempts exist."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))
    report = observer.get_report()
    assert "error" in report


def test_report_structure(tmp_path):
    """Report contains all expected keys after completed attempts."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))

    aid = observer.record_agent_attempt("claude", timeout=10.0)
    observer.record_agent_completion(aid, output="result")

    aid2 = observer.record_agent_attempt("gpt", timeout=10.0)
    observer.record_agent_completion(aid2, output=None, error=Exception("fail"))

    report = observer.get_report()

    assert report["total_attempts"] == 2
    assert report["failed_attempts"] == 1
    assert report["failure_rate"] == pytest.approx(0.5)
    assert report["null_byte_incidents"] == 0
    assert report["timeout_incidents"] == 0


def test_report_counts_null_byte_incidents(tmp_path):
    """Report correctly counts null byte incidents."""
    observer = SimpleObserver(log_file=str(tmp_path / "health.log"))

    aid = observer.record_agent_attempt("claude", timeout=10.0)
    observer.record_agent_completion(aid, output="clean output")

    aid2 = observer.record_agent_attempt("gpt", timeout=10.0)
    observer.record_agent_completion(aid2, output="has\x00null")

    report = observer.get_report()
    assert report["null_byte_incidents"] == 1

"""Tests for observability/metrics/oracle.py."""

from unittest.mock import patch

import pytest

from aragora.observability.metrics import oracle as mod
from aragora.observability.metrics.oracle import (
    get_oracle_stream_metrics_summary,
    init_oracle_metrics,
    record_oracle_session_outcome,
    record_oracle_session_started,
    record_oracle_stream_phase_duration,
    record_oracle_stream_stall,
    record_oracle_time_to_first_token,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    mod._initialized = False
    mod._snapshot = mod._OracleSnapshot()
    yield
    mod._initialized = False
    mod._snapshot = mod._OracleSnapshot()


def test_init_noop_when_metrics_disabled() -> None:
    with patch("aragora.observability.metrics.oracle.get_metrics_enabled", return_value=False):
        init_oracle_metrics()
    assert mod._initialized is True
    summary = get_oracle_stream_metrics_summary()
    assert summary["available"] is True


def test_records_session_lifecycle_and_ttft_summary() -> None:
    with patch("aragora.observability.metrics.oracle.get_metrics_enabled", return_value=False):
        init_oracle_metrics()

    record_oracle_session_started()
    record_oracle_time_to_first_token("deep", 0.42)
    record_oracle_stream_phase_duration("deep", 1.3)
    record_oracle_stream_stall("waiting_first_token", phase="deep")
    record_oracle_session_outcome("completed")

    summary = get_oracle_stream_metrics_summary()
    assert summary["sessions_started"] == 1
    assert summary["sessions_completed"] == 1
    assert summary["active_sessions"] == 0
    assert summary["stalls_waiting_first_token"] == 1
    assert summary["ttft_samples"] == 1
    assert summary["ttft_avg_ms"] == 420.0
    assert summary["ttft_last_ms"] == 420.0


def test_normalizes_unknown_outcome_and_stall_reason() -> None:
    with patch("aragora.observability.metrics.oracle.get_metrics_enabled", return_value=False):
        init_oracle_metrics()

    record_oracle_session_started()
    record_oracle_stream_stall("unknown-reason", phase="deep")
    record_oracle_session_outcome("unexpected")

    summary = get_oracle_stream_metrics_summary()
    assert summary["sessions_errors"] == 1
    assert summary["stalls_stream_inactive"] == 1

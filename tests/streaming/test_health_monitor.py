"""Tests for aragora.streaming.health_monitor."""

from __future__ import annotations

import time

import pytest

from aragora.streaming.health_monitor import (
    StreamHealthMonitor,
    StreamHealthSnapshot,
    get_stream_health_monitor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton between tests."""
    StreamHealthMonitor.reset_instance()
    yield
    StreamHealthMonitor.reset_instance()


def _make_monitor(**kwargs) -> StreamHealthMonitor:
    return StreamHealthMonitor(**kwargs)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestStreamHealthMonitorRecording:
    """Test recording of connections, errors, and messages."""

    def test_record_connection(self):
        mon = _make_monitor()
        mon.record_connection("d1", "c1")
        health = mon.get_health()
        assert health.active_connections == 1
        assert health.active_debates == 1

    def test_record_disconnection(self):
        mon = _make_monitor()
        mon.record_connection("d1", "c1")
        mon.record_disconnection("d1", "c1")
        health = mon.get_health()
        assert health.active_connections == 0

    def test_disconnection_does_not_go_negative(self):
        mon = _make_monitor()
        mon.record_disconnection("d1", "c1")
        health = mon.get_health()
        assert health.active_connections == 0

    def test_record_reconnect(self):
        mon = _make_monitor()
        mon.record_reconnect("d1", "c1")
        health = mon.get_health()
        assert health.total_reconnects == 1

    def test_record_message_delivered(self):
        mon = _make_monitor()
        mon.record_message_delivered("d1", count=5)
        health = mon.get_health()
        assert health.total_messages_delivered == 5

    def test_record_message_failed(self):
        mon = _make_monitor()
        mon.record_message_failed("d1", count=2)
        health = mon.get_health()
        assert health.total_messages_failed == 2

    def test_record_error(self):
        mon = _make_monitor(error_rate_window=300.0)
        mon.record_error("d1", "timeout")
        health = mon.get_health()
        assert health.error_rate_5m > 0

    def test_remove_debate(self):
        mon = _make_monitor()
        mon.record_connection("d1", "c1")
        mon.record_connection("d2", "c2")
        mon.remove_debate("d1")
        health = mon.get_health()
        assert health.active_debates == 1


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------


class TestStreamHealthMonitorStatus:
    """Test health status calculation."""

    def test_healthy_when_no_issues(self):
        mon = _make_monitor()
        mon.record_connection("d1")
        mon.record_message_delivered("d1", count=100)
        health = mon.get_health()
        assert health.status == "healthy"
        assert len(health.slo_violations) == 0

    def test_degraded_on_low_delivery_rate(self):
        mon = _make_monitor()
        mon.record_message_delivered("d1", count=990)
        mon.record_message_failed("d1", count=15)
        health = mon.get_health()
        # Delivery rate = 990/1005 = ~98.5%, below 99.5% SLO
        assert health.status in ("degraded", "unhealthy")
        assert len(health.slo_violations) > 0

    def test_unhealthy_on_very_low_delivery_rate(self):
        mon = _make_monitor()
        mon.record_message_delivered("d1", count=80)
        mon.record_message_failed("d1", count=20)
        health = mon.get_health()
        # Delivery rate = 80%, well below 95% threshold
        assert health.status == "unhealthy"

    def test_delivery_rate_defaults_to_1_when_no_messages(self):
        mon = _make_monitor()
        health = mon.get_health()
        assert health.message_delivery_rate == 1.0

    def test_snapshot_to_dict(self):
        mon = _make_monitor()
        mon.record_connection("d1")
        health = mon.get_health()
        d = health.to_dict()
        assert "status" in d
        assert "active_connections" in d
        assert "message_delivery_rate" in d
        assert "slo_violations" in d
        assert "debates" in d


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestStreamHealthMonitorSingleton:
    """Test singleton behavior."""

    def test_get_instance_returns_same_object(self):
        m1 = StreamHealthMonitor.get_instance()
        m2 = StreamHealthMonitor.get_instance()
        assert m1 is m2

    def test_reset_instance_clears_singleton(self):
        m1 = StreamHealthMonitor.get_instance()
        StreamHealthMonitor.reset_instance()
        m2 = StreamHealthMonitor.get_instance()
        assert m1 is not m2

    def test_get_stream_health_monitor_returns_singleton(self):
        m1 = get_stream_health_monitor()
        m2 = get_stream_health_monitor()
        assert m1 is m2


# ---------------------------------------------------------------------------
# Per-debate metrics
# ---------------------------------------------------------------------------


class TestStreamHealthMonitorPerDebate:
    """Test per-debate metrics in health snapshot."""

    def test_per_debate_summary(self):
        mon = _make_monitor()
        mon.record_connection("d1")
        mon.record_message_delivered("d1", count=50)
        mon.record_message_failed("d1", count=1)
        mon.record_reconnect("d1")
        mon.record_error("d1", "send_error")

        health = mon.get_health()
        assert "d1" in health.debates
        d1 = health.debates["d1"]
        assert d1["active_connections"] == 1
        assert d1["messages_delivered"] == 50
        assert d1["messages_failed"] == 1
        assert d1["reconnect_count"] == 1
        assert d1["last_error_type"] == "send_error"

    def test_health_check_interval(self):
        mon = _make_monitor(health_check_interval=0.05)
        assert mon.should_run_health_check() is True
        mon.get_health()  # Updates last check time
        assert mon.should_run_health_check() is False
        time.sleep(0.06)
        assert mon.should_run_health_check() is True

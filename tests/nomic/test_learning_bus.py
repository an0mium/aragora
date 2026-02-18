"""Tests for the cross-agent learning bus."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.nomic.learning_bus import Finding, LearningBus


@pytest.fixture(autouse=True)
def _reset_bus():
    """Ensure each test gets a fresh singleton."""
    LearningBus.reset_instance()
    yield
    LearningBus.reset_instance()


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "agent_id": "agent-1",
        "topic": "pattern_bug",
        "description": "Found bare except",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        a = LearningBus.get_instance()
        b = LearningBus.get_instance()
        assert a is b

    def test_reset_clears_instance(self):
        a = LearningBus.get_instance()
        LearningBus.reset_instance()
        b = LearningBus.get_instance()
        assert a is not b


class TestPublishAndQuery:
    def test_publish_and_get_all(self):
        bus = LearningBus.get_instance()
        f = _make_finding()
        bus.publish(f)
        assert bus.get_findings() == [f]

    def test_get_findings_by_topic(self):
        bus = LearningBus.get_instance()
        bus.publish(_make_finding(topic="pattern_bug"))
        bus.publish(_make_finding(topic="test_failure"))
        bus.publish(_make_finding(topic="pattern_bug"))

        result = bus.get_findings(topic="pattern_bug")
        assert len(result) == 2
        assert all(f.topic == "pattern_bug" for f in result)

    def test_get_findings_since(self):
        bus = LearningBus.get_instance()
        old = _make_finding(timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc))
        new = _make_finding(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        bus.publish(old)
        bus.publish(new)

        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = bus.get_findings(since=cutoff)
        assert len(result) == 1
        assert result[0] is new

    def test_get_findings_for_files(self):
        bus = LearningBus.get_instance()
        bus.publish(_make_finding(affected_files=["a.py", "b.py"]))
        bus.publish(_make_finding(affected_files=["c.py"]))

        result = bus.get_findings_for_files(["b.py", "d.py"])
        assert len(result) == 1
        assert "b.py" in result[0].affected_files

    def test_get_findings_for_files_empty(self):
        bus = LearningBus.get_instance()
        bus.publish(_make_finding(affected_files=["a.py"]))
        assert bus.get_findings_for_files(["z.py"]) == []


class TestSubscriptions:
    def test_subscribe_receives_callback(self):
        bus = LearningBus.get_instance()
        received = []
        bus.subscribe("pattern_bug", received.append)
        f = _make_finding(topic="pattern_bug")
        bus.publish(f)
        assert received == [f]

    def test_subscribe_wrong_topic_no_callback(self):
        bus = LearningBus.get_instance()
        received = []
        bus.subscribe("test_failure", received.append)
        bus.publish(_make_finding(topic="pattern_bug"))
        assert received == []

    def test_unsubscribe_stops_callbacks(self):
        bus = LearningBus.get_instance()
        received = []
        bus.subscribe("pattern_bug", received.append)
        bus.unsubscribe("pattern_bug", received.append)
        bus.publish(_make_finding(topic="pattern_bug"))
        assert received == []

    def test_unsubscribe_nonexistent_no_error(self):
        bus = LearningBus.get_instance()
        bus.unsubscribe("nope", lambda f: None)  # Should not raise

    def test_subscriber_exception_does_not_break_publish(self):
        bus = LearningBus.get_instance()

        def bad_cb(f):
            raise RuntimeError("boom")

        good_received = []
        bus.subscribe("pattern_bug", bad_cb)
        bus.subscribe("pattern_bug", good_received.append)

        f = _make_finding(topic="pattern_bug")
        bus.publish(f)
        # Finding is still stored despite callback error
        assert bus.get_findings() == [f]


class TestClearAndSummary:
    def test_clear_removes_everything(self):
        bus = LearningBus.get_instance()
        bus.publish(_make_finding())
        bus.subscribe("pattern_bug", lambda f: None)
        bus.clear()
        assert bus.get_findings() == []
        assert bus.summary()["total"] == 0

    def test_summary_counts(self):
        bus = LearningBus.get_instance()
        bus.publish(_make_finding(topic="pattern_bug", severity="warning"))
        bus.publish(_make_finding(topic="pattern_bug", severity="critical"))
        bus.publish(_make_finding(topic="test_failure", severity="warning"))

        s = bus.summary()
        assert s["total"] == 3
        assert s["by_topic"] == {"pattern_bug": 2, "test_failure": 1}
        assert s["by_severity"] == {"warning": 2, "critical": 1}


class TestFindingDataclass:
    def test_defaults(self):
        f = Finding(agent_id="a", topic="t", description="d")
        assert f.severity == "info"
        assert f.suggested_action is None
        assert f.affected_files == []
        assert f.metadata == {}
        assert isinstance(f.timestamp, datetime)

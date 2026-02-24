"""Tests for aragora.nomic.cycle_telemetry — CycleTelemetryCollector."""

from __future__ import annotations

import json
import time
import tempfile
import os

import pytest

from aragora.nomic.cycle_telemetry import CycleRecord, CycleTelemetryCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_telemetry.db")


@pytest.fixture
def collector(db_path):
    """Create a fresh CycleTelemetryCollector with a temp database."""
    return CycleTelemetryCollector(db_path=db_path)


def _make_record(**overrides) -> CycleRecord:
    """Helper to build a CycleRecord with sensible defaults."""
    defaults = {
        "goal": "Improve tests",
        "cycle_time_seconds": 10.0,
        "success": True,
        "quality_delta": 0.1,
        "cost_usd": 0.05,
        "agents_used": ["claude", "gemini"],
        "debate_ids": ["d1"],
        "branch_name": "nomic/test",
        "commit_sha": "abc123",
    }
    defaults.update(overrides)
    return CycleRecord(**defaults)


# ---------------------------------------------------------------------------
# CycleRecord dataclass
# ---------------------------------------------------------------------------


class TestCycleRecord:
    def test_auto_id(self):
        r = CycleRecord(goal="x")
        assert r.cycle_id.startswith("cycle_")

    def test_auto_timestamp(self):
        before = time.time()
        r = CycleRecord(goal="x")
        assert r.timestamp >= before

    def test_explicit_id_preserved(self):
        r = CycleRecord(cycle_id="my_id", goal="x")
        assert r.cycle_id == "my_id"

    def test_to_dict(self):
        r = _make_record(cycle_id="c1")
        d = r.to_dict()
        assert d["cycle_id"] == "c1"
        assert d["agents_used"] == ["claude", "gemini"]
        assert isinstance(d["success"], bool)

    def test_from_dict_roundtrip(self):
        original = _make_record(cycle_id="rt")
        d = original.to_dict()
        restored = CycleRecord.from_dict(d)
        assert restored.cycle_id == "rt"
        assert restored.goal == original.goal
        assert restored.agents_used == original.agents_used

    def test_from_dict_missing_fields(self):
        r = CycleRecord.from_dict({})
        assert r.cycle_id == ""
        assert r.success is False


# ---------------------------------------------------------------------------
# record_cycle + get_recent_cycles
# ---------------------------------------------------------------------------


class TestRecordAndQuery:
    def test_record_single(self, collector):
        r = _make_record(cycle_id="s1")
        collector.record_cycle(r)
        recent = collector.get_recent_cycles(n=10)
        assert len(recent) == 1
        assert recent[0].cycle_id == "s1"

    def test_record_multiple_ordered_by_timestamp(self, collector):
        for i in range(5):
            r = _make_record(cycle_id=f"c{i}", timestamp=1000.0 + i)
            collector.record_cycle(r)
        recent = collector.get_recent_cycles(n=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].cycle_id == "c4"
        assert recent[2].cycle_id == "c2"

    def test_get_recent_respects_limit(self, collector):
        for i in range(10):
            collector.record_cycle(_make_record(cycle_id=f"lim{i}"))
        assert len(collector.get_recent_cycles(n=3)) == 3

    def test_replace_existing_record(self, collector):
        collector.record_cycle(_make_record(cycle_id="dup", quality_delta=0.1))
        collector.record_cycle(_make_record(cycle_id="dup", quality_delta=0.9))
        recent = collector.get_recent_cycles(n=10)
        assert len(recent) == 1
        assert recent[0].quality_delta == 0.9

    def test_empty_store(self, collector):
        assert collector.get_recent_cycles() == []

    def test_agents_and_debates_serialized(self, collector):
        r = _make_record(
            cycle_id="ser",
            agents_used=["a", "b", "c"],
            debate_ids=["d1", "d2"],
        )
        collector.record_cycle(r)
        loaded = collector.get_recent_cycles(n=1)[0]
        assert loaded.agents_used == ["a", "b", "c"]
        assert loaded.debate_ids == ["d1", "d2"]


# ---------------------------------------------------------------------------
# Aggregation queries
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_success_rate_all_success(self, collector):
        for i in range(5):
            collector.record_cycle(_make_record(cycle_id=f"s{i}", success=True))
        assert collector.get_success_rate(window_days=7) == 1.0

    def test_success_rate_mixed(self, collector):
        collector.record_cycle(_make_record(cycle_id="ok", success=True))
        collector.record_cycle(_make_record(cycle_id="fail", success=False))
        rate = collector.get_success_rate(window_days=7)
        assert rate == pytest.approx(0.5)

    def test_success_rate_empty(self, collector):
        assert collector.get_success_rate() == 0.0

    def test_success_rate_window_excludes_old(self, collector):
        # Record old cycle (outside window)
        old_time = time.time() - 30 * 86400  # 30 days ago
        collector.record_cycle(_make_record(cycle_id="old", success=False, timestamp=old_time))
        # Record recent cycle
        collector.record_cycle(_make_record(cycle_id="new", success=True))
        rate = collector.get_success_rate(window_days=7)
        assert rate == 1.0

    def test_avg_cost_per_improvement(self, collector):
        collector.record_cycle(_make_record(cycle_id="c1", success=True, cost_usd=0.10))
        collector.record_cycle(_make_record(cycle_id="c2", success=True, cost_usd=0.20))
        collector.record_cycle(_make_record(cycle_id="c3", success=False, cost_usd=0.50))
        avg = collector.get_avg_cost_per_improvement()
        assert avg == pytest.approx(0.15)

    def test_avg_cost_no_successes(self, collector):
        collector.record_cycle(_make_record(cycle_id="f1", success=False))
        assert collector.get_avg_cost_per_improvement() == 0.0

    def test_top_goals_by_impact(self, collector):
        collector.record_cycle(_make_record(cycle_id="g1", quality_delta=0.5, success=True, goal="Big"))
        collector.record_cycle(_make_record(cycle_id="g2", quality_delta=0.1, success=True, goal="Small"))
        collector.record_cycle(_make_record(cycle_id="g3", quality_delta=0.9, success=True, goal="Biggest"))
        top = collector.get_top_goals_by_impact(n=2)
        assert len(top) == 2
        assert top[0]["goal"] == "Biggest"
        assert top[1]["goal"] == "Big"

    def test_top_goals_excludes_failures(self, collector):
        collector.record_cycle(_make_record(cycle_id="ok", quality_delta=0.1, success=True))
        collector.record_cycle(_make_record(cycle_id="bad", quality_delta=9.0, success=False))
        top = collector.get_top_goals_by_impact(n=5)
        assert len(top) == 1

    def test_total_cost(self, collector):
        collector.record_cycle(_make_record(cycle_id="a", cost_usd=0.10))
        collector.record_cycle(_make_record(cycle_id="b", cost_usd=0.25))
        assert collector.get_total_cost() == pytest.approx(0.35)

    def test_total_cost_empty(self, collector):
        assert collector.get_total_cost() == 0.0

    def test_cycle_count(self, collector):
        for i in range(7):
            collector.record_cycle(_make_record(cycle_id=f"cnt{i}"))
        assert collector.get_cycle_count() == 7

    def test_consecutive_failures(self, collector):
        collector.record_cycle(_make_record(cycle_id="a", success=True, timestamp=1.0))
        collector.record_cycle(_make_record(cycle_id="b", success=False, timestamp=2.0))
        collector.record_cycle(_make_record(cycle_id="c", success=False, timestamp=3.0))
        assert collector.get_consecutive_failures() == 2

    def test_consecutive_failures_zero_when_last_succeeds(self, collector):
        collector.record_cycle(_make_record(cycle_id="a", success=False, timestamp=1.0))
        collector.record_cycle(_make_record(cycle_id="b", success=True, timestamp=2.0))
        assert collector.get_consecutive_failures() == 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json(self, collector):
        collector.record_cycle(_make_record(cycle_id="e1"))
        collector.record_cycle(_make_record(cycle_id="e2"))
        exported = collector.export_json(n=10)
        data = json.loads(exported)
        assert len(data) == 2
        assert all("cycle_id" in d for d in data)

    def test_export_json_limited(self, collector):
        for i in range(5):
            collector.record_cycle(_make_record(cycle_id=f"x{i}"))
        data = json.loads(collector.export_json(n=2))
        assert len(data) == 2

    def test_export_json_all(self, collector):
        for i in range(3):
            collector.record_cycle(_make_record(cycle_id=f"a{i}"))
        data = json.loads(collector.export_json(n=None))
        assert len(data) == 3

    def test_export_json_empty(self, collector):
        data = json.loads(collector.export_json())
        assert data == []


# ---------------------------------------------------------------------------
# Schema / persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_survives_reopen(self, db_path):
        c1 = CycleTelemetryCollector(db_path=db_path)
        c1.record_cycle(_make_record(cycle_id="persist"))

        c2 = CycleTelemetryCollector(db_path=db_path)
        recent = c2.get_recent_cycles()
        assert len(recent) == 1
        assert recent[0].cycle_id == "persist"

    def test_in_memory_mode(self):
        collector = CycleTelemetryCollector(db_path=":memory:")
        collector.record_cycle(_make_record(cycle_id="mem"))
        assert collector.get_cycle_count() == 1

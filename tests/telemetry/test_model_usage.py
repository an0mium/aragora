"""Tests for the model usage telemetry pipeline.

Covers ModelCallRecord, DebateUsageSummary, ModelUsagePipeline,
event constructors, and module-level singleton access.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from aragora.telemetry.model_usage import (
    DebateUsageSummary,
    ModelCallRecord,
    ModelUsagePipeline,
    ModelUsageEventType,
    debate_cost_summary_event,
    get_model_usage_pipeline,
    model_call_event,
    set_model_usage_pipeline,
)
from aragora.telemetry.research_events import TelemetryEventType


# ---------------------------------------------------------------------------
# ModelCallRecord
# ---------------------------------------------------------------------------


class TestModelCallRecord:
    """Unit tests for ModelCallRecord dataclass."""

    def test_defaults(self):
        r = ModelCallRecord()
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.latency_ms == 0.0
        assert r.cost_usd == Decimal("0")
        assert r.id  # UUID generated

    def test_total_tokens(self):
        r = ModelCallRecord(tokens_in=100, tokens_out=50)
        assert r.total_tokens() == 150

    def test_calculate_cost_returns_decimal(self):
        r = ModelCallRecord(
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=1000,
            tokens_out=500,
        )
        cost = r.calculate_cost()
        assert isinstance(cost, Decimal)
        assert cost == r.cost_usd

    def test_to_dict_keys(self):
        r = ModelCallRecord(
            debate_id="d1",
            agent_id="a1",
            provider="openai",
            model="gpt-4o",
            tokens_in=200,
            tokens_out=100,
            latency_ms=500.0,
            operation="proposal",
        )
        d = r.to_dict()
        assert d["debate_id"] == "d1"
        assert d["agent_id"] == "a1"
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["tokens_in"] == 200
        assert d["tokens_out"] == 100
        assert d["latency_ms"] == 500.0
        assert d["operation"] == "proposal"
        assert "id" in d
        assert "timestamp" in d
        assert "cost_usd" in d

    def test_to_dict_cost_is_string(self):
        r = ModelCallRecord(cost_usd=Decimal("1.23"))
        d = r.to_dict()
        assert d["cost_usd"] == "1.23"


# ---------------------------------------------------------------------------
# DebateUsageSummary
# ---------------------------------------------------------------------------


class TestDebateUsageSummary:
    """Unit tests for DebateUsageSummary dataclass."""

    def test_defaults(self):
        s = DebateUsageSummary()
        assert s.total_calls == 0
        assert s.total_tokens == 0
        assert s.avg_latency_ms == 0.0
        assert s.wall_time_ms == 0.0

    def test_avg_latency(self):
        s = DebateUsageSummary(total_calls=4, total_latency_ms=1000.0)
        assert s.avg_latency_ms == 250.0

    def test_total_tokens_property(self):
        s = DebateUsageSummary(total_tokens_in=300, total_tokens_out=200)
        assert s.total_tokens == 500

    def test_wall_time(self):
        t1 = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(seconds=5)
        s = DebateUsageSummary(first_call=t1, last_call=t2)
        assert s.wall_time_ms == 5000.0

    def test_to_dict_includes_all_keys(self):
        s = DebateUsageSummary(debate_id="d1", total_calls=2)
        d = s.to_dict()
        expected_keys = {
            "debate_id",
            "workspace_id",
            "total_calls",
            "total_tokens_in",
            "total_tokens_out",
            "total_tokens_cached",
            "total_tokens",
            "total_cost_usd",
            "total_latency_ms",
            "avg_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "wall_time_ms",
            "by_agent",
            "by_model",
            "by_operation",
            "first_call",
            "last_call",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_min_latency_zero_when_no_calls(self):
        s = DebateUsageSummary()
        d = s.to_dict()
        assert d["min_latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# Event constructors
# ---------------------------------------------------------------------------


class TestEventConstructors:
    """Tests for telemetry event bridge functions."""

    def test_model_call_event(self):
        record = ModelCallRecord(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=500,
            tokens_out=200,
            latency_ms=800.0,
            operation="critique",
        )
        event = model_call_event(record)
        assert event.event_type == TelemetryEventType.BENCHMARK_RUN
        assert event.debate_id == "d1"
        assert event.duration_ms == 800.0
        props = event.properties
        assert props["model_usage_type"] == ModelUsageEventType.MODEL_CALL.value
        assert props["provider"] == "anthropic"
        assert props["model"] == "claude-sonnet-4.6"
        assert props["tokens_in"] == 500
        assert props["tokens_out"] == 200
        assert props["operation"] == "critique"

    def test_debate_cost_summary_event(self):
        summary = DebateUsageSummary(
            debate_id="d2",
            workspace_id="ws1",
            total_calls=10,
            total_tokens_in=5000,
            total_tokens_out=3000,
            total_cost_usd=Decimal("0.42"),
            total_latency_ms=12000.0,
        )
        summary.by_model["anthropic/claude-sonnet-4.6"] = {"calls": 10}
        summary.by_agent["agent-1"] = {"calls": 5}

        event = debate_cost_summary_event(summary)
        assert event.event_type == TelemetryEventType.BENCHMARK_RUN
        assert event.debate_id == "d2"
        assert event.workspace_id == "ws1"
        assert event.duration_ms == 12000.0
        props = event.properties
        assert props["model_usage_type"] == ModelUsageEventType.DEBATE_COST_SUMMARY.value
        assert props["total_calls"] == 10
        assert props["total_tokens"] == 8000
        assert "anthropic/claude-sonnet-4.6" in props["models_used"]
        assert "agent-1" in props["agents_used"]

    def test_event_to_dict_serializable(self):
        record = ModelCallRecord(debate_id="d1", provider="openai", model="gpt-4o")
        event = model_call_event(record)
        d = event.to_dict()
        assert isinstance(d, dict)
        assert d["event_type"] == "benchmark_run"


# ---------------------------------------------------------------------------
# ModelUsagePipeline
# ---------------------------------------------------------------------------


class TestModelUsagePipeline:
    """Tests for the aggregation pipeline."""

    def test_record_model_call_returns_record(self):
        p = ModelUsagePipeline()
        rec = p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=1000,
            tokens_out=500,
            latency_ms=1200.0,
            operation="proposal",
        )
        assert isinstance(rec, ModelCallRecord)
        assert rec.debate_id == "d1"
        assert rec.tokens_in == 1000

    def test_summary_created_on_first_call(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="openai",
            model="gpt-4o",
            tokens_in=100,
            tokens_out=50,
            latency_ms=300.0,
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert summary.total_calls == 1
        assert summary.total_tokens_in == 100
        assert summary.total_tokens_out == 50

    def test_summary_accumulates(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            operation="proposal",
        )
        p.record_model_call(
            debate_id="d1",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=200,
            tokens_out=100,
            latency_ms=400.0,
            operation="critique",
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert summary.total_calls == 2
        assert summary.total_tokens_in == 300
        assert summary.total_tokens_out == 150
        assert summary.min_latency_ms == 200.0
        assert summary.max_latency_ms == 400.0

    def test_by_agent_breakdown(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=150,
            tokens_out=75,
            latency_ms=300.0,
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert "a1" in summary.by_agent
        assert summary.by_agent["a1"]["calls"] == 2
        assert summary.by_agent["a1"]["tokens_in"] == 250

    def test_by_model_breakdown(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.record_model_call(
            debate_id="d1",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=200,
            tokens_out=100,
            latency_ms=400.0,
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert "anthropic/claude-sonnet-4.6" in summary.by_model
        assert "openai/gpt-4o" in summary.by_model
        assert summary.by_model["openai/gpt-4o"]["tokens_in"] == 200

    def test_by_operation_breakdown(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            operation="proposal",
        )
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=300,
            tokens_out=150,
            latency_ms=600.0,
            operation="critique",
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert "proposal" in summary.by_operation
        assert "critique" in summary.by_operation
        assert summary.by_operation["proposal"]["calls"] == 1
        assert summary.by_operation["critique"]["tokens_in"] == 300

    def test_empty_operation_not_tracked(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="openai",
            model="gpt-4o",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            operation="",
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert len(summary.by_operation) == 0

    def test_multiple_debates_independent(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.record_model_call(
            debate_id="d2",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=200,
            tokens_out=100,
            latency_ms=400.0,
        )
        assert p.get_debate_summary("d1") is not None
        assert p.get_debate_summary("d2") is not None
        assert p.get_debate_summary("d1").total_calls == 1
        assert p.get_debate_summary("d2").total_calls == 1

    def test_get_debate_summary_returns_none_for_unknown(self):
        p = ModelUsagePipeline()
        assert p.get_debate_summary("nonexistent") is None

    def test_get_debate_records(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        records = p.get_debate_records("d1")
        assert len(records) == 1
        assert records[0].agent_id == "a1"

    def test_get_debate_records_returns_empty_for_unknown(self):
        p = ModelUsagePipeline()
        assert p.get_debate_records("nonexistent") == []

    def test_get_all_debate_ids(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.record_model_call(
            debate_id="d2",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        ids = p.get_all_debate_ids()
        assert set(ids) == {"d1", "d2"}

    def test_clear_debate(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.clear_debate("d1")
        assert p.get_debate_summary("d1") is None
        assert p.get_debate_records("d1") == []

    def test_clear_all(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        p.record_model_call(
            debate_id="d2",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=200,
            tokens_out=100,
            latency_ms=400.0,
        )
        p.clear_all()
        assert p.get_all_debate_ids() == []

    def test_cost_computed_on_record(self):
        p = ModelUsagePipeline()
        rec = p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=1_000_000,
            tokens_out=500_000,
            latency_ms=5000.0,
        )
        # Cost should be computed (non-zero for known model)
        assert rec.cost_usd > 0

        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert summary.total_cost_usd > 0

    def test_model_version_recorded(self):
        p = ModelUsagePipeline()
        rec = p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            model_version="claude-sonnet-4-6-20260320",
        )
        assert rec.model_version == "claude-sonnet-4-6-20260320"

    def test_tokens_cached_tracked(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            tokens_cached=30,
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert summary.total_tokens_cached == 30

    def test_workspace_id_propagated(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            workspace_id="ws-42",
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        assert summary.workspace_id == "ws-42"

    def test_metadata_stored(self):
        p = ModelUsagePipeline()
        rec = p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
            metadata={"temperature": 0.7},
        )
        assert rec.metadata == {"temperature": 0.7}


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for module-level pipeline singleton."""

    def test_get_returns_pipeline(self):
        pipeline = get_model_usage_pipeline()
        assert isinstance(pipeline, ModelUsagePipeline)

    def test_set_and_get(self):
        original = get_model_usage_pipeline()
        custom = ModelUsagePipeline()
        set_model_usage_pipeline(custom)
        assert get_model_usage_pipeline() is custom
        # Restore
        set_model_usage_pipeline(original)

    def test_get_returns_same_instance(self):
        p1 = get_model_usage_pipeline()
        p2 = get_model_usage_pipeline()
        assert p1 is p2


# ---------------------------------------------------------------------------
# Summary serialization round-trip
# ---------------------------------------------------------------------------


class TestSummarySerialization:
    """Tests for DebateUsageSummary.to_dict output."""

    def test_summary_to_dict_after_calls(self):
        p = ModelUsagePipeline()
        p.record_model_call(
            debate_id="d1",
            agent_id="a1",
            provider="anthropic",
            model="claude-sonnet-4.6",
            tokens_in=500,
            tokens_out=250,
            latency_ms=1000.0,
            operation="proposal",
        )
        p.record_model_call(
            debate_id="d1",
            agent_id="a2",
            provider="openai",
            model="gpt-4o",
            tokens_in=300,
            tokens_out=150,
            latency_ms=800.0,
            operation="critique",
        )
        summary = p.get_debate_summary("d1")
        assert summary is not None
        d = summary.to_dict()

        assert d["debate_id"] == "d1"
        assert d["total_calls"] == 2
        assert d["total_tokens_in"] == 800
        assert d["total_tokens_out"] == 400
        assert d["total_tokens"] == 1200
        assert d["total_latency_ms"] == 1800.0
        assert d["avg_latency_ms"] == 900.0
        assert d["min_latency_ms"] == 800.0
        assert d["max_latency_ms"] == 1000.0
        assert "anthropic/claude-sonnet-4.6" in d["by_model"]
        assert "openai/gpt-4o" in d["by_model"]
        assert "proposal" in d["by_operation"]
        assert "critique" in d["by_operation"]

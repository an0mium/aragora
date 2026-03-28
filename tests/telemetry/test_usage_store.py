"""Tests for aragora.telemetry.usage_store — ModelUsageStore."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from aragora.telemetry.models import (
    AggregationInterval,
    ModelUsageRecord,
    UsageOperation,
    UsageQuery,
)
from aragora.telemetry.usage_store import ModelUsageStore


@pytest.fixture
def store(tmp_path: Path) -> ModelUsageStore:
    """Create a fresh ModelUsageStore backed by a temp SQLite DB."""
    return ModelUsageStore(db_path=tmp_path / "test_usage.db")


def _make_record(**overrides) -> ModelUsageRecord:
    """Helper to create a ModelUsageRecord with sensible defaults."""
    defaults = dict(
        debate_id="debate-1",
        session_id="session-1",
        workspace_id="ws-1",
        agent_id="agent-1",
        agent_name="claude",
        provider="anthropic",
        model="claude-3-opus",
        model_version="2024-02-15",
        tokens_input=500,
        tokens_output=200,
        tokens_cached=0,
        cost_usd=Decimal("0.0250"),
        latency_ms=800.0,
        operation=UsageOperation.DEBATE_ROUND,
        round_number=1,
        timestamp=datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ModelUsageRecord(**defaults)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestInsertAndGet:
    def test_insert_and_get_by_id(self, store: ModelUsageStore):
        rec = _make_record()
        store.insert(rec)
        fetched = store.get_by_id(rec.id)
        assert fetched is not None
        assert fetched.id == rec.id
        assert fetched.debate_id == "debate-1"
        assert fetched.provider == "anthropic"
        assert fetched.tokens_input == 500
        assert fetched.tokens_output == 200
        assert fetched.latency_ms == 800.0

    def test_get_nonexistent_returns_none(self, store: ModelUsageStore):
        assert store.get_by_id("no-such-id") is None

    def test_insert_batch(self, store: ModelUsageStore):
        records = [_make_record(round_number=i) for i in range(5)]
        count = store.insert_batch(records)
        assert count == 5
        assert store.count_records() == 5

    def test_insert_batch_empty(self, store: ModelUsageStore):
        assert store.insert_batch([]) == 0


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_by_debate_id(self, store: ModelUsageStore):
        store.insert(_make_record(debate_id="d-1"))
        store.insert(_make_record(debate_id="d-2"))
        results = store.query(UsageQuery(debate_id="d-1"))
        assert len(results) == 1
        assert results[0].debate_id == "d-1"

    def test_query_by_provider(self, store: ModelUsageStore):
        store.insert(_make_record(provider="anthropic"))
        store.insert(_make_record(provider="openai"))
        results = store.query(UsageQuery(provider="openai"))
        assert len(results) == 1
        assert results[0].provider == "openai"

    def test_query_by_operation(self, store: ModelUsageStore):
        store.insert(_make_record(operation=UsageOperation.PROPOSAL))
        store.insert(_make_record(operation=UsageOperation.CRITIQUE))
        results = store.query(UsageQuery(operation=UsageOperation.CRITIQUE))
        assert len(results) == 1
        assert results[0].operation == UsageOperation.CRITIQUE

    def test_query_time_range(self, store: ModelUsageStore):
        t1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 15, tzinfo=timezone.utc)
        t3 = datetime(2026, 3, 27, tzinfo=timezone.utc)
        store.insert(_make_record(timestamp=t1))
        store.insert(_make_record(timestamp=t2))
        store.insert(_make_record(timestamp=t3))
        results = store.query(
            UsageQuery(
                start_time=datetime(2026, 3, 10, tzinfo=timezone.utc),
                end_time=datetime(2026, 3, 20, tzinfo=timezone.utc),
            )
        )
        assert len(results) == 1

    def test_query_limit_offset(self, store: ModelUsageStore):
        for i in range(10):
            store.insert(
                _make_record(
                    round_number=i,
                    timestamp=datetime(2026, 3, 27, 10, i, 0, tzinfo=timezone.utc),
                )
            )
        results = store.query(UsageQuery(limit=3, offset=0))
        assert len(results) == 3

    def test_count_records_with_filter(self, store: ModelUsageStore):
        store.insert(_make_record(provider="anthropic"))
        store.insert(_make_record(provider="openai"))
        store.insert(_make_record(provider="anthropic"))
        assert store.count_records(UsageQuery(provider="anthropic")) == 2
        assert store.count_records() == 3


# ---------------------------------------------------------------------------
# Session Summary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_session_summary(self, store: ModelUsageStore):
        base_time = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        store.insert(
            _make_record(
                debate_id="d-1",
                tokens_input=500,
                tokens_output=200,
                cost_usd=Decimal("0.02"),
                latency_ms=100.0,
                timestamp=base_time,
            )
        )
        store.insert(
            _make_record(
                debate_id="d-1",
                tokens_input=300,
                tokens_output=150,
                cost_usd=Decimal("0.01"),
                latency_ms=200.0,
                provider="openai",
                model="gpt-4",
                timestamp=base_time + timedelta(seconds=30),
            )
        )

        summary = store.get_session_summary("d-1")
        assert summary is not None
        assert summary.debate_id == "d-1"
        assert summary.total_requests == 2
        assert summary.total_tokens_input == 800
        assert summary.total_tokens_output == 350
        assert summary.total_cost_usd == Decimal("0.03")
        assert summary.avg_latency_ms == 150.0
        assert summary.max_latency_ms == 200.0
        assert summary.min_latency_ms == 100.0
        assert summary.p95_latency_ms >= 100.0
        assert "claude-3-opus" in summary.models_used
        assert "gpt-4" in summary.models_used
        assert "anthropic" in summary.providers_used
        assert "openai" in summary.providers_used
        assert summary.duration_seconds == 30.0

    def test_session_summary_nonexistent(self, store: ModelUsageStore):
        assert store.get_session_summary("no-such") is None


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


class TestAggregations:
    def test_aggregate_by_model(self, store: ModelUsageStore):
        store.insert(_make_record(model="opus", cost_usd=Decimal("0.05")))
        store.insert(_make_record(model="opus", cost_usd=Decimal("0.03")))
        store.insert(_make_record(model="sonnet", cost_usd=Decimal("0.01")))

        aggs = store.aggregate_by("model")
        assert len(aggs) == 2
        # Sorted by cost desc
        assert aggs[0].group_value == "opus"
        assert aggs[0].total_requests == 2
        assert aggs[0].total_cost_usd == Decimal("0.08")

    def test_aggregate_by_provider(self, store: ModelUsageStore):
        store.insert(_make_record(provider="anthropic"))
        store.insert(_make_record(provider="openai"))
        aggs = store.aggregate_by("provider")
        assert len(aggs) == 2

    def test_aggregate_by_invalid_dimension(self, store: ModelUsageStore):
        with pytest.raises(ValueError, match="Invalid dimension"):
            store.aggregate_by("nonexistent_column")

    def test_aggregate_with_filter(self, store: ModelUsageStore):
        store.insert(_make_record(workspace_id="ws-1", model="opus"))
        store.insert(_make_record(workspace_id="ws-1", model="sonnet"))
        store.insert(_make_record(workspace_id="ws-2", model="opus"))

        aggs = store.aggregate_by("model", q=UsageQuery(workspace_id="ws-1"))
        assert len(aggs) == 2
        total_reqs = sum(a.total_requests for a in aggs)
        assert total_reqs == 2

    def test_time_series_hourly(self, store: ModelUsageStore):
        base = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        store.insert(_make_record(timestamp=base))
        store.insert(_make_record(timestamp=base + timedelta(minutes=30)))
        store.insert(_make_record(timestamp=base + timedelta(hours=1)))

        series = store.get_time_series(AggregationInterval.HOURLY)
        assert len(series) == 2
        assert series[0]["total_requests"] == 2  # first hour
        assert series[1]["total_requests"] == 1  # second hour

    def test_time_series_daily(self, store: ModelUsageStore):
        d1 = datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)
        store.insert(_make_record(timestamp=d1))
        store.insert(_make_record(timestamp=d2))
        series = store.get_time_series(AggregationInterval.DAILY)
        assert len(series) == 2


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


class TestMaintenance:
    def test_delete_before(self, store: ModelUsageStore):
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2026, 3, 27, tzinfo=timezone.utc)
        store.insert(_make_record(timestamp=old))
        store.insert(_make_record(timestamp=recent))

        deleted = store.delete_before(datetime(2026, 2, 1, tzinfo=timezone.utc))
        assert deleted == 1
        assert store.count_records() == 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_version(self, store: ModelUsageStore):
        assert store.get_schema_version() == 1

    def test_table_exists(self, store: ModelUsageStore):
        info = store.get_table_info("model_usage")
        col_names = {c["name"] for c in info}
        assert "id" in col_names
        assert "debate_id" in col_names
        assert "tokens_input" in col_names
        assert "tokens_output" in col_names
        assert "latency_ms" in col_names
        assert "cost_usd" in col_names
        assert "model_version" in col_names
        assert "timestamp" in col_names
        assert "metadata_json" in col_names

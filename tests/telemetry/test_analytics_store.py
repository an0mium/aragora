"""Tests for TelemetryAnalyticsStore."""

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from aragora.telemetry.analytics_store import TelemetryAnalyticsStore
from aragora.telemetry.models import (
    DebateUsageSummary,
    ModelUsageRecord,
    RequestStatus,
    TokenCount,
)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test_analytics.db"
    return TelemetryAnalyticsStore(db)


def _make_record(
    debate_id="d-1",
    workspace_id="ws-1",
    agent_id="a-1",
    provider="anthropic",
    model="claude-sonnet-4.6",
    input_tokens=500,
    output_tokens=200,
    latency_ms=1200.0,
    cost_usd="0.0085",
    status=RequestStatus.SUCCESS,
    round_number=1,
    timestamp=None,
    **kwargs,
):
    return ModelUsageRecord(
        debate_id=debate_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
        provider=provider,
        model=model,
        tokens=TokenCount(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=latency_ms,
        cost_usd=Decimal(cost_usd),
        status=status,
        round_number=round_number,
        timestamp=timestamp or datetime.now(timezone.utc),
        **kwargs,
    )


class TestRecordAndGet:
    def test_record_and_get(self, store):
        r = _make_record()
        store.record(r)
        fetched = store.get(r.id)
        assert fetched is not None
        assert fetched.debate_id == "d-1"
        assert fetched.tokens.input_tokens == 500
        assert fetched.cost_usd == Decimal("0.0085")

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_upsert(self, store):
        r = _make_record()
        store.record(r)
        r.cost_usd = Decimal("0.01")
        store.record(r)
        fetched = store.get(r.id)
        assert fetched.cost_usd == Decimal("0.01")


class TestRecordBatch:
    def test_batch_insert(self, store):
        records = [_make_record(agent_id=f"a-{i}") for i in range(50)]
        count = store.record_batch(records)
        assert count == 50
        assert len(store.query_by_debate("d-1")) == 50

    def test_batch_empty(self, store):
        assert store.record_batch([]) == 0


class TestQueryByDebate:
    def test_returns_ordered(self, store):
        now = datetime.now(timezone.utc)
        for i in range(5):
            store.record(
                _make_record(
                    agent_id=f"a-{i}",
                    timestamp=now + timedelta(seconds=i),
                )
            )
        results = store.query_by_debate("d-1")
        assert len(results) == 5
        assert results[0].timestamp <= results[-1].timestamp

    def test_filters_debate(self, store):
        store.record(_make_record(debate_id="d-1"))
        store.record(_make_record(debate_id="d-2"))
        assert len(store.query_by_debate("d-1")) == 1
        assert len(store.query_by_debate("d-2")) == 1


class TestQueryByWorkspace:
    def test_basic(self, store):
        store.record(_make_record(workspace_id="ws-A"))
        store.record(_make_record(workspace_id="ws-B"))
        results = store.query_by_workspace("ws-A")
        assert len(results) == 1

    def test_time_window(self, store):
        now = datetime.now(timezone.utc)
        store.record(_make_record(timestamp=now - timedelta(hours=2)))
        store.record(_make_record(timestamp=now, agent_id="a-2"))
        results = store.query_by_workspace("ws-1", since=now - timedelta(hours=1))
        assert len(results) == 1

    def test_limit(self, store):
        for i in range(10):
            store.record(_make_record(agent_id=f"a-{i}"))
        results = store.query_by_workspace("ws-1", limit=3)
        assert len(results) == 3


class TestQueryErrors:
    def test_filters_non_success(self, store):
        store.record(_make_record(status=RequestStatus.SUCCESS))
        store.record(
            _make_record(
                status=RequestStatus.ERROR,
                agent_id="a-err",
                error_message="boom",
            )
        )
        store.record(
            _make_record(
                status=RequestStatus.TIMEOUT,
                agent_id="a-timeout",
            )
        )
        errors = store.query_errors()
        assert len(errors) == 2
        assert all(r.status != RequestStatus.SUCCESS for r in errors)


class TestSummariseDebate:
    def test_empty_debate(self, store):
        s = store.summarise_debate("nonexistent")
        assert s.total_requests == 0
        assert s.total_cost_usd == Decimal("0")

    def test_full_summary(self, store):
        now = datetime.now(timezone.utc)
        store.record(
            _make_record(
                agent_id="a-1",
                provider="anthropic",
                model="claude-sonnet-4.6",
                cost_usd="0.01",
                latency_ms=100.0,
                round_number=1,
                timestamp=now,
            )
        )
        store.record(
            _make_record(
                agent_id="a-2",
                provider="openai",
                model="gpt-4o",
                cost_usd="0.02",
                latency_ms=200.0,
                round_number=1,
                timestamp=now + timedelta(seconds=1),
            )
        )
        store.record(
            _make_record(
                agent_id="a-3",
                provider="anthropic",
                model="claude-sonnet-4.6",
                cost_usd="0.005",
                latency_ms=50.0,
                round_number=2,
                status=RequestStatus.ERROR,
                timestamp=now + timedelta(seconds=2),
            )
        )

        s = store.summarise_debate("d-1")
        assert s.total_requests == 3
        assert s.successful_requests == 2
        assert s.failed_requests == 1
        assert s.total_input_tokens == 1500
        assert s.total_output_tokens == 600
        assert s.total_cost_usd == Decimal("0.035")
        assert s.cost_by_provider["anthropic"] == Decimal("0.015")
        assert s.cost_by_provider["openai"] == Decimal("0.02")
        assert s.cost_by_model["claude-sonnet-4.6"] == Decimal("0.015")
        assert s.requests_by_round[1] == 2
        assert s.requests_by_round[2] == 1
        assert s.min_latency_ms == 50.0
        assert s.max_latency_ms == 200.0
        assert s.started_at is not None
        assert s.ended_at is not None

    def test_latency_p95(self, store):
        now = datetime.now(timezone.utc)
        for i in range(20):
            store.record(
                _make_record(
                    agent_id=f"a-{i}",
                    latency_ms=float(100 + i * 10),
                    timestamp=now + timedelta(seconds=i),
                )
            )
        s = store.summarise_debate("d-1")
        assert s.p95_latency_ms >= s.avg_latency_ms


class TestCostByProvider:
    def test_grouped(self, store):
        store.record(_make_record(provider="anthropic", cost_usd="0.01"))
        store.record(_make_record(provider="openai", cost_usd="0.02", agent_id="a-2"))
        store.record(_make_record(provider="anthropic", cost_usd="0.03", agent_id="a-3"))
        result = store.cost_by_provider("ws-1")
        assert len(result) == 2
        # Anthropic total should be ~0.04
        assert result["anthropic"] > result["openai"]


class TestTotalTokens:
    def test_sums(self, store):
        store.record(_make_record(input_tokens=100, output_tokens=50))
        store.record(_make_record(input_tokens=200, output_tokens=100, agent_id="a-2"))
        totals = store.total_tokens("ws-1")
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 150


class TestModelVersionTracking:
    def test_version_stored(self, store):
        r = _make_record(model_version="2026-03-01")
        store.record(r)
        fetched = store.get(r.id)
        assert fetched.model_version == "2026-03-01"

    def test_version_null(self, store):
        r = _make_record()
        store.record(r)
        fetched = store.get(r.id)
        assert fetched.model_version is None


class TestMetadataStorage:
    def test_metadata_round_trip(self, store):
        r = _make_record(metadata={"thinking_tokens": 150, "stream": True})
        store.record(r)
        fetched = store.get(r.id)
        assert fetched.metadata["thinking_tokens"] == 150
        assert fetched.metadata["stream"] is True

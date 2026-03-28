"""Tests for the model usage telemetry pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from aragora.telemetry.model_usage import (
    AggregationWindow,
    ModelUsageAggregator,
    ModelUsageCollector,
    ModelUsageRecord,
    ModelUsageStore,
    UsageAggregate,
    track_model_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    debate_id: str = "d-1",
    session_id: str = "s-1",
    agent_id: str = "a-1",
    agent_name: str = "claude",
    provider: str = "anthropic",
    model: str = "claude-opus-4-6",
    model_version: str = "2026-03-01",
    operation: str = "propose",
    tokens_in: int = 500,
    tokens_out: int = 200,
    tokens_cached: int = 0,
    latency_ms: float = 1200.0,
    cost_usd: str = "0.0125",
    success: bool = True,
    error_type: str | None = None,
    metadata: dict | None = None,
) -> ModelUsageRecord:
    return ModelUsageRecord(
        record_id="rec-test-001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        debate_id=debate_id,
        session_id=session_id,
        agent_id=agent_id,
        agent_name=agent_name,
        provider=provider,
        model=model,
        model_version=model_version,
        operation=operation,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_cached=tokens_cached,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        success=success,
        error_type=error_type,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------


class TestModelUsageRecord:
    def test_total_tokens(self):
        rec = _make_record(tokens_in=300, tokens_out=100)
        assert rec.total_tokens == 400

    def test_to_dict_round_trip(self):
        rec = _make_record()
        d = rec.to_dict()
        assert d["provider"] == "anthropic"
        assert d["model"] == "claude-opus-4-6"
        assert d["tokens_in"] == 500
        assert d["cost_usd"] == "0.0125"

    def test_frozen(self):
        rec = _make_record()
        with pytest.raises(AttributeError):
            rec.tokens_in = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelUsageStore
# ---------------------------------------------------------------------------


class TestModelUsageStore:
    @pytest.fixture
    def store(self) -> ModelUsageStore:
        return ModelUsageStore()

    @pytest.mark.asyncio
    async def test_store_and_count(self, store: ModelUsageStore):
        await store.store([_make_record(), _make_record()])
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_query_by_debate(self, store: ModelUsageStore):
        await store.store(
            [
                _make_record(debate_id="d-1"),
                _make_record(debate_id="d-2"),
                _make_record(debate_id="d-1"),
            ]
        )
        results = await store.query(debate_id="d-1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_provider(self, store: ModelUsageStore):
        await store.store(
            [
                _make_record(provider="anthropic"),
                _make_record(provider="openai"),
            ]
        )
        results = await store.query(provider="openai")
        assert len(results) == 1
        assert results[0].provider == "openai"

    @pytest.mark.asyncio
    async def test_query_by_model(self, store: ModelUsageStore):
        await store.store(
            [
                _make_record(model="claude-opus-4-6"),
                _make_record(model="gpt-4.1"),
            ]
        )
        results = await store.query(model="gpt-4.1")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_agent(self, store: ModelUsageStore):
        await store.store(
            [
                _make_record(agent_id="a-1"),
                _make_record(agent_id="a-2"),
            ]
        )
        results = await store.query(agent_id="a-2")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_limit(self, store: ModelUsageStore):
        await store.store([_make_record() for _ in range(10)])
        results = await store.query(limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_since_until(self, store: ModelUsageStore):
        r1 = ModelUsageRecord(
            record_id="r1",
            timestamp="2026-03-01T00:00:00+00:00",
            debate_id="d-1",
            session_id="s-1",
            agent_id="a-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4-6",
            model_version="",
            operation="propose",
            tokens_in=100,
            tokens_out=50,
            tokens_cached=0,
            latency_ms=500.0,
            cost_usd="0.01",
            success=True,
        )
        r2 = ModelUsageRecord(
            record_id="r2",
            timestamp="2026-03-15T00:00:00+00:00",
            debate_id="d-1",
            session_id="s-1",
            agent_id="a-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4-6",
            model_version="",
            operation="propose",
            tokens_in=100,
            tokens_out=50,
            tokens_cached=0,
            latency_ms=500.0,
            cost_usd="0.01",
            success=True,
        )
        await store.store([r1, r2])
        results = await store.query(
            since=datetime(2026, 3, 10, tzinfo=timezone.utc),
        )
        assert len(results) == 1
        assert results[0].record_id == "r2"

    @pytest.mark.asyncio
    async def test_clear(self, store: ModelUsageStore):
        await store.store([_make_record()])
        await store.clear()
        assert await store.count() == 0


# ---------------------------------------------------------------------------
# ModelUsageCollector
# ---------------------------------------------------------------------------


class TestModelUsageCollector:
    @pytest.fixture
    def store(self) -> ModelUsageStore:
        return ModelUsageStore()

    @pytest.fixture
    def collector(self, store: ModelUsageStore) -> ModelUsageCollector:
        return ModelUsageCollector(backend=store, buffer_size=10)

    @pytest.mark.asyncio
    async def test_record_and_flush(self, collector: ModelUsageCollector, store: ModelUsageStore):
        await collector.record(_make_record())
        assert collector.pending == 1
        flushed = await collector.flush()
        assert flushed == 1
        assert collector.pending == 0
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_full(
        self,
        store: ModelUsageStore,
    ):
        collector = ModelUsageCollector(backend=store, buffer_size=3)
        for _ in range(3):
            await collector.record(_make_record())
        # buffer_size reached → auto-flushed
        assert collector.pending == 0
        assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_track_call_success(
        self,
        collector: ModelUsageCollector,
        store: ModelUsageStore,
    ):
        async with collector.track_call(
            debate_id="d-1",
            session_id="s-1",
            agent_id="a-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4-6",
            operation="propose",
        ) as ctx:
            ctx.set_tokens(tokens_in=500, tokens_out=200)
            ctx.set_cost("0.0125")
            ctx.set_model_version("2026-03-01")
            ctx.add_metadata("round", 1)

        await collector.flush()
        records = await store.query()
        assert len(records) == 1
        rec = records[0]
        assert rec.tokens_in == 500
        assert rec.tokens_out == 200
        assert rec.cost_usd == "0.0125"
        assert rec.model_version == "2026-03-01"
        assert rec.success is True
        assert rec.latency_ms >= 0  # may round to 0.0 on fast machines
        assert rec.metadata["round"] == 1

    @pytest.mark.asyncio
    async def test_track_call_error(
        self,
        collector: ModelUsageCollector,
        store: ModelUsageStore,
    ):
        with pytest.raises(ValueError, match="boom"):
            async with collector.track_call(
                debate_id="d-1",
                session_id="s-1",
                agent_id="a-1",
                agent_name="claude",
                provider="anthropic",
                model="claude-opus-4-6",
                operation="propose",
            ) as ctx:
                ctx.set_tokens(tokens_in=100, tokens_out=0)
                raise ValueError("boom")

        await collector.flush()
        records = await store.query()
        assert len(records) == 1
        rec = records[0]
        assert rec.success is False
        assert rec.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_flush_empty_returns_zero(self, collector: ModelUsageCollector):
        assert await collector.flush() == 0

    @pytest.mark.asyncio
    async def test_track_model_call_convenience(
        self,
        collector: ModelUsageCollector,
        store: ModelUsageStore,
    ):
        async with track_model_call(
            collector,
            debate_id="d-1",
            session_id="s-1",
            agent_id="a-1",
            agent_name="gpt",
            provider="openai",
            model="gpt-4.1",
            operation="critique",
        ) as ctx:
            ctx.set_tokens(tokens_in=300, tokens_out=150)
            ctx.set_cost("0.008")

        await collector.flush()
        records = await store.query()
        assert len(records) == 1
        assert records[0].provider == "openai"

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, store: ModelUsageStore):
        collector = ModelUsageCollector(backend=store, flush_interval_seconds=1)
        await collector.start()
        await collector.record(_make_record())
        # Wait for periodic flush
        await asyncio.sleep(1.5)
        assert await store.count() >= 1
        await collector.stop()


# ---------------------------------------------------------------------------
# ModelUsageAggregator
# ---------------------------------------------------------------------------


class TestModelUsageAggregator:
    @pytest.fixture
    def records(self) -> list[ModelUsageRecord]:
        return [
            _make_record(
                provider="anthropic",
                model="claude-opus-4-6",
                agent_name="claude",
                debate_id="d-1",
                operation="propose",
                tokens_in=500,
                tokens_out=200,
                latency_ms=1200.0,
                cost_usd="0.0125",
            ),
            _make_record(
                provider="anthropic",
                model="claude-opus-4-6",
                agent_name="claude",
                debate_id="d-1",
                operation="critique",
                tokens_in=400,
                tokens_out=150,
                latency_ms=900.0,
                cost_usd="0.010",
            ),
            _make_record(
                provider="openai",
                model="gpt-4.1",
                agent_name="gpt",
                debate_id="d-2",
                operation="propose",
                tokens_in=600,
                tokens_out=250,
                latency_ms=1500.0,
                cost_usd="0.008",
                success=False,
                error_type="TimeoutError",
            ),
        ]

    def test_by_model(self, records: list[ModelUsageRecord]):
        groups = ModelUsageAggregator.by_model(records)
        assert len(groups) == 2
        claude = groups["anthropic/claude-opus-4-6"]
        assert claude.record_count == 2
        assert claude.total_tokens_in == 900
        assert claude.total_tokens_out == 350
        assert claude.total_cost_usd == Decimal("0.0225")
        assert claude.error_count == 0

        gpt = groups["openai/gpt-4.1"]
        assert gpt.record_count == 1
        assert gpt.error_count == 1

    def test_by_debate(self, records: list[ModelUsageRecord]):
        groups = ModelUsageAggregator.by_debate(records)
        assert groups["d-1"].record_count == 2
        assert groups["d-2"].record_count == 1

    def test_by_agent(self, records: list[ModelUsageRecord]):
        groups = ModelUsageAggregator.by_agent(records)
        assert groups["claude"].record_count == 2
        assert groups["gpt"].record_count == 1

    def test_by_operation(self, records: list[ModelUsageRecord]):
        groups = ModelUsageAggregator.by_operation(records)
        assert groups["propose"].record_count == 2
        assert groups["critique"].record_count == 1

    def test_total(self, records: list[ModelUsageRecord]):
        agg = ModelUsageAggregator.total(records)
        assert agg.record_count == 3
        assert agg.total_tokens == 500 + 200 + 400 + 150 + 600 + 250
        assert agg.total_cost_usd == Decimal("0.0305")
        assert agg.error_count == 1
        assert agg.error_rate == pytest.approx(1 / 3, abs=0.01)

    def test_latency_stats(self, records: list[ModelUsageRecord]):
        agg = ModelUsageAggregator.total(records)
        assert agg.min_latency_ms == 900.0
        assert agg.max_latency_ms == 1500.0
        assert agg.avg_latency_ms == pytest.approx(1200.0)

    def test_to_dict(self, records: list[ModelUsageRecord]):
        agg = ModelUsageAggregator.total(records)
        d = agg.to_dict()
        assert d["group_key"] == "total"
        assert d["record_count"] == 3
        assert "total_tokens" in d
        assert "avg_latency_ms" in d
        assert "error_rate" in d

    def test_empty_aggregate(self):
        agg = ModelUsageAggregator.total([])
        assert agg.record_count == 0
        assert agg.avg_latency_ms == 0.0
        assert agg.error_rate == 0.0


# ---------------------------------------------------------------------------
# UsageAggregate edge cases
# ---------------------------------------------------------------------------


class TestUsageAggregate:
    def test_defaults(self):
        agg = UsageAggregate(group_key="test")
        assert agg.total_tokens == 0
        assert agg.avg_latency_ms == 0.0
        assert agg.error_rate == 0.0
        assert agg.min_latency_ms == float("inf")

    def test_to_dict_min_latency_inf(self):
        agg = UsageAggregate(group_key="empty")
        d = agg.to_dict()
        # inf should survive serialization
        assert d["min_latency_ms"] == float("inf")

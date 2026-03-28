"""Tests for aragora.telemetry.model_usage — model usage telemetry pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from aragora.telemetry.model_usage import (
    ModelUsageAggregate,
    ModelUsageAggregator,
    ModelUsagePipeline,
    ModelUsageRecord,
    ModelUsageStore,
    ModelUsageTimer,
    get_model_usage_pipeline,
    reset_model_usage_pipeline,
    set_model_usage_pipeline,
)


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------


class TestModelUsageRecord:
    """Tests for the ModelUsageRecord dataclass."""

    def test_defaults(self):
        """Record fields default to sensible zero/empty values."""
        record = ModelUsageRecord()
        assert record.agent_name == ""
        assert record.provider == ""
        assert record.model == ""
        assert record.tokens_in == 0
        assert record.tokens_out == 0
        assert record.tokens_cached == 0
        assert record.latency_ms == 0.0
        assert record.cost_usd == Decimal("0")
        assert record.debate_id == ""
        assert record.round_number is None
        assert record.operation == ""
        assert record.metadata == {}
        assert record.id  # UUID should be generated

    def test_total_tokens(self):
        """total_tokens sums input + output + cached."""
        record = ModelUsageRecord(tokens_in=100, tokens_out=50, tokens_cached=25)
        assert record.total_tokens == 175

    def test_calculate_cost(self):
        """calculate_cost should compute a non-zero cost for known models."""
        record = ModelUsageRecord(
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=1_000_000,
            tokens_out=0,
        )
        cost = record.calculate_cost()
        assert cost > Decimal("0")
        assert record.cost_usd == cost

    def test_to_dict_round_trip(self):
        """to_dict / from_dict should preserve all fields."""
        record = ModelUsageRecord(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=500,
            tokens_out=200,
            tokens_cached=50,
            latency_ms=1234.5,
            cost_usd=Decimal("0.0035"),
            debate_id="d-1",
            round_number=3,
            operation="proposal",
            metadata={"key": "value"},
        )
        data = record.to_dict()
        restored = ModelUsageRecord.from_dict(data)
        assert restored.agent_name == "claude"
        assert restored.provider == "anthropic"
        assert restored.model == "claude-opus-4.6"
        assert restored.tokens_in == 500
        assert restored.tokens_out == 200
        assert restored.tokens_cached == 50
        assert restored.latency_ms == 1234.5
        assert restored.cost_usd == Decimal("0.0035")
        assert restored.debate_id == "d-1"
        assert restored.round_number == 3
        assert restored.operation == "proposal"
        assert restored.metadata == {"key": "value"}

    def test_from_dict_with_defaults(self):
        """from_dict should handle missing keys gracefully."""
        record = ModelUsageRecord.from_dict({})
        assert record.tokens_in == 0
        assert record.agent_name == ""


# ---------------------------------------------------------------------------
# ModelUsageAggregate
# ---------------------------------------------------------------------------


class TestModelUsageAggregate:
    """Tests for ModelUsageAggregate."""

    def test_empty_aggregate(self):
        """An empty aggregate should have zero values."""
        agg = ModelUsageAggregate()
        assert agg.total_requests == 0
        assert agg.avg_latency_ms == 0.0
        assert agg.avg_tokens_per_request == 0.0
        assert agg.avg_cost_per_request == Decimal("0")

    def test_add_record(self):
        """add_record should update aggregate stats."""
        agg = ModelUsageAggregate()
        record = ModelUsageRecord(
            tokens_in=100,
            tokens_out=50,
            latency_ms=500.0,
            cost_usd=Decimal("0.01"),
            model="gpt-4o",
            provider="openai",
        )
        agg.add_record(record)
        assert agg.total_requests == 1
        assert agg.total_tokens_in == 100
        assert agg.total_tokens_out == 50
        assert agg.total_cost_usd == Decimal("0.01")
        assert agg.avg_latency_ms == 500.0
        assert "gpt-4o" in agg.models_seen
        assert "openai" in agg.providers_seen

    def test_multiple_records(self):
        """Aggregate correctly accumulates multiple records."""
        agg = ModelUsageAggregate()
        for i in range(3):
            agg.add_record(
                ModelUsageRecord(
                    tokens_in=100,
                    tokens_out=50,
                    latency_ms=float(100 * (i + 1)),
                    cost_usd=Decimal("0.01"),
                )
            )
        assert agg.total_requests == 3
        assert agg.total_tokens_in == 300
        assert agg.total_tokens_out == 150
        assert agg.min_latency_ms == 100.0
        assert agg.max_latency_ms == 300.0
        assert agg.avg_latency_ms == 200.0

    def test_to_dict(self):
        """to_dict produces a serializable dictionary."""
        agg = ModelUsageAggregate()
        agg.add_record(
            ModelUsageRecord(
                tokens_in=100,
                tokens_out=50,
                latency_ms=250.0,
                cost_usd=Decimal("0.005"),
                model="claude-opus-4.6",
                provider="anthropic",
            )
        )
        d = agg.to_dict()
        assert d["total_requests"] == 1
        assert d["total_tokens_in"] == 100
        assert d["models_seen"] == ["claude-opus-4.6"]
        assert d["providers_seen"] == ["anthropic"]


# ---------------------------------------------------------------------------
# ModelUsageStore
# ---------------------------------------------------------------------------


class TestModelUsageStore:
    """Tests for ModelUsageStore."""

    def test_add_and_count(self):
        """Records can be added and counted."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(agent_name="a"))
        store.add(ModelUsageRecord(agent_name="b"))
        assert store.count() == 2

    def test_capacity_eviction(self):
        """Oldest records are evicted when capacity is exceeded."""
        store = ModelUsageStore(max_records=3)
        for i in range(5):
            store.add(ModelUsageRecord(agent_name=f"agent-{i}"))
        assert store.count() == 3
        names = [r.agent_name for r in store.get_all()]
        assert names == ["agent-2", "agent-3", "agent-4"]

    def test_query_by_debate_id(self):
        """Query filters by debate_id."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(debate_id="d-1"))
        store.add(ModelUsageRecord(debate_id="d-2"))
        store.add(ModelUsageRecord(debate_id="d-1"))
        results = store.query(debate_id="d-1")
        assert len(results) == 2

    def test_query_by_agent_name(self):
        """Query filters by agent_name."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(agent_name="claude"))
        store.add(ModelUsageRecord(agent_name="gpt"))
        results = store.query(agent_name="claude")
        assert len(results) == 1

    def test_query_by_model(self):
        """Query filters by model."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(model="claude-opus-4.6"))
        store.add(ModelUsageRecord(model="gpt-4o"))
        results = store.query(model="gpt-4o")
        assert len(results) == 1

    def test_query_by_provider(self):
        """Query filters by provider."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(provider="anthropic"))
        store.add(ModelUsageRecord(provider="openai"))
        results = store.query(provider="anthropic")
        assert len(results) == 1

    def test_query_by_operation(self):
        """Query filters by operation."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(operation="proposal"))
        store.add(ModelUsageRecord(operation="critique"))
        results = store.query(operation="proposal")
        assert len(results) == 1

    def test_query_since_until(self):
        """Query filters by timestamp range."""
        store = ModelUsageStore()
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)
        store.add(ModelUsageRecord(timestamp=t1))
        store.add(ModelUsageRecord(timestamp=t2))
        store.add(ModelUsageRecord(timestamp=t3))
        results = store.query(since=t2)
        assert len(results) == 2
        results = store.query(until=t2)
        assert len(results) == 2
        results = store.query(since=t2, until=t2)
        assert len(results) == 1

    def test_query_limit(self):
        """Query respects limit."""
        store = ModelUsageStore()
        for _ in range(10):
            store.add(ModelUsageRecord())
        results = store.query(limit=3)
        assert len(results) == 3

    def test_query_combined_filters(self):
        """Multiple filters are AND-combined."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord(agent_name="claude", debate_id="d-1"))
        store.add(ModelUsageRecord(agent_name="claude", debate_id="d-2"))
        store.add(ModelUsageRecord(agent_name="gpt", debate_id="d-1"))
        results = store.query(agent_name="claude", debate_id="d-1")
        assert len(results) == 1

    def test_clear(self):
        """Clear removes all records."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord())
        store.clear()
        assert store.count() == 0

    def test_get_all_returns_copy(self):
        """get_all returns a copy, not a reference."""
        store = ModelUsageStore()
        store.add(ModelUsageRecord())
        all_records = store.get_all()
        all_records.clear()
        assert store.count() == 1


# ---------------------------------------------------------------------------
# ModelUsageAggregator
# ---------------------------------------------------------------------------


class TestModelUsageAggregator:
    """Tests for ModelUsageAggregator static methods."""

    @pytest.fixture()
    def sample_records(self) -> list[ModelUsageRecord]:
        """Create a set of sample records for aggregation tests."""
        return [
            ModelUsageRecord(
                agent_name="claude",
                provider="anthropic",
                model="claude-opus-4.6",
                tokens_in=1000,
                tokens_out=500,
                latency_ms=200.0,
                cost_usd=Decimal("0.01"),
                debate_id="d-1",
                operation="proposal",
            ),
            ModelUsageRecord(
                agent_name="gpt",
                provider="openai",
                model="gpt-4o",
                tokens_in=800,
                tokens_out=400,
                latency_ms=150.0,
                cost_usd=Decimal("0.008"),
                debate_id="d-1",
                operation="critique",
            ),
            ModelUsageRecord(
                agent_name="claude",
                provider="anthropic",
                model="claude-opus-4.6",
                tokens_in=600,
                tokens_out=300,
                latency_ms=180.0,
                cost_usd=Decimal("0.007"),
                debate_id="d-2",
                operation="proposal",
            ),
        ]

    def test_by_model(self, sample_records: list[ModelUsageRecord]):
        """by_model groups records by model name."""
        result = ModelUsageAggregator.by_model(sample_records)
        assert "claude-opus-4.6" in result
        assert "gpt-4o" in result
        assert result["claude-opus-4.6"].total_requests == 2
        assert result["gpt-4o"].total_requests == 1

    def test_by_agent(self, sample_records: list[ModelUsageRecord]):
        """by_agent groups records by agent name."""
        result = ModelUsageAggregator.by_agent(sample_records)
        assert "claude" in result
        assert "gpt" in result
        assert result["claude"].total_requests == 2
        assert result["gpt"].total_requests == 1

    def test_by_provider(self, sample_records: list[ModelUsageRecord]):
        """by_provider groups records by provider."""
        result = ModelUsageAggregator.by_provider(sample_records)
        assert "anthropic" in result
        assert "openai" in result

    def test_by_debate(self, sample_records: list[ModelUsageRecord]):
        """by_debate groups records by debate ID."""
        result = ModelUsageAggregator.by_debate(sample_records)
        assert "d-1" in result
        assert "d-2" in result
        assert result["d-1"].total_requests == 2

    def test_by_operation(self, sample_records: list[ModelUsageRecord]):
        """by_operation groups records by operation type."""
        result = ModelUsageAggregator.by_operation(sample_records)
        assert "proposal" in result
        assert "critique" in result
        assert result["proposal"].total_requests == 2

    def test_total(self, sample_records: list[ModelUsageRecord]):
        """total produces a single aggregate across all records."""
        result = ModelUsageAggregator.total(sample_records)
        assert result.total_requests == 3
        assert result.total_tokens_in == 2400
        assert result.total_tokens_out == 1200

    def test_empty_records(self):
        """Aggregation over empty records returns empty results."""
        assert ModelUsageAggregator.by_model([]) == {}
        total = ModelUsageAggregator.total([])
        assert total.total_requests == 0


# ---------------------------------------------------------------------------
# ModelUsagePipeline
# ---------------------------------------------------------------------------


class TestModelUsagePipeline:
    """Tests for ModelUsagePipeline."""

    def test_record_creates_entry(self):
        """record() adds a record to the store and calculates cost."""
        pipeline = ModelUsagePipeline()
        record = pipeline.record(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=1_000_000,
            tokens_out=0,
            latency_ms=500.0,
            debate_id="d-1",
        )
        assert record.cost_usd > Decimal("0")
        assert pipeline.store.count() == 1

    def test_record_from_dict(self):
        """record_from_dict creates and stores a record from a dict."""
        pipeline = ModelUsagePipeline()
        record = pipeline.record_from_dict(
            {
                "agent_name": "gpt",
                "provider": "openai",
                "model": "gpt-4o",
                "tokens_in": 500,
                "tokens_out": 200,
            }
        )
        assert record.agent_name == "gpt"
        assert pipeline.store.count() == 1

    def test_query_passthrough(self):
        """query() delegates to the store."""
        pipeline = ModelUsagePipeline()
        pipeline.record(agent_name="a", debate_id="d-1")
        pipeline.record(agent_name="b", debate_id="d-2")
        results = pipeline.query(debate_id="d-1")
        assert len(results) == 1

    def test_aggregate_by_model(self):
        """aggregate_by_model groups all records."""
        pipeline = ModelUsagePipeline()
        pipeline.record(model="m1", tokens_in=100, tokens_out=50)
        pipeline.record(model="m1", tokens_in=200, tokens_out=100)
        pipeline.record(model="m2", tokens_in=50, tokens_out=25)
        result = pipeline.aggregate_by_model()
        assert result["m1"].total_requests == 2
        assert result["m2"].total_requests == 1

    def test_aggregate_by_agent(self):
        """aggregate_by_agent groups all records."""
        pipeline = ModelUsagePipeline()
        pipeline.record(agent_name="claude")
        pipeline.record(agent_name="claude")
        pipeline.record(agent_name="gpt")
        result = pipeline.aggregate_by_agent()
        assert result["claude"].total_requests == 2

    def test_aggregate_by_provider(self):
        """aggregate_by_provider groups all records."""
        pipeline = ModelUsagePipeline()
        pipeline.record(provider="anthropic")
        pipeline.record(provider="openai")
        result = pipeline.aggregate_by_provider()
        assert len(result) == 2

    def test_aggregate_by_debate(self):
        """aggregate_by_debate groups by debate_id."""
        pipeline = ModelUsagePipeline()
        pipeline.record(debate_id="d-1")
        pipeline.record(debate_id="d-1")
        pipeline.record(debate_id="d-2")
        result = pipeline.aggregate_by_debate()
        assert result["d-1"].total_requests == 2

    def test_aggregate_by_operation(self):
        """aggregate_by_operation groups by operation."""
        pipeline = ModelUsagePipeline()
        pipeline.record(operation="proposal")
        pipeline.record(operation="critique")
        result = pipeline.aggregate_by_operation()
        assert len(result) == 2

    def test_get_total(self):
        """get_total produces a single aggregate."""
        pipeline = ModelUsagePipeline()
        pipeline.record(tokens_in=100, tokens_out=50)
        pipeline.record(tokens_in=200, tokens_out=100)
        total = pipeline.get_total()
        assert total.total_requests == 2
        assert total.total_tokens_in == 300

    def test_get_summary(self):
        """get_summary returns a serializable dict."""
        pipeline = ModelUsagePipeline()
        pipeline.record(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=100,
            tokens_out=50,
        )
        summary = pipeline.get_summary()
        assert summary["record_count"] == 1
        assert "total" in summary
        assert "by_model" in summary
        assert "by_agent" in summary
        assert "by_provider" in summary

    def test_aggregate_with_query_filter(self):
        """Aggregation methods accept query filters."""
        pipeline = ModelUsagePipeline()
        pipeline.record(agent_name="claude", debate_id="d-1")
        pipeline.record(agent_name="gpt", debate_id="d-2")
        result = pipeline.aggregate_by_agent(debate_id="d-1")
        assert len(result) == 1
        assert "claude" in result

    def test_clear(self):
        """clear() empties the store."""
        pipeline = ModelUsagePipeline()
        pipeline.record(agent_name="a")
        pipeline.clear()
        assert pipeline.store.count() == 0

    def test_custom_store(self):
        """Pipeline accepts a custom store."""
        store = ModelUsageStore(max_records=5)
        pipeline = ModelUsagePipeline(store=store)
        assert pipeline.store is store


# ---------------------------------------------------------------------------
# ModelUsageTimer
# ---------------------------------------------------------------------------


class TestModelUsageTimer:
    """Tests for ModelUsageTimer context manager."""

    def test_timer_records_latency(self):
        """Timer captures elapsed time as latency_ms."""
        pipeline = ModelUsagePipeline()
        with ModelUsageTimer(
            pipeline,
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
        ) as timer:
            time.sleep(0.01)  # 10ms minimum
            timer.set_tokens(tokens_in=100, tokens_out=50)

        assert timer.record is not None
        assert timer.record.latency_ms >= 10.0
        assert timer.record.tokens_in == 100
        assert timer.record.tokens_out == 50
        assert pipeline.store.count() == 1

    def test_timer_without_tokens(self):
        """Timer works even without calling set_tokens."""
        pipeline = ModelUsagePipeline()
        with ModelUsageTimer(pipeline, agent_name="test"):
            pass
        assert pipeline.store.count() == 1

    def test_timer_preserves_context(self):
        """Timer passes through debate_id, round, operation, metadata."""
        pipeline = ModelUsagePipeline()
        with ModelUsageTimer(
            pipeline,
            debate_id="d-1",
            round_number=2,
            operation="vote",
            metadata={"key": "val"},
        ) as timer:
            timer.set_tokens(tokens_in=10, tokens_out=5)

        record = timer.record
        assert record is not None
        assert record.debate_id == "d-1"
        assert record.round_number == 2
        assert record.operation == "vote"
        assert record.metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------


class TestSingletonAccess:
    """Tests for module-level singleton management."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_model_usage_pipeline()

    def teardown_method(self):
        """Clean up singleton after each test."""
        reset_model_usage_pipeline()

    def test_get_creates_default(self):
        """get_model_usage_pipeline creates a pipeline on first call."""
        pipeline = get_model_usage_pipeline()
        assert isinstance(pipeline, ModelUsagePipeline)

    def test_get_returns_same_instance(self):
        """Repeated calls return the same instance."""
        p1 = get_model_usage_pipeline()
        p2 = get_model_usage_pipeline()
        assert p1 is p2

    def test_set_replaces_singleton(self):
        """set_model_usage_pipeline replaces the global instance."""
        custom = ModelUsagePipeline(max_records=5)
        set_model_usage_pipeline(custom)
        assert get_model_usage_pipeline() is custom

    def test_reset_clears_singleton(self):
        """reset_model_usage_pipeline clears the singleton."""
        _ = get_model_usage_pipeline()
        reset_model_usage_pipeline()
        # Next call should create a new instance
        p2 = get_model_usage_pipeline()
        assert isinstance(p2, ModelUsagePipeline)


# ---------------------------------------------------------------------------
# Integration: end-to-end debate session
# ---------------------------------------------------------------------------


class TestDebateSessionIntegration:
    """End-to-end test simulating a debate session with model usage tracking."""

    def test_full_debate_session(self):
        """Simulate recording multiple agent calls across a debate."""
        pipeline = ModelUsagePipeline()

        # Round 1: proposals
        pipeline.record(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=2000,
            tokens_out=1000,
            latency_ms=1500.0,
            debate_id="debate-abc",
            round_number=1,
            operation="proposal",
        )
        pipeline.record(
            agent_name="gpt",
            provider="openai",
            model="gpt-4o",
            tokens_in=2000,
            tokens_out=900,
            latency_ms=1200.0,
            debate_id="debate-abc",
            round_number=1,
            operation="proposal",
        )

        # Round 1: critiques
        pipeline.record(
            agent_name="claude",
            provider="anthropic",
            model="claude-opus-4.6",
            tokens_in=3000,
            tokens_out=800,
            latency_ms=1100.0,
            debate_id="debate-abc",
            round_number=1,
            operation="critique",
        )

        # Round 2: revision
        pipeline.record(
            agent_name="gpt",
            provider="openai",
            model="gpt-4o",
            tokens_in=2500,
            tokens_out=1200,
            latency_ms=1400.0,
            debate_id="debate-abc",
            round_number=2,
            operation="revision",
        )

        # Verify totals
        total = pipeline.get_total()
        assert total.total_requests == 4
        assert total.total_tokens_in == 9500
        assert total.total_tokens_out == 3900
        assert total.total_cost_usd > Decimal("0")

        # Verify by-model aggregation
        by_model = pipeline.aggregate_by_model()
        assert by_model["claude-opus-4.6"].total_requests == 2
        assert by_model["gpt-4o"].total_requests == 2

        # Verify by-agent aggregation
        by_agent = pipeline.aggregate_by_agent()
        assert by_agent["claude"].total_tokens_in == 5000
        assert by_agent["gpt"].total_tokens_in == 4500

        # Verify debate-scoped query
        debate_records = pipeline.query(debate_id="debate-abc")
        assert len(debate_records) == 4

        # Verify operation aggregation
        by_op = pipeline.aggregate_by_operation()
        assert by_op["proposal"].total_requests == 2
        assert by_op["critique"].total_requests == 1
        assert by_op["revision"].total_requests == 1

        # Verify summary is serializable
        summary = pipeline.get_summary()
        assert summary["record_count"] == 4
        assert "claude-opus-4.6" in summary["by_model"]

    def test_multi_debate_isolation(self):
        """Records from different debates can be queried independently."""
        pipeline = ModelUsagePipeline()
        pipeline.record(debate_id="d-1", tokens_in=100, tokens_out=50)
        pipeline.record(debate_id="d-1", tokens_in=200, tokens_out=100)
        pipeline.record(debate_id="d-2", tokens_in=300, tokens_out=150)

        by_debate = pipeline.aggregate_by_debate()
        assert by_debate["d-1"].total_requests == 2
        assert by_debate["d-2"].total_requests == 1
        assert by_debate["d-1"].total_tokens_in == 300
        assert by_debate["d-2"].total_tokens_in == 300

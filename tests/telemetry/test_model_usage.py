"""Tests for the model usage telemetry pipeline."""

from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from aragora.telemetry.model_usage import (
    ModelUsageCollector,
    ModelUsageRecord,
    get_model_usage_collector,
    record_model_usage,
    set_model_usage_collector,
)


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------


class TestModelUsageRecord:
    def test_defaults(self):
        r = ModelUsageRecord()
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.total_tokens == 0
        assert r.cost_usd == Decimal("0")
        assert r.record_id  # auto-generated

    def test_total_tokens(self):
        r = ModelUsageRecord(tokens_in=100, tokens_out=50)
        assert r.total_tokens == 150

    def test_immutable(self):
        r = ModelUsageRecord(tokens_in=10)
        with pytest.raises(AttributeError):
            r.tokens_in = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelUsageCollector — basic recording
# ---------------------------------------------------------------------------


class TestModelUsageCollector:
    def test_record_and_retrieve(self):
        c = ModelUsageCollector()
        r = ModelUsageRecord(debate_id="d1", agent_id="a1", tokens_in=100)
        c.record(r)

        records = c.get_records("d1")
        assert len(records) == 1
        assert records[0].tokens_in == 100

    def test_separate_debates(self):
        c = ModelUsageCollector()
        c.record(ModelUsageRecord(debate_id="d1"))
        c.record(ModelUsageRecord(debate_id="d2"))
        c.record(ModelUsageRecord(debate_id="d1"))

        assert len(c.get_records("d1")) == 2
        assert len(c.get_records("d2")) == 1

    def test_get_all_debate_ids(self):
        c = ModelUsageCollector()
        c.record(ModelUsageRecord(debate_id="d1"))
        c.record(ModelUsageRecord(debate_id="d2"))
        ids = c.get_all_debate_ids()
        assert set(ids) == {"d1", "d2"}

    def test_max_records_per_debate(self):
        c = ModelUsageCollector(max_records_per_debate=3)
        for _ in range(5):
            c.record(ModelUsageRecord(debate_id="d1"))
        assert len(c.get_records("d1")) == 3

    def test_clear_debate(self):
        c = ModelUsageCollector()
        c.record(ModelUsageRecord(debate_id="d1"))
        c.record(ModelUsageRecord(debate_id="d1"))
        removed = c.clear_debate("d1")
        assert removed == 2
        assert c.get_records("d1") == []

    def test_clear_all(self):
        c = ModelUsageCollector()
        c.record(ModelUsageRecord(debate_id="d1"))
        c.record(ModelUsageRecord(debate_id="d2"))
        c.clear_all()
        assert c.get_all_debate_ids() == []

    def test_empty_debate_returns_empty(self):
        c = ModelUsageCollector()
        assert c.get_records("nonexistent") == []


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


class TestSummariseDebate:
    def _make_collector(self) -> ModelUsageCollector:
        c = ModelUsageCollector()
        c.record(
            ModelUsageRecord(
                debate_id="d1",
                agent_id="agent-1",
                provider="anthropic",
                model="claude-3-opus",
                model_version="2024-02-29",
                tokens_in=1000,
                tokens_out=500,
                tokens_cached=200,
                latency_ms=350.5,
                cost_usd=Decimal("0.0450"),
                operation="proposal",
                round_number=1,
            )
        )
        c.record(
            ModelUsageRecord(
                debate_id="d1",
                agent_id="agent-2",
                provider="openai",
                model="gpt-4o",
                model_version="2024-05-13",
                tokens_in=800,
                tokens_out=400,
                latency_ms=280.0,
                cost_usd=Decimal("0.0200"),
                operation="critique",
                round_number=1,
            )
        )
        c.record(
            ModelUsageRecord(
                debate_id="d1",
                agent_id="agent-1",
                provider="anthropic",
                model="claude-3-opus",
                model_version="2024-02-29",
                tokens_in=600,
                tokens_out=300,
                latency_ms=200.0,
                cost_usd=Decimal("0.0270"),
                operation="revision",
                round_number=2,
            )
        )
        return c

    def test_totals(self):
        c = self._make_collector()
        s = c.summarise_debate("d1")
        assert s["total_cost_usd"] == "0.0920"
        assert s["total_tokens_in"] == 2400
        assert s["total_tokens_out"] == 1200
        assert s["total_tokens_cached"] == 200
        assert s["call_count"] == 3
        assert s["total_latency_ms"] == pytest.approx(830.5, abs=0.01)

    def test_by_model(self):
        c = self._make_collector()
        s = c.summarise_debate("d1")
        by_model = s["by_model"]
        assert "anthropic/claude-3-opus" in by_model
        assert "openai/gpt-4o" in by_model
        anthropic = by_model["anthropic/claude-3-opus"]
        assert anthropic["call_count"] == 2
        assert anthropic["tokens_in"] == 1600
        assert anthropic["model_version"] == "2024-02-29"

    def test_by_agent(self):
        c = self._make_collector()
        s = c.summarise_debate("d1")
        by_agent = s["by_agent"]
        assert by_agent["agent-1"]["call_count"] == 2
        assert by_agent["agent-2"]["call_count"] == 1

    def test_by_operation(self):
        c = self._make_collector()
        s = c.summarise_debate("d1")
        by_op = s["by_operation"]
        assert "proposal" in by_op
        assert "critique" in by_op
        assert "revision" in by_op
        assert by_op["proposal"]["call_count"] == 1

    def test_empty_debate_summary(self):
        c = ModelUsageCollector()
        s = c.summarise_debate("nonexistent")
        assert s["call_count"] == 0
        assert s["total_cost_usd"] == "0"
        assert s["by_model"] == {}

    def test_cost_serialised_as_string(self):
        """Ensure Decimal costs are serialised as strings for JSON compat."""
        c = self._make_collector()
        s = c.summarise_debate("d1")
        for model_data in s["by_model"].values():
            assert isinstance(model_data["cost_usd"], str)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_recording(self):
        c = ModelUsageCollector()
        n_threads = 8
        n_per_thread = 200
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(n_per_thread):
                c.record(
                    ModelUsageRecord(
                        debate_id="d1",
                        agent_id=f"agent-{tid}",
                        tokens_in=i,
                    )
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(c.get_records("d1")) == n_threads * n_per_thread


# ---------------------------------------------------------------------------
# Singleton / convenience function
# ---------------------------------------------------------------------------


class TestSingletonAndConvenience:
    def test_get_returns_same_instance(self):
        c1 = get_model_usage_collector()
        c2 = get_model_usage_collector()
        assert c1 is c2

    def test_set_replaces(self):
        original = get_model_usage_collector()
        custom = ModelUsageCollector()
        set_model_usage_collector(custom)
        assert get_model_usage_collector() is custom
        # restore
        set_model_usage_collector(original)

    def test_record_model_usage_convenience(self):
        custom = ModelUsageCollector()
        set_model_usage_collector(custom)
        try:
            r = record_model_usage(
                debate_id="conv-test",
                agent_id="a1",
                provider="anthropic",
                model="claude-3-haiku",
                tokens_in=50,
                tokens_out=25,
                cost_usd="0.001",
                operation="vote",
            )
            assert r.debate_id == "conv-test"
            assert r.cost_usd == Decimal("0.001")
            records = custom.get_records("conv-test")
            assert len(records) == 1
        finally:
            set_model_usage_collector(get_model_usage_collector())

    def test_record_model_usage_returns_record(self):
        custom = ModelUsageCollector()
        set_model_usage_collector(custom)
        try:
            r = record_model_usage(
                debate_id="ret-test",
                tokens_in=10,
                cost_usd=0.5,
            )
            assert isinstance(r, ModelUsageRecord)
            assert r.cost_usd == Decimal("0.5")
        finally:
            set_model_usage_collector(get_model_usage_collector())

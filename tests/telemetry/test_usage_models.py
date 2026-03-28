"""Tests for aragora.telemetry.models — telemetry data models."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from aragora.telemetry.models import (
    AggregationInterval,
    ModelUsageAggregate,
    ModelUsageRecord,
    SessionUsageSummary,
    UsageOperation,
    UsageQuery,
)


# ---------------------------------------------------------------------------
# ModelUsageRecord
# ---------------------------------------------------------------------------


class TestModelUsageRecord:
    def test_defaults(self):
        rec = ModelUsageRecord()
        assert rec.id  # UUID generated
        assert rec.tokens_total == 0
        assert rec.cost_usd == Decimal("0")
        assert rec.operation == UsageOperation.OTHER
        assert rec.metadata == {}

    def test_tokens_total_auto_calculated(self):
        rec = ModelUsageRecord(tokens_input=100, tokens_output=50)
        assert rec.tokens_total == 150

    def test_tokens_total_explicit(self):
        rec = ModelUsageRecord(tokens_input=100, tokens_output=50, tokens_total=200)
        assert rec.tokens_total == 200  # explicit takes precedence

    def test_to_dict_roundtrip(self):
        rec = ModelUsageRecord(
            debate_id="d-1",
            session_id="s-1",
            workspace_id="ws-1",
            agent_id="a-1",
            agent_name="claude",
            provider="anthropic",
            model="claude-3-opus",
            model_version="2024-02-15",
            tokens_input=1000,
            tokens_output=500,
            tokens_cached=200,
            cost_usd=Decimal("0.0450"),
            latency_ms=1234.5,
            time_to_first_token_ms=120.0,
            operation=UsageOperation.DEBATE_ROUND,
            round_number=2,
            metadata={"temperature": 0.7},
        )
        d = rec.to_dict()
        restored = ModelUsageRecord.from_dict(d)

        assert restored.debate_id == "d-1"
        assert restored.provider == "anthropic"
        assert restored.model == "claude-3-opus"
        assert restored.model_version == "2024-02-15"
        assert restored.tokens_input == 1000
        assert restored.tokens_output == 500
        assert restored.cost_usd == Decimal("0.0450")
        assert restored.latency_ms == 1234.5
        assert restored.operation == UsageOperation.DEBATE_ROUND
        assert restored.round_number == 2
        assert restored.metadata == {"temperature": 0.7}

    def test_from_dict_unknown_operation(self):
        rec = ModelUsageRecord.from_dict({"operation": "nonexistent_op"})
        assert rec.operation == UsageOperation.OTHER

    def test_from_dict_missing_fields(self):
        rec = ModelUsageRecord.from_dict({})
        assert rec.provider == ""
        assert rec.tokens_input == 0
        assert rec.cost_usd == Decimal("0")

    def test_from_row(self):
        row = (
            "id-1",  # id
            "d-1",  # debate_id
            "s-1",  # session_id
            "ws-1",  # workspace_id
            "a-1",  # agent_id
            "claude",  # agent_name
            "anthropic",  # provider
            "opus",  # model
            "v1",  # model_version
            500,  # tokens_input
            200,  # tokens_output
            100,  # tokens_cached
            700,  # tokens_total
            800.0,  # latency_ms
            50.0,  # time_to_first_token_ms
            "debate_round",  # operation
            0.035,  # cost_usd
            "2026-03-27T10:00:00+00:00",  # timestamp
            1,  # round_number
        )
        rec = ModelUsageRecord.from_row(row)
        assert rec.id == "id-1"
        assert rec.debate_id == "d-1"
        assert rec.tokens_total == 700
        assert rec.operation == UsageOperation.DEBATE_ROUND


# ---------------------------------------------------------------------------
# UsageQuery
# ---------------------------------------------------------------------------


class TestUsageQuery:
    def test_empty_query(self):
        q = UsageQuery()
        where, params = q.to_where_clause()
        assert where == "1=1"
        assert params == []

    def test_single_filter(self):
        q = UsageQuery(debate_id="d-1")
        where, params = q.to_where_clause()
        assert "debate_id = ?" in where
        assert params == ["d-1"]

    def test_multiple_filters(self):
        q = UsageQuery(
            workspace_id="ws-1",
            provider="anthropic",
            operation=UsageOperation.PROPOSAL,
        )
        where, params = q.to_where_clause()
        assert "workspace_id = ?" in where
        assert "provider = ?" in where
        assert "operation = ?" in where
        assert "ws-1" in params
        assert "anthropic" in params
        assert "proposal" in params

    def test_time_range(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, tzinfo=timezone.utc)
        q = UsageQuery(start_time=start, end_time=end)
        where, params = q.to_where_clause()
        assert "timestamp >= ?" in where
        assert "timestamp <= ?" in where
        assert len(params) == 2


# ---------------------------------------------------------------------------
# SessionUsageSummary
# ---------------------------------------------------------------------------


class TestSessionUsageSummary:
    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        s = SessionUsageSummary(
            debate_id="d-1",
            total_requests=10,
            total_tokens=5000,
            total_cost_usd=Decimal("1.23"),
            first_request_at=now,
            last_request_at=now,
        )
        d = s.to_dict()
        assert d["debate_id"] == "d-1"
        assert d["total_requests"] == 10
        assert d["total_cost_usd"] == "1.23"
        assert d["first_request_at"] is not None


# ---------------------------------------------------------------------------
# ModelUsageAggregate
# ---------------------------------------------------------------------------


class TestModelUsageAggregate:
    def test_to_dict(self):
        agg = ModelUsageAggregate(
            group_key="model",
            group_value="claude-3-opus",
            total_requests=50,
            total_tokens=25000,
            total_cost_usd=Decimal("5.00"),
            avg_latency_ms=400.0,
        )
        d = agg.to_dict()
        assert d["group_key"] == "model"
        assert d["group_value"] == "claude-3-opus"
        assert d["total_cost_usd"] == "5.00"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_usage_operation_values(self):
        assert UsageOperation.DEBATE_ROUND.value == "debate_round"
        assert UsageOperation.EMBEDDING.value == "embedding"

    def test_aggregation_interval_values(self):
        assert AggregationInterval.HOURLY.value == "hourly"
        assert AggregationInterval.MONTHLY.value == "monthly"

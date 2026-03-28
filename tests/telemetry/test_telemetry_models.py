"""Tests for telemetry data models."""

from datetime import datetime, timezone
from decimal import Decimal

from aragora.telemetry.models import (
    DebateUsageSummary,
    ModelUsageRecord,
    RequestStatus,
    TokenCount,
    TokenDirection,
)


class TestTokenCount:
    def test_defaults(self):
        tc = TokenCount()
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0
        assert tc.cached_tokens == 0
        assert tc.total == 0

    def test_total(self):
        tc = TokenCount(input_tokens=100, output_tokens=50, cached_tokens=20)
        assert tc.total == 170

    def test_immutable(self):
        tc = TokenCount(input_tokens=10)
        try:
            tc.input_tokens = 20  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_round_trip(self):
        tc = TokenCount(input_tokens=100, output_tokens=50, cached_tokens=20)
        d = tc.to_dict()
        tc2 = TokenCount.from_dict(d)
        assert tc == tc2

    def test_from_dict_defaults(self):
        tc = TokenCount.from_dict({})
        assert tc.total == 0


class TestModelUsageRecord:
    def _make(self, **kwargs):
        defaults = {
            "debate_id": "d-1",
            "workspace_id": "ws-1",
            "agent_id": "a-1",
            "agent_name": "claude",
            "provider": "anthropic",
            "model": "claude-sonnet-4.6",
            "tokens": TokenCount(input_tokens=500, output_tokens=200),
            "latency_ms": 1234.5,
            "cost_usd": Decimal("0.0085"),
        }
        defaults.update(kwargs)
        return ModelUsageRecord(**defaults)

    def test_defaults(self):
        r = ModelUsageRecord()
        assert r.id  # auto-generated
        assert r.status == RequestStatus.SUCCESS
        assert r.cost_usd == Decimal("0")
        assert r.tokens.total == 0

    def test_fingerprint_deterministic(self):
        r = self._make()
        assert r.fingerprint == r.fingerprint

    def test_fingerprint_varies(self):
        r1 = self._make(agent_id="a-1")
        r2 = self._make(agent_id="a-2")
        assert r1.fingerprint != r2.fingerprint

    def test_round_trip(self):
        r = self._make(
            round_number=2,
            model_version="2024-03-01",
            error_message=None,
            metadata={"key": "val"},
        )
        d = r.to_dict()
        r2 = ModelUsageRecord.from_dict(d)
        assert r2.debate_id == r.debate_id
        assert r2.tokens.input_tokens == 500
        assert r2.tokens.output_tokens == 200
        assert r2.cost_usd == Decimal("0.0085")
        assert r2.model_version == "2024-03-01"
        assert r2.metadata == {"key": "val"}
        assert r2.status == RequestStatus.SUCCESS

    def test_from_dict_with_string_metadata(self):
        d = {
            "metadata": '{"a": 1}',
            "status": "error",
        }
        r = ModelUsageRecord.from_dict(d)
        assert r.metadata == {"a": 1}
        assert r.status == RequestStatus.ERROR

    def test_from_dict_missing_timestamp(self):
        r = ModelUsageRecord.from_dict({})
        assert isinstance(r.timestamp, datetime)


class TestDebateUsageSummary:
    def test_to_dict(self):
        s = DebateUsageSummary(
            debate_id="d-1",
            total_requests=10,
            total_cost_usd=Decimal("0.50"),
            cost_by_provider={"anthropic": Decimal("0.50")},
        )
        d = s.to_dict()
        assert d["debate_id"] == "d-1"
        assert d["total_requests"] == 10
        assert d["total_cost_usd"] == "0.50"
        assert d["cost_by_provider"]["anthropic"] == "0.50"

    def test_empty_summary(self):
        s = DebateUsageSummary(debate_id="d-empty")
        d = s.to_dict()
        assert d["total_requests"] == 0
        assert d["total_cost_usd"] == "0"


class TestRequestStatus:
    def test_values(self):
        assert RequestStatus.SUCCESS.value == "success"
        assert RequestStatus.RATE_LIMITED.value == "rate_limited"


class TestTokenDirection:
    def test_values(self):
        assert TokenDirection.INPUT.value == "input"
        assert TokenDirection.CACHED.value == "cached"

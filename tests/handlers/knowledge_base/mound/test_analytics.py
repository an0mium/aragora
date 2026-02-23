"""Tests for AnalyticsOperationsMixin.

Covers all six async endpoints on the mixin:
- GET  /api/knowledge/mound/analytics/coverage        - analyze_coverage
- GET  /api/knowledge/mound/analytics/usage            - analyze_usage
- POST /api/knowledge/mound/analytics/usage/record     - record_usage_event
- POST /api/knowledge/mound/analytics/quality/snapshot - capture_quality_snapshot
- GET  /api/knowledge/mound/analytics/quality/trend    - get_quality_trend
- GET  /api/knowledge/mound/analytics/stats            - get_analytics_stats

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Invalid parameter values (400)
- Internal errors from mound operations (500)
- Edge cases and response structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.analytics import UsageEventType
from aragora.server.handlers.knowledge_base.mound.analytics import (
    AnalyticsOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock dataclasses for mound return values
# ---------------------------------------------------------------------------


@dataclass
class MockCoverageReport:
    """Mock coverage report returned by mound.analyze_coverage."""

    workspace_id: str = "ws-001"
    total_items: int = 200
    domains_covered: int = 5
    stale_items: int = 15
    coverage_score: float = 0.82
    domain_breakdown: dict[str, Any] = field(
        default_factory=lambda: {
            "engineering": {"items": 80, "stale": 3, "coverage": 0.95},
            "marketing": {"items": 40, "stale": 5, "coverage": 0.75},
            "legal": {"items": 30, "stale": 2, "coverage": 0.85},
            "hr": {"items": 25, "stale": 3, "coverage": 0.70},
            "finance": {"items": 25, "stale": 2, "coverage": 0.80},
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_items": self.total_items,
            "domains_covered": self.domains_covered,
            "stale_items": self.stale_items,
            "coverage_score": self.coverage_score,
            "domain_breakdown": self.domain_breakdown,
        }


@dataclass
class MockUsageReport:
    """Mock usage report returned by mound.analyze_usage."""

    workspace_id: str = "ws-001"
    days: int = 30
    total_events: int = 1500
    unique_users: int = 25
    top_queries: list[str] = field(
        default_factory=lambda: ["rate limiter", "auth flow", "deployment"]
    )
    events_by_type: dict[str, int] = field(
        default_factory=lambda: {
            "query": 800,
            "view": 400,
            "cite": 150,
            "share": 100,
            "export": 50,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "days": self.days,
            "total_events": self.total_events,
            "unique_users": self.unique_users,
            "top_queries": self.top_queries,
            "events_by_type": self.events_by_type,
        }


@dataclass
class MockUsageEvent:
    """Mock usage event returned by mound.record_usage_event."""

    id: str = "evt-001"
    event_type: UsageEventType = UsageEventType.QUERY
    item_id: str | None = "item-001"
    user_id: str | None = "user-001"
    workspace_id: str | None = "ws-001"
    query: str | None = "search query"
    timestamp: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockQualitySnapshot:
    """Mock quality snapshot returned by mound.capture_quality_snapshot."""

    workspace_id: str = "ws-001"
    total_items: int = 200
    average_confidence: float = 0.78
    high_quality_count: int = 120
    low_quality_count: int = 20
    stale_count: int = 15
    captured_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_items": self.total_items,
            "average_confidence": self.average_confidence,
            "high_quality_count": self.high_quality_count,
            "low_quality_count": self.low_quality_count,
            "stale_count": self.stale_count,
            "captured_at": self.captured_at.isoformat(),
        }


@dataclass
class MockQualityTrend:
    """Mock quality trend returned by mound.get_quality_trend."""

    workspace_id: str = "ws-001"
    days: int = 30
    snapshots: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"date": "2026-01-15", "average_confidence": 0.75, "total_items": 180},
            {"date": "2026-01-20", "average_confidence": 0.77, "total_items": 190},
            {"date": "2026-02-01", "average_confidence": 0.78, "total_items": 200},
        ]
    )
    trend_direction: str = "improving"
    confidence_change: float = 0.03

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "days": self.days,
            "snapshots": self.snapshots,
            "trend_direction": self.trend_direction,
            "confidence_change": self.confidence_change,
        }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class AnalyticsTestHandler(AnalyticsOperationsMixin):
    """Concrete handler for testing the analytics mixin."""

    def __init__(self, mound=None):
        self._mound = mound
        self.ctx: dict[str, Any] = {}

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with analytics methods."""
    mound = MagicMock()
    mound.analyze_coverage = AsyncMock(return_value=MockCoverageReport())
    mound.analyze_usage = AsyncMock(return_value=MockUsageReport())
    mound.record_usage_event = AsyncMock(return_value=MockUsageEvent())
    mound.capture_quality_snapshot = AsyncMock(return_value=MockQualitySnapshot())
    mound.get_quality_trend = AsyncMock(return_value=MockQualityTrend())
    mound.get_analytics_stats = MagicMock(
        return_value={
            "total_events": 5000,
            "total_snapshots": 30,
            "coverage_reports": 10,
            "usage_reports": 20,
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create an AnalyticsTestHandler with a mocked mound."""
    return AnalyticsTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create an AnalyticsTestHandler with no mound (None)."""
    return AnalyticsTestHandler(mound=None)


# ============================================================================
# Tests: analyze_coverage
# ============================================================================


class TestAnalyzeCoverage:
    """Test analyze_coverage (GET /api/knowledge/mound/analytics/coverage)."""

    @pytest.mark.asyncio
    async def test_success_returns_coverage_report(self, handler, mock_mound):
        """Successful coverage analysis returns report data from mound."""
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"
        assert body["total_items"] == 200
        assert body["domains_covered"] == 5
        assert body["stale_items"] == 15
        assert body["coverage_score"] == 0.82

    @pytest.mark.asyncio
    async def test_success_forwards_workspace_id(self, handler, mock_mound):
        """workspace_id is forwarded to mound.analyze_coverage."""
        await handler.analyze_coverage(workspace_id="ws-test-123")
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-test-123",
            stale_threshold_days=90,
        )

    @pytest.mark.asyncio
    async def test_success_with_custom_stale_threshold(self, handler, mock_mound):
        """Custom stale_threshold_days is forwarded to mound."""
        await handler.analyze_coverage(
            workspace_id="ws-001", stale_threshold_days=30
        )
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-001",
            stale_threshold_days=30,
        )

    @pytest.mark.asyncio
    async def test_default_stale_threshold_is_90(self, handler, mock_mound):
        """Default stale_threshold_days is 90."""
        await handler.analyze_coverage(workspace_id="ws-001")
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-001",
            stale_threshold_days=90,
        )

    @pytest.mark.asyncio
    async def test_domain_breakdown_in_response(self, handler, mock_mound):
        """Domain breakdown is included in response."""
        result = await handler.analyze_coverage(workspace_id="ws-001")
        body = _body(result)
        assert "domain_breakdown" in body
        assert "engineering" in body["domain_breakdown"]
        assert body["domain_breakdown"]["engineering"]["items"] == 80

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.analyze_coverage(workspace_id="")
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=KeyError("missing key")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=ValueError("bad value")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=OSError("disk failure")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=TypeError("wrong type")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=AttributeError("no such attr")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_error_body_contains_sanitized_message(self, handler, mock_mound):
        """Error response contains a sanitized error message, not raw exception text."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=ValueError("internal secret details")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        body = _body(result)
        assert "error" in body
        # safe_error_message should not leak the raw exception text
        assert "internal secret details" not in body["error"]

    @pytest.mark.asyncio
    async def test_custom_stale_threshold_zero(self, handler, mock_mound):
        """stale_threshold_days=0 is forwarded."""
        await handler.analyze_coverage(
            workspace_id="ws-001", stale_threshold_days=0
        )
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-001",
            stale_threshold_days=0,
        )

    @pytest.mark.asyncio
    async def test_large_stale_threshold(self, handler, mock_mound):
        """Large stale_threshold_days is forwarded."""
        await handler.analyze_coverage(
            workspace_id="ws-001", stale_threshold_days=365
        )
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-001",
            stale_threshold_days=365,
        )

    @pytest.mark.asyncio
    async def test_empty_coverage_report(self, handler, mock_mound):
        """Coverage report with zero items is handled."""
        report = MockCoverageReport(
            total_items=0,
            domains_covered=0,
            stale_items=0,
            coverage_score=0.0,
            domain_breakdown={},
        )
        mock_mound.analyze_coverage = AsyncMock(return_value=report)
        result = await handler.analyze_coverage(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["total_items"] == 0
        assert body["domain_breakdown"] == {}


# ============================================================================
# Tests: analyze_usage
# ============================================================================


class TestAnalyzeUsage:
    """Test analyze_usage (GET /api/knowledge/mound/analytics/usage)."""

    @pytest.mark.asyncio
    async def test_success_returns_usage_report(self, handler, mock_mound):
        """Successful usage analysis returns report data."""
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"
        assert body["total_events"] == 1500
        assert body["unique_users"] == 25

    @pytest.mark.asyncio
    async def test_success_forwards_workspace_id(self, handler, mock_mound):
        """workspace_id is forwarded to mound.analyze_usage."""
        await handler.analyze_usage(workspace_id="ws-usage-test")
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-usage-test",
            days=30,
        )

    @pytest.mark.asyncio
    async def test_default_days_is_30(self, handler, mock_mound):
        """Default days parameter is 30."""
        await handler.analyze_usage(workspace_id="ws-001")
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-001",
            days=30,
        )

    @pytest.mark.asyncio
    async def test_custom_days_parameter(self, handler, mock_mound):
        """Custom days parameter is forwarded."""
        await handler.analyze_usage(workspace_id="ws-001", days=7)
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-001",
            days=7,
        )

    @pytest.mark.asyncio
    async def test_top_queries_in_response(self, handler, mock_mound):
        """Top queries are included in response."""
        result = await handler.analyze_usage(workspace_id="ws-001")
        body = _body(result)
        assert "top_queries" in body
        assert "rate limiter" in body["top_queries"]

    @pytest.mark.asyncio
    async def test_events_by_type_in_response(self, handler, mock_mound):
        """Events by type breakdown is included."""
        result = await handler.analyze_usage(workspace_id="ws-001")
        body = _body(result)
        assert "events_by_type" in body
        assert body["events_by_type"]["query"] == 800

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.analyze_usage(workspace_id="")
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=KeyError("missing")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=ValueError("bad")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=OSError("disk")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=TypeError("wrong")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=AttributeError("missing attr")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_days_zero(self, handler, mock_mound):
        """days=0 is forwarded to mound."""
        await handler.analyze_usage(workspace_id="ws-001", days=0)
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-001", days=0,
        )

    @pytest.mark.asyncio
    async def test_large_days_value(self, handler, mock_mound):
        """Large days value is forwarded."""
        await handler.analyze_usage(workspace_id="ws-001", days=365)
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-001", days=365,
        )

    @pytest.mark.asyncio
    async def test_empty_usage_report(self, handler, mock_mound):
        """Usage report with zero events is handled."""
        report = MockUsageReport(
            total_events=0,
            unique_users=0,
            top_queries=[],
            events_by_type={},
        )
        mock_mound.analyze_usage = AsyncMock(return_value=report)
        result = await handler.analyze_usage(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["total_events"] == 0
        assert body["top_queries"] == []


# ============================================================================
# Tests: record_usage_event
# ============================================================================


class TestRecordUsageEvent:
    """Test record_usage_event (POST /api/knowledge/mound/analytics/usage/record)."""

    @pytest.mark.asyncio
    async def test_success_returns_event_data(self, handler, mock_mound):
        """Successful recording returns event details."""
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["event_id"] == "evt-001"
        assert body["event_type"] == "query"
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_success_with_all_params(self, handler, mock_mound):
        """All optional parameters are forwarded to mound."""
        await handler.record_usage_event(
            event_type="view",
            item_id="item-abc",
            user_id="user-xyz",
            workspace_id="ws-123",
            query="test query",
        )
        mock_mound.record_usage_event.assert_awaited_once_with(
            event_type=UsageEventType.VIEW,
            item_id="item-abc",
            user_id="user-xyz",
            workspace_id="ws-123",
            query="test query",
        )

    @pytest.mark.asyncio
    async def test_success_query_event_type(self, handler, mock_mound):
        """event_type 'query' is converted to UsageEventType.QUERY."""
        await handler.record_usage_event(event_type="query")
        call_kwargs = mock_mound.record_usage_event.call_args.kwargs
        assert call_kwargs["event_type"] == UsageEventType.QUERY

    @pytest.mark.asyncio
    async def test_success_view_event_type(self, handler, mock_mound):
        """event_type 'view' is converted to UsageEventType.VIEW."""
        event = MockUsageEvent(event_type=UsageEventType.VIEW)
        mock_mound.record_usage_event = AsyncMock(return_value=event)
        result = await handler.record_usage_event(event_type="view")
        assert _status(result) == 200
        body = _body(result)
        assert body["event_type"] == "view"

    @pytest.mark.asyncio
    async def test_success_cite_event_type(self, handler, mock_mound):
        """event_type 'cite' is converted to UsageEventType.CITE."""
        event = MockUsageEvent(event_type=UsageEventType.CITE)
        mock_mound.record_usage_event = AsyncMock(return_value=event)
        result = await handler.record_usage_event(event_type="cite")
        assert _status(result) == 200
        body = _body(result)
        assert body["event_type"] == "cite"

    @pytest.mark.asyncio
    async def test_success_share_event_type(self, handler, mock_mound):
        """event_type 'share' is converted to UsageEventType.SHARE."""
        event = MockUsageEvent(event_type=UsageEventType.SHARE)
        mock_mound.record_usage_event = AsyncMock(return_value=event)
        result = await handler.record_usage_event(event_type="share")
        assert _status(result) == 200
        body = _body(result)
        assert body["event_type"] == "share"

    @pytest.mark.asyncio
    async def test_success_export_event_type(self, handler, mock_mound):
        """event_type 'export' is converted to UsageEventType.EXPORT."""
        event = MockUsageEvent(event_type=UsageEventType.EXPORT)
        mock_mound.record_usage_event = AsyncMock(return_value=event)
        result = await handler.record_usage_event(event_type="export")
        assert _status(result) == 200
        body = _body(result)
        assert body["event_type"] == "export"

    @pytest.mark.asyncio
    async def test_success_with_minimal_params(self, handler, mock_mound):
        """Only event_type is required; optional params default to None."""
        await handler.record_usage_event(event_type="query")
        mock_mound.record_usage_event.assert_awaited_once_with(
            event_type=UsageEventType.QUERY,
            item_id=None,
            user_id=None,
            workspace_id=None,
            query=None,
        )

    @pytest.mark.asyncio
    async def test_success_timestamp_is_iso_format(self, handler, mock_mound):
        """Timestamp in response is ISO format."""
        result = await handler.record_usage_event(event_type="query")
        body = _body(result)
        # Should parse without error
        datetime.fromisoformat(body["timestamp"])

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.record_usage_event(event_type="query")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_event_type_returns_400(self, handler):
        """Empty event_type returns 400."""
        result = await handler.record_usage_event(event_type="")
        assert _status(result) == 400
        body = _body(result)
        assert "event_type" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_event_type_returns_400(self, handler):
        """Invalid event_type returns 400 with valid types listed."""
        result = await handler.record_usage_event(event_type="invalid_type")
        assert _status(result) == 400
        body = _body(result)
        assert "invalid event_type" in body["error"].lower()
        # Should list valid types
        assert "query" in body["error"].lower()
        assert "view" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_event_type_lists_all_valid_types(self, handler):
        """Invalid event_type error message lists all valid UsageEventType values."""
        result = await handler.record_usage_event(event_type="bogus")
        body = _body(result)
        for evt in UsageEventType:
            assert evt.value in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.record_usage_event = AsyncMock(
            side_effect=KeyError("missing")
        )
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.record_usage_event = AsyncMock(
            side_effect=ValueError("bad")
        )
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.record_usage_event = AsyncMock(
            side_effect=OSError("disk")
        )
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.record_usage_event = AsyncMock(
            side_effect=TypeError("wrong")
        )
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.record_usage_event = AsyncMock(
            side_effect=AttributeError("no attr")
        )
        result = await handler.record_usage_event(event_type="query")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_with_only_item_id(self, handler, mock_mound):
        """item_id alone is accepted."""
        await handler.record_usage_event(
            event_type="view", item_id="item-only"
        )
        call_kwargs = mock_mound.record_usage_event.call_args.kwargs
        assert call_kwargs["item_id"] == "item-only"
        assert call_kwargs["user_id"] is None

    @pytest.mark.asyncio
    async def test_with_only_user_id(self, handler, mock_mound):
        """user_id alone is accepted."""
        await handler.record_usage_event(
            event_type="query", user_id="user-only"
        )
        call_kwargs = mock_mound.record_usage_event.call_args.kwargs
        assert call_kwargs["user_id"] == "user-only"
        assert call_kwargs["item_id"] is None

    @pytest.mark.asyncio
    async def test_with_query_string(self, handler, mock_mound):
        """query string is forwarded."""
        await handler.record_usage_event(
            event_type="query", query="how to deploy"
        )
        call_kwargs = mock_mound.record_usage_event.call_args.kwargs
        assert call_kwargs["query"] == "how to deploy"


# ============================================================================
# Tests: capture_quality_snapshot
# ============================================================================


class TestCaptureQualitySnapshot:
    """Test capture_quality_snapshot (POST /api/knowledge/mound/analytics/quality/snapshot)."""

    @pytest.mark.asyncio
    async def test_success_returns_snapshot(self, handler, mock_mound):
        """Successful snapshot returns quality data."""
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"
        assert body["total_items"] == 200
        assert body["average_confidence"] == 0.78
        assert body["high_quality_count"] == 120

    @pytest.mark.asyncio
    async def test_success_forwards_workspace_id(self, handler, mock_mound):
        """workspace_id is forwarded to mound.capture_quality_snapshot."""
        await handler.capture_quality_snapshot(workspace_id="ws-snap-test")
        mock_mound.capture_quality_snapshot.assert_awaited_once_with(
            workspace_id="ws-snap-test",
        )

    @pytest.mark.asyncio
    async def test_low_quality_count_in_response(self, handler, mock_mound):
        """low_quality_count is included in response."""
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        body = _body(result)
        assert body["low_quality_count"] == 20

    @pytest.mark.asyncio
    async def test_stale_count_in_response(self, handler, mock_mound):
        """stale_count is included in response."""
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        body = _body(result)
        assert body["stale_count"] == 15

    @pytest.mark.asyncio
    async def test_captured_at_in_response(self, handler, mock_mound):
        """captured_at timestamp is included in response."""
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        body = _body(result)
        assert "captured_at" in body
        datetime.fromisoformat(body["captured_at"])

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.capture_quality_snapshot(
            workspace_id="ws-001"
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.capture_quality_snapshot(workspace_id="")
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.capture_quality_snapshot = AsyncMock(
            side_effect=KeyError("missing")
        )
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.capture_quality_snapshot = AsyncMock(
            side_effect=ValueError("bad")
        )
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.capture_quality_snapshot = AsyncMock(
            side_effect=OSError("disk")
        )
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.capture_quality_snapshot = AsyncMock(
            side_effect=TypeError("wrong")
        )
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.capture_quality_snapshot = AsyncMock(
            side_effect=AttributeError("attr")
        )
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_empty_snapshot(self, handler, mock_mound):
        """Snapshot with zero items is handled."""
        snapshot = MockQualitySnapshot(
            total_items=0,
            average_confidence=0.0,
            high_quality_count=0,
            low_quality_count=0,
            stale_count=0,
        )
        mock_mound.capture_quality_snapshot = AsyncMock(return_value=snapshot)
        result = await handler.capture_quality_snapshot(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["total_items"] == 0
        assert body["average_confidence"] == 0.0


# ============================================================================
# Tests: get_quality_trend
# ============================================================================


class TestGetQualityTrend:
    """Test get_quality_trend (GET /api/knowledge/mound/analytics/quality/trend)."""

    @pytest.mark.asyncio
    async def test_success_returns_trend_data(self, handler, mock_mound):
        """Successful trend query returns trend data."""
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"
        assert body["trend_direction"] == "improving"
        assert body["confidence_change"] == 0.03

    @pytest.mark.asyncio
    async def test_success_forwards_workspace_id(self, handler, mock_mound):
        """workspace_id is forwarded to mound.get_quality_trend."""
        await handler.get_quality_trend(workspace_id="ws-trend-test")
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-trend-test",
            days=30,
        )

    @pytest.mark.asyncio
    async def test_default_days_is_30(self, handler, mock_mound):
        """Default days parameter is 30."""
        await handler.get_quality_trend(workspace_id="ws-001")
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-001",
            days=30,
        )

    @pytest.mark.asyncio
    async def test_custom_days_parameter(self, handler, mock_mound):
        """Custom days parameter is forwarded."""
        await handler.get_quality_trend(workspace_id="ws-001", days=14)
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-001",
            days=14,
        )

    @pytest.mark.asyncio
    async def test_snapshots_in_response(self, handler, mock_mound):
        """Snapshots list is included in response."""
        result = await handler.get_quality_trend(workspace_id="ws-001")
        body = _body(result)
        assert "snapshots" in body
        assert len(body["snapshots"]) == 3
        assert body["snapshots"][0]["date"] == "2026-01-15"

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.get_quality_trend(workspace_id="")
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_quality_trend = AsyncMock(
            side_effect=KeyError("missing")
        )
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_quality_trend = AsyncMock(
            side_effect=ValueError("bad")
        )
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_quality_trend = AsyncMock(
            side_effect=OSError("disk")
        )
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_quality_trend = AsyncMock(
            side_effect=TypeError("wrong")
        )
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_quality_trend = AsyncMock(
            side_effect=AttributeError("attr")
        )
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_days_zero(self, handler, mock_mound):
        """days=0 is forwarded to mound."""
        await handler.get_quality_trend(workspace_id="ws-001", days=0)
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-001", days=0,
        )

    @pytest.mark.asyncio
    async def test_days_large_value(self, handler, mock_mound):
        """Large days value is forwarded."""
        await handler.get_quality_trend(workspace_id="ws-001", days=365)
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-001", days=365,
        )

    @pytest.mark.asyncio
    async def test_empty_snapshots(self, handler, mock_mound):
        """Trend with no snapshots is handled."""
        trend = MockQualityTrend(
            snapshots=[],
            trend_direction="stable",
            confidence_change=0.0,
        )
        mock_mound.get_quality_trend = AsyncMock(return_value=trend)
        result = await handler.get_quality_trend(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["snapshots"] == []
        assert body["trend_direction"] == "stable"

    @pytest.mark.asyncio
    async def test_declining_trend(self, handler, mock_mound):
        """Declining trend data is returned correctly."""
        trend = MockQualityTrend(
            trend_direction="declining",
            confidence_change=-0.05,
            snapshots=[
                {"date": "2026-01-15", "average_confidence": 0.85, "total_items": 200},
                {"date": "2026-02-01", "average_confidence": 0.80, "total_items": 195},
            ],
        )
        mock_mound.get_quality_trend = AsyncMock(return_value=trend)
        result = await handler.get_quality_trend(workspace_id="ws-001")
        body = _body(result)
        assert body["trend_direction"] == "declining"
        assert body["confidence_change"] == -0.05


# ============================================================================
# Tests: get_analytics_stats
# ============================================================================


class TestGetAnalyticsStats:
    """Test get_analytics_stats (GET /api/knowledge/mound/analytics/stats)."""

    @pytest.mark.asyncio
    async def test_success_returns_stats(self, handler, mock_mound):
        """Successful stats query returns analytics statistics."""
        result = await handler.get_analytics_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_events"] == 5000
        assert body["total_snapshots"] == 30
        assert body["coverage_reports"] == 10
        assert body["usage_reports"] == 20

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_analytics_stats()
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_analytics_stats = MagicMock(
            side_effect=KeyError("missing")
        )
        result = await handler.get_analytics_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_analytics_stats = MagicMock(
            side_effect=ValueError("bad")
        )
        result = await handler.get_analytics_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_analytics_stats = MagicMock(
            side_effect=OSError("disk")
        )
        result = await handler.get_analytics_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_analytics_stats = MagicMock(
            side_effect=TypeError("wrong")
        )
        result = await handler.get_analytics_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_analytics_stats = MagicMock(
            side_effect=AttributeError("attr")
        )
        result = await handler.get_analytics_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_stats_is_sync_not_async(self, handler, mock_mound):
        """get_analytics_stats calls a sync method on the mound (not async)."""
        result = await handler.get_analytics_stats()
        assert _status(result) == 200
        mock_mound.get_analytics_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stats(self, handler, mock_mound):
        """Empty stats dict is returned correctly."""
        mock_mound.get_analytics_stats = MagicMock(return_value={})
        result = await handler.get_analytics_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body == {}

    @pytest.mark.asyncio
    async def test_stats_with_extra_fields(self, handler, mock_mound):
        """Stats with additional fields are passed through."""
        mock_mound.get_analytics_stats = MagicMock(
            return_value={
                "total_events": 100,
                "custom_field": "custom_value",
                "nested": {"key": "val"},
            }
        )
        result = await handler.get_analytics_stats()
        body = _body(result)
        assert body["total_events"] == 100
        assert body["custom_field"] == "custom_value"
        assert body["nested"]["key"] == "val"


# ============================================================================
# Tests: Cross-cutting concerns
# ============================================================================


class TestCrossCutting:
    """Cross-cutting tests for the AnalyticsOperationsMixin."""

    @pytest.mark.asyncio
    async def test_mixin_has_all_six_methods(self):
        """AnalyticsOperationsMixin exposes all six endpoint methods."""
        methods = [
            "analyze_coverage",
            "analyze_usage",
            "record_usage_event",
            "capture_quality_snapshot",
            "get_quality_trend",
            "get_analytics_stats",
        ]
        for method in methods:
            assert hasattr(AnalyticsOperationsMixin, method), f"Missing method: {method}"

    @pytest.mark.asyncio
    async def test_mixin_has_get_mound_stub(self):
        """AnalyticsOperationsMixin has a _get_mound stub."""
        assert hasattr(AnalyticsOperationsMixin, "_get_mound")

    @pytest.mark.asyncio
    async def test_handler_with_mound_returning_none(self):
        """Handler where _get_mound returns None gives 503 for all endpoints."""
        h = AnalyticsTestHandler(mound=None)
        results = [
            await h.analyze_coverage(workspace_id="ws-001"),
            await h.analyze_usage(workspace_id="ws-001"),
            await h.record_usage_event(event_type="query"),
            await h.capture_quality_snapshot(workspace_id="ws-001"),
            await h.get_quality_trend(workspace_id="ws-001"),
            await h.get_analytics_stats(),
        ]
        for result in results:
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_different_workspace_ids(self, handler, mock_mound):
        """Each endpoint correctly passes through different workspace IDs."""
        await handler.analyze_coverage(workspace_id="ws-alpha")
        mock_mound.analyze_coverage.assert_awaited_once_with(
            workspace_id="ws-alpha", stale_threshold_days=90,
        )

        await handler.analyze_usage(workspace_id="ws-beta")
        mock_mound.analyze_usage.assert_awaited_once_with(
            workspace_id="ws-beta", days=30,
        )

        await handler.capture_quality_snapshot(workspace_id="ws-gamma")
        mock_mound.capture_quality_snapshot.assert_awaited_once_with(
            workspace_id="ws-gamma",
        )

        await handler.get_quality_trend(workspace_id="ws-delta")
        mock_mound.get_quality_trend.assert_awaited_once_with(
            workspace_id="ws-delta", days=30,
        )

    @pytest.mark.asyncio
    async def test_os_error_gives_sanitized_message(self, handler, mock_mound):
        """OSError returns 'Resource not found' (sanitized), not raw message."""
        mock_mound.analyze_coverage = AsyncMock(
            side_effect=OSError("/etc/secret/path not found")
        )
        result = await handler.analyze_coverage(workspace_id="ws-001")
        body = _body(result)
        assert "/etc/secret/path" not in body["error"]

    @pytest.mark.asyncio
    async def test_value_error_gives_sanitized_message(self, handler, mock_mound):
        """ValueError returns sanitized message."""
        mock_mound.analyze_usage = AsyncMock(
            side_effect=ValueError("password=hunter2")
        )
        result = await handler.analyze_usage(workspace_id="ws-001")
        body = _body(result)
        assert "hunter2" not in body["error"]

"""
Tests for KnowledgeMound Analytics Handler operations.

Tests knowledge analytics functionality:
- Domain coverage analysis
- Usage pattern analysis
- Usage event recording
- Quality snapshots
- Quality trends
- Analytics statistics
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_mound():
    """Create mock KnowledgeMound instance with analytics support."""
    mound = MagicMock()

    # Mock coverage report
    mock_coverage = MagicMock()
    mock_coverage.to_dict.return_value = {
        "workspace_id": "ws_123",
        "total_items": 500,
        "by_domain": {
            "engineering": {"count": 200, "coverage": 0.85},
            "product": {"count": 150, "coverage": 0.72},
            "sales": {"count": 150, "coverage": 0.68},
        },
        "stale_items": 42,
        "gaps": ["security", "compliance"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    mound.analyze_coverage = AsyncMock(return_value=mock_coverage)

    # Mock usage report
    mock_usage = MagicMock()
    mock_usage.to_dict.return_value = {
        "workspace_id": "ws_123",
        "period_days": 30,
        "total_events": 1500,
        "by_type": {"query": 800, "view": 500, "cite": 150, "share": 50},
        "top_queries": [
            {"query": "authentication flow", "count": 45},
            {"query": "api endpoints", "count": 38},
        ],
        "most_accessed_items": [
            {"item_id": "item_1", "accesses": 120},
            {"item_id": "item_2", "accesses": 95},
        ],
    }
    mound.analyze_usage = AsyncMock(return_value=mock_usage)

    # Mock usage event recording
    mock_event = MagicMock()
    mock_event.id = "event_123"
    mock_event.event_type.value = "query"
    mock_event.timestamp = datetime.now(timezone.utc)
    mound.record_usage_event = AsyncMock(return_value=mock_event)

    # Mock quality snapshot
    mock_snapshot = MagicMock()
    mock_snapshot.to_dict.return_value = {
        "workspace_id": "ws_123",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_score": 0.78,
        "metrics": {
            "accuracy": 0.82,
            "completeness": 0.75,
            "freshness": 0.80,
            "consistency": 0.73,
        },
        "item_count": 500,
    }
    mound.capture_quality_snapshot = AsyncMock(return_value=mock_snapshot)

    # Mock quality trend
    mock_trend = MagicMock()
    mock_trend.to_dict.return_value = {
        "workspace_id": "ws_123",
        "period_days": 30,
        "snapshots": [
            {"date": "2024-01-01", "score": 0.72},
            {"date": "2024-01-15", "score": 0.75},
            {"date": "2024-01-30", "score": 0.78},
        ],
        "trend": "improving",
        "change_percent": 8.3,
    }
    mound.get_quality_trend = AsyncMock(return_value=mock_trend)

    # Mock analytics stats
    mound.get_analytics_stats = MagicMock(
        return_value={
            "total_usage_events": 15000,
            "total_snapshots": 60,
            "active_workspaces": 12,
            "avg_quality_score": 0.76,
        }
    )

    return mound


@pytest.fixture
def analytics_mixin(mock_mound):
    """Create handler instance with analytics mixin."""
    from aragora.server.handlers.knowledge_base.mound.analytics import (
        AnalyticsOperationsMixin,
    )

    class MockHandler(AnalyticsOperationsMixin):
        def __init__(self, mound):
            self.ctx = {}
            self._mound = mound

        def _get_mound(self):
            return self._mound

    return MockHandler(mock_mound)


# ===========================================================================
# Test Coverage Analysis
# ===========================================================================


class TestAnalyzeCoverage:
    """Tests for analyze_coverage endpoint."""

    async def test_analyze_coverage_success(self, analytics_mixin, mock_mound):
        """Test successful coverage analysis."""
        result = await analytics_mixin.analyze_coverage(
            workspace_id="ws_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["workspace_id"] == "ws_123"
        assert body["total_items"] == 500
        assert "by_domain" in body
        assert "gaps" in body
        mock_mound.analyze_coverage.assert_called_once()

    async def test_analyze_coverage_missing_workspace(self, analytics_mixin):
        """Test coverage fails without workspace_id."""
        result = await analytics_mixin.analyze_coverage(
            workspace_id="",
        )

        assert result.status_code == 400

    async def test_analyze_coverage_with_threshold(self, analytics_mixin, mock_mound):
        """Test coverage with custom stale threshold."""
        result = await analytics_mixin.analyze_coverage(
            workspace_id="ws_123",
            stale_threshold_days=60,
        )

        assert result.status_code == 200
        call_kwargs = mock_mound.analyze_coverage.call_args.kwargs
        assert call_kwargs["stale_threshold_days"] == 60

    async def test_analyze_coverage_mound_unavailable(self, analytics_mixin):
        """Test coverage when mound unavailable."""
        analytics_mixin._mound = None

        result = await analytics_mixin.analyze_coverage(
            workspace_id="ws_123",
        )

        assert result.status_code == 503


# ===========================================================================
# Test Usage Analysis
# ===========================================================================


class TestAnalyzeUsage:
    """Tests for analyze_usage endpoint."""

    async def test_analyze_usage_success(self, analytics_mixin, mock_mound):
        """Test successful usage analysis."""
        result = await analytics_mixin.analyze_usage(
            workspace_id="ws_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["workspace_id"] == "ws_123"
        assert body["total_events"] == 1500
        assert "by_type" in body
        assert "top_queries" in body
        mock_mound.analyze_usage.assert_called_once()

    async def test_analyze_usage_missing_workspace(self, analytics_mixin):
        """Test usage fails without workspace_id."""
        result = await analytics_mixin.analyze_usage(
            workspace_id="",
        )

        assert result.status_code == 400

    async def test_analyze_usage_with_days(self, analytics_mixin, mock_mound):
        """Test usage with custom days parameter."""
        result = await analytics_mixin.analyze_usage(
            workspace_id="ws_123",
            days=7,
        )

        assert result.status_code == 200
        call_kwargs = mock_mound.analyze_usage.call_args.kwargs
        assert call_kwargs["days"] == 7


# ===========================================================================
# Test Usage Event Recording
# ===========================================================================


class TestRecordUsageEvent:
    """Tests for record_usage_event endpoint."""

    async def test_record_event_success(self, analytics_mixin, mock_mound):
        """Test successful event recording."""
        result = await analytics_mixin.record_usage_event(
            event_type="query",
            item_id="item_123",
            user_id="user_456",
            workspace_id="ws_789",
            query="search term",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["event_id"] == "event_123"
        mock_mound.record_usage_event.assert_called_once()

    async def test_record_event_missing_type(self, analytics_mixin):
        """Test event recording fails without event_type."""
        result = await analytics_mixin.record_usage_event(
            event_type="",
        )

        assert result.status_code == 400

    async def test_record_event_invalid_type(self, analytics_mixin):
        """Test event recording fails with invalid event_type."""
        result = await analytics_mixin.record_usage_event(
            event_type="invalid_event_type",
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert "invalid event_type" in body.get("error", "").lower()

    async def test_record_event_view(self, analytics_mixin, mock_mound):
        """Test recording view event."""
        result = await analytics_mixin.record_usage_event(
            event_type="view",
            item_id="item_123",
        )

        assert result.status_code == 200

    async def test_record_event_cite(self, analytics_mixin, mock_mound):
        """Test recording cite event."""
        result = await analytics_mixin.record_usage_event(
            event_type="cite",
            item_id="item_123",
            user_id="user_456",
        )

        assert result.status_code == 200


# ===========================================================================
# Test Quality Snapshot
# ===========================================================================


class TestCaptureQualitySnapshot:
    """Tests for capture_quality_snapshot endpoint."""

    async def test_capture_snapshot_success(self, analytics_mixin, mock_mound):
        """Test successful quality snapshot capture."""
        result = await analytics_mixin.capture_quality_snapshot(
            workspace_id="ws_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["workspace_id"] == "ws_123"
        assert body["overall_score"] == 0.78
        assert "metrics" in body
        mock_mound.capture_quality_snapshot.assert_called_once()

    async def test_capture_snapshot_missing_workspace(self, analytics_mixin):
        """Test snapshot fails without workspace_id."""
        result = await analytics_mixin.capture_quality_snapshot(
            workspace_id="",
        )

        assert result.status_code == 400


# ===========================================================================
# Test Quality Trend
# ===========================================================================


class TestGetQualityTrend:
    """Tests for get_quality_trend endpoint."""

    async def test_get_trend_success(self, analytics_mixin, mock_mound):
        """Test successful quality trend retrieval."""
        result = await analytics_mixin.get_quality_trend(
            workspace_id="ws_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["workspace_id"] == "ws_123"
        assert body["trend"] == "improving"
        assert len(body["snapshots"]) == 3
        mock_mound.get_quality_trend.assert_called_once()

    async def test_get_trend_missing_workspace(self, analytics_mixin):
        """Test trend fails without workspace_id."""
        result = await analytics_mixin.get_quality_trend(
            workspace_id="",
        )

        assert result.status_code == 400

    async def test_get_trend_with_days(self, analytics_mixin, mock_mound):
        """Test trend with custom days parameter."""
        result = await analytics_mixin.get_quality_trend(
            workspace_id="ws_123",
            days=60,
        )

        assert result.status_code == 200
        call_kwargs = mock_mound.get_quality_trend.call_args.kwargs
        assert call_kwargs["days"] == 60


# ===========================================================================
# Test Analytics Stats
# ===========================================================================


class TestGetAnalyticsStats:
    """Tests for get_analytics_stats endpoint."""

    async def test_get_stats_success(self, analytics_mixin, mock_mound):
        """Test successful analytics stats retrieval."""
        result = await analytics_mixin.get_analytics_stats()

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["total_usage_events"] == 15000
        assert body["active_workspaces"] == 12
        mock_mound.get_analytics_stats.assert_called_once()

    async def test_get_stats_mound_unavailable(self, analytics_mixin):
        """Test stats when mound unavailable."""
        analytics_mixin._mound = None

        result = await analytics_mixin.get_analytics_stats()

        assert result.status_code == 503


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in analytics operations."""

    async def test_coverage_exception(self, analytics_mixin, mock_mound):
        """Test coverage analysis handles exceptions."""
        mock_mound.analyze_coverage = AsyncMock(side_effect=Exception("Analysis failed"))

        result = await analytics_mixin.analyze_coverage(
            workspace_id="ws_123",
        )

        assert result.status_code == 500

    async def test_usage_exception(self, analytics_mixin, mock_mound):
        """Test usage analysis handles exceptions."""
        mock_mound.analyze_usage = AsyncMock(side_effect=Exception("Usage query failed"))

        result = await analytics_mixin.analyze_usage(
            workspace_id="ws_123",
        )

        assert result.status_code == 500

    async def test_record_event_exception(self, analytics_mixin, mock_mound):
        """Test event recording handles exceptions."""
        mock_mound.record_usage_event = AsyncMock(side_effect=Exception("Event storage failed"))

        result = await analytics_mixin.record_usage_event(
            event_type="query",
        )

        assert result.status_code == 500

    async def test_snapshot_exception(self, analytics_mixin, mock_mound):
        """Test snapshot capture handles exceptions."""
        mock_mound.capture_quality_snapshot = AsyncMock(side_effect=Exception("Snapshot failed"))

        result = await analytics_mixin.capture_quality_snapshot(
            workspace_id="ws_123",
        )

        assert result.status_code == 500

    async def test_trend_exception(self, analytics_mixin, mock_mound):
        """Test trend retrieval handles exceptions."""
        mock_mound.get_quality_trend = AsyncMock(side_effect=Exception("Trend failed"))

        result = await analytics_mixin.get_quality_trend(
            workspace_id="ws_123",
        )

        assert result.status_code == 500

    async def test_stats_exception(self, analytics_mixin, mock_mound):
        """Test stats retrieval handles exceptions."""
        mock_mound.get_analytics_stats = MagicMock(side_effect=Exception("Stats query failed"))

        result = await analytics_mixin.get_analytics_stats()

        assert result.status_code == 500

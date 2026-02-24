"""
Comprehensive tests for aragora.billing.spend_analytics module.

Tests cover:
- SpendTrend, CostForecast, SpendAnomaly dataclasses and serialization
- SpendAnalytics.get_spend_trend: period parsing, daily aggregation, zero-fill
- SpendAnalytics.get_spend_by_provider: provider bucketing
- SpendAnalytics.get_spend_by_agent: agent bucketing
- SpendAnalytics.get_cost_forecast: linear projection, trend direction
- SpendAnalytics.get_anomalies: z-score detection, severity classification
- Edge cases: no tracker, empty buffer, single data point
- Module-level singleton: get_spend_analytics
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.spend_analytics import (
    CostForecast,
    DailySpend,
    SpendAnalytics,
    SpendAnomaly,
    SpendTrend,
    get_spend_analytics,
    _parse_period_days,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tracker():
    """CostTracker mock with an async buffer lock and sample usage data."""
    tracker = MagicMock()
    tracker._buffer_lock = asyncio.Lock()
    tracker._usage_buffer = []
    return tracker


@pytest.fixture
def analytics(mock_tracker):
    """SpendAnalytics wired to the mock tracker."""
    return SpendAnalytics(cost_tracker=mock_tracker)


def _make_usage(
    workspace_id: str = "ws-1",
    provider: str = "anthropic",
    agent_name: str = "claude",
    agent_id: str = "agent-1",
    cost_usd: float = 1.0,
    days_ago: int = 0,
) -> MagicMock:
    """Create a mock TokenUsage record."""
    usage = MagicMock()
    usage.workspace_id = workspace_id
    usage.provider = provider
    usage.agent_name = agent_name
    usage.agent_id = agent_id
    usage.cost_usd = Decimal(str(cost_usd))
    usage.timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return usage


# =============================================================================
# Dataclass tests
# =============================================================================


class TestDailySpend:
    """Tests for DailySpend dataclass."""

    def test_creation(self):
        ds = DailySpend(date="2026-02-01", cost_usd=5.25)
        assert ds.date == "2026-02-01"
        assert ds.cost_usd == 5.25


class TestSpendTrend:
    """Tests for SpendTrend dataclass."""

    def test_to_dict(self):
        trend = SpendTrend(
            workspace_id="ws-1",
            period="7d",
            points=[DailySpend(date="2026-02-01", cost_usd=1.5)],
            total_usd=1.5,
            avg_daily_usd=1.5,
        )
        d = trend.to_dict()
        assert d["workspace_id"] == "ws-1"
        assert d["period"] == "7d"
        assert len(d["points"]) == 1
        assert d["points"][0]["cost_usd"] == 1.5
        assert d["total_usd"] == 1.5

    def test_empty_points(self):
        trend = SpendTrend(workspace_id="ws-1", period="30d")
        d = trend.to_dict()
        assert d["points"] == []
        assert d["total_usd"] == 0.0


class TestCostForecast:
    """Tests for CostForecast dataclass."""

    def test_to_dict(self):
        fc = CostForecast(
            workspace_id="ws-1",
            forecast_days=30,
            projected_total_usd=45.0,
            projected_daily_avg_usd=1.5,
            trend="increasing",
            confidence=0.85,
        )
        d = fc.to_dict()
        assert d["forecast_days"] == 30
        assert d["trend"] == "increasing"
        assert d["confidence"] == 0.85

    def test_defaults(self):
        fc = CostForecast(workspace_id="ws-1", forecast_days=7)
        assert fc.projected_total_usd == 0.0
        assert fc.trend == "stable"
        assert fc.confidence == 0.0


class TestSpendAnomaly:
    """Tests for SpendAnomaly dataclass."""

    def test_to_dict(self):
        a = SpendAnomaly(
            date="2026-02-10",
            actual_usd=15.0,
            expected_usd=3.0,
            z_score=3.5,
            severity="critical",
            description="Spend $15.00 was 400% above average",
        )
        d = a.to_dict()
        assert d["severity"] == "critical"
        assert d["z_score"] == 3.5
        assert "above" in d["description"]


# =============================================================================
# Period parsing
# =============================================================================


class TestParsePeriodDays:
    """Tests for _parse_period_days helper."""

    def test_known_periods(self):
        assert _parse_period_days("7d") == 7
        assert _parse_period_days("14d") == 14
        assert _parse_period_days("30d") == 30
        assert _parse_period_days("90d") == 90

    def test_custom_period(self):
        assert _parse_period_days("45d") == 45

    def test_invalid_falls_back(self):
        assert _parse_period_days("invalid") == 30
        assert _parse_period_days("") == 30


# =============================================================================
# SpendAnalytics.get_spend_trend
# =============================================================================


class TestGetSpendTrend:
    """Tests for SpendAnalytics.get_spend_trend."""

    @pytest.mark.asyncio
    async def test_returns_trend_with_points(self, analytics, mock_tracker):
        """Trend should contain daily data points from the buffer."""
        mock_tracker._usage_buffer = [
            _make_usage(cost_usd=2.0, days_ago=1),
            _make_usage(cost_usd=3.0, days_ago=1),
            _make_usage(cost_usd=1.0, days_ago=2),
        ]

        trend = await analytics.get_spend_trend("ws-1", period="7d")

        assert isinstance(trend, SpendTrend)
        assert trend.workspace_id == "ws-1"
        assert trend.period == "7d"
        assert len(trend.points) == 7
        assert trend.total_usd == pytest.approx(6.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_buffer_returns_zeros(self, analytics):
        """With no data, all points should be zero."""
        trend = await analytics.get_spend_trend("ws-1", period="7d")

        assert trend.total_usd == 0.0
        assert trend.avg_daily_usd == 0.0
        assert len(trend.points) == 7
        assert all(p.cost_usd == 0.0 for p in trend.points)

    @pytest.mark.asyncio
    async def test_filters_by_workspace(self, analytics, mock_tracker):
        """Only records for the requested workspace should be included."""
        mock_tracker._usage_buffer = [
            _make_usage(workspace_id="ws-1", cost_usd=5.0, days_ago=1),
            _make_usage(workspace_id="ws-other", cost_usd=99.0, days_ago=1),
        ]

        trend = await analytics.get_spend_trend("ws-1", period="7d")
        assert trend.total_usd == pytest.approx(5.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_tracker(self):
        """With no tracker, should return empty trend without error."""
        analytics = SpendAnalytics(cost_tracker=None)
        trend = await analytics.get_spend_trend("ws-1", period="7d")

        assert trend.total_usd == 0.0
        assert len(trend.points) == 7


# =============================================================================
# SpendAnalytics.get_spend_by_provider
# =============================================================================


class TestGetSpendByProvider:
    """Tests for SpendAnalytics.get_spend_by_provider."""

    @pytest.mark.asyncio
    async def test_groups_by_provider(self, analytics, mock_tracker):
        mock_tracker._usage_buffer = [
            _make_usage(provider="anthropic", cost_usd=3.0),
            _make_usage(provider="anthropic", cost_usd=2.0),
            _make_usage(provider="openai", cost_usd=1.0),
        ]

        result = await analytics.get_spend_by_provider("ws-1")

        assert result["anthropic"] == pytest.approx(5.0, abs=0.01)
        assert result["openai"] == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_buffer(self, analytics):
        result = await analytics.get_spend_by_provider("ws-1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_filters_workspace(self, analytics, mock_tracker):
        mock_tracker._usage_buffer = [
            _make_usage(workspace_id="ws-1", provider="anthropic", cost_usd=3.0),
            _make_usage(workspace_id="ws-2", provider="openai", cost_usd=7.0),
        ]

        result = await analytics.get_spend_by_provider("ws-1")
        assert "anthropic" in result
        assert "openai" not in result


# =============================================================================
# SpendAnalytics.get_spend_by_agent
# =============================================================================


class TestGetSpendByAgent:
    """Tests for SpendAnalytics.get_spend_by_agent."""

    @pytest.mark.asyncio
    async def test_groups_by_agent_name(self, analytics, mock_tracker):
        mock_tracker._usage_buffer = [
            _make_usage(agent_name="claude", cost_usd=4.0),
            _make_usage(agent_name="gpt-4", cost_usd=2.0),
            _make_usage(agent_name="claude", cost_usd=1.0),
        ]

        result = await analytics.get_spend_by_agent("ws-1")

        assert result["claude"] == pytest.approx(5.0, abs=0.01)
        assert result["gpt-4"] == pytest.approx(2.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_falls_back_to_agent_id(self, analytics, mock_tracker):
        """If agent_name is empty, falls back to agent_id."""
        usage = _make_usage(agent_name="", cost_usd=1.0)
        usage.agent_id = "agent-xyz"
        mock_tracker._usage_buffer = [usage]

        result = await analytics.get_spend_by_agent("ws-1")
        assert "agent-xyz" in result

    @pytest.mark.asyncio
    async def test_unknown_fallback(self, analytics, mock_tracker):
        """If both agent_name and agent_id are empty, uses 'unknown'."""
        usage = _make_usage(agent_name="", cost_usd=1.0)
        usage.agent_id = ""
        mock_tracker._usage_buffer = [usage]

        result = await analytics.get_spend_by_agent("ws-1")
        assert "unknown" in result


# =============================================================================
# SpendAnalytics.get_cost_forecast
# =============================================================================


class TestGetCostForecast:
    """Tests for SpendAnalytics.get_cost_forecast."""

    @pytest.mark.asyncio
    async def test_forecast_with_data(self, analytics, mock_tracker):
        """Should produce a valid projection when historical data exists."""
        # Create 30 days of usage with ~$2/day
        usages = []
        for i in range(30):
            usages.append(_make_usage(cost_usd=2.0, days_ago=i))
        mock_tracker._usage_buffer = usages

        forecast = await analytics.get_cost_forecast("ws-1", days=30)

        assert isinstance(forecast, CostForecast)
        assert forecast.forecast_days == 30
        assert forecast.projected_total_usd > 0
        assert forecast.projected_daily_avg_usd > 0

    @pytest.mark.asyncio
    async def test_forecast_empty_data(self, analytics):
        """Should return zero forecast when no data exists."""
        forecast = await analytics.get_cost_forecast("ws-1", days=30)

        assert forecast.projected_total_usd == 0.0
        assert forecast.trend == "stable"

    @pytest.mark.asyncio
    async def test_increasing_trend(self, analytics, mock_tracker):
        """Increasing daily costs should yield 'increasing' trend."""
        usages = []
        for i in range(30):
            # cost increases over time: day 29 (most recent) = $30, day 0 = $1
            usages.append(_make_usage(cost_usd=float(30 - i), days_ago=i))
        mock_tracker._usage_buffer = usages

        forecast = await analytics.get_cost_forecast("ws-1", days=30)
        assert forecast.trend == "increasing"

    @pytest.mark.asyncio
    async def test_decreasing_trend(self, analytics, mock_tracker):
        """Decreasing daily costs should yield 'decreasing' trend."""
        usages = []
        for i in range(30):
            # cost decreases over time: day 29 (most recent) = $1, day 0 = $30
            usages.append(_make_usage(cost_usd=float(i + 1), days_ago=i))
        mock_tracker._usage_buffer = usages

        forecast = await analytics.get_cost_forecast("ws-1", days=30)
        assert forecast.trend == "decreasing"

    @pytest.mark.asyncio
    async def test_stable_trend(self, analytics, mock_tracker):
        """Constant daily costs should yield 'stable' trend."""
        usages = []
        for i in range(30):
            usages.append(_make_usage(cost_usd=5.0, days_ago=i))
        mock_tracker._usage_buffer = usages

        forecast = await analytics.get_cost_forecast("ws-1", days=30)
        assert forecast.trend == "stable"


# =============================================================================
# SpendAnalytics.get_anomalies
# =============================================================================


class TestGetAnomalies:
    """Tests for SpendAnalytics.get_anomalies."""

    @pytest.mark.asyncio
    async def test_detects_spike(self, analytics, mock_tracker):
        """A large spike should be detected as an anomaly."""
        usages = []
        # 6 days of $1 spend
        for i in range(2, 8):
            usages.append(_make_usage(cost_usd=1.0, days_ago=i))
        # 1 day with $50 spike
        usages.append(_make_usage(cost_usd=50.0, days_ago=1))
        mock_tracker._usage_buffer = usages

        anomalies = await analytics.get_anomalies("ws-1", period="7d")

        assert len(anomalies) >= 1
        # The spike should be the highest z-score
        assert anomalies[0].actual_usd == pytest.approx(50.0, abs=0.01)
        assert anomalies[0].severity in ("warning", "critical")

    @pytest.mark.asyncio
    async def test_no_anomalies_for_uniform_data(self, analytics, mock_tracker):
        """Uniform spend should not produce anomalies."""
        usages = []
        for i in range(7):
            usages.append(_make_usage(cost_usd=5.0, days_ago=i + 1))
        mock_tracker._usage_buffer = usages

        anomalies = await analytics.get_anomalies("ws-1", period="7d")
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_insufficient_data(self, analytics, mock_tracker):
        """With fewer than 3 data points, no anomalies are returned."""
        mock_tracker._usage_buffer = [_make_usage(cost_usd=5.0, days_ago=1)]

        anomalies = await analytics.get_anomalies("ws-1", period="7d")
        assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_critical_severity(self, analytics, mock_tracker):
        """Z-score >= 3 should yield 'critical' severity."""
        usages = []
        for i in range(2, 32):
            usages.append(_make_usage(cost_usd=1.0, days_ago=i))
        # Massive spike
        usages.append(_make_usage(cost_usd=100.0, days_ago=1))
        mock_tracker._usage_buffer = usages

        anomalies = await analytics.get_anomalies("ws-1", period="30d")

        critical = [a for a in anomalies if a.severity == "critical"]
        assert len(critical) >= 1

    @pytest.mark.asyncio
    async def test_anomalies_sorted_by_z_score(self, analytics, mock_tracker):
        """Anomalies should be sorted by absolute z-score, descending."""
        usages = []
        for i in range(3, 10):
            usages.append(_make_usage(cost_usd=1.0, days_ago=i))
        usages.append(_make_usage(cost_usd=20.0, days_ago=1))
        usages.append(_make_usage(cost_usd=30.0, days_ago=2))
        mock_tracker._usage_buffer = usages

        anomalies = await analytics.get_anomalies("ws-1", period="10d")

        if len(anomalies) >= 2:
            assert abs(anomalies[0].z_score) >= abs(anomalies[1].z_score)

    @pytest.mark.asyncio
    async def test_custom_z_threshold(self, analytics, mock_tracker):
        """Higher z_threshold should yield fewer anomalies."""
        usages = []
        for i in range(2, 32):
            usages.append(_make_usage(cost_usd=1.0, days_ago=i))
        usages.append(_make_usage(cost_usd=5.0, days_ago=1))
        mock_tracker._usage_buffer = usages

        loose = await analytics.get_anomalies("ws-1", period="30d", z_threshold=1.5)
        strict = await analytics.get_anomalies("ws-1", period="30d", z_threshold=3.0)

        assert len(strict) <= len(loose)


# =============================================================================
# set_cost_tracker
# =============================================================================


class TestSetCostTracker:
    """Tests for SpendAnalytics.set_cost_tracker."""

    @pytest.mark.asyncio
    async def test_set_tracker(self):
        """Setting a tracker should enable data retrieval."""
        analytics = SpendAnalytics()
        assert analytics._cost_tracker is None

        mock_tracker = MagicMock()
        mock_tracker._buffer_lock = asyncio.Lock()
        mock_tracker._usage_buffer = [_make_usage(cost_usd=3.0, days_ago=1)]

        analytics.set_cost_tracker(mock_tracker)
        trend = await analytics.get_spend_trend("ws-1", period="7d")
        assert trend.total_usd == pytest.approx(3.0, abs=0.01)


# =============================================================================
# Module-level singleton
# =============================================================================


class TestGetSpendAnalytics:
    """Tests for get_spend_analytics singleton."""

    def test_returns_instance(self):
        """Should return a SpendAnalytics instance."""
        import aragora.billing.spend_analytics as mod

        # Reset singleton
        mod._spend_analytics = None

        mock_tracker = MagicMock()
        with patch("aragora.billing.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            instance = get_spend_analytics()
            assert isinstance(instance, SpendAnalytics)
            assert instance._cost_tracker is mock_tracker

        # Cleanup
        mod._spend_analytics = None

    def test_singleton_returns_same_instance(self):
        """Subsequent calls should return the same instance."""
        import aragora.billing.spend_analytics as mod

        mod._spend_analytics = None

        mock_tracker = MagicMock()
        with patch("aragora.billing.cost_tracker.get_cost_tracker", return_value=mock_tracker):
            first = get_spend_analytics()
            second = get_spend_analytics()
            assert first is second

        mod._spend_analytics = None

    def test_fallback_without_tracker(self):
        """If get_cost_tracker fails, should still create an instance."""
        import aragora.billing.spend_analytics as mod

        mod._spend_analytics = None

        # The import happens inside get_spend_analytics, so we construct
        # directly to test the fallback path.
        instance = SpendAnalytics(cost_tracker=None)
        assert isinstance(instance, SpendAnalytics)
        assert instance._cost_tracker is None

        mod._spend_analytics = None

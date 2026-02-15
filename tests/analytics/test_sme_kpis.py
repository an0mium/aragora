"""
Tests for SME KPI computation module.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aragora.analytics.sme_kpis import (
    DEFAULT_HOURLY_RATE,
    DEFAULT_MANUAL_DECISION_HOURS,
    PERIOD_DAYS,
    SMEKPIs,
    get_sme_kpis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_debate_stats(total=20, completed=18, consensus=15, avg_duration=120.0):
    """Create a mock DebateStats-like object."""
    stats = MagicMock()
    stats.total_debates = total
    stats.completed_debates = completed
    stats.consensus_reached = consensus
    stats.consensus_rate = consensus / completed if completed > 0 else 0.0
    stats.avg_duration_seconds = avg_duration
    return stats


def _mock_workspace_stats(total_cost="10.00", api_calls=100):
    return {
        "total_cost_usd": total_cost,
        "total_api_calls": api_calls,
    }


# ---------------------------------------------------------------------------
# SMEKPIs dataclass
# ---------------------------------------------------------------------------


class TestSMEKPIs:
    def test_default_values(self):
        kpis = SMEKPIs()
        assert kpis.cost_per_decision == 0.0
        assert kpis.decision_velocity == 0.0
        assert kpis.time_saved_hours == 0.0
        assert kpis.roi_percentage == 0.0
        assert kpis.total_debates == 0

    def test_to_dict(self):
        kpis = SMEKPIs(
            cost_per_decision=1.5,
            decision_velocity=3.2,
            time_saved_hours=10.5,
            roi_percentage=250.3,
            total_debates=25,
            total_cost_usd=37.5,
            period_days=30,
        )
        d = kpis.to_dict()
        assert d["cost_per_decision"] == 1.5
        assert d["decision_velocity"] == 3.2
        assert d["time_saved_hours"] == 10.5
        assert d["roi_percentage"] == 250.3
        assert d["total_debates"] == 25
        assert d["total_cost_usd"] == 37.5
        assert d["period_days"] == 30

    def test_to_dict_rounds_values(self):
        kpis = SMEKPIs(cost_per_decision=1.23456, roi_percentage=99.999)
        d = kpis.to_dict()
        assert d["cost_per_decision"] == 1.23
        assert d["roi_percentage"] == 100.0

    def test_to_dict_has_all_keys(self):
        d = SMEKPIs().to_dict()
        expected = [
            "cost_per_decision",
            "decision_velocity",
            "time_saved_hours",
            "roi_percentage",
            "total_debates",
            "total_cost_usd",
            "avg_debate_duration_minutes",
            "period_days",
        ]
        for key in expected:
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Period constants
# ---------------------------------------------------------------------------


class TestPeriodDays:
    def test_week(self):
        assert PERIOD_DAYS["week"] == 7

    def test_month(self):
        assert PERIOD_DAYS["month"] == 30

    def test_quarter(self):
        assert PERIOD_DAYS["quarter"] == 90

    def test_year(self):
        assert PERIOD_DAYS["year"] == 365


# ---------------------------------------------------------------------------
# get_sme_kpis with real analytics
# ---------------------------------------------------------------------------


class TestGetSMEKPIsWithAnalytics:
    """Tests with DebateAnalytics providing real data."""

    def _setup_mocks(
        self,
        debate_stats=None,
        workspace_stats=None,
    ):
        """Set up analytics and cost tracker mocks."""
        mock_analytics = MagicMock()
        stats = debate_stats or _mock_debate_stats()

        async def fake_get_stats(**kwargs):
            return stats

        mock_analytics.get_debate_stats = fake_get_stats

        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = workspace_stats or _mock_workspace_stats()

        return mock_analytics, mock_tracker

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_cost_per_decision(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=10),
            workspace_stats=_mock_workspace_stats(total_cost="20.00"),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.cost_per_decision == 2.0  # $20 / 10 debates

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_decision_velocity(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=28),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123", period="month")
        # 28 debates / (30/7 weeks) = 28 / 4.286 = ~6.53
        assert kpis.decision_velocity == pytest.approx(6.53, abs=0.1)

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_time_saved_hours(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=20, avg_duration=300.0),  # 5 min
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        # Manual: 20 * 0.75h = 15h; AI: 20 * 5min/60 = 1.67h; Saved: 13.33h
        assert kpis.time_saved_hours == pytest.approx(13.33, abs=0.1)

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_roi_percentage(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=20, avg_duration=300.0),
            workspace_stats=_mock_workspace_stats(total_cost="10.00"),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        # time_saved = 13.33h * $75 = $1000; ROI = ($1000-$10)/$10 * 100 = 9900%
        assert kpis.roi_percentage > 0
        assert kpis.roi_percentage == pytest.approx(9900, abs=100)

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_total_debates_from_analytics(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=42),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.total_debates == 42

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_avg_duration_from_analytics(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=10, avg_duration=180.0),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.avg_debate_duration_minutes == 3.0  # 180s / 60

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_week_period(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=7),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123", period="week")
        assert kpis.period_days == 7
        assert kpis.decision_velocity == 7.0  # 7 debates / 1 week

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_custom_benchmarks(self, mock_get_analytics, mock_get_tracker):
        analytics, tracker = self._setup_mocks(
            debate_stats=_mock_debate_stats(total=10, avg_duration=300.0),
            workspace_stats=_mock_workspace_stats(total_cost="5.00"),
        )
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis(
            "org_123",
            manual_decision_hours=1.0,
            hourly_rate=100.0,
        )
        # Manual: 10 * 1h = 10h; AI: 10 * 5min/60 = 0.833h; Saved: 9.167h
        assert kpis.time_saved_hours == pytest.approx(9.17, abs=0.1)
        # ROI: (9.167 * $100 - $5) / $5 * 100
        assert kpis.roi_percentage > 0


# ---------------------------------------------------------------------------
# get_sme_kpis fallback scenarios
# ---------------------------------------------------------------------------


class TestGetSMEKPIsFallback:
    """Tests when analytics or cost tracker are unavailable."""

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_no_analytics_estimates_from_api_calls(self, mock_get_analytics, mock_get_tracker):
        mock_get_analytics.return_value = None
        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _mock_workspace_stats(
            total_cost="10.00", api_calls=200
        )
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.total_debates == 20  # 200 // 10

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_no_cost_tracker(self, mock_get_analytics, mock_get_tracker):
        analytics, _ = MagicMock(), None
        stats = _mock_debate_stats(total=10)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = None

        kpis = get_sme_kpis("org_123")
        assert kpis.total_cost_usd == 0.0
        assert kpis.cost_per_decision == 0.0
        assert kpis.total_debates == 10

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_both_unavailable(self, mock_get_analytics, mock_get_tracker):
        mock_get_analytics.return_value = None
        mock_get_tracker.return_value = None

        kpis = get_sme_kpis("org_123")
        assert kpis.total_debates == 0
        assert kpis.cost_per_decision == 0.0
        assert kpis.decision_velocity == 0.0
        assert kpis.time_saved_hours == 0.0
        assert kpis.roi_percentage == 0.0

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_analytics_exception(self, mock_get_analytics, mock_get_tracker):
        analytics = MagicMock()

        async def failing_stats(**kwargs):
            raise RuntimeError("DB error")

        analytics.get_debate_stats = failing_stats
        mock_get_analytics.return_value = analytics

        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _mock_workspace_stats(api_calls=50)
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.total_debates == 5  # Fallback: 50//10

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_zero_debates_from_analytics(self, mock_get_analytics, mock_get_tracker):
        analytics = MagicMock()
        stats = _mock_debate_stats(total=0, completed=0, consensus=0)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics

        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _mock_workspace_stats(api_calls=60)
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        # Falls back to estimation when analytics returns 0 debates
        assert kpis.total_debates == 6  # 60//10

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_zero_cost_roi_is_zero(self, mock_get_analytics, mock_get_tracker):
        analytics = MagicMock()
        stats = _mock_debate_stats(total=10)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics

        tracker = MagicMock()
        tracker.get_workspace_stats.return_value = _mock_workspace_stats(total_cost="0")
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.roi_percentage == 0.0  # Can't divide by zero

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_invalid_period_defaults_to_month(self, mock_get_analytics, mock_get_tracker):
        mock_get_analytics.return_value = None
        mock_get_tracker.return_value = None

        kpis = get_sme_kpis("org_123", period="invalid")
        assert kpis.period_days == 30  # Default

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_cost_tracker_exception(self, mock_get_analytics, mock_get_tracker):
        mock_get_analytics.return_value = None
        tracker = MagicMock()
        tracker.get_workspace_stats.side_effect = RuntimeError("Connection refused")
        mock_get_tracker.return_value = tracker

        kpis = get_sme_kpis("org_123")
        assert kpis.total_cost_usd == 0.0

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_time_saved_never_negative(self, mock_get_analytics, mock_get_tracker):
        """Time saved should never be negative even with long AI durations."""
        analytics = MagicMock()
        # AI duration longer than manual benchmark (edge case)
        stats = _mock_debate_stats(total=10, avg_duration=3600.0)  # 60 min

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = MagicMock(
            get_workspace_stats=MagicMock(return_value=_mock_workspace_stats())
        )

        kpis = get_sme_kpis("org_123")
        assert kpis.time_saved_hours >= 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_quarter_period_velocity(self, mock_get_analytics, mock_get_tracker):
        analytics = MagicMock()
        stats = _mock_debate_stats(total=90)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = MagicMock(
            get_workspace_stats=MagicMock(return_value=_mock_workspace_stats())
        )

        kpis = get_sme_kpis("org_123", period="quarter")
        # 90 debates / (90/7 weeks) = 90 / 12.86 = ~7.0
        assert kpis.decision_velocity == pytest.approx(7.0, abs=0.1)

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_year_period(self, mock_get_analytics, mock_get_tracker):
        analytics = MagicMock()
        stats = _mock_debate_stats(total=500)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = MagicMock(
            get_workspace_stats=MagicMock(return_value=_mock_workspace_stats())
        )

        kpis = get_sme_kpis("org_123", period="year")
        assert kpis.period_days == 365
        assert kpis.total_debates == 500

    @patch("aragora.analytics.sme_kpis._get_cost_tracker")
    @patch("aragora.analytics.sme_kpis._get_debate_analytics")
    def test_zero_avg_duration_defaults(self, mock_get_analytics, mock_get_tracker):
        """When avg_duration_seconds is 0, use default 5 minutes."""
        analytics = MagicMock()
        stats = _mock_debate_stats(total=10, avg_duration=0.0)

        async def fake_get_stats(**kwargs):
            return stats

        analytics.get_debate_stats = fake_get_stats
        mock_get_analytics.return_value = analytics
        mock_get_tracker.return_value = MagicMock(
            get_workspace_stats=MagicMock(return_value=_mock_workspace_stats())
        )

        kpis = get_sme_kpis("org_123")
        assert kpis.avg_debate_duration_minutes == 5.0

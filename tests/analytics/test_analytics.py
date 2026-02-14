"""
Comprehensive tests for the aragora analytics module.

Covers:
- debate_analytics.py: SQLite-backed DebateAnalytics, dataclasses, enums
- dashboard.py: In-memory AnalyticsDashboard, caching, metrics, compliance
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# debate_analytics imports
from aragora.analytics.debate_analytics import (
    AgentPerformance,
    CostBreakdown,
    DebateAnalytics,
    DebateDashboardSummary,
    DebateMetricType,
    DebateStats,
    DebateTimeGranularity,
    UsageTrendPoint,
    get_debate_analytics,
)

# dashboard imports
from aragora.analytics.dashboard import (
    AgentMetrics,
    AnalyticsDashboard,
    AuditCostMetrics,
    ComplianceScore,
    DashboardSummary,
    FindingTrend,
    Granularity,
    RemediationMetrics,
    RiskHeatmapCell,
    TimeRange,
    get_analytics_dashboard,
)


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestDebateTimeGranularity:
    def test_values(self):
        assert DebateTimeGranularity.HOURLY == "hourly"
        assert DebateTimeGranularity.DAILY == "daily"
        assert DebateTimeGranularity.WEEKLY == "weekly"
        assert DebateTimeGranularity.MONTHLY == "monthly"

    def test_is_str_enum(self):
        assert isinstance(DebateTimeGranularity.HOURLY, str)

    def test_all_members(self):
        members = list(DebateTimeGranularity)
        assert len(members) == 4


class TestDebateMetricType:
    def test_values(self):
        assert DebateMetricType.DEBATE_COUNT == "debate_count"
        assert DebateMetricType.CONSENSUS_RATE == "consensus_rate"
        assert DebateMetricType.AVG_ROUNDS == "avg_rounds"
        assert DebateMetricType.AVG_DURATION == "avg_duration"
        assert DebateMetricType.AGENT_RESPONSE_TIME == "agent_response_time"
        assert DebateMetricType.AGENT_ACCURACY == "agent_accuracy"
        assert DebateMetricType.TOKEN_USAGE == "token_usage"
        assert DebateMetricType.COST_TOTAL == "cost_total"
        assert DebateMetricType.USER_ACTIVITY == "user_activity"
        assert DebateMetricType.ERROR_RATE == "error_rate"

    def test_all_members(self):
        assert len(list(DebateMetricType)) == 10


class TestTimeRange:
    def test_values(self):
        assert TimeRange.LAST_24_HOURS == "24h"
        assert TimeRange.LAST_7_DAYS == "7d"
        assert TimeRange.LAST_30_DAYS == "30d"
        assert TimeRange.LAST_90_DAYS == "90d"
        assert TimeRange.LAST_365_DAYS == "365d"
        assert TimeRange.ALL_TIME == "all"

    def test_to_timedelta(self):
        assert TimeRange.LAST_24_HOURS.to_timedelta() == timedelta(hours=24)
        assert TimeRange.LAST_7_DAYS.to_timedelta() == timedelta(days=7)
        assert TimeRange.LAST_30_DAYS.to_timedelta() == timedelta(days=30)
        assert TimeRange.LAST_90_DAYS.to_timedelta() == timedelta(days=90)
        assert TimeRange.LAST_365_DAYS.to_timedelta() == timedelta(days=365)
        assert TimeRange.ALL_TIME.to_timedelta() is None


class TestGranularity:
    def test_values(self):
        assert Granularity.HOURLY == "hour"
        assert Granularity.DAILY == "day"
        assert Granularity.WEEKLY == "week"
        assert Granularity.MONTHLY == "month"

    def test_all_members(self):
        assert len(list(Granularity)) == 4


# ===========================================================================
# Dataclass Tests - debate_analytics
# ===========================================================================


class TestDebateStats:
    def test_defaults(self):
        stats = DebateStats()
        assert stats.total_debates == 0
        assert stats.completed_debates == 0
        assert stats.failed_debates == 0
        assert stats.consensus_reached == 0
        assert stats.consensus_rate == 0.0
        assert stats.avg_rounds == 0.0
        assert stats.avg_duration_seconds == 0.0
        assert stats.avg_agents_per_debate == 0.0
        assert stats.total_messages == 0
        assert stats.total_votes == 0
        assert stats.period_start is None
        assert stats.period_end is None
        assert stats.by_protocol == {}

    def test_to_dict(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        stats = DebateStats(
            total_debates=10,
            completed_debates=8,
            failed_debates=2,
            consensus_reached=6,
            consensus_rate=0.75,
            avg_rounds=3.333,
            avg_duration_seconds=120.567,
            avg_agents_per_debate=3.5,
            total_messages=100,
            total_votes=40,
            period_start=now,
            period_end=now + timedelta(days=30),
            by_protocol={"majority": 5, "unanimous": 3},
        )
        d = stats.to_dict()
        assert d["total_debates"] == 10
        assert d["completed_debates"] == 8
        assert d["consensus_rate"] == 0.75
        assert d["avg_rounds"] == 3.33
        assert d["avg_duration_seconds"] == 120.57
        assert d["period_start"] == now.isoformat()
        assert d["period_end"] == (now + timedelta(days=30)).isoformat()
        assert d["by_protocol"] == {"majority": 5, "unanimous": 3}

    def test_to_dict_none_periods(self):
        d = DebateStats().to_dict()
        assert d["period_start"] is None
        assert d["period_end"] is None


class TestAgentPerformance:
    def test_defaults(self):
        perf = AgentPerformance()
        assert perf.agent_id == ""
        assert perf.current_elo == 1500.0
        assert perf.total_cost == Decimal("0")
        assert perf.rank == 0

    def test_to_dict(self):
        perf = AgentPerformance(
            agent_id="claude",
            agent_name="Claude",
            provider="anthropic",
            model="claude-3",
            debates_participated=5,
            messages_sent=20,
            avg_response_time_ms=150.456,
            error_count=1,
            error_rate=0.05,
            total_tokens_in=5000,
            total_tokens_out=3000,
            total_cost=Decimal("1.50"),
            avg_cost_per_debate=Decimal("0.30"),
            current_elo=1600.0,
            elo_change_period=100.0,
            rank=1,
        )
        d = perf.to_dict()
        assert d["agent_id"] == "claude"
        assert d["provider"] == "anthropic"
        assert d["total_cost"] == "1.50"
        assert d["avg_cost_per_debate"] == "0.30"
        assert d["current_elo"] == 1600.0
        assert d["avg_response_time_ms"] == 150.46
        assert d["rank"] == 1


class TestUsageTrendPoint:
    def test_to_dict(self):
        ts = datetime(2025, 6, 15, tzinfo=timezone.utc)
        point = UsageTrendPoint(
            timestamp=ts,
            value=42.0,
            metric=DebateMetricType.DEBATE_COUNT,
        )
        d = point.to_dict()
        assert d["timestamp"] == ts.isoformat()
        assert d["value"] == 42.0
        assert d["metric"] == "debate_count"


class TestCostBreakdown:
    def test_to_dict(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        cb = CostBreakdown(
            period_start=start,
            period_end=end,
            total_cost=Decimal("100.50"),
            by_provider={"anthropic": Decimal("60"), "openai": Decimal("40.50")},
            by_model={"claude-3": Decimal("60"), "gpt-4": Decimal("40.50")},
            by_user={"user1": Decimal("50"), "user2": Decimal("50.50")},
            by_org={"org1": Decimal("100.50")},
            daily_costs=[("2025-01-01", Decimal("5.00"))],
            projected_monthly=Decimal("150"),
            cost_per_debate=Decimal("10.05"),
            cost_per_consensus=Decimal("16.75"),
        )
        d = cb.to_dict()
        assert d["total_cost"] == "100.50"
        assert d["by_provider"]["anthropic"] == "60"
        assert d["projected_monthly"] == "150"
        assert d["daily_costs"] == [("2025-01-01", "5.00")]


class TestDebateDashboardSummary:
    def test_defaults(self):
        summary = DebateDashboardSummary()
        assert summary.total_debates == 0
        assert summary.total_users == 0
        assert summary.debate_stats is None
        assert summary.cost_breakdown is None
        assert summary.top_agents == []
        assert summary.alerts == []

    def test_to_dict_with_nested(self):
        stats = DebateStats(total_debates=5)
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 31, tzinfo=timezone.utc)
        cb = CostBreakdown(period_start=start, period_end=end)
        agent = AgentPerformance(agent_id="a1")
        trend = UsageTrendPoint(
            timestamp=start,
            value=3.0,
            metric=DebateMetricType.DEBATE_COUNT,
        )

        summary = DebateDashboardSummary(
            period_start=start,
            period_end=end,
            total_debates=5,
            debate_stats=stats,
            cost_breakdown=cb,
            top_agents=[agent],
            debate_trend=[trend],
            alerts=[{"level": "warning", "message": "test"}],
        )
        d = summary.to_dict()
        assert d["total_debates"] == 5
        assert d["debate_stats"]["total_debates"] == 5
        assert d["cost_breakdown"]["total_cost"] == "0"
        assert len(d["top_agents"]) == 1
        assert len(d["debate_trend"]) == 1
        assert len(d["alerts"]) == 1

    def test_to_dict_none_nested(self):
        d = DebateDashboardSummary().to_dict()
        assert d["debate_stats"] is None
        assert d["cost_breakdown"] is None
        assert d["period_start"] is None
        assert d["period_end"] is None


# ===========================================================================
# Dataclass Tests - dashboard
# ===========================================================================


class TestFindingTrend:
    def test_to_dict(self):
        ts = datetime(2025, 3, 1, tzinfo=timezone.utc)
        ft = FindingTrend(
            timestamp=ts,
            total=10,
            by_severity={"critical": 2, "high": 8},
            by_category={"auth": 5, "crypto": 5},
            by_status={"open": 7, "resolved": 3},
        )
        d = ft.to_dict()
        assert d["total"] == 10
        assert d["by_severity"]["critical"] == 2
        assert d["timestamp"] == ts.isoformat()


class TestRemediationMetrics:
    def test_to_dict(self):
        rm = RemediationMetrics(
            total_resolved=50,
            total_open=10,
            mttr_hours=24.567,
            mttr_by_severity={"critical": 4.123, "high": 12.999},
            false_positive_rate=0.05432,
            accepted_risk_rate=0.10001,
        )
        d = rm.to_dict()
        assert d["total_resolved"] == 50
        assert d["mttr_hours"] == 24.57
        assert d["mttr_by_severity"]["critical"] == 4.12
        assert d["false_positive_rate"] == 0.0543
        assert d["accepted_risk_rate"] == 0.1


class TestAgentMetricsDashboard:
    def test_to_dict(self):
        am = AgentMetrics(
            agent_id="a1",
            agent_name="Agent1",
            total_findings=20,
            agreement_rate=0.85123,
            precision=0.92456,
            finding_distribution={"critical": 5, "high": 15},
            avg_response_time_ms=123.456,
        )
        d = am.to_dict()
        assert d["agent_id"] == "a1"
        assert d["agreement_rate"] == 0.8512
        assert d["precision"] == 0.9246
        assert d["avg_response_time_ms"] == 123.46


class TestAuditCostMetrics:
    def test_to_dict(self):
        acm = AuditCostMetrics(
            total_audits=100,
            total_cost_usd=250.567,
            avg_cost_per_audit=2.50567,
            cost_by_type={"security": 150.123, "compliance": 100.444},
            token_usage={"input": 50000, "output": 30000},
        )
        d = acm.to_dict()
        assert d["total_audits"] == 100
        assert d["total_cost_usd"] == 250.57
        assert d["avg_cost_per_audit"] == 2.51
        assert d["cost_by_type"]["security"] == 150.12
        assert d["token_usage"]["input"] == 50000


class TestComplianceScore:
    def test_to_dict(self):
        cs = ComplianceScore(
            framework="SOC2",
            score=0.85123,
            passing_controls=17,
            failing_controls=3,
            not_applicable=2,
            critical_gaps=["access_control", "monitoring"],
        )
        d = cs.to_dict()
        assert d["framework"] == "SOC2"
        assert d["score"] == 0.8512
        assert d["passing_controls"] == 17
        assert d["critical_gaps"] == ["access_control", "monitoring"]


class TestRiskHeatmapCell:
    def test_to_dict(self):
        cell = RiskHeatmapCell(
            category="auth",
            severity="critical",
            count=5,
            trend="up",
        )
        d = cell.to_dict()
        assert d["category"] == "auth"
        assert d["severity"] == "critical"
        assert d["count"] == 5
        assert d["trend"] == "up"


class TestDashboardSummary:
    def test_to_dict(self):
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        ds = DashboardSummary(
            workspace_id="ws-123",
            time_range=TimeRange.LAST_30_DAYS,
            generated_at=now,
            total_findings=100,
            open_findings=40,
            critical_findings=5,
            resolved_last_period=20,
            finding_trend="up",
            trend_percentage=15.678,
            top_categories=[("auth", 30), ("crypto", 25)],
            recent_critical=[{"id": "f1", "title": "Critical bug"}],
        )
        d = ds.to_dict()
        assert d["workspace_id"] == "ws-123"
        assert d["time_range"] == "30d"
        assert d["total_findings"] == 100
        assert d["trend_percentage"] == 15.68
        assert d["top_categories"] == [
            {"category": "auth", "count": 30},
            {"category": "crypto", "count": 25},
        ]
        assert d["recent_critical"] == [{"id": "f1", "title": "Critical bug"}]


# ===========================================================================
# DebateAnalytics Tests (SQLite-backed)
# ===========================================================================


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_analytics.db")


class TestDebateAnalyticsInit:
    def test_creates_in_memory_db(self):
        analytics = DebateAnalytics()
        assert analytics.db_path == ":memory:"

    def test_creates_tables(self, tmp_db_path):
        import sqlite3

        analytics = DebateAnalytics(db_path=tmp_db_path)
        with sqlite3.connect(analytics.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
        assert "debate_records" in tables
        assert "agent_records" in tables
        assert "elo_records" in tables


class TestDebateAnalyticsRecordDebate:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_record_debate(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=120.5,
            agents=["claude", "gpt-4"],
            org_id="org1",
            user_id="user1",
            protocol="majority",
            total_messages=20,
            total_votes=6,
            total_cost=Decimal("0.50"),
        )

        import sqlite3

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM debate_records").fetchone()
        assert row["debate_id"] == "d1"
        assert row["rounds"] == 3
        assert row["consensus_reached"] == 1
        assert row["status"] == "completed"
        assert row["org_id"] == "org1"
        assert row["total_cost"] == "0.50"

    @pytest.mark.asyncio
    async def test_record_failed_debate(self, analytics):
        await analytics.record_debate(
            debate_id="d2",
            rounds=1,
            consensus_reached=False,
            duration_seconds=10.0,
            agents=["claude"],
            status="failed",
        )

        import sqlite3

        with sqlite3.connect(analytics.db_path) as conn:
            row = conn.execute("SELECT * FROM debate_records").fetchone()
        assert row[4] == "failed"  # status column
        assert row[14] is None  # completed_at is None for failed


class TestDebateAnalyticsRecordAgentActivity:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_record_agent_activity(self, analytics):
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=250.5,
            tokens_in=1000,
            tokens_out=500,
            cost=Decimal("0.05"),
            agent_name="Claude",
            provider="anthropic",
            model="claude-3",
        )

        import sqlite3

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM agent_records").fetchone()
        assert row["agent_id"] == "claude"
        assert row["response_time_ms"] == 250.5
        assert row["tokens_in"] == 1000
        assert row["provider"] == "anthropic"
        assert row["error"] == 0

    @pytest.mark.asyncio
    async def test_record_agent_error(self, analytics):
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=0,
            error=True,
        )

        import sqlite3

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM agent_records").fetchone()
        assert row["error"] == 1


class TestDebateAnalyticsRecordElo:
    @pytest.mark.asyncio
    async def test_record_elo_update(self, tmp_db_path):
        analytics = DebateAnalytics(db_path=tmp_db_path)
        await analytics.record_elo_update("claude", 1650.0, debate_id="d1")

        import sqlite3

        with sqlite3.connect(analytics.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM elo_records").fetchone()
        assert row["agent_id"] == "claude"
        assert row["elo_rating"] == 1650.0
        assert row["debate_id"] == "d1"


class TestDebateAnalyticsGetDebateStats:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_empty_stats(self, analytics):
        stats = await analytics.get_debate_stats()
        assert stats.total_debates == 0
        assert stats.completed_debates == 0
        assert stats.consensus_rate == 0.0
        assert stats.period_start is not None

    @pytest.mark.asyncio
    async def test_stats_with_data(self, analytics):
        # Record several debates
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a", "b"],
            status="completed",
            total_messages=10,
            total_votes=4,
        )
        await analytics.record_debate(
            debate_id="d2",
            rounds=5,
            consensus_reached=False,
            duration_seconds=200,
            agents=["a", "b", "c"],
            status="completed",
            total_messages=20,
            total_votes=6,
        )
        await analytics.record_debate(
            debate_id="d3",
            rounds=1,
            consensus_reached=False,
            duration_seconds=10,
            agents=["a"],
            status="failed",
            total_messages=2,
            total_votes=0,
        )

        stats = await analytics.get_debate_stats()
        assert stats.total_debates == 3
        assert stats.completed_debates == 2
        assert stats.failed_debates == 1
        assert stats.consensus_reached == 1
        assert stats.consensus_rate == 0.5  # 1 consensus / 2 completed
        assert stats.avg_rounds == pytest.approx(3.0, abs=0.01)
        assert stats.total_messages == 32
        assert stats.total_votes == 10

    @pytest.mark.asyncio
    async def test_stats_filtered_by_org(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a"],
            org_id="org1",
        )
        await analytics.record_debate(
            debate_id="d2",
            rounds=5,
            consensus_reached=False,
            duration_seconds=200,
            agents=["a"],
            org_id="org2",
        )

        stats = await analytics.get_debate_stats(org_id="org1")
        assert stats.total_debates == 1


class TestDebateAnalyticsGetAgentPerformance:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_no_data_returns_defaults(self, analytics):
        perf = await analytics.get_agent_performance("nonexistent")
        assert perf.agent_id == "nonexistent"
        assert perf.debates_participated == 0
        assert perf.current_elo == 1500.0

    @pytest.mark.asyncio
    async def test_with_data(self, analytics):
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=200.0,
            tokens_in=1000,
            tokens_out=500,
            cost=Decimal("0.10"),
            agent_name="Claude",
            provider="anthropic",
            model="claude-3",
        )
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d2",
            response_time_ms=300.0,
            tokens_in=2000,
            tokens_out=1000,
            cost=Decimal("0.20"),
            agent_name="Claude",
            provider="anthropic",
            model="claude-3",
        )
        await analytics.record_elo_update("claude", 1650.0)

        perf = await analytics.get_agent_performance("claude")
        assert perf.agent_id == "claude"
        assert perf.agent_name == "Claude"
        assert perf.provider == "anthropic"
        assert perf.debates_participated == 2
        assert perf.messages_sent == 2
        assert perf.avg_response_time_ms == pytest.approx(250.0, abs=0.1)
        assert perf.total_tokens_in == 3000
        assert perf.total_tokens_out == 1500
        assert perf.current_elo == 1650.0
        assert perf.error_count == 0
        assert perf.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_error_rate(self, analytics):
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d2",
            response_time_ms=0.0,
            error=True,
        )
        perf = await analytics.get_agent_performance("gpt")
        assert perf.error_count == 1
        assert perf.error_rate == pytest.approx(0.5, abs=0.01)


class TestDebateAnalyticsLeaderboard:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_empty_leaderboard(self, analytics):
        lb = await analytics.get_agent_leaderboard()
        assert lb == []

    @pytest.mark.asyncio
    async def test_leaderboard_sort_by_elo(self, analytics):
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=150.0,
        )
        await analytics.record_elo_update("claude", 1700.0)
        await analytics.record_elo_update("gpt", 1600.0)

        lb = await analytics.get_agent_leaderboard(sort_by="elo")
        assert len(lb) == 2
        assert lb[0].agent_id == "claude"
        assert lb[0].rank == 1
        assert lb[1].agent_id == "gpt"
        assert lb[1].rank == 2

    @pytest.mark.asyncio
    async def test_leaderboard_sort_by_messages(self, analytics):
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=150.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d2",
            response_time_ms=150.0,
        )

        lb = await analytics.get_agent_leaderboard(sort_by="messages")
        assert lb[0].agent_id == "gpt"
        assert lb[0].messages_sent == 2

    @pytest.mark.asyncio
    async def test_leaderboard_sort_by_debates(self, analytics):
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d2",
            response_time_ms=100.0,
        )

        lb = await analytics.get_agent_leaderboard(sort_by="debates")
        assert lb[0].agent_id == "gpt"

    @pytest.mark.asyncio
    async def test_leaderboard_limit(self, analytics):
        for i in range(5):
            await analytics.record_agent_activity(
                agent_id=f"agent{i}",
                debate_id="d1",
                response_time_ms=100.0,
            )
        lb = await analytics.get_agent_leaderboard(limit=3)
        assert len(lb) == 3


class TestDebateAnalyticsUsageTrends:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_empty_trends(self, analytics):
        trends = await analytics.get_usage_trends(DebateMetricType.DEBATE_COUNT)
        assert trends == []

    @pytest.mark.asyncio
    async def test_debate_count_trend(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a"],
        )
        trends = await analytics.get_usage_trends(
            DebateMetricType.DEBATE_COUNT,
            DebateTimeGranularity.DAILY,
        )
        # Should have at least one data point for today
        assert len(trends) >= 1
        assert trends[0].metric == DebateMetricType.DEBATE_COUNT
        assert trends[0].value >= 1.0

    @pytest.mark.asyncio
    async def test_trend_with_org_filter(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a"],
            org_id="org1",
        )
        await analytics.record_debate(
            debate_id="d2",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a"],
            org_id="org2",
        )
        trends = await analytics.get_usage_trends(
            DebateMetricType.DEBATE_COUNT,
            org_id="org1",
        )
        total = sum(t.value for t in trends)
        assert total == 1.0


class TestDebateAnalyticsCostBreakdown:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_empty_cost_breakdown(self, analytics):
        cb = await analytics.get_cost_breakdown()
        assert cb.total_cost == Decimal("0")
        assert cb.by_provider == {}
        assert cb.by_model == {}

    @pytest.mark.asyncio
    async def test_cost_breakdown_with_data(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["claude"],
            total_cost=Decimal("1.50"),
        )
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=200.0,
            cost=Decimal("1.50"),
            provider="anthropic",
            model="claude-3",
        )

        cb = await analytics.get_cost_breakdown()
        assert cb.total_cost == Decimal("1.5")
        assert "anthropic" in cb.by_provider
        assert "claude-3" in cb.by_model

    @pytest.mark.asyncio
    async def test_cost_breakdown_org_filter(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["claude"],
            total_cost=Decimal("1.50"),
            org_id="org1",
        )
        await analytics.record_debate(
            debate_id="d2",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["claude"],
            total_cost=Decimal("3.00"),
            org_id="org2",
        )

        cb = await analytics.get_cost_breakdown(org_id="org1")
        assert cb.total_cost == Decimal("1.5")


class TestDebateAnalyticsDashboardSummary:
    @pytest.fixture
    def analytics(self, tmp_db_path):
        return DebateAnalytics(db_path=tmp_db_path)

    @pytest.mark.asyncio
    async def test_empty_dashboard(self, analytics):
        summary = await analytics.get_dashboard_summary()
        assert summary.total_debates == 0
        assert summary.active_agents == 0
        assert summary.debate_stats is not None
        assert summary.cost_breakdown is not None
        assert summary.top_agents == []
        # Low consensus rate should trigger alert
        assert any(a["metric"] == "consensus_rate" for a in summary.alerts)

    @pytest.mark.asyncio
    async def test_dashboard_with_data(self, analytics):
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["claude", "gpt"],
            total_cost=Decimal("1.00"),
        )
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=200.0,
        )
        await analytics.record_agent_activity(
            agent_id="gpt",
            debate_id="d1",
            response_time_ms=300.0,
        )

        summary = await analytics.get_dashboard_summary()
        assert summary.total_debates == 1
        assert summary.active_agents == 2
        assert len(summary.top_agents) == 2

    @pytest.mark.asyncio
    async def test_dashboard_high_consensus_no_alert(self, analytics):
        # All debates reach consensus -> no low consensus alert
        for i in range(5):
            await analytics.record_debate(
                debate_id=f"d{i}",
                rounds=3,
                consensus_reached=True,
                duration_seconds=100,
                agents=["claude"],
            )
        summary = await analytics.get_dashboard_summary()
        assert not any(a.get("metric") == "consensus_rate" for a in summary.alerts)


# ===========================================================================
# Singleton / get_debate_analytics tests
# ===========================================================================


class TestGetDebateAnalytics:
    @pytest.fixture(autouse=True)
    def _reset_analytics_singleton(self):
        """Reset debate analytics singleton before/after each test."""
        import aragora.analytics.debate_analytics as mod

        mod._debate_analytics = None
        yield
        mod._debate_analytics = None

    def test_returns_instance(self):
        instance = get_debate_analytics()
        assert isinstance(instance, DebateAnalytics)

    def test_singleton_returns_same_instance(self):
        a = get_debate_analytics()
        b = get_debate_analytics()
        assert a is b


# ===========================================================================
# AnalyticsDashboard Tests (in-memory cache)
# ===========================================================================


class TestAnalyticsDashboardCache:
    def test_cache_miss(self):
        dashboard = AnalyticsDashboard()
        assert dashboard._get_cached("nonexistent") is None

    def test_cache_hit(self):
        dashboard = AnalyticsDashboard()
        dashboard._set_cached("key", "value")
        assert dashboard._get_cached("key") == "value"

    def test_cache_expiry(self):
        dashboard = AnalyticsDashboard()
        dashboard._set_cached("key", "value")
        # Manually expire
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        dashboard._cache["key"] = ("value", old_time)
        assert dashboard._get_cached("key") is None
        # Should have been removed
        assert "key" not in dashboard._cache


class TestAnalyticsDashboardBucketHelpers:
    def test_bucket_key_hourly(self):
        dashboard = AnalyticsDashboard()
        dt = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert dashboard._get_bucket_key(dt, Granularity.HOURLY) == "2025-06-15-14"

    def test_bucket_key_daily(self):
        dashboard = AnalyticsDashboard()
        dt = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert dashboard._get_bucket_key(dt, Granularity.DAILY) == "2025-06-15"

    def test_bucket_key_weekly(self):
        dashboard = AnalyticsDashboard()
        # 2025-06-15 is a Sunday, weekday()=6, so start of week is 2025-06-09
        dt = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = dashboard._get_bucket_key(dt, Granularity.WEEKLY)
        assert result == "2025-06-09"

    def test_bucket_key_monthly(self):
        dashboard = AnalyticsDashboard()
        dt = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert dashboard._get_bucket_key(dt, Granularity.MONTHLY) == "2025-06"

    def test_parse_bucket_key_hourly(self):
        dashboard = AnalyticsDashboard()
        result = dashboard._parse_bucket_key("2025-06-15-14", Granularity.HOURLY)
        assert result == datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)

    def test_parse_bucket_key_daily(self):
        dashboard = AnalyticsDashboard()
        result = dashboard._parse_bucket_key("2025-06-15", Granularity.DAILY)
        assert result == datetime(2025, 6, 15, tzinfo=timezone.utc)

    def test_parse_bucket_key_weekly(self):
        dashboard = AnalyticsDashboard()
        result = dashboard._parse_bucket_key("2025-06-09", Granularity.WEEKLY)
        assert result == datetime(2025, 6, 9, tzinfo=timezone.utc)

    def test_parse_bucket_key_monthly(self):
        dashboard = AnalyticsDashboard()
        result = dashboard._parse_bucket_key("2025-06", Granularity.MONTHLY)
        assert result == datetime(2025, 6, 1, tzinfo=timezone.utc)


class TestAnalyticsDashboardGetSummary:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_summary_with_no_findings(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            summary = await dashboard.get_summary("ws-1")
        assert summary.total_findings == 0
        assert summary.open_findings == 0
        assert summary.critical_findings == 0
        assert summary.finding_trend == "stable"
        assert summary.trend_percentage == 0

    @pytest.mark.asyncio
    async def test_summary_with_findings(self, dashboard):
        findings = [
            {
                "id": "f1",
                "severity": "critical",
                "status": "open",
                "category": "auth",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "f2",
                "severity": "high",
                "status": "open",
                "category": "auth",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "f3",
                "severity": "high",
                "status": "resolved",
                "category": "crypto",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        prev_findings = [
            {"id": "f0", "severity": "high", "status": "open", "category": "auth"},
        ]

        call_count = 0

        async def mock_get_findings(ws_id, time_range, offset=None):
            nonlocal call_count
            call_count += 1
            if offset is not None:
                return prev_findings
            return findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            summary = await dashboard.get_summary("ws-1")

        assert summary.total_findings == 3
        assert summary.open_findings == 2
        assert summary.critical_findings == 1
        assert summary.resolved_last_period == 1
        assert summary.finding_trend == "up"  # 3 vs 1 = 200% increase
        assert summary.trend_percentage == pytest.approx(200.0)
        assert len(summary.top_categories) == 2

    @pytest.mark.asyncio
    async def test_summary_caching(self, dashboard):
        mock = AsyncMock(return_value=[])
        with patch.object(dashboard, "_get_findings", mock):
            s1 = await dashboard.get_summary("ws-1")
            s2 = await dashboard.get_summary("ws-1")
        assert s1 is s2  # Same cached object
        # _get_findings called twice for first call (current + previous), 0 for cached
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_summary_trend_down(self, dashboard):
        findings = [{"id": "f1", "severity": "low", "status": "open", "category": "misc"}]
        prev_findings = [
            {"id": f"f{i}", "severity": "low", "status": "open", "category": "misc"}
            for i in range(10)
        ]

        async def mock_get_findings(ws_id, time_range, offset=None):
            if offset is not None:
                return prev_findings
            return findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            summary = await dashboard.get_summary("ws-1")
        assert summary.finding_trend == "down"

    @pytest.mark.asyncio
    async def test_summary_trend_stable(self, dashboard):
        findings = [{"id": "f1", "severity": "low", "status": "open", "category": "misc"}]
        # Same count -> 0% change -> stable
        prev = [{"id": "f1", "severity": "low", "status": "open", "category": "misc"}]

        async def mock_get_findings(ws_id, time_range, offset=None):
            return prev if offset else findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            summary = await dashboard.get_summary("ws-1")
        assert summary.finding_trend == "stable"

    @pytest.mark.asyncio
    async def test_summary_all_time_trend(self, dashboard):
        """ALL_TIME has no timedelta, so trend should be stable."""
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            summary = await dashboard.get_summary("ws-1", time_range=TimeRange.ALL_TIME)
        assert summary.finding_trend == "stable"
        assert summary.trend_percentage == 0

    @pytest.mark.asyncio
    async def test_summary_new_findings_no_previous(self, dashboard):
        """When previous period has 0 findings but current has some."""
        findings = [
            {"id": "f1", "severity": "high", "status": "open", "category": "auth"},
        ]

        async def mock_get_findings(ws_id, time_range, offset=None):
            return [] if offset else findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            summary = await dashboard.get_summary("ws-1")
        assert summary.trend_percentage == 100.0
        assert summary.finding_trend == "up"


class TestAnalyticsDashboardFindingTrends:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_empty_trends(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            trends = await dashboard.get_finding_trends("ws-1")
        assert trends == []

    @pytest.mark.asyncio
    async def test_trends_bucketing(self, dashboard):
        now = datetime.now(timezone.utc)
        findings = [
            {
                "id": "f1",
                "severity": "critical",
                "status": "open",
                "category": "auth",
                "created_at": now.isoformat(),
            },
            {
                "id": "f2",
                "severity": "high",
                "status": "open",
                "category": "crypto",
                "created_at": now.isoformat(),
            },
        ]

        with patch.object(
            dashboard, "_get_findings", new_callable=AsyncMock, return_value=findings
        ):
            trends = await dashboard.get_finding_trends("ws-1", granularity=Granularity.DAILY)

        assert len(trends) >= 1
        assert trends[0].total == 2
        assert trends[0].by_severity.get("critical") == 1
        assert trends[0].by_severity.get("high") == 1
        assert trends[0].by_category.get("auth") == 1

    @pytest.mark.asyncio
    async def test_trends_missing_created_at(self, dashboard):
        findings = [{"id": "f1", "severity": "high", "status": "open", "category": "auth"}]
        with patch.object(
            dashboard, "_get_findings", new_callable=AsyncMock, return_value=findings
        ):
            trends = await dashboard.get_finding_trends("ws-1")
        # Finding without created_at should be skipped
        assert trends == []

    @pytest.mark.asyncio
    async def test_trends_string_datetime(self, dashboard):
        """Test that string datetimes are parsed correctly."""
        findings = [
            {
                "id": "f1",
                "severity": "high",
                "status": "open",
                "category": "auth",
                "created_at": "2025-06-15T10:00:00+00:00",
            },
        ]
        with patch.object(
            dashboard, "_get_findings", new_callable=AsyncMock, return_value=findings
        ):
            trends = await dashboard.get_finding_trends("ws-1", granularity=Granularity.DAILY)
        assert len(trends) == 1


class TestAnalyticsDashboardRemediationMetrics:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_empty_remediation(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            metrics = await dashboard.get_remediation_metrics("ws-1")
        assert metrics.total_resolved == 0
        assert metrics.total_open == 0
        assert metrics.mttr_hours == 0
        assert metrics.false_positive_rate == 0.0
        assert metrics.accepted_risk_rate == 0.0

    @pytest.mark.asyncio
    async def test_remediation_with_data(self, dashboard):
        now = datetime.now(timezone.utc)
        findings = [
            {
                "id": "f1",
                "status": "resolved",
                "severity": "critical",
                "created_at": (now - timedelta(hours=48)).isoformat(),
                "resolved_at": now.isoformat(),
            },
            {
                "id": "f2",
                "status": "resolved",
                "severity": "high",
                "created_at": (now - timedelta(hours=24)).isoformat(),
                "resolved_at": now.isoformat(),
            },
            {"id": "f3", "status": "open", "severity": "high"},
            {"id": "f4", "status": "false_positive", "severity": "low"},
            {"id": "f5", "status": "accepted_risk", "severity": "medium"},
        ]

        with patch.object(
            dashboard, "_get_findings", new_callable=AsyncMock, return_value=findings
        ):
            metrics = await dashboard.get_remediation_metrics("ws-1")

        assert metrics.total_resolved == 2
        assert metrics.total_open == 1
        assert metrics.mttr_hours == pytest.approx(36.0, abs=0.1)
        assert "critical" in metrics.mttr_by_severity
        assert metrics.mttr_by_severity["critical"] == pytest.approx(48.0, abs=0.1)
        assert metrics.false_positive_rate == pytest.approx(0.2, abs=0.01)
        assert metrics.accepted_risk_rate == pytest.approx(0.2, abs=0.01)

    @pytest.mark.asyncio
    async def test_remediation_zero_division(self, dashboard):
        """No findings -> zero rates, no division error."""
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            metrics = await dashboard.get_remediation_metrics("ws-1")
        assert metrics.false_positive_rate == 0.0


class TestAnalyticsDashboardAgentMetrics:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_empty_agent_metrics(self, dashboard):
        with patch.object(dashboard, "_get_sessions", new_callable=AsyncMock, return_value=[]):
            metrics = await dashboard.get_agent_metrics("ws-1")
        assert metrics == []

    @pytest.mark.asyncio
    async def test_agent_metrics_with_data(self, dashboard):
        sessions = [
            {
                "id": "s1",
                "agent_results": [
                    {
                        "agent_id": "claude",
                        "agent_name": "Claude",
                        "findings": [
                            {"severity": "critical", "status": "open"},
                            {"severity": "high", "status": "false_positive"},
                        ],
                        "agreed_with_consensus": True,
                        "response_time_ms": 200.0,
                    },
                    {
                        "agent_id": "gpt",
                        "agent_name": "GPT-4",
                        "findings": [
                            {"severity": "critical", "status": "open"},
                        ],
                        "agreed_with_consensus": False,
                        "response_time_ms": 300.0,
                    },
                ],
            },
        ]

        with patch.object(
            dashboard, "_get_sessions", new_callable=AsyncMock, return_value=sessions
        ):
            metrics = await dashboard.get_agent_metrics("ws-1")

        assert len(metrics) == 2
        # Sorted by total_findings descending
        assert metrics[0].agent_id == "claude"
        assert metrics[0].total_findings == 2
        assert metrics[0].agreement_rate == 1.0
        assert metrics[0].precision == 0.5  # 1 TP / 2 total
        assert metrics[0].avg_response_time_ms == 200.0

        assert metrics[1].agent_id == "gpt"
        assert metrics[1].agreement_rate == 0.0
        assert metrics[1].precision == 1.0  # no false positives

    @pytest.mark.asyncio
    async def test_agent_metrics_zero_decisions(self, dashboard):
        """Agent with no agreement data should have 0 agreement rate."""
        sessions = [
            {
                "id": "s1",
                "agent_results": [
                    {
                        "agent_id": "agent1",
                        "agent_name": "Agent1",
                        "findings": [],
                        # No agreed_with_consensus key
                    },
                ],
            },
        ]
        with patch.object(
            dashboard, "_get_sessions", new_callable=AsyncMock, return_value=sessions
        ):
            metrics = await dashboard.get_agent_metrics("ws-1")
        assert len(metrics) == 1
        assert metrics[0].agreement_rate == 0.0
        assert metrics[0].precision == 0.0


class TestAnalyticsDashboardCostMetrics:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_empty_cost_metrics(self, dashboard):
        with patch.object(dashboard, "_get_sessions", new_callable=AsyncMock, return_value=[]):
            metrics = await dashboard.get_cost_metrics("ws-1")
        assert metrics.total_audits == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.avg_cost_per_audit == 0.0

    @pytest.mark.asyncio
    async def test_cost_metrics_with_data(self, dashboard):
        sessions = [
            {
                "id": "s1",
                "cost_usd": 1.50,
                "audit_type": "security",
                "token_usage": {"input_tokens": 1000, "output_tokens": 500},
            },
            {
                "id": "s2",
                "cost_usd": 2.50,
                "audit_type": "compliance",
                "token_usage": {"input_tokens": 2000, "output_tokens": 1000},
            },
        ]
        with patch.object(
            dashboard, "_get_sessions", new_callable=AsyncMock, return_value=sessions
        ):
            metrics = await dashboard.get_cost_metrics("ws-1")

        assert metrics.total_audits == 2
        assert metrics.total_cost_usd == pytest.approx(4.0)
        assert metrics.avg_cost_per_audit == pytest.approx(2.0)
        assert metrics.cost_by_type["security"] == pytest.approx(1.5)
        assert metrics.cost_by_type["compliance"] == pytest.approx(2.5)
        assert metrics.token_usage["input"] == 3000
        assert metrics.token_usage["output"] == 1500


class TestAnalyticsDashboardComplianceScorecard:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_default_frameworks(self, dashboard):
        with patch.object(
            dashboard,
            "_calculate_compliance_score",
            new_callable=AsyncMock,
            return_value=ComplianceScore(
                framework="SOC2",
                score=1.0,
                passing_controls=4,
                failing_controls=0,
                not_applicable=0,
            ),
        ):
            scores = await dashboard.get_compliance_scorecard("ws-1")
        assert len(scores) == 4  # SOC2, GDPR, HIPAA, PCI-DSS

    @pytest.mark.asyncio
    async def test_custom_frameworks(self, dashboard):
        with patch.object(
            dashboard,
            "_calculate_compliance_score",
            new_callable=AsyncMock,
            return_value=ComplianceScore(
                framework="SOC2",
                score=1.0,
                passing_controls=4,
                failing_controls=0,
                not_applicable=0,
            ),
        ):
            scores = await dashboard.get_compliance_scorecard("ws-1", frameworks=["SOC2"])
        assert len(scores) == 1

    @pytest.mark.asyncio
    async def test_compliance_with_no_findings(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            score = await dashboard._calculate_compliance_score("ws-1", "SOC2")
        # All controls pass when there are no findings
        assert score.framework == "SOC2"
        assert score.passing_controls == 4
        assert score.failing_controls == 0
        assert score.score == 1.0

    @pytest.mark.asyncio
    async def test_compliance_with_critical_gaps(self, dashboard):
        findings = [
            {"id": "f1", "severity": "critical", "status": "open", "category": "authentication"},
        ]
        with patch.object(
            dashboard, "_get_findings", new_callable=AsyncMock, return_value=findings
        ):
            score = await dashboard._calculate_compliance_score("ws-1", "SOC2")
        assert score.failing_controls >= 1
        assert "access_control" in score.critical_gaps
        assert score.score < 1.0

    @pytest.mark.asyncio
    async def test_compliance_unknown_framework(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            score = await dashboard._calculate_compliance_score("ws-1", "UNKNOWN")
        assert score.score == 0
        assert score.passing_controls == 0
        assert score.failing_controls == 0


class TestAnalyticsDashboardRiskHeatmap:
    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_empty_heatmap(self, dashboard):
        with patch.object(dashboard, "_get_findings", new_callable=AsyncMock, return_value=[]):
            cells = await dashboard.get_risk_heatmap("ws-1")
        assert cells == []

    @pytest.mark.asyncio
    async def test_heatmap_trends(self, dashboard):
        current_findings = [
            {"id": "f1", "category": "auth", "severity": "critical"},
            {"id": "f2", "category": "auth", "severity": "critical"},
            {"id": "f3", "category": "crypto", "severity": "high"},
        ]
        prev_findings = [
            {"id": "f4", "category": "auth", "severity": "critical"},
            {"id": "f5", "category": "crypto", "severity": "high"},
            {"id": "f6", "category": "crypto", "severity": "high"},
        ]

        call_count = 0

        async def mock_get_findings(ws_id, time_range, offset=None):
            nonlocal call_count
            call_count += 1
            if offset is not None:
                return prev_findings
            return current_findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            cells = await dashboard.get_risk_heatmap("ws-1")

        auth_critical = next(c for c in cells if c.category == "auth" and c.severity == "critical")
        assert auth_critical.count == 2
        assert auth_critical.trend == "up"  # 2 > 1

        crypto_high = next(c for c in cells if c.category == "crypto" and c.severity == "high")
        assert crypto_high.count == 1
        assert crypto_high.trend == "down"  # 1 < 2

    @pytest.mark.asyncio
    async def test_heatmap_stable_trend(self, dashboard):
        findings = [{"id": "f1", "category": "auth", "severity": "high"}]

        async def mock_get_findings(ws_id, time_range, offset=None):
            return findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            cells = await dashboard.get_risk_heatmap("ws-1")

        assert cells[0].trend == "stable"

    @pytest.mark.asyncio
    async def test_heatmap_all_time_no_previous(self, dashboard):
        findings = [{"id": "f1", "category": "auth", "severity": "high"}]

        async def mock_get_findings(ws_id, time_range, offset=None):
            return [] if offset else findings

        with patch.object(dashboard, "_get_findings", side_effect=mock_get_findings):
            cells = await dashboard.get_risk_heatmap("ws-1", time_range=TimeRange.ALL_TIME)
        # ALL_TIME has delta=None, so prev_findings will be []
        assert cells[0].trend == "up"


class TestAnalyticsDashboardGetFindings:
    """Test _get_findings when the import fails (no aragora.audit module)."""

    @pytest.fixture
    def dashboard(self):
        return AnalyticsDashboard()

    @pytest.mark.asyncio
    async def test_get_findings_import_failure(self, dashboard):
        """When aragora.audit is unavailable, return empty list."""
        with patch.dict("sys.modules", {"aragora.audit": None}):
            result = await dashboard._get_findings("ws-1", TimeRange.LAST_30_DAYS)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_sessions_import_failure(self, dashboard):
        """When aragora.audit is unavailable, return empty list."""
        with patch.dict("sys.modules", {"aragora.audit": None}):
            result = await dashboard._get_sessions("ws-1", TimeRange.LAST_30_DAYS)
        assert result == []


# ===========================================================================
# Singleton / get_analytics_dashboard tests
# ===========================================================================


class TestGetAnalyticsDashboard:
    def test_returns_instance(self):
        import aragora.analytics.dashboard as mod

        mod._dashboard = None
        instance = get_analytics_dashboard()
        assert isinstance(instance, AnalyticsDashboard)

    def test_singleton_returns_same_instance(self):
        import aragora.analytics.dashboard as mod

        mod._dashboard = None
        a = get_analytics_dashboard()
        b = get_analytics_dashboard()
        assert a is b
        # Clean up
        mod._dashboard = None


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_debate_analytics_multiple_elo_updates(self, tmp_db_path):
        """Latest ELO should be used."""
        analytics = DebateAnalytics(db_path=tmp_db_path)
        await analytics.record_agent_activity(
            agent_id="claude",
            debate_id="d1",
            response_time_ms=100.0,
        )
        await analytics.record_elo_update("claude", 1500.0)
        await analytics.record_elo_update("claude", 1600.0)
        await analytics.record_elo_update("claude", 1550.0)

        perf = await analytics.get_agent_performance("claude")
        assert perf.current_elo == 1550.0

    @pytest.mark.asyncio
    async def test_debate_stats_zero_completed_no_division_error(self, tmp_db_path):
        """When no debates are completed, consensus_rate should be 0."""
        analytics = DebateAnalytics(db_path=tmp_db_path)
        await analytics.record_debate(
            debate_id="d1",
            rounds=1,
            consensus_reached=False,
            duration_seconds=10.0,
            agents=["a"],
            status="failed",
        )
        stats = await analytics.get_debate_stats()
        assert stats.consensus_rate == 0.0

    @pytest.mark.asyncio
    async def test_agent_performance_zero_messages_no_division_error(self, tmp_db_path):
        """Agent with no records should not cause division error."""
        analytics = DebateAnalytics(db_path=tmp_db_path)
        perf = await analytics.get_agent_performance("ghost")
        assert perf.error_rate == 0.0
        assert perf.avg_cost_per_debate == Decimal("0")

    @pytest.mark.asyncio
    async def test_cost_breakdown_no_agent_records(self, tmp_db_path):
        """Cost breakdown with debates but no agent records."""
        analytics = DebateAnalytics(db_path=tmp_db_path)
        await analytics.record_debate(
            debate_id="d1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=100,
            agents=["a"],
            total_cost=Decimal("5.00"),
        )
        cb = await analytics.get_cost_breakdown()
        assert cb.total_cost == Decimal("5.0")
        assert cb.by_provider == {}

    def test_remediation_metrics_to_dict_rounding(self):
        rm = RemediationMetrics(
            total_resolved=1,
            total_open=1,
            mttr_hours=0.0,
            false_positive_rate=0.0,
            accepted_risk_rate=0.0,
        )
        d = rm.to_dict()
        assert d["mttr_hours"] == 0.0
        assert d["false_positive_rate"] == 0.0

    def test_usage_trend_granularities(self):
        """Ensure all granularity strings are covered."""
        for gran in DebateTimeGranularity:
            assert isinstance(gran.value, str)

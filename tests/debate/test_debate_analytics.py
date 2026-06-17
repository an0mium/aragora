"""
Tests for aragora.analytics.debate_analytics module.

Tests cover:
- Enum values
- Dataclass creation and defaults
- to_dict serialization methods
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from aragora.analytics.debate_analytics import (
    AgentPerformance,
    CostBreakdown,
    DebateMetricType,
    DebateStats,
    DebateTimeGranularity,
    UsageTrendPoint,
)


class TestDebateTimeGranularity:
    """Tests for DebateTimeGranularity enum."""

    def test_all_values(self):
        """Test all granularity values exist."""
        assert DebateTimeGranularity.HOURLY.value == "hourly"
        assert DebateTimeGranularity.DAILY.value == "daily"
        assert DebateTimeGranularity.WEEKLY.value == "weekly"
        assert DebateTimeGranularity.MONTHLY.value == "monthly"

    def test_count(self):
        """Test there are exactly 4 granularities."""
        assert len(DebateTimeGranularity) == 4


class TestDebateMetricType:
    """Tests for DebateMetricType enum."""

    def test_all_values(self):
        """Test all metric types exist."""
        assert DebateMetricType.DEBATE_COUNT.value == "debate_count"
        assert DebateMetricType.CONSENSUS_RATE.value == "consensus_rate"
        assert DebateMetricType.AVG_ROUNDS.value == "avg_rounds"
        assert DebateMetricType.AVG_DURATION.value == "avg_duration"
        assert DebateMetricType.AGENT_RESPONSE_TIME.value == "agent_response_time"
        assert DebateMetricType.AGENT_ACCURACY.value == "agent_accuracy"
        assert DebateMetricType.TOKEN_USAGE.value == "token_usage"
        assert DebateMetricType.COST_TOTAL.value == "cost_total"
        assert DebateMetricType.USER_ACTIVITY.value == "user_activity"
        assert DebateMetricType.ERROR_RATE.value == "error_rate"

    def test_count(self):
        """Test there are exactly 10 metric types."""
        assert len(DebateMetricType) == 10


class TestDebateStats:
    """Tests for DebateStats dataclass."""

    def test_defaults(self):
        """Test default values."""
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

    def test_with_values(self):
        """Test with provided values."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        stats = DebateStats(
            total_debates=100,
            completed_debates=95,
            failed_debates=5,
            consensus_reached=80,
            consensus_rate=0.8421,
            avg_rounds=3.5,
            avg_duration_seconds=45.7,
            avg_agents_per_debate=3.2,
            total_messages=500,
            total_votes=300,
            period_start=yesterday,
            period_end=now,
            by_protocol={"majority": 80, "unanimous": 20},
        )

        assert stats.total_debates == 100
        assert stats.consensus_rate == 0.8421
        assert stats.by_protocol["majority"] == 80

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now(timezone.utc)
        stats = DebateStats(
            total_debates=50,
            consensus_rate=0.84567,
            avg_rounds=3.456,
            period_start=now,
        )

        result = stats.to_dict()

        assert result["total_debates"] == 50
        assert result["consensus_rate"] == 0.8457  # Rounded to 4 decimals
        assert result["avg_rounds"] == 3.46  # Rounded to 2 decimals
        assert result["period_start"] == now.isoformat()
        assert result["period_end"] is None

    def test_to_dict_with_protocol_breakdown(self):
        """Test to_dict with protocol breakdown."""
        stats = DebateStats(
            by_protocol={"majority": 60, "unanimous": 30, "hybrid": 10},
        )

        result = stats.to_dict()

        assert result["by_protocol"] == {"majority": 60, "unanimous": 30, "hybrid": 10}


class TestAgentPerformance:
    """Tests for AgentPerformance dataclass."""

    def test_defaults(self):
        """Test default values."""
        perf = AgentPerformance()
        assert perf.agent_id == ""
        assert perf.agent_name == ""
        assert perf.provider == ""
        assert perf.model == ""
        assert perf.debates_participated == 0
        assert perf.messages_sent == 0
        assert perf.avg_response_time_ms == 0.0
        assert perf.p95_response_time_ms == 0.0
        assert perf.p99_response_time_ms == 0.0
        assert perf.error_count == 0
        assert perf.error_rate == 0.0
        assert perf.votes_received == 0
        assert perf.positive_votes == 0
        assert perf.vote_ratio == 0.0
        assert perf.consensus_contributions == 0
        assert perf.total_tokens_in == 0
        assert perf.total_tokens_out == 0
        assert perf.total_cost == Decimal("0")
        assert perf.avg_cost_per_debate == Decimal("0")
        assert perf.current_elo == 1500.0
        assert perf.elo_change_period == 0.0
        assert perf.rank == 0

    def test_with_values(self):
        """Test with provided values."""
        perf = AgentPerformance(
            agent_id="claude-001",
            agent_name="Claude",
            provider="anthropic",
            model="claude-3-opus",
            debates_participated=50,
            messages_sent=200,
            avg_response_time_ms=1500.5,
            error_rate=0.02,
            current_elo=1650.5,
            total_cost=Decimal("25.50"),
        )

        assert perf.agent_id == "claude-001"
        assert perf.current_elo == 1650.5
        assert perf.total_cost == Decimal("25.50")

    def test_to_dict(self):
        """Test to_dict serialization."""
        perf = AgentPerformance(
            agent_id="gpt-001",
            agent_name="GPT-4",
            provider="openai",
            avg_response_time_ms=1234.567,
            error_rate=0.01234,
            vote_ratio=0.85678,
            current_elo=1650.123,
            total_cost=Decimal("10.50"),
            avg_cost_per_debate=Decimal("0.21"),
        )

        result = perf.to_dict()

        assert result["agent_id"] == "gpt-001"
        assert result["avg_response_time_ms"] == 1234.57  # Rounded to 2 decimals
        assert result["error_rate"] == 0.0123  # Rounded to 4 decimals
        assert result["vote_ratio"] == 0.8568  # Rounded to 4 decimals
        assert result["current_elo"] == 1650.12  # Rounded to 2 decimals
        assert result["total_cost"] == "10.50"  # String representation
        assert result["avg_cost_per_debate"] == "0.21"


class TestUsageTrendPoint:
    """Tests for UsageTrendPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        now = datetime.now(timezone.utc)
        point = UsageTrendPoint(
            timestamp=now,
            value=42.5,
            metric=DebateMetricType.DEBATE_COUNT,
        )

        assert point.timestamp == now
        assert point.value == 42.5
        assert point.metric == DebateMetricType.DEBATE_COUNT

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now(timezone.utc)
        point = UsageTrendPoint(
            timestamp=now,
            value=0.85,
            metric=DebateMetricType.CONSENSUS_RATE,
        )

        result = point.to_dict()

        assert result["timestamp"] == now.isoformat()
        assert result["value"] == 0.85
        assert result["metric"] == "consensus_rate"


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_creation_with_required_fields(self):
        """Test creation with required period fields."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        breakdown = CostBreakdown(
            period_start=yesterday,
            period_end=now,
        )

        assert breakdown.period_start == yesterday
        assert breakdown.period_end == now
        assert breakdown.total_cost == Decimal("0")
        assert breakdown.by_provider == {}
        assert breakdown.by_model == {}
        assert breakdown.by_user == {}
        assert breakdown.by_org == {}
        assert breakdown.daily_costs == []
        assert breakdown.projected_monthly == Decimal("0")
        assert breakdown.cost_per_debate == Decimal("0")
        assert breakdown.cost_per_consensus == Decimal("0")

    def test_with_full_data(self):
        """Test with complete breakdown data."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        breakdown = CostBreakdown(
            period_start=yesterday,
            period_end=now,
            total_cost=Decimal("100.50"),
            by_provider={"anthropic": Decimal("60.00"), "openai": Decimal("40.50")},
            by_model={"claude-3-opus": Decimal("60.00"), "gpt-4": Decimal("40.50")},
            by_user={"user-1": Decimal("50.25"), "user-2": Decimal("50.25")},
            by_org={"org-1": Decimal("100.50")},
            daily_costs=[("2024-01-01", Decimal("50.25")), ("2024-01-02", Decimal("50.25"))],
            projected_monthly=Decimal("3015.00"),
            cost_per_debate=Decimal("1.005"),
            cost_per_consensus=Decimal("1.25"),
        )

        assert breakdown.total_cost == Decimal("100.50")
        assert breakdown.by_provider["anthropic"] == Decimal("60.00")
        assert len(breakdown.daily_costs) == 2

    def test_to_dict(self):
        """Test to_dict serialization."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        breakdown = CostBreakdown(
            period_start=yesterday,
            period_end=now,
            total_cost=Decimal("100.50"),
            by_provider={"anthropic": Decimal("60.00")},
            daily_costs=[("2024-01-01", Decimal("100.50"))],
            projected_monthly=Decimal("3015.00"),
        )

        result = breakdown.to_dict()

        assert result["period_start"] == yesterday.isoformat()
        assert result["period_end"] == now.isoformat()
        assert result["total_cost"] == "100.50"
        assert result["by_provider"] == {"anthropic": "60.00"}
        assert result["daily_costs"] == [("2024-01-01", "100.50")]
        assert result["projected_monthly"] == "3015.00"

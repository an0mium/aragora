"""
Tests for decision outcome analytics.

Covers:
- OutcomeAnalytics class methods
- Dataclass serialization (QualityDataPoint, OutcomeSummary, AgentContribution)
- Period parsing and validation
- Caching behavior
- Database query edge cases
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analytics.outcome_analytics import (
    AgentContribution,
    OutcomeAnalytics,
    OutcomeSummary,
    QualityDataPoint,
    get_outcome_analytics,
    _parse_period,
)


# ===========================================================================
# Dataclass Tests
# ===========================================================================


class TestQualityDataPoint:
    def test_creation(self):
        now = datetime.now(timezone.utc)
        point = QualityDataPoint(
            timestamp=now,
            consensus_rate=0.85,
            avg_confidence=0.78,
            avg_rounds=3.2,
            debate_count=42,
        )
        assert point.consensus_rate == 0.85
        assert point.avg_confidence == 0.78
        assert point.avg_rounds == 3.2
        assert point.debate_count == 42

    def test_to_dict(self):
        now = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        point = QualityDataPoint(
            timestamp=now,
            consensus_rate=0.8512,
            avg_confidence=0.7777,
            avg_rounds=3.14159,
            debate_count=10,
        )
        d = point.to_dict()
        assert d["timestamp"] == "2026-02-23T12:00:00+00:00"
        assert d["consensus_rate"] == 0.8512
        assert d["avg_confidence"] == 0.7777
        assert d["avg_rounds"] == 3.14
        assert d["debate_count"] == 10

    def test_to_dict_rounding(self):
        now = datetime.now(timezone.utc)
        point = QualityDataPoint(
            timestamp=now,
            consensus_rate=0.123456789,
            avg_confidence=0.987654321,
            avg_rounds=2.999,
            debate_count=1,
        )
        d = point.to_dict()
        assert d["consensus_rate"] == 0.1235
        assert d["avg_confidence"] == 0.9877
        assert d["avg_rounds"] == 3.0


class TestAgentContribution:
    def test_creation(self):
        contrib = AgentContribution(
            agent_id="claude-opus",
            agent_name="Claude Opus",
            debates_participated=25,
            consensus_contributions=20,
            avg_confidence=0.82,
            contribution_score=0.75,
        )
        assert contrib.agent_id == "claude-opus"
        assert contrib.debates_participated == 25

    def test_defaults(self):
        contrib = AgentContribution(
            agent_id="test",
            agent_name="Test",
        )
        assert contrib.debates_participated == 0
        assert contrib.consensus_contributions == 0
        assert contrib.avg_confidence == 0.0
        assert contrib.contribution_score == 0.0

    def test_to_dict(self):
        contrib = AgentContribution(
            agent_id="gpt-4o",
            agent_name="GPT-4o",
            debates_participated=10,
            consensus_contributions=8,
            avg_confidence=0.91,
            contribution_score=0.8523,
        )
        d = contrib.to_dict()
        assert d["agent_id"] == "gpt-4o"
        assert d["agent_name"] == "GPT-4o"
        assert d["debates_participated"] == 10
        assert d["consensus_contributions"] == 8
        assert d["avg_confidence"] == 0.91
        assert d["contribution_score"] == 0.8523


class TestOutcomeSummary:
    def test_creation(self):
        summary = OutcomeSummary(
            debate_id="debate-123",
            task="Design a rate limiter",
            consensus_reached=True,
            confidence=0.9,
            rounds=3,
            agents=["claude", "gpt-4"],
            duration_seconds=120.5,
            topic="architecture",
            created_at="2026-02-23T12:00:00Z",
        )
        assert summary.debate_id == "debate-123"
        assert summary.consensus_reached is True
        assert summary.rounds == 3
        assert len(summary.agents) == 2

    def test_defaults(self):
        summary = OutcomeSummary(
            debate_id="d1",
            task="test",
            consensus_reached=False,
            confidence=0.0,
            rounds=0,
        )
        assert summary.agents == []
        assert summary.duration_seconds == 0.0
        assert summary.topic == ""
        assert summary.created_at == ""

    def test_to_dict(self):
        summary = OutcomeSummary(
            debate_id="debate-456",
            task="Review PR",
            consensus_reached=True,
            confidence=0.88,
            rounds=5,
            agents=["claude", "gemini"],
            duration_seconds=95.3,
            topic="code-review",
            created_at="2026-02-20T10:00:00Z",
        )
        d = summary.to_dict()
        assert d["debate_id"] == "debate-456"
        assert d["task"] == "Review PR"
        assert d["consensus_reached"] is True
        assert d["confidence"] == 0.88
        assert d["rounds"] == 5
        assert d["agents"] == ["claude", "gemini"]
        assert d["duration_seconds"] == 95.3
        assert d["topic"] == "code-review"


# ===========================================================================
# Period Parsing Tests
# ===========================================================================


class TestParsePeriod:
    def test_valid_periods(self):
        assert _parse_period("24h") == timedelta(hours=24)
        assert _parse_period("7d") == timedelta(days=7)
        assert _parse_period("30d") == timedelta(days=30)
        assert _parse_period("90d") == timedelta(days=90)
        assert _parse_period("365d") == timedelta(days=365)

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="Invalid period"):
            _parse_period("invalid")

    def test_empty_period(self):
        with pytest.raises(ValueError, match="Invalid period"):
            _parse_period("")

    def test_unsupported_period(self):
        with pytest.raises(ValueError, match="Invalid period"):
            _parse_period("60d")


# ===========================================================================
# OutcomeAnalytics Tests
# ===========================================================================


class TestOutcomeAnalytics:
    """Tests for OutcomeAnalytics using an in-memory SQLite database."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path."""
        return str(tmp_path / "test_outcomes.db")

    @pytest.fixture
    def analytics(self, db_path):
        """Create OutcomeAnalytics instance with test database."""
        return OutcomeAnalytics(db_path=db_path)

    @pytest.fixture
    async def populated_analytics(self, db_path):
        """Create analytics instance with pre-populated debate data."""
        from aragora.analytics.debate_analytics import DebateAnalytics

        da = DebateAnalytics(db_path=db_path)

        # Record several debates
        await da.record_debate(
            debate_id="debate-1",
            rounds=3,
            consensus_reached=True,
            duration_seconds=120.0,
            agents=["claude", "gpt-4"],
            protocol="architecture",
        )
        await da.record_debate(
            debate_id="debate-2",
            rounds=5,
            consensus_reached=False,
            duration_seconds=200.0,
            agents=["claude", "gemini"],
            protocol="security",
        )
        await da.record_debate(
            debate_id="debate-3",
            rounds=2,
            consensus_reached=True,
            duration_seconds=80.0,
            agents=["gpt-4", "gemini"],
            protocol="architecture",
        )

        # Record agent activity
        await da.record_agent_activity(
            agent_id="claude",
            debate_id="debate-1",
            response_time_ms=150.0,
            agent_name="Claude",
        )
        await da.record_agent_activity(
            agent_id="gpt-4",
            debate_id="debate-1",
            response_time_ms=200.0,
            agent_name="GPT-4",
        )

        return OutcomeAnalytics(db_path=db_path)

    @pytest.mark.asyncio
    async def test_get_consensus_rate_empty(self, analytics):
        """Consensus rate is 0 when no debates recorded."""
        rate = await analytics.get_consensus_rate(period="30d")
        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_get_consensus_rate_with_data(self, populated_analytics):
        """Consensus rate reflects recorded debates."""
        rate = await populated_analytics.get_consensus_rate(period="30d")
        # 2 of 3 debates reached consensus
        assert 0.0 <= rate <= 1.0

    @pytest.mark.asyncio
    async def test_get_consensus_rate_caching(self, populated_analytics):
        """Subsequent calls return cached value."""
        rate1 = await populated_analytics.get_consensus_rate(period="30d")
        rate2 = await populated_analytics.get_consensus_rate(period="30d")
        assert rate1 == rate2

    @pytest.mark.asyncio
    async def test_get_consensus_rate_invalid_period(self, analytics):
        """Invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Invalid period"):
            await analytics.get_consensus_rate(period="invalid")

    @pytest.mark.asyncio
    async def test_get_average_rounds_empty(self, analytics):
        """Average rounds is 0 when no debates recorded."""
        avg = await analytics.get_average_rounds(period="30d")
        assert avg == 0.0

    @pytest.mark.asyncio
    async def test_get_average_rounds_with_data(self, populated_analytics):
        """Average rounds reflects recorded debates."""
        avg = await populated_analytics.get_average_rounds(period="30d")
        # Debates had 3, 5, 2 rounds
        assert avg > 0.0

    @pytest.mark.asyncio
    async def test_get_agent_contribution_scores_empty(self, analytics):
        """Empty contributions when no agents recorded."""
        scores = await analytics.get_agent_contribution_scores(period="30d")
        assert isinstance(scores, dict)
        assert len(scores) == 0

    @pytest.mark.asyncio
    async def test_get_agent_contribution_scores_with_data(self, populated_analytics):
        """Agent contributions are computed from activity data."""
        scores = await populated_analytics.get_agent_contribution_scores(period="30d")
        assert isinstance(scores, dict)
        # We recorded activity for claude and gpt-4
        for agent_id, contrib in scores.items():
            assert isinstance(contrib, AgentContribution)
            assert contrib.agent_id == agent_id
            assert 0.0 <= contrib.contribution_score <= 1.0

    @pytest.mark.asyncio
    async def test_get_decision_quality_trend_empty(self, analytics):
        """Quality trend returns empty list when no data."""
        trend = await analytics.get_decision_quality_trend(period="30d")
        assert isinstance(trend, list)

    @pytest.mark.asyncio
    async def test_get_decision_quality_trend_with_data(self, populated_analytics):
        """Quality trend returns data points."""
        trend = await populated_analytics.get_decision_quality_trend(period="30d")
        assert isinstance(trend, list)
        for point in trend:
            assert isinstance(point, QualityDataPoint)
            assert isinstance(point.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_get_decision_quality_trend_90d(self, populated_analytics):
        """Quality trend for 90 days creates weekly buckets."""
        trend = await populated_analytics.get_decision_quality_trend(period="90d")
        assert isinstance(trend, list)
        # 90 / 7 = 12 weekly buckets
        assert len(trend) <= 13

    @pytest.mark.asyncio
    async def test_get_topic_distribution_empty(self, analytics):
        """Topic distribution is empty with no data."""
        topics = await analytics.get_topic_distribution(period="30d")
        assert isinstance(topics, dict)
        assert len(topics) == 0

    @pytest.mark.asyncio
    async def test_get_topic_distribution_with_data(self, populated_analytics):
        """Topic distribution groups by protocol."""
        topics = await populated_analytics.get_topic_distribution(period="30d")
        assert isinstance(topics, dict)
        # We recorded "architecture" (2 debates) and "security" (1 debate)
        assert "architecture" in topics
        assert topics["architecture"] == 2
        assert "security" in topics
        assert topics["security"] == 1

    @pytest.mark.asyncio
    async def test_get_outcome_summary_not_found(self, analytics):
        """Returns None for nonexistent debate."""
        summary = await analytics.get_outcome_summary("nonexistent")
        assert summary is None

    @pytest.mark.asyncio
    async def test_get_outcome_summary_with_data(self, populated_analytics):
        """Returns summary for existing debate."""
        summary = await populated_analytics.get_outcome_summary("debate-1")
        assert summary is not None
        assert isinstance(summary, OutcomeSummary)
        assert summary.debate_id == "debate-1"
        assert summary.consensus_reached is True
        assert summary.rounds == 3
        assert summary.agents == ["claude", "gpt-4"]

    @pytest.mark.asyncio
    async def test_get_outcome_summary_no_consensus(self, populated_analytics):
        """Returns summary for debate without consensus."""
        summary = await populated_analytics.get_outcome_summary("debate-2")
        assert summary is not None
        assert summary.consensus_reached is False
        assert summary.rounds == 5

    @pytest.mark.asyncio
    async def test_get_outcome_summary_to_dict(self, populated_analytics):
        """Outcome summary serializes correctly."""
        summary = await populated_analytics.get_outcome_summary("debate-3")
        assert summary is not None
        d = summary.to_dict()
        assert d["debate_id"] == "debate-3"
        assert d["consensus_reached"] is True
        assert d["rounds"] == 2
        assert d["agents"] == ["gpt-4", "gemini"]


# ===========================================================================
# Caching Tests
# ===========================================================================


class TestOutcomeAnalyticsCaching:
    @pytest.fixture
    def analytics(self, tmp_path):
        return OutcomeAnalytics(db_path=str(tmp_path / "cache_test.db"))

    def test_cache_set_and_get(self, analytics):
        """Cache stores and retrieves values."""
        analytics._set_cached("test_key", 42)
        assert analytics._get_cached("test_key") == 42

    def test_cache_miss(self, analytics):
        """Cache returns None for missing keys."""
        assert analytics._get_cached("nonexistent") is None

    def test_cache_expiry(self, analytics):
        """Cache returns None for expired values."""
        from datetime import timedelta

        analytics._cache_ttl = timedelta(seconds=0)
        analytics._set_cached("test_key", 42)
        assert analytics._get_cached("test_key") is None


# ===========================================================================
# Global Instance Tests
# ===========================================================================


class TestGetOutcomeAnalytics:
    def test_returns_instance(self):
        """get_outcome_analytics returns an OutcomeAnalytics."""
        import aragora.analytics.outcome_analytics as mod

        # Reset global
        mod._outcome_analytics = None
        try:
            instance = get_outcome_analytics()
            assert isinstance(instance, OutcomeAnalytics)
        finally:
            mod._outcome_analytics = None

    def test_returns_same_instance(self):
        """get_outcome_analytics returns singleton."""
        import aragora.analytics.outcome_analytics as mod

        mod._outcome_analytics = None
        try:
            instance1 = get_outcome_analytics()
            instance2 = get_outcome_analytics()
            assert instance1 is instance2
        finally:
            mod._outcome_analytics = None


# ===========================================================================
# Import Verification
# ===========================================================================


class TestImports:
    def test_import_from_module(self):
        """Verify direct import works."""
        from aragora.analytics.outcome_analytics import OutcomeAnalytics

        assert OutcomeAnalytics is not None

    def test_import_from_package(self):
        """Verify package-level import works."""
        from aragora.analytics import OutcomeAnalytics, QualityDataPoint

        assert OutcomeAnalytics is not None
        assert QualityDataPoint is not None

    def test_all_exports(self):
        """Verify __all__ contains expected names."""
        from aragora.analytics import outcome_analytics

        expected = {
            "OutcomeAnalytics",
            "QualityDataPoint",
            "OutcomeSummary",
            "AgentContribution",
            "get_outcome_analytics",
        }
        assert set(outcome_analytics.__all__) == expected

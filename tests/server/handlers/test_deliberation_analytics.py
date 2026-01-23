"""Tests for deliberation analytics endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler


@pytest.fixture
def handler():
    """Create a handler instance with mock server context."""
    server_context = MagicMock()
    server_context.config = {}
    return AnalyticsDashboardHandler(server_context)


@pytest.fixture
def mock_debate_store():
    """Create a mock debate store with sample data."""
    store = MagicMock()

    # Mock get_deliberation_stats
    store.get_deliberation_stats.return_value = {
        "total": 150,
        "completed": 145,
        "in_progress": 3,
        "failed": 2,
        "consensus_reached": 120,
        "avg_rounds": 3.5,
        "avg_duration_seconds": 45.2,
        "by_template": {
            "code_review": 45,
            "security_audit": 30,
            "architecture_decision": 25,
        },
        "by_priority": {
            "high": 50,
            "normal": 80,
            "low": 20,
        },
    }

    # Mock get_deliberation_stats_by_channel
    store.get_deliberation_stats_by_channel.return_value = [
        {
            "platform": "slack",
            "channel_id": "C123456",
            "channel_name": "#engineering",
            "total_deliberations": 45,
            "consensus_reached": 38,
            "consensus_rate": "84%",
            "total_duration": 1800,
            "avg_duration_seconds": 40.0,
            "top_templates": ["code_review", "architecture_decision"],
        },
        {
            "platform": "teams",
            "channel_id": "T789",
            "channel_name": "General",
            "total_deliberations": 30,
            "consensus_reached": 24,
            "consensus_rate": "80%",
            "total_duration": 1500,
            "avg_duration_seconds": 50.0,
            "top_templates": ["security_audit"],
        },
    ]

    # Mock get_consensus_stats
    store.get_consensus_stats.return_value = {
        "overall_consensus_rate": "82%",
        "by_team_size": {
            "2": {"count": 30, "consensus_rate": "90%"},
            "3": {"count": 80, "consensus_rate": "85%"},
            "5": {"count": 40, "consensus_rate": "75%"},
        },
        "by_agent": [
            {
                "agent_id": "anthropic-api",
                "agent_name": "Claude",
                "participations": 120,
                "consensus_contributions": 100,
                "consensus_rate": "83%",
                "avg_agreement_score": 0.87,
            },
            {
                "agent_id": "openai-api",
                "agent_name": "GPT-4",
                "participations": 100,
                "consensus_contributions": 82,
                "consensus_rate": "82%",
                "avg_agreement_score": 0.85,
            },
        ],
        "top_teams": [
            {
                "team": ["anthropic-api", "openai-api", "gemini"],
                "deliberations": 25,
                "consensus_rate": "92%",
            },
        ],
    }

    # Mock get_deliberation_performance
    store.get_deliberation_performance.return_value = {
        "summary": {
            "total_deliberations": 150,
            "total_cost_usd": "125.50",
            "avg_cost_per_deliberation": "0.84",
            "total_tokens": 5000000,
            "avg_tokens_per_deliberation": 33333,
            "avg_duration_seconds": 45.2,
            "p50_duration_seconds": 38.0,
            "p95_duration_seconds": 120.0,
            "avg_rounds": 3.5,
        },
        "by_template": [
            {
                "template": "code_review",
                "count": 45,
                "avg_cost": "0.75",
                "avg_duration_seconds": 35.0,
                "avg_rounds": 3.0,
            },
        ],
        "trends": [
            {
                "date": "2026-01-15",
                "count": 5,
                "avg_duration_seconds": 42.0,
                "total_cost": "4.20",
            },
        ],
        "cost_by_agent": {
            "anthropic-api": "60.00",
            "openai-api": "45.50",
            "gemini": "20.00",
        },
    }

    return store


class TestDeliberationAnalyticsRouting:
    """Test routing for deliberation analytics endpoints."""

    def test_can_handle_deliberations_endpoint(self, handler):
        """Test that handler recognizes deliberation analytics routes."""
        assert handler.can_handle("/api/v1/analytics/deliberations")
        assert handler.can_handle("/api/v1/analytics/deliberations/channels")
        assert handler.can_handle("/api/v1/analytics/deliberations/consensus")
        assert handler.can_handle("/api/v1/analytics/deliberations/performance")

    def test_cannot_handle_unknown_endpoint(self, handler):
        """Test that handler rejects unknown routes."""
        assert not handler.can_handle("/api/v1/analytics/unknown")
        assert not handler.can_handle("/api/v1/deliberations")


class TestDeliberationSummary:
    """Tests for the deliberation summary endpoint."""

    def test_get_deliberation_summary_success(self, handler, mock_debate_store):
        """Test successful retrieval of deliberation summary."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            # Get the unwrapped method and call it
            method = handler._get_deliberation_summary
            # Skip decorators by accessing __wrapped__ twice (require_user_auth, handle_errors)
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org", "days": "30"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["total_deliberations"] == 150
            assert body["consensus_rate"] == "82.8%"

    def test_get_deliberation_summary_missing_org_id(self, handler):
        """Test error when org_id is missing."""
        method = handler._get_deliberation_summary
        unwrapped = method.__wrapped__.__wrapped__
        result = unwrapped(handler, {}, user=MagicMock())

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "org_id is required" in body["error"]


class TestDeliberationByChannel:
    """Tests for the deliberation by channel endpoint."""

    def test_get_deliberation_by_channel_success(self, handler, mock_debate_store):
        """Test successful retrieval of channel breakdown."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_deliberation_by_channel
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org", "days": "30"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "channels" in body
            assert "by_platform" in body
            assert len(body["channels"]) == 2
            assert body["channels"][0]["platform"] == "slack"

    def test_aggregates_by_platform(self, handler, mock_debate_store):
        """Test that channel data is aggregated by platform."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_deliberation_by_channel
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "slack" in body["by_platform"]
            assert body["by_platform"]["slack"]["count"] == 45


class TestConsensusRates:
    """Tests for the consensus rates endpoint."""

    def test_get_consensus_rates_success(self, handler, mock_debate_store):
        """Test successful retrieval of consensus rates."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_consensus_rates
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org", "days": "30"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["overall_consensus_rate"] == "82%"
            assert "by_team_size" in body
            assert "by_agent" in body
            assert "top_teams" in body

    def test_includes_agent_breakdown(self, handler, mock_debate_store):
        """Test that agent-level consensus data is included."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_consensus_rates
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            agents = body["by_agent"]
            assert len(agents) == 2
            assert agents[0]["agent_id"] == "anthropic-api"
            assert agents[0]["consensus_rate"] == "83%"


class TestDeliberationPerformance:
    """Tests for the deliberation performance endpoint."""

    def test_get_deliberation_performance_success(self, handler, mock_debate_store):
        """Test successful retrieval of performance metrics."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_deliberation_performance
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(
                handler,
                {"org_id": "test-org", "days": "30", "granularity": "day"},
                user=MagicMock(),
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "summary" in body
            assert "by_template" in body
            assert "trends" in body
            assert "cost_by_agent" in body

    def test_includes_cost_breakdown(self, handler, mock_debate_store):
        """Test that cost breakdown by agent is included."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_deliberation_performance
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(handler, {"org_id": "test-org"}, user=MagicMock())

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["cost_by_agent"]["anthropic-api"] == "60.00"
            assert body["cost_by_agent"]["openai-api"] == "45.50"

    def test_granularity_parameter(self, handler, mock_debate_store):
        """Test that granularity parameter is passed correctly."""
        with patch(
            "aragora.memory.debate_store.get_debate_store",
            return_value=mock_debate_store,
        ):
            method = handler._get_deliberation_performance
            unwrapped = method.__wrapped__.__wrapped__
            result = unwrapped(
                handler, {"org_id": "test-org", "granularity": "week"}, user=MagicMock()
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["granularity"] == "week"

            # Verify store was called with correct granularity
            call_args = mock_debate_store.get_deliberation_performance.call_args
            assert call_args[1]["granularity"] == "week"


class TestDebateStore:
    """Tests for the DebateStore class."""

    def test_store_initialization(self, tmp_path):
        """Test that DebateStore initializes correctly."""
        from aragora.memory.debate_store import DebateStore

        db_path = tmp_path / "test_debate.db"
        store = DebateStore(db_path=db_path)

        assert store.db_path == db_path
        assert db_path.exists()

    def test_record_and_update_deliberation(self, tmp_path):
        """Test recording and updating a deliberation."""
        from aragora.memory.debate_store import DebateStore

        db_path = tmp_path / "test_debate.db"
        store = DebateStore(db_path=db_path)

        # Record a new deliberation
        store.record_deliberation(
            deliberation_id="test-123",
            org_id="org-1",
            question="Should we use microservices?",
            status="in_progress",
            template="architecture_decision",
            priority="high",
            platform="slack",
            channel_id="C123",
            team_agents=["anthropic-api", "openai-api"],
        )

        # Update with result
        store.update_deliberation_result(
            deliberation_id="test-123",
            status="completed",
            consensus_reached=True,
            rounds=4,
            duration_seconds=60.5,
            total_tokens=10000,
            total_cost_usd=0.50,
        )

        # Verify stats
        now = datetime.now(timezone.utc)
        stats = store.get_deliberation_stats(
            org_id="org-1",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert stats["total"] == 1
        assert stats["completed"] == 1
        assert stats["consensus_reached"] == 1

    def test_record_agent_participation(self, tmp_path):
        """Test recording agent participation."""
        from aragora.memory.debate_store import DebateStore

        db_path = tmp_path / "test_debate.db"
        store = DebateStore(db_path=db_path)

        # Record deliberation first
        store.record_deliberation(
            deliberation_id="test-456",
            org_id="org-1",
            question="Test question",
        )

        # Record agent participation
        store.record_agent_participation(
            deliberation_id="test-456",
            agent_id="anthropic-api",
            agent_name="Claude",
            tokens_used=5000,
            cost_usd=0.25,
            agreed_with_consensus=True,
        )

        store.record_agent_participation(
            deliberation_id="test-456",
            agent_id="openai-api",
            agent_name="GPT-4",
            tokens_used=4000,
            cost_usd=0.20,
            agreed_with_consensus=False,
        )

        # Verify in consensus stats
        store.update_deliberation_result(
            deliberation_id="test-456",
            status="completed",
            consensus_reached=True,
            rounds=3,
            duration_seconds=45.0,
        )

        now = datetime.now(timezone.utc)
        consensus_stats = store.get_consensus_stats(
            org_id="org-1",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert len(consensus_stats["by_agent"]) == 2

    def test_get_stats_by_channel(self, tmp_path):
        """Test getting statistics grouped by channel."""
        from aragora.memory.debate_store import DebateStore

        db_path = tmp_path / "test_debate.db"
        store = DebateStore(db_path=db_path)

        # Record deliberations in different channels
        store.record_deliberation(
            deliberation_id="d1",
            org_id="org-1",
            question="Q1",
            platform="slack",
            channel_id="C1",
            channel_name="#dev",
            template="code_review",
        )
        store.update_deliberation_result("d1", "completed", True, 3, 30.0)

        store.record_deliberation(
            deliberation_id="d2",
            org_id="org-1",
            question="Q2",
            platform="slack",
            channel_id="C1",
            channel_name="#dev",
            template="code_review",
        )
        store.update_deliberation_result("d2", "completed", True, 2, 25.0)

        store.record_deliberation(
            deliberation_id="d3",
            org_id="org-1",
            question="Q3",
            platform="teams",
            channel_id="T1",
            channel_name="General",
        )
        store.update_deliberation_result("d3", "completed", False, 5, 60.0)

        now = datetime.now(timezone.utc)
        channel_stats = store.get_deliberation_stats_by_channel(
            org_id="org-1",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert len(channel_stats) == 2
        slack_channel = next(c for c in channel_stats if c["platform"] == "slack")
        assert slack_channel["total_deliberations"] == 2
        assert slack_channel["consensus_rate"] == "100%"

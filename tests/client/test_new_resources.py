"""Tests for new SDK API resources (Consensus, Pulse, System, Tournaments)."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources import (
    ConsensusAPI,
    PulseAPI,
    SystemAPI,
    TournamentsAPI,
)
from aragora.client.resources.consensus import (
    ConsensusStats,
    Dissent,
    RiskWarning,
    SettledTopic,
    SimilarDebate,
)
from aragora.client.resources.pulse import (
    DebateSuggestion,
    PulseAnalytics,
    TrendingTopic,
)
from aragora.client.resources.system import (
    CircuitBreakerStatus,
    HealthStatus,
    SystemInfo,
    SystemStats,
)
from aragora.client.resources.tournaments import (
    Tournament,
    TournamentStanding,
    TournamentSummary,
)


class TestConsensusAPI:
    """Tests for ConsensusAPI resource."""

    def test_consensus_api_exists(self):
        """Test that ConsensusAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.consensus, ConsensusAPI)

    def test_consensus_api_has_methods(self):
        """Test that ConsensusAPI has required methods."""
        client = AragoraClient()
        assert hasattr(client.consensus, "find_similar")
        assert hasattr(client.consensus, "find_similar_async")
        assert hasattr(client.consensus, "get_settled")
        assert hasattr(client.consensus, "get_settled_async")
        assert hasattr(client.consensus, "get_dissents")
        assert hasattr(client.consensus, "get_dissents_async")
        assert hasattr(client.consensus, "get_risk_warnings")
        assert hasattr(client.consensus, "get_risk_warnings_async")
        assert hasattr(client.consensus, "get_contrarian_views")
        assert hasattr(client.consensus, "get_contrarian_views_async")
        assert hasattr(client.consensus, "get_stats")
        assert hasattr(client.consensus, "get_stats_async")


class TestConsensusModels:
    """Tests for Consensus model classes."""

    def test_similar_debate_from_dict(self):
        """Test SimilarDebate.from_dict."""
        data = {
            "id": "debate_123",
            "topic": "Should we use microservices?",
            "conclusion": "Yes, for large teams",
            "strength": "strong",
            "confidence": 0.85,
            "similarity": 0.92,
            "timestamp": "2024-01-15T10:30:00Z",
            "dissent_count": 2,
        }
        debate = SimilarDebate.from_dict(data)
        assert debate.id == "debate_123"
        assert debate.topic == "Should we use microservices?"
        assert debate.confidence == 0.85
        assert debate.similarity == 0.92
        assert debate.dissent_count == 2

    def test_settled_topic_from_dict(self):
        """Test SettledTopic.from_dict."""
        data = {
            "topic": "REST vs GraphQL",
            "conclusion": "Use REST for simple APIs, GraphQL for complex queries",
            "confidence": 0.90,
            "strength": "strong",
            "last_debated": "2024-01-10T08:00:00Z",
            "debate_count": 5,
        }
        topic = SettledTopic.from_dict(data)
        assert topic.topic == "REST vs GraphQL"
        assert topic.confidence == 0.90
        assert topic.debate_count == 5

    def test_dissent_from_dict(self):
        """Test Dissent.from_dict."""
        data = {
            "id": "dissent_001",
            "debate_id": "debate_123",
            "agent_id": "claude",
            "dissent_type": "alternative_approach",
            "content": "Consider event-driven architecture",
            "reasoning": "Better for real-time requirements",
            "confidence": 0.75,
            "acknowledged": True,
            "rebuttal": "Valid point for specific use cases",
            "timestamp": "2024-01-15T11:00:00Z",
        }
        dissent = Dissent.from_dict(data)
        assert dissent.id == "dissent_001"
        assert dissent.dissent_type == "alternative_approach"
        assert dissent.acknowledged is True

    def test_risk_warning_from_dict(self):
        """Test RiskWarning.from_dict."""
        data = {
            "id": "warning_001",
            "debate_id": "debate_123",
            "agent_id": "gpt4",
            "content": "Security implications of this approach",
            "reasoning": "May expose sensitive data",
            "severity": "high",
            "acknowledged": False,
            "timestamp": "2024-01-15T12:00:00Z",
        }
        warning = RiskWarning.from_dict(data)
        assert warning.id == "warning_001"
        assert warning.severity == "high"
        assert warning.acknowledged is False

    def test_consensus_stats_from_dict(self):
        """Test ConsensusStats.from_dict."""
        data = {
            "total_consensuses": 150,
            "total_dissents": 45,
            "by_strength": {"strong": 80, "moderate": 50, "weak": 20},
            "by_domain": {"architecture": 60, "security": 40, "performance": 50},
            "avg_confidence": 0.82,
        }
        stats = ConsensusStats.from_dict(data)
        assert stats.total_consensuses == 150
        assert stats.total_dissents == 45
        assert stats.avg_confidence == 0.82


class TestPulseAPI:
    """Tests for PulseAPI resource."""

    def test_pulse_api_exists(self):
        """Test that PulseAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.pulse, PulseAPI)

    def test_pulse_api_has_methods(self):
        """Test that PulseAPI has required methods."""
        client = AragoraClient()
        assert hasattr(client.pulse, "trending")
        assert hasattr(client.pulse, "trending_async")
        assert hasattr(client.pulse, "suggest")
        assert hasattr(client.pulse, "suggest_async")
        assert hasattr(client.pulse, "get_analytics")
        assert hasattr(client.pulse, "get_analytics_async")
        assert hasattr(client.pulse, "refresh")
        assert hasattr(client.pulse, "refresh_async")


class TestPulseModels:
    """Tests for Pulse model classes."""

    def test_trending_topic_from_dict(self):
        """Test TrendingTopic.from_dict."""
        data = {
            "title": "AI Safety Research Breakthroughs",
            "source": "arxiv",
            "score": 95.5,
            "category": "technology",
            "url": "https://arxiv.org/abs/2401.xxxxx",
            "summary": "New approaches to AI alignment",
            "suggested_agents": ["claude", "gpt4"],
        }
        topic = TrendingTopic.from_dict(data)
        assert topic.title == "AI Safety Research Breakthroughs"
        assert topic.source == "arxiv"
        assert topic.score == 95.5
        assert topic.suggested_agents == ["claude", "gpt4"]

    def test_debate_suggestion_from_dict(self):
        """Test DebateSuggestion.from_dict."""
        data = {
            "topic": "Should companies adopt LLMs for code review?",
            "rationale": "High impact, multiple valid perspectives",
            "difficulty": "hard",
            "estimated_rounds": 5,
            "suggested_agents": ["claude", "gpt4", "gemini"],
            "related_topics": ["AI in DevOps", "Code quality automation"],
        }
        suggestion = DebateSuggestion.from_dict(data)
        assert suggestion.topic == "Should companies adopt LLMs for code review?"
        assert suggestion.difficulty == "hard"
        assert suggestion.estimated_rounds == 5

    def test_pulse_analytics_from_dict(self):
        """Test PulseAnalytics.from_dict."""
        data = {
            "total_topics": 500,
            "by_source": {"hackernews": 200, "arxiv": 150, "reddit": 150},
            "by_category": {"technology": 300, "science": 150, "business": 50},
            "top_categories": ["technology", "science", "business"],
            "freshness_hours": 12.0,
        }
        analytics = PulseAnalytics.from_dict(data)
        assert analytics.total_topics == 500
        assert analytics.freshness_hours == 12.0


class TestSystemAPI:
    """Tests for SystemAPI resource."""

    def test_system_api_exists(self):
        """Test that SystemAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.system, SystemAPI)

    def test_system_api_has_methods(self):
        """Test that SystemAPI has required methods."""
        client = AragoraClient()
        assert hasattr(client.system, "health")
        assert hasattr(client.system, "health_async")
        assert hasattr(client.system, "info")
        assert hasattr(client.system, "info_async")
        assert hasattr(client.system, "stats")
        assert hasattr(client.system, "stats_async")
        assert hasattr(client.system, "circuit_breakers")
        assert hasattr(client.system, "circuit_breakers_async")
        assert hasattr(client.system, "reset_circuit_breaker")
        assert hasattr(client.system, "reset_circuit_breaker_async")
        assert hasattr(client.system, "modes")
        assert hasattr(client.system, "modes_async")


class TestSystemModels:
    """Tests for System model classes."""

    def test_health_status_from_dict(self):
        """Test HealthStatus.from_dict."""
        data = {
            "status": "healthy",
            "version": "1.5.0",
            "uptime_seconds": 86400.0,
            "checks": {"database": True, "redis": True, "weaviate": False},
            "timestamp": "2024-01-15T12:00:00Z",
        }
        health = HealthStatus.from_dict(data)
        assert health.status == "healthy"
        assert health.version == "1.5.0"
        assert health.is_healthy is False  # weaviate is False

    def test_health_status_is_healthy(self):
        """Test HealthStatus.is_healthy property."""
        healthy_data = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
            "checks": {"db": True, "cache": True},
            "timestamp": "2024-01-15T12:00:00Z",
        }
        unhealthy_data = {
            "status": "degraded",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
            "checks": {"db": True, "cache": True},
            "timestamp": "2024-01-15T12:00:00Z",
        }

        healthy = HealthStatus.from_dict(healthy_data)
        unhealthy = HealthStatus.from_dict(unhealthy_data)

        assert healthy.is_healthy is True
        assert unhealthy.is_healthy is False

    def test_system_info_from_dict(self):
        """Test SystemInfo.from_dict."""
        data = {
            "version": "1.5.0",
            "environment": "production",
            "python_version": "3.10.13",
            "platform": "darwin",
            "agents_available": ["claude", "gpt4", "gemini"],
            "features_enabled": ["streaming", "consensus", "memory"],
            "memory_mb": 512.5,
            "cpu_percent": 25.0,
        }
        info = SystemInfo.from_dict(data)
        assert info.version == "1.5.0"
        assert "claude" in info.agents_available
        assert info.memory_mb == 512.5

    def test_system_stats_from_dict(self):
        """Test SystemStats.from_dict."""
        data = {
            "total_debates": 1500,
            "total_agents": 12,
            "active_debates": 5,
            "debates_today": 45,
            "debates_this_week": 200,
            "avg_debate_duration_seconds": 180.5,
            "memory_entries": 50000,
            "consensus_rate": 0.78,
        }
        stats = SystemStats.from_dict(data)
        assert stats.total_debates == 1500
        assert stats.consensus_rate == 0.78

    def test_circuit_breaker_status_from_dict(self):
        """Test CircuitBreakerStatus.from_dict."""
        data = {
            "agent_id": "claude",
            "state": "closed",
            "failure_count": 0,
            "success_count": 150,
            "last_failure": None,
            "last_success": "2024-01-15T12:00:00Z",
        }
        status = CircuitBreakerStatus.from_dict(data)
        assert status.agent_id == "claude"
        assert status.state == "closed"
        assert status.is_open is False

    def test_circuit_breaker_is_open(self):
        """Test CircuitBreakerStatus.is_open property."""
        open_data = {
            "agent_id": "gpt4",
            "state": "open",
            "failure_count": 5,
            "success_count": 10,
        }
        closed_data = {
            "agent_id": "claude",
            "state": "closed",
            "failure_count": 0,
            "success_count": 100,
        }

        open_breaker = CircuitBreakerStatus.from_dict(open_data)
        closed_breaker = CircuitBreakerStatus.from_dict(closed_data)

        assert open_breaker.is_open is True
        assert closed_breaker.is_open is False


class TestTournamentsAPI:
    """Tests for TournamentsAPI resource."""

    def test_tournaments_api_exists(self):
        """Test that TournamentsAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.tournaments, TournamentsAPI)

    def test_tournaments_api_has_methods(self):
        """Test that TournamentsAPI has required methods."""
        client = AragoraClient()
        assert hasattr(client.tournaments, "list")
        assert hasattr(client.tournaments, "list_async")
        assert hasattr(client.tournaments, "get")
        assert hasattr(client.tournaments, "get_async")
        assert hasattr(client.tournaments, "get_standings")
        assert hasattr(client.tournaments, "get_standings_async")
        assert hasattr(client.tournaments, "create")
        assert hasattr(client.tournaments, "create_async")
        assert hasattr(client.tournaments, "cancel")
        assert hasattr(client.tournaments, "cancel_async")


class TestTournamentModels:
    """Tests for Tournament model classes."""

    def test_tournament_standing_from_dict(self):
        """Test TournamentStanding.from_dict."""
        data = {
            "agent_id": "claude",
            "rank": 1,
            "wins": 8,
            "losses": 2,
            "draws": 1,
            "points": 25.0,
            "elo_change": 45.5,
        }
        standing = TournamentStanding.from_dict(data)
        assert standing.agent_id == "claude"
        assert standing.rank == 1
        assert standing.wins == 8
        assert standing.elo_change == 45.5

    def test_tournament_summary_from_dict(self):
        """Test TournamentSummary.from_dict."""
        data = {
            "id": "tournament_001",
            "name": "Weekly Championship",
            "status": "completed",
            "participants": 8,
            "rounds_completed": 7,
            "total_rounds": 7,
            "created_at": "2024-01-08T10:00:00Z",
            "completed_at": "2024-01-15T18:00:00Z",
            "winner": "claude",
        }
        summary = TournamentSummary.from_dict(data)
        assert summary.id == "tournament_001"
        assert summary.name == "Weekly Championship"
        assert summary.winner == "claude"

    def test_tournament_from_dict(self):
        """Test Tournament.from_dict."""
        data = {
            "id": "tournament_001",
            "name": "Weekly Championship",
            "status": "completed",
            "format": "round_robin",
            "participants": ["claude", "gpt4", "gemini", "llama"],
            "standings": [
                {
                    "agent_id": "claude",
                    "rank": 1,
                    "wins": 3,
                    "losses": 0,
                    "draws": 0,
                    "points": 9.0,
                },
                {"agent_id": "gpt4", "rank": 2, "wins": 2, "losses": 1, "draws": 0, "points": 6.0},
            ],
            "rounds_completed": 3,
            "total_rounds": 3,
            "created_at": "2024-01-08T10:00:00Z",
            "completed_at": "2024-01-10T18:00:00Z",
            "metadata": {"topic": "AI ethics"},
        }
        tournament = Tournament.from_dict(data)
        assert tournament.id == "tournament_001"
        assert tournament.format == "round_robin"
        assert len(tournament.standings) == 2
        assert tournament.standings[0].agent_id == "claude"
        assert tournament.metadata["topic"] == "AI ethics"

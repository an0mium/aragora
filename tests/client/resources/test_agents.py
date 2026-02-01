"""Tests for AgentsAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.agents import AgentsAPI


class TestAgentsAPI:
    """Tests for AgentsAPI resource."""

    def test_agents_api_exists(self):
        """Test that AgentsAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.agents, AgentsAPI)

    def test_agents_api_has_list_method(self):
        """Test that AgentsAPI has list methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "list")
        assert hasattr(client.agents, "list_async")
        assert callable(client.agents.list)
        assert callable(client.agents.list_async)

    def test_agents_api_has_get_method(self):
        """Test that AgentsAPI has get methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get")
        assert hasattr(client.agents, "get_async")

    def test_agents_api_has_profile_methods(self):
        """Test that AgentsAPI has profile methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_profile")
        assert hasattr(client.agents, "get_profile_async")

    def test_agents_api_has_calibration_methods(self):
        """Test that AgentsAPI has calibration methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_calibration")
        assert hasattr(client.agents, "get_calibration_async")

    def test_agents_api_has_performance_methods(self):
        """Test that AgentsAPI has performance methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_performance")
        assert hasattr(client.agents, "get_performance_async")

    def test_agents_api_has_head_to_head_methods(self):
        """Test that AgentsAPI has head-to-head methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_head_to_head")
        assert hasattr(client.agents, "get_head_to_head_async")

    def test_agents_api_has_opponent_briefing_methods(self):
        """Test that AgentsAPI has opponent briefing methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_opponent_briefing")
        assert hasattr(client.agents, "get_opponent_briefing_async")

    def test_agents_api_has_consistency_methods(self):
        """Test that AgentsAPI has consistency methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_consistency")
        assert hasattr(client.agents, "get_consistency_async")

    def test_agents_api_has_flips_methods(self):
        """Test that AgentsAPI has flips methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_flips")
        assert hasattr(client.agents, "get_flips_async")

    def test_agents_api_has_network_methods(self):
        """Test that AgentsAPI has network methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_network")
        assert hasattr(client.agents, "get_network_async")

    def test_agents_api_has_moments_methods(self):
        """Test that AgentsAPI has moments methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_moments")
        assert hasattr(client.agents, "get_moments_async")

    def test_agents_api_has_positions_methods(self):
        """Test that AgentsAPI has positions methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_positions")
        assert hasattr(client.agents, "get_positions_async")

    def test_agents_api_has_domains_methods(self):
        """Test that AgentsAPI has domains methods."""
        client = AragoraClient()
        assert hasattr(client.agents, "get_domains")
        assert hasattr(client.agents, "get_domains_async")


class TestAgentModels:
    """Tests for Agent model classes."""

    def test_agent_profile_import(self):
        """Test AgentProfile model can be imported."""
        from aragora.client.models import AgentProfile

        profile = AgentProfile(
            agent_id="claude",
            name="Claude",
            provider="anthropic",
        )
        assert profile.agent_id == "claude"
        assert profile.name == "Claude"

    def test_agent_calibration_import(self):
        """Test AgentCalibration model can be imported."""
        from aragora.client.models import AgentCalibration

        calibration = AgentCalibration(
            agent="claude",
            overall_score=0.85,
        )
        assert calibration.overall_score == 0.85
        assert calibration.agent == "claude"

    def test_agent_performance_import(self):
        """Test AgentPerformance model can be imported."""
        from aragora.client.models import AgentPerformance

        # Model import check
        assert AgentPerformance is not None

    def test_head_to_head_stats_import(self):
        """Test HeadToHeadStats model can be imported."""
        from aragora.client.models import HeadToHeadStats

        stats = HeadToHeadStats(
            agent="claude",
            opponent="gpt4",
            wins=15,
            losses=10,
            draws=5,
        )
        assert stats.wins == 15

    def test_opponent_briefing_import(self):
        """Test OpponentBriefing model can be imported."""
        from aragora.client.models import OpponentBriefing

        # Model import check
        assert OpponentBriefing is not None

    def test_agent_consistency_import(self):
        """Test AgentConsistency model can be imported."""
        from aragora.client.models import AgentConsistency

        consistency = AgentConsistency(
            agent="claude",
        )
        assert consistency.agent == "claude"

    def test_agent_flip_import(self):
        """Test AgentFlip model can be imported."""
        from aragora.client.models import AgentFlip

        flip = AgentFlip(
            flip_id="flip_001",
            agent="claude",
            debate_id="debate_123",
        )
        assert flip.agent == "claude"

    def test_agent_network_import(self):
        """Test AgentNetwork model can be imported."""
        from aragora.client.models import AgentNetwork

        network = AgentNetwork(
            agent="claude",
        )
        assert network.agent == "claude"

    def test_agent_moment_import(self):
        """Test AgentMoment model can be imported."""
        from aragora.client.models import AgentMoment

        moment = AgentMoment(
            moment_id="moment_001",
            agent="claude",
            debate_id="debate_123",
            type="breakthrough",
        )
        assert moment.type == "breakthrough"

    def test_agent_position_import(self):
        """Test AgentPosition model can be imported."""
        from aragora.client.models import AgentPosition

        position = AgentPosition(
            position_id="pos_001",
            agent="claude",
            debate_id="debate_123",
        )
        assert position.agent == "claude"

    def test_domain_rating_import(self):
        """Test DomainRating model can be imported."""
        from aragora.client.models import DomainRating

        rating = DomainRating(
            domain="security",
        )
        assert rating.domain == "security"

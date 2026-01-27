"""Tests for AgentHierarchy integration with TeamSelector."""

import pytest
from unittest.mock import MagicMock

from aragora.debate.hierarchy import (
    AgentHierarchy,
    HierarchyConfig,
    HierarchyRole,
)
from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig
from aragora.routing.selection import AgentProfile


class TestHierarchyTeamSelectorIntegration:
    """Tests for integration between AgentHierarchy and TeamSelector."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = []
        for name in ["claude-opus", "gpt-4", "gemini-pro", "deepseek"]:
            agent = MagicMock()
            agent.name = name
            agent.agent_type = name.split("-")[0]
            agent.capabilities = set()
            agent.metadata = {}
            agents.append(agent)
        return agents

    @pytest.fixture
    def agent_profiles(self):
        """Create AgentProfile instances for hierarchy testing."""
        return [
            AgentProfile(
                name="claude-opus",
                agent_type="claude",
                elo_rating=1800.0,
                capabilities={"reasoning", "synthesis", "coordination", "analysis"},
                task_affinity={"code": 0.9},
            ),
            AgentProfile(
                name="gpt-4",
                agent_type="gpt",
                elo_rating=1750.0,
                capabilities={"reasoning", "synthesis", "analysis"},
                task_affinity={"code": 0.8},
            ),
            AgentProfile(
                name="gemini-pro",
                agent_type="gemini",
                elo_rating=1650.0,
                capabilities={"analysis", "quality_assessment"},
                task_affinity={"research": 0.9},
            ),
            AgentProfile(
                name="deepseek",
                agent_type="deepseek",
                elo_rating=1500.0,
                capabilities={"coding", "reasoning"},
                task_affinity={"code": 0.95},
            ),
        ]

    def test_team_selector_with_hierarchy(self, mock_agents):
        """Test TeamSelector with AgentHierarchy integration."""
        hierarchy = AgentHierarchy(HierarchyConfig(max_monitors=1))
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # Select agents with a debate_id to trigger hierarchy assignment
        selected = selector.select(
            agents=mock_agents,
            domain="code",
            debate_id="test-debate-1",
        )

        # All agents should be returned (no filtering by role)
        assert len(selected) == 4

        # Hierarchy should have been assigned
        status = selector.get_hierarchy_status("test-debate-1")
        assert status is not None
        assert status["orchestrator"] is not None
        assert len(status["monitors"]) >= 0
        assert len(status["workers"]) >= 0

    def test_hierarchy_role_assignment(self, agent_profiles):
        """Test that hierarchy assigns roles correctly."""
        hierarchy = AgentHierarchy(
            HierarchyConfig(max_orchestrators=1, max_monitors=1, min_workers=2)
        )

        assignments = hierarchy.assign_roles(
            debate_id="test-debate",
            agents=agent_profiles,
            task_type="code",
        )

        # Should have exactly one orchestrator
        orchestrators = [a for a, r in assignments.items() if r.role == HierarchyRole.ORCHESTRATOR]
        assert len(orchestrators) == 1

        # Claude should be orchestrator (highest ELO + best capabilities)
        assert orchestrators[0] == "claude-opus"

        # Should have one monitor
        monitors = [a for a, r in assignments.items() if r.role == HierarchyRole.MONITOR]
        assert len(monitors) == 1

        # Gemini should be monitor (has quality_assessment)
        assert monitors[0] == "gemini-pro"

        # Rest should be workers
        workers = [a for a, r in assignments.items() if r.role == HierarchyRole.WORKER]
        assert len(workers) == 2

    def test_filter_by_hierarchy_role(self, mock_agents):
        """Test filtering agents by hierarchy role."""
        hierarchy = AgentHierarchy()
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # First assign roles
        selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-2",
        )

        # Filter for workers only
        workers_only = selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-2",
            required_hierarchy_roles={"worker"},
        )

        # Should have fewer agents than total
        assert len(workers_only) < len(mock_agents)

        # All returned agents should be workers
        for agent in workers_only:
            role = selector._get_agent_hierarchy_role(agent, "test-debate-2")
            assert role == "worker"

    def test_filter_orchestrator_only(self, mock_agents):
        """Test filtering for orchestrator role."""
        hierarchy = AgentHierarchy()
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # First assign roles
        selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-3",
        )

        # Filter for orchestrator only
        orchestrators = selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-3",
            required_hierarchy_roles={"orchestrator"},
        )

        # Should have exactly one agent
        assert len(orchestrators) == 1

        # That agent should be the orchestrator
        role = selector._get_agent_hierarchy_role(orchestrators[0], "test-debate-3")
        assert role == "orchestrator"

    def test_hierarchy_fallback_when_no_match(self, mock_agents):
        """Test fallback when no agents match required roles."""
        hierarchy = AgentHierarchy()
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            hierarchy_filter_fallback=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # Assign roles first
        selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-4",
        )

        # Request a non-existent role (should fall back to all agents)
        result = selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-4",
            required_hierarchy_roles={"nonexistent_role"},
        )

        # Should fall back to all agents
        assert len(result) == len(mock_agents)

    def test_clear_hierarchy_cache(self, mock_agents):
        """Test clearing hierarchy cache."""
        hierarchy = AgentHierarchy()
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # Assign roles
        selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-5",
        )

        # Verify hierarchy exists
        assert selector.get_hierarchy_status("test-debate-5") is not None

        # Clear cache
        selector.clear_hierarchy_cache("test-debate-5")

        # Hierarchy should be cleared
        status = selector.get_hierarchy_status("test-debate-5")
        assert status is None or status.get("status") == "not_initialized"

    def test_no_hierarchy_when_not_provided(self, mock_agents):
        """Test that selector works without hierarchy."""
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=False,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(config=config)

        # Should work without hierarchy
        selected = selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-6",
        )

        assert len(selected) == len(mock_agents)
        assert selector.get_hierarchy_status("test-debate-6") is None

    def test_hierarchy_with_elo_scoring(self, mock_agents):
        """Test that hierarchy respects ELO scores."""
        # Create a mock ELO system
        elo_system = MagicMock()
        elo_system.get_rating.side_effect = lambda name: {
            "claude-opus": 1800,
            "gpt-4": 1750,
            "gemini-pro": 1650,
            "deepseek": 1500,
        }.get(name, 1000)

        hierarchy = AgentHierarchy(
            HierarchyConfig(
                elo_weight=0.5,  # High weight for ELO
                capability_weight=0.3,
                affinity_weight=0.2,
            )
        )
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            elo_system=elo_system,
            agent_hierarchy=hierarchy,
            config=config,
        )

        # Select should use hierarchy with ELO weighting
        selected = selector.select(
            agents=mock_agents,
            domain="general",
            debate_id="test-debate-7",
        )

        # All agents should be returned
        assert len(selected) == 4

        # Orchestrator should be the highest-ELO agent with coordination capabilities
        orchestrator = hierarchy.get_orchestrator("test-debate-7")
        # Claude-opus has highest ELO and good capabilities
        assert orchestrator == "claude-opus"

    def test_multiple_debates_independent(self, mock_agents):
        """Test that multiple debates have independent hierarchy assignments."""
        hierarchy = AgentHierarchy()
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )

        selector = TeamSelector(
            agent_hierarchy=hierarchy,
            config=config,
        )

        # Assign roles for two debates
        selector.select(
            agents=mock_agents,
            domain="code",
            debate_id="debate-A",
        )
        selector.select(
            agents=mock_agents,
            domain="research",
            debate_id="debate-B",
        )

        # Both debates should have independent hierarchy
        status_a = selector.get_hierarchy_status("debate-A")
        status_b = selector.get_hierarchy_status("debate-B")

        assert status_a is not None
        assert status_b is not None

        # Clear one, the other should remain
        selector.clear_hierarchy_cache("debate-A")

        assert (
            selector.get_hierarchy_status("debate-A") is None
            or selector.get_hierarchy_status("debate-A").get("status") == "not_initialized"
        )
        assert selector.get_hierarchy_status("debate-B") is not None

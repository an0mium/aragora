"""Tests for Gastown Agent Role integration with team selection.

Tests the integration of Gastown-style hierarchical roles (Mayor, Witness,
Polecat, Crew) with the debate team selector.
"""

import pytest
from unittest.mock import MagicMock

from aragora.agents.spec import AgentSpec
from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig


class TestAgentSpecHierarchyRole:
    """Tests for hierarchy_role field on AgentSpec."""

    def test_hierarchy_role_defaults_to_none(self):
        """hierarchy_role should default to None."""
        spec = AgentSpec(provider="anthropic-api")
        assert spec.hierarchy_role is None

    def test_hierarchy_role_can_be_set(self):
        """hierarchy_role should be settable to valid values."""
        spec = AgentSpec(provider="anthropic-api", hierarchy_role="mayor")
        assert spec.hierarchy_role == "mayor"

    def test_hierarchy_role_normalized_to_lowercase(self):
        """hierarchy_role should be normalized to lowercase."""
        spec = AgentSpec(provider="anthropic-api", hierarchy_role="MAYOR")
        assert spec.hierarchy_role == "mayor"

    def test_invalid_hierarchy_role_raises(self):
        """Invalid hierarchy_role should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hierarchy role"):
            AgentSpec(provider="anthropic-api", hierarchy_role="invalid_role")

    def test_all_valid_hierarchy_roles(self):
        """All valid hierarchy roles should be accepted."""
        for role in ["mayor", "witness", "polecat", "crew"]:
            spec = AgentSpec(provider="anthropic-api", hierarchy_role=role)
            assert spec.hierarchy_role == role

    def test_gastown_role_property(self):
        """gastown_role property should return AgentRole enum."""
        spec = AgentSpec(provider="anthropic-api", hierarchy_role="mayor")
        gastown = spec.gastown_role
        assert gastown is not None
        assert gastown.value == "mayor"

    def test_gastown_role_none_when_not_set(self):
        """gastown_role should return None when hierarchy_role not set."""
        spec = AgentSpec(provider="anthropic-api")
        assert spec.gastown_role is None

    def test_is_coordinator(self):
        """is_coordinator should return True for mayor role."""
        spec = AgentSpec(provider="anthropic-api", hierarchy_role="mayor")
        assert spec.is_coordinator() is True
        spec2 = AgentSpec(provider="anthropic-api", hierarchy_role="crew")
        assert spec2.is_coordinator() is False

    def test_is_monitor(self):
        """is_monitor should return True for witness role."""
        spec = AgentSpec(provider="anthropic-api", hierarchy_role="witness")
        assert spec.is_monitor() is True
        spec2 = AgentSpec(provider="anthropic-api", hierarchy_role="crew")
        assert spec2.is_monitor() is False

    def test_is_worker(self):
        """is_worker should return True for polecat and crew roles."""
        for role in ["polecat", "crew"]:
            spec = AgentSpec(provider="anthropic-api", hierarchy_role=role)
            assert spec.is_worker() is True
        spec_mayor = AgentSpec(provider="anthropic-api", hierarchy_role="mayor")
        assert spec_mayor.is_worker() is False

    def test_is_ephemeral(self):
        """is_ephemeral should return True only for polecat role."""
        spec_polecat = AgentSpec(provider="anthropic-api", hierarchy_role="polecat")
        assert spec_polecat.is_ephemeral() is True
        spec_crew = AgentSpec(provider="anthropic-api", hierarchy_role="crew")
        assert spec_crew.is_ephemeral() is False

    def test_has_gastown_capability(self):
        """has_gastown_capability should check role capabilities."""
        spec_mayor = AgentSpec(provider="anthropic-api", hierarchy_role="mayor")
        assert spec_mayor.has_gastown_capability("create_convoy") is True
        assert spec_mayor.has_gastown_capability("execute_task") is False

        spec_crew = AgentSpec(provider="anthropic-api", hierarchy_role="crew")
        assert spec_crew.has_gastown_capability("execute_task") is True
        assert spec_crew.has_gastown_capability("create_convoy") is False


class TestTeamSelectionConfigHierarchy:
    """Tests for hierarchy role config in TeamSelectionConfig."""

    def test_hierarchy_filtering_disabled_by_default(self):
        """Hierarchy filtering should be disabled by default."""
        config = TeamSelectionConfig()
        assert config.enable_hierarchy_filtering is False

    def test_hierarchy_fallback_enabled_by_default(self):
        """Hierarchy fallback should be enabled by default."""
        config = TeamSelectionConfig()
        assert config.hierarchy_filter_fallback is True


class TestTeamSelectorHierarchyFiltering:
    """Tests for TeamSelector hierarchy role filtering."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents with different hierarchy roles."""
        agents = []

        # Mayor agent
        mayor = MagicMock()
        mayor.name = "claude-mayor"
        mayor.hierarchy_role = "mayor"
        agents.append(mayor)

        # Witness agent
        witness = MagicMock()
        witness.name = "gemini-witness"
        witness.hierarchy_role = "witness"
        agents.append(witness)

        # Crew agents
        crew1 = MagicMock()
        crew1.name = "gpt-crew-1"
        crew1.hierarchy_role = "crew"
        agents.append(crew1)

        crew2 = MagicMock()
        crew2.name = "mistral-crew-2"
        crew2.hierarchy_role = "crew"
        agents.append(crew2)

        # Polecat agent
        polecat = MagicMock()
        polecat.name = "deepseek-polecat"
        polecat.hierarchy_role = "polecat"
        agents.append(polecat)

        # Agent without hierarchy role
        no_role = MagicMock()
        no_role.name = "llama-worker"
        no_role.hierarchy_role = None
        agents.append(no_role)

        return agents

    def test_no_filtering_when_disabled(self, mock_agents):
        """Should return all agents when hierarchy filtering is disabled."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=False)
        selector = TeamSelector(config=config)

        result = selector._filter_by_hierarchy_role(mock_agents, {"mayor"})
        assert len(result) == len(mock_agents)

    def test_no_filtering_when_no_roles_specified(self, mock_agents):
        """Should return all agents when no required roles specified."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        result = selector._filter_by_hierarchy_role(mock_agents, None)
        assert len(result) == len(mock_agents)

        result2 = selector._filter_by_hierarchy_role(mock_agents, set())
        assert len(result2) == len(mock_agents)

    def test_filter_by_single_role(self, mock_agents):
        """Should filter to agents with specific role."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        # Filter to mayors only
        result = selector._filter_by_hierarchy_role(mock_agents, {"mayor"})
        assert len(result) == 1
        assert result[0].name == "claude-mayor"

        # Filter to crew only
        result = selector._filter_by_hierarchy_role(mock_agents, {"crew"})
        assert len(result) == 2
        assert all(a.hierarchy_role == "crew" for a in result)

    def test_filter_by_multiple_roles(self, mock_agents):
        """Should filter to agents with any of the specified roles."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        # Filter to mayor or crew
        result = selector._filter_by_hierarchy_role(mock_agents, {"mayor", "crew"})
        assert len(result) == 3
        roles = {a.hierarchy_role for a in result}
        assert roles == {"mayor", "crew"}

    def test_fallback_when_no_matches(self, mock_agents):
        """Should fall back to all agents when no matches and fallback enabled."""
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            hierarchy_filter_fallback=True,
        )
        selector = TeamSelector(config=config)

        # Filter to nonexistent role
        result = selector._filter_by_hierarchy_role(mock_agents, {"nonexistent"})
        assert len(result) == len(mock_agents)

    def test_empty_when_no_matches_and_fallback_disabled(self, mock_agents):
        """Should return empty when no matches and fallback disabled."""
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            hierarchy_filter_fallback=False,
        )
        selector = TeamSelector(config=config)

        # Filter to nonexistent role
        result = selector._filter_by_hierarchy_role(mock_agents, {"nonexistent"})
        assert len(result) == 0

    def test_case_insensitive_matching(self, mock_agents):
        """Should match roles case-insensitively."""
        config = TeamSelectionConfig(enable_hierarchy_filtering=True)
        selector = TeamSelector(config=config)

        result = selector._filter_by_hierarchy_role(mock_agents, {"MAYOR", "CREW"})
        assert len(result) == 3

    def test_select_with_hierarchy_roles(self, mock_agents):
        """Should integrate hierarchy filtering into full select flow."""
        config = TeamSelectionConfig(
            enable_hierarchy_filtering=True,
            enable_domain_filtering=False,
        )
        selector = TeamSelector(config=config)

        result = selector.select(
            mock_agents,
            required_hierarchy_roles={"crew", "polecat"},
        )
        # Should include crew (2) and polecat (1)
        assert len(result) == 3
        roles = {a.hierarchy_role for a in result}
        assert roles == {"crew", "polecat"}


class TestAgentHierarchyRoleDetection:
    """Tests for _get_agent_hierarchy_role method."""

    def test_direct_attribute(self):
        """Should detect hierarchy_role from direct attribute."""
        agent = MagicMock()
        agent.hierarchy_role = "mayor"

        selector = TeamSelector()
        assert selector._get_agent_hierarchy_role(agent) == "mayor"

    def test_from_spec(self):
        """Should detect hierarchy_role from agent.spec."""
        agent = MagicMock(spec=[])  # Clear default spec
        agent.hierarchy_role = None
        agent.spec = MagicMock()
        agent.spec.hierarchy_role = "witness"

        selector = TeamSelector()
        assert selector._get_agent_hierarchy_role(agent) == "witness"

    def test_from_metadata(self):
        """Should detect hierarchy_role from agent.metadata."""
        agent = MagicMock(spec=[])
        agent.hierarchy_role = None
        agent.spec = None
        agent.metadata = {"hierarchy_role": "crew"}

        selector = TeamSelector()
        assert selector._get_agent_hierarchy_role(agent) == "crew"

    def test_none_when_not_found(self):
        """Should return None when hierarchy_role not found."""
        agent = MagicMock(spec=[])
        agent.hierarchy_role = None
        agent.spec = None
        agent.metadata = {}

        selector = TeamSelector()
        assert selector._get_agent_hierarchy_role(agent) is None


class TestRoleCapabilitiesIntegration:
    """Tests for role capabilities integration."""

    def test_role_capabilities_defined(self):
        """All roles should have defined capabilities."""
        from aragora.nomic.agent_roles import ROLE_CAPABILITIES, AgentRole

        for role in AgentRole:
            assert role in ROLE_CAPABILITIES
            assert len(ROLE_CAPABILITIES[role]) > 0

    def test_mayor_has_coordinator_capabilities(self):
        """Mayor role should have coordination capabilities."""
        from aragora.nomic.agent_roles import ROLE_CAPABILITIES, AgentRole, RoleCapability

        mayor_caps = ROLE_CAPABILITIES[AgentRole.MAYOR]
        assert RoleCapability.CREATE_CONVOY in mayor_caps
        assert RoleCapability.ASSIGN_WORK in mayor_caps
        assert RoleCapability.COORDINATE in mayor_caps

    def test_witness_has_monitoring_capabilities(self):
        """Witness role should have monitoring capabilities."""
        from aragora.nomic.agent_roles import ROLE_CAPABILITIES, AgentRole, RoleCapability

        witness_caps = ROLE_CAPABILITIES[AgentRole.WITNESS]
        assert RoleCapability.MONITOR_AGENTS in witness_caps
        assert RoleCapability.DETECT_STUCK in witness_caps

    def test_crew_has_worker_capabilities(self):
        """Crew role should have worker capabilities."""
        from aragora.nomic.agent_roles import ROLE_CAPABILITIES, AgentRole, RoleCapability

        crew_caps = ROLE_CAPABILITIES[AgentRole.CREW]
        assert RoleCapability.EXECUTE_TASK in crew_caps
        assert RoleCapability.CLAIM_BEAD in crew_caps
        assert RoleCapability.MAINTAIN_CONTEXT in crew_caps

    def test_polecat_has_limited_capabilities(self):
        """Polecat role should have basic worker capabilities only."""
        from aragora.nomic.agent_roles import ROLE_CAPABILITIES, AgentRole, RoleCapability

        polecat_caps = ROLE_CAPABILITIES[AgentRole.POLECAT]
        assert RoleCapability.EXECUTE_TASK in polecat_caps
        # Polecat shouldn't have context maintenance
        assert RoleCapability.MAINTAIN_CONTEXT not in polecat_caps

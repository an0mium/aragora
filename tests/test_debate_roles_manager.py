"""
Tests for aragora.debate.roles_manager module.

Tests RolesManager class which handles role assignment, stance assignment,
and agreement intensity for debate agents.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.debate.roles_manager import RolesManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_protocol():
    """Create a mock DebateProtocol."""
    protocol = MagicMock()
    protocol.proposer_count = 1
    protocol.asymmetric_stances = False
    protocol.rotate_stances = False
    protocol.role_rotation = False
    protocol.role_matching = False
    protocol.role_rotation_config = None
    protocol.role_matching_config = None
    protocol.agreement_intensity = None
    return protocol


@pytest.fixture
def mock_agents():
    """Create a list of mock agents."""
    agents = []
    for i, name in enumerate(["claude", "gpt", "gemini"]):
        agent = MagicMock()
        agent.name = name
        agent.role = ""
        agent.stance = ""
        agent.system_prompt = ""
        agents.append(agent)
    return agents


@pytest.fixture
def mock_prompt_builder():
    """Create a mock PromptBuilder."""
    builder = MagicMock()
    builder.get_stance_guidance = MagicMock(return_value="Stance guidance text")
    return builder


@pytest.fixture
def roles_manager(mock_agents, mock_protocol):
    """Create a basic RolesManager."""
    with patch.object(RolesManager, "_init_role_systems"):
        manager = RolesManager(
            agents=mock_agents,
            protocol=mock_protocol,
        )
        manager.role_rotator = None
        manager.role_matcher = None
        manager.current_role_assignments = {}
        return manager


# ============================================================================
# Initialization Tests
# ============================================================================


class TestRolesManagerInit:
    """Tests for RolesManager initialization."""

    def test_basic_init(self, mock_agents, mock_protocol):
        """Test basic initialization."""
        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=mock_agents,
                protocol=mock_protocol,
            )
            assert manager.agents == mock_agents
            assert manager.protocol == mock_protocol
            assert manager.prompt_builder is None

    def test_init_with_prompt_builder(self, mock_agents, mock_protocol, mock_prompt_builder):
        """Test initialization with prompt builder."""
        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=mock_agents,
                protocol=mock_protocol,
                prompt_builder=mock_prompt_builder,
            )
            assert manager.prompt_builder == mock_prompt_builder

    def test_init_with_calibration_tracker(self, mock_agents, mock_protocol):
        """Test initialization with calibration tracker."""
        mock_tracker = MagicMock()
        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=mock_agents,
                protocol=mock_protocol,
                calibration_tracker=mock_tracker,
            )
            assert manager.calibration_tracker == mock_tracker

    def test_init_with_persona_manager(self, mock_agents, mock_protocol):
        """Test initialization with persona manager."""
        mock_personas = MagicMock()
        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=mock_agents,
                protocol=mock_protocol,
                persona_manager=mock_personas,
            )
            assert manager.persona_manager == mock_personas

    def test_init_calls_init_role_systems(self, mock_agents, mock_protocol):
        """Test that initialization calls _init_role_systems."""
        mock_protocol.role_rotation = False
        mock_protocol.role_matching = False

        manager = RolesManager(
            agents=mock_agents,
            protocol=mock_protocol,
        )
        # Should complete without error
        assert manager is not None


# ============================================================================
# Init Role Systems Tests
# ============================================================================


class TestInitRoleSystems:
    """Tests for _init_role_systems method."""

    def test_role_rotation_enabled(self, mock_agents, mock_protocol):
        """Test role rotator is created when role_rotation enabled."""
        mock_protocol.role_rotation = True
        mock_protocol.role_matching = False

        manager = RolesManager(
            agents=mock_agents,
            protocol=mock_protocol,
        )
        assert manager.role_rotator is not None

    def test_role_matching_enabled(self, mock_agents, mock_protocol):
        """Test role matcher is created when role_matching enabled."""
        mock_protocol.role_rotation = False
        mock_protocol.role_matching = True

        manager = RolesManager(
            agents=mock_agents,
            protocol=mock_protocol,
        )
        assert manager.role_matcher is not None

    def test_role_matching_priority_over_rotation(self, mock_agents, mock_protocol):
        """Test role matching takes priority over rotation."""
        mock_protocol.role_rotation = True
        mock_protocol.role_matching = True

        manager = RolesManager(
            agents=mock_agents,
            protocol=mock_protocol,
        )
        # Role matcher should be created, rotator should not
        assert manager.role_matcher is not None
        assert manager.role_rotator is None


# ============================================================================
# Assign Initial Roles Tests
# ============================================================================


class TestAssignInitialRoles:
    """Tests for assign_initial_roles method."""

    def test_assigns_roles(self, roles_manager, mock_agents):
        """Test basic role assignment."""
        for agent in mock_agents:
            agent.role = ""

        roles_manager.assign_initial_roles()

        # Should assign roles
        assigned_roles = [a.role for a in mock_agents]
        assert "proposer" in assigned_roles
        assert "critic" in assigned_roles
        assert "synthesizer" in assigned_roles

    def test_respects_existing_roles(self, roles_manager, mock_agents):
        """Test existing roles are not overwritten."""
        for agent in mock_agents:
            agent.role = "existing_role"

        roles_manager.assign_initial_roles()

        # All should keep existing roles
        for agent in mock_agents:
            assert agent.role == "existing_role"

    def test_first_agent_is_proposer(self, roles_manager, mock_agents):
        """Test first agent gets proposer role."""
        for agent in mock_agents:
            agent.role = ""

        roles_manager.assign_initial_roles()

        assert mock_agents[0].role == "proposer"

    def test_last_agent_is_synthesizer(self, roles_manager, mock_agents):
        """Test last agent gets synthesizer role."""
        for agent in mock_agents:
            agent.role = ""

        roles_manager.assign_initial_roles()

        assert mock_agents[-1].role == "synthesizer"

    def test_middle_agents_are_critics(self, roles_manager, mock_agents):
        """Test middle agents get critic role."""
        for agent in mock_agents:
            agent.role = ""

        roles_manager.assign_initial_roles()

        # Middle agent (index 1) should be critic
        assert mock_agents[1].role == "critic"

    def test_max_proposers_capped(self, roles_manager, mock_agents):
        """Test proposer count is capped to ensure critic and synthesizer."""
        for agent in mock_agents:
            agent.role = ""
        roles_manager.protocol.proposer_count = 10  # More than available

        roles_manager.assign_initial_roles()

        # Should still have critic and synthesizer
        roles = [a.role for a in mock_agents]
        assert "critic" in roles
        assert "synthesizer" in roles


# ============================================================================
# Assign Stances Tests
# ============================================================================


class TestAssignStances:
    """Tests for assign_stances method."""

    def test_no_asymmetric_stances(self, roles_manager, mock_agents):
        """Test no stances assigned when asymmetric_stances disabled."""
        roles_manager.protocol.asymmetric_stances = False

        roles_manager.assign_stances()

        # Stances should be unchanged (empty)
        for agent in mock_agents:
            assert agent.stance == ""

    def test_assigns_stances(self, roles_manager, mock_agents):
        """Test stances assigned when enabled."""
        roles_manager.protocol.asymmetric_stances = True

        roles_manager.assign_stances()

        stances = [a.stance for a in mock_agents]
        assert "affirmative" in stances
        assert "negative" in stances
        assert "neutral" in stances

    def test_first_agent_affirmative(self, roles_manager, mock_agents):
        """Test first agent gets affirmative stance."""
        roles_manager.protocol.asymmetric_stances = True

        roles_manager.assign_stances()

        assert mock_agents[0].stance == "affirmative"

    def test_stance_rotation(self, roles_manager, mock_agents):
        """Test stances rotate with round number."""
        roles_manager.protocol.asymmetric_stances = True
        roles_manager.protocol.rotate_stances = True

        # Round 0
        roles_manager.assign_stances(round_num=0)
        round0_stances = [a.stance for a in mock_agents]

        # Round 1 - should be rotated
        roles_manager.assign_stances(round_num=1)
        round1_stances = [a.stance for a in mock_agents]

        assert round0_stances != round1_stances


# ============================================================================
# Apply Agreement Intensity Tests
# ============================================================================


class TestApplyAgreementIntensity:
    """Tests for apply_agreement_intensity method."""

    def test_no_intensity_set(self, roles_manager, mock_agents):
        """Test no guidance added when intensity not set."""
        roles_manager.protocol.agreement_intensity = None

        roles_manager.apply_agreement_intensity()

        # System prompts should be unchanged or empty
        for agent in mock_agents:
            assert agent.system_prompt == ""

    def test_appends_to_existing_prompt(self, roles_manager, mock_agents):
        """Test guidance appended to existing system prompt."""
        roles_manager.protocol.agreement_intensity = 5
        mock_agents[0].system_prompt = "Existing prompt"

        roles_manager.apply_agreement_intensity()

        assert "Existing prompt" in mock_agents[0].system_prompt
        assert "merits" in mock_agents[0].system_prompt

    def test_sets_prompt_when_empty(self, roles_manager, mock_agents):
        """Test guidance set when system prompt is empty."""
        roles_manager.protocol.agreement_intensity = 5
        mock_agents[0].system_prompt = ""

        roles_manager.apply_agreement_intensity()

        assert len(mock_agents[0].system_prompt) > 0


# ============================================================================
# Get Agreement Intensity Guidance Tests
# ============================================================================


class TestGetAgreementIntensityGuidance:
    """Tests for _get_agreement_intensity_guidance method."""

    def test_no_intensity(self, roles_manager):
        """Test returns empty when intensity not set."""
        roles_manager.protocol.agreement_intensity = None
        result = roles_manager._get_agreement_intensity_guidance()
        assert result == ""

    def test_very_low_intensity(self, roles_manager):
        """Test adversarial guidance for intensity 0-1."""
        roles_manager.protocol.agreement_intensity = 1
        result = roles_manager._get_agreement_intensity_guidance()
        assert "strongly disagree" in result

    def test_low_intensity(self, roles_manager):
        """Test skeptical guidance for intensity 2-3."""
        roles_manager.protocol.agreement_intensity = 3
        result = roles_manager._get_agreement_intensity_guidance()
        assert "skepticism" in result

    def test_medium_intensity(self, roles_manager):
        """Test balanced guidance for intensity 4-6."""
        roles_manager.protocol.agreement_intensity = 5
        result = roles_manager._get_agreement_intensity_guidance()
        assert "merits" in result

    def test_high_intensity(self, roles_manager):
        """Test collaborative guidance for intensity 7-8."""
        roles_manager.protocol.agreement_intensity = 8
        result = roles_manager._get_agreement_intensity_guidance()
        assert "common ground" in result

    def test_very_high_intensity(self, roles_manager):
        """Test synthesis guidance for intensity 9-10."""
        roles_manager.protocol.agreement_intensity = 10
        result = roles_manager._get_agreement_intensity_guidance()
        assert "collaborative" in result


# ============================================================================
# Get Stance Guidance Tests
# ============================================================================


class TestGetStanceGuidance:
    """Tests for get_stance_guidance method."""

    def test_with_prompt_builder(self, roles_manager, mock_agents, mock_prompt_builder):
        """Test delegates to prompt builder when available."""
        roles_manager.prompt_builder = mock_prompt_builder
        result = roles_manager.get_stance_guidance(mock_agents[0])

        assert result == "Stance guidance text"
        mock_prompt_builder.get_stance_guidance.assert_called_once()

    def test_fallback_affirmative(self, roles_manager, mock_agents):
        """Test fallback guidance for affirmative stance."""
        mock_agents[0].stance = "affirmative"
        result = roles_manager.get_stance_guidance(mock_agents[0])

        assert "IN FAVOR" in result

    def test_fallback_negative(self, roles_manager, mock_agents):
        """Test fallback guidance for negative stance."""
        mock_agents[0].stance = "negative"
        result = roles_manager.get_stance_guidance(mock_agents[0])

        assert "AGAINST" in result

    def test_fallback_neutral(self, roles_manager, mock_agents):
        """Test fallback guidance for neutral stance."""
        mock_agents[0].stance = "neutral"
        result = roles_manager.get_stance_guidance(mock_agents[0])

        assert "NEUTRAL" in result


# ============================================================================
# Rotate Roles for Round Tests
# ============================================================================


class TestRotateRolesForRound:
    """Tests for rotate_roles_for_round method."""

    def test_no_role_rotator(self, roles_manager, mock_agents):
        """Test does nothing without role_rotator."""
        roles_manager.role_rotator = None

        roles_manager.rotate_roles_for_round(1)

        assert roles_manager.current_role_assignments == {}

    def test_with_role_rotator(self, roles_manager, mock_agents):
        """Test calls role_rotator.rotate when available."""
        mock_rotator = MagicMock()
        mock_rotator.rotate = MagicMock(return_value={"claude": "assignment"})
        roles_manager.role_rotator = mock_rotator

        roles_manager.rotate_roles_for_round(1)

        mock_rotator.rotate.assert_called_once_with(mock_agents, 1)
        assert roles_manager.current_role_assignments == {"claude": "assignment"}


# ============================================================================
# Match Roles for Task Tests
# ============================================================================


class TestMatchRolesForTask:
    """Tests for match_roles_for_task method."""

    def test_no_role_matcher(self, roles_manager):
        """Test returns empty dict without role_matcher."""
        roles_manager.role_matcher = None

        result = roles_manager.match_roles_for_task("Test task")

        assert result == {}

    def test_with_role_matcher(self, roles_manager, mock_agents):
        """Test calls role_matcher.match when available."""
        mock_matcher = MagicMock()
        mock_matcher.match = MagicMock(return_value={"claude": "matched_role"})
        roles_manager.role_matcher = mock_matcher

        result = roles_manager.match_roles_for_task("Test task")

        mock_matcher.match.assert_called_once_with(mock_agents, "Test task")
        assert result == {"claude": "matched_role"}
        assert roles_manager.current_role_assignments == {"claude": "matched_role"}


# ============================================================================
# Integration Tests
# ============================================================================


class TestRolesManagerIntegration:
    """Integration tests for RolesManager."""

    def test_full_initialization_flow(self, mock_agents, mock_protocol):
        """Test full initialization with role assignment and stances."""
        mock_protocol.asymmetric_stances = True

        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=mock_agents,
                protocol=mock_protocol,
            )
            manager.role_rotator = None
            manager.role_matcher = None

        # Clear existing roles
        for agent in mock_agents:
            agent.role = ""

        manager.assign_initial_roles()
        manager.assign_stances()

        # Verify all agents have roles and stances
        for agent in mock_agents:
            assert agent.role != ""
            assert agent.stance != ""

    def test_multiple_rounds_with_stance_rotation(self, roles_manager, mock_agents):
        """Test stance rotation across multiple rounds."""
        roles_manager.protocol.asymmetric_stances = True
        roles_manager.protocol.rotate_stances = True

        round_stances = []
        for round_num in range(3):
            roles_manager.assign_stances(round_num=round_num)
            stances = tuple(a.stance for a in mock_agents)
            round_stances.append(stances)

        # Each round should have different stance distribution
        assert len(set(round_stances)) > 1


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestRolesManagerEdgeCases:
    """Tests for edge cases."""

    def test_single_agent(self, mock_protocol):
        """Test with single agent."""
        single_agent = MagicMock()
        single_agent.name = "claude"
        single_agent.role = ""
        single_agent.stance = ""
        single_agent.system_prompt = ""

        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=[single_agent],
                protocol=mock_protocol,
            )
            manager.role_rotator = None

        manager.assign_initial_roles()

        # Single agent should get proposer role
        assert single_agent.role == "proposer"

    def test_two_agents(self, mock_protocol):
        """Test with two agents."""
        agents = [
            MagicMock(name=f"agent-{i}", role="", stance="", system_prompt="") for i in range(2)
        ]
        for i, agent in enumerate(agents):
            agent.name = f"agent-{i}"
            agent.role = ""
            agent.stance = ""
            agent.system_prompt = ""

        with patch.object(RolesManager, "_init_role_systems"):
            manager = RolesManager(
                agents=agents,
                protocol=mock_protocol,
            )
            manager.role_rotator = None

        manager.assign_initial_roles()

        # Should have proposer and synthesizer
        roles = [a.role for a in agents]
        assert "proposer" in roles
        assert "synthesizer" in roles


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports RolesManager."""
        from aragora.debate import roles_manager

        assert "RolesManager" in roles_manager.__all__

    def test_import(self):
        """Test RolesManager can be imported."""
        from aragora.debate.roles_manager import RolesManager as RM

        assert RM is RolesManager

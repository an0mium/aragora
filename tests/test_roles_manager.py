"""
Tests for RolesManager in debate phases.

Tests role assignment, stance assignment, and agreement intensity guidance.
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

from aragora.debate.phases.roles_manager import RolesManager


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: Optional[str] = None
    stance: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclass
class MockProtocol:
    """Mock protocol for testing."""

    proposer_count: int = 1
    asymmetric_stances: bool = False
    rotate_stances: bool = False
    agreement_intensity: int = 5


@pytest.fixture
def three_agents():
    """Three mock agents without roles."""
    return [
        MockAgent(name="agent1"),
        MockAgent(name="agent2"),
        MockAgent(name="agent3"),
    ]


@pytest.fixture
def five_agents():
    """Five mock agents without roles."""
    return [
        MockAgent(name="agent1"),
        MockAgent(name="agent2"),
        MockAgent(name="agent3"),
        MockAgent(name="agent4"),
        MockAgent(name="agent5"),
    ]


@pytest.fixture
def protocol():
    """Default protocol."""
    return MockProtocol()


# =============================================================================
# RolesManager Initialization Tests
# =============================================================================


class TestRolesManagerInit:
    """Tests for RolesManager initialization."""

    def test_init_stores_protocol(self, protocol, three_agents):
        """Should store protocol reference."""
        manager = RolesManager(protocol, three_agents)
        assert manager.protocol is protocol

    def test_init_stores_agents(self, protocol, three_agents):
        """Should store agents list."""
        manager = RolesManager(protocol, three_agents)
        assert manager.agents == three_agents


# =============================================================================
# Role Assignment Tests
# =============================================================================


class TestAssignRoles:
    """Tests for assign_roles method."""

    def test_assigns_proposer_first(self, protocol, three_agents):
        """First agent should be proposer."""
        manager = RolesManager(protocol, three_agents)
        manager.assign_roles()
        assert three_agents[0].role == "proposer"

    def test_assigns_synthesizer_last(self, protocol, three_agents):
        """Last agent should be synthesizer."""
        manager = RolesManager(protocol, three_agents)
        manager.assign_roles()
        assert three_agents[-1].role == "synthesizer"

    def test_assigns_critic_middle(self, protocol, three_agents):
        """Middle agents should be critics."""
        manager = RolesManager(protocol, three_agents)
        manager.assign_roles()
        assert three_agents[1].role == "critic"

    def test_respects_existing_roles(self, protocol):
        """Should not override existing roles."""
        agents = [
            MockAgent(name="a1", role="proposer"),
            MockAgent(name="a2", role="critic"),
            MockAgent(name="a3", role="synthesizer"),
        ]
        manager = RolesManager(protocol, agents)
        manager.assign_roles()
        # Should keep existing roles
        assert agents[0].role == "proposer"
        assert agents[1].role == "critic"
        assert agents[2].role == "synthesizer"

    def test_multiple_proposers_with_high_count(self, five_agents):
        """Should assign multiple proposers when proposer_count > 1."""
        protocol = MockProtocol(proposer_count=2)
        manager = RolesManager(protocol, five_agents)
        manager.assign_roles()

        proposers = [a for a in five_agents if a.role == "proposer"]
        assert len(proposers) == 2
        assert five_agents[0].role == "proposer"
        assert five_agents[1].role == "proposer"

    def test_safety_bounds_max_proposers(self, three_agents):
        """Should cap proposers to ensure at least 1 critic and 1 synthesizer."""
        protocol = MockProtocol(proposer_count=10)  # Requesting more than possible
        manager = RolesManager(protocol, three_agents)
        manager.assign_roles()

        # With 3 agents, max 1 proposer to ensure critic + synthesizer
        proposers = [a for a in three_agents if a.role == "proposer"]
        critics = [a for a in three_agents if a.role == "critic"]
        synthesizers = [a for a in three_agents if a.role == "synthesizer"]

        assert len(proposers) == 1
        assert len(critics) >= 1
        assert len(synthesizers) == 1

    def test_two_agents_distribution(self):
        """With 2 agents, should have proposer and synthesizer."""
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]
        protocol = MockProtocol(proposer_count=1)
        manager = RolesManager(protocol, agents)
        manager.assign_roles()

        assert agents[0].role == "proposer"
        assert agents[1].role == "synthesizer"

    def test_single_agent(self):
        """Single agent should be proposer."""
        agents = [MockAgent(name="solo")]
        protocol = MockProtocol()
        manager = RolesManager(protocol, agents)
        manager.assign_roles()

        assert agents[0].role == "proposer"


# =============================================================================
# Stance Assignment Tests
# =============================================================================


class TestAssignStances:
    """Tests for assign_stances method."""

    def test_no_stances_when_disabled(self, protocol, three_agents):
        """Should not assign stances when asymmetric_stances is False."""
        protocol.asymmetric_stances = False
        manager = RolesManager(protocol, three_agents)
        manager.assign_stances()

        for agent in three_agents:
            assert agent.stance is None

    def test_assigns_stances_when_enabled(self, three_agents):
        """Should assign stances when asymmetric_stances is True."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        manager.assign_stances()

        stances = [a.stance for a in three_agents]
        assert "affirmative" in stances
        assert "negative" in stances
        assert "neutral" in stances

    def test_stance_rotation(self, three_agents):
        """Stances should rotate when rotate_stances is enabled."""
        protocol = MockProtocol(asymmetric_stances=True, rotate_stances=True)
        manager = RolesManager(protocol, three_agents)

        # Round 0
        manager.assign_stances(round_num=0)
        stances_r0 = [a.stance for a in three_agents]

        # Round 1
        manager.assign_stances(round_num=1)
        stances_r1 = [a.stance for a in three_agents]

        # Stances should have rotated
        assert stances_r0 != stances_r1

    def test_no_rotation_when_disabled(self, three_agents):
        """Stances should stay same when rotate_stances is False."""
        protocol = MockProtocol(asymmetric_stances=True, rotate_stances=False)
        manager = RolesManager(protocol, three_agents)

        # Round 0
        manager.assign_stances(round_num=0)
        stances_r0 = [a.stance for a in three_agents]

        # Round 1
        manager.assign_stances(round_num=1)
        stances_r1 = [a.stance for a in three_agents]

        # Stances should be the same
        assert stances_r0 == stances_r1


# =============================================================================
# Agreement Intensity Tests
# =============================================================================


class TestAgreementIntensity:
    """Tests for agreement intensity guidance."""

    def test_adversarial_mode_low_intensity(self, three_agents):
        """Low intensity (0-3) should produce adversarial guidance."""
        protocol = MockProtocol(agreement_intensity=2)
        manager = RolesManager(protocol, three_agents)

        guidance = manager.get_agreement_intensity_guidance()
        assert "ADVERSARIAL" in guidance
        assert "challenge" in guidance.lower()

    def test_collaborative_mode_high_intensity(self, three_agents):
        """High intensity (7-10) should produce collaborative guidance."""
        protocol = MockProtocol(agreement_intensity=8)
        manager = RolesManager(protocol, three_agents)

        guidance = manager.get_agreement_intensity_guidance()
        assert "COLLABORATIVE" in guidance
        assert "common ground" in guidance.lower()

    def test_balanced_mode_medium_intensity(self, three_agents):
        """Medium intensity (4-6) should produce balanced guidance."""
        protocol = MockProtocol(agreement_intensity=5)
        manager = RolesManager(protocol, three_agents)

        guidance = manager.get_agreement_intensity_guidance()
        assert "BALANCED" in guidance
        assert "merits" in guidance.lower()

    def test_boundary_value_intensity_3(self, three_agents):
        """Intensity 3 should be adversarial (boundary)."""
        protocol = MockProtocol(agreement_intensity=3)
        manager = RolesManager(protocol, three_agents)

        guidance = manager.get_agreement_intensity_guidance()
        assert "ADVERSARIAL" in guidance

    def test_boundary_value_intensity_7(self, three_agents):
        """Intensity 7 should be collaborative (boundary)."""
        protocol = MockProtocol(agreement_intensity=7)
        manager = RolesManager(protocol, three_agents)

        guidance = manager.get_agreement_intensity_guidance()
        assert "COLLABORATIVE" in guidance

    def test_apply_agreement_intensity_modifies_prompts(self, three_agents):
        """apply_agreement_intensity should modify agent system prompts."""
        protocol = MockProtocol(agreement_intensity=2)
        three_agents[0].system_prompt = "Base prompt"
        manager = RolesManager(protocol, three_agents)

        manager.apply_agreement_intensity()

        assert "ADVERSARIAL" in three_agents[0].system_prompt
        assert "Base prompt" in three_agents[0].system_prompt

    def test_apply_agreement_intensity_sets_prompt_if_none(self, three_agents):
        """Should set prompt even if agent has no system_prompt."""
        protocol = MockProtocol(agreement_intensity=8)
        manager = RolesManager(protocol, three_agents)

        manager.apply_agreement_intensity()

        assert three_agents[0].system_prompt is not None
        assert "COLLABORATIVE" in three_agents[0].system_prompt


# =============================================================================
# Stance Guidance Tests
# =============================================================================


class TestStanceGuidance:
    """Tests for stance-specific guidance."""

    def test_affirmative_stance_guidance(self, three_agents):
        """Should generate affirmative stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        three_agents[0].stance = "affirmative"

        guidance = manager.get_stance_guidance(three_agents[0])

        assert "AFFIRMATIVE" in guidance
        assert "DEFEND" in guidance or "SUPPORT" in guidance

    def test_negative_stance_guidance(self, three_agents):
        """Should generate negative stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        three_agents[0].stance = "negative"

        guidance = manager.get_stance_guidance(three_agents[0])

        assert "NEGATIVE" in guidance
        assert "CHALLENGE" in guidance or "CRITIQUE" in guidance

    def test_neutral_stance_guidance(self, three_agents):
        """Should generate neutral stance guidance."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        three_agents[0].stance = "neutral"

        guidance = manager.get_stance_guidance(three_agents[0])

        assert "NEUTRAL" in guidance
        assert "EVALUATE" in guidance

    def test_no_guidance_when_stances_disabled(self, three_agents):
        """Should return empty guidance when asymmetric_stances is False."""
        protocol = MockProtocol(asymmetric_stances=False)
        manager = RolesManager(protocol, three_agents)
        three_agents[0].stance = "affirmative"

        guidance = manager.get_stance_guidance(three_agents[0])

        assert guidance == ""

    def test_no_guidance_when_no_stance(self, three_agents):
        """Should return empty guidance when agent has no stance."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        # three_agents[0].stance is None

        guidance = manager.get_stance_guidance(three_agents[0])

        assert guidance == ""


# =============================================================================
# Summary Tests
# =============================================================================


class TestRoleSummary:
    """Tests for role summary generation."""

    def test_get_role_summary(self, protocol, three_agents):
        """Should return correct role summary."""
        manager = RolesManager(protocol, three_agents)
        manager.assign_roles()

        summary = manager.get_role_summary()

        assert "proposer" in summary
        assert "critic" in summary
        assert "synthesizer" in summary
        assert "agent1" in summary["proposer"]
        assert "agent2" in summary["critic"]
        assert "agent3" in summary["synthesizer"]

    def test_role_summary_handles_unknown_roles(self, protocol):
        """Should handle unknown roles gracefully."""
        agents = [MockAgent(name="a1", role="custom_role")]
        manager = RolesManager(protocol, agents)

        summary = manager.get_role_summary()

        assert "custom_role" in summary
        assert "a1" in summary["custom_role"]


class TestStanceSummary:
    """Tests for stance summary generation."""

    def test_get_stance_summary(self, three_agents):
        """Should return correct stance summary."""
        protocol = MockProtocol(asymmetric_stances=True)
        manager = RolesManager(protocol, three_agents)
        manager.assign_stances()

        summary = manager.get_stance_summary()

        assert "affirmative" in summary
        assert "negative" in summary
        assert "neutral" in summary

        # Each stance should have exactly one agent
        total_assigned = (
            len(summary["affirmative"]) + len(summary["negative"]) + len(summary["neutral"])
        )
        assert total_assigned == 3

    def test_stance_summary_with_no_stances(self, protocol, three_agents):
        """Should return empty lists when no stances assigned."""
        manager = RolesManager(protocol, three_agents)

        summary = manager.get_stance_summary()

        assert summary["affirmative"] == []
        assert summary["negative"] == []
        assert summary["neutral"] == []

"""Tests for aragora.debate.roles_manager — RolesManager."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs to avoid importing heavy Arena/Protocol dependencies
# ---------------------------------------------------------------------------


@dataclass
class FakeAgent:
    """Minimal Agent stand-in."""

    name: str = "claude"
    role: str = ""
    stance: str = ""
    system_prompt: str = ""


@dataclass
class FakeProtocol:
    """Minimal DebateProtocol stand-in."""

    proposer_count: int = 1
    rounds: int = 3
    asymmetric_stances: bool = False
    rotate_stances: bool = False
    agreement_intensity: int | None = None
    role_rotation: bool = False
    role_rotation_config: object = None
    role_matching: bool = False
    role_matching_config: object = None


# ---------------------------------------------------------------------------
# RolesManager import — patch role systems init to avoid heavy imports
# ---------------------------------------------------------------------------


@pytest.fixture
def roles_manager_cls():
    """Import RolesManager with role system init patched."""
    from aragora.debate.roles_manager import RolesManager

    return RolesManager


def _make_manager(agents, protocol=None, **kwargs):
    """Create a RolesManager with _init_role_systems patched out."""
    from aragora.debate.roles_manager import RolesManager

    protocol = protocol or FakeProtocol()
    with patch.object(RolesManager, "_init_role_systems"):
        mgr = RolesManager(agents=agents, protocol=protocol, **kwargs)
    mgr.role_rotator = None
    mgr.role_matcher = None
    mgr.current_role_assignments = {}
    return mgr


# ---------------------------------------------------------------------------
# assign_initial_roles
# ---------------------------------------------------------------------------


class TestAssignInitialRoles:
    def test_single_agent_gets_proposer(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents, FakeProtocol(proposer_count=1))
        mgr.assign_initial_roles()
        assert agents[0].role == "proposer"

    def test_two_agents(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents, FakeProtocol(proposer_count=1))
        mgr.assign_initial_roles()
        assert agents[0].role == "proposer"
        assert agents[1].role == "synthesizer"

    def test_three_agents_one_proposer(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b"), FakeAgent(name="c")]
        mgr = _make_manager(agents, FakeProtocol(proposer_count=1))
        mgr.assign_initial_roles()
        assert agents[0].role == "proposer"
        assert agents[1].role == "critic"
        assert agents[2].role == "synthesizer"

    def test_three_agents_many_proposers_capped(self):
        # With 3 agents, max proposers is 1 (to ensure 1 critic + 1 synthesizer)
        agents = [FakeAgent(name="a"), FakeAgent(name="b"), FakeAgent(name="c")]
        mgr = _make_manager(agents, FakeProtocol(proposer_count=5))
        mgr.assign_initial_roles()
        assert agents[0].role == "proposer"
        assert agents[1].role == "critic"
        assert agents[2].role == "synthesizer"

    def test_five_agents_two_proposers(self):
        agents = [FakeAgent(name=f"a{i}") for i in range(5)]
        mgr = _make_manager(agents, FakeProtocol(proposer_count=2))
        mgr.assign_initial_roles()
        assert agents[0].role == "proposer"
        assert agents[1].role == "proposer"
        assert agents[2].role == "critic"
        assert agents[3].role == "critic"
        assert agents[4].role == "synthesizer"

    def test_respects_existing_roles(self):
        agents = [
            FakeAgent(name="a", role="custom1"),
            FakeAgent(name="b", role="custom2"),
        ]
        mgr = _make_manager(agents)
        mgr.assign_initial_roles()
        assert agents[0].role == "custom1"
        assert agents[1].role == "custom2"

    def test_assign_roles_alias(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        mgr.assign_roles()
        assert agents[0].role == "proposer"


# ---------------------------------------------------------------------------
# assign_stances
# ---------------------------------------------------------------------------


class TestAssignStances:
    def test_no_asymmetric_stances(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents, FakeProtocol(asymmetric_stances=False))
        mgr.assign_stances()
        assert agents[0].stance == ""
        assert agents[1].stance == ""

    def test_assigns_stances(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b"), FakeAgent(name="c")]
        mgr = _make_manager(agents, FakeProtocol(asymmetric_stances=True))
        mgr.assign_stances(round_num=0)
        assert agents[0].stance == "affirmative"
        assert agents[1].stance == "negative"
        assert agents[2].stance == "neutral"

    def test_rotates_stances(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b"), FakeAgent(name="c")]
        mgr = _make_manager(agents, FakeProtocol(asymmetric_stances=True, rotate_stances=True))
        mgr.assign_stances(round_num=1)
        # With rotation, agent 0 gets stance (0+1)%3 = 1 = "negative"
        assert agents[0].stance == "negative"
        assert agents[1].stance == "neutral"
        assert agents[2].stance == "affirmative"

    def test_no_rotation(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents, FakeProtocol(asymmetric_stances=True, rotate_stances=False))
        mgr.assign_stances(round_num=0)
        mgr.assign_stances(round_num=1)
        # Without rotation, stances are the same regardless of round
        assert agents[0].stance == "affirmative"
        assert agents[1].stance == "negative"


# ---------------------------------------------------------------------------
# apply_agreement_intensity
# ---------------------------------------------------------------------------


class TestApplyAgreementIntensity:
    def test_none_intensity(self):
        agents = [FakeAgent(name="a", system_prompt="base")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=None))
        mgr.apply_agreement_intensity()
        # _get_agreement_intensity_guidance returns "" but system_prompt still gets \n\n appended
        assert agents[0].system_prompt.strip() == "base"

    def test_adversarial(self):
        agents = [FakeAgent(name="a", system_prompt="base")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=2))
        mgr.apply_agreement_intensity()
        assert "ADVERSARIAL" in agents[0].system_prompt

    def test_balanced(self):
        agents = [FakeAgent(name="a", system_prompt="base")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=5))
        mgr.apply_agreement_intensity()
        assert "BALANCED" in agents[0].system_prompt

    def test_collaborative(self):
        agents = [FakeAgent(name="a", system_prompt="base")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=8))
        mgr.apply_agreement_intensity()
        assert "COLLABORATIVE" in agents[0].system_prompt

    def test_empty_system_prompt(self):
        agents = [FakeAgent(name="a", system_prompt="")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=5))
        mgr.apply_agreement_intensity()
        assert "BALANCED" in agents[0].system_prompt

    def test_boundary_low(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=3))
        assert "ADVERSARIAL" in mgr._get_agreement_intensity_guidance()

    def test_boundary_high(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=7))
        assert "COLLABORATIVE" in mgr._get_agreement_intensity_guidance()

    def test_get_agreement_intensity_guidance_public(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents, FakeProtocol(agreement_intensity=5))
        assert mgr.get_agreement_intensity_guidance() == mgr._get_agreement_intensity_guidance()


# ---------------------------------------------------------------------------
# get_stance_guidance
# ---------------------------------------------------------------------------


class TestGetStanceGuidance:
    def test_affirmative(self):
        agent = FakeAgent(name="a", stance="affirmative")
        mgr = _make_manager([agent], FakeProtocol(asymmetric_stances=True))
        guidance = mgr.get_stance_guidance(agent)
        assert "AFFIRMATIVE" in guidance
        assert "DEFEND" in guidance

    def test_negative(self):
        agent = FakeAgent(name="a", stance="negative")
        mgr = _make_manager([agent], FakeProtocol(asymmetric_stances=True))
        guidance = mgr.get_stance_guidance(agent)
        assert "NEGATIVE" in guidance
        assert "CHALLENGE" in guidance

    def test_neutral(self):
        agent = FakeAgent(name="a", stance="neutral")
        mgr = _make_manager([agent], FakeProtocol(asymmetric_stances=True))
        guidance = mgr.get_stance_guidance(agent)
        assert "NEUTRAL" in guidance
        assert "EVALUATE" in guidance

    def test_no_stance(self):
        agent = FakeAgent(name="a")
        mgr = _make_manager([agent], FakeProtocol(asymmetric_stances=True))
        assert mgr.get_stance_guidance(agent) == ""

    def test_no_asymmetric_stances(self):
        agent = FakeAgent(name="a", stance="affirmative")
        mgr = _make_manager([agent], FakeProtocol(asymmetric_stances=False))
        assert mgr.get_stance_guidance(agent) == ""

    def test_uses_prompt_builder_if_available(self):
        agent = FakeAgent(name="a", stance="affirmative")
        builder = MagicMock()
        builder.get_stance_guidance.return_value = "custom guidance"
        mgr = _make_manager([agent], prompt_builder=builder)
        assert mgr.get_stance_guidance(agent) == "custom guidance"


# ---------------------------------------------------------------------------
# rotate_roles_for_round / match_roles_for_task / update_role_assignments
# ---------------------------------------------------------------------------


class TestRoleRotation:
    def test_rotate_without_rotator_is_noop(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        mgr.rotate_roles_for_round(1)
        assert mgr.current_role_assignments == {}

    def test_rotate_with_rotator(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents, FakeProtocol(rounds=3))
        mock_rotator = MagicMock()
        mock_rotator.get_assignments.return_value = {"a": "explorer", "b": "critic"}
        mgr.role_rotator = mock_rotator
        mgr.rotate_roles_for_round(1)
        mock_rotator.get_assignments.assert_called_once_with(["a", "b"], 1, 3)

    def test_match_roles_for_task(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents)
        mock_matcher = MagicMock()
        mock_result = MagicMock()
        mock_result.assignments = {"a": "proposer", "b": "critic"}
        mock_matcher.match_roles.return_value = mock_result
        mgr.role_matcher = mock_matcher
        result = mgr.match_roles_for_task("security topic", round_num=1)
        assert result == {"a": "proposer", "b": "critic"}

    def test_match_roles_no_matcher(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        result = mgr.match_roles_for_task("topic")
        assert result == {}

    def test_update_role_assignments_with_matcher(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents)
        mock_matcher = MagicMock()
        mock_result = MagicMock()
        mock_result.assignments = {"a": "explorer"}
        mock_matcher.match_roles.return_value = mock_result
        mgr.role_matcher = mock_matcher
        mgr.update_role_assignments(round_num=2, debate_domain="security")
        mock_matcher.match_roles.assert_called_once_with(
            agent_names=["a", "b"],
            round_num=2,
            debate_domain="security",
        )

    def test_update_role_assignments_with_rotator(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents, FakeProtocol(rounds=5))
        mock_rotator = MagicMock()
        mock_rotator.get_assignments.return_value = {"a": "critic"}
        mgr.role_rotator = mock_rotator
        mgr.update_role_assignments(round_num=2)
        mock_rotator.get_assignments.assert_called_once()


# ---------------------------------------------------------------------------
# get_role_context
# ---------------------------------------------------------------------------


class TestGetRoleContext:
    def test_no_rotator(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        assert mgr.get_role_context(agents[0]) == ""

    def test_agent_not_in_assignments(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        mock_rotator = MagicMock()
        mgr.role_rotator = mock_rotator
        assert mgr.get_role_context(agents[0]) == ""

    def test_agent_in_assignments(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        mock_rotator = MagicMock()
        mock_rotator.format_role_context.return_value = "You are an explorer."
        mgr.role_rotator = mock_rotator
        mock_assignment = MagicMock()
        mgr.current_role_assignments = {"a": mock_assignment}
        assert mgr.get_role_context(agents[0]) == "You are an explorer."
        mock_rotator.format_role_context.assert_called_once_with(mock_assignment)


# ---------------------------------------------------------------------------
# get_role_summary / get_stance_summary
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_role_summary(self):
        agents = [
            FakeAgent(name="a", role="proposer"),
            FakeAgent(name="b", role="critic"),
            FakeAgent(name="c", role="synthesizer"),
        ]
        mgr = _make_manager(agents)
        summary = mgr.get_role_summary()
        assert summary["proposer"] == ["a"]
        assert summary["critic"] == ["b"]
        assert summary["synthesizer"] == ["c"]

    def test_role_summary_custom_role(self):
        agents = [FakeAgent(name="a", role="judge")]
        mgr = _make_manager(agents)
        summary = mgr.get_role_summary()
        assert "judge" in summary
        assert summary["judge"] == ["a"]

    def test_stance_summary(self):
        agents = [
            FakeAgent(name="a", stance="affirmative"),
            FakeAgent(name="b", stance="negative"),
            FakeAgent(name="c", stance="neutral"),
        ]
        mgr = _make_manager(agents)
        summary = mgr.get_stance_summary()
        assert summary["affirmative"] == ["a"]
        assert summary["negative"] == ["b"]
        assert summary["neutral"] == ["c"]

    def test_stance_summary_no_stance(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        summary = mgr.get_stance_summary()
        assert all(len(v) == 0 for v in summary.values())


# ---------------------------------------------------------------------------
# format_role_assignments_for_log / log_role_assignments
# ---------------------------------------------------------------------------


class TestLogFormatting:
    def test_format_empty(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        assert mgr.format_role_assignments_for_log() == ""

    def test_format_with_assignments(self):
        agents = [FakeAgent(name="a"), FakeAgent(name="b")]
        mgr = _make_manager(agents)
        mock_a = MagicMock()
        mock_a.role.value = "explorer"
        mock_b = MagicMock()
        mock_b.role.value = "critic"
        mgr.current_role_assignments = {"a": mock_a, "b": mock_b}
        result = mgr.format_role_assignments_for_log()
        assert "a: explorer" in result
        assert "b: critic" in result

    def test_log_role_assignments_no_assignments(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        # Should not raise
        mgr.log_role_assignments(round_num=1)

    def test_log_role_assignments_with_assignments(self):
        agents = [FakeAgent(name="a")]
        mgr = _make_manager(agents)
        mock_a = MagicMock()
        mock_a.role.value = "explorer"
        mgr.current_role_assignments = {"a": mock_a}
        # Should not raise
        mgr.log_role_assignments(round_num=2)

"""Tests for Cognitive Role System module."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.debate.roles import (
    ROLE_PROMPTS,
    CognitiveRole,
    RoleAssignment,
    RoleRotationConfig,
    RoleRotator,
    create_role_rotation,
    inject_role_into_prompt,
)


# ---------------------------------------------------------------------------
# CognitiveRole enum
# ---------------------------------------------------------------------------


class TestCognitiveRole:
    """Test CognitiveRole enum values and completeness."""

    def test_all_roles_defined(self):
        """Test all expected cognitive roles are present."""
        expected = {
            "ANALYST",
            "SKEPTIC",
            "LATERAL_THINKER",
            "SYNTHESIZER",
            "ADVOCATE",
            "DEVIL_ADVOCATE",
            "QUALITY_CHALLENGER",
        }
        actual = {role.name for role in CognitiveRole}
        assert actual == expected

    def test_role_values_are_strings(self):
        """Test all role values are lowercase strings."""
        for role in CognitiveRole:
            assert isinstance(role.value, str)
            assert role.value == role.value.lower()

    def test_role_count(self):
        """Test there are exactly 7 cognitive roles."""
        assert len(CognitiveRole) == 7


# ---------------------------------------------------------------------------
# ROLE_PROMPTS
# ---------------------------------------------------------------------------


class TestRolePrompts:
    """Test ROLE_PROMPTS dictionary."""

    def test_all_roles_have_prompts(self):
        """Test every CognitiveRole has a corresponding prompt."""
        for role in CognitiveRole:
            assert role in ROLE_PROMPTS, f"Missing prompt for {role.name}"

    def test_prompts_are_non_empty_strings(self):
        """Test all prompts are non-empty strings."""
        for role, prompt in ROLE_PROMPTS.items():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0, f"Empty prompt for {role.name}"

    def test_analyst_prompt_contains_keyword(self):
        """Test ANALYST prompt references analyst perspective."""
        prompt = ROLE_PROMPTS[CognitiveRole.ANALYST]
        assert "Analyst" in prompt
        assert "evidence" in prompt.lower()

    def test_skeptic_prompt_contains_keyword(self):
        """Test SKEPTIC prompt references challenge assumptions."""
        prompt = ROLE_PROMPTS[CognitiveRole.SKEPTIC]
        assert "Skeptic" in prompt
        assert "challenge" in prompt.lower() or "question" in prompt.lower()

    def test_lateral_thinker_prompt_contains_keyword(self):
        """Test LATERAL_THINKER prompt references unconventional thinking."""
        prompt = ROLE_PROMPTS[CognitiveRole.LATERAL_THINKER]
        assert "Lateral" in prompt
        assert "unconventional" in prompt.lower()

    def test_synthesizer_prompt_contains_keyword(self):
        """Test SYNTHESIZER prompt references integration."""
        prompt = ROLE_PROMPTS[CognitiveRole.SYNTHESIZER]
        assert "Synthesizer" in prompt
        assert "integrate" in prompt.lower() or "common ground" in prompt.lower()

    def test_advocate_prompt_contains_keyword(self):
        """Test ADVOCATE prompt references building a case."""
        prompt = ROLE_PROMPTS[CognitiveRole.ADVOCATE]
        assert "Advocate" in prompt

    def test_devil_advocate_prompt_contains_keyword(self):
        """Test DEVIL_ADVOCATE prompt references arguing against consensus."""
        prompt = ROLE_PROMPTS[CognitiveRole.DEVIL_ADVOCATE]
        assert "Devil" in prompt or "Advocate" in prompt
        assert "consensus" in prompt.lower()

    def test_quality_challenger_prompt_contains_keyword(self):
        """Test QUALITY_CHALLENGER prompt references evidence and hollow consensus."""
        prompt = ROLE_PROMPTS[CognitiveRole.QUALITY_CHALLENGER]
        assert "Quality Challenger" in prompt
        assert "hollow consensus" in prompt.lower()

    def test_no_extra_prompts(self):
        """Test there are no prompts for non-existent roles."""
        assert len(ROLE_PROMPTS) == len(CognitiveRole)


# ---------------------------------------------------------------------------
# RoleAssignment
# ---------------------------------------------------------------------------


class TestRoleAssignment:
    """Test RoleAssignment dataclass."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        ra = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ANALYST,
            round_num=0,
        )
        assert ra.agent_name == "claude"
        assert ra.role == CognitiveRole.ANALYST
        assert ra.round_num == 0

    def test_auto_populates_role_prompt(self):
        """Test role_prompt is auto-populated from ROLE_PROMPTS when empty."""
        ra = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.SKEPTIC,
            round_num=1,
        )
        assert ra.role_prompt == ROLE_PROMPTS[CognitiveRole.SKEPTIC]

    def test_custom_role_prompt_preserved(self):
        """Test custom role_prompt is not overwritten."""
        custom_prompt = "Custom prompt for testing"
        ra = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ANALYST,
            round_num=0,
            role_prompt=custom_prompt,
        )
        assert ra.role_prompt == custom_prompt

    def test_empty_string_prompt_gets_replaced(self):
        """Test empty string role_prompt triggers auto-population."""
        ra = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ADVOCATE,
            round_num=2,
            role_prompt="",
        )
        assert ra.role_prompt == ROLE_PROMPTS[CognitiveRole.ADVOCATE]

    def test_all_roles_produce_valid_assignments(self):
        """Test creating assignments for every role succeeds."""
        for role in CognitiveRole:
            ra = RoleAssignment(agent_name="test_agent", role=role, round_num=0)
            assert ra.role_prompt != ""
            assert ra.role == role


# ---------------------------------------------------------------------------
# RoleRotationConfig
# ---------------------------------------------------------------------------


class TestRoleRotationConfig:
    """Test RoleRotationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = RoleRotationConfig()
        assert cfg.enabled is True
        assert cfg.ensure_coverage is True
        assert cfg.synthesizer_final_round is True
        assert len(cfg.roles) == 4
        assert CognitiveRole.ANALYST in cfg.roles
        assert CognitiveRole.SKEPTIC in cfg.roles
        assert CognitiveRole.LATERAL_THINKER in cfg.roles
        assert CognitiveRole.SYNTHESIZER in cfg.roles

    def test_custom_roles(self):
        """Test custom role list."""
        custom_roles = [CognitiveRole.ANALYST, CognitiveRole.ADVOCATE]
        cfg = RoleRotationConfig(roles=custom_roles)
        assert len(cfg.roles) == 2
        assert CognitiveRole.ADVOCATE in cfg.roles

    def test_disabled_config(self):
        """Test disabled configuration."""
        cfg = RoleRotationConfig(enabled=False)
        assert cfg.enabled is False

    def test_no_synthesizer_final_round(self):
        """Test disabling synthesizer in final round."""
        cfg = RoleRotationConfig(synthesizer_final_round=False)
        assert cfg.synthesizer_final_round is False


# ---------------------------------------------------------------------------
# RoleRotator
# ---------------------------------------------------------------------------


class TestRoleRotator:
    """Test RoleRotator class."""

    def test_init_default_config(self):
        """Test RoleRotator with default config."""
        rotator = RoleRotator()
        assert rotator.config.enabled is True

    def test_init_custom_config(self):
        """Test RoleRotator with custom config."""
        cfg = RoleRotationConfig(enabled=False)
        rotator = RoleRotator(config=cfg)
        assert rotator.config.enabled is False

    def test_disabled_returns_empty(self):
        """Test disabled rotator returns empty assignments."""
        cfg = RoleRotationConfig(enabled=False)
        rotator = RoleRotator(config=cfg)
        assignments = rotator.get_assignments(["a1", "a2"], round_num=0, total_rounds=3)
        assert assignments == {}

    def test_single_agent_single_round(self):
        """Test assignment for a single agent."""
        rotator = RoleRotator()
        assignments = rotator.get_assignments(["claude"], round_num=0, total_rounds=1)
        # Final round with single agent => synthesizer
        assert "claude" in assignments
        assert assignments["claude"].role == CognitiveRole.SYNTHESIZER

    def test_multiple_agents_round_zero(self):
        """Test assignments for multiple agents in round 0."""
        rotator = RoleRotator()
        agents = ["claude", "gpt", "gemini"]
        assignments = rotator.get_assignments(agents, round_num=0, total_rounds=3)
        assert len(assignments) == 3
        # Each agent should have a valid role
        for agent_name in agents:
            assert agent_name in assignments
            assert isinstance(assignments[agent_name], RoleAssignment)
            assert assignments[agent_name].round_num == 0

    def test_roles_rotate_across_rounds(self):
        """Test roles change between rounds for the same agent."""
        cfg = RoleRotationConfig(synthesizer_final_round=False)
        rotator = RoleRotator(config=cfg)
        agents = ["claude", "gpt"]

        round0 = rotator.get_assignments(agents, round_num=0, total_rounds=4)
        round1 = rotator.get_assignments(agents, round_num=1, total_rounds=4)

        # At least one agent should have a different role in round 1
        roles_r0 = {a: r.role for a, r in round0.items()}
        roles_r1 = {a: r.role for a, r in round1.items()}
        # With default 4 roles and 2 agents, rotation should change something
        changed = any(roles_r0[a] != roles_r1[a] for a in agents)
        assert changed, "Roles should rotate across rounds"

    def test_final_round_synthesizer(self):
        """Test first agent gets SYNTHESIZER in final round."""
        rotator = RoleRotator()
        agents = ["claude", "gpt", "gemini"]
        assignments = rotator.get_assignments(agents, round_num=2, total_rounds=3)
        # First agent should be synthesizer in final round
        assert assignments["claude"].role == CognitiveRole.SYNTHESIZER

    def test_final_round_synthesizer_disabled(self):
        """Test no forced synthesizer when disabled."""
        cfg = RoleRotationConfig(synthesizer_final_round=False)
        rotator = RoleRotator(config=cfg)
        agents = ["claude", "gpt"]
        assignments = rotator.get_assignments(agents, round_num=2, total_rounds=3)
        # First agent may or may not be synthesizer (depends on rotation)
        # but it should not be forced
        assert len(assignments) == 2

    def test_empty_agents_list(self):
        """Test empty agents list produces empty assignments."""
        rotator = RoleRotator()
        assignments = rotator.get_assignments([], round_num=0, total_rounds=3)
        assert assignments == {}

    def test_all_agents_get_assignments(self):
        """Test every agent gets an assignment."""
        rotator = RoleRotator()
        agents = ["a1", "a2", "a3", "a4", "a5"]
        assignments = rotator.get_assignments(agents, round_num=0, total_rounds=5)
        assert len(assignments) == len(agents)

    def test_assignments_have_correct_round_num(self):
        """Test all assignments carry the correct round number."""
        rotator = RoleRotator()
        agents = ["claude", "gpt"]
        for rnd in range(4):
            assignments = rotator.get_assignments(agents, round_num=rnd, total_rounds=5)
            for assignment in assignments.values():
                assert assignment.round_num == rnd

    def test_custom_role_set(self):
        """Test rotation uses custom role set."""
        cfg = RoleRotationConfig(
            roles=[CognitiveRole.ADVOCATE, CognitiveRole.DEVIL_ADVOCATE],
            synthesizer_final_round=False,
        )
        rotator = RoleRotator(config=cfg)
        agents = ["claude"]
        assignments = rotator.get_assignments(agents, round_num=0, total_rounds=2)
        assert assignments["claude"].role in (
            CognitiveRole.ADVOCATE,
            CognitiveRole.DEVIL_ADVOCATE,
        )


class TestRoleRotatorPromptMethods:
    """Test RoleRotator prompt-related methods."""

    def test_get_role_prompt_returns_string(self):
        """Test get_role_prompt returns a string."""
        rotator = RoleRotator()
        prompt = rotator.get_role_prompt("claude", round_num=0)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_format_role_context_contains_round(self):
        """Test format_role_context includes round information."""
        rotator = RoleRotator()
        assignment = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ANALYST,
            round_num=2,
        )
        ctx = rotator.format_role_context(assignment)
        assert "Round 3" in ctx  # round_num + 1
        assert "COGNITIVE ROLE ASSIGNMENT" in ctx
        assert "END ROLE ASSIGNMENT" in ctx

    def test_format_role_context_includes_prompt(self):
        """Test format_role_context includes the role prompt."""
        rotator = RoleRotator()
        assignment = RoleAssignment(
            agent_name="gpt",
            role=CognitiveRole.SKEPTIC,
            round_num=0,
        )
        ctx = rotator.format_role_context(assignment)
        # Should include the skeptic prompt content
        assert "Skeptic" in ctx


# ---------------------------------------------------------------------------
# create_role_rotation (module-level function)
# ---------------------------------------------------------------------------


class TestCreateRoleRotation:
    """Test the create_role_rotation convenience function."""

    def _make_agents(self, names):
        """Create simple agent-like objects with .name attribute."""
        return [SimpleNamespace(name=n) for n in names]

    def test_returns_schedule_list(self):
        """Test returns a list of round assignments."""
        agents = self._make_agents(["claude", "gpt"])
        schedule = create_role_rotation(agents, total_rounds=3)
        assert isinstance(schedule, list)
        assert len(schedule) == 3

    def test_each_round_is_dict(self):
        """Test each round in the schedule is a dict of assignments."""
        agents = self._make_agents(["claude", "gpt"])
        schedule = create_role_rotation(agents, total_rounds=2)
        for round_assignments in schedule:
            assert isinstance(round_assignments, dict)

    def test_all_agents_assigned_each_round(self):
        """Test all agents get assignments in every round."""
        agents = self._make_agents(["a1", "a2", "a3"])
        schedule = create_role_rotation(agents, total_rounds=4)
        for round_assignments in schedule:
            for agent in agents:
                assert agent.name in round_assignments

    def test_custom_config_passed_through(self):
        """Test custom config is respected."""
        agents = self._make_agents(["claude"])
        cfg = RoleRotationConfig(enabled=False)
        schedule = create_role_rotation(agents, total_rounds=3, config=cfg)
        # All rounds should be empty when disabled
        for round_assignments in schedule:
            assert round_assignments == {}

    def test_final_round_has_synthesizer(self):
        """Test final round assigns synthesizer to first agent."""
        agents = self._make_agents(["claude", "gpt", "gemini"])
        schedule = create_role_rotation(agents, total_rounds=3)
        final_round = schedule[-1]
        assert final_round["claude"].role == CognitiveRole.SYNTHESIZER

    def test_single_round_debate(self):
        """Test single round debate works correctly."""
        agents = self._make_agents(["claude", "gpt"])
        schedule = create_role_rotation(agents, total_rounds=1)
        assert len(schedule) == 1
        # Single round is also the final round
        assert schedule[0]["claude"].role == CognitiveRole.SYNTHESIZER

    def test_empty_agents_list(self):
        """Test empty agents produce empty schedule."""
        schedule = create_role_rotation([], total_rounds=3)
        assert len(schedule) == 3
        for round_assignments in schedule:
            assert round_assignments == {}

    def test_zero_rounds(self):
        """Test zero rounds produces empty schedule."""
        agents = self._make_agents(["claude"])
        schedule = create_role_rotation(agents, total_rounds=0)
        assert schedule == []

    def test_roles_differ_across_rounds(self):
        """Test agents get different roles across rounds."""
        agents = self._make_agents(["claude", "gpt"])
        cfg = RoleRotationConfig(synthesizer_final_round=False)
        schedule = create_role_rotation(agents, total_rounds=4, config=cfg)
        claude_roles = [r["claude"].role for r in schedule]
        # With 4 default roles and 4 rounds, claude should cycle through roles
        assert len(set(claude_roles)) > 1, "Agent should get different roles across rounds"


# ---------------------------------------------------------------------------
# inject_role_into_prompt (module-level function)
# ---------------------------------------------------------------------------


class TestInjectRoleIntoPrompt:
    """Test the inject_role_into_prompt function."""

    def test_prepends_role_context(self):
        """Test role context is prepended to the base prompt."""
        assignment = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ANALYST,
            round_num=0,
        )
        result = inject_role_into_prompt("Please analyze this topic.", assignment)
        # Role context should come before the base prompt
        analyst_idx = result.index("COGNITIVE ROLE ASSIGNMENT")
        base_idx = result.index("Please analyze this topic.")
        assert analyst_idx < base_idx

    def test_base_prompt_preserved(self):
        """Test the original base prompt is fully preserved."""
        base = "This is the original prompt with special chars: !@#$%"
        assignment = RoleAssignment(
            agent_name="gpt",
            role=CognitiveRole.SKEPTIC,
            round_num=1,
        )
        result = inject_role_into_prompt(base, assignment)
        assert base in result

    def test_role_prompt_included(self):
        """Test the role-specific prompt text is included."""
        assignment = RoleAssignment(
            agent_name="gemini",
            role=CognitiveRole.LATERAL_THINKER,
            round_num=0,
        )
        result = inject_role_into_prompt("base", assignment)
        assert "Lateral Thinker" in result

    def test_round_number_displayed(self):
        """Test round number is correctly displayed (1-indexed)."""
        assignment = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.SYNTHESIZER,
            round_num=4,
        )
        result = inject_role_into_prompt("base", assignment)
        assert "Round 5" in result

    def test_empty_base_prompt(self):
        """Test with empty base prompt."""
        assignment = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ADVOCATE,
            round_num=0,
        )
        result = inject_role_into_prompt("", assignment)
        assert "COGNITIVE ROLE ASSIGNMENT" in result

    def test_all_roles_inject_successfully(self):
        """Test injection works for every cognitive role."""
        for role in CognitiveRole:
            assignment = RoleAssignment(
                agent_name="agent",
                role=role,
                round_num=0,
            )
            result = inject_role_into_prompt("base prompt", assignment)
            assert "base prompt" in result
            assert "COGNITIVE ROLE ASSIGNMENT" in result

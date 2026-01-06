"""
Tests for cognitive role system.

Tests CognitiveRole enum, ROLE_PROMPTS, RoleRotationConfig,
RoleRotator, and role injection utilities.
"""

import pytest
from dataclasses import fields

from aragora.debate.roles import (
    CognitiveRole,
    ROLE_PROMPTS,
    RoleAssignment,
    RoleRotationConfig,
    RoleRotator,
    create_role_rotation,
    inject_role_into_prompt,
)


# =============================================================================
# CognitiveRole Tests
# =============================================================================

class TestCognitiveRole:
    """Tests for CognitiveRole enum."""

    def test_all_roles_defined(self):
        """All expected cognitive roles should be defined."""
        expected_roles = [
            "analyst",
            "skeptic",
            "lateral_thinker",
            "synthesizer",
            "advocate",
            "devil_advocate",
        ]

        role_values = [r.value for r in CognitiveRole]
        for role in expected_roles:
            assert role in role_values, f"Missing role: {role}"

    def test_role_enum_values(self):
        """Role enum values should match expected strings."""
        assert CognitiveRole.ANALYST.value == "analyst"
        assert CognitiveRole.SKEPTIC.value == "skeptic"
        assert CognitiveRole.LATERAL_THINKER.value == "lateral_thinker"
        assert CognitiveRole.SYNTHESIZER.value == "synthesizer"
        assert CognitiveRole.ADVOCATE.value == "advocate"
        assert CognitiveRole.DEVIL_ADVOCATE.value == "devil_advocate"


# =============================================================================
# ROLE_PROMPTS Tests
# =============================================================================

class TestRolePrompts:
    """Tests for role prompt definitions."""

    def test_all_roles_have_prompts(self):
        """Every CognitiveRole should have a corresponding prompt."""
        for role in CognitiveRole:
            assert role in ROLE_PROMPTS, f"Missing prompt for {role}"

    def test_prompts_not_empty(self):
        """All role prompts should have content."""
        for role, prompt in ROLE_PROMPTS.items():
            assert len(prompt.strip()) > 100, f"Prompt too short for {role}"

    def test_analyst_prompt_content(self):
        """Analyst prompt should mention investigation and evidence."""
        prompt = ROLE_PROMPTS[CognitiveRole.ANALYST]
        assert "ANALYST" in prompt
        assert "evidence" in prompt.lower()
        assert "investigate" in prompt.lower() or "investigation" in prompt.lower()

    def test_skeptic_prompt_content(self):
        """Skeptic prompt should mention challenge and assumptions."""
        prompt = ROLE_PROMPTS[CognitiveRole.SKEPTIC]
        assert "SKEPTIC" in prompt
        assert "challenge" in prompt.lower()
        assert "assumption" in prompt.lower()

    def test_synthesizer_prompt_content(self):
        """Synthesizer prompt should mention integration and common ground."""
        prompt = ROLE_PROMPTS[CognitiveRole.SYNTHESIZER]
        assert "SYNTHESIZER" in prompt
        assert "integrate" in prompt.lower() or "common ground" in prompt.lower()

    def test_devil_advocate_prompt_content(self):
        """Devil's advocate prompt should mention challenging consensus."""
        prompt = ROLE_PROMPTS[CognitiveRole.DEVIL_ADVOCATE]
        assert "DEVIL" in prompt
        assert "consensus" in prompt.lower()


# =============================================================================
# RoleAssignment Tests
# =============================================================================

class TestRoleAssignment:
    """Tests for RoleAssignment dataclass."""

    def test_basic_assignment(self):
        """Should create assignment with required fields."""
        assignment = RoleAssignment(
            agent_name="claude",
            role=CognitiveRole.ANALYST,
            round_num=0,
        )

        assert assignment.agent_name == "claude"
        assert assignment.role == CognitiveRole.ANALYST
        assert assignment.round_num == 0

    def test_auto_populate_role_prompt(self):
        """Role prompt should auto-populate from ROLE_PROMPTS."""
        assignment = RoleAssignment(
            agent_name="test",
            role=CognitiveRole.SKEPTIC,
            round_num=1,
        )

        assert len(assignment.role_prompt) > 0
        assert "SKEPTIC" in assignment.role_prompt

    def test_custom_role_prompt(self):
        """Should accept custom role prompt."""
        custom_prompt = "Custom analyst behavior"
        assignment = RoleAssignment(
            agent_name="test",
            role=CognitiveRole.ANALYST,
            round_num=0,
            role_prompt=custom_prompt,
        )

        assert assignment.role_prompt == custom_prompt


# =============================================================================
# RoleRotationConfig Tests
# =============================================================================

class TestRoleRotationConfig:
    """Tests for RoleRotationConfig dataclass."""

    def test_default_values(self):
        """Default configuration should have sensible values."""
        config = RoleRotationConfig()

        assert config.enabled is True
        assert config.ensure_coverage is True
        assert config.synthesizer_final_round is True

    def test_default_roles(self):
        """Default roles should include core four."""
        config = RoleRotationConfig()

        assert CognitiveRole.ANALYST in config.roles
        assert CognitiveRole.SKEPTIC in config.roles
        assert CognitiveRole.LATERAL_THINKER in config.roles
        assert CognitiveRole.SYNTHESIZER in config.roles

    def test_custom_roles(self):
        """Should accept custom role list."""
        config = RoleRotationConfig(
            roles=[CognitiveRole.ANALYST, CognitiveRole.SKEPTIC]
        )

        assert len(config.roles) == 2
        assert CognitiveRole.ANALYST in config.roles
        assert CognitiveRole.SKEPTIC in config.roles


# =============================================================================
# RoleRotator Tests
# =============================================================================

class TestRoleRotator:
    """Tests for RoleRotator class."""

    @pytest.fixture
    def rotator(self):
        """Default rotator instance."""
        return RoleRotator()

    @pytest.fixture
    def agent_names(self):
        """Sample agent names."""
        return ["claude", "gpt4", "gemini"]

    def test_disabled_rotation_returns_empty(self, agent_names):
        """Disabled rotation should return empty dict."""
        config = RoleRotationConfig(enabled=False)
        rotator = RoleRotator(config)

        assignments = rotator.get_assignments(agent_names, 0, 5)
        assert assignments == {}

    def test_assigns_roles_to_all_agents(self, rotator, agent_names):
        """Should assign roles to all agents."""
        assignments = rotator.get_assignments(agent_names, 0, 5)

        for agent in agent_names:
            assert agent in assignments

    def test_rotation_changes_roles(self, rotator, agent_names):
        """Roles should change between rounds."""
        round_0 = rotator.get_assignments(agent_names, 0, 5)
        round_1 = rotator.get_assignments(agent_names, 1, 5)

        # At least one agent should have a different role
        different_roles = False
        for agent in agent_names:
            if round_0[agent].role != round_1[agent].role:
                different_roles = True
                break

        assert different_roles, "Roles should change between rounds"

    def test_synthesizer_in_final_round(self, agent_names):
        """Final round should assign synthesizer when configured."""
        config = RoleRotationConfig(synthesizer_final_round=True)
        rotator = RoleRotator(config)

        # Final round (round 4 of 5)
        assignments = rotator.get_assignments(agent_names, 4, 5)

        # At least one agent should be synthesizer
        synthesizer_found = any(
            a.role == CognitiveRole.SYNTHESIZER
            for a in assignments.values()
        )
        assert synthesizer_found

    def test_format_role_context(self, rotator):
        """Should format role context correctly."""
        assignment = RoleAssignment(
            agent_name="test",
            role=CognitiveRole.ANALYST,
            round_num=2,
        )

        context = rotator.format_role_context(assignment)

        assert "Round 3" in context  # 0-indexed round_num displayed as 1-indexed
        assert "ANALYST" in context


# =============================================================================
# create_role_rotation Tests
# =============================================================================

class TestCreateRoleRotation:
    """Tests for create_role_rotation utility function."""

    class MockAgent:
        """Mock agent with name attribute."""
        def __init__(self, name):
            self.name = name

    @pytest.fixture
    def agents(self):
        """Sample agents."""
        return [
            self.MockAgent("claude"),
            self.MockAgent("gpt4"),
            self.MockAgent("gemini"),
        ]

    def test_creates_schedule_for_all_rounds(self, agents):
        """Should create assignments for all rounds."""
        schedule = create_role_rotation(agents, total_rounds=5)

        assert len(schedule) == 5

    def test_each_round_has_all_agents(self, agents):
        """Each round should have assignments for all agents."""
        schedule = create_role_rotation(agents, total_rounds=3)

        for round_assignments in schedule:
            for agent in agents:
                assert agent.name in round_assignments

    def test_respects_config(self, agents):
        """Should respect custom configuration."""
        config = RoleRotationConfig(enabled=False)
        schedule = create_role_rotation(agents, total_rounds=3, config=config)

        # All rounds should have empty assignments
        for round_assignments in schedule:
            assert round_assignments == {}


# =============================================================================
# inject_role_into_prompt Tests
# =============================================================================

class TestInjectRoleIntoPrompt:
    """Tests for inject_role_into_prompt utility function."""

    def test_injects_role_context(self):
        """Should inject role context into prompt."""
        base_prompt = "You are an AI assistant."
        assignment = RoleAssignment(
            agent_name="test",
            role=CognitiveRole.SKEPTIC,
            round_num=0,
        )

        result = inject_role_into_prompt(base_prompt, assignment)

        assert "SKEPTIC" in result
        assert base_prompt in result

    def test_role_context_prepended(self):
        """Role context should come before base prompt."""
        base_prompt = "Base prompt content"
        assignment = RoleAssignment(
            agent_name="test",
            role=CognitiveRole.ANALYST,
            round_num=0,
        )

        result = inject_role_into_prompt(base_prompt, assignment)

        # Role should appear before base prompt
        role_pos = result.find("ANALYST")
        base_pos = result.find(base_prompt)
        assert role_pos < base_pos

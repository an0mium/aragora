"""Tests for essay-specific agent roles and round phases — written FIRST (TDD)."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. test_essay_phases_are_ordered — phases[i].number == i
# ---------------------------------------------------------------------------


def test_essay_phases_are_ordered():
    """Every phase's .number attribute must equal its list index."""
    from aragora.essay.roles import ESSAY_ROUND_PHASES

    for i, phase in enumerate(ESSAY_ROUND_PHASES):
        assert phase.number == i, f"Phase at index {i} has number={phase.number}, expected {i}"


# ---------------------------------------------------------------------------
# 2. test_essay_phases_have_required_fields — all text fields non-empty
# ---------------------------------------------------------------------------


def test_essay_phases_have_required_fields():
    """Every phase must have non-empty name, description, focus, and cognitive_mode."""
    from aragora.essay.roles import ESSAY_ROUND_PHASES

    for phase in ESSAY_ROUND_PHASES:
        assert phase.name, f"Phase {phase.number} has empty name"
        assert phase.description, f"Phase {phase.number} has empty description"
        assert phase.focus, f"Phase {phase.number} has empty focus"
        assert phase.cognitive_mode, f"Phase {phase.number} has empty cognitive_mode"


# ---------------------------------------------------------------------------
# 3. test_essay_agent_roles_defined — required roles are present
# ---------------------------------------------------------------------------


def test_essay_agent_roles_defined():
    """drafter, critic, synthesizer, and judge must all be present."""
    from aragora.essay.roles import ESSAY_AGENT_ROLES

    required = {"drafter", "critic", "synthesizer", "judge"}
    missing = required - set(ESSAY_AGENT_ROLES.keys())
    assert not missing, f"Missing roles: {missing}"

    for role_name, config in ESSAY_AGENT_ROLES.items():
        assert "role" in config, f"Role '{role_name}' missing 'role' key"
        assert "description" in config, f"Role '{role_name}' missing 'description' key"
        assert config["role"], f"Role '{role_name}' has empty 'role' value"
        assert config["description"], f"Role '{role_name}' has empty 'description' value"

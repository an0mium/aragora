"""Tests for the execution-pattern selector (pure, deterministic)."""

from __future__ import annotations

import pytest

from aragora.modes.selector import (
    EXECUTOR_MODULES,
    ModeDecisionContext,
    OperationalModeSelector,
    OrchestrationPattern,
    PatternDecision,
    estimate_goal_abstractness,
)

SELECTOR = OperationalModeSelector()


def _pick(**kwargs: object) -> PatternDecision:
    return SELECTOR.select_pattern(ModeDecisionContext(**kwargs))  # type: ignore[arg-type]


def test_high_risk_forces_agent_teams() -> None:
    d = _pick(risk_tier=3, complexity_score=1.0)
    assert d.pattern is OrchestrationPattern.AGENT_TEAMS
    assert d.confidence >= 0.9


def test_consensus_required_forces_agent_teams() -> None:
    d = _pick(consensus_required=True, complexity_score=1.0)
    assert d.pattern is OrchestrationPattern.AGENT_TEAMS


def test_design_routes_to_agent_teams() -> None:
    assert _pick(is_design=True, complexity_score=2.0).pattern is OrchestrationPattern.AGENT_TEAMS


def test_error_diagnosis_routes_to_agent_teams() -> None:
    assert _pick(involves_error=True).pattern is OrchestrationPattern.AGENT_TEAMS


def test_very_high_complexity_routes_to_agent_teams() -> None:
    assert _pick(complexity_score=9.0).pattern is OrchestrationPattern.AGENT_TEAMS


def test_abstract_noncode_goal_routes_to_goal_anchored() -> None:
    d = _pick(goal_abstractness=0.7, is_code_change=False, complexity_score=2.0)
    assert d.pattern is OrchestrationPattern.GOAL_ANCHORED


def test_abstract_but_code_change_does_not_take_goal_anchored() -> None:
    # Abstract phrasing on a concrete code change should not become goal_anchored.
    d = _pick(goal_abstractness=0.7, is_code_change=True, complexity_score=1.0)
    assert d.pattern is not OrchestrationPattern.GOAL_ANCHORED


def test_structured_medium_complexity_routes_to_dynamic_workflow() -> None:
    assert _pick(complexity_score=5.0).pattern is OrchestrationPattern.DYNAMIC_WORKFLOW


def test_default_is_dynamic_workflow_not_agent_teams() -> None:
    # Critique fix: an unclassified low-complexity task must NOT default to the
    # most expensive executor.
    d = _pick(complexity_score=1.0)
    assert d.pattern is OrchestrationPattern.DYNAMIC_WORKFLOW


def test_prior_pattern_breaks_tie_in_default_branch() -> None:
    d = _pick(complexity_score=1.0, prior_pattern=OrchestrationPattern.GOAL_ANCHORED)
    assert d.pattern is OrchestrationPattern.GOAL_ANCHORED
    assert "prior" in d.rationale.lower()


def test_risk_gate_precedes_prior_pattern() -> None:
    # Safety override wins even if a prior pattern hint is present.
    d = _pick(risk_tier=4, prior_pattern=OrchestrationPattern.DYNAMIC_WORKFLOW)
    assert d.pattern is OrchestrationPattern.AGENT_TEAMS


def test_decision_is_deterministic() -> None:
    ctx = ModeDecisionContext(complexity_score=5.0, domain="software")
    assert SELECTOR.select_pattern(ctx).pattern is SELECTOR.select_pattern(ctx).pattern


def test_executor_module_mapping_complete() -> None:
    for pattern in OrchestrationPattern:
        assert pattern in EXECUTOR_MODULES
    d = _pick(complexity_score=5.0)
    assert d.executor_module == EXECUTOR_MODULES[OrchestrationPattern.DYNAMIC_WORKFLOW]


def test_to_dict_shape() -> None:
    d = _pick(complexity_score=5.0).to_dict()
    assert set(d) == {"pattern", "confidence", "rationale", "executor_module"}
    assert d["pattern"] == "dynamic_workflow"


@pytest.mark.parametrize(
    "text,expected_high",
    [
        ("maximize SME utility", True),
        ("optimize the cache", True),
        ("add a login button", False),
        ("", False),
    ],
)
def test_estimate_goal_abstractness(text: str, expected_high: bool) -> None:
    score = estimate_goal_abstractness(text)
    assert (score >= 0.6) is expected_high


@pytest.mark.parametrize("bad", [-1, 5])
def test_invalid_risk_tier_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="risk_tier"):
        ModeDecisionContext(risk_tier=bad)


def test_invalid_complexity_rejected() -> None:
    with pytest.raises(ValueError, match="complexity_score"):
        ModeDecisionContext(complexity_score=11.0)

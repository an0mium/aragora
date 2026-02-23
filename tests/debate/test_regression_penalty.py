"""Tests for regression penalty scoring in TeamSelector."""

from unittest.mock import MagicMock, patch

from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def test_regression_penalty_returns_zero_with_no_data():
    """No regression history should yield 0.0 penalty."""
    selector = TeamSelector(config=TeamSelectionConfig())
    agent = _make_agent("claude")

    with patch(
        "aragora.nomic.outcome_tracker.NomicOutcomeTracker.get_regression_history",
        return_value=[],
    ):
        penalty = selector._compute_regression_penalty(agent)
    assert penalty == 0.0


def test_regression_penalty_negative_with_regressions():
    """Agents mentioned in regression recommendations get penalized."""
    selector = TeamSelector(config=TeamSelectionConfig())
    agent = _make_agent("claude")

    regressions = [
        {
            "cycle_id": "cycle-1",
            "regressed_metrics": ["consensus_rate"],
            "recommendation": "Claude caused consensus regression",
        },
        {
            "cycle_id": "cycle-2",
            "regressed_metrics": ["avg_rounds"],
            "recommendation": "Claude and gemini increased round count",
        },
    ]

    with patch(
        "aragora.nomic.outcome_tracker.NomicOutcomeTracker.get_regression_history",
        return_value=regressions,
    ):
        penalty = selector._compute_regression_penalty(agent)
    assert penalty < 0.0
    assert penalty >= -0.5


def test_regression_penalty_zero_for_uninvolved_agent():
    """Agents not mentioned in regressions should get 0 penalty."""
    selector = TeamSelector(config=TeamSelectionConfig())
    agent = _make_agent("deepseek")

    regressions = [
        {
            "cycle_id": "cycle-1",
            "regressed_metrics": ["consensus_rate"],
            "recommendation": "Claude caused consensus regression",
        },
    ]

    with patch(
        "aragora.nomic.outcome_tracker.NomicOutcomeTracker.get_regression_history",
        return_value=regressions,
    ):
        penalty = selector._compute_regression_penalty(agent)
    assert penalty == 0.0


def test_regression_penalty_disabled():
    """When disabled in config, penalty should not affect score."""
    config = TeamSelectionConfig(enable_regression_penalty=False)
    selector = TeamSelector(config=config)
    agent = _make_agent("claude")

    # _compute_score should not call _compute_regression_penalty
    with patch.object(selector, "_compute_regression_penalty") as mock_penalty:
        selector._compute_score(agent, domain="test")
        mock_penalty.assert_not_called()

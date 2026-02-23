"""Tests for introspection scoring in TeamSelector."""

from unittest.mock import MagicMock, patch

from aragora.debate.team_selector import TeamSelector, TeamSelectionConfig


def _make_agent(name: str) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def test_introspection_score_returns_weighted_values():
    """Introspection score returns average of reputation and calibration."""
    selector = TeamSelector(config=TeamSelectionConfig())
    agent = _make_agent("claude")

    snapshot = MagicMock()
    snapshot.reputation_score = 0.8
    snapshot.calibration_score = 0.6

    with patch(
        "aragora.introspection.api.get_agent_introspection",
        return_value=snapshot,
    ):
        score = selector._compute_introspection_score(agent)
    assert score == (0.8 + 0.6) / 2.0


def test_introspection_score_zero_on_import_error():
    """Score should be 0.0 when introspection module is unavailable."""
    selector = TeamSelector(config=TeamSelectionConfig())
    agent = _make_agent("claude")

    with patch(
        "aragora.debate.team_selector.TeamSelector._compute_introspection_score",
        wraps=selector._compute_introspection_score,
    ):
        with patch.dict("sys.modules", {"aragora.introspection.api": None}):
            score = selector._compute_introspection_score(agent)
    assert score == 0.0


def test_introspection_scoring_disabled():
    """When disabled in config, introspection should not affect score."""
    config = TeamSelectionConfig(enable_introspection_scoring=False)
    selector = TeamSelector(config=config)
    agent = _make_agent("claude")

    with patch.object(selector, "_compute_introspection_score") as mock_score:
        selector._compute_score(agent, domain="test")
        mock_score.assert_not_called()

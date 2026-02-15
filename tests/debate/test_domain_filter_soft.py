"""Tests for soft domain filtering mode in TeamSelector.

Verifies that domain_filter_mode="soft" keeps all agents but applies a scoring
penalty to non-matching agents, and that ELO win rate can override the penalty.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.team_selector import TeamSelectionConfig, TeamSelector


def _make_agent(name: str) -> MagicMock:
    """Create a mock agent with the given name."""
    agent = MagicMock()
    agent.name = name
    agent.agent_type = name
    agent.model = name
    return agent


class TestDomainFilterHardMode:
    """Hard mode preserves existing filter behavior (default)."""

    def test_hard_mode_filters_non_matching(self):
        config = TeamSelectionConfig(domain_filter_mode="hard")
        selector = TeamSelector(config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        # code domain prefers: claude, codex, codestral, deepseek, gpt
        names = [a.name for a in result]
        assert "claude" in names
        assert "gpt" in names
        assert "llama" not in names

    def test_hard_mode_is_default(self):
        config = TeamSelectionConfig()
        assert config.domain_filter_mode == "hard"


class TestDomainFilterSoftMode:
    """Soft mode keeps all agents but penalizes non-matching."""

    def test_soft_mode_keeps_all_agents(self):
        config = TeamSelectionConfig(domain_filter_mode="soft")
        selector = TeamSelector(config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        # All agents should be returned
        assert len(result) == 3
        names = [a.name for a in result]
        assert "llama" in names

    def test_soft_mode_populates_non_preferred(self):
        config = TeamSelectionConfig(domain_filter_mode="soft")
        selector = TeamSelector(config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        selector._filter_by_domain_capability(agents, "code")

        # llama is not in the code domain capability map
        assert "llama" in selector._domain_non_preferred
        assert "claude" not in selector._domain_non_preferred
        assert "gpt" not in selector._domain_non_preferred

    def test_soft_mode_scoring_penalty(self):
        """Non-preferred agents get a scoring penalty in soft mode."""
        config = TeamSelectionConfig(
            domain_filter_mode="soft",
            domain_soft_penalty=0.3,
            # Disable domain capability scoring to isolate the penalty effect
            enable_domain_filtering=True,
        )
        selector = TeamSelector(config=config)

        # Use two agents that both match the code domain to isolate
        # the soft penalty. "gemini" matches code; "llama" does not.
        gemini = _make_agent("gemini")
        llama = _make_agent("llama")

        # Directly set non-preferred to isolate the penalty
        selector._domain_non_preferred = {"llama"}

        score_gemini = selector._compute_score(gemini, domain="code")
        score_llama = selector._compute_score(llama, domain="code")

        # llama should score lower due to the penalty
        assert score_llama < score_gemini

        # Now test with two identical-domain agents, one penalized
        agent_a = _make_agent("agent_a")
        agent_b = _make_agent("agent_b")
        selector._domain_non_preferred = {"agent_b"}

        score_a = selector._compute_score(agent_a, domain="general")
        score_b = selector._compute_score(agent_b, domain="general")

        # Exactly the penalty difference (no other domain factors for "general")
        assert abs((score_a - score_b) - 0.3) < 0.01

    def test_elo_win_rate_can_override_penalty(self):
        """High ELO win rate should be able to overcome the soft penalty."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1000

        # Create a top agent rating for llama with high win rate
        llama_rating = MagicMock()
        llama_rating.agent_name = "llama"
        llama_rating.win_rate = 0.9  # Very high win rate
        mock_elo.get_top_agents_for_domain.return_value = [llama_rating]

        config = TeamSelectionConfig(
            domain_filter_mode="soft",
            domain_soft_penalty=0.3,
            enable_elo_win_rate=True,
            elo_win_rate_weight=0.5,
        )
        selector = TeamSelector(elo_system=mock_elo, config=config)

        llama = _make_agent("llama")
        selector._domain_non_preferred = {"llama"}

        # With penalty=0.3 and ELO win rate boost:
        # win_rate 0.9 -> (0.9-0.5)*2.0 = 0.8, * weight 0.5 = 0.4
        # Net effect: +0.4 - 0.3 = +0.1 (positive, so ELO overrides penalty)
        score = selector._compute_score(llama, domain="code")
        base_score = config.base_score
        assert score > base_score, "High ELO win rate should overcome the soft penalty"


class TestDomainFilterDisabledMode:
    """Disabled mode skips domain filtering entirely."""

    def test_disabled_mode_returns_all_agents(self):
        config = TeamSelectionConfig(domain_filter_mode="disabled")
        selector = TeamSelector(config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        assert len(result) == 3
        assert len(selector._domain_non_preferred) == 0


class TestAutoSoftOnFeedbackData:
    """Auto-switch to soft mode when feedback loop has domain data."""

    def test_auto_soft_when_feedback_data_exists(self):
        mock_feedback = MagicMock()
        mock_feedback.get_domain_weights.return_value = {"claude": 0.1, "llama": -0.05}

        config = TeamSelectionConfig(domain_filter_mode="hard")
        selector = TeamSelector(feedback_loop=mock_feedback, config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        # Should auto-switch to soft mode because feedback data exists
        assert len(result) == 3
        assert "llama" in selector._domain_non_preferred

    def test_stays_hard_when_no_feedback_data(self):
        mock_feedback = MagicMock()
        mock_feedback.get_domain_weights.return_value = {}

        config = TeamSelectionConfig(domain_filter_mode="hard")
        selector = TeamSelector(feedback_loop=mock_feedback, config=config)
        agents = [_make_agent("claude"), _make_agent("gpt"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        # Should stay in hard mode since no feedback data
        names = [a.name for a in result]
        assert "llama" not in names

    def test_disabled_not_overridden_by_feedback(self):
        mock_feedback = MagicMock()
        mock_feedback.get_domain_weights.return_value = {"claude": 0.1}

        config = TeamSelectionConfig(domain_filter_mode="disabled")
        selector = TeamSelector(feedback_loop=mock_feedback, config=config)
        agents = [_make_agent("claude"), _make_agent("llama")]

        result = selector._filter_by_domain_capability(agents, "code")

        # disabled mode should NOT be overridden to soft
        assert len(result) == 2
        assert len(selector._domain_non_preferred) == 0

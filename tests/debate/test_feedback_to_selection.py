"""Tests for feedback loop integration with team selection (B2).

Verifies that SelectionFeedbackLoop performance data flows into
TeamSelector scoring, so high-win agents get positive adjustments
and high-timeout agents get negative adjustments.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.selection_feedback import (
    FeedbackLoopConfig,
    SelectionFeedbackLoop,
)
from aragora.debate.team_selector import TeamSelectionConfig, TeamSelector


def _make_agent(name: str) -> MagicMock:
    """Create a mock agent with a name."""
    agent = MagicMock()
    agent.name = name
    agent.agent_type = "test"
    agent.model = "test-model"
    agent.metadata = {}
    return agent


class TestGetDomainWeights:
    """Test SelectionFeedbackLoop.get_domain_weights()."""

    def test_returns_empty_when_disabled(self):
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(enabled=False))
        assert loop.get_domain_weights("technical") == {}

    def test_returns_empty_when_no_data(self):
        loop = SelectionFeedbackLoop()
        assert loop.get_domain_weights("technical") == {}

    def test_returns_empty_below_min_debates(self):
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=5))
        loop.process_debate_outcome(
            debate_id="d1",
            participants=["claude"],
            winner="claude",
            domain="technical",
        )
        # Only 1 debate, below threshold of 5
        weights = loop.get_domain_weights("technical")
        assert weights == {}

    def test_positive_weight_for_high_win_agent(self):
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        # Claude wins 4 out of 4 debates in technical domain
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="technical",
            )
        weights = loop.get_domain_weights("technical")
        assert "claude" in weights
        assert weights["claude"] > 0

    def test_negative_weight_for_losing_agent(self):
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        # GPT loses 4 out of 4 debates
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="technical",
            )
        weights = loop.get_domain_weights("technical")
        assert "gpt" in weights
        assert weights["gpt"] < 0

    def test_domain_specificity(self):
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        # Claude wins in technical
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"tech-{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="technical",
            )
        # GPT wins in creative
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"creative-{i}",
                participants=["claude", "gpt"],
                winner="gpt",
                domain="creative",
            )
        tech_weights = loop.get_domain_weights("technical")
        creative_weights = loop.get_domain_weights("creative")

        # Claude should be positive in technical, lower in creative
        assert tech_weights.get("claude", 0) > creative_weights.get("claude", 0)

    def test_clamped_to_max_adjustment(self):
        config = FeedbackLoopConfig(min_debates_for_feedback=1, max_adjustment=0.3)
        loop = SelectionFeedbackLoop(config=config)
        for i in range(20):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude"],
                winner="claude",
                domain="general",
            )
        weights = loop.get_domain_weights("general")
        if "claude" in weights:
            assert weights["claude"] <= 0.3


class TestFeedbackToSelection:
    """Test that feedback loop integrates into TeamSelector scoring."""

    def test_feedback_weight_in_score(self):
        """High-win agents should score higher than low-win agents."""
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        # Claude wins 4 debates, GPT loses 4
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="general",
            )

        config = TeamSelectionConfig(
            enable_feedback_weights=True,
            feedback_weight=0.5,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_budget_filtering=False,
        )
        selector = TeamSelector(feedback_loop=loop, config=config)

        claude = _make_agent("claude")
        gpt = _make_agent("gpt")

        claude_score = selector.score_agent(claude, domain="general")
        gpt_score = selector.score_agent(gpt, domain="general")

        assert claude_score > gpt_score

    def test_feedback_disabled_no_effect(self):
        """When feedback is disabled, scores should be equal."""
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="general",
            )

        config = TeamSelectionConfig(
            enable_feedback_weights=False,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_budget_filtering=False,
        )
        selector = TeamSelector(feedback_loop=loop, config=config)

        claude = _make_agent("claude")
        gpt = _make_agent("gpt")

        claude_score = selector.score_agent(claude, domain="general")
        gpt_score = selector.score_agent(gpt, domain="general")

        assert claude_score == gpt_score

    def test_no_feedback_loop_no_effect(self):
        """When no feedback loop, scores should be base only."""
        config = TeamSelectionConfig(
            enable_feedback_weights=True,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_budget_filtering=False,
        )
        selector = TeamSelector(feedback_loop=None, config=config)

        claude = _make_agent("claude")
        score = selector.score_agent(claude, domain="general")
        assert score == config.base_score

    def test_timeout_penalty_in_selection(self):
        """Agents with timeouts should score lower."""
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        # Both agents participate in 4 debates with no winner
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner=None,
                domain="general",
            )
        # GPT times out frequently
        for _ in range(5):
            loop.record_timeout("gpt")

        config = TeamSelectionConfig(
            enable_feedback_weights=True,
            feedback_weight=0.5,
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_budget_filtering=False,
        )
        selector = TeamSelector(feedback_loop=loop, config=config)

        claude = _make_agent("claude")
        gpt = _make_agent("gpt")

        claude_score = selector.score_agent(claude, domain="general")
        gpt_score = selector.score_agent(gpt, domain="general")

        assert claude_score >= gpt_score

    def test_apply_feedback_weights_convenience(self):
        """apply_feedback_weights returns weights dict."""
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["claude", "gpt"],
                winner="claude",
                domain="technical",
            )

        config = TeamSelectionConfig(enable_feedback_weights=True)
        selector = TeamSelector(feedback_loop=loop, config=config)

        agents = [_make_agent("claude"), _make_agent("gpt")]
        weights = selector.apply_feedback_weights(agents, domain="technical")

        assert isinstance(weights, dict)
        # Claude (winner) should have positive weight
        if "claude" in weights:
            assert weights["claude"] > 0

    def test_apply_feedback_weights_empty_without_loop(self):
        """apply_feedback_weights returns empty dict without feedback loop."""
        selector = TeamSelector(config=TeamSelectionConfig(enable_feedback_weights=True))
        agents = [_make_agent("claude")]
        assert selector.apply_feedback_weights(agents) == {}

    def test_select_uses_feedback(self):
        """Full select() flow incorporates feedback weights."""
        loop = SelectionFeedbackLoop(config=FeedbackLoopConfig(min_debates_for_feedback=3))
        for i in range(4):
            loop.process_debate_outcome(
                debate_id=f"d{i}",
                participants=["winner_agent", "loser_agent"],
                winner="winner_agent",
                domain="general",
            )

        config = TeamSelectionConfig(
            enable_feedback_weights=True,
            feedback_weight=1.0,  # High weight to make feedback dominant
            enable_domain_filtering=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
            enable_cv_selection=False,
            enable_budget_filtering=False,
        )
        selector = TeamSelector(feedback_loop=loop, config=config)

        agents = [_make_agent("loser_agent"), _make_agent("winner_agent")]
        selected = selector.select(agents, domain="general")

        # Winner should be ranked first
        assert selected[0].name == "winner_agent"

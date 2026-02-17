"""
Tests for aragora.introspection.active module.

Tests cover:
- RoundMetrics dataclass construction and serialization
- IntrospectionGoals dataclass construction and serialization
- AgentPerformanceSummary computed properties
- ActiveIntrospectionTracker metric tracking and round updates
- MetaReasoningEngine prompt generation across all guidance categories
- Round-by-round update flow through IntrospectionCache
- Edge cases (no data, zero rounds, missing agents)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.introspection.active import (
    ActiveIntrospectionTracker,
    AgentPerformanceSummary,
    IntrospectionGoals,
    MetaReasoningEngine,
    RoundMetrics,
)


# ---------------------------------------------------------------------------
# RoundMetrics tests
# ---------------------------------------------------------------------------


class TestRoundMetrics:
    """Tests for the RoundMetrics dataclass."""

    def test_defaults(self):
        """RoundMetrics should have sensible defaults."""
        m = RoundMetrics()
        assert m.round_number == 0
        assert m.proposals_made == 0
        assert m.proposals_accepted == 0
        assert m.critiques_given == 0
        assert m.critiques_led_to_changes == 0
        assert m.votes_received == 0
        assert m.votes_cast == 0
        assert m.agreements == {}
        assert m.argument_influence == 0.0

    def test_custom_values(self):
        """RoundMetrics should accept custom values."""
        m = RoundMetrics(
            round_number=2,
            proposals_made=3,
            proposals_accepted=2,
            critiques_given=5,
            critiques_led_to_changes=3,
            votes_received=4,
            votes_cast=2,
            agreements={"gpt": True, "gemini": False},
            argument_influence=0.75,
        )
        assert m.round_number == 2
        assert m.proposals_made == 3
        assert m.proposals_accepted == 2
        assert m.agreements == {"gpt": True, "gemini": False}
        assert m.argument_influence == 0.75

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        m = RoundMetrics(
            round_number=1,
            proposals_made=2,
            agreements={"gpt": True},
            argument_influence=0.5,
        )
        d = m.to_dict()
        assert d["round_number"] == 1
        assert d["proposals_made"] == 2
        assert d["agreements"] == {"gpt": True}
        assert d["argument_influence"] == 0.5


# ---------------------------------------------------------------------------
# IntrospectionGoals tests
# ---------------------------------------------------------------------------


class TestIntrospectionGoals:
    """Tests for the IntrospectionGoals dataclass."""

    def test_defaults(self):
        """Goals should have sensible defaults."""
        g = IntrospectionGoals(agent_name="claude")
        assert g.agent_name == "claude"
        assert g.target_acceptance_rate == 0.6
        assert g.target_critique_quality == 0.5
        assert g.focus_expertise == []
        assert g.collaboration_targets == []
        assert g.strategic_notes == ""

    def test_custom_values(self):
        """Goals should accept custom values."""
        g = IntrospectionGoals(
            agent_name="claude",
            target_acceptance_rate=0.8,
            focus_expertise=["security", "testing"],
            strategic_notes="Focus on edge cases",
        )
        assert g.target_acceptance_rate == 0.8
        assert g.focus_expertise == ["security", "testing"]
        assert g.strategic_notes == "Focus on edge cases"

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        g = IntrospectionGoals(
            agent_name="claude",
            focus_expertise=["api"],
        )
        d = g.to_dict()
        assert d["agent_name"] == "claude"
        assert d["target_acceptance_rate"] == 0.6
        assert d["focus_expertise"] == ["api"]


# ---------------------------------------------------------------------------
# AgentPerformanceSummary tests
# ---------------------------------------------------------------------------


class TestAgentPerformanceSummary:
    """Tests for the AgentPerformanceSummary dataclass."""

    def test_proposal_acceptance_rate_zero(self):
        """Rate should be 0.0 when no proposals made."""
        s = AgentPerformanceSummary(agent_name="claude")
        assert s.proposal_acceptance_rate == 0.0

    def test_proposal_acceptance_rate(self):
        """Rate should reflect acceptance ratio."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            total_proposals=10,
            total_accepted=7,
        )
        assert s.proposal_acceptance_rate == pytest.approx(0.7)

    def test_critique_effectiveness_zero(self):
        """Effectiveness should be 0.0 when no critiques given."""
        s = AgentPerformanceSummary(agent_name="claude")
        assert s.critique_effectiveness == 0.0

    def test_critique_effectiveness(self):
        """Effectiveness should reflect ratio."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            total_critiques=10,
            total_critiques_effective=6,
        )
        assert s.critique_effectiveness == pytest.approx(0.6)

    def test_average_influence_zero(self):
        """Average influence should be 0.0 when no rounds completed."""
        s = AgentPerformanceSummary(agent_name="claude")
        assert s.average_influence == 0.0

    def test_average_influence(self):
        """Average influence should be total / rounds."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            rounds_completed=4,
            total_argument_influence=2.0,
        )
        assert s.average_influence == pytest.approx(0.5)

    def test_get_agreement_rate_none(self):
        """Agreement rate should return None for unknown agents."""
        s = AgentPerformanceSummary(agent_name="claude")
        assert s.get_agreement_rate("gpt") is None

    def test_get_agreement_rate(self):
        """Agreement rate should reflect pattern."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            agreement_patterns={"gpt": [True, True, False]},
        )
        assert s.get_agreement_rate("gpt") == pytest.approx(2 / 3)

    def test_get_top_disagreers(self):
        """Should return agents sorted by disagreement rate."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            agreement_patterns={
                "gpt": [False, False, True],  # 67% disagree
                "gemini": [True, True, True],  # 0% disagree
                "mistral": [False, False, False],  # 100% disagree
            },
        )
        disagreers = s.get_top_disagreers(limit=2)
        assert len(disagreers) == 2
        assert disagreers[0][0] == "mistral"
        assert disagreers[0][1] == pytest.approx(1.0)
        assert disagreers[1][0] == "gpt"

    def test_get_top_allies(self):
        """Should return agents sorted by agreement rate."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            agreement_patterns={
                "gpt": [True, True, False],  # 67% agree
                "gemini": [True, True, True],  # 100% agree
            },
        )
        allies = s.get_top_allies(limit=2)
        assert len(allies) == 2
        assert allies[0][0] == "gemini"
        assert allies[0][1] == pytest.approx(1.0)

    def test_to_dict(self):
        """to_dict should include all fields and computed properties."""
        s = AgentPerformanceSummary(
            agent_name="claude",
            rounds_completed=2,
            total_proposals=5,
            total_accepted=3,
            total_critiques=4,
            total_critiques_effective=2,
            total_votes_received=6,
            total_argument_influence=1.0,
        )
        d = s.to_dict()
        assert d["agent_name"] == "claude"
        assert d["proposal_acceptance_rate"] == pytest.approx(0.6)
        assert d["critique_effectiveness"] == pytest.approx(0.5)
        assert d["average_influence"] == pytest.approx(0.5)
        assert d["goals"] is None
        assert d["round_history"] == []


# ---------------------------------------------------------------------------
# ActiveIntrospectionTracker tests
# ---------------------------------------------------------------------------


class TestActiveIntrospectionTracker:
    """Tests for the ActiveIntrospectionTracker class."""

    def test_init_empty(self):
        """Fresh tracker should have no agents."""
        tracker = ActiveIntrospectionTracker()
        assert tracker.agent_count == 0
        assert tracker.get_summary("claude") is None

    def test_update_round_creates_summary(self):
        """update_round should create summary for new agent."""
        tracker = ActiveIntrospectionTracker()
        metrics = RoundMetrics(proposals_made=2, proposals_accepted=1)

        tracker.update_round("claude", round_num=1, metrics=metrics)

        assert tracker.agent_count == 1
        summary = tracker.get_summary("claude")
        assert summary is not None
        assert summary.total_proposals == 2
        assert summary.total_accepted == 1
        assert summary.rounds_completed == 1

    def test_update_round_accumulates(self):
        """Multiple rounds should accumulate metrics."""
        tracker = ActiveIntrospectionTracker()

        r1 = RoundMetrics(proposals_made=2, proposals_accepted=1, argument_influence=0.6)
        r2 = RoundMetrics(proposals_made=1, proposals_accepted=1, argument_influence=0.8)

        tracker.update_round("claude", round_num=1, metrics=r1)
        tracker.update_round("claude", round_num=2, metrics=r2)

        summary = tracker.get_summary("claude")
        assert summary.total_proposals == 3
        assert summary.total_accepted == 2
        assert summary.rounds_completed == 2
        assert summary.total_argument_influence == pytest.approx(1.4)
        assert len(summary.round_history) == 2

    def test_update_round_tracks_agreements(self):
        """Agreement patterns should accumulate across rounds."""
        tracker = ActiveIntrospectionTracker()

        r1 = RoundMetrics(agreements={"gpt": True, "gemini": False})
        r2 = RoundMetrics(agreements={"gpt": False, "gemini": False})

        tracker.update_round("claude", round_num=1, metrics=r1)
        tracker.update_round("claude", round_num=2, metrics=r2)

        summary = tracker.get_summary("claude")
        assert summary.agreement_patterns["gpt"] == [True, False]
        assert summary.agreement_patterns["gemini"] == [False, False]

    def test_set_goals(self):
        """set_goals should create summary and attach goals."""
        tracker = ActiveIntrospectionTracker()
        goals = IntrospectionGoals(
            agent_name="claude",
            target_acceptance_rate=0.8,
            focus_expertise=["security"],
        )

        tracker.set_goals("claude", goals)

        summary = tracker.get_summary("claude")
        assert summary is not None
        assert summary.goals is not None
        assert summary.goals.target_acceptance_rate == 0.8
        assert summary.goals.focus_expertise == ["security"]

    def test_set_goals_before_update(self):
        """Goals set before updates should persist through updates."""
        tracker = ActiveIntrospectionTracker()
        goals = IntrospectionGoals(agent_name="claude", focus_expertise=["api"])

        tracker.set_goals("claude", goals)
        tracker.update_round("claude", round_num=1, metrics=RoundMetrics(proposals_made=1))

        summary = tracker.get_summary("claude")
        assert summary.goals is not None
        assert summary.goals.focus_expertise == ["api"]
        assert summary.total_proposals == 1

    def test_get_all_summaries(self):
        """get_all_summaries should return copy of all tracked agents."""
        tracker = ActiveIntrospectionTracker()
        tracker.update_round("claude", 1, RoundMetrics())
        tracker.update_round("gpt", 1, RoundMetrics())

        all_summaries = tracker.get_all_summaries()
        assert len(all_summaries) == 2
        assert "claude" in all_summaries
        assert "gpt" in all_summaries

        # Should be a copy
        all_summaries.pop("claude")
        assert tracker.get_summary("claude") is not None

    def test_reset(self):
        """reset should clear all data."""
        tracker = ActiveIntrospectionTracker()
        tracker.set_goals("claude", IntrospectionGoals(agent_name="claude"))
        tracker.update_round("claude", 1, RoundMetrics())

        tracker.reset()

        assert tracker.agent_count == 0
        assert tracker.get_summary("claude") is None

    def test_update_round_sets_round_number(self):
        """update_round should set round_number on metrics if not set."""
        tracker = ActiveIntrospectionTracker()
        metrics = RoundMetrics()  # round_number defaults to 0

        tracker.update_round("claude", round_num=3, metrics=metrics)

        summary = tracker.get_summary("claude")
        assert summary.round_history[0].round_number == 3


# ---------------------------------------------------------------------------
# MetaReasoningEngine tests
# ---------------------------------------------------------------------------


class TestMetaReasoningEngine:
    """Tests for the MetaReasoningEngine class."""

    def _make_summary(self, **overrides) -> AgentPerformanceSummary:
        defaults = dict(agent_name="claude", rounds_completed=2)
        defaults.update(overrides)
        return AgentPerformanceSummary(**defaults)

    def test_no_guidance_for_zero_rounds(self):
        """Should return empty list when no rounds completed."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(rounds_completed=0)
        guidance = engine.generate_guidance(summary)
        assert guidance == []

    def test_proposal_high_acceptance(self):
        """Should generate positive proposal guidance for high acceptance."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_proposals=10, total_accepted=9)
        guidance = engine.generate_guidance(summary)
        proposal_lines = [g for g in guidance if "acceptance rate" in g.lower() or "proposals" in g.lower()]
        assert len(proposal_lines) >= 1
        assert "strong" in proposal_lines[0].lower()

    def test_proposal_low_acceptance(self):
        """Should generate improvement guidance for low acceptance."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_proposals=10, total_accepted=2)
        guidance = engine.generate_guidance(summary)
        proposal_lines = [g for g in guidance if "acceptance" in g.lower() or "proposals" in g.lower()]
        assert len(proposal_lines) >= 1
        assert "low" in proposal_lines[0].lower()

    def test_proposal_zero_acceptance(self):
        """Should generate guidance when no proposals accepted."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_proposals=5, total_accepted=0)
        guidance = engine.generate_guidance(summary)
        proposal_lines = [g for g in guidance if "proposals" in g.lower() or "accepted" in g.lower()]
        assert len(proposal_lines) >= 1
        assert "none" in proposal_lines[0].lower() or "different" in proposal_lines[0].lower()

    def test_critique_high_effectiveness(self):
        """Should generate positive critique guidance."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_critiques=10, total_critiques_effective=8)
        guidance = engine.generate_guidance(summary)
        critique_lines = [g for g in guidance if "critique" in g.lower()]
        assert len(critique_lines) >= 1
        assert "frequently" in critique_lines[0].lower() or "valued" in critique_lines[0].lower()

    def test_critique_low_effectiveness(self):
        """Should generate improvement guidance for low critique effectiveness."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_critiques=10, total_critiques_effective=2)
        guidance = engine.generate_guidance(summary)
        critique_lines = [g for g in guidance if "critique" in g.lower()]
        assert len(critique_lines) >= 1
        assert "few" in critique_lines[0].lower() or "concrete" in critique_lines[0].lower()

    def test_relationship_disagreer(self):
        """Should generate guidance about high-disagreement agents."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(
            agreement_patterns={"gpt": [False, False, False]},
        )
        guidance = engine.generate_guidance(summary)
        rel_lines = [g for g in guidance if "gpt" in g.lower()]
        assert len(rel_lines) >= 1
        assert "disagrees" in rel_lines[0].lower()

    def test_relationship_ally(self):
        """Should generate guidance about high-agreement agents."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(
            agreement_patterns={"gemini": [True, True, True]},
        )
        guidance = engine.generate_guidance(summary)
        rel_lines = [g for g in guidance if "gemini" in g.lower()]
        assert len(rel_lines) >= 1
        assert "agrees" in rel_lines[0].lower()

    def test_high_influence(self):
        """Should generate positive influence guidance."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_argument_influence=1.6)  # avg = 0.8
        guidance = engine.generate_guidance(summary)
        influence_lines = [g for g in guidance if "influence" in g.lower()]
        assert len(influence_lines) >= 1
        assert "strong" in influence_lines[0].lower()

    def test_low_influence(self):
        """Should generate improvement influence guidance."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(
            rounds_completed=3,
            total_argument_influence=0.3,  # avg = 0.1
        )
        guidance = engine.generate_guidance(summary)
        influence_lines = [g for g in guidance if "influence" in g.lower()]
        assert len(influence_lines) >= 1
        assert "limited" in influence_lines[0].lower()

    def test_goal_below_target(self):
        """Should generate guidance when below acceptance target."""
        engine = MetaReasoningEngine()
        goals = IntrospectionGoals(
            agent_name="claude",
            target_acceptance_rate=0.8,
        )
        summary = self._make_summary(
            total_proposals=10,
            total_accepted=3,
            goals=goals,
        )
        guidance = engine.generate_guidance(summary)
        goal_lines = [g for g in guidance if "target" in g.lower() or "goal" in g.lower()]
        assert len(goal_lines) >= 1
        assert "30%" in goal_lines[0] and "80%" in goal_lines[0]

    def test_goal_expertise_reminder(self):
        """Should remind about focus expertise."""
        engine = MetaReasoningEngine()
        goals = IntrospectionGoals(
            agent_name="claude",
            focus_expertise=["security", "testing"],
        )
        summary = self._make_summary(goals=goals)
        guidance = engine.generate_guidance(summary)
        expertise_lines = [g for g in guidance if "expertise" in g.lower()]
        assert len(expertise_lines) >= 1
        assert "security" in expertise_lines[0].lower()

    def test_format_for_prompt_empty(self):
        """Should return empty string when no guidance available."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(rounds_completed=0)
        result = engine.format_for_prompt(summary)
        assert result == ""

    def test_format_for_prompt_header(self):
        """Formatted output should include the right header."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(total_proposals=5, total_accepted=3)
        result = engine.format_for_prompt(summary)
        assert "## YOUR PERFORMANCE THIS DEBATE" in result
        assert "Proposal acceptance: 60%" in result

    def test_format_for_prompt_max_chars(self):
        """Formatted output should respect max_chars."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(
            total_proposals=10,
            total_accepted=8,
            total_critiques=10,
            total_critiques_effective=7,
            total_argument_influence=1.5,
            agreement_patterns={
                "gpt": [False, False, False],
                "gemini": [True, True, True],
            },
            goals=IntrospectionGoals(
                agent_name="claude",
                target_acceptance_rate=0.9,
                focus_expertise=["security", "api", "testing"],
            ),
        )
        result = engine.format_for_prompt(summary, max_chars=200)
        assert len(result) <= 200

    def test_format_for_prompt_includes_bullets(self):
        """Formatted output should include guidance as bullets."""
        engine = MetaReasoningEngine()
        summary = self._make_summary(
            total_proposals=10,
            total_accepted=9,
            total_critiques=5,
            total_critiques_effective=4,
        )
        result = engine.format_for_prompt(summary)
        assert "- " in result


# ---------------------------------------------------------------------------
# IntrospectionCache round-by-round update tests
# ---------------------------------------------------------------------------


class TestIntrospectionCacheRoundUpdates:
    """Tests for round-by-round updates through IntrospectionCache."""

    def _make_agent(self, name):
        return SimpleNamespace(name=name)

    def test_cache_initializes_active_tracker(self):
        """Warming cache should initialize the active tracker."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])

        assert cache.has_active_tracker

    def test_update_round_through_cache(self):
        """update_round on cache should delegate to active tracker."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])

        metrics = RoundMetrics(proposals_made=3, proposals_accepted=2)
        cache.update_round("claude", round_num=1, metrics=metrics)

        summary = cache.get_active_summary("claude")
        assert summary is not None
        assert summary.total_proposals == 3
        assert summary.total_accepted == 2

    def test_get_last_round_updated(self):
        """Should track the last round updated for each agent."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])

        assert cache.get_last_round_updated("claude") is None

        cache.update_round("claude", 1, RoundMetrics())
        assert cache.get_last_round_updated("claude") == 1

        cache.update_round("claude", 2, RoundMetrics())
        assert cache.get_last_round_updated("claude") == 2

    def test_get_active_summary_none_without_updates(self):
        """Should return None when no rounds have been recorded."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])

        assert cache.get_active_summary("claude") is None

    def test_invalidate_resets_active_tracker(self):
        """Invalidating cache should reset active tracker data."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        cache.warm(agents=[self._make_agent("claude")])
        cache.update_round("claude", 1, RoundMetrics(proposals_made=1))

        cache.invalidate()

        assert cache.get_active_summary("claude") is None

    def test_update_round_safe_without_tracker(self):
        """update_round should no-op when tracker is not available."""
        from aragora.introspection.cache import IntrospectionCache

        cache = IntrospectionCache()
        # Don't call warm() - tracker won't be initialized
        cache.update_round("claude", 1, RoundMetrics())  # Should not raise
        assert cache.get_active_summary("claude") is None


# ---------------------------------------------------------------------------
# Integration: end-to-end round update flow
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Integration test for the full active introspection flow."""

    def test_full_debate_flow(self):
        """Simulate a multi-round debate with active introspection."""
        tracker = ActiveIntrospectionTracker()
        engine = MetaReasoningEngine()

        # Set goals
        tracker.set_goals(
            "claude",
            IntrospectionGoals(
                agent_name="claude",
                focus_expertise=["security"],
                target_acceptance_rate=0.7,
            ),
        )

        # Round 1: Claude proposes, gets partial acceptance
        r1 = RoundMetrics(
            proposals_made=2,
            proposals_accepted=1,
            critiques_given=1,
            critiques_led_to_changes=1,
            agreements={"gpt": True, "gemini": False},
            argument_influence=0.5,
        )
        tracker.update_round("claude", 1, r1)

        summary = tracker.get_summary("claude")
        assert summary.rounds_completed == 1
        assert summary.proposal_acceptance_rate == 0.5

        # Round 2: Better performance
        r2 = RoundMetrics(
            proposals_made=1,
            proposals_accepted=1,
            critiques_given=2,
            critiques_led_to_changes=2,
            agreements={"gpt": False, "gemini": False},
            argument_influence=0.8,
        )
        tracker.update_round("claude", 2, r2)

        summary = tracker.get_summary("claude")
        assert summary.rounds_completed == 2
        assert summary.total_proposals == 3
        assert summary.total_accepted == 2
        assert summary.proposal_acceptance_rate == pytest.approx(2 / 3)

        # Generate guidance
        guidance = engine.generate_guidance(summary)
        assert len(guidance) > 0

        # Format for prompt
        prompt_section = engine.format_for_prompt(summary)
        assert "## YOUR PERFORMANCE THIS DEBATE" in prompt_section
        assert "Rounds: 2" in prompt_section

        # Verify agreement patterns accumulated
        assert summary.agreement_patterns["gpt"] == [True, False]
        assert summary.agreement_patterns["gemini"] == [False, False]

        # Check disagreer detection
        disagreers = summary.get_top_disagreers()
        assert any(name == "gemini" for name, _ in disagreers)


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public symbols are importable from the module."""

    def test_import_from_active(self):
        """All active introspection classes should be importable."""
        from aragora.introspection.active import (
            ActiveIntrospectionTracker,
            AgentPerformanceSummary,
            IntrospectionGoals,
            MetaReasoningEngine,
            RoundMetrics,
        )
        assert ActiveIntrospectionTracker is not None
        assert AgentPerformanceSummary is not None
        assert IntrospectionGoals is not None
        assert MetaReasoningEngine is not None
        assert RoundMetrics is not None

    def test_import_from_package(self):
        """Active classes should be importable from package __init__."""
        from aragora.introspection import (
            ActiveIntrospectionTracker,
            AgentPerformanceSummary,
            IntrospectionGoals,
            MetaReasoningEngine,
            RoundMetrics,
        )
        assert ActiveIntrospectionTracker is not None
        assert MetaReasoningEngine is not None

    def test_all_exports_updated(self):
        """__all__ should include active introspection symbols."""
        import aragora.introspection as mod
        expected_new = {
            "ActiveIntrospectionTracker",
            "AgentPerformanceSummary",
            "IntrospectionGoals",
            "MetaReasoningEngine",
            "RoundMetrics",
        }
        assert expected_new.issubset(set(mod.__all__))

"""Tests for Phase-Tagged ELO System."""

from __future__ import annotations

import pytest
from aragora.ranking.phase_elo import DEFAULT_ELO, PhaseELOTracker, PhaseRating


@pytest.fixture
def tracker():
    return PhaseELOTracker()


class TestPhaseRating:
    def test_defaults(self):
        pr = PhaseRating(agent_name="claude", domain="tech", phase="debate")
        assert pr.rating == DEFAULT_ELO
        assert pr.matches == 0
        assert pr.win_rate == 0.0

    def test_domain_phase_key(self):
        pr = PhaseRating(agent_name="claude", domain="tech", phase="debate")
        assert pr.domain_phase_key == "tech:debate"

    def test_win_rate(self):
        pr = PhaseRating(
            agent_name="claude", domain="tech", phase="debate", matches=10, wins=7, losses=3
        )
        assert pr.win_rate == 0.7


class TestUpdateDomainElo:
    def test_win_increases_rating(self, tracker):
        rating = tracker.update_domain_elo("claude", "tech:debate", won=True)
        assert rating > DEFAULT_ELO

    def test_loss_decreases_rating(self, tracker):
        rating = tracker.update_domain_elo("claude", "tech:debate", won=False)
        assert rating < DEFAULT_ELO

    def test_multiple_updates_accumulate(self, tracker):
        for _ in range(5):
            tracker.update_domain_elo("claude", "tech:debate", won=True)
        pr = tracker.get_rating("claude", "tech", "debate")
        assert pr is not None
        assert pr.matches == 5
        assert pr.wins == 5
        assert pr.rating > DEFAULT_ELO + 50

    def test_separate_phases(self, tracker):
        tracker.update_domain_elo("claude", "tech:debate", won=True)
        tracker.update_domain_elo("claude", "tech:execution", won=False)
        debate = tracker.get_rating("claude", "tech", "debate")
        execution = tracker.get_rating("claude", "tech", "execution")
        assert debate.rating > execution.rating

    def test_general_phase_default(self, tracker):
        tracker.update_domain_elo("claude", "tech", won=True)
        pr = tracker.get_rating("claude", "tech", "general")
        assert pr is not None


class TestBestAgentsForPhase:
    def test_ranking_order(self, tracker):
        for _ in range(5):
            tracker.update_domain_elo("claude", "tech:debate", won=True)
            tracker.update_domain_elo("gpt4", "tech:debate", won=False)
        best = tracker.get_best_agents_for_phase("tech", "debate")
        assert len(best) == 2
        assert best[0].agent_name == "claude"

    def test_limit(self, tracker):
        for i in range(10):
            tracker.update_domain_elo(f"agent-{i}", "tech:debate", won=True)
        best = tracker.get_best_agents_for_phase("tech", "debate", limit=3)
        assert len(best) == 3

    def test_empty(self, tracker):
        assert tracker.get_best_agents_for_phase("tech", "unknown") == []


class TestPhaseLeaderboard:
    def test_cross_domain(self, tracker):
        tracker.update_domain_elo("claude", "tech:debate", won=True)
        tracker.update_domain_elo("gpt4", "finance:debate", won=True)
        board = tracker.get_phase_leaderboard("debate")
        assert len(board) == 2

    def test_excludes_other_phases(self, tracker):
        tracker.update_domain_elo("claude", "tech:debate", won=True)
        tracker.update_domain_elo("claude", "tech:execution", won=True)
        board = tracker.get_phase_leaderboard("debate")
        assert len(board) == 1


class TestImprovementTrend:
    def test_trend(self, tracker):
        for i in range(5):
            tracker.update_domain_elo("claude", "tech:debate", won=True, influence_score=i * 0.2)
        trend = tracker.get_improvement_trend("claude", "tech", "debate")
        assert len(trend) == 5
        assert trend[-1] > trend[0]


class TestAgentProfile:
    def test_profile(self, tracker):
        tracker.update_domain_elo("claude", "tech:debate", won=True)
        tracker.update_domain_elo("claude", "tech:execution", won=False)
        tracker.update_domain_elo("claude", "finance:debate", won=True)
        profile = tracker.get_agent_profile("claude")
        assert len(profile) == 3

    def test_empty_profile(self, tracker):
        assert tracker.get_agent_profile("unknown") == {}


class TestSerialization:
    def test_to_dict(self, tracker):
        tracker.update_domain_elo("claude", "tech:debate", won=True)
        d = tracker.to_dict()
        assert "claude" in d
        assert "tech:debate" in d["claude"]
        assert d["claude"]["tech:debate"]["phase"] == "debate"
        assert d["claude"]["tech:debate"]["matches"] == 1

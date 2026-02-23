"""Tests for aragora.debate.novelty_selection_bridge — NoveltySelectionBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from aragora.debate.novelty_selection_bridge import (
    AgentNoveltyStats,
    NoveltySelectionBridge,
    NoveltySelectionBridgeConfig,
    create_novelty_selection_bridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNoveltyResult:
    """Minimal NoveltyResult stand-in."""

    round_num: int = 1
    per_agent_novelty: dict[str, float] = field(default_factory=dict)
    low_novelty_agents: list[str] = field(default_factory=list)


@dataclass
class FakeNoveltyTracker:
    """Minimal NoveltyTracker stand-in."""

    scores: list[FakeNoveltyResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentNoveltyStats
# ---------------------------------------------------------------------------


class TestAgentNoveltyStats:
    def test_defaults(self):
        s = AgentNoveltyStats(agent_name="claude")
        assert s.total_rounds == 0
        assert s.low_novelty_rounds == 0
        assert s.total_novelty_score == 0.0

    def test_avg_novelty_no_rounds(self):
        s = AgentNoveltyStats(agent_name="claude")
        assert s.avg_novelty == 1.0  # default to max

    def test_avg_novelty(self):
        s = AgentNoveltyStats(agent_name="claude", total_rounds=4, total_novelty_score=2.0)
        assert s.avg_novelty == pytest.approx(0.5)

    def test_low_novelty_rate_no_rounds(self):
        s = AgentNoveltyStats(agent_name="claude")
        assert s.low_novelty_rate == 0.0

    def test_low_novelty_rate(self):
        s = AgentNoveltyStats(agent_name="claude", total_rounds=10, low_novelty_rounds=3)
        assert s.low_novelty_rate == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# NoveltySelectionBridgeConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = NoveltySelectionBridgeConfig()
        assert c.min_rounds_for_penalty == 5
        assert c.low_novelty_penalty_weight == pytest.approx(0.2)
        assert c.high_novelty_bonus_weight == pytest.approx(0.1)
        assert c.max_penalty == pytest.approx(0.3)
        assert c.max_bonus == pytest.approx(0.2)
        assert c.decay_factor == pytest.approx(0.95)

    def test_custom(self):
        c = NoveltySelectionBridgeConfig(min_rounds_for_penalty=10, max_penalty=0.5)
        assert c.min_rounds_for_penalty == 10
        assert c.max_penalty == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — record_round_novelty
# ---------------------------------------------------------------------------


class TestRecordRoundNovelty:
    def test_records_agent_stats(self):
        bridge = NoveltySelectionBridge()
        result = FakeNoveltyResult(
            round_num=1,
            per_agent_novelty={"claude": 0.8, "gpt": 0.3},
            low_novelty_agents=["gpt"],
        )
        adjustments = bridge.record_round_novelty(result)
        assert "claude" in adjustments
        assert "gpt" in adjustments

        stats_claude = bridge.get_agent_stats("claude")
        assert stats_claude is not None
        assert stats_claude.total_rounds == 1
        assert stats_claude.total_novelty_score == pytest.approx(0.8)
        assert stats_claude.low_novelty_rounds == 0

        stats_gpt = bridge.get_agent_stats("gpt")
        assert stats_gpt is not None
        assert stats_gpt.low_novelty_rounds == 1

    def test_accumulates_across_rounds(self):
        bridge = NoveltySelectionBridge()
        for i in range(3):
            result = FakeNoveltyResult(
                round_num=i,
                per_agent_novelty={"claude": 0.5},
                low_novelty_agents=[],
            )
            bridge.record_round_novelty(result)

        stats = bridge.get_agent_stats("claude")
        assert stats.total_rounds == 3
        assert stats.total_novelty_score == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — _compute_adjustment
# ---------------------------------------------------------------------------


class TestComputeAdjustment:
    def test_below_min_rounds_no_adjustment(self):
        bridge = NoveltySelectionBridge()
        stats = AgentNoveltyStats(
            agent_name="claude",
            total_rounds=3,  # below default 5
            low_novelty_rounds=3,
            total_novelty_score=0.1,
        )
        assert bridge._compute_adjustment(stats) == 0.0

    def test_penalty_for_high_low_novelty_rate(self):
        bridge = NoveltySelectionBridge()
        stats = AgentNoveltyStats(
            agent_name="claude",
            total_rounds=10,
            low_novelty_rounds=5,  # 50% low novelty rate
            total_novelty_score=5.0,  # avg 0.5
        )
        adj = bridge._compute_adjustment(stats)
        assert adj < 0  # negative = penalty

    def test_bonus_for_high_novelty(self):
        bridge = NoveltySelectionBridge()
        stats = AgentNoveltyStats(
            agent_name="claude",
            total_rounds=10,
            low_novelty_rounds=0,
            total_novelty_score=8.0,  # avg 0.8, above threshold 0.6
        )
        adj = bridge._compute_adjustment(stats)
        assert adj > 0  # positive = bonus

    def test_penalty_capped_at_max(self):
        bridge = NoveltySelectionBridge(config=NoveltySelectionBridgeConfig(max_penalty=0.3))
        stats = AgentNoveltyStats(
            agent_name="claude",
            total_rounds=10,
            low_novelty_rounds=10,  # 100% low novelty
            total_novelty_score=0.5,  # avg 0.05
        )
        adj = bridge._compute_adjustment(stats)
        assert adj >= -0.35  # bounded

    def test_very_low_avg_novelty_extra_penalty(self):
        bridge = NoveltySelectionBridge()
        stats = AgentNoveltyStats(
            agent_name="claude",
            total_rounds=10,
            low_novelty_rounds=8,
            total_novelty_score=1.0,  # avg 0.1, below threshold*2=0.3
        )
        adj = bridge._compute_adjustment(stats)
        assert adj < 0


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — get_novelty_penalty / bonus / combined
# ---------------------------------------------------------------------------


class TestGetAdjustments:
    def _build_bridge_with_stats(self):
        bridge = NoveltySelectionBridge()
        # Record enough rounds for penalty
        for i in range(6):
            result = FakeNoveltyResult(
                round_num=i,
                per_agent_novelty={"bad": 0.05, "good": 0.9},
                low_novelty_agents=["bad"],
            )
            bridge.record_round_novelty(result)
        return bridge

    def test_penalty_negative(self):
        bridge = self._build_bridge_with_stats()
        assert bridge.get_novelty_penalty("bad") <= 0

    def test_penalty_unknown_agent(self):
        bridge = NoveltySelectionBridge()
        assert bridge.get_novelty_penalty("unknown") == 0.0

    def test_bonus_positive(self):
        bridge = self._build_bridge_with_stats()
        assert bridge.get_novelty_bonus("good") >= 0

    def test_bonus_unknown_agent(self):
        bridge = NoveltySelectionBridge()
        assert bridge.get_novelty_bonus("unknown") == 0.0

    def test_combined_adjustment(self):
        bridge = self._build_bridge_with_stats()
        adj = bridge.get_combined_adjustment("bad")
        assert isinstance(adj, float)

    def test_all_adjustments(self):
        bridge = self._build_bridge_with_stats()
        all_adj = bridge.get_all_adjustments()
        assert "bad" in all_adj
        assert "good" in all_adj


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — get_low/high_novelty_agents
# ---------------------------------------------------------------------------


class TestAgentLists:
    def _build_bridge(self):
        bridge = NoveltySelectionBridge()
        for i in range(6):
            result = FakeNoveltyResult(
                round_num=i,
                per_agent_novelty={"bad": 0.05, "good": 0.9, "mid": 0.4},
                low_novelty_agents=["bad"],
            )
            bridge.record_round_novelty(result)
        return bridge

    def test_low_novelty_agents(self):
        bridge = self._build_bridge()
        low = bridge.get_low_novelty_agents()
        assert "bad" in low
        assert "good" not in low

    def test_high_novelty_agents(self):
        bridge = self._build_bridge()
        high = bridge.get_high_novelty_agents()
        assert "good" in high
        assert "bad" not in high

    def test_below_min_rounds_excluded(self):
        bridge = NoveltySelectionBridge()
        result = FakeNoveltyResult(
            round_num=1,
            per_agent_novelty={"new": 0.05},
            low_novelty_agents=["new"],
        )
        bridge.record_round_novelty(result)
        assert bridge.get_low_novelty_agents() == []


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — record_from_tracker
# ---------------------------------------------------------------------------


class TestRecordFromTracker:
    def test_no_tracker(self):
        bridge = NoveltySelectionBridge(novelty_tracker=None)
        assert bridge.record_from_tracker() == {}

    def test_with_tracker(self):
        tracker = FakeNoveltyTracker(
            scores=[
                FakeNoveltyResult(
                    round_num=i,
                    per_agent_novelty={"claude": 0.5},
                    low_novelty_agents=[],
                )
                for i in range(3)
            ]
        )
        bridge = NoveltySelectionBridge(novelty_tracker=tracker)
        adj = bridge.record_from_tracker()
        assert "claude" in adj
        assert bridge.get_agent_stats("claude").total_rounds == 3


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — sync_to_selection_feedback
# ---------------------------------------------------------------------------


class TestSyncToFeedback:
    def test_no_feedback_loop(self):
        bridge = NoveltySelectionBridge(selection_feedback=None)
        assert bridge.sync_to_selection_feedback() == 0

    def test_syncs_adjustments(self):
        feedback = MagicMock()
        feedback.get_selection_adjustment.return_value = 0.0
        feedback.get_agent_state.return_value = MagicMock()  # truthy
        feedback._selection_adjustments = {}

        bridge = NoveltySelectionBridge(selection_feedback=feedback)
        # Record enough rounds
        for i in range(6):
            result = FakeNoveltyResult(
                round_num=i,
                per_agent_novelty={"claude": 0.1},
                low_novelty_agents=["claude"],
            )
            bridge.record_round_novelty(result)

        updated = bridge.sync_to_selection_feedback()
        assert updated >= 1
        assert "claude" in feedback._selection_adjustments

    def test_skips_unknown_agents(self):
        feedback = MagicMock()
        feedback.get_selection_adjustment.return_value = 0.0
        feedback.get_agent_state.return_value = None  # not found

        bridge = NoveltySelectionBridge(selection_feedback=feedback)
        bridge._novelty_adjustments = {"ghost": -0.1}
        updated = bridge.sync_to_selection_feedback()
        assert updated == 0


# ---------------------------------------------------------------------------
# NoveltySelectionBridge — decay, stats, reset
# ---------------------------------------------------------------------------


class TestDecayStatsReset:
    def _build_bridge(self):
        bridge = NoveltySelectionBridge()
        for i in range(6):
            bridge.record_round_novelty(
                FakeNoveltyResult(
                    round_num=i,
                    per_agent_novelty={"claude": 0.5},
                    low_novelty_agents=[],
                )
            )
        return bridge

    def test_apply_decay(self):
        bridge = self._build_bridge()
        original_score = bridge.get_agent_stats("claude").total_novelty_score
        bridge.apply_decay()
        decayed_score = bridge.get_agent_stats("claude").total_novelty_score
        assert decayed_score < original_score

    def test_get_stats(self):
        bridge = self._build_bridge()
        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 1
        assert stats["total_rounds_recorded"] == 6
        assert "low_novelty_agents" in stats
        assert "high_novelty_agents" in stats
        assert "avg_adjustment" in stats

    def test_get_all_stats(self):
        bridge = self._build_bridge()
        all_stats = bridge.get_all_stats()
        assert "claude" in all_stats
        assert isinstance(all_stats["claude"], AgentNoveltyStats)

    def test_reset(self):
        bridge = self._build_bridge()
        bridge.reset()
        assert bridge.get_agent_stats("claude") is None
        assert bridge.get_all_adjustments() == {}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        bridge = create_novelty_selection_bridge()
        assert isinstance(bridge, NoveltySelectionBridge)
        assert bridge.novelty_tracker is None
        assert bridge.selection_feedback is None

    def test_create_with_config_kwargs(self):
        bridge = create_novelty_selection_bridge(
            min_rounds_for_penalty=10,
            max_penalty=0.5,
        )
        assert bridge.config.min_rounds_for_penalty == 10
        assert bridge.config.max_penalty == pytest.approx(0.5)

    def test_create_with_tracker(self):
        tracker = FakeNoveltyTracker()
        bridge = create_novelty_selection_bridge(novelty_tracker=tracker)
        assert bridge.novelty_tracker is tracker

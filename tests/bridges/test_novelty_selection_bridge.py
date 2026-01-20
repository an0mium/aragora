"""Tests for NoveltySelectionBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from aragora.debate.novelty_selection_bridge import (
    NoveltySelectionBridge,
    NoveltySelectionBridgeConfig,
    AgentNoveltyStats,
    create_novelty_selection_bridge,
)


@dataclass
class MockNoveltyResult:
    """Mock novelty result for testing."""

    round_num: int = 1
    per_agent_novelty: Dict[str, float] = field(default_factory=dict)
    low_novelty_agents: List[str] = field(default_factory=list)
    avg_novelty: float = 0.5


class MockNoveltyTracker:
    """Mock novelty tracker."""

    def __init__(self):
        self.scores: List[MockNoveltyResult] = []

    def add_result(self, result: MockNoveltyResult) -> None:
        """Add a novelty result."""
        self.scores.append(result)


class MockSelectionFeedbackLoop:
    """Mock selection feedback loop."""

    def __init__(self):
        self._selection_adjustments: Dict[str, float] = {}
        self._agent_states: Dict[str, object] = {}

    def get_selection_adjustment(self, agent_name: str) -> float:
        """Get current adjustment."""
        return self._selection_adjustments.get(agent_name, 0.0)

    def get_agent_state(self, agent_name: str) -> Optional[object]:
        """Get agent state."""
        return self._agent_states.get(agent_name)

    def add_agent_state(self, agent_name: str) -> None:
        """Add agent state."""
        self._agent_states[agent_name] = object()


class TestAgentNoveltyStats:
    """Tests for AgentNoveltyStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = AgentNoveltyStats(agent_name="test")
        assert stats.total_rounds == 0
        assert stats.low_novelty_rounds == 0
        assert stats.avg_novelty == 1.0  # Default max

    def test_avg_novelty(self):
        """Test average novelty calculation."""
        stats = AgentNoveltyStats(
            agent_name="test",
            total_rounds=10,
            total_novelty_score=7.0,
        )
        assert stats.avg_novelty == 0.7

    def test_low_novelty_rate(self):
        """Test low novelty rate calculation."""
        stats = AgentNoveltyStats(
            agent_name="test",
            total_rounds=10,
            low_novelty_rounds=3,
        )
        assert stats.low_novelty_rate == 0.3


class TestNoveltySelectionBridge:
    """Tests for NoveltySelectionBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = NoveltySelectionBridge()
        assert bridge.novelty_tracker is None
        assert bridge.selection_feedback is None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = NoveltySelectionBridgeConfig(
            min_rounds_for_penalty=3,
            low_novelty_penalty_weight=0.3,
        )
        bridge = NoveltySelectionBridge(config=config)
        assert bridge.config.min_rounds_for_penalty == 3
        assert bridge.config.low_novelty_penalty_weight == 0.3

    def test_record_round_novelty(self):
        """Test recording novelty results."""
        bridge = NoveltySelectionBridge()

        result = MockNoveltyResult(
            round_num=1,
            per_agent_novelty={"claude": 0.8, "gpt-4": 0.3},
            low_novelty_agents=["gpt-4"],
        )

        adjustments = bridge.record_round_novelty(result)

        assert "claude" in adjustments
        assert "gpt-4" in adjustments
        assert bridge._agent_stats["gpt-4"].low_novelty_rounds == 1

    def test_record_from_tracker(self):
        """Test recording from tracker."""
        tracker = MockNoveltyTracker()
        tracker.add_result(
            MockNoveltyResult(
                round_num=1,
                per_agent_novelty={"claude": 0.7},
            )
        )
        tracker.add_result(
            MockNoveltyResult(
                round_num=2,
                per_agent_novelty={"claude": 0.8},
            )
        )

        bridge = NoveltySelectionBridge(novelty_tracker=tracker)
        adjustments = bridge.record_from_tracker()

        assert "claude" in adjustments
        stats = bridge.get_agent_stats("claude")
        assert stats.total_rounds == 2

    def test_get_novelty_penalty(self):
        """Test getting novelty penalty."""
        bridge = NoveltySelectionBridge(
            config=NoveltySelectionBridgeConfig(min_rounds_for_penalty=5)
        )

        # Record multiple low novelty rounds
        for i in range(10):
            result = MockNoveltyResult(
                round_num=i,
                per_agent_novelty={"low-novelty-agent": 0.1},
                low_novelty_agents=["low-novelty-agent"],
            )
            bridge.record_round_novelty(result)

        penalty = bridge.get_novelty_penalty("low-novelty-agent")
        assert penalty <= 0  # Should be negative (penalty)

    def test_get_novelty_bonus(self):
        """Test getting novelty bonus."""
        bridge = NoveltySelectionBridge(
            config=NoveltySelectionBridgeConfig(
                min_rounds_for_penalty=5,
                high_novelty_threshold=0.6,
            )
        )

        # Record high novelty rounds
        for i in range(10):
            result = MockNoveltyResult(
                round_num=i,
                per_agent_novelty={"high-novelty-agent": 0.9},
            )
            bridge.record_round_novelty(result)

        bonus = bridge.get_novelty_bonus("high-novelty-agent")
        assert bonus >= 0  # Should be positive (bonus)

    def test_get_combined_adjustment(self):
        """Test getting combined adjustment."""
        bridge = NoveltySelectionBridge()
        bridge._novelty_adjustments["test"] = -0.1

        adjustment = bridge.get_combined_adjustment("test")
        assert adjustment == -0.1

        # Unknown agent
        assert bridge.get_combined_adjustment("unknown") == 0.0

    def test_get_low_novelty_agents(self):
        """Test getting low novelty agents."""
        bridge = NoveltySelectionBridge(
            config=NoveltySelectionBridgeConfig(min_rounds_for_penalty=5)
        )

        # Add agent with high low-novelty rate
        bridge._agent_stats["low"] = AgentNoveltyStats(
            agent_name="low",
            total_rounds=10,
            low_novelty_rounds=5,
        )
        # Add agent with low low-novelty rate
        bridge._agent_stats["high"] = AgentNoveltyStats(
            agent_name="high",
            total_rounds=10,
            low_novelty_rounds=1,
        )

        low_novelty = bridge.get_low_novelty_agents(threshold=0.3)
        assert "low" in low_novelty
        assert "high" not in low_novelty

    def test_get_high_novelty_agents(self):
        """Test getting high novelty agents."""
        bridge = NoveltySelectionBridge(
            config=NoveltySelectionBridgeConfig(min_rounds_for_penalty=5)
        )

        bridge._agent_stats["high"] = AgentNoveltyStats(
            agent_name="high",
            total_rounds=10,
            total_novelty_score=8.0,  # avg 0.8
        )
        bridge._agent_stats["low"] = AgentNoveltyStats(
            agent_name="low",
            total_rounds=10,
            total_novelty_score=3.0,  # avg 0.3
        )

        high_novelty = bridge.get_high_novelty_agents(threshold=0.6)
        assert "high" in high_novelty
        assert "low" not in high_novelty

    def test_sync_to_selection_feedback(self):
        """Test syncing to selection feedback."""
        feedback = MockSelectionFeedbackLoop()
        feedback.add_agent_state("claude")

        bridge = NoveltySelectionBridge(selection_feedback=feedback)
        bridge._novelty_adjustments["claude"] = -0.1

        updated = bridge.sync_to_selection_feedback()
        assert updated == 1
        assert feedback._selection_adjustments["claude"] == -0.1

    def test_sync_no_feedback(self):
        """Test sync with no feedback loop."""
        bridge = NoveltySelectionBridge()
        bridge._novelty_adjustments["test"] = 0.1

        updated = bridge.sync_to_selection_feedback()
        assert updated == 0

    def test_get_all_stats(self):
        """Test getting all stats."""
        bridge = NoveltySelectionBridge()
        bridge._agent_stats["claude"] = AgentNoveltyStats(agent_name="claude")
        bridge._agent_stats["gpt-4"] = AgentNoveltyStats(agent_name="gpt-4")

        all_stats = bridge.get_all_stats()
        assert len(all_stats) == 2

    def test_apply_decay(self):
        """Test applying decay to stats."""
        bridge = NoveltySelectionBridge(
            config=NoveltySelectionBridgeConfig(decay_factor=0.5)
        )
        bridge._agent_stats["test"] = AgentNoveltyStats(
            agent_name="test",
            total_rounds=100,
            low_novelty_rounds=50,
            total_novelty_score=70.0,
        )

        bridge.apply_decay()

        stats = bridge.get_agent_stats("test")
        assert stats.total_rounds == 50  # Decayed
        assert stats.low_novelty_rounds == 25  # Decayed

    def test_reset(self):
        """Test resetting bridge."""
        bridge = NoveltySelectionBridge()
        bridge._agent_stats["test"] = AgentNoveltyStats(agent_name="test")
        bridge._novelty_adjustments["test"] = 0.1

        bridge.reset()

        assert len(bridge._agent_stats) == 0
        assert len(bridge._novelty_adjustments) == 0

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = NoveltySelectionBridge()
        bridge._agent_stats["test"] = AgentNoveltyStats(
            agent_name="test", total_rounds=5
        )

        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 1
        assert stats["total_rounds_recorded"] == 5

    def test_factory_function(self):
        """Test factory function."""
        bridge = create_novelty_selection_bridge(
            min_rounds_for_penalty=3,
            low_novelty_penalty_weight=0.25,
        )
        assert bridge.config.min_rounds_for_penalty == 3
        assert bridge.config.low_novelty_penalty_weight == 0.25

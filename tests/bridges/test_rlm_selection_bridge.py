"""Tests for RLMSelectionBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aragora.rlm.rlm_selection_bridge import (
    RLMSelectionBridge,
    RLMSelectionBridgeConfig,
    AgentRLMStats,
    create_rlm_selection_bridge,
)


@dataclass
class MockCompressionResult:
    """Mock compression result for testing."""

    estimated_fidelity: float = 0.85
    time_seconds: float = 1.5
    cache_hits: int = 3
    compression_ratio: Dict[str, float] = field(default_factory=dict)


@dataclass
class MockRLMResult:
    """Mock RLM query result for testing."""

    confidence: float = 0.8
    sub_calls_made: int = 5
    tokens_processed: int = 1000
    time_seconds: float = 2.0


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


class TestAgentRLMStats:
    """Tests for AgentRLMStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = AgentRLMStats(agent_name="test")
        assert stats.total_operations == 0
        assert stats.total_compressions == 0
        assert stats.avg_fidelity == 1.0  # Default max

    def test_avg_fidelity(self):
        """Test average fidelity calculation."""
        stats = AgentRLMStats(
            agent_name="test",
            total_compressions=10,
            total_fidelity=8.5,
        )
        assert stats.avg_fidelity == 0.85

    def test_avg_confidence(self):
        """Test average confidence calculation."""
        stats = AgentRLMStats(
            agent_name="test",
            total_queries=10,
            total_confidence=7.0,
        )
        assert stats.avg_confidence == 0.7

    def test_avg_sub_calls_per_query(self):
        """Test average sub-calls per query."""
        stats = AgentRLMStats(
            agent_name="test",
            total_queries=10,
            total_sub_calls=30,
        )
        assert stats.avg_sub_calls_per_query == 3.0

    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        stats = AgentRLMStats(
            agent_name="test",
            total_operations=100,
            total_compressions=50,
            total_queries=50,
            total_fidelity=42.5,  # avg 0.85
            total_confidence=40.0,  # avg 0.8
            total_sub_calls=150,  # avg 3
            total_cache_hits=30,
        )
        score = stats.efficiency_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Good stats should score well

    def test_efficiency_score_no_operations(self):
        """Test efficiency score with no operations."""
        stats = AgentRLMStats(agent_name="test")
        assert stats.efficiency_score == 0.5  # Neutral


class TestRLMSelectionBridge:
    """Tests for RLMSelectionBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = RLMSelectionBridge()
        assert bridge.rlm_bridge is None
        assert bridge.selection_feedback is None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = RLMSelectionBridgeConfig(
            min_operations_for_boost=3,
            compression_boost_weight=0.2,
        )
        bridge = RLMSelectionBridge(config=config)
        assert bridge.config.min_operations_for_boost == 3
        assert bridge.config.compression_boost_weight == 0.2

    def test_record_compression(self):
        """Test recording compression result."""
        bridge = RLMSelectionBridge()

        result = MockCompressionResult(
            estimated_fidelity=0.9,
            compression_ratio={"SUMMARY": 0.3, "ABSTRACT": 0.1},
        )

        adjustment = bridge.record_compression("claude", result)

        stats = bridge.get_agent_stats("claude")
        assert stats.total_compressions == 1
        assert stats.total_fidelity == 0.9
        assert isinstance(adjustment, float)

    def test_record_query(self):
        """Test recording query result."""
        bridge = RLMSelectionBridge()

        result = MockRLMResult(
            confidence=0.85,
            sub_calls_made=3,
            tokens_processed=500,
        )

        adjustment = bridge.record_query("claude", result)

        stats = bridge.get_agent_stats("claude")
        assert stats.total_queries == 1
        assert stats.total_confidence == 0.85
        assert stats.total_sub_calls == 3

    def test_record_rlm_operation_compression(self):
        """Test generic record with compression result."""
        bridge = RLMSelectionBridge()

        result = MockCompressionResult(estimated_fidelity=0.9)
        adjustment = bridge.record_rlm_operation("claude", result)

        stats = bridge.get_agent_stats("claude")
        assert stats.total_compressions == 1

    def test_record_rlm_operation_query(self):
        """Test generic record with query result."""
        bridge = RLMSelectionBridge()

        result = MockRLMResult(confidence=0.8)
        adjustment = bridge.record_rlm_operation("claude", result)

        stats = bridge.get_agent_stats("claude")
        assert stats.total_queries == 1

    def test_get_compression_boost(self):
        """Test getting compression boost."""
        bridge = RLMSelectionBridge(config=RLMSelectionBridgeConfig(min_operations_for_boost=3))

        # Record multiple high-quality compressions
        for _ in range(5):
            bridge.record_compression(
                "good-compressor",
                MockCompressionResult(estimated_fidelity=0.95),
            )

        boost = bridge.get_compression_boost("good-compressor")
        assert boost >= 0

    def test_get_compression_penalty(self):
        """Test getting compression penalty."""
        bridge = RLMSelectionBridge(
            config=RLMSelectionBridgeConfig(
                min_operations_for_boost=3,
                min_fidelity_threshold=0.7,
            )
        )

        # Record low-quality compressions
        for _ in range(5):
            bridge.record_compression(
                "bad-compressor",
                MockCompressionResult(estimated_fidelity=0.3),
            )

        penalty = bridge.get_compression_penalty("bad-compressor")
        assert penalty <= 0  # Penalty is negative

    def test_get_rlm_efficient_agents(self):
        """Test getting RLM efficient agents."""
        bridge = RLMSelectionBridge(config=RLMSelectionBridgeConfig(min_operations_for_boost=5))

        # Add efficient agent
        bridge._agent_stats["efficient"] = AgentRLMStats(
            agent_name="efficient",
            total_operations=100,
            total_compressions=50,
            total_queries=50,
            total_fidelity=47.5,  # 0.95 avg
            total_confidence=42.5,  # 0.85 avg
            total_sub_calls=100,  # 2 avg
        )

        # Add inefficient agent
        bridge._agent_stats["inefficient"] = AgentRLMStats(
            agent_name="inefficient",
            total_operations=100,
            total_compressions=50,
            total_queries=50,
            total_fidelity=25.0,  # 0.5 avg
            total_confidence=25.0,  # 0.5 avg
            total_sub_calls=500,  # 10 avg
        )

        efficient = bridge.get_rlm_efficient_agents(threshold=0.6)
        assert "efficient" in efficient
        assert "inefficient" not in efficient

    def test_get_best_agents_for_long_debates(self):
        """Test getting best agents for long debates."""
        bridge = RLMSelectionBridge(config=RLMSelectionBridgeConfig(min_operations_for_boost=5))

        # Add agents with different efficiency
        bridge._agent_stats["best"] = AgentRLMStats(
            agent_name="best",
            total_operations=50,
            total_compressions=25,
            total_queries=25,
            total_fidelity=23.75,
            total_confidence=21.25,
            total_sub_calls=50,
        )
        bridge._agent_stats["good"] = AgentRLMStats(
            agent_name="good",
            total_operations=50,
            total_compressions=25,
            total_queries=25,
            total_fidelity=20.0,
            total_confidence=17.5,
            total_sub_calls=100,
        )

        best = bridge.get_best_agents_for_long_debates(top_n=1)
        assert best[0] == "best"

    def test_sync_to_selection_feedback(self):
        """Test syncing to selection feedback."""
        feedback = MockSelectionFeedbackLoop()
        feedback.add_agent_state("claude")

        bridge = RLMSelectionBridge(selection_feedback=feedback)
        bridge._rlm_adjustments["claude"] = 0.1

        updated = bridge.sync_to_selection_feedback()
        assert updated == 1
        assert feedback._selection_adjustments["claude"] == 0.1

    def test_sync_no_feedback(self):
        """Test sync with no feedback loop."""
        bridge = RLMSelectionBridge()
        bridge._rlm_adjustments["test"] = 0.1

        updated = bridge.sync_to_selection_feedback()
        assert updated == 0

    def test_get_all_adjustments(self):
        """Test getting all adjustments."""
        bridge = RLMSelectionBridge()
        bridge._rlm_adjustments["a"] = 0.1
        bridge._rlm_adjustments["b"] = -0.05

        adjustments = bridge.get_all_adjustments()
        assert "a" in adjustments
        assert "b" in adjustments

    def test_get_all_stats(self):
        """Test getting all stats."""
        bridge = RLMSelectionBridge()
        bridge._agent_stats["a"] = AgentRLMStats(agent_name="a")
        bridge._agent_stats["b"] = AgentRLMStats(agent_name="b")

        all_stats = bridge.get_all_stats()
        assert len(all_stats) == 2

    def test_reset(self):
        """Test resetting bridge."""
        bridge = RLMSelectionBridge()
        bridge._agent_stats["test"] = AgentRLMStats(agent_name="test")
        bridge._rlm_adjustments["test"] = 0.1

        bridge.reset()

        assert len(bridge._agent_stats) == 0
        assert len(bridge._rlm_adjustments) == 0

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = RLMSelectionBridge()
        bridge._agent_stats["test"] = AgentRLMStats(
            agent_name="test",
            total_operations=10,
            total_compressions=5,
            total_queries=5,
        )

        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 1
        assert stats["total_operations"] == 10
        assert stats["total_compressions"] == 5
        assert stats["total_queries"] == 5

    def test_factory_function(self):
        """Test factory function."""
        bridge = create_rlm_selection_bridge(
            min_operations_for_boost=10,
            compression_boost_weight=0.25,
        )
        assert bridge.config.min_operations_for_boost == 10
        assert bridge.config.compression_boost_weight == 0.25

    def test_adjustment_bounded(self):
        """Test that adjustments are bounded."""
        bridge = RLMSelectionBridge(
            config=RLMSelectionBridgeConfig(
                min_operations_for_boost=3,
                max_boost=0.25,
                low_fidelity_penalty=0.1,
            )
        )

        # Record many high-quality operations
        for _ in range(100):
            bridge.record_compression(
                "excellent",
                MockCompressionResult(estimated_fidelity=0.99),
            )
            bridge.record_query(
                "excellent",
                MockRLMResult(confidence=0.99, sub_calls_made=1),
            )

        adjustment = bridge.get_combined_adjustment("excellent")
        assert adjustment <= bridge.config.max_boost

        # Record many low-quality operations
        for _ in range(100):
            bridge.record_compression(
                "poor",
                MockCompressionResult(estimated_fidelity=0.1),
            )

        penalty = bridge.get_combined_adjustment("poor")
        assert penalty >= -bridge.config.low_fidelity_penalty

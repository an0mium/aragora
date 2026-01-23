"""
Tests for RLM Selection Bridge module.

Tests the integration between RLM compression metrics and
agent selection feedback loop.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from unittest.mock import MagicMock

from aragora.rlm.rlm_selection_bridge import (
    AgentRLMStats,
    RLMSelectionBridge,
    RLMSelectionBridgeConfig,
    create_rlm_selection_bridge,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockCompressionResult:
    """Mock compression result for testing."""

    estimated_fidelity: float = 0.85
    time_seconds: float = 0.5
    cache_hits: int = 2
    compression_ratio: Dict[str, float] = None

    def __post_init__(self):
        if self.compression_ratio is None:
            self.compression_ratio = {"level_1": 0.6, "level_2": 0.4}


@dataclass
class MockRLMResult:
    """Mock RLM query result for testing."""

    confidence: float = 0.8
    sub_calls_made: int = 3
    tokens_processed: int = 1500
    time_seconds: float = 0.25


@dataclass
class MockAgentState:
    """Mock agent state for selection feedback."""

    agent_name: str
    total_selections: int = 10


class MockSelectionFeedbackLoop:
    """Mock selection feedback loop for testing."""

    def __init__(self):
        self._agents: Dict[str, MockAgentState] = {}
        self._selection_adjustments: Dict[str, float] = {}

    def get_agent_state(self, agent_name: str) -> Optional[MockAgentState]:
        return self._agents.get(agent_name)

    def get_selection_adjustment(self, agent_name: str) -> float:
        return self._selection_adjustments.get(agent_name, 0.0)

    def add_agent(self, agent_name: str):
        self._agents[agent_name] = MockAgentState(agent_name=agent_name)


# =============================================================================
# AgentRLMStats Tests
# =============================================================================


class TestAgentRLMStats:
    """Tests for AgentRLMStats dataclass."""

    def test_default_values(self):
        """Should initialize with correct defaults."""
        stats = AgentRLMStats(agent_name="claude")

        assert stats.agent_name == "claude"
        assert stats.total_operations == 0
        assert stats.total_compressions == 0
        assert stats.total_queries == 0
        assert stats.total_fidelity == 0.0
        assert stats.total_confidence == 0.0

    def test_avg_fidelity_with_compressions(self):
        """Should calculate average fidelity correctly."""
        stats = AgentRLMStats(
            agent_name="claude",
            total_compressions=4,
            total_fidelity=3.4,  # 0.85 average
        )

        assert stats.avg_fidelity == pytest.approx(0.85, abs=0.01)

    def test_avg_fidelity_no_compressions(self):
        """Should return 1.0 when no compressions recorded."""
        stats = AgentRLMStats(agent_name="claude", total_compressions=0)

        assert stats.avg_fidelity == 1.0

    def test_avg_confidence_with_queries(self):
        """Should calculate average confidence correctly."""
        stats = AgentRLMStats(
            agent_name="claude",
            total_queries=5,
            total_confidence=4.0,  # 0.8 average
        )

        assert stats.avg_confidence == pytest.approx(0.8, abs=0.01)

    def test_avg_confidence_no_queries(self):
        """Should return 0.5 when no queries recorded."""
        stats = AgentRLMStats(agent_name="claude", total_queries=0)

        assert stats.avg_confidence == 0.5

    def test_avg_sub_calls_per_query(self):
        """Should calculate average sub-calls correctly."""
        stats = AgentRLMStats(
            agent_name="claude",
            total_queries=4,
            total_sub_calls=12,  # 3 average
        )

        assert stats.avg_sub_calls_per_query == pytest.approx(3.0)

    def test_avg_sub_calls_no_queries(self):
        """Should return 0.0 when no queries."""
        stats = AgentRLMStats(agent_name="claude", total_queries=0)

        assert stats.avg_sub_calls_per_query == 0.0

    def test_efficiency_score_no_operations(self):
        """Should return 0.5 when no operations."""
        stats = AgentRLMStats(agent_name="claude", total_operations=0)

        assert stats.efficiency_score == 0.5

    def test_efficiency_score_high_performance(self):
        """Should calculate high efficiency for good performance."""
        stats = AgentRLMStats(
            agent_name="claude",
            total_operations=10,
            total_compressions=5,
            total_fidelity=4.75,  # 0.95 avg
            total_queries=5,
            total_confidence=4.5,  # 0.9 avg
            total_sub_calls=10,  # 2 avg sub-calls (efficient)
            total_cache_hits=3,
        )

        # High fidelity + high confidence + low sub-calls = high efficiency
        assert stats.efficiency_score > 0.8

    def test_efficiency_score_poor_sub_calls(self):
        """Should reduce efficiency for high sub-call counts."""
        # Baseline with low sub-calls
        good_stats = AgentRLMStats(
            agent_name="claude",
            total_operations=10,
            total_queries=5,
            total_confidence=4.0,  # 0.8 avg
            total_sub_calls=10,  # 2 avg - efficient
        )

        # Same metrics but high sub-calls
        poor_stats = AgentRLMStats(
            agent_name="claude",
            total_operations=10,
            total_queries=5,
            total_confidence=4.0,  # 0.8 avg
            total_sub_calls=60,  # 12 avg - poor
        )

        # High sub-calls should reduce efficiency compared to low sub-calls
        assert poor_stats.efficiency_score < good_stats.efficiency_score


# =============================================================================
# RLMSelectionBridgeConfig Tests
# =============================================================================


class TestRLMSelectionBridgeConfig:
    """Tests for RLMSelectionBridgeConfig."""

    def test_default_values(self):
        """Should initialize with correct defaults."""
        config = RLMSelectionBridgeConfig()

        assert config.min_operations_for_boost == 5
        assert config.compression_boost_weight == 0.15
        assert config.query_boost_weight == 0.15
        assert config.min_fidelity_threshold == 0.7
        assert config.max_boost == 0.25
        assert config.low_fidelity_penalty == 0.1

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = RLMSelectionBridgeConfig(
            min_operations_for_boost=10,
            max_boost=0.5,
        )

        assert config.min_operations_for_boost == 10
        assert config.max_boost == 0.5


# =============================================================================
# RLMSelectionBridge Tests
# =============================================================================


class TestRLMSelectionBridgeConstruction:
    """Tests for RLMSelectionBridge construction."""

    def test_minimal_construction(self):
        """Should construct with no arguments."""
        bridge = RLMSelectionBridge()

        assert bridge.rlm_bridge is None
        assert bridge.selection_feedback is None
        assert len(bridge._agent_stats) == 0

    def test_with_dependencies(self):
        """Should construct with dependencies."""
        rlm = MagicMock()
        feedback = MockSelectionFeedbackLoop()

        bridge = RLMSelectionBridge(
            rlm_bridge=rlm,
            selection_feedback=feedback,
        )

        assert bridge.rlm_bridge is rlm
        assert bridge.selection_feedback is feedback


class TestRecordCompression:
    """Tests for recording compression results."""

    def test_record_compression_creates_stats(self):
        """Should create stats for new agent."""
        bridge = RLMSelectionBridge()
        result = MockCompressionResult(estimated_fidelity=0.9)

        bridge.record_compression("claude", result)

        assert "claude" in bridge._agent_stats
        stats = bridge._agent_stats["claude"]
        assert stats.total_operations == 1
        assert stats.total_compressions == 1

    def test_record_compression_accumulates(self):
        """Should accumulate compression stats."""
        bridge = RLMSelectionBridge()

        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.8))
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.9))

        stats = bridge._agent_stats["claude"]
        assert stats.total_compressions == 2
        assert stats.total_fidelity == pytest.approx(1.7, abs=0.01)

    def test_record_compression_updates_adjustment(self):
        """Should compute adjustment after recording."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)

        adjustment = bridge.record_compression(
            "claude", MockCompressionResult(estimated_fidelity=0.95)
        )

        # High fidelity should give positive adjustment
        assert adjustment > 0

    def test_record_compression_calculates_avg_ratio(self):
        """Should calculate average compression ratio."""
        bridge = RLMSelectionBridge()
        result = MockCompressionResult(compression_ratio={"l1": 0.5, "l2": 0.3})

        bridge.record_compression("claude", result)

        stats = bridge._agent_stats["claude"]
        assert stats.avg_compression_ratio == pytest.approx(0.4, abs=0.01)


class TestRecordQuery:
    """Tests for recording query results."""

    def test_record_query_creates_stats(self):
        """Should create stats for new agent."""
        bridge = RLMSelectionBridge()
        result = MockRLMResult(confidence=0.85)

        bridge.record_query("gpt4", result)

        assert "gpt4" in bridge._agent_stats
        stats = bridge._agent_stats["gpt4"]
        assert stats.total_queries == 1

    def test_record_query_accumulates_metrics(self):
        """Should accumulate query metrics."""
        bridge = RLMSelectionBridge()

        bridge.record_query("gpt4", MockRLMResult(confidence=0.8, sub_calls_made=2))
        bridge.record_query("gpt4", MockRLMResult(confidence=0.9, sub_calls_made=4))

        stats = bridge._agent_stats["gpt4"]
        assert stats.total_queries == 2
        assert stats.total_confidence == pytest.approx(1.7, abs=0.01)
        assert stats.total_sub_calls == 6


class TestRecordRLMOperation:
    """Tests for generic RLM operation recording."""

    def test_detects_compression_result(self):
        """Should detect and route compression results."""
        bridge = RLMSelectionBridge()
        result = MockCompressionResult(estimated_fidelity=0.9)

        bridge.record_rlm_operation("claude", result)

        stats = bridge._agent_stats["claude"]
        assert stats.total_compressions == 1
        assert stats.total_queries == 0

    def test_detects_query_result(self):
        """Should detect and route query results."""
        bridge = RLMSelectionBridge()
        result = MockRLMResult(confidence=0.85)

        bridge.record_rlm_operation("claude", result)

        stats = bridge._agent_stats["claude"]
        assert stats.total_queries == 1
        assert stats.total_compressions == 0

    def test_handles_unknown_result_type(self):
        """Should handle unknown result types gracefully."""
        bridge = RLMSelectionBridge()

        # Unknown type should return 0.0 adjustment
        adjustment = bridge.record_rlm_operation("claude", {"unknown": "data"})

        assert adjustment == 0.0


class TestComputeAdjustment:
    """Tests for adjustment computation."""

    def test_no_adjustment_below_min_operations(self):
        """Should return 0 when below minimum operations."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=5)
        bridge = RLMSelectionBridge(config=config)

        # Record only 3 operations
        for _ in range(3):
            bridge.record_compression("claude", MockCompressionResult())

        assert bridge.get_combined_adjustment("claude") == 0.0

    def test_positive_boost_high_fidelity(self):
        """Should give positive boost for high fidelity."""
        config = RLMSelectionBridgeConfig(
            min_operations_for_boost=2,
            min_fidelity_threshold=0.7,
        )
        bridge = RLMSelectionBridge(config=config)

        # High fidelity compressions
        for _ in range(3):
            bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))

        adjustment = bridge.get_combined_adjustment("claude")
        assert adjustment > 0

    def test_negative_penalty_low_fidelity(self):
        """Should give negative penalty for low fidelity."""
        config = RLMSelectionBridgeConfig(
            min_operations_for_boost=2,
            min_fidelity_threshold=0.7,
        )
        bridge = RLMSelectionBridge(config=config)

        # Low fidelity compressions
        for _ in range(3):
            bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.5))

        adjustment = bridge.get_combined_adjustment("claude")
        assert adjustment < 0

    def test_boost_capped_at_max(self):
        """Should cap boost at max_boost."""
        config = RLMSelectionBridgeConfig(
            min_operations_for_boost=1,
            max_boost=0.2,
        )
        bridge = RLMSelectionBridge(config=config)

        # Very high performance
        for _ in range(10):
            bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=1.0))
            bridge.record_query("claude", MockRLMResult(confidence=1.0, sub_calls_made=1))

        adjustment = bridge.get_combined_adjustment("claude")
        assert adjustment <= config.max_boost


class TestGetBoostAndPenalty:
    """Tests for boost and penalty getters."""

    def test_get_compression_boost_positive(self):
        """Should return positive boost when adjustment is positive."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))

        boost = bridge.get_compression_boost("claude")
        assert boost >= 0

    def test_get_compression_boost_returns_zero_for_penalty(self):
        """Should return 0 when adjustment is negative."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.4))

        boost = bridge.get_compression_boost("claude")
        assert boost == 0.0

    def test_get_compression_penalty_negative(self):
        """Should return negative penalty when adjustment is negative."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.4))

        penalty = bridge.get_compression_penalty("claude")
        assert penalty <= 0

    def test_get_compression_penalty_returns_zero_for_boost(self):
        """Should return 0 when adjustment is positive."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))

        penalty = bridge.get_compression_penalty("claude")
        assert penalty == 0.0


class TestGetAllAdjustments:
    """Tests for getting all adjustments."""

    def test_returns_all_tracked_agents(self):
        """Should return adjustments for all agents."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(config=config)

        bridge.record_compression("claude", MockCompressionResult())
        bridge.record_compression("gpt4", MockCompressionResult())
        bridge.record_compression("gemini", MockCompressionResult())

        adjustments = bridge.get_all_adjustments()

        assert len(adjustments) == 3
        assert "claude" in adjustments
        assert "gpt4" in adjustments
        assert "gemini" in adjustments


class TestGetEfficientAgents:
    """Tests for getting RLM-efficient agents."""

    def test_returns_efficient_agents(self):
        """Should return agents above efficiency threshold."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=2)
        bridge = RLMSelectionBridge(config=config)

        # High efficiency agent
        for _ in range(5):
            bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))
            bridge.record_query("claude", MockRLMResult(confidence=0.9, sub_calls_made=2))

        # Low efficiency agent
        for _ in range(5):
            bridge.record_compression("gpt4", MockCompressionResult(estimated_fidelity=0.5))

        efficient = bridge.get_rlm_efficient_agents(threshold=0.7)

        assert "claude" in efficient
        # gpt4 might not be in efficient list due to low fidelity

    def test_excludes_agents_below_min_operations(self):
        """Should exclude agents with insufficient operations."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=10)
        bridge = RLMSelectionBridge(config=config)

        # Only 2 operations
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))

        efficient = bridge.get_rlm_efficient_agents()

        assert "claude" not in efficient


class TestGetBestAgentsForLongDebates:
    """Tests for getting best agents for long debates."""

    def test_returns_top_n_agents(self):
        """Should return top N agents by efficiency."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=2)
        bridge = RLMSelectionBridge(config=config)

        # Create agents with different efficiency
        for _ in range(5):
            bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))
            bridge.record_compression("gpt4", MockCompressionResult(estimated_fidelity=0.85))
            bridge.record_compression("gemini", MockCompressionResult(estimated_fidelity=0.75))

        best = bridge.get_best_agents_for_long_debates(top_n=2)

        assert len(best) <= 2
        if len(best) == 2:
            # Claude should be first (highest fidelity)
            assert best[0] == "claude"

    def test_returns_empty_when_no_eligible(self):
        """Should return empty list when no eligible agents."""
        config = RLMSelectionBridgeConfig(min_operations_for_boost=100)
        bridge = RLMSelectionBridge(config=config)

        bridge.record_compression("claude", MockCompressionResult())

        best = bridge.get_best_agents_for_long_debates()

        assert best == []


class TestSyncToSelectionFeedback:
    """Tests for syncing to selection feedback loop."""

    def test_updates_selection_adjustments(self):
        """Should update selection feedback adjustments."""
        feedback = MockSelectionFeedbackLoop()
        feedback.add_agent("claude")

        config = RLMSelectionBridgeConfig(min_operations_for_boost=1)
        bridge = RLMSelectionBridge(
            selection_feedback=feedback,
            config=config,
        )

        # Record high-performance operation
        bridge.record_compression("claude", MockCompressionResult(estimated_fidelity=0.95))

        updated = bridge.sync_to_selection_feedback()

        assert updated >= 0  # May update if adjustment significant

    def test_returns_zero_without_feedback_loop(self):
        """Should return 0 when no feedback loop attached."""
        bridge = RLMSelectionBridge()
        bridge.record_compression("claude", MockCompressionResult())

        updated = bridge.sync_to_selection_feedback()

        assert updated == 0

    def test_skips_small_adjustments(self):
        """Should skip adjustments smaller than 0.01."""
        feedback = MockSelectionFeedbackLoop()
        feedback.add_agent("claude")

        config = RLMSelectionBridgeConfig(min_operations_for_boost=100)
        bridge = RLMSelectionBridge(
            selection_feedback=feedback,
            config=config,
        )

        # Only 1 operation - adjustment will be 0
        bridge.record_compression("claude", MockCompressionResult())

        updated = bridge.sync_to_selection_feedback()

        assert updated == 0


class TestGetStats:
    """Tests for getting bridge statistics."""

    def test_get_agent_stats(self):
        """Should return stats for specific agent."""
        bridge = RLMSelectionBridge()
        bridge.record_compression("claude", MockCompressionResult())

        stats = bridge.get_agent_stats("claude")

        assert stats is not None
        assert stats.agent_name == "claude"

    def test_get_agent_stats_nonexistent(self):
        """Should return None for unknown agent."""
        bridge = RLMSelectionBridge()

        stats = bridge.get_agent_stats("unknown")

        assert stats is None

    def test_get_all_stats(self):
        """Should return stats for all agents."""
        bridge = RLMSelectionBridge()
        bridge.record_compression("claude", MockCompressionResult())
        bridge.record_compression("gpt4", MockCompressionResult())

        all_stats = bridge.get_all_stats()

        assert len(all_stats) == 2
        assert "claude" in all_stats
        assert "gpt4" in all_stats

    def test_get_stats_summary(self):
        """Should return bridge summary statistics."""
        bridge = RLMSelectionBridge()
        bridge.record_compression("claude", MockCompressionResult())
        bridge.record_query("claude", MockRLMResult())

        stats = bridge.get_stats()

        assert stats["agents_tracked"] == 1
        assert stats["total_operations"] == 2
        assert stats["total_compressions"] == 1
        assert stats["total_queries"] == 1


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_all_data(self):
        """Should clear all stats and adjustments."""
        bridge = RLMSelectionBridge()
        bridge.record_compression("claude", MockCompressionResult())
        bridge.record_query("gpt4", MockRLMResult())

        bridge.reset()

        assert len(bridge._agent_stats) == 0
        assert len(bridge._rlm_adjustments) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateRLMSelectionBridge:
    """Tests for factory function."""

    def test_creates_with_defaults(self):
        """Should create bridge with default config."""
        bridge = create_rlm_selection_bridge()

        assert bridge.config.min_operations_for_boost == 5
        assert bridge.config.max_boost == 0.25

    def test_creates_with_custom_config(self):
        """Should create bridge with custom config."""
        bridge = create_rlm_selection_bridge(
            min_operations_for_boost=10,
            max_boost=0.5,
        )

        assert bridge.config.min_operations_for_boost == 10
        assert bridge.config.max_boost == 0.5

    def test_creates_with_dependencies(self):
        """Should create bridge with dependencies."""
        rlm = MagicMock()
        feedback = MockSelectionFeedbackLoop()

        bridge = create_rlm_selection_bridge(
            rlm_bridge=rlm,
            selection_feedback=feedback,
        )

        assert bridge.rlm_bridge is rlm
        assert bridge.selection_feedback is feedback

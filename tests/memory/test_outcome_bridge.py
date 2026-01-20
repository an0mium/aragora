"""Tests for OutcomeMemoryBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import Mock, MagicMock

from aragora.memory.outcome_bridge import (
    OutcomeMemoryBridge,
    MemoryUsageRecord,
    PromotionResult,
    ProcessingResult,
    create_outcome_bridge,
)


@dataclass
class MockMemoryEntry:
    """Mock ContinuumMemoryEntry for testing."""

    id: str
    tier: str = "medium"
    content: str = "test content"
    importance: float = 0.5
    surprise_score: float = 0.5
    consolidation_score: float = 0.5
    update_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = "2026-01-01T00:00:00"
    updated_at: str = "2026-01-01T00:00:00"
    metadata: Dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class MockConsensusOutcome:
    """Mock ConsensusOutcome for testing."""

    debate_id: str
    consensus_text: str = "Test consensus"
    consensus_confidence: float = 0.8
    implementation_attempted: bool = True
    implementation_succeeded: bool = True


class MockMemoryTier:
    """Mock MemoryTier enum."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"

    def __init__(self, value: str):
        self.value = value


class TestMemoryUsageTracking:
    """Test memory usage tracking during debates."""

    def test_record_memory_usage(self) -> None:
        """Can record that a memory was used in a debate."""
        bridge = OutcomeMemoryBridge()
        bridge.record_memory_usage("mem-1", "debate-1")

        memories = bridge.get_memories_for_debate("debate-1")
        assert "mem-1" in memories

    def test_record_multiple_memories(self) -> None:
        """Can record multiple memories for same debate."""
        bridge = OutcomeMemoryBridge()
        bridge.record_memory_usage("mem-1", "debate-1")
        bridge.record_memory_usage("mem-2", "debate-1")
        bridge.record_memory_usage("mem-3", "debate-1")

        memories = bridge.get_memories_for_debate("debate-1")
        assert len(memories) == 3
        assert "mem-1" in memories
        assert "mem-2" in memories
        assert "mem-3" in memories

    def test_no_duplicate_usage_records(self) -> None:
        """Duplicate recordings are ignored."""
        bridge = OutcomeMemoryBridge()
        bridge.record_memory_usage("mem-1", "debate-1")
        bridge.record_memory_usage("mem-1", "debate-1")  # Duplicate

        memories = bridge.get_memories_for_debate("debate-1")
        assert len(memories) == 1

    def test_separate_debates_tracked_separately(self) -> None:
        """Different debates have separate memory tracking."""
        bridge = OutcomeMemoryBridge()
        bridge.record_memory_usage("mem-1", "debate-1")
        bridge.record_memory_usage("mem-2", "debate-2")

        assert bridge.get_memories_for_debate("debate-1") == ["mem-1"]
        assert bridge.get_memories_for_debate("debate-2") == ["mem-2"]


class TestOutcomeProcessing:
    """Test outcome processing."""

    def test_process_outcome_no_memories(self) -> None:
        """Processing with no tracked memories returns empty result."""
        bridge = OutcomeMemoryBridge()
        outcome = MockConsensusOutcome(debate_id="debate-1")

        result = bridge.process_outcome(outcome)

        assert result.debate_id == "debate-1"
        assert result.memories_updated == 0

    def test_process_outcome_explicit_memory_ids(self) -> None:
        """Can process with explicit memory IDs."""
        mock_memory = MockMemoryEntry(id="mem-1")
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(continuum_memory=mock_continuum)
        outcome = MockConsensusOutcome(debate_id="debate-1")

        result = bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert result.memories_updated == 1
        mock_continuum.update_entry.assert_called_once()

    def test_process_outcome_skips_low_confidence(self) -> None:
        """Low confidence debates don't update memories."""
        mock_memory = MockMemoryEntry(id="mem-1")
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory

        bridge = OutcomeMemoryBridge(
            continuum_memory=mock_continuum,
            min_confidence_for_update=0.6,
        )
        outcome = MockConsensusOutcome(debate_id="debate-1", consensus_confidence=0.4)

        result = bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert result.memories_updated == 0
        mock_continuum.get_entry.assert_not_called()

    def test_process_successful_outcome_increases_success_count(self) -> None:
        """Successful outcome increases memory success count."""
        mock_memory = MockMemoryEntry(id="mem-1", success_count=0)
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(continuum_memory=mock_continuum)
        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=True,
        )

        bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert mock_memory.success_count == 1
        assert mock_memory.failure_count == 0

    def test_process_failed_outcome_increases_failure_count(self) -> None:
        """Failed outcome increases memory failure count."""
        mock_memory = MockMemoryEntry(id="mem-1", failure_count=0)
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(continuum_memory=mock_continuum)
        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=False,
        )

        bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert mock_memory.failure_count == 1
        assert mock_memory.success_count == 0


class TestImportanceAdjustment:
    """Test importance adjustments based on outcomes."""

    def test_success_boosts_importance(self) -> None:
        """Successful outcome boosts memory importance."""
        mock_memory = MockMemoryEntry(id="mem-1", importance=0.5)
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(
            continuum_memory=mock_continuum,
            success_boost_weight=0.1,
        )
        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=True,
        )

        bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert mock_memory.importance == 0.6

    def test_failure_reduces_importance(self) -> None:
        """Failed outcome reduces memory importance."""
        mock_memory = MockMemoryEntry(id="mem-1", importance=0.5)
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(
            continuum_memory=mock_continuum,
            failure_penalty_weight=0.1,
        )
        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=False,
        )

        bridge.process_outcome(outcome, used_memory_ids=["mem-1"])

        assert mock_memory.importance == 0.4

    def test_importance_clamped_at_bounds(self) -> None:
        """Importance stays within 0-1 bounds."""
        # Test upper bound
        mock_memory_high = MockMemoryEntry(id="mem-1", importance=0.98)
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory_high
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(
            continuum_memory=mock_continuum,
            success_boost_weight=0.1,
        )
        outcome = MockConsensusOutcome(debate_id="debate-1", implementation_succeeded=True)

        bridge.process_outcome(outcome, used_memory_ids=["mem-1"])
        assert mock_memory_high.importance <= 1.0

        # Test lower bound
        mock_memory_low = MockMemoryEntry(id="mem-2", importance=0.02)
        mock_continuum.get_entry.return_value = mock_memory_low

        outcome_fail = MockConsensusOutcome(
            debate_id="debate-2",
            implementation_succeeded=False,
        )
        bridge.process_outcome(outcome_fail, used_memory_ids=["mem-2"])
        assert mock_memory_low.importance >= 0.0


class TestMemoryStats:
    """Test memory statistics tracking."""

    def test_get_memory_stats(self) -> None:
        """Can get stats for a memory."""
        bridge = OutcomeMemoryBridge()

        # Simulate some outcomes
        bridge._memory_success_counts["mem-1"] = 5
        bridge._memory_failure_counts["mem-1"] = 2

        stats = bridge.get_memory_stats("mem-1")

        assert stats["memory_id"] == "mem-1"
        assert stats["success_count"] == 5
        assert stats["failure_count"] == 2
        assert stats["total_uses"] == 7
        assert abs(stats["success_rate"] - (5 / 7)) < 0.001

    def test_get_memory_stats_empty(self) -> None:
        """Stats for unknown memory returns zeros."""
        bridge = OutcomeMemoryBridge()
        stats = bridge.get_memory_stats("unknown")

        assert stats["success_count"] == 0
        assert stats["failure_count"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_top_memories(self) -> None:
        """Can get top memories by success rate."""
        bridge = OutcomeMemoryBridge()

        # Simulate outcomes
        bridge._memory_success_counts["good"] = 8
        bridge._memory_failure_counts["good"] = 2
        bridge._memory_success_counts["bad"] = 2
        bridge._memory_failure_counts["bad"] = 8
        bridge._memory_success_counts["medium"] = 5
        bridge._memory_failure_counts["medium"] = 5

        top = bridge.get_top_memories(limit=2)

        assert len(top) == 2
        assert top[0]["memory_id"] == "good"
        assert top[0]["success_rate"] == 0.8


class TestCleanup:
    """Test cleanup functionality."""

    def test_clear_tracking_data(self) -> None:
        """clear_tracking_data removes all tracked data."""
        bridge = OutcomeMemoryBridge()

        # Add some data
        bridge.record_memory_usage("mem-1", "debate-1")
        bridge._memory_success_counts["mem-1"] = 5
        bridge._memory_failure_counts["mem-1"] = 2

        bridge.clear_tracking_data()

        assert len(bridge._usage_records) == 0
        assert len(bridge._memory_success_counts) == 0
        assert len(bridge._memory_failure_counts) == 0

    def test_usage_records_cleared_after_processing(self) -> None:
        """Usage records are cleared after processing an outcome."""
        mock_memory = MockMemoryEntry(id="mem-1")
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(continuum_memory=mock_continuum)
        bridge.record_memory_usage("mem-1", "debate-1")

        outcome = MockConsensusOutcome(debate_id="debate-1")
        bridge.process_outcome(outcome)

        # Usage records for this debate should be cleared
        assert bridge.get_memories_for_debate("debate-1") == []


class TestPromotionLogic:
    """Test tier promotion logic."""

    def test_promotion_after_threshold_successes(self) -> None:
        """Memory promoted after reaching success threshold."""
        from aragora.memory.tier_manager import MemoryTier

        mock_memory = MockMemoryEntry(
            id="mem-1",
            tier=MemoryTier.MEDIUM,
            success_count=0,
        )
        mock_continuum = Mock()
        mock_continuum.get_entry.return_value = mock_memory
        mock_continuum.update_entry = Mock()
        mock_continuum.promote_entry = Mock()
        mock_continuum.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(
            continuum_memory=mock_continuum,
            usage_count_threshold=3,
            success_threshold=0.7,
        )

        # Simulate 3 successful outcomes
        for i in range(3):
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=True,
                consensus_confidence=0.9,
            )
            bridge.record_memory_usage("mem-1", f"debate-{i}")
            mock_memory.success_count = i + 1  # Simulate increment
            bridge.process_outcome(outcome)

        # Should have attempted promotion
        assert mock_continuum.promote_entry.called


class TestCreateHelper:
    """Test create_outcome_bridge helper."""

    def test_creates_with_defaults(self) -> None:
        """Creates bridge with default values."""
        bridge = create_outcome_bridge()
        assert isinstance(bridge, OutcomeMemoryBridge)

    def test_creates_with_custom_config(self) -> None:
        """Creates bridge with custom configuration."""
        bridge = create_outcome_bridge(
            success_threshold=0.9,
            usage_count_threshold=5,
        )
        assert bridge.success_threshold == 0.9
        assert bridge.usage_count_threshold == 5

    def test_creates_with_dependencies(self) -> None:
        """Creates bridge with dependencies."""
        mock_tracker = Mock()
        mock_memory = Mock()

        bridge = create_outcome_bridge(
            outcome_tracker=mock_tracker,
            continuum_memory=mock_memory,
        )

        assert bridge.outcome_tracker is mock_tracker
        assert bridge.continuum_memory is mock_memory

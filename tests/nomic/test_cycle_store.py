"""Tests for Nomic cross-cycle learning persistence."""

import os
import tempfile
import time

import pytest

from aragora.nomic.cycle_record import (
    AgentContribution,
    NomicCycleRecord,
    PatternReinforcement,
    SurpriseEvent,
)
from aragora.nomic.cycle_store import CycleLearningStore


class TestNomicCycleRecord:
    """Tests for NomicCycleRecord dataclass."""

    def test_basic_creation(self):
        """Should create record with required fields."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )

        assert record.cycle_id == "test-123"
        assert record.started_at > 0
        assert record.completed_at is None
        assert record.success is False

    def test_mark_complete(self):
        """mark_complete should set timing and success."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time() - 10,
        )
        record.mark_complete(success=True)

        assert record.completed_at is not None
        assert record.success is True
        assert record.duration_seconds > 0

    def test_mark_complete_with_error(self):
        """mark_complete with error should record failure."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )
        record.mark_complete(success=False, error="Test error")

        assert record.success is False
        assert record.error_message == "Test error"

    def test_add_agent_contribution(self):
        """Should track agent contributions."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )
        record.add_agent_contribution(
            agent_name="claude",
            proposals_made=5,
            proposals_accepted=3,
        )

        assert "claude" in record.agent_contributions
        contrib = record.agent_contributions["claude"]
        assert contrib.proposals_made == 5
        assert contrib.proposals_accepted == 3

    def test_add_agent_contribution_accumulates(self):
        """Multiple calls should accumulate contributions."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )
        record.add_agent_contribution("claude", proposals_made=2)
        record.add_agent_contribution("claude", proposals_made=3)

        assert record.agent_contributions["claude"].proposals_made == 5

    def test_add_surprise(self):
        """Should record surprise events."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )
        record.add_surprise(
            phase="design",
            description="Unexpected complexity",
            expected="Simple refactor",
            actual="Major rewrite needed",
            impact="high",
        )

        assert len(record.surprise_events) == 1
        assert record.surprise_events[0].phase == "design"
        assert record.surprise_events[0].impact == "high"

    def test_add_pattern_reinforcement(self):
        """Should record pattern reinforcements."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
        )
        record.add_pattern_reinforcement(
            pattern_type="bugfix",
            description="Incremental fix worked",
            success=True,
            confidence=0.9,
        )

        assert len(record.pattern_reinforcements) == 1
        assert record.pattern_reinforcements[0].success is True

    def test_to_dict_and_back(self):
        """Should serialize and deserialize correctly."""
        record = NomicCycleRecord(
            cycle_id="test-123",
            started_at=time.time(),
            topics_debated=["Add feature", "Fix bug"],
        )
        record.add_agent_contribution("claude", proposals_made=3)
        record.add_surprise("test", "Test", "A", "B")
        record.add_pattern_reinforcement("refactor", "Worked", True)
        record.mark_complete(success=True)

        data = record.to_dict()
        restored = NomicCycleRecord.from_dict(data)

        assert restored.cycle_id == record.cycle_id
        assert restored.topics_debated == record.topics_debated
        assert "claude" in restored.agent_contributions
        assert len(restored.surprise_events) == 1
        assert len(restored.pattern_reinforcements) == 1
        assert restored.success is True


class TestAgentContribution:
    """Tests for AgentContribution dataclass."""

    def test_creation(self):
        """Should create with all fields."""
        contrib = AgentContribution(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=5,
            critiques_valuable=4,
            quality_score=0.85,
        )

        assert contrib.agent_name == "claude"
        assert contrib.proposals_made == 10
        assert contrib.quality_score == 0.85


class TestSurpriseEvent:
    """Tests for SurpriseEvent dataclass."""

    def test_creation(self):
        """Should create with all fields."""
        event = SurpriseEvent(
            phase="implement",
            description="Tests failed unexpectedly",
            expected="All tests pass",
            actual="3 tests failed",
            impact="medium",
        )

        assert event.phase == "implement"
        assert event.impact == "medium"


class TestCycleLearningStore:
    """Tests for CycleLearningStore persistence."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        store = CycleLearningStore(db_path=db_path)
        yield store

        # Cleanup
        try:
            os.unlink(db_path)
        except OSError:
            pass

    def test_save_and_load_cycle(self, temp_store):
        """Should save and load a cycle record."""
        record = NomicCycleRecord(
            cycle_id="test-abc",
            started_at=time.time(),
            topics_debated=["Test topic"],
        )
        record.mark_complete(success=True)

        temp_store.save_cycle(record)
        loaded = temp_store.load_cycle("test-abc")

        assert loaded is not None
        assert loaded.cycle_id == "test-abc"
        assert loaded.success is True

    def test_load_nonexistent(self, temp_store):
        """Loading nonexistent cycle should return None."""
        loaded = temp_store.load_cycle("nonexistent")
        assert loaded is None

    def test_get_recent_cycles(self, temp_store):
        """Should return cycles in reverse chronological order."""
        now = time.time()

        for i in range(5):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=now + i,
            )
            temp_store.save_cycle(record)

        recent = temp_store.get_recent_cycles(3)

        assert len(recent) == 3
        assert recent[0].cycle_id == "cycle-4"  # Most recent
        assert recent[2].cycle_id == "cycle-2"

    def test_get_successful_cycles(self, temp_store):
        """Should filter to only successful cycles."""
        for i in range(5):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            record.mark_complete(success=(i % 2 == 0))  # Alternate success
            temp_store.save_cycle(record)

        successful = temp_store.get_successful_cycles(10)

        assert len(successful) == 3  # 0, 2, 4
        assert all(r.success for r in successful)

    def test_query_by_topic(self, temp_store):
        """Should find cycles matching topic."""
        record1 = NomicCycleRecord(
            cycle_id="auth-cycle",
            started_at=time.time(),
            topics_debated=["Fix authentication bug"],
        )
        record2 = NomicCycleRecord(
            cycle_id="ui-cycle",
            started_at=time.time() + 1,
            topics_debated=["Update dashboard UI"],
        )
        temp_store.save_cycle(record1)
        temp_store.save_cycle(record2)

        auth_results = temp_store.query_by_topic("auth")
        assert len(auth_results) == 1
        assert auth_results[0].cycle_id == "auth-cycle"

    def test_get_agent_trajectory(self, temp_store):
        """Should return agent performance over time."""
        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            record.add_agent_contribution(
                "claude",
                proposals_made=5,
                proposals_accepted=3 + i,
            )
            record.mark_complete(success=True)
            temp_store.save_cycle(record)

        trajectory = temp_store.get_agent_trajectory("claude")

        assert len(trajectory) == 3
        # Acceptance rate should improve over cycles
        assert trajectory[0]["proposals_accepted"] == 5  # Most recent
        assert trajectory[2]["proposals_accepted"] == 3

    def test_get_pattern_statistics(self, temp_store):
        """Should aggregate pattern statistics."""
        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            record.add_pattern_reinforcement(
                "bugfix", f"Fix {i}", success=(i != 1)
            )
            temp_store.save_cycle(record)

        stats = temp_store.get_pattern_statistics()

        assert "bugfix" in stats
        assert stats["bugfix"]["success_count"] == 2
        assert stats["bugfix"]["failure_count"] == 1

    def test_cleanup_old_cycles(self, temp_store):
        """Should delete old cycles while keeping recent ones."""
        for i in range(10):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            temp_store.save_cycle(record)

        deleted = temp_store.cleanup_old_cycles(keep_count=5)

        assert deleted == 5
        assert temp_store.get_cycle_count() == 5

    def test_get_cycle_count(self, temp_store):
        """Should return correct count."""
        assert temp_store.get_cycle_count() == 0

        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time(),
            )
            temp_store.save_cycle(record)

        assert temp_store.get_cycle_count() == 3

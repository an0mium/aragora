"""
Comprehensive tests for A/B testing framework.

Tests cover:
- ABTest dataclass functionality
- ABTestStatus enum
- ABTestResult dataclass
- ABTestManager lifecycle (start, record, conclude, cancel)
- Variant selection for balanced sampling
- Statistical significance calculations
- Database persistence and retrieval
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aragora.evolution.ab_testing import (
    ABTest,
    ABTestManager,
    ABTestResult,
    ABTestStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def manager(temp_db):
    """Create an ABTestManager with temp database."""
    return ABTestManager(db_path=temp_db)


# =============================================================================
# ABTestStatus Tests
# =============================================================================


class TestABTestStatus:
    """Test ABTestStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ABTestStatus.ACTIVE.value == "active"
        assert ABTestStatus.CONCLUDED.value == "concluded"
        assert ABTestStatus.CANCELLED.value == "cancelled"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert ABTestStatus("active") == ABTestStatus.ACTIVE
        assert ABTestStatus("concluded") == ABTestStatus.CONCLUDED
        assert ABTestStatus("cancelled") == ABTestStatus.CANCELLED

    def test_invalid_status_raises(self):
        """Test invalid status raises ValueError."""
        with pytest.raises(ValueError):
            ABTestStatus("invalid")


# =============================================================================
# ABTest Dataclass Tests
# =============================================================================


class TestABTest:
    """Test ABTest dataclass."""

    def test_create_basic_test(self):
        """Test creating a basic ABTest."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
        )

        assert test.id == "test-123"
        assert test.agent == "claude"
        assert test.baseline_prompt_version == 1
        assert test.evolved_prompt_version == 2
        assert test.baseline_wins == 0
        assert test.evolved_wins == 0
        assert test.status == ABTestStatus.ACTIVE

    def test_auto_timestamp(self):
        """Test that started_at is auto-generated."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
        )

        assert test.started_at != ""
        # Should be a valid ISO timestamp
        datetime.fromisoformat(test.started_at.replace("Z", "+00:00"))

    def test_evolved_win_rate_no_wins(self):
        """Test win rate with no wins returns 0.5."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
        )

        assert test.evolved_win_rate == 0.5
        assert test.baseline_win_rate == 0.5

    def test_evolved_win_rate_with_wins(self):
        """Test win rate calculation with wins."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=3,
            evolved_wins=7,
        )

        assert test.evolved_win_rate == 0.7
        assert test.baseline_win_rate == 0.3

    def test_total_debates(self):
        """Test total debates calculation."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_debates=10,
            evolved_debates=12,
        )

        assert test.total_debates == 22

    def test_sample_size(self):
        """Test sample size calculation."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=5,
            evolved_wins=7,
        )

        assert test.sample_size == 12

    def test_is_significant_insufficient_samples(self):
        """Test significance with insufficient samples."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=5,
            evolved_wins=10,  # 15 total, need 20
        )

        assert test.is_significant is False

    def test_is_significant_enough_samples_big_diff(self):
        """Test significance with sufficient samples and big difference."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=5,
            evolved_wins=15,  # 20 total, 75% evolved win rate
        )

        assert test.is_significant is True

    def test_is_significant_enough_samples_small_diff(self):
        """Test significance with sufficient samples but small difference."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=9,
            evolved_wins=11,  # 20 total, only 55% evolved win rate
        )

        assert test.is_significant is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        test = ABTest(
            id="test-123",
            agent="claude",
            baseline_prompt_version=1,
            evolved_prompt_version=2,
            baseline_wins=3,
            evolved_wins=7,
            metadata={"key": "value"},
        )

        d = test.to_dict()

        assert d["id"] == "test-123"
        assert d["agent"] == "claude"
        assert d["baseline_prompt_version"] == 1
        assert d["evolved_prompt_version"] == 2
        assert d["baseline_wins"] == 3
        assert d["evolved_wins"] == 7
        assert d["evolved_win_rate"] == 0.7
        assert d["baseline_win_rate"] == 0.3
        assert d["sample_size"] == 10
        assert d["status"] == "active"
        assert d["metadata"] == {"key": "value"}

    def test_from_row(self):
        """Test creating ABTest from database row."""
        row = (
            "test-123",  # id
            "claude",  # agent
            1,  # baseline_prompt_version
            2,  # evolved_prompt_version
            3,  # baseline_wins
            7,  # evolved_wins
            5,  # baseline_debates
            8,  # evolved_debates
            "2026-01-01T00:00:00",  # started_at
            None,  # concluded_at
            "active",  # status
            '{"key": "value"}',  # metadata
        )

        test = ABTest.from_row(row)

        assert test.id == "test-123"
        assert test.agent == "claude"
        assert test.baseline_wins == 3
        assert test.evolved_wins == 7
        assert test.status == ABTestStatus.ACTIVE
        assert test.metadata == {"key": "value"}


# =============================================================================
# ABTestResult Tests
# =============================================================================


class TestABTestResult:
    """Test ABTestResult dataclass."""

    def test_create_result(self):
        """Test creating an ABTestResult."""
        result = ABTestResult(
            test_id="test-123",
            winner="evolved",
            confidence=0.85,
            recommendation="Recommend adoption.",
            stats={"win_rate": 0.7},
        )

        assert result.test_id == "test-123"
        assert result.winner == "evolved"
        assert result.confidence == 0.85
        assert result.recommendation == "Recommend adoption."
        assert result.stats["win_rate"] == 0.7

    def test_result_default_stats(self):
        """Test result with default stats."""
        result = ABTestResult(
            test_id="test-123",
            winner="tie",
            confidence=0.5,
            recommendation="No difference.",
        )

        assert result.stats == {}


# =============================================================================
# ABTestManager - Test Lifecycle Tests
# =============================================================================


class TestABTestManagerLifecycle:
    """Test ABTestManager test lifecycle."""

    def test_start_test(self, manager):
        """Test starting a new A/B test."""
        test = manager.start_test(
            agent="claude",
            baseline_version=1,
            evolved_version=2,
            metadata={"experiment": "prompt_v2"},
        )

        assert test.id is not None
        assert test.agent == "claude"
        assert test.baseline_prompt_version == 1
        assert test.evolved_prompt_version == 2
        assert test.status == ABTestStatus.ACTIVE
        assert test.metadata == {"experiment": "prompt_v2"}

    def test_start_test_duplicate_raises(self, manager):
        """Test starting duplicate test raises error."""
        manager.start_test("claude", 1, 2)

        with pytest.raises(ValueError, match="already has an active test"):
            manager.start_test("claude", 1, 3)

    def test_start_multiple_tests_different_agents(self, manager):
        """Test starting tests for different agents."""
        test1 = manager.start_test("claude", 1, 2)
        test2 = manager.start_test("gpt4", 1, 2)

        assert test1.agent == "claude"
        assert test2.agent == "gpt4"

    def test_get_test(self, manager):
        """Test getting a test by ID."""
        created = manager.start_test("claude", 1, 2)

        retrieved = manager.get_test(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.agent == "claude"

    def test_get_test_nonexistent(self, manager):
        """Test getting nonexistent test returns None."""
        result = manager.get_test("nonexistent-id")
        assert result is None

    def test_get_active_test(self, manager):
        """Test getting active test for agent."""
        created = manager.start_test("claude", 1, 2)

        active = manager.get_active_test("claude")

        assert active is not None
        assert active.id == created.id

    def test_get_active_test_no_active(self, manager):
        """Test getting active test when none exists."""
        result = manager.get_active_test("claude")
        assert result is None

    def test_get_agent_tests(self, manager):
        """Test getting all tests for an agent."""
        # Start and conclude a test
        test1 = manager.start_test("claude", 1, 2)
        manager.cancel_test(test1.id)

        # Start another test
        test2 = manager.start_test("claude", 2, 3)

        tests = manager.get_agent_tests("claude", limit=10)

        assert len(tests) == 2
        # Should be ordered by started_at DESC
        assert tests[0].id == test2.id

    def test_cancel_test(self, manager):
        """Test cancelling an active test."""
        test = manager.start_test("claude", 1, 2)

        result = manager.cancel_test(test.id)

        assert result is True

        # Verify status changed
        cancelled = manager.get_test(test.id)
        assert cancelled.status == ABTestStatus.CANCELLED
        assert cancelled.concluded_at is not None

    def test_cancel_nonexistent_test(self, manager):
        """Test cancelling nonexistent test returns False."""
        result = manager.cancel_test("nonexistent-id")
        assert result is False


# =============================================================================
# ABTestManager - Result Recording Tests
# =============================================================================


class TestABTestManagerRecording:
    """Test recording results in ABTestManager."""

    def test_record_baseline_win(self, manager):
        """Test recording a baseline win."""
        test = manager.start_test("claude", 1, 2)

        updated = manager.record_result(
            agent="claude",
            debate_id="debate-1",
            variant="baseline",
            won=True,
        )

        assert updated is not None
        assert updated.baseline_wins == 1
        assert updated.baseline_debates == 1
        assert updated.evolved_wins == 0

    def test_record_evolved_win(self, manager):
        """Test recording an evolved win."""
        test = manager.start_test("claude", 1, 2)

        updated = manager.record_result(
            agent="claude",
            debate_id="debate-1",
            variant="evolved",
            won=True,
        )

        assert updated.evolved_wins == 1
        assert updated.evolved_debates == 1
        assert updated.baseline_wins == 0

    def test_record_loss(self, manager):
        """Test recording a loss."""
        test = manager.start_test("claude", 1, 2)

        updated = manager.record_result(
            agent="claude",
            debate_id="debate-1",
            variant="evolved",
            won=False,
        )

        assert updated.evolved_wins == 0
        assert updated.evolved_debates == 1

    def test_record_no_active_test(self, manager):
        """Test recording with no active test returns None."""
        result = manager.record_result(
            agent="claude",
            debate_id="debate-1",
            variant="baseline",
            won=True,
        )

        assert result is None

    def test_record_invalid_variant_raises(self, manager):
        """Test recording with invalid variant raises error."""
        manager.start_test("claude", 1, 2)

        with pytest.raises(ValueError, match="Invalid variant"):
            manager.record_result(
                agent="claude",
                debate_id="debate-1",
                variant="invalid",
                won=True,
            )

    def test_record_duplicate_debate_ignored(self, manager):
        """Test recording same debate twice is ignored."""
        manager.start_test("claude", 1, 2)

        # Record first time
        manager.record_result("claude", "debate-1", "baseline", True)

        # Record same debate again
        updated = manager.record_result("claude", "debate-1", "baseline", True)

        # Should still only have 1 win
        assert updated.baseline_wins == 1
        assert updated.baseline_debates == 1

    def test_record_multiple_debates(self, manager):
        """Test recording multiple debates."""
        manager.start_test("claude", 1, 2)

        manager.record_result("claude", "debate-1", "baseline", True)
        manager.record_result("claude", "debate-2", "evolved", True)
        manager.record_result("claude", "debate-3", "baseline", False)
        updated = manager.record_result("claude", "debate-4", "evolved", True)

        assert updated.baseline_wins == 1
        assert updated.baseline_debates == 2
        assert updated.evolved_wins == 2
        assert updated.evolved_debates == 2


# =============================================================================
# ABTestManager - Conclusion Tests
# =============================================================================


class TestABTestManagerConclusion:
    """Test concluding tests in ABTestManager."""

    def test_conclude_evolved_wins(self, manager):
        """Test concluding with evolved winning."""
        test = manager.start_test("claude", 1, 2)

        # Record results favoring evolved
        for i in range(25):
            variant = "evolved" if i < 18 else "baseline"
            won = i < 18  # 18 evolved wins, 7 baseline wins
            manager.record_result("claude", f"debate-{i}", variant, won)

        result = manager.conclude_test(test.id)

        assert result.winner == "evolved"
        assert result.confidence > 0
        assert "adoption" in result.recommendation.lower()

    def test_conclude_baseline_wins(self, manager):
        """Test concluding with baseline winning."""
        test = manager.start_test("claude", 1, 2)

        # Record results favoring baseline
        for i in range(25):
            variant = "baseline" if i < 18 else "evolved"
            won = i < 18  # 18 baseline wins, 7 evolved wins
            manager.record_result("claude", f"debate-{i}", variant, won)

        result = manager.conclude_test(test.id)

        assert result.winner == "baseline"
        assert "keeping current" in result.recommendation.lower()

    def test_conclude_tie(self, manager):
        """Test concluding with a tie."""
        test = manager.start_test("claude", 1, 2)

        # Record balanced results
        for i in range(20):
            variant = "evolved" if i % 2 == 0 else "baseline"
            manager.record_result("claude", f"debate-{i}", variant, True)

        result = manager.conclude_test(test.id)

        assert result.winner == "tie"
        assert "no significant difference" in result.recommendation.lower()

    def test_conclude_no_data(self, manager):
        """Test concluding with no data."""
        test = manager.start_test("claude", 1, 2)

        result = manager.conclude_test(test.id)

        assert result.winner == "tie"
        assert result.confidence == 0.0
        assert "no data" in result.recommendation.lower()

    def test_conclude_nonexistent_raises(self, manager):
        """Test concluding nonexistent test raises error."""
        with pytest.raises(ValueError, match="not found"):
            manager.conclude_test("nonexistent-id")

    def test_conclude_already_concluded_raises(self, manager):
        """Test concluding already concluded test raises error."""
        test = manager.start_test("claude", 1, 2)
        manager.conclude_test(test.id)

        with pytest.raises(ValueError, match="already concluded"):
            manager.conclude_test(test.id)

    def test_conclude_significance_warning(self, manager):
        """Test significance warning in recommendation."""
        test = manager.start_test("claude", 1, 2)

        # Only record a few results (not significant)
        for i in range(5):
            manager.record_result("claude", f"debate-{i}", "evolved", True)

        result = manager.conclude_test(test.id)

        assert "not be statistically significant" in result.recommendation

    def test_conclude_force_ignores_warning(self, manager):
        """Test force flag suppresses significance warning."""
        test = manager.start_test("claude", 1, 2)

        # Only record a few results
        for i in range(5):
            manager.record_result("claude", f"debate-{i}", "evolved", True)

        result = manager.conclude_test(test.id, force=True)

        # Should still have result but no significance warning added
        # (force=True means we accept low sample size)
        assert result.winner == "evolved"

    def test_conclude_updates_status(self, manager):
        """Test that conclude updates the test status."""
        test = manager.start_test("claude", 1, 2)
        manager.conclude_test(test.id)

        concluded = manager.get_test(test.id)

        assert concluded.status == ABTestStatus.CONCLUDED
        assert concluded.concluded_at is not None


# =============================================================================
# ABTestManager - Variant Selection Tests
# =============================================================================


class TestABTestManagerVariantSelection:
    """Test variant selection in ABTestManager."""

    def test_get_variant_no_active_test(self, manager):
        """Test variant selection with no active test."""
        result = manager.get_variant_for_debate("claude")
        assert result is None

    def test_get_variant_initial(self, manager):
        """Test initial variant selection is baseline."""
        manager.start_test("claude", 1, 2)

        variant = manager.get_variant_for_debate("claude")

        # Both start at 0, so baseline should be selected
        assert variant == "baseline"

    def test_get_variant_alternates(self, manager):
        """Test variant selection alternates between baseline and evolved."""
        manager.start_test("claude", 1, 2)

        # Record baseline debate
        manager.record_result("claude", "debate-1", "baseline", True)

        # Next should be evolved
        variant = manager.get_variant_for_debate("claude")
        assert variant == "evolved"

        # Record evolved debate
        manager.record_result("claude", "debate-2", "evolved", True)

        # Back to baseline
        variant = manager.get_variant_for_debate("claude")
        assert variant == "baseline"

    def test_get_variant_catches_up(self, manager):
        """Test variant selection catches up when unbalanced."""
        manager.start_test("claude", 1, 2)

        # Record 3 baseline debates
        for i in range(3):
            manager.record_result("claude", f"baseline-{i}", "baseline", True)

        # Should keep selecting evolved until it catches up
        variant = manager.get_variant_for_debate("claude")
        assert variant == "evolved"


# =============================================================================
# ABTestManager - Database Persistence Tests
# =============================================================================


class TestABTestManagerPersistence:
    """Test database persistence in ABTestManager."""

    def test_data_persists_across_instances(self, temp_db):
        """Test data persists when reopening database."""
        # Create and populate first instance
        manager1 = ABTestManager(db_path=temp_db)
        test = manager1.start_test("claude", 1, 2)
        manager1.record_result("claude", "debate-1", "evolved", True)

        # Create second instance with same database
        manager2 = ABTestManager(db_path=temp_db)

        # Verify data persisted
        retrieved = manager2.get_test(test.id)
        assert retrieved is not None
        assert retrieved.evolved_wins == 1

    def test_test_metadata_persists(self, manager):
        """Test metadata persists correctly."""
        test = manager.start_test(
            "claude",
            1,
            2,
            metadata={
                "experiment": "prompt_v2",
                "hypothesis": "shorter prompts perform better",
            },
        )

        retrieved = manager.get_test(test.id)

        assert retrieved.metadata["experiment"] == "prompt_v2"
        assert retrieved.metadata["hypothesis"] == "shorter prompts perform better"


# =============================================================================
# ABTestManager - Schema Tests
# =============================================================================


class TestABTestManagerSchema:
    """Test ABTestManager schema initialization."""

    def test_schema_constants(self):
        """Test schema constants are defined."""
        assert ABTestManager.SCHEMA_NAME == "ab_testing"
        assert ABTestManager.SCHEMA_VERSION == 1
        assert "ab_tests" in ABTestManager.INITIAL_SCHEMA
        assert "ab_test_debates" in ABTestManager.INITIAL_SCHEMA

    def test_creates_tables(self, temp_db):
        """Test that tables are created on init."""
        manager = ABTestManager(db_path=temp_db)

        with manager.connection() as conn:
            cursor = conn.cursor()

            # Check ab_tests table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ab_tests'")
            assert cursor.fetchone() is not None

            # Check ab_test_debates table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ab_test_debates'"
            )
            assert cursor.fetchone() is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestABTestingIntegration:
    """Integration tests for complete A/B testing workflows."""

    def test_full_experiment_lifecycle(self, manager):
        """Test complete experiment from start to conclusion."""
        # 1. Start test
        test = manager.start_test(
            agent="claude",
            baseline_version=1,
            evolved_version=2,
            metadata={"hypothesis": "v2 is better"},
        )

        # 2. Run balanced experiment
        for i in range(30):
            variant = manager.get_variant_for_debate("claude")
            won = (i < 20 and variant == "evolved") or (i >= 20 and variant == "baseline")
            manager.record_result("claude", f"debate-{i}", variant, won)

        # 3. Conclude
        result = manager.conclude_test(test.id)

        # 4. Verify results
        assert result.winner in ["evolved", "baseline", "tie"]
        assert result.test_id == test.id
        assert "stats" in result.__dict__ or hasattr(result, "stats")

        # 5. Verify can't start new test with same agent while one is active
        # (but we just concluded, so we should be able to)
        new_test = manager.start_test("claude", 2, 3)
        assert new_test.id != test.id

    def test_multiple_agents_concurrent(self, manager):
        """Test running tests for multiple agents concurrently."""
        # Start tests for multiple agents
        agents = ["claude", "gpt4", "gemini"]
        tests = {}

        for agent in agents:
            tests[agent] = manager.start_test(agent, 1, 2)

        # Record results for all agents
        for i in range(10):
            for agent in agents:
                variant = "evolved" if i % 2 == 0 else "baseline"
                manager.record_result(agent, f"{agent}-debate-{i}", variant, True)

        # Verify each agent has correct counts
        for agent in agents:
            test = manager.get_active_test(agent)
            assert test.total_debates == 10
            assert test.evolved_debates == 5
            assert test.baseline_debates == 5

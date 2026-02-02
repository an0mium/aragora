"""Tests for PostgreSQL Cycle Learning Store."""

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.cycle_record import (
    AgentContribution,
    NomicCycleRecord,
    PatternReinforcement,
    SurpriseEvent,
)
from aragora.nomic.postgres_cycle_store import PostgresCycleLearningStore


# =========================================================================
# Test Fixtures
# =========================================================================


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.get_size.return_value = 5
    pool.get_min_size.return_value = 2
    pool.get_max_size.return_value = 10
    pool.get_idle_size.return_value = 3
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@pytest.fixture
def store_with_mock_pool(mock_pool, mock_connection):
    """Create a PostgresCycleLearningStore with mocked pool."""
    # Mock the connection context manager
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    # Patch ASYNCPG_AVAILABLE to True
    with patch("aragora.storage.postgres_store.ASYNCPG_AVAILABLE", True):
        store = PostgresCycleLearningStore(pool=mock_pool, use_resilient=False)
    return store, mock_connection


@pytest.fixture
def sample_cycle_record():
    """Create a sample NomicCycleRecord for testing."""
    record = NomicCycleRecord(
        cycle_id="test-cycle-123",
        started_at=time.time() - 100,
        topics_debated=["Add rate limiting", "Improve security"],
    )
    record.phases_completed = ["context", "debate", "design"]
    record.add_agent_contribution(
        "claude",
        proposals_made=5,
        proposals_accepted=3,
        critiques_given=2,
        critiques_valuable=1,
    )
    record.add_agent_contribution(
        "gpt4",
        proposals_made=4,
        proposals_accepted=2,
    )
    record.add_surprise(
        phase="implement",
        description="Tests took longer than expected",
        expected="5 minutes",
        actual="20 minutes",
        impact="medium",
    )
    record.add_pattern_reinforcement(
        pattern_type="refactor",
        description="Incremental refactor succeeded",
        success=True,
        confidence=0.85,
    )
    record.mark_complete(success=True)
    return record


@pytest.fixture
def sample_cycle_dict(sample_cycle_record):
    """Get sample cycle as dictionary."""
    return sample_cycle_record.to_dict()


# =========================================================================
# CRUD Operations Tests
# =========================================================================


class TestSaveCycle:
    """Tests for save_cycle and save_cycle_async."""

    @pytest.mark.asyncio
    async def test_save_cycle_async_basic(self, store_with_mock_pool, sample_cycle_record):
        """Should save a cycle record to the database."""
        store, mock_conn = store_with_mock_pool

        await store.save_cycle_async(sample_cycle_record)

        # Verify execute was called with correct SQL
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]

        assert "INSERT INTO cycles" in sql
        assert "ON CONFLICT (cycle_id) DO UPDATE" in sql
        # Verify parameters
        assert call_args[0][1] == sample_cycle_record.cycle_id
        assert call_args[0][2] == sample_cycle_record.started_at

    @pytest.mark.asyncio
    async def test_save_cycle_async_serializes_topics(
        self, store_with_mock_pool, sample_cycle_record
    ):
        """Should serialize topics as JSON."""
        store, mock_conn = store_with_mock_pool

        await store.save_cycle_async(sample_cycle_record)

        call_args = mock_conn.execute.call_args[0]
        topics_json = call_args[6]  # 7th parameter is topics

        # Verify it's valid JSON
        topics = json.loads(topics_json)
        assert topics == sample_cycle_record.topics_debated

    @pytest.mark.asyncio
    async def test_save_cycle_async_serializes_full_data(
        self, store_with_mock_pool, sample_cycle_record
    ):
        """Should serialize full record data as JSON."""
        store, mock_conn = store_with_mock_pool

        await store.save_cycle_async(sample_cycle_record)

        call_args = mock_conn.execute.call_args[0]
        data_json = call_args[7]  # 8th parameter is data

        # Verify it's valid JSON with expected fields
        data = json.loads(data_json)
        assert data["cycle_id"] == sample_cycle_record.cycle_id
        assert data["success"] == sample_cycle_record.success
        assert "agent_contributions" in data

    def test_save_cycle_sync_wrapper(self, store_with_mock_pool, sample_cycle_record):
        """sync save_cycle should call async version."""
        store, mock_conn = store_with_mock_pool

        with patch.object(store, "save_cycle_async", new_callable=AsyncMock) as mock_async:
            with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
                mock_run.return_value = None
                store.save_cycle(sample_cycle_record)

                # Verify run_async was called with the coroutine
                mock_run.assert_called_once()


class TestLoadCycle:
    """Tests for load_cycle and load_cycle_async."""

    @pytest.mark.asyncio
    async def test_load_cycle_async_found(self, store_with_mock_pool, sample_cycle_dict):
        """Should load and deserialize a cycle record."""
        store, mock_conn = store_with_mock_pool

        # Mock the fetchrow response
        mock_conn.fetchrow.return_value = {"data": sample_cycle_dict}

        result = await store.load_cycle_async("test-cycle-123")

        assert result is not None
        assert result.cycle_id == "test-cycle-123"
        assert result.success is True
        assert "claude" in result.agent_contributions

    @pytest.mark.asyncio
    async def test_load_cycle_async_not_found(self, store_with_mock_pool):
        """Should return None for nonexistent cycle."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetchrow.return_value = None

        result = await store.load_cycle_async("nonexistent-cycle")

        assert result is None

    @pytest.mark.asyncio
    async def test_load_cycle_async_handles_string_data(
        self, store_with_mock_pool, sample_cycle_dict
    ):
        """Should handle data returned as JSON string."""
        store, mock_conn = store_with_mock_pool

        # Some drivers return JSON as string
        mock_conn.fetchrow.return_value = {"data": json.dumps(sample_cycle_dict)}

        result = await store.load_cycle_async("test-cycle-123")

        assert result is not None
        assert result.cycle_id == "test-cycle-123"

    def test_load_cycle_sync_wrapper(self, store_with_mock_pool):
        """sync load_cycle should call async version."""
        store, mock_conn = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = None
            result = store.load_cycle("test-123")

            mock_run.assert_called_once()


class TestGetRecentCycles:
    """Tests for get_recent_cycles and get_recent_cycles_async."""

    @pytest.mark.asyncio
    async def test_get_recent_cycles_async_returns_ordered(self, store_with_mock_pool):
        """Should return cycles ordered by started_at DESC."""
        store, mock_conn = store_with_mock_pool

        # Create mock cycle data
        cycles = []
        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_recent_cycles_async(3)

        assert len(result) == 3
        # Verify correct SQL was used
        call_args = mock_conn.fetch.call_args[0]
        assert "ORDER BY started_at DESC" in call_args[0]
        assert "LIMIT" in call_args[0]

    @pytest.mark.asyncio
    async def test_get_recent_cycles_async_empty(self, store_with_mock_pool):
        """Should return empty list when no cycles exist."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetch.return_value = []

        result = await store.get_recent_cycles_async(10)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_recent_cycles_async_default_limit(self, store_with_mock_pool):
        """Should use default limit of 10."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetch.return_value = []

        await store.get_recent_cycles_async()

        call_args = mock_conn.fetch.call_args[0]
        assert call_args[1] == 10  # Default limit


class TestGetSuccessfulCycles:
    """Tests for get_successful_cycles and get_successful_cycles_async."""

    @pytest.mark.asyncio
    async def test_get_successful_cycles_async_filters_success(self, store_with_mock_pool):
        """Should filter to only successful cycles."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="success-1", started_at=time.time())
        record.mark_complete(success=True)
        mock_conn.fetch.return_value = [{"data": record.to_dict()}]

        result = await store.get_successful_cycles_async(5)

        # Verify SQL includes success filter
        call_args = mock_conn.fetch.call_args[0]
        assert "WHERE success = TRUE" in call_args[0]
        assert len(result) == 1
        assert result[0].success is True

    @pytest.mark.asyncio
    async def test_get_successful_cycles_async_empty(self, store_with_mock_pool):
        """Should return empty list when no successful cycles."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetch.return_value = []

        result = await store.get_successful_cycles_async(10)

        assert result == []


# =========================================================================
# Query By Topic Tests
# =========================================================================


class TestQueryByTopic:
    """Tests for query_by_topic and query_by_topic_async."""

    @pytest.mark.asyncio
    async def test_query_by_topic_async_uses_fulltext_search(self, store_with_mock_pool):
        """Should use PostgreSQL full-text search."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(
            cycle_id="auth-cycle",
            started_at=time.time(),
            topics_debated=["Fix authentication bug"],
        )
        mock_conn.fetch.return_value = [{"data": record.to_dict()}]

        result = await store.query_by_topic_async("authentication", limit=5)

        # Verify full-text search SQL
        call_args = mock_conn.fetch.call_args[0]
        assert "search_vector @@ plainto_tsquery" in call_args[0]
        assert call_args[1] == "authentication"
        assert call_args[2] == 5

    @pytest.mark.asyncio
    async def test_query_by_topic_async_returns_matching_cycles(self, store_with_mock_pool):
        """Should return cycles matching the topic."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(
            cycle_id="security-cycle",
            started_at=time.time(),
            topics_debated=["Add security headers"],
        )
        mock_conn.fetch.return_value = [{"data": record.to_dict()}]

        result = await store.query_by_topic_async("security")

        assert len(result) == 1
        assert result[0].cycle_id == "security-cycle"

    @pytest.mark.asyncio
    async def test_query_by_topic_async_no_matches(self, store_with_mock_pool):
        """Should return empty list when no matches."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetch.return_value = []

        result = await store.query_by_topic_async("nonexistent-topic")

        assert result == []

    @pytest.mark.asyncio
    async def test_query_by_topic_async_default_limit(self, store_with_mock_pool):
        """Should use default limit of 10."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetch.return_value = []

        await store.query_by_topic_async("topic")

        call_args = mock_conn.fetch.call_args[0]
        assert call_args[2] == 10  # Default limit


# =========================================================================
# Agent Trajectory Tests
# =========================================================================


class TestGetAgentTrajectory:
    """Tests for get_agent_trajectory and get_agent_trajectory_async."""

    @pytest.mark.asyncio
    async def test_get_agent_trajectory_async_extracts_contributions(self, store_with_mock_pool):
        """Should extract agent contributions from cycles."""
        store, mock_conn = store_with_mock_pool

        # Create cycles with agent contributions
        cycles = []
        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            record.add_agent_contribution(
                "claude",
                proposals_made=5,
                proposals_accepted=2 + i,
                critiques_given=3,
                critiques_valuable=1,
            )
            record.agent_contributions["claude"].quality_score = 0.8 + (i * 0.05)
            record.mark_complete(success=True)
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude", n=3)

        assert len(result) == 3
        # Verify trajectory structure
        assert "cycle_id" in result[0]
        assert "timestamp" in result[0]
        assert "proposals_made" in result[0]
        assert "proposals_accepted" in result[0]
        assert "acceptance_rate" in result[0]
        assert "quality_score" in result[0]
        assert "cycle_success" in result[0]

    @pytest.mark.asyncio
    async def test_get_agent_trajectory_async_calculates_acceptance_rate(
        self, store_with_mock_pool
    ):
        """Should calculate acceptance rate correctly."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_agent_contribution("claude", proposals_made=10, proposals_accepted=8)
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude")

        assert len(result) == 1
        assert result[0]["acceptance_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_get_agent_trajectory_async_handles_zero_proposals(self, store_with_mock_pool):
        """Should handle zero proposals without division error."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_agent_contribution("claude", proposals_made=0, proposals_accepted=0)
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude")

        assert len(result) == 1
        assert result[0]["acceptance_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_agent_trajectory_async_filters_agent(self, store_with_mock_pool):
        """Should only include cycles where agent participated."""
        store, mock_conn = store_with_mock_pool

        # Create cycles - only some have the target agent
        cycles = []
        for i in range(3):
            record = NomicCycleRecord(cycle_id=f"cycle-{i}", started_at=time.time() + i)
            if i % 2 == 0:  # Only even cycles have claude
                record.add_agent_contribution("claude", proposals_made=5)
            else:
                record.add_agent_contribution("gpt4", proposals_made=3)
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude")

        assert len(result) == 2  # Only 2 cycles have claude

    @pytest.mark.asyncio
    async def test_get_agent_trajectory_async_no_participation(self, store_with_mock_pool):
        """Should return empty list if agent never participated."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_agent_contribution("gpt4", proposals_made=5)
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude")

        assert result == []


# =========================================================================
# Pattern Statistics Tests
# =========================================================================


class TestGetPatternStatistics:
    """Tests for get_pattern_statistics and get_pattern_statistics_async."""

    @pytest.mark.asyncio
    async def test_get_pattern_statistics_async_aggregates(self, store_with_mock_pool):
        """Should aggregate pattern statistics across cycles."""
        store, mock_conn = store_with_mock_pool

        cycles = []
        for i in range(3):
            record = NomicCycleRecord(cycle_id=f"cycle-{i}", started_at=time.time() + i)
            record.add_pattern_reinforcement(
                "bugfix",
                f"Fix bug {i}",
                success=(i != 1),  # 2 successes, 1 failure
                confidence=0.7 + (i * 0.1),
            )
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_pattern_statistics_async()

        assert "bugfix" in result
        assert result["bugfix"]["success_count"] == 2
        assert result["bugfix"]["failure_count"] == 1
        assert "success_rate" in result["bugfix"]
        assert "avg_confidence" in result["bugfix"]

    @pytest.mark.asyncio
    async def test_get_pattern_statistics_async_calculates_rates(self, store_with_mock_pool):
        """Should calculate success rate and avg confidence."""
        store, mock_conn = store_with_mock_pool

        cycles = []
        for i in range(4):
            record = NomicCycleRecord(cycle_id=f"cycle-{i}", started_at=time.time() + i)
            record.add_pattern_reinforcement(
                "refactor",
                f"Refactor {i}",
                success=(i < 3),  # 3 successes, 1 failure
                confidence=0.8,
            )
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_pattern_statistics_async()

        assert result["refactor"]["success_rate"] == 0.75
        assert result["refactor"]["avg_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_get_pattern_statistics_async_multiple_patterns(self, store_with_mock_pool):
        """Should track multiple pattern types."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_pattern_reinforcement("bugfix", "Fix", True)
        record.add_pattern_reinforcement("refactor", "Refactor", True)
        record.add_pattern_reinforcement("security", "Security", False)
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_pattern_statistics_async()

        assert len(result) == 3
        assert "bugfix" in result
        assert "refactor" in result
        assert "security" in result

    @pytest.mark.asyncio
    async def test_get_pattern_statistics_async_keeps_examples(self, store_with_mock_pool):
        """Should keep up to 3 examples per pattern."""
        store, mock_conn = store_with_mock_pool

        cycles = []
        for i in range(5):
            record = NomicCycleRecord(cycle_id=f"cycle-{i}", started_at=time.time() + i)
            record.add_pattern_reinforcement("bugfix", f"Example {i}", True)
            cycles.append({"data": record.to_dict()})

        mock_conn.fetch.return_value = cycles

        result = await store.get_pattern_statistics_async()

        assert len(result["bugfix"]["examples"]) <= 3

    @pytest.mark.asyncio
    async def test_get_pattern_statistics_async_empty(self, store_with_mock_pool):
        """Should return empty dict when no patterns."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_pattern_statistics_async()

        assert result == {}


# =========================================================================
# Surprise Summary Tests
# =========================================================================


class TestGetSurpriseSummary:
    """Tests for get_surprise_summary and get_surprise_summary_async."""

    @pytest.mark.asyncio
    async def test_get_surprise_summary_async_groups_by_phase(self, store_with_mock_pool):
        """Should group surprises by phase."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_surprise("debate", "Debate surprise", "A", "B")
        record.add_surprise("implement", "Implement surprise", "C", "D")
        record.add_surprise("debate", "Another debate surprise", "E", "F")
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_surprise_summary_async()

        assert "debate" in result
        assert "implement" in result
        assert len(result["debate"]) == 2
        assert len(result["implement"]) == 1

    @pytest.mark.asyncio
    async def test_get_surprise_summary_async_includes_details(self, store_with_mock_pool):
        """Should include surprise details in summary."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        record.add_surprise(
            phase="verify",
            description="Tests failed",
            expected="All pass",
            actual="3 failures",
            impact="high",
        )
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_surprise_summary_async()

        surprise = result["verify"][0]
        assert surprise["cycle_id"] == "cycle-1"
        assert surprise["description"] == "Tests failed"
        assert surprise["expected"] == "All pass"
        assert surprise["actual"] == "3 failures"
        assert surprise["impact"] == "high"

    @pytest.mark.asyncio
    async def test_get_surprise_summary_async_empty(self, store_with_mock_pool):
        """Should return empty dict when no surprises."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="cycle-1", started_at=time.time())
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_surprise_summary_async()

        assert result == {}


# =========================================================================
# Cleanup Tests
# =========================================================================


class TestCleanupOldCycles:
    """Tests for cleanup_old_cycles and cleanup_old_cycles_async."""

    @pytest.mark.asyncio
    async def test_cleanup_old_cycles_async_deletes_old(self, store_with_mock_pool):
        """Should delete cycles older than cutoff."""
        store, mock_conn = store_with_mock_pool

        # Mock cutoff query
        mock_conn.fetchrow.return_value = {"started_at": time.time() - 1000}
        mock_conn.execute.return_value = "DELETE 5"

        result = await store.cleanup_old_cycles_async(keep_count=100)

        assert result == 5
        # Verify delete was called
        assert mock_conn.execute.called

    @pytest.mark.asyncio
    async def test_cleanup_old_cycles_async_no_old_cycles(self, store_with_mock_pool):
        """Should return 0 when no cycles to delete."""
        store, mock_conn = store_with_mock_pool

        # No cutoff row means fewer cycles than keep_count
        mock_conn.fetchrow.return_value = None

        result = await store.cleanup_old_cycles_async(keep_count=100)

        assert result == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_cycles_async_handles_parse_error(self, store_with_mock_pool):
        """Should handle malformed delete result."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetchrow.return_value = {"started_at": time.time()}
        mock_conn.execute.return_value = "UNEXPECTED"

        result = await store.cleanup_old_cycles_async(keep_count=10)

        assert result == 0


# =========================================================================
# Cycle Count Tests
# =========================================================================


class TestGetCycleCount:
    """Tests for get_cycle_count and get_cycle_count_async."""

    @pytest.mark.asyncio
    async def test_get_cycle_count_async_returns_count(self, store_with_mock_pool):
        """Should return the correct count."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetchrow.return_value = {"count": 42}

        result = await store.get_cycle_count_async()

        assert result == 42

    @pytest.mark.asyncio
    async def test_get_cycle_count_async_zero(self, store_with_mock_pool):
        """Should return 0 for empty table."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetchrow.return_value = {"count": 0}

        result = await store.get_cycle_count_async()

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_cycle_count_async_handles_none(self, store_with_mock_pool):
        """Should return 0 if no row returned."""
        store, mock_conn = store_with_mock_pool

        mock_conn.fetchrow.return_value = None

        result = await store.get_cycle_count_async()

        assert result == 0


# =========================================================================
# Connection Handling Tests
# =========================================================================


class TestConnectionHandling:
    """Tests for connection management and close."""

    def test_close_is_noop(self, store_with_mock_pool):
        """close() should be a no-op for pool-based store."""
        store, mock_conn = store_with_mock_pool

        # Should not raise
        store.close()

    def test_schema_constants(self, store_with_mock_pool):
        """Should have correct schema constants."""
        store, _ = store_with_mock_pool

        assert store.SCHEMA_NAME == "cycle_learning"
        assert store.SCHEMA_VERSION == 1
        assert "CREATE TABLE IF NOT EXISTS cycles" in store.INITIAL_SCHEMA


# =========================================================================
# Edge Cases Tests
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_save_cycle_with_empty_topics(self, store_with_mock_pool):
        """Should handle cycle with no topics."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(
            cycle_id="empty-topics",
            started_at=time.time(),
            topics_debated=[],
        )

        await store.save_cycle_async(record)

        call_args = mock_conn.execute.call_args[0]
        topics_json = call_args[6]
        assert json.loads(topics_json) == []

    @pytest.mark.asyncio
    async def test_save_cycle_with_special_characters(self, store_with_mock_pool):
        """Should handle special characters in topics."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(
            cycle_id="special-chars",
            started_at=time.time(),
            topics_debated=['Fix "quoted" bug', "Handle <html> tags", "Use $variables"],
        )

        await store.save_cycle_async(record)

        call_args = mock_conn.execute.call_args[0]
        topics_json = call_args[6]
        topics = json.loads(topics_json)
        assert len(topics) == 3

    @pytest.mark.asyncio
    async def test_load_cycle_with_missing_fields(self, store_with_mock_pool):
        """Should handle cycle data with missing optional fields."""
        store, mock_conn = store_with_mock_pool

        minimal_data = {
            "cycle_id": "minimal",
            "started_at": time.time(),
        }
        mock_conn.fetchrow.return_value = {"data": minimal_data}

        result = await store.load_cycle_async("minimal")

        assert result is not None
        assert result.cycle_id == "minimal"
        assert result.topics_debated == []
        assert result.agent_contributions == {}

    @pytest.mark.asyncio
    async def test_trajectory_with_complex_contributions(self, store_with_mock_pool):
        """Should handle contributions with all metrics."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(cycle_id="complex", started_at=time.time())
        contrib = AgentContribution(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=15,
            critiques_valuable=12,
            votes_cast=20,
            consensus_aligned=18,
            quality_score=0.92,
        )
        record.agent_contributions["claude"] = contrib
        record.mark_complete(success=True)
        cycles = [{"data": record.to_dict()}]
        mock_conn.fetch.return_value = cycles

        result = await store.get_agent_trajectory_async("claude")

        assert len(result) == 1
        traj = result[0]
        assert traj["proposals_made"] == 10
        assert traj["critiques_given"] == 15
        assert traj["critiques_valuable"] == 12
        assert traj["quality_score"] == 0.92

    @pytest.mark.asyncio
    async def test_pattern_statistics_with_zero_total(self, store_with_mock_pool):
        """Should handle pattern with zero total (shouldn't happen but test anyway)."""
        store, mock_conn = store_with_mock_pool

        # Empty cycles list
        mock_conn.fetch.return_value = []

        result = await store.get_pattern_statistics_async()

        assert result == {}


# =========================================================================
# Sync Wrapper Tests
# =========================================================================


class TestSyncWrappers:
    """Tests for sync wrapper methods."""

    def test_get_recent_cycles_sync(self, store_with_mock_pool):
        """sync get_recent_cycles should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = []
            result = store.get_recent_cycles(5)

            mock_run.assert_called_once()
            assert result == []

    def test_get_successful_cycles_sync(self, store_with_mock_pool):
        """sync get_successful_cycles should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = []
            result = store.get_successful_cycles(5)

            mock_run.assert_called_once()

    def test_query_by_topic_sync(self, store_with_mock_pool):
        """sync query_by_topic should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = []
            result = store.query_by_topic("test", limit=5)

            mock_run.assert_called_once()

    def test_get_agent_trajectory_sync(self, store_with_mock_pool):
        """sync get_agent_trajectory should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = []
            result = store.get_agent_trajectory("claude", n=10)

            mock_run.assert_called_once()

    def test_get_pattern_statistics_sync(self, store_with_mock_pool):
        """sync get_pattern_statistics should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = {}
            result = store.get_pattern_statistics()

            mock_run.assert_called_once()

    def test_get_surprise_summary_sync(self, store_with_mock_pool):
        """sync get_surprise_summary should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = {}
            result = store.get_surprise_summary(n=20)

            mock_run.assert_called_once()

    def test_cleanup_old_cycles_sync(self, store_with_mock_pool):
        """sync cleanup_old_cycles should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = 0
            result = store.cleanup_old_cycles(keep_count=50)

            mock_run.assert_called_once()

    def test_get_cycle_count_sync(self, store_with_mock_pool):
        """sync get_cycle_count should call async version."""
        store, _ = store_with_mock_pool

        with patch("aragora.nomic.postgres_cycle_store.run_async") as mock_run:
            mock_run.return_value = 42
            result = store.get_cycle_count()

            mock_run.assert_called_once()
            assert result == 42


# =========================================================================
# Update/Upsert Tests
# =========================================================================


class TestUpdateCycle:
    """Tests for update behavior (ON CONFLICT DO UPDATE)."""

    @pytest.mark.asyncio
    async def test_save_cycle_updates_existing(self, store_with_mock_pool):
        """Should update existing cycle on conflict."""
        store, mock_conn = store_with_mock_pool

        record = NomicCycleRecord(
            cycle_id="existing-cycle",
            started_at=time.time(),
        )

        await store.save_cycle_async(record)

        # Verify ON CONFLICT clause is present
        call_args = mock_conn.execute.call_args[0]
        sql = call_args[0]
        assert "ON CONFLICT (cycle_id) DO UPDATE" in sql
        assert "started_at = EXCLUDED.started_at" in sql

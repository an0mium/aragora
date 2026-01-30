"""
Comprehensive tests for Debate Store.

Tests the DebateStore persistent storage and analytics system including:
- Store initialization and schema creation
- Recording deliberations
- Updating deliberation results
- Recording agent participation
- Deliberation statistics
- Channel-level breakdown
- Consensus statistics by team composition
- Performance metrics (latency, cost, efficiency)
- CRUD operations
- Data integrity
- Concurrent access patterns
"""

import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.debate_store import DebateStore, get_debate_store


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return tmp_path / "test_debate_store.db"


@pytest.fixture
def store(temp_db_path):
    """Create a DebateStore instance with isolated database."""
    return DebateStore(db_path=temp_db_path)


@pytest.fixture
def populated_store(store):
    """Store with pre-populated deliberations for testing."""
    org_id = "test_org"
    now = datetime.utcnow()

    # Create 10 deliberations with varying properties
    for i in range(10):
        delib_id = f"delib_{i}"
        template = ["brainstorm", "decision", "review"][i % 3]
        priority = ["low", "normal", "high"][i % 3]
        platform = ["slack", "api", "telegram"][i % 3]
        consensus = i % 2 == 0  # Half reach consensus
        team = ["claude", "gpt", "gemini"][: (i % 3) + 1]

        store.record_deliberation(
            deliberation_id=delib_id,
            org_id=org_id,
            question=f"Test question {i}",
            status="completed",
            template=template,
            priority=priority,
            platform=platform,
            channel_id=f"channel_{i % 3}",
            channel_name=f"Channel {i % 3}",
            team_agents=team,
        )

        store.update_deliberation_result(
            deliberation_id=delib_id,
            status="completed",
            consensus_reached=consensus,
            rounds=i + 1,
            duration_seconds=float(10 + i * 5),
            total_tokens=1000 * (i + 1),
            total_cost_usd=0.01 * (i + 1),
        )

        # Record agent participations
        for agent in team:
            store.record_agent_participation(
                deliberation_id=delib_id,
                agent_id=agent,
                agent_name=agent.capitalize(),
                tokens_used=500,
                cost_usd=0.005,
                agreed_with_consensus=consensus,
            )

    return store, org_id


@pytest.fixture
def time_range():
    """Provide a default time range for queries."""
    now = datetime.utcnow()
    return now - timedelta(hours=1), now + timedelta(hours=1)


# =============================================================================
# Test Store Initialization
# =============================================================================


class TestStoreInitialization:
    """Test DebateStore initialization and schema creation."""

    @pytest.mark.smoke
    def test_default_initialization(self, temp_db_path):
        """Test that DebateStore initializes with default settings."""
        store = DebateStore(db_path=temp_db_path)
        assert store.db_path == temp_db_path

    def test_schema_creation(self, temp_db_path):
        """Test that database schema is created on initialization."""
        store = DebateStore(db_path=temp_db_path)

        with store._connection() as conn:
            cursor = conn.cursor()

            # Check deliberations table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='deliberations'"
            )
            assert cursor.fetchone() is not None

            # Check deliberation_agents table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='deliberation_agents'"
            )
            assert cursor.fetchone() is not None

    def test_indices_created(self, temp_db_path):
        """Test that indices are created on initialization."""
        store = DebateStore(db_path=temp_db_path)

        with store._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indices = {row[0] for row in cursor.fetchall()}

            assert "idx_deliberations_org_id" in indices
            assert "idx_deliberations_created_at" in indices
            assert "idx_deliberations_status" in indices
            assert "idx_deliberations_platform" in indices
            assert "idx_delib_agents_agent_id" in indices

    def test_connection_context_manager(self, store):
        """Test that connection context manager works correctly."""
        with store._connection() as conn:
            assert conn is not None
            assert conn.row_factory == sqlite3.Row

    def test_db_path_from_config(self):
        """Test that default db_path comes from config."""
        with patch("aragora.memory.debate_store.get_db_path") as mock_get_path:
            mock_get_path.return_value = Path("/tmp/test.db")
            store = DebateStore()
            assert store.db_path == Path("/tmp/test.db")


# =============================================================================
# Test Recording Deliberations
# =============================================================================


class TestRecordDeliberation:
    """Test recording new deliberations."""

    def test_record_basic_deliberation(self, store):
        """Test recording a basic deliberation."""
        store.record_deliberation(
            deliberation_id="test_1",
            org_id="org_1",
            question="What is the best approach?",
        )

        with store._connection() as conn:
            row = conn.execute("SELECT * FROM deliberations WHERE id = ?", ("test_1",)).fetchone()

            assert row is not None
            assert row["org_id"] == "org_1"
            assert row["question"] == "What is the best approach?"
            assert row["status"] == "pending"
            assert row["priority"] == "normal"

    def test_record_deliberation_with_all_fields(self, store):
        """Test recording deliberation with all optional fields."""
        store.record_deliberation(
            deliberation_id="full_1",
            org_id="org_1",
            question="Complete question",
            status="in_progress",
            template="decision",
            priority="high",
            platform="slack",
            channel_id="C123",
            channel_name="general",
            team_agents=["claude", "gpt", "gemini"],
            metadata={"source": "api", "tags": ["urgent"]},
        )

        with store._connection() as conn:
            row = conn.execute("SELECT * FROM deliberations WHERE id = ?", ("full_1",)).fetchone()

            assert row["status"] == "in_progress"
            assert row["template"] == "decision"
            assert row["priority"] == "high"
            assert row["platform"] == "slack"
            assert row["channel_id"] == "C123"
            assert row["channel_name"] == "general"
            assert row["team_agents"] == "claude,gpt,gemini"
            assert "source" in row["metadata"]

    def test_record_deliberation_sets_created_at(self, store):
        """Test that created_at is automatically set."""
        before = datetime.utcnow()
        store.record_deliberation(
            deliberation_id="time_1",
            org_id="org_1",
            question="Time test",
        )
        after = datetime.utcnow()

        with store._connection() as conn:
            row = conn.execute(
                "SELECT created_at FROM deliberations WHERE id = ?", ("time_1",)
            ).fetchone()

            created = datetime.fromisoformat(row["created_at"])
            assert before <= created <= after

    def test_record_deliberation_without_team_agents(self, store):
        """Test recording deliberation without team_agents."""
        store.record_deliberation(
            deliberation_id="no_team",
            org_id="org_1",
            question="Solo question",
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT team_agents FROM deliberations WHERE id = ?", ("no_team",)
            ).fetchone()

            assert row["team_agents"] is None


# =============================================================================
# Test Updating Deliberation Results
# =============================================================================


class TestUpdateDeliberationResult:
    """Test updating deliberation results."""

    def test_update_basic_result(self, store):
        """Test updating a deliberation with basic result."""
        store.record_deliberation(
            deliberation_id="update_1",
            org_id="org_1",
            question="To be updated",
        )

        store.update_deliberation_result(
            deliberation_id="update_1",
            status="completed",
            consensus_reached=True,
            rounds=3,
            duration_seconds=45.5,
        )

        with store._connection() as conn:
            row = conn.execute("SELECT * FROM deliberations WHERE id = ?", ("update_1",)).fetchone()

            assert row["status"] == "completed"
            assert row["consensus_reached"] == 1
            assert row["rounds"] == 3
            assert row["duration_seconds"] == 45.5

    def test_update_result_with_cost_metrics(self, store):
        """Test updating result with cost metrics."""
        store.record_deliberation(
            deliberation_id="cost_1",
            org_id="org_1",
            question="Cost test",
        )

        store.update_deliberation_result(
            deliberation_id="cost_1",
            status="completed",
            consensus_reached=True,
            rounds=5,
            duration_seconds=120.0,
            total_tokens=15000,
            total_cost_usd=0.15,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT total_tokens, total_cost_usd FROM deliberations WHERE id = ?",
                ("cost_1",),
            ).fetchone()

            assert row["total_tokens"] == 15000
            assert row["total_cost_usd"] == 0.15

    def test_update_result_sets_completed_at(self, store):
        """Test that completed_at is set on update."""
        store.record_deliberation(
            deliberation_id="complete_1",
            org_id="org_1",
            question="Complete test",
        )

        before = datetime.utcnow()
        store.update_deliberation_result(
            deliberation_id="complete_1",
            status="completed",
            consensus_reached=True,
            rounds=1,
            duration_seconds=10.0,
        )
        after = datetime.utcnow()

        with store._connection() as conn:
            row = conn.execute(
                "SELECT completed_at FROM deliberations WHERE id = ?", ("complete_1",)
            ).fetchone()

            completed = datetime.fromisoformat(row["completed_at"])
            assert before <= completed <= after

    def test_update_result_consensus_false(self, store):
        """Test updating result without consensus."""
        store.record_deliberation(
            deliberation_id="no_consensus",
            org_id="org_1",
            question="No consensus",
        )

        store.update_deliberation_result(
            deliberation_id="no_consensus",
            status="completed",
            consensus_reached=False,
            rounds=5,
            duration_seconds=300.0,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT consensus_reached FROM deliberations WHERE id = ?",
                ("no_consensus",),
            ).fetchone()

            assert row["consensus_reached"] == 0

    def test_update_failed_deliberation(self, store):
        """Test updating a failed deliberation."""
        store.record_deliberation(
            deliberation_id="failed_1",
            org_id="org_1",
            question="Will fail",
        )

        store.update_deliberation_result(
            deliberation_id="failed_1",
            status="failed",
            consensus_reached=False,
            rounds=0,
            duration_seconds=5.0,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT status FROM deliberations WHERE id = ?", ("failed_1",)
            ).fetchone()

            assert row["status"] == "failed"


# =============================================================================
# Test Recording Agent Participation
# =============================================================================


class TestRecordAgentParticipation:
    """Test recording agent participation in deliberations."""

    def test_record_basic_participation(self, store):
        """Test recording basic agent participation."""
        store.record_deliberation(
            deliberation_id="part_1",
            org_id="org_1",
            question="Participation test",
        )

        store.record_agent_participation(
            deliberation_id="part_1",
            agent_id="claude",
        )

        with store._connection() as conn:
            row = conn.execute(
                """SELECT * FROM deliberation_agents
                   WHERE deliberation_id = ? AND agent_id = ?""",
                ("part_1", "claude"),
            ).fetchone()

            assert row is not None
            assert row["tokens_used"] == 0
            assert row["cost_usd"] == 0

    def test_record_participation_with_metrics(self, store):
        """Test recording participation with full metrics."""
        store.record_deliberation(
            deliberation_id="metrics_1",
            org_id="org_1",
            question="Metrics test",
        )

        store.record_agent_participation(
            deliberation_id="metrics_1",
            agent_id="gpt",
            agent_name="GPT-4",
            tokens_used=5000,
            cost_usd=0.05,
            agreed_with_consensus=True,
        )

        with store._connection() as conn:
            row = conn.execute(
                """SELECT * FROM deliberation_agents
                   WHERE deliberation_id = ? AND agent_id = ?""",
                ("metrics_1", "gpt"),
            ).fetchone()

            assert row["agent_name"] == "GPT-4"
            assert row["tokens_used"] == 5000
            assert row["cost_usd"] == 0.05
            assert row["agreed_with_consensus"] == 1

    def test_record_multiple_agents(self, store):
        """Test recording multiple agents for same deliberation."""
        store.record_deliberation(
            deliberation_id="multi_1",
            org_id="org_1",
            question="Multi agent test",
        )

        agents = [("claude", "Claude"), ("gpt", "GPT-4"), ("gemini", "Gemini")]
        for agent_id, agent_name in agents:
            store.record_agent_participation(
                deliberation_id="multi_1",
                agent_id=agent_id,
                agent_name=agent_name,
            )

        with store._connection() as conn:
            rows = conn.execute(
                "SELECT agent_id FROM deliberation_agents WHERE deliberation_id = ?",
                ("multi_1",),
            ).fetchall()

            assert len(rows) == 3

    def test_record_participation_upsert(self, store):
        """Test that recording participation updates existing record."""
        store.record_deliberation(
            deliberation_id="upsert_1",
            org_id="org_1",
            question="Upsert test",
        )

        # First record
        store.record_agent_participation(
            deliberation_id="upsert_1",
            agent_id="claude",
            tokens_used=1000,
        )

        # Update with new values
        store.record_agent_participation(
            deliberation_id="upsert_1",
            agent_id="claude",
            tokens_used=2000,
        )

        with store._connection() as conn:
            rows = conn.execute(
                """SELECT tokens_used FROM deliberation_agents
                   WHERE deliberation_id = ? AND agent_id = ?""",
                ("upsert_1", "claude"),
            ).fetchall()

            # Should only have one row
            assert len(rows) == 1
            assert rows[0]["tokens_used"] == 2000


# =============================================================================
# Test Deliberation Statistics
# =============================================================================


class TestDeliberationStats:
    """Test deliberation statistics retrieval."""

    def test_get_stats_empty(self, store, time_range):
        """Test getting stats when no deliberations exist."""
        start, end = time_range
        stats = store.get_deliberation_stats("org_1", start, end)

        assert stats["total"] == 0
        assert stats["completed"] == 0
        assert stats["in_progress"] == 0
        assert stats["failed"] == 0
        assert stats["consensus_reached"] == 0
        assert stats["avg_rounds"] == 0
        assert stats["avg_duration_seconds"] == 0

    def test_get_stats_with_data(self, populated_store, time_range):
        """Test getting stats with existing deliberations."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats(org_id, start, end)

        assert stats["total"] == 10
        assert stats["completed"] == 10
        assert stats["consensus_reached"] == 5  # Half reach consensus

    def test_get_stats_by_template(self, populated_store, time_range):
        """Test getting stats broken down by template."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats(org_id, start, end)

        assert "by_template" in stats
        assert len(stats["by_template"]) > 0

    def test_get_stats_by_priority(self, populated_store, time_range):
        """Test getting stats broken down by priority."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats(org_id, start, end)

        assert "by_priority" in stats
        assert len(stats["by_priority"]) > 0

    def test_get_stats_time_filtering(self, store):
        """Test that stats respect time range filtering."""
        # Create old deliberation
        store.record_deliberation(
            deliberation_id="old_1",
            org_id="org_1",
            question="Old question",
        )

        # Manually set old created_at
        with store._connection() as conn:
            old_time = (datetime.utcnow() - timedelta(days=30)).isoformat()
            conn.execute(
                "UPDATE deliberations SET created_at = ? WHERE id = ?",
                (old_time, "old_1"),
            )
            conn.commit()

        # Query with recent time range
        now = datetime.utcnow()
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        stats = store.get_deliberation_stats("org_1", start, end)

        assert stats["total"] == 0  # Old deliberation should be excluded

    def test_get_stats_avg_rounds(self, populated_store, time_range):
        """Test that average rounds is calculated correctly."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats(org_id, start, end)

        # Rounds are 1-10, average is 5.5
        assert stats["avg_rounds"] == 5.5

    def test_get_stats_avg_duration(self, populated_store, time_range):
        """Test that average duration is calculated correctly."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats(org_id, start, end)

        # Durations are 10, 15, 20, ... 55, average is 32.5
        assert stats["avg_duration_seconds"] == 32.5


# =============================================================================
# Test Channel Statistics
# =============================================================================


class TestChannelStats:
    """Test channel-level statistics."""

    def test_get_stats_by_channel_empty(self, store, time_range):
        """Test getting channel stats when no deliberations exist."""
        start, end = time_range
        stats = store.get_deliberation_stats_by_channel("org_1", start, end)

        assert stats == []

    def test_get_stats_by_channel_with_data(self, populated_store, time_range):
        """Test getting channel stats with existing deliberations."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats_by_channel(org_id, start, end)

        assert len(stats) > 0
        for channel in stats:
            assert "platform" in channel
            assert "channel_id" in channel
            assert "total_deliberations" in channel
            assert "consensus_rate" in channel

    def test_channel_stats_consensus_rate(self, populated_store, time_range):
        """Test that consensus rate is calculated per channel."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats_by_channel(org_id, start, end)

        for channel in stats:
            # Consensus rate should be a percentage string
            assert "%" in channel["consensus_rate"]

    def test_channel_stats_top_templates(self, populated_store, time_range):
        """Test that top templates are included."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats_by_channel(org_id, start, end)

        for channel in stats:
            assert "top_templates" in channel
            assert isinstance(channel["top_templates"], list)
            # Should be limited to 3
            assert len(channel["top_templates"]) <= 3

    def test_channel_stats_default_platform(self, store, time_range):
        """Test that missing platform defaults to 'api'."""
        store.record_deliberation(
            deliberation_id="no_platform",
            org_id="org_1",
            question="No platform",
            platform=None,
        )

        start, end = time_range
        stats = store.get_deliberation_stats_by_channel("org_1", start, end)

        assert len(stats) == 1
        assert stats[0]["platform"] == "api"

    def test_channel_stats_ordered_by_count(self, store, time_range):
        """Test that channels are ordered by deliberation count."""
        # Create channels with different counts
        for i in range(5):
            store.record_deliberation(
                deliberation_id=f"ch_a_{i}",
                org_id="org_1",
                question=f"Question {i}",
                channel_id="channel_a",
            )

        for i in range(2):
            store.record_deliberation(
                deliberation_id=f"ch_b_{i}",
                org_id="org_1",
                question=f"Question {i}",
                channel_id="channel_b",
            )

        start, end = time_range
        stats = store.get_deliberation_stats_by_channel("org_1", start, end)

        # First channel should have more deliberations
        assert stats[0]["total_deliberations"] >= stats[1]["total_deliberations"]


# =============================================================================
# Test Consensus Statistics
# =============================================================================


class TestConsensusStats:
    """Test consensus statistics by team composition."""

    def test_get_consensus_stats_empty(self, store, time_range):
        """Test getting consensus stats when no deliberations exist."""
        start, end = time_range
        stats = store.get_consensus_stats("org_1", start, end)

        assert stats["overall_consensus_rate"] == "0%"
        assert stats["by_team_size"] == {}
        assert stats["by_agent"] == []
        assert stats["top_teams"] == []

    def test_get_consensus_stats_overall_rate(self, populated_store, time_range):
        """Test overall consensus rate calculation."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_consensus_stats(org_id, start, end)

        # Half of 10 deliberations reach consensus
        assert stats["overall_consensus_rate"] == "50%"

    def test_get_consensus_stats_by_agent(self, populated_store, time_range):
        """Test per-agent consensus statistics."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_consensus_stats(org_id, start, end)

        assert len(stats["by_agent"]) > 0
        for agent in stats["by_agent"]:
            assert "agent_id" in agent
            assert "participations" in agent
            assert "consensus_rate" in agent
            assert "avg_agreement_score" in agent

    def test_get_consensus_stats_by_team_size(self, populated_store, time_range):
        """Test consensus stats by team size."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_consensus_stats(org_id, start, end)

        # Our populated store has varying team sizes
        assert "by_team_size" in stats

    def test_top_teams_require_minimum_deliberations(self, store, time_range):
        """Test that top teams require minimum 3 deliberations."""
        # Create team with only 2 deliberations
        for i in range(2):
            store.record_deliberation(
                deliberation_id=f"small_team_{i}",
                org_id="org_1",
                question=f"Question {i}",
                team_agents=["claude", "gpt"],
                status="completed",
            )
            store.update_deliberation_result(
                deliberation_id=f"small_team_{i}",
                status="completed",
                consensus_reached=True,
                rounds=1,
                duration_seconds=10.0,
            )

        start, end = time_range
        stats = store.get_consensus_stats("org_1", start, end)

        # Team with only 2 deliberations should not appear in top_teams
        team_delib_counts = [t["deliberations"] for t in stats["top_teams"]]
        assert all(count >= 3 for count in team_delib_counts)


# =============================================================================
# Test Performance Statistics
# =============================================================================


class TestPerformanceStats:
    """Test deliberation performance metrics."""

    def test_get_performance_empty(self, store, time_range):
        """Test getting performance stats when no deliberations exist."""
        start, end = time_range
        stats = store.get_deliberation_performance("org_1", start, end)

        assert stats["summary"]["total_deliberations"] == 0
        assert stats["by_template"] == []
        assert stats["trends"] == []

    def test_get_performance_summary(self, populated_store, time_range):
        """Test performance summary statistics."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end)
        summary = stats["summary"]

        assert summary["total_deliberations"] == 10
        assert float(summary["total_cost_usd"]) > 0
        assert summary["total_tokens"] > 0
        assert summary["avg_duration_seconds"] > 0
        assert summary["avg_rounds"] > 0

    def test_get_performance_percentiles(self, populated_store, time_range):
        """Test duration percentiles calculation."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end)
        summary = stats["summary"]

        assert "p50_duration_seconds" in summary
        assert "p95_duration_seconds" in summary
        # P95 should be >= P50
        assert summary["p95_duration_seconds"] >= summary["p50_duration_seconds"]

    def test_get_performance_by_template(self, populated_store, time_range):
        """Test performance breakdown by template."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end)

        assert len(stats["by_template"]) > 0
        for template in stats["by_template"]:
            assert "template" in template
            assert "count" in template
            assert "avg_cost" in template
            assert "avg_duration_seconds" in template

    def test_get_performance_trends_daily(self, populated_store, time_range):
        """Test daily performance trends."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end, granularity="day")

        assert "trends" in stats
        for trend in stats["trends"]:
            assert "date" in trend
            assert "count" in trend
            assert "avg_duration_seconds" in trend
            assert "total_cost" in trend

    def test_get_performance_trends_weekly(self, populated_store, time_range):
        """Test weekly performance trends."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end, granularity="week")

        # Should use week format
        for trend in stats["trends"]:
            if trend["date"]:
                # Weekly format includes W
                assert "W" in str(trend["date"]) or "-" in str(trend["date"])

    def test_get_performance_cost_by_agent(self, populated_store, time_range):
        """Test cost breakdown by agent."""
        store, org_id = populated_store
        start, end = time_range

        stats = store.get_deliberation_performance(org_id, start, end)

        assert "cost_by_agent" in stats
        # Should have agents from populated store
        assert len(stats["cost_by_agent"]) > 0


# =============================================================================
# Test Data Integrity
# =============================================================================


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_unique_deliberation_id(self, store):
        """Test that deliberation IDs must be unique."""
        store.record_deliberation(
            deliberation_id="unique_1",
            org_id="org_1",
            question="First",
        )

        with pytest.raises(sqlite3.IntegrityError):
            store.record_deliberation(
                deliberation_id="unique_1",
                org_id="org_1",
                question="Duplicate",
            )

    def test_agent_participation_composite_key(self, store):
        """Test that agent participation uses composite primary key."""
        store.record_deliberation(
            deliberation_id="comp_1",
            org_id="org_1",
            question="Composite key test",
        )

        # Same agent in same deliberation uses upsert
        store.record_agent_participation(
            deliberation_id="comp_1",
            agent_id="claude",
            tokens_used=100,
        )

        store.record_agent_participation(
            deliberation_id="comp_1",
            agent_id="claude",
            tokens_used=200,
        )

        with store._connection() as conn:
            rows = conn.execute(
                """SELECT COUNT(*) as count FROM deliberation_agents
                   WHERE deliberation_id = ? AND agent_id = ?""",
                ("comp_1", "claude"),
            ).fetchone()

            assert rows["count"] == 1

    def test_data_persists_across_connections(self, temp_db_path):
        """Test that data persists across store instances."""
        store1 = DebateStore(db_path=temp_db_path)
        store1.record_deliberation(
            deliberation_id="persist_1",
            org_id="org_1",
            question="Persistence test",
        )

        store2 = DebateStore(db_path=temp_db_path)
        with store2._connection() as conn:
            row = conn.execute(
                "SELECT * FROM deliberations WHERE id = ?", ("persist_1",)
            ).fetchone()

            assert row is not None
            assert row["question"] == "Persistence test"


# =============================================================================
# Test Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Test thread-safety and concurrent access."""

    def test_concurrent_record_deliberations(self, store):
        """Test concurrent recording of deliberations."""
        errors = []

        def record_deliberations(thread_id):
            try:
                for i in range(10):
                    store.record_deliberation(
                        deliberation_id=f"concurrent_{thread_id}_{i}",
                        org_id="org_1",
                        question=f"Question {thread_id}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_deliberations, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        with store._connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM deliberations").fetchone()
            assert row["count"] == 50  # 5 threads * 10 deliberations

    def test_concurrent_reads_and_writes(self, populated_store, time_range):
        """Test concurrent read and write operations."""
        store, org_id = populated_store
        start, end = time_range
        errors = []

        def read_stats():
            try:
                for _ in range(10):
                    store.get_deliberation_stats(org_id, start, end)
            except Exception as e:
                errors.append(e)

        def write_deliberations(thread_id):
            try:
                for i in range(5):
                    store.record_deliberation(
                        deliberation_id=f"rw_{thread_id}_{i}",
                        org_id=org_id,
                        question=f"RW Question {thread_id}_{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_stats),
            threading.Thread(target=write_deliberations, args=(0,)),
            threading.Thread(target=read_stats),
            threading.Thread(target=write_deliberations, args=(1,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_updates(self, store):
        """Test concurrent updates to same deliberation."""
        store.record_deliberation(
            deliberation_id="update_target",
            org_id="org_1",
            question="Update target",
        )

        errors = []

        def update_result(rounds):
            try:
                store.update_deliberation_result(
                    deliberation_id="update_target",
                    status="completed",
                    consensus_reached=True,
                    rounds=rounds,
                    duration_seconds=float(rounds * 10),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_result, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Test Singleton Access
# =============================================================================


class TestSingletonAccess:
    """Test global singleton access."""

    def test_get_debate_store_creates_singleton(self, monkeypatch):
        """Test that get_debate_store creates a singleton."""
        # Reset the global singleton
        import aragora.memory.debate_store as ds

        ds._debate_store = None

        with patch.object(ds, "get_db_path") as mock_path:
            mock_path.return_value = Path(tempfile.mktemp(suffix=".db"))

            store1 = get_debate_store()
            store2 = get_debate_store()

            assert store1 is store2

        # Reset again for other tests
        ds._debate_store = None

    def test_singleton_persists(self, monkeypatch):
        """Test that singleton persists across calls."""
        import aragora.memory.debate_store as ds

        ds._debate_store = None

        with patch.object(ds, "get_db_path") as mock_path:
            mock_path.return_value = Path(tempfile.mktemp(suffix=".db"))

            store = get_debate_store()
            store.record_deliberation(
                deliberation_id="singleton_test",
                org_id="org_1",
                question="Singleton test",
            )

            store2 = get_debate_store()
            with store2._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM deliberations WHERE id = ?", ("singleton_test",)
                ).fetchone()
                assert row is not None

        ds._debate_store = None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_team_agents_list(self, store):
        """Test recording deliberation with empty team list."""
        store.record_deliberation(
            deliberation_id="empty_team",
            org_id="org_1",
            question="Empty team",
            team_agents=[],
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT team_agents FROM deliberations WHERE id = ?", ("empty_team",)
            ).fetchone()

            # Empty list should result in None or empty string
            assert row["team_agents"] is None or row["team_agents"] == ""

    def test_very_long_question(self, store):
        """Test recording deliberation with very long question."""
        long_question = "Q" * 10000
        store.record_deliberation(
            deliberation_id="long_q",
            org_id="org_1",
            question=long_question,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT question FROM deliberations WHERE id = ?", ("long_q",)
            ).fetchone()

            assert len(row["question"]) == 10000

    def test_special_characters_in_question(self, store):
        """Test recording deliberation with special characters."""
        special_question = "What about 'quotes' and \"double quotes\" and emoji?"
        store.record_deliberation(
            deliberation_id="special_q",
            org_id="org_1",
            question=special_question,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT question FROM deliberations WHERE id = ?", ("special_q",)
            ).fetchone()

            assert row["question"] == special_question

    def test_zero_duration(self, store):
        """Test updating deliberation with zero duration."""
        store.record_deliberation(
            deliberation_id="zero_dur",
            org_id="org_1",
            question="Zero duration",
        )

        store.update_deliberation_result(
            deliberation_id="zero_dur",
            status="completed",
            consensus_reached=True,
            rounds=1,
            duration_seconds=0.0,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT duration_seconds FROM deliberations WHERE id = ?", ("zero_dur",)
            ).fetchone()

            assert row["duration_seconds"] == 0.0

    def test_very_high_token_count(self, store):
        """Test updating deliberation with very high token count."""
        store.record_deliberation(
            deliberation_id="high_tokens",
            org_id="org_1",
            question="High tokens",
        )

        store.update_deliberation_result(
            deliberation_id="high_tokens",
            status="completed",
            consensus_reached=True,
            rounds=10,
            duration_seconds=600.0,
            total_tokens=10_000_000,
            total_cost_usd=100.0,
        )

        with store._connection() as conn:
            row = conn.execute(
                "SELECT total_tokens FROM deliberations WHERE id = ?", ("high_tokens",)
            ).fetchone()

            assert row["total_tokens"] == 10_000_000

    def test_query_nonexistent_org(self, populated_store, time_range):
        """Test querying stats for non-existent organization."""
        store, _ = populated_store
        start, end = time_range

        stats = store.get_deliberation_stats("nonexistent_org", start, end)

        assert stats["total"] == 0

    def test_time_range_excludes_future(self, store):
        """Test that future time range excludes current data."""
        store.record_deliberation(
            deliberation_id="current_1",
            org_id="org_1",
            question="Current",
        )

        future_start = datetime.utcnow() + timedelta(days=1)
        future_end = datetime.utcnow() + timedelta(days=2)

        stats = store.get_deliberation_stats("org_1", future_start, future_end)

        assert stats["total"] == 0

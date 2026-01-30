"""
Comprehensive tests for evolution tracker.

Tests cover:
- OutcomeRecord dataclass
- EvolutionTracker class
- Outcome recording
- Agent statistics
- Generation metrics
- Performance delta calculations
- Generation trends
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.evolution.tracker import EvolutionTracker, OutcomeRecord


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
def tracker(temp_db):
    """Create an EvolutionTracker with temp database."""
    return EvolutionTracker(db_path=temp_db)


# =============================================================================
# OutcomeRecord Tests
# =============================================================================


class TestOutcomeRecord:
    """Test OutcomeRecord dataclass."""

    def test_create_basic_record(self):
        """Test creating a basic OutcomeRecord."""
        record = OutcomeRecord(
            agent="claude",
            won=True,
        )

        assert record.agent == "claude"
        assert record.won is True
        assert record.debate_id is None
        assert record.generation == 0
        assert record.recorded_at != ""

    def test_create_full_record(self):
        """Test creating OutcomeRecord with all fields."""
        record = OutcomeRecord(
            agent="gpt4",
            won=False,
            debate_id="debate-123",
            generation=5,
        )

        assert record.agent == "gpt4"
        assert record.won is False
        assert record.debate_id == "debate-123"
        assert record.generation == 5

    def test_auto_timestamp(self):
        """Test automatic timestamp generation."""
        record = OutcomeRecord(agent="test", won=True)

        # Should be a valid ISO timestamp
        timestamp = datetime.fromisoformat(record.recorded_at.replace("Z", "+00:00"))
        assert timestamp is not None

    def test_custom_timestamp(self):
        """Test custom timestamp is preserved."""
        custom_time = "2026-01-15T12:00:00+00:00"
        record = OutcomeRecord(
            agent="test",
            won=True,
            recorded_at=custom_time,
        )

        assert record.recorded_at == custom_time


# =============================================================================
# EvolutionTracker Initialization Tests
# =============================================================================


class TestEvolutionTrackerInit:
    """Test EvolutionTracker initialization."""

    def test_schema_constants(self):
        """Test schema constants are defined."""
        assert EvolutionTracker.SCHEMA_NAME == "evolution_tracker"
        assert EvolutionTracker.SCHEMA_VERSION == 1
        assert "outcomes" in EvolutionTracker.INITIAL_SCHEMA

    def test_creates_tables(self, temp_db):
        """Test that tables are created on init."""
        tracker = EvolutionTracker(db_path=temp_db)

        with tracker.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'")
            assert cursor.fetchone() is not None

    def test_creates_indexes(self, temp_db):
        """Test that indexes are created."""
        tracker = EvolutionTracker(db_path=temp_db)

        with tracker.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            )
            indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_outcomes_agent" in indexes
        assert "idx_outcomes_generation" in indexes


# =============================================================================
# Outcome Recording Tests
# =============================================================================


class TestOutcomeRecording:
    """Test outcome recording functionality."""

    def test_record_win(self, tracker):
        """Test recording a win."""
        tracker.record_outcome(
            agent="claude",
            won=True,
            debate_id="debate-1",
            generation=0,
        )

        stats = tracker.get_agent_stats("claude")
        assert stats["wins"] == 1
        assert stats["total"] == 1
        assert stats["win_rate"] == 1.0

    def test_record_loss(self, tracker):
        """Test recording a loss."""
        tracker.record_outcome(
            agent="claude",
            won=False,
            debate_id="debate-1",
            generation=0,
        )

        stats = tracker.get_agent_stats("claude")
        assert stats["losses"] == 1
        assert stats["total"] == 1
        assert stats["win_rate"] == 0.0

    def test_record_multiple_outcomes(self, tracker):
        """Test recording multiple outcomes."""
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("claude", won=False)
        tracker.record_outcome("claude", won=True)

        stats = tracker.get_agent_stats("claude")
        assert stats["wins"] == 3
        assert stats["losses"] == 1
        assert stats["total"] == 4
        assert stats["win_rate"] == 0.75

    def test_record_with_debate_id(self, tracker):
        """Test recording with debate ID."""
        tracker.record_outcome(
            agent="claude",
            won=True,
            debate_id="specific-debate-123",
        )

        # Verify stored (through stats since we can't query directly)
        stats = tracker.get_agent_stats("claude")
        assert stats["total"] == 1

    def test_record_different_generations(self, tracker):
        """Test recording outcomes for different generations."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=False, generation=1)

        gen0 = tracker.get_generation_metrics(0)
        gen1 = tracker.get_generation_metrics(1)

        assert gen0["total_debates"] == 2
        assert gen0["wins"] == 2
        assert gen1["total_debates"] == 2
        assert gen1["wins"] == 1

    def test_record_different_agents(self, tracker):
        """Test recording outcomes for different agents."""
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("gpt4", won=False)
        tracker.record_outcome("gemini", won=True)

        assert tracker.get_agent_stats("claude")["wins"] == 1
        assert tracker.get_agent_stats("gpt4")["losses"] == 1
        assert tracker.get_agent_stats("gemini")["wins"] == 1


# =============================================================================
# Agent Statistics Tests
# =============================================================================


class TestAgentStats:
    """Test agent statistics retrieval."""

    def test_get_stats_empty(self, tracker):
        """Test getting stats for agent with no outcomes."""
        stats = tracker.get_agent_stats("nonexistent")

        assert stats["agent"] == "nonexistent"
        assert stats["wins"] == 0
        assert stats["losses"] == 0
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0

    def test_get_stats_single_win(self, tracker):
        """Test stats with single win."""
        tracker.record_outcome("claude", won=True)

        stats = tracker.get_agent_stats("claude")

        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["total"] == 1
        assert stats["win_rate"] == 1.0

    def test_get_stats_single_loss(self, tracker):
        """Test stats with single loss."""
        tracker.record_outcome("claude", won=False)

        stats = tracker.get_agent_stats("claude")

        assert stats["wins"] == 0
        assert stats["losses"] == 1
        assert stats["total"] == 1
        assert stats["win_rate"] == 0.0

    def test_get_stats_mixed(self, tracker):
        """Test stats with mixed outcomes."""
        for _ in range(7):
            tracker.record_outcome("claude", won=True)
        for _ in range(3):
            tracker.record_outcome("claude", won=False)

        stats = tracker.get_agent_stats("claude")

        assert stats["wins"] == 7
        assert stats["losses"] == 3
        assert stats["total"] == 10
        assert stats["win_rate"] == 0.7


# =============================================================================
# Generation Metrics Tests
# =============================================================================


class TestGenerationMetrics:
    """Test generation metrics retrieval."""

    def test_get_metrics_empty_generation(self, tracker):
        """Test getting metrics for generation with no outcomes."""
        metrics = tracker.get_generation_metrics(99)

        assert metrics["generation"] == 99
        assert metrics["total_debates"] == 0
        assert metrics["wins"] == 0
        assert metrics["losses"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["unique_agents"] == 0

    def test_get_metrics_single_agent(self, tracker):
        """Test metrics with single agent."""
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=False, generation=1)

        metrics = tracker.get_generation_metrics(1)

        assert metrics["total_debates"] == 3
        assert metrics["wins"] == 2
        assert metrics["losses"] == 1
        assert metrics["unique_agents"] == 1
        assert abs(metrics["win_rate"] - 2 / 3) < 0.01

    def test_get_metrics_multiple_agents(self, tracker):
        """Test metrics with multiple agents."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("gpt4", won=True, generation=0)
        tracker.record_outcome("gemini", won=False, generation=0)

        metrics = tracker.get_generation_metrics(0)

        assert metrics["total_debates"] == 3
        assert metrics["wins"] == 2
        assert metrics["unique_agents"] == 3

    def test_get_metrics_isolates_generations(self, tracker):
        """Test that metrics are isolated by generation."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=True, generation=1)

        metrics0 = tracker.get_generation_metrics(0)
        metrics1 = tracker.get_generation_metrics(1)

        assert metrics0["total_debates"] == 1
        assert metrics1["total_debates"] == 2


# =============================================================================
# Performance Delta Tests
# =============================================================================


class TestPerformanceDelta:
    """Test performance delta calculations."""

    def test_delta_improvement(self, tracker):
        """Test delta showing improvement."""
        # Gen 0: 50% win rate
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=False, generation=0)

        # Gen 1: 75% win rate
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=False, generation=1)

        delta = tracker.get_performance_delta("claude", gen1=0, gen2=1)

        assert delta["agent"] == "claude"
        assert delta["gen1"] == 0
        assert delta["gen2"] == 1
        assert delta["gen1_win_rate"] == 0.5
        assert delta["gen2_win_rate"] == 0.75
        assert delta["win_rate_delta"] == 0.25
        assert delta["improved"] is True

    def test_delta_regression(self, tracker):
        """Test delta showing regression."""
        # Gen 0: 80% win rate
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=False, generation=0)

        # Gen 1: 50% win rate
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=False, generation=1)

        delta = tracker.get_performance_delta("claude", gen1=0, gen2=1)

        assert delta["gen1_win_rate"] == 0.8
        assert delta["gen2_win_rate"] == 0.5
        assert abs(delta["win_rate_delta"] - (-0.3)) < 0.001
        assert delta["improved"] is False

    def test_delta_no_change(self, tracker):
        """Test delta with no change."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=False, generation=0)
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=False, generation=1)

        delta = tracker.get_performance_delta("claude", gen1=0, gen2=1)

        assert delta["win_rate_delta"] == 0.0
        assert delta["improved"] is False

    def test_delta_empty_generations(self, tracker):
        """Test delta with empty generations."""
        delta = tracker.get_performance_delta("claude", gen1=0, gen2=1)

        assert delta["gen1_win_rate"] == 0.0
        assert delta["gen2_win_rate"] == 0.0
        assert delta["win_rate_delta"] == 0.0


# =============================================================================
# Agent List Tests
# =============================================================================


class TestGetAllAgents:
    """Test getting all agents list."""

    def test_get_all_agents_empty(self, tracker):
        """Test getting agents when none recorded."""
        agents = tracker.get_all_agents()
        assert agents == []

    def test_get_all_agents_single(self, tracker):
        """Test getting single agent."""
        tracker.record_outcome("claude", won=True)

        agents = tracker.get_all_agents()

        assert agents == ["claude"]

    def test_get_all_agents_multiple(self, tracker):
        """Test getting multiple agents."""
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("gpt4", won=False)
        tracker.record_outcome("gemini", won=True)

        agents = tracker.get_all_agents()

        assert set(agents) == {"claude", "gemini", "gpt4"}

    def test_get_all_agents_unique(self, tracker):
        """Test agents are unique."""
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("claude", won=True)
        tracker.record_outcome("claude", won=False)

        agents = tracker.get_all_agents()

        assert agents == ["claude"]

    def test_get_all_agents_sorted(self, tracker):
        """Test agents are sorted alphabetically."""
        tracker.record_outcome("zebra", won=True)
        tracker.record_outcome("alpha", won=True)
        tracker.record_outcome("mid", won=True)

        agents = tracker.get_all_agents()

        assert agents == ["alpha", "mid", "zebra"]


# =============================================================================
# Generation Trend Tests
# =============================================================================


class TestGenerationTrend:
    """Test generation trend retrieval."""

    def test_get_trend_empty(self, tracker):
        """Test trend for agent with no outcomes."""
        trend = tracker.get_generation_trend("nonexistent")
        assert trend == []

    def test_get_trend_single_generation(self, tracker):
        """Test trend with single generation."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=False, generation=0)

        trend = tracker.get_generation_trend("claude")

        assert len(trend) == 1
        assert trend[0]["generation"] == 0
        assert trend[0]["wins"] == 2
        assert trend[0]["total"] == 3
        assert abs(trend[0]["win_rate"] - 2 / 3) < 0.01

    def test_get_trend_multiple_generations(self, tracker):
        """Test trend with multiple generations."""
        # Gen 0
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=False, generation=0)

        # Gen 1
        tracker.record_outcome("claude", won=True, generation=1)
        tracker.record_outcome("claude", won=True, generation=1)

        # Gen 2
        tracker.record_outcome("claude", won=True, generation=2)

        trend = tracker.get_generation_trend("claude")

        assert len(trend) == 3
        assert trend[0]["generation"] == 0
        assert trend[1]["generation"] == 1
        assert trend[2]["generation"] == 2

    def test_get_trend_sorted(self, tracker):
        """Test trend is sorted by generation."""
        tracker.record_outcome("claude", won=True, generation=5)
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("claude", won=True, generation=3)

        trend = tracker.get_generation_trend("claude")

        generations = [t["generation"] for t in trend]
        assert generations == sorted(generations)

    def test_get_trend_max_generations(self, tracker):
        """Test trend respects max_generations limit."""
        for gen in range(20):
            tracker.record_outcome("claude", won=True, generation=gen)

        trend = tracker.get_generation_trend("claude", max_generations=5)

        assert len(trend) == 5

    def test_get_trend_isolates_agents(self, tracker):
        """Test trend is isolated per agent."""
        tracker.record_outcome("claude", won=True, generation=0)
        tracker.record_outcome("gpt4", won=True, generation=0)
        tracker.record_outcome("gpt4", won=True, generation=1)

        claude_trend = tracker.get_generation_trend("claude")
        gpt4_trend = tracker.get_generation_trend("gpt4")

        assert len(claude_trend) == 1
        assert len(gpt4_trend) == 2


# =============================================================================
# Data Persistence Tests
# =============================================================================


class TestDataPersistence:
    """Test data persistence across instances."""

    def test_data_persists(self, temp_db):
        """Test data persists when reopening database."""
        tracker1 = EvolutionTracker(db_path=temp_db)
        tracker1.record_outcome("claude", won=True, generation=0)
        tracker1.record_outcome("claude", won=True, generation=0)

        tracker2 = EvolutionTracker(db_path=temp_db)
        stats = tracker2.get_agent_stats("claude")

        assert stats["wins"] == 2
        assert stats["total"] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrackerIntegration:
    """Integration tests for EvolutionTracker."""

    def test_evolution_analysis_workflow(self, tracker):
        """Test typical evolution analysis workflow."""
        # Record outcomes for multiple generations
        agents = ["claude", "gpt4", "gemini"]
        generations = 5

        for gen in range(generations):
            for agent in agents:
                # Simulate improving performance over generations
                wins = gen + 1  # More wins in later generations
                losses = 2

                for _ in range(wins):
                    tracker.record_outcome(agent, won=True, generation=gen)
                for _ in range(losses):
                    tracker.record_outcome(agent, won=False, generation=gen)

        # Analyze results
        all_agents = tracker.get_all_agents()
        assert len(all_agents) == 3

        # Check improvement over generations
        for agent in agents:
            delta = tracker.get_performance_delta(agent, gen1=0, gen2=4)
            assert delta["improved"] is True
            assert delta["win_rate_delta"] > 0

        # Check generation metrics
        final_gen = tracker.get_generation_metrics(4)
        assert final_gen["total_debates"] == 21  # 3 agents * (5 wins + 2 losses)
        assert final_gen["unique_agents"] == 3

    def test_competing_agents_tracking(self, tracker):
        """Test tracking competing agents."""
        # Simulate a tournament
        matchups = [
            ("claude", "gpt4", "claude"),
            ("claude", "gemini", "claude"),
            ("gpt4", "gemini", "gpt4"),
            ("claude", "gpt4", "gpt4"),
            ("gemini", "claude", "gemini"),
        ]

        for agent1, agent2, winner in matchups:
            tracker.record_outcome(agent1, won=(winner == agent1))
            tracker.record_outcome(agent2, won=(winner == agent2))

        # Analyze rankings
        all_agents = tracker.get_all_agents()
        rankings = sorted(
            [(a, tracker.get_agent_stats(a)["win_rate"]) for a in all_agents],
            key=lambda x: x[1],
            reverse=True,
        )

        # Claude has 2/3 wins
        # GPT4 has 2/3 wins
        # Gemini has 1/3 wins
        assert rankings[2][0] == "gemini"  # Gemini is lowest

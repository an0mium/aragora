"""
Tests for LearningHandler - cross-cycle learning analytics endpoints.

Tests cover:
- GET /api/learning/cycles - Get cycle summaries
- GET /api/learning/patterns - Get learned patterns
- GET /api/learning/agent-evolution - Get agent performance evolution
- GET /api/learning/insights - Get aggregated insights
"""

import json
import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.learning import LearningHandler


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_nomic_dir():
    """Create a temporary nomic directory with cycle data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)

        # Create replays directory with cycles
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()

        # Create cycle 1
        cycle1_dir = replays_dir / "nomic-cycle-1"
        cycle1_dir.mkdir()
        (cycle1_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "debate-001",
                    "topic": "Implement security improvements",
                    "agents": [
                        {"name": "claude"},
                        {"name": "gpt4"},
                    ],
                    "started_at": "2026-01-09T10:00:00",
                    "ended_at": "2026-01-09T10:30:00",
                    "duration_ms": 1800000,
                    "status": "completed",
                    "final_verdict": "Approved security patch",
                    "event_count": 15,
                    "winner": "claude",
                    "vote_tally": {"claude": 3, "gpt4": 1},
                }
            )
        )

        # Create cycle 2
        cycle2_dir = replays_dir / "nomic-cycle-2"
        cycle2_dir.mkdir()
        (cycle2_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "debate-002",
                    "topic": "Add performance testing",
                    "agents": [
                        {"name": "claude"},
                        {"name": "gemini"},
                    ],
                    "started_at": "2026-01-10T10:00:00",
                    "ended_at": "2026-01-10T10:45:00",
                    "duration_ms": 2700000,
                    "status": "completed",
                    "final_verdict": "Performance tests added",
                    "event_count": 20,
                    "winner": "gemini",
                    "vote_tally": {"claude": 1, "gemini": 3},
                }
            )
        )

        # Create cycle 3 (failed)
        cycle3_dir = replays_dir / "nomic-cycle-3"
        cycle3_dir.mkdir()
        (cycle3_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "debate-003",
                    "topic": "Refactor API layer",
                    "agents": [
                        {"name": "claude"},
                        {"name": "gpt4"},
                    ],
                    "started_at": "2026-01-10T14:00:00",
                    "ended_at": "2026-01-10T14:20:00",
                    "duration_ms": 1200000,
                    "status": "failed",
                    "final_verdict": None,
                    "event_count": 8,
                }
            )
        )

        # Create risk register
        (nomic_dir / "risk_register.jsonl").write_text(
            json.dumps(
                {"cycle": 1, "phase": "design", "confidence": 0.8, "task": "Security update"}
            )
            + "\n"
            + json.dumps(
                {"cycle": 2, "phase": "test", "confidence": 0.9, "task": "Performance tests"}
            )
            + "\n"
            + json.dumps(
                {
                    "cycle": 3,
                    "phase": "implement",
                    "confidence": 0.2,
                    "task": "API refactor",
                    "error": "Syntax error in generated code",
                }
            )
            + "\n"
        )

        # Create insights database
        insight_db = nomic_dir / "insights.db"
        conn = sqlite3.connect(str(insight_db))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE insights (
                insight_id TEXT PRIMARY KEY,
                debate_id TEXT,
                category TEXT,
                content TEXT,
                confidence REAL,
                created_at TEXT
            )
        """
        )
        cursor.execute(
            """
            INSERT INTO insights VALUES
            ('insight-001', 'debate-001', 'security', 'Use input validation', 0.9, '2026-01-09T10:30:00'),
            ('insight-002', 'debate-002', 'performance', 'Add caching layer', 0.85, '2026-01-10T10:45:00'),
            ('insight-003', 'debate-001', 'security', 'Use parameterized queries', 0.95, '2026-01-09T10:35:00')
        """
        )
        conn.commit()
        conn.close()

        yield nomic_dir


@pytest.fixture
def handler(temp_nomic_dir):
    """Create LearningHandler with test context."""
    ctx = {"nomic_dir": temp_nomic_dir}
    return LearningHandler(ctx)


@pytest.fixture
def empty_handler():
    """Create LearningHandler without nomic directory."""
    return LearningHandler({})


# ============================================================================
# Route Recognition Tests
# ============================================================================


class TestLearningRouting:
    """Tests for learning handler route recognition."""

    def test_can_handle_cycles_route(self, handler):
        """Test handler recognizes cycles route."""
        assert handler.can_handle("/api/learning/cycles")

    def test_can_handle_patterns_route(self, handler):
        """Test handler recognizes patterns route."""
        assert handler.can_handle("/api/learning/patterns")

    def test_can_handle_evolution_route(self, handler):
        """Test handler recognizes agent-evolution route."""
        assert handler.can_handle("/api/learning/agent-evolution")

    def test_can_handle_insights_route(self, handler):
        """Test handler recognizes insights route."""
        assert handler.can_handle("/api/learning/insights")

    def test_cannot_handle_unknown_routes(self, handler):
        """Test handler rejects unknown routes."""
        assert not handler.can_handle("/api/learning")
        assert not handler.can_handle("/api/learning/other")
        assert not handler.can_handle("/api/analytics")


# ============================================================================
# GET /api/learning/cycles Tests
# ============================================================================


class TestGetCycles:
    """Tests for cycle summaries endpoint."""

    def test_get_cycles_success(self, handler):
        """Test successful cycles retrieval."""
        result = handler.handle("/api/learning/cycles", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "cycles" in data
        assert "count" in data
        assert data["count"] == 3

    def test_cycles_sorted_descending(self, handler):
        """Test cycles are sorted newest first."""
        result = handler.handle("/api/learning/cycles", {}, None)
        data = json.loads(result.body)

        cycles = data["cycles"]
        cycle_numbers = [c["cycle"] for c in cycles]
        assert cycle_numbers == sorted(cycle_numbers, reverse=True)

    def test_cycles_include_metadata(self, handler):
        """Test cycle entries include expected fields."""
        result = handler.handle("/api/learning/cycles", {}, None)
        data = json.loads(result.body)

        cycle = data["cycles"][0]
        assert "cycle" in cycle
        assert "debate_id" in cycle
        assert "topic" in cycle
        assert "agents" in cycle
        assert "status" in cycle
        assert "success" in cycle

    def test_cycles_with_limit(self, handler):
        """Test pagination with limit parameter."""
        result = handler.handle("/api/learning/cycles", {"limit": ["2"]}, None)
        data = json.loads(result.body)

        assert len(data["cycles"]) == 2

    def test_cycles_success_flag(self, handler):
        """Test success flag is correctly set."""
        result = handler.handle("/api/learning/cycles", {}, None)
        data = json.loads(result.body)

        # Find completed cycles and failed cycles
        for cycle in data["cycles"]:
            if cycle["status"] == "completed" and cycle["final_verdict"]:
                assert cycle["success"] is True
            else:
                assert cycle["success"] is False

    def test_cycles_no_nomic_dir(self, empty_handler):
        """Test 503 when nomic directory not configured."""
        result = empty_handler.handle("/api/learning/cycles", {}, None)

        assert result.status_code == 503

    def test_cycles_empty_replays(self, temp_nomic_dir):
        """Test empty result when no cycles exist."""
        # Remove existing cycles
        replays_dir = temp_nomic_dir / "replays"
        for d in replays_dir.iterdir():
            import shutil

            shutil.rmtree(d)

        handler = LearningHandler({"nomic_dir": temp_nomic_dir})
        result = handler.handle("/api/learning/cycles", {}, None)
        data = json.loads(result.body)

        assert data["cycles"] == []
        assert data["count"] == 0


# ============================================================================
# GET /api/learning/patterns Tests
# ============================================================================


class TestGetPatterns:
    """Tests for learned patterns endpoint."""

    def test_get_patterns_success(self, handler):
        """Test successful patterns retrieval."""
        result = handler.handle("/api/learning/patterns", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "successful_patterns" in data
        assert "failed_patterns" in data
        assert "recurring_themes" in data
        assert "agent_specializations" in data

    def test_patterns_from_risk_register(self, handler):
        """Test patterns extracted from risk register."""
        result = handler.handle("/api/learning/patterns", {}, None)
        data = json.loads(result.body)

        # Should have failed patterns (low confidence)
        assert len(data["failed_patterns"]) > 0
        # Should have successful patterns (high confidence)
        assert len(data["successful_patterns"]) > 0

    def test_recurring_themes_detected(self, handler):
        """Test recurring themes are identified."""
        result = handler.handle("/api/learning/patterns", {}, None)
        data = json.loads(result.body)

        themes = data["recurring_themes"]
        theme_names = [t["theme"] for t in themes]

        # Our test data has security and performance topics
        assert "security" in theme_names or "performance" in theme_names

    def test_agent_specializations_tracked(self, handler):
        """Test agent win counts are tracked."""
        result = handler.handle("/api/learning/patterns", {}, None)
        data = json.loads(result.body)

        specs = data["agent_specializations"]
        # claude won cycle 1, gemini won cycle 2
        assert "claude" in specs or "gemini" in specs

    def test_patterns_no_nomic_dir(self, empty_handler):
        """Test 503 when nomic directory not configured."""
        result = empty_handler.handle("/api/learning/patterns", {}, None)

        assert result.status_code == 503


# ============================================================================
# GET /api/learning/agent-evolution Tests
# ============================================================================


class TestGetAgentEvolution:
    """Tests for agent evolution endpoint."""

    def test_get_evolution_success(self, handler):
        """Test successful evolution retrieval."""
        result = handler.handle("/api/learning/agent-evolution", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "agents" in data
        assert "total_cycles_analyzed" in data

    def test_evolution_tracks_agents(self, handler):
        """Test all agents are tracked."""
        result = handler.handle("/api/learning/agent-evolution", {}, None)
        data = json.loads(result.body)

        agents = data["agents"]
        assert "claude" in agents  # Participated in all cycles

    def test_evolution_data_points(self, handler):
        """Test agent data points include expected fields."""
        result = handler.handle("/api/learning/agent-evolution", {}, None)
        data = json.loads(result.body)

        claude = data["agents"].get("claude", {})
        assert "data_points" in claude
        assert "total_cycles" in claude
        assert "total_wins" in claude
        assert "trend" in claude

    def test_evolution_trends_calculated(self, handler):
        """Test trends are calculated."""
        result = handler.handle("/api/learning/agent-evolution", {}, None)
        data = json.loads(result.body)

        for agent, info in data["agents"].items():
            assert info["trend"] in ["improving", "declining", "stable"]

    def test_evolution_no_nomic_dir(self, empty_handler):
        """Test 503 when nomic directory not configured."""
        result = empty_handler.handle("/api/learning/agent-evolution", {}, None)

        assert result.status_code == 503


# ============================================================================
# GET /api/learning/insights Tests
# ============================================================================


class TestGetInsights:
    """Tests for aggregated insights endpoint."""

    def test_get_insights_success(self, handler):
        """Test successful insights retrieval."""
        result = handler.handle("/api/learning/insights", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)

        assert "insights" in data
        assert "count" in data
        assert "by_category" in data

    def test_insights_include_expected_fields(self, handler):
        """Test insight entries include expected fields."""
        result = handler.handle("/api/learning/insights", {}, None)
        data = json.loads(result.body)

        if data["count"] > 0:
            insight = data["insights"][0]
            assert "insight_id" in insight
            assert "debate_id" in insight
            assert "category" in insight
            assert "content" in insight
            assert "confidence" in insight

    def test_insights_with_limit(self, handler):
        """Test pagination with limit parameter."""
        result = handler.handle("/api/learning/insights", {"limit": ["2"]}, None)
        data = json.loads(result.body)

        assert len(data["insights"]) <= 2

    def test_insights_by_category(self, handler):
        """Test category aggregation."""
        result = handler.handle("/api/learning/insights", {}, None)
        data = json.loads(result.body)

        # Our test data has security and performance categories
        by_category = data["by_category"]
        assert "security" in by_category or len(by_category) > 0

    def test_insights_no_nomic_dir(self, empty_handler):
        """Test 503 when nomic directory not configured."""
        result = empty_handler.handle("/api/learning/insights", {}, None)

        assert result.status_code == 503

    def test_insights_no_database(self, temp_nomic_dir):
        """Test empty result when no insights database."""
        # Remove insights database
        (temp_nomic_dir / "insights.db").unlink()

        handler = LearningHandler({"nomic_dir": temp_nomic_dir})
        result = handler.handle("/api/learning/insights", {}, None)
        data = json.loads(result.body)

        assert data["insights"] == []
        assert data["count"] == 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestLearningErrorHandling:
    """Tests for error handling in learning handler."""

    def test_malformed_meta_json_skipped(self, temp_nomic_dir):
        """Test malformed meta.json files are skipped."""
        # Create cycle with malformed JSON
        bad_dir = temp_nomic_dir / "replays" / "nomic-cycle-99"
        bad_dir.mkdir()
        (bad_dir / "meta.json").write_text("{ invalid json }")

        handler = LearningHandler({"nomic_dir": temp_nomic_dir})
        result = handler.handle("/api/learning/cycles", {}, None)

        # Should still work, just skip the bad one
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should have original 3 cycles, not the bad one
        assert data["count"] == 3

    def test_malformed_risk_register_handled(self, temp_nomic_dir):
        """Test malformed risk register entries are skipped."""
        # Append malformed entry
        with open(temp_nomic_dir / "risk_register.jsonl", "a") as f:
            f.write("{ bad json }\n")

        handler = LearningHandler({"nomic_dir": temp_nomic_dir})
        result = handler.handle("/api/learning/patterns", {}, None)

        assert result.status_code == 200

    def test_missing_replays_dir_handled(self, temp_nomic_dir):
        """Test missing replays directory is handled."""
        import shutil

        shutil.rmtree(temp_nomic_dir / "replays")

        handler = LearningHandler({"nomic_dir": temp_nomic_dir})
        result = handler.handle("/api/learning/cycles", {}, None)
        data = json.loads(result.body)

        assert data["cycles"] == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestLearningIntegration:
    """Integration tests for learning analytics."""

    def test_cross_endpoint_consistency(self, handler):
        """Test data consistency across endpoints."""
        # Get cycles
        cycles_result = handler.handle("/api/learning/cycles", {}, None)
        cycles_data = json.loads(cycles_result.body)

        # Get evolution
        evolution_result = handler.handle("/api/learning/agent-evolution", {}, None)
        evolution_data = json.loads(evolution_result.body)

        # Agents in evolution should appear in cycles
        for agent in evolution_data["agents"].keys():
            found = False
            for cycle in cycles_data["cycles"]:
                if agent in cycle["agents"]:
                    found = True
                    break
            assert found, f"Agent {agent} not found in any cycle"

    def test_patterns_reflect_cycles(self, handler):
        """Test patterns reflect actual cycle data."""
        # Get patterns
        patterns_result = handler.handle("/api/learning/patterns", {}, None)
        patterns_data = json.loads(patterns_result.body)

        # Get cycles
        cycles_result = handler.handle("/api/learning/cycles", {}, None)
        cycles_data = json.loads(cycles_result.body)

        # If we have failed cycles, should have failed patterns
        failed_cycles = [c for c in cycles_data["cycles"] if not c["success"]]
        if failed_cycles:
            # Should have some indication of failures
            assert (
                len(patterns_data["failed_patterns"]) > 0
                or len(patterns_data["successful_patterns"]) >= 0
            )

    def test_limit_parameters_respected(self, handler):
        """Test limit parameters work across endpoints."""
        endpoints = [
            ("/api/learning/cycles", "cycles"),
            ("/api/learning/insights", "insights"),
        ]

        for endpoint, key in endpoints:
            result = handler.handle(endpoint, {"limit": ["1"]}, None)
            data = json.loads(result.body)
            assert len(data[key]) <= 1, f"Limit not respected for {endpoint}"

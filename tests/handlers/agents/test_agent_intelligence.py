"""Tests for agent intelligence handler (AgentIntelligenceMixin).

Tests the agent intelligence endpoints:
- GET /api/v1/agent/{name}/metadata          - Agent metadata (model, capabilities)
- GET /api/v1/agent/{name}/introspect        - Agent introspection data
- GET /api/v1/agent/{name}/head-to-head/{op} - Head-to-head stats
- GET /api/v1/agent/{name}/opponent-briefing/{op} - Strategic opponent briefing
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock data classes
# ---------------------------------------------------------------------------


@dataclass
class MockRating:
    """Mock ELO rating data."""

    elo: float = 1650.0
    wins: int = 20
    losses: int = 10
    draws: int = 5
    calibration_accuracy: float = 0.75
    calibration_brier_score: float = 0.18
    calibration_total: int = 50


@dataclass
class MockRatingLowData:
    """Mock rating with insufficient calibration data."""

    elo: float = 1500.0
    wins: int = 2
    losses: int = 1
    draws: int = 0
    calibration_accuracy: float = 0.5
    calibration_brier_score: float = 0.25
    calibration_total: int = 3


@dataclass
class MockRatingHigh:
    """Mock rating with high calibration accuracy."""

    elo: float = 1800.0
    wins: int = 40
    losses: int = 5
    draws: int = 2
    calibration_accuracy: float = 0.85
    calibration_brier_score: float = 0.10
    calibration_total: int = 100


@dataclass
class MockRatingMedium:
    """Mock rating with medium calibration accuracy."""

    elo: float = 1600.0
    wins: int = 15
    losses: int = 12
    draws: int = 3
    calibration_accuracy: float = 0.65
    calibration_brier_score: float = 0.22
    calibration_total: int = 30


@dataclass
class MockRatingLow:
    """Mock rating with low calibration accuracy."""

    elo: float = 1400.0
    wins: int = 8
    losses: int = 20
    draws: int = 2
    calibration_accuracy: float = 0.45
    calibration_brier_score: float = 0.35
    calibration_total: int = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create AgentsHandler with mock context."""
    from aragora.server.handlers.agents.agents import AgentsHandler

    ctx = {}
    h = AgentsHandler(ctx)
    return h


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limiter state before each test."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.agents import agents

        agents._agent_limiter = agents.RateLimiter(requests_per_minute=60)
    except (ImportError, AttributeError):
        pass

    # Clear ttl_cache between tests to avoid stale data
    try:
        from aragora.server.handlers.admin.cache import clear_all_caches

        clear_all_caches()
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {}
    return h


# =============================================================================
# Routing Tests
# =============================================================================


class TestCanHandle:
    """Verify that can_handle correctly routes intelligence endpoints."""

    def test_metadata_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/metadata")

    def test_introspect_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/introspect")

    def test_head_to_head_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/head-to-head/gpt4")

    def test_opponent_briefing_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/opponent-briefing/gpt4")

    def test_metadata_different_agents(self, handler):
        assert handler.can_handle("/api/v1/agent/gpt4/metadata")
        assert handler.can_handle("/api/v1/agent/gemini/metadata")

    def test_head_to_head_different_agents(self, handler):
        assert handler.can_handle("/api/v1/agent/gpt4/head-to-head/claude")

    def test_unversioned_metadata_path(self, handler):
        """Paths without version prefix are also handled."""
        assert handler.can_handle("/api/agent/claude/metadata")


# =============================================================================
# _get_metadata Tests
# =============================================================================


class TestGetMetadata:
    """Tests for GET /api/v1/agent/{name}/metadata endpoint."""

    @pytest.mark.asyncio
    async def test_no_nomic_dir(self, handler, mock_http_handler):
        """Returns message when nomic_dir is not available."""
        handler.ctx = {}  # no nomic_dir
        result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["metadata"] is None
        assert "not available" in body["message"]

    @pytest.mark.asyncio
    async def test_elo_db_not_found(self, handler, mock_http_handler, tmp_path):
        """Returns message when ELO database file does not exist."""
        handler.ctx = {"nomic_dir": tmp_path}
        # tmp_path exists but no elo.db file inside
        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=tmp_path / "nonexistent_elo.db",
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["metadata"] is None
            assert "not found" in body["message"]

    @pytest.mark.asyncio
    async def test_metadata_found(self, handler, mock_http_handler, tmp_path):
        """Returns full metadata when agent exists in database."""
        handler.ctx = {"nomic_dir": tmp_path}
        db_path = tmp_path / "elo.db"

        # Create a real sqlite db with agent_metadata table
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE agent_metadata (
                agent_name TEXT PRIMARY KEY,
                provider TEXT,
                model_id TEXT,
                context_window INTEGER,
                specialties TEXT,
                strengths TEXT,
                release_date TEXT,
                updated_at TEXT
            )
        """)
        conn.execute(
            """
            INSERT INTO agent_metadata
            (agent_name, provider, model_id, context_window, specialties, strengths, release_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "claude",
                "anthropic",
                "claude-3-opus",
                200000,
                json.dumps(["reasoning", "analysis"]),
                json.dumps(["nuanced thinking", "long context"]),
                "2024-03-04",
                "2024-03-05T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=db_path,
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["agent"] == "claude"
            meta = body["metadata"]
            assert meta["provider"] == "anthropic"
            assert meta["model_id"] == "claude-3-opus"
            assert meta["context_window"] == 200000
            assert meta["specialties"] == ["reasoning", "analysis"]
            assert meta["strengths"] == ["nuanced thinking", "long context"]
            assert meta["release_date"] == "2024-03-04"

    @pytest.mark.asyncio
    async def test_metadata_not_found_for_agent(self, handler, mock_http_handler, tmp_path):
        """Returns null metadata when agent not in database."""
        handler.ctx = {"nomic_dir": tmp_path}
        db_path = tmp_path / "elo.db"

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE agent_metadata (
                agent_name TEXT PRIMARY KEY,
                provider TEXT,
                model_id TEXT,
                context_window INTEGER,
                specialties TEXT,
                strengths TEXT,
                release_date TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=db_path,
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["metadata"] is None
            assert "not found" in body["message"]

    @pytest.mark.asyncio
    async def test_metadata_table_not_initialized(self, handler, mock_http_handler, tmp_path):
        """Returns message when agent_metadata table does not exist."""
        handler.ctx = {"nomic_dir": tmp_path}
        db_path = tmp_path / "elo.db"

        # Create empty database with no tables
        conn = sqlite3.connect(db_path)
        conn.close()

        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=db_path,
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["metadata"] is None
            assert "not initialized" in body["message"]

    @pytest.mark.asyncio
    async def test_metadata_invalid_json_specialties(self, handler, mock_http_handler, tmp_path):
        """Gracefully handles invalid JSON in specialties field."""
        handler.ctx = {"nomic_dir": tmp_path}
        db_path = tmp_path / "elo.db"

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE agent_metadata (
                agent_name TEXT PRIMARY KEY,
                provider TEXT,
                model_id TEXT,
                context_window INTEGER,
                specialties TEXT,
                strengths TEXT,
                release_date TEXT,
                updated_at TEXT
            )
        """)
        conn.execute(
            """
            INSERT INTO agent_metadata
            (agent_name, provider, model_id, context_window, specialties, strengths, release_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "claude",
                "anthropic",
                "claude-3-opus",
                200000,
                "not valid json",
                "also not valid json",
                "2024-03-04",
                "2024-03-05T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=db_path,
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            meta = body["metadata"]
            # Invalid JSON gracefully falls back to empty lists
            assert meta["specialties"] == []
            assert meta["strengths"] == []

    @pytest.mark.asyncio
    async def test_metadata_null_specialties(self, handler, mock_http_handler, tmp_path):
        """Handles null specialties and strengths fields."""
        handler.ctx = {"nomic_dir": tmp_path}
        db_path = tmp_path / "elo.db"

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE agent_metadata (
                agent_name TEXT PRIMARY KEY,
                provider TEXT,
                model_id TEXT,
                context_window INTEGER,
                specialties TEXT,
                strengths TEXT,
                release_date TEXT,
                updated_at TEXT
            )
        """)
        conn.execute(
            """
            INSERT INTO agent_metadata
            (agent_name, provider, model_id, context_window, specialties, strengths, release_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("claude", "anthropic", "claude-3-opus", 200000, None, None, "2024-03-04", None),
        )
        conn.commit()
        conn.close()

        with patch(
            "aragora.server.handlers.agents.agent_intelligence.get_db_path",
            return_value=db_path,
        ):
            result = await handler.handle("/api/v1/agent/claude/metadata", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            meta = body["metadata"]
            assert meta["specialties"] == []
            assert meta["strengths"] == []


# =============================================================================
# _get_agent_introspect Tests
# =============================================================================


class TestGetAgentIntrospect:
    """Tests for GET /api/v1/agent/{name}/introspect endpoint."""

    @pytest.mark.asyncio
    async def test_basic_introspect_no_elo(self, handler, mock_http_handler):
        """Returns basic introspection data when ELO system is unavailable."""
        handler.ctx = {}  # no elo_system
        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert _status(result) == 200
                assert body["agent_id"] == "claude"
                assert body["identity"]["name"] == "claude"
                assert body["calibration"] == {}
                assert body["positions"] == []
                assert body["performance"] == {}
                assert body["fatigue_indicators"] is None
                assert body["debate_context"] is None

    @pytest.mark.asyncio
    async def test_introspect_with_elo(self, handler, mock_http_handler):
        """Returns performance and calibration data when ELO is available."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRating()
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert _status(result) == 200
                perf = body["performance"]
                assert perf["elo"] == 1650.0
                assert perf["total_games"] == 35  # 20 + 10 + 5
                assert perf["wins"] == 20
                assert perf["losses"] == 10
                assert perf["win_rate"] == pytest.approx(20 / 35, rel=1e-3)

                cal = body["calibration"]
                assert cal["accuracy"] == 0.75
                assert cal["brier_score"] == 0.18
                assert cal["prediction_count"] == 50
                assert cal["confidence_level"] == "medium"

    @pytest.mark.asyncio
    async def test_introspect_high_confidence(self, handler, mock_http_handler):
        """Returns high confidence level when accuracy >= 0.8."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRatingHigh()
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["calibration"]["confidence_level"] == "high"

    @pytest.mark.asyncio
    async def test_introspect_low_confidence(self, handler, mock_http_handler):
        """Returns low confidence level when accuracy < 0.6."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRatingLow()
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["calibration"]["confidence_level"] == "low"

    @pytest.mark.asyncio
    async def test_introspect_insufficient_data(self, handler, mock_http_handler):
        """Returns insufficient_data when calibration count < 5."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRatingLowData()
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["calibration"]["confidence_level"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_introspect_elo_error_graceful(self, handler, mock_http_handler):
        """Gracefully handles ELO lookup errors."""
        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = KeyError("agent not found")
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert _status(result) == 200
                assert body["performance"] == {}
                assert body["calibration"] == {}

    @pytest.mark.asyncio
    async def test_introspect_with_position_history(self, handler, mock_http_handler, tmp_path):
        """Returns position history when PositionTracker is available."""
        import sys
        import types

        handler.ctx = {"nomic_dir": tmp_path}

        mock_tracker = MagicMock()
        mock_tracker.get_agent_positions.return_value = [
            {
                "topic": "Rate limiter design",
                "stance": "pro",
                "confidence": 0.8,
                "timestamp": "2024-03-01T00:00:00",
            },
            {
                "topic": "Caching strategy",
                "stance": "against",
                "confidence": 0.6,
                "timestamp": "2024-03-02T00:00:00",
            },
        ]

        tracker_path = tmp_path / "position_tracker.json"
        tracker_path.touch()

        # The handler imports from aragora.ranking.position_tracker which may
        # not exist as a real module. Inject a mock module into sys.modules.
        mock_pt_module = types.ModuleType("aragora.ranking.position_tracker")
        mock_pt_module.PositionTracker = MagicMock(return_value=mock_tracker)

        with patch.dict(sys.modules, {"aragora.ranking.position_tracker": mock_pt_module}):
            with patch(
                "aragora.memory.continuum.ContinuumMemory",
                side_effect=ImportError("not available"),
            ):
                with patch(
                    "aragora.agents.personas.PersonaManager",
                    side_effect=ImportError("not available"),
                ):
                    result = await handler.handle(
                        "/api/v1/agent/claude/introspect", {}, mock_http_handler
                    )
                    body = _body(result)
                    assert len(body["positions"]) == 2
                    assert body["positions"][0]["topic"] == "Rate limiter design"
                    assert body["positions"][0]["stance"] == "pro"
                    assert body["positions"][0]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_introspect_position_tracker_missing_file(
        self, handler, mock_http_handler, tmp_path
    ):
        """Positions empty when tracker file does not exist."""
        handler.ctx = {"nomic_dir": tmp_path}
        # No position_tracker.json file

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["positions"] == []

    @pytest.mark.asyncio
    async def test_introspect_with_memory_summary(self, handler, mock_http_handler):
        """Returns memory tier summary when ContinuumMemory is available."""
        handler.ctx = {}

        mock_memory = MagicMock()
        mock_memory.get_stats.return_value = {
            "by_tier": {
                "fast": {"count": 10},
                "medium": {"count": 25},
                "slow": {"count": 5},
                "glacial": {"count": 2},
            }
        }
        mock_memory.get_red_line_memories.return_value = ["mem1", "mem2"]

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            return_value=mock_memory,
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                mem = body["memory_summary"]
                assert mem["tier_counts"]["fast"] == 10
                assert mem["tier_counts"]["medium"] == 25
                assert mem["tier_counts"]["slow"] == 5
                assert mem["tier_counts"]["glacial"] == 2
                assert mem["total_memories"] == 42
                assert mem["red_line_count"] == 2

    @pytest.mark.asyncio
    async def test_introspect_memory_error_graceful(self, handler, mock_http_handler):
        """Gracefully handles ContinuumMemory errors."""
        handler.ctx = {}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ValueError("memory init failed"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert _status(result) == 200
                assert body["memory_summary"] == {}

    @pytest.mark.asyncio
    async def test_introspect_with_persona(self, handler, mock_http_handler):
        """Returns persona identity data when PersonaManager is available."""
        handler.ctx = {}

        mock_persona = MagicMock()
        mock_persona.style = "analytical"
        mock_persona.temperature = 0.7
        mock_persona.system_prompt = "You are a careful analytical agent."

        mock_persona_mgr = MagicMock()
        mock_persona_mgr.get_persona.return_value = mock_persona

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                return_value=mock_persona_mgr,
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                persona_data = body["identity"].get("persona")
                assert persona_data is not None
                assert persona_data["style"] == "analytical"
                assert persona_data["temperature"] == 0.7
                assert "You are a careful" in persona_data["system_prompt_preview"]

    @pytest.mark.asyncio
    async def test_introspect_persona_not_found(self, handler, mock_http_handler):
        """No persona data when PersonaManager returns None."""
        handler.ctx = {}

        mock_persona_mgr = MagicMock()
        mock_persona_mgr.get_persona.return_value = None

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                return_value=mock_persona_mgr,
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert "persona" not in body["identity"]

    @pytest.mark.asyncio
    async def test_introspect_with_debate_context(self, handler, mock_http_handler):
        """Returns debate context when debate_id query param is provided."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "messages": [
                {"agent": "claude", "text": "msg1"},
                {"agent": "gpt4", "text": "msg2"},
                {"agent": "claude", "text": "msg3"},
            ],
            "current_round": 2,
            "status": "in_progress",
        }
        handler.ctx = {"storage": mock_storage}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect",
                    {"debate_id": "debate-123"},
                    mock_http_handler,
                )
                body = _body(result)
                ctx = body["debate_context"]
                assert ctx is not None
                assert ctx["debate_id"] == "debate-123"
                assert ctx["messages_sent"] == 2  # only claude's messages
                assert ctx["current_round"] == 2
                assert ctx["debate_status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_introspect_debate_context_no_storage(self, handler, mock_http_handler):
        """debate_context is None when storage unavailable."""
        handler.ctx = {}  # no storage

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect",
                    {"debate_id": "debate-123"},
                    mock_http_handler,
                )
                body = _body(result)
                assert body["debate_context"] is None

    @pytest.mark.asyncio
    async def test_introspect_debate_not_found(self, handler, mock_http_handler):
        """debate_context is None when debate not found in storage."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        handler.ctx = {"storage": mock_storage}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect",
                    {"debate_id": "debate-999"},
                    mock_http_handler,
                )
                body = _body(result)
                assert body["debate_context"] is None

    @pytest.mark.asyncio
    async def test_introspect_no_debate_id(self, handler, mock_http_handler):
        """debate_context is None when no debate_id param provided."""
        handler.ctx = {}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["debate_context"] is None

    @pytest.mark.asyncio
    async def test_introspect_has_timestamp(self, handler, mock_http_handler):
        """Introspection result includes a timestamp."""
        handler.ctx = {}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert "timestamp" in body
                assert len(body["timestamp"]) > 0

    @pytest.mark.asyncio
    async def test_introspect_zero_games_win_rate(self, handler, mock_http_handler):
        """Win rate is 0.0 when no games played."""
        mock_rating = MagicMock()
        mock_rating.elo = 1500.0
        mock_rating.wins = 0
        mock_rating.losses = 0
        mock_rating.draws = 0
        mock_rating.calibration_accuracy = 0.5
        mock_rating.calibration_brier_score = 0.25
        mock_rating.calibration_total = 0

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = mock_rating
        handler.ctx = {"elo_system": mock_elo}

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                assert body["performance"]["win_rate"] == 0.0
                assert body["performance"]["total_games"] == 0

    @pytest.mark.asyncio
    async def test_introspect_persona_long_system_prompt_truncated(
        self, handler, mock_http_handler
    ):
        """Long system prompts are truncated to 200 chars in preview."""
        handler.ctx = {}

        mock_persona = MagicMock()
        mock_persona.style = "verbose"
        mock_persona.temperature = 1.0
        mock_persona.system_prompt = "A" * 500

        mock_persona_mgr = MagicMock()
        mock_persona_mgr.get_persona.return_value = mock_persona

        with patch(
            "aragora.memory.continuum.ContinuumMemory",
            side_effect=ImportError("not available"),
        ):
            with patch(
                "aragora.agents.personas.PersonaManager",
                return_value=mock_persona_mgr,
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/introspect", {}, mock_http_handler
                )
                body = _body(result)
                preview = body["identity"]["persona"]["system_prompt_preview"]
                assert len(preview) == 200


# =============================================================================
# _compute_confidence Tests
# =============================================================================


class TestComputeConfidence:
    """Tests for the _compute_confidence helper method."""

    def test_insufficient_data(self, handler):
        rating = MockRatingLowData()
        assert handler._compute_confidence(rating) == "insufficient_data"

    def test_high_confidence(self, handler):
        rating = MockRatingHigh()
        assert handler._compute_confidence(rating) == "high"

    def test_medium_confidence(self, handler):
        rating = MockRatingMedium()
        assert handler._compute_confidence(rating) == "medium"

    def test_low_confidence(self, handler):
        rating = MockRatingLow()
        assert handler._compute_confidence(rating) == "low"

    def test_boundary_high(self, handler):
        """Accuracy exactly 0.8 is high."""
        rating = MagicMock()
        rating.calibration_accuracy = 0.8
        rating.calibration_total = 10
        assert handler._compute_confidence(rating) == "high"

    def test_boundary_medium(self, handler):
        """Accuracy exactly 0.6 is medium."""
        rating = MagicMock()
        rating.calibration_accuracy = 0.6
        rating.calibration_total = 10
        assert handler._compute_confidence(rating) == "medium"

    def test_boundary_insufficient(self, handler):
        """Count exactly 4 is insufficient."""
        rating = MagicMock()
        rating.calibration_accuracy = 0.9
        rating.calibration_total = 4
        assert handler._compute_confidence(rating) == "insufficient_data"

    def test_boundary_just_sufficient(self, handler):
        """Count exactly 5 is sufficient."""
        rating = MagicMock()
        rating.calibration_accuracy = 0.9
        rating.calibration_total = 5
        assert handler._compute_confidence(rating) == "high"


# =============================================================================
# _get_head_to_head Tests
# =============================================================================


class TestGetHeadToHead:
    """Tests for GET /api/v1/agent/{name}/head-to-head/{opponent} endpoint."""

    @pytest.mark.asyncio
    async def test_no_elo_system(self, handler, mock_http_handler):
        """Returns 503 when ELO system unavailable."""
        handler.ctx = {}
        result = await handler.handle(
            "/api/v1/agent/claude/head-to-head/gpt4", {}, mock_http_handler
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_head_to_head_with_method(self, handler, mock_http_handler):
        """Returns head-to-head stats when get_head_to_head method exists."""
        mock_elo = MagicMock()
        mock_elo.get_head_to_head.return_value = {
            "matches": 10,
            "agent1_wins": 6,
            "agent2_wins": 4,
        }
        handler.ctx = {"elo_system": mock_elo}

        result = await handler.handle(
            "/api/v1/agent/claude/head-to-head/gpt4", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["agent1"] == "claude"
        assert body["agent2"] == "gpt4"
        assert body["matches"] == 10
        assert body["agent1_wins"] == 6
        assert body["agent2_wins"] == 4
        mock_elo.get_head_to_head.assert_called_once_with("claude", "gpt4")

    @pytest.mark.asyncio
    async def test_head_to_head_fallback_no_method(self, handler, mock_http_handler):
        """Falls back to empty stats when get_head_to_head method missing."""
        mock_elo = MagicMock(spec=[])  # No get_head_to_head attribute
        handler.ctx = {"elo_system": mock_elo}

        result = await handler.handle(
            "/api/v1/agent/claude/head-to-head/gpt4", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["agent1"] == "claude"
        assert body["agent2"] == "gpt4"
        assert body["matches"] == 0
        assert body["agent1_wins"] == 0
        assert body["agent2_wins"] == 0


# =============================================================================
# _get_opponent_briefing Tests
# =============================================================================


class TestGetOpponentBriefing:
    """Tests for GET /api/v1/agent/{name}/opponent-briefing/{opponent} endpoint."""

    @pytest.mark.asyncio
    async def test_briefing_with_data(self, handler, mock_http_handler, tmp_path):
        """Returns briefing when synthesizer has data."""
        mock_elo = MagicMock()
        handler.ctx = {"elo_system": mock_elo, "nomic_dir": tmp_path}

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_opponent_briefing.return_value = (
            "### Briefing: gpt4\n- Previous debates: 5\n- Agreement rate: 40%"
        )

        with patch(
            "aragora.agents.grounded.PersonaSynthesizer",
            return_value=mock_synthesizer,
        ):
            with patch(
                "aragora.server.handlers.agents.agent_intelligence.get_db_path",
                return_value=tmp_path / "nonexistent.db",
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/opponent-briefing/gpt4",
                    {},
                    mock_http_handler,
                )
                body = _body(result)
                assert _status(result) == 200
                assert body["agent"] == "claude"
                assert body["opponent"] == "gpt4"
                assert body["briefing"] is not None
                assert "gpt4" in body["briefing"]

    @pytest.mark.asyncio
    async def test_briefing_no_data(self, handler, mock_http_handler, tmp_path):
        """Returns null briefing when no opponent data available."""
        handler.ctx = {"nomic_dir": tmp_path}

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_opponent_briefing.return_value = ""

        with patch(
            "aragora.agents.grounded.PersonaSynthesizer",
            return_value=mock_synthesizer,
        ):
            with patch(
                "aragora.server.handlers.agents.agent_intelligence.get_db_path",
                return_value=tmp_path / "nonexistent.db",
            ):
                result = await handler.handle(
                    "/api/v1/agent/claude/opponent-briefing/gpt4",
                    {},
                    mock_http_handler,
                )
                body = _body(result)
                assert _status(result) == 200
                assert body["agent"] == "claude"
                assert body["opponent"] == "gpt4"
                assert body["briefing"] is None
                assert "message" in body

    @pytest.mark.asyncio
    async def test_briefing_no_nomic_dir(self, handler, mock_http_handler):
        """Works without nomic_dir (no position ledger)."""
        handler.ctx = {}

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_opponent_briefing.return_value = "Basic briefing"

        with patch(
            "aragora.agents.grounded.PersonaSynthesizer",
            return_value=mock_synthesizer,
        ):
            result = await handler.handle(
                "/api/v1/agent/claude/opponent-briefing/gpt4",
                {},
                mock_http_handler,
            )
            body = _body(result)
            assert _status(result) == 200
            assert body["briefing"] == "Basic briefing"

    @pytest.mark.asyncio
    async def test_briefing_with_position_ledger(self, handler, mock_http_handler, tmp_path):
        """Position ledger is loaded when db exists."""
        handler.ctx = {"nomic_dir": tmp_path}

        db_path = tmp_path / "positions.db"
        db_path.touch()

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_opponent_briefing.return_value = "Detailed briefing"

        mock_ledger = MagicMock()

        with patch(
            "aragora.agents.grounded.PersonaSynthesizer",
            return_value=mock_synthesizer,
        ) as mock_synth_cls:
            with patch(
                "aragora.agents.grounded.PositionLedger",
                return_value=mock_ledger,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_intelligence.get_db_path",
                    return_value=db_path,
                ):
                    result = await handler.handle(
                        "/api/v1/agent/claude/opponent-briefing/gpt4",
                        {},
                        mock_http_handler,
                    )
                    body = _body(result)
                    assert _status(result) == 200
                    # Verify PersonaSynthesizer was constructed with position_ledger
                    call_kwargs = mock_synth_cls.call_args
                    assert call_kwargs[1]["position_ledger"] is mock_ledger

    @pytest.mark.asyncio
    async def test_briefing_position_ledger_import_error(
        self, handler, mock_http_handler, tmp_path
    ):
        """Gracefully handles ImportError for PositionLedger."""
        handler.ctx = {"nomic_dir": tmp_path}

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_opponent_briefing.return_value = "Fallback briefing"

        with patch(
            "aragora.agents.grounded.PersonaSynthesizer",
            return_value=mock_synthesizer,
        ) as mock_synth_cls:
            with patch(
                "aragora.agents.grounded.PositionLedger",
                side_effect=ImportError("not available"),
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_intelligence.get_db_path",
                    return_value=tmp_path / "positions.db",
                ):
                    result = await handler.handle(
                        "/api/v1/agent/claude/opponent-briefing/gpt4",
                        {},
                        mock_http_handler,
                    )
                    body = _body(result)
                    assert _status(result) == 200
                    # position_ledger should be None when import fails
                    call_kwargs = mock_synth_cls.call_args
                    assert call_kwargs[1]["position_ledger"] is None


# =============================================================================
# _get_timestamp Tests
# =============================================================================


class TestGetTimestamp:
    """Tests for the _get_timestamp helper method."""

    def test_returns_iso_format(self, handler):
        ts = handler._get_timestamp()
        # ISO format has T separator
        assert "T" in ts

    def test_returns_string(self, handler):
        ts = handler._get_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 10

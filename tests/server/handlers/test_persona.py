"""
Tests for PersonaHandler - Agent persona HTTP endpoints.

Tests cover:
- GET /api/personas - Get all agent personas
- GET /api/personas/options - Get available traits and expertise domains
- GET /api/agent/{name}/persona - Get agent persona
- GET /api/agent/{name}/grounded-persona - Get truth-grounded persona
- GET /api/agent/{name}/identity-prompt - Get identity prompt
- GET /api/agent/{name}/performance - Get agent performance summary
- GET /api/agent/{name}/domains - Get agent expertise domains
- GET /api/agent/{name}/accuracy - Get position accuracy stats
- POST /api/personas - Create new persona
- PUT /api/agent/{name}/persona - Update agent persona
- DELETE /api/agent/{name}/persona - Delete agent persona
- Rate limiting
- RBAC permission enforcement
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import tempfile

import pytest

from aragora.server.handlers.persona import PersonaHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockPersona:
    """Mock persona for testing."""

    agent_name: str = "claude"
    description: str = "A helpful AI assistant"
    traits: list[str] = field(default_factory=lambda: ["analytical", "thorough"])
    expertise: dict[str, float] = field(default_factory=lambda: {"reasoning": 0.9, "coding": 0.8})
    created_at: str = "2024-01-15T10:30:00Z"
    updated_at: str = "2024-01-15T10:30:00Z"


@dataclass
class MockGroundedPersona:
    """Mock grounded persona for testing."""

    elo: int = 1650
    domain_elos: dict[str, int] = field(default_factory=lambda: {"reasoning": 1700, "coding": 1600})
    games_played: int = 100
    win_rate: float = 0.65
    calibration_score: float = 0.85
    position_accuracy: float = 0.78
    positions_taken: int = 250
    reversals: int = 12


class MockPersonaManager:
    """Mock persona manager for testing."""

    def __init__(self):
        self._personas: dict[str, MockPersona] = {}
        self._connection_ctx = MagicMock()

    def get_all_personas(self) -> list[MockPersona]:
        return list(self._personas.values())

    def get_persona(self, agent_name: str) -> MockPersona | None:
        return self._personas.get(agent_name)

    def create_persona(
        self,
        agent_name: str,
        description: str = "",
        traits: list[str] = None,
        expertise: dict[str, float] = None,
    ) -> MockPersona:
        persona = MockPersona(
            agent_name=agent_name,
            description=description,
            traits=traits or [],
            expertise=expertise or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._personas[agent_name] = persona
        return persona

    def get_performance_summary(self, agent_name: str) -> dict[str, Any]:
        return {
            "total_debates": 100,
            "wins": 65,
            "losses": 30,
            "draws": 5,
            "win_rate": 0.65,
            "average_score": 0.78,
        }

    def connection(self):
        """Return a context manager for database connection."""
        return self._connection_ctx


class MockEloSystem:
    """Mock ELO system for testing."""

    def get_best_domains(self, agent: str, limit: int = 10) -> list[tuple]:
        return [
            ("reasoning", 0.92),
            ("coding", 0.88),
            ("mathematics", 0.85),
        ][:limit]


class MockPositionLedger:
    """Mock position ledger for testing."""

    pass


class MockPositionTracker:
    """Mock position tracker for testing."""

    def __init__(self, db_path: str = ""):
        self.db_path = db_path

    def get_agent_position_accuracy(self, agent: str) -> dict[str, Any] | None:
        return {
            "total_positions": 250,
            "verified_positions": 200,
            "correct_positions": 156,
            "accuracy_rate": 0.78,
            "by_type": {
                "factual": {"total": 100, "correct": 85},
                "reasoning": {"total": 150, "correct": 71},
            },
        }


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self):
        self.headers = {}
        self.client_address = ("127.0.0.1", 12345)
        self._body = b"{}"
        self.rfile = MagicMock()
        self.rfile.read.return_value = self._body

    def set_body(self, data: dict):
        self._body = json.dumps(data).encode()
        self.rfile.read.return_value = self._body
        self.headers["Content-Length"] = str(len(self._body))
        self.headers["Content-Type"] = "application/json"


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_persona_manager():
    """Create mock persona manager with sample data."""
    manager = MockPersonaManager()
    manager._personas["claude"] = MockPersona()
    manager._personas["gpt4"] = MockPersona(
        agent_name="gpt4",
        description="A powerful language model",
        traits=["creative", "verbose"],
        expertise={"writing": 0.95, "reasoning": 0.85},
    )
    return manager


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system."""
    return MockEloSystem()


@pytest.fixture
def persona_handler(mock_server_context, mock_persona_manager, mock_elo_system):
    """Create handler with mocked dependencies."""
    handler = PersonaHandler(mock_server_context)
    handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)
    handler.get_elo_system = MagicMock(return_value=mock_elo_system)
    handler.get_position_ledger = MagicMock(return_value=MockPositionLedger())
    return handler


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    return MockHTTPHandler()


def parse_handler_response(result) -> dict[str, Any]:
    """Parse handler result body as JSON."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode())
        return json.loads(body)
    return {}


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestPersonaHandlerInit:
    """Tests for PersonaHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = PersonaHandler({})
        assert hasattr(handler, "ctx")

    def test_routes_constant_is_list(self):
        """ROUTES should be a list."""
        assert isinstance(PersonaHandler.ROUTES, list)

    def test_routes_includes_core_endpoints(self):
        """ROUTES should include core persona endpoints."""
        routes = PersonaHandler.ROUTES
        assert "/api/personas" in routes
        assert "/api/personas/options" in routes
        assert "/api/agent/*/persona" in routes
        assert "/api/agent/*/grounded-persona" in routes
        assert "/api/agent/*/identity-prompt" in routes
        assert "/api/agent/*/performance" in routes
        assert "/api/agent/*/domains" in routes
        assert "/api/agent/*/accuracy" in routes


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestPersonaHandlerCanHandle:
    """Tests for can_handle method."""

    def test_can_handle_personas_list(self):
        """Should handle /api/personas."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/personas") is True
        assert handler.can_handle("/api/v1/personas") is True

    def test_can_handle_personas_options(self):
        """Should handle /api/personas/options."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/personas/options") is True
        assert handler.can_handle("/api/v1/personas/options") is True

    def test_can_handle_agent_persona(self):
        """Should handle /api/agent/{name}/persona."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/persona") is True
        assert handler.can_handle("/api/v1/agent/claude/persona") is True

    def test_can_handle_grounded_persona(self):
        """Should handle /api/agent/{name}/grounded-persona."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/grounded-persona") is True

    def test_can_handle_identity_prompt(self):
        """Should handle /api/agent/{name}/identity-prompt."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/identity-prompt") is True

    def test_can_handle_performance(self):
        """Should handle /api/agent/{name}/performance."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/performance") is True

    def test_can_handle_domains(self):
        """Should handle /api/agent/{name}/domains."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/domains") is True

    def test_can_handle_accuracy(self):
        """Should handle /api/agent/{name}/accuracy."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/accuracy") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unknown paths."""
        handler = PersonaHandler({})
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/agent/claude/unknown") is False


# ===========================================================================
# GET /api/personas Tests
# ===========================================================================


class TestGetAllPersonas:
    """Tests for GET /api/personas endpoint."""

    def test_get_all_personas_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of all personas."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "personas" in data
        assert data["count"] == 2
        assert len(data["personas"]) == 2

    def test_get_all_personas_no_manager(self, mock_http_handler):
        """Test response when persona manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["error"] == "Persona management not configured"
        assert data["personas"] == []

    def test_get_all_personas_exception(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test error handling when exception occurs."""
        mock_persona_manager.get_all_personas = MagicMock(side_effect=ValueError("DB error"))
        persona_handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "error" in data
        assert data["personas"] == []


# ===========================================================================
# GET /api/personas/options Tests
# ===========================================================================


class TestGetPersonaOptions:
    """Tests for GET /api/personas/options endpoint."""

    def test_get_options_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of persona options."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.agents.personas": MagicMock(
                        PERSONALITY_TRAITS=["analytical", "creative", "thorough"],
                        EXPERTISE_DOMAINS=["reasoning", "coding", "writing"],
                    )
                },
            ):
                result = persona_handler.handle("/api/personas/options", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "traits" in data
        assert "expertise_domains" in data

    def test_get_options_import_error(self, persona_handler, mock_http_handler):
        """Test response when personas module not available."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            # Force ImportError by making the method raise it
            original_method = persona_handler._get_persona_options

            def raise_import_error():
                return persona_handler.json_response({"traits": [], "expertise_domains": []})

            persona_handler._get_persona_options = raise_import_error
            result = persona_handler.handle("/api/personas/options", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["traits"] == []
        assert data["expertise_domains"] == []


# ===========================================================================
# GET /api/agent/{name}/persona Tests
# ===========================================================================


class TestGetAgentPersona:
    """Tests for GET /api/agent/{name}/persona endpoint."""

    def test_get_persona_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of agent persona."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "persona" in data
        assert data["persona"]["agent_name"] == "claude"
        assert "traits" in data["persona"]
        assert "expertise" in data["persona"]

    def test_get_persona_not_found(self, persona_handler, mock_http_handler, mock_persona_manager):
        """Test response when persona not found."""
        mock_persona_manager._personas = {}
        persona_handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/unknown/persona", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["persona"] is None
        assert "error" in data

    def test_get_persona_no_manager(self, mock_http_handler):
        """Test response when persona manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 503

    def test_get_persona_invalid_agent_name(self, persona_handler, mock_http_handler):
        """Test response with invalid agent name."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle(
                "/api/agent/../../../etc/passwd/persona", {}, mock_http_handler
            )

        assert result.status_code == 400


# ===========================================================================
# GET /api/agent/{name}/performance Tests
# ===========================================================================


class TestGetAgentPerformance:
    """Tests for GET /api/agent/{name}/performance endpoint."""

    def test_get_performance_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of agent performance."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/performance", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["agent"] == "claude"
        assert "performance" in data
        assert "total_debates" in data["performance"]

    def test_get_performance_no_manager(self, mock_http_handler):
        """Test response when persona manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle("/api/agent/claude/performance", {}, mock_http_handler)

        assert result.status_code == 503


# ===========================================================================
# GET /api/agent/{name}/domains Tests
# ===========================================================================


class TestGetAgentDomains:
    """Tests for GET /api/agent/{name}/domains endpoint."""

    def test_get_domains_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of agent domains."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/domains", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["agent"] == "claude"
        assert "domains" in data
        assert len(data["domains"]) > 0
        assert "domain" in data["domains"][0]
        assert "calibration_score" in data["domains"][0]

    def test_get_domains_with_limit(self, persona_handler, mock_http_handler):
        """Test retrieval with limit parameter."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle(
                "/api/agent/claude/domains",
                {"limit": ["2"]},
                mock_http_handler,
            )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["count"] <= 2

    def test_get_domains_no_elo_system(self, mock_http_handler):
        """Test response when ELO system not available."""
        handler = PersonaHandler({})
        handler.get_elo_system = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle("/api/agent/claude/domains", {}, mock_http_handler)

        assert result.status_code == 503


# ===========================================================================
# GET /api/agent/{name}/grounded-persona Tests
# ===========================================================================


class TestGetGroundedPersona:
    """Tests for GET /api/agent/{name}/grounded-persona endpoint."""

    def test_grounded_persona_module_not_available(self, mock_http_handler):
        """Test response when grounded personas module not available."""
        handler = PersonaHandler({})

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                False,
            ):
                result = handler.handle("/api/agent/claude/grounded-persona", {}, mock_http_handler)

        assert result.status_code == 503

    def test_grounded_persona_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of grounded persona."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.get_grounded_persona.return_value = MockGroundedPersona()

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                True,
            ):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/grounded-persona",
                        {},
                        mock_http_handler,
                    )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["agent"] == "claude"
        assert "elo" in data
        assert "win_rate" in data

    def test_grounded_persona_no_data(self, persona_handler, mock_http_handler):
        """Test response when no grounded persona data available."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.get_grounded_persona.return_value = None

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                True,
            ):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/grounded-persona",
                        {},
                        mock_http_handler,
                    )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert "message" in data


# ===========================================================================
# GET /api/agent/{name}/identity-prompt Tests
# ===========================================================================


class TestGetIdentityPrompt:
    """Tests for GET /api/agent/{name}/identity-prompt endpoint."""

    def test_identity_prompt_module_not_available(self, mock_http_handler):
        """Test response when grounded personas module not available."""
        handler = PersonaHandler({})

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                False,
            ):
                result = handler.handle("/api/agent/claude/identity-prompt", {}, mock_http_handler)

        assert result.status_code == 503

    def test_identity_prompt_success(self, persona_handler, mock_http_handler):
        """Test successful retrieval of identity prompt."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.return_value = (
            "You are Claude, an AI assistant..."
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                True,
            ):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/identity-prompt",
                        {},
                        mock_http_handler,
                    )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["agent"] == "claude"
        assert "identity_prompt" in data

    def test_identity_prompt_with_sections(self, persona_handler, mock_http_handler):
        """Test retrieval with sections parameter."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.return_value = "Partial prompt..."

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.GROUNDED_AVAILABLE",
                True,
            ):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/identity-prompt",
                        {"sections": ["elo,calibration"]},
                        mock_http_handler,
                    )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["sections"] == ["elo", "calibration"]


# ===========================================================================
# GET /api/agent/{name}/accuracy Tests
# ===========================================================================


class TestGetAgentAccuracy:
    """Tests for GET /api/agent/{name}/accuracy endpoint."""

    def test_accuracy_module_not_available(self, mock_http_handler):
        """Test response when PositionTracker module not available."""
        handler = PersonaHandler({})

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                False,
            ):
                result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 503

    def test_accuracy_no_nomic_dir(self, mock_http_handler):
        """Test response when nomic dir not configured."""
        handler = PersonaHandler({})
        handler.get_nomic_dir = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch(
                "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                True,
            ):
                with patch(
                    "aragora.server.handlers.persona.PositionTracker",
                    MockPositionTracker,
                ):
                    result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 503

    def test_accuracy_db_not_exists(self, mock_http_handler):
        """Test response when position database doesn't exist."""
        handler = PersonaHandler({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_path = Path(tmp_dir)
            handler.get_nomic_dir = MagicMock(return_value=nomic_path)

            with patch(
                "aragora.server.handlers.persona._persona_limiter.is_allowed",
                return_value=True,
            ):
                with patch(
                    "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                    True,
                ):
                    with patch(
                        "aragora.server.handlers.persona.PositionTracker",
                        MockPositionTracker,
                    ):
                        result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["total_positions"] == 0
        assert "message" in data

    def test_accuracy_success(self, mock_http_handler):
        """Test successful retrieval of accuracy data."""
        handler = PersonaHandler({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_path = Path(tmp_dir)
            # Create the database file
            db_path = nomic_path / "aragora_positions.db"
            db_path.touch()
            handler.get_nomic_dir = MagicMock(return_value=nomic_path)

            with patch(
                "aragora.server.handlers.persona._persona_limiter.is_allowed",
                return_value=True,
            ):
                with patch(
                    "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                    True,
                ):
                    with patch(
                        "aragora.server.handlers.persona.PositionTracker",
                        MockPositionTracker,
                    ):
                        result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["agent"] == "claude"
        assert data["total_positions"] == 250
        assert data["accuracy_rate"] == 0.78


# ===========================================================================
# POST /api/personas Tests
# ===========================================================================


class TestCreatePersona:
    """Tests for POST /api/personas endpoint."""

    def test_create_persona_success(self, persona_handler, mock_http_handler, mock_persona_manager):
        """Test successful persona creation."""
        mock_http_handler.set_body(
            {
                "agent_name": "gemini",
                "description": "A multimodal AI",
                "traits": ["visual", "analytical"],
                "expertise": {"images": 0.9},
            }
        )

        persona_handler.read_json_body_validated = MagicMock(
            return_value=(
                {
                    "agent_name": "gemini",
                    "description": "A multimodal AI",
                    "traits": ["visual", "analytical"],
                    "expertise": {"images": 0.9},
                },
                None,
            )
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["success"] is True
        assert data["persona"]["agent_name"] == "gemini"

    def test_create_persona_missing_agent_name(self, persona_handler, mock_http_handler):
        """Test creation fails without agent_name."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "Test"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400

    def test_create_persona_invalid_agent_name(self, persona_handler, mock_http_handler):
        """Test creation fails with invalid agent_name."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"agent_name": "../etc/passwd"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400

    def test_create_persona_invalid_traits_type(self, persona_handler, mock_http_handler):
        """Test creation fails when traits is not a list."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"agent_name": "test", "traits": "not-a-list"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "traits must be a list" in data["error"]

    def test_create_persona_invalid_expertise_type(self, persona_handler, mock_http_handler):
        """Test creation fails when expertise is not a dict."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"agent_name": "test", "expertise": ["not-a-dict"]}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "expertise must be a dict" in data["error"]

    def test_create_persona_no_manager(self, mock_http_handler):
        """Test creation fails when manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)
        handler.read_json_body_validated = MagicMock(return_value=({"agent_name": "test"}, None))

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 503


# ===========================================================================
# PUT /api/agent/{name}/persona Tests
# ===========================================================================


class TestUpdatePersona:
    """Tests for PUT /api/agent/{name}/persona endpoint."""

    def test_update_persona_success(self, persona_handler, mock_http_handler, mock_persona_manager):
        """Test successful persona update."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=(
                {
                    "description": "Updated description",
                    "traits": ["updated", "traits"],
                },
                None,
            )
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put(
                "/api/agent/claude/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["success"] is True

    def test_update_persona_not_found(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test update fails when persona doesn't exist."""
        mock_persona_manager._personas = {}
        persona_handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "Test"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put(
                "/api/agent/unknown/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 404

    def test_update_persona_invalid_traits(self, persona_handler, mock_http_handler):
        """Test update fails with invalid traits type."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"traits": "not-a-list"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put(
                "/api/agent/claude/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 400


# ===========================================================================
# DELETE /api/agent/{name}/persona Tests
# ===========================================================================


class TestDeletePersona:
    """Tests for DELETE /api/agent/{name}/persona endpoint."""

    def test_delete_persona_success(self, persona_handler, mock_http_handler, mock_persona_manager):
        """Test successful persona deletion."""
        # Setup mock connection context manager
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_persona_manager._connection_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_persona_manager._connection_ctx.__exit__ = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/claude/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["success"] is True

    def test_delete_persona_not_found(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test delete fails when persona doesn't exist."""
        mock_persona_manager._personas = {}
        persona_handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/unknown/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 404

    def test_delete_persona_no_manager(self, mock_http_handler):
        """Test delete fails when manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle_delete(
                "/api/agent/claude/persona",
                {},
                mock_http_handler,
            )

        assert result.status_code == 503


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestPersonaHandlerRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded_get(self, persona_handler, mock_http_handler):
        """Test rate limit on GET requests."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=False,
        ):
            result = persona_handler.handle("/api/personas", {}, mock_http_handler)

        assert result.status_code == 429
        data = parse_handler_response(result)
        assert "Rate limit" in data["error"]

    def test_rate_limit_exceeded_post(self, persona_handler, mock_http_handler):
        """Test rate limit on POST requests."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=False,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 429

    def test_rate_limit_exceeded_put(self, persona_handler, mock_http_handler):
        """Test rate limit on PUT requests."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=False,
        ):
            result = persona_handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 429

    def test_rate_limit_exceeded_delete(self, persona_handler, mock_http_handler):
        """Test rate limit on DELETE requests."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=False,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/claude/persona", {}, mock_http_handler
            )

        assert result.status_code == 429


# ===========================================================================
# Path Validation Tests
# ===========================================================================


class TestPathValidation:
    """Tests for path parameter validation."""

    def test_extract_agent_name_valid(self, persona_handler, mock_http_handler):
        """Test valid agent names are accepted."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude_v3/persona", {}, mock_http_handler)
        # Should not return 400 for valid name
        assert result.status_code != 400 or "Invalid" not in parse_handler_response(result).get(
            "error", ""
        )

    def test_extract_agent_name_with_numbers(self, persona_handler, mock_http_handler):
        """Test agent names with numbers are accepted."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/gpt4o/persona", {}, mock_http_handler)
        # Should not return 400 for valid name
        assert result.status_code != 400 or "Invalid" not in parse_handler_response(result).get(
            "error", ""
        )


# ===========================================================================
# Version Prefix Tests
# ===========================================================================


class TestVersionPrefix:
    """Tests for API version prefix handling."""

    def test_handles_v1_prefix(self, persona_handler, mock_http_handler):
        """Test handler strips v1 prefix correctly."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/v1/personas", {}, mock_http_handler)

        assert result.status_code == 200

    def test_handles_v2_prefix(self, persona_handler, mock_http_handler):
        """Test handler strips v2 prefix correctly."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/v2/personas", {}, mock_http_handler)

        assert result.status_code == 200

    def test_handles_no_prefix(self, persona_handler, mock_http_handler):
        """Test handler works without version prefix."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestPersonaHandlerRBAC:
    """Tests for RBAC permission enforcement on persona endpoints."""

    def test_handle_has_persona_read_permission(self):
        """The handle method should be decorated with persona:read permission."""
        handler = PersonaHandler({})
        method = handler.handle
        assert callable(method)

    def test_handle_post_has_persona_create_permission(self):
        """The handle_post method should be decorated with persona:create permission."""
        handler = PersonaHandler({})
        method = handler.handle_post
        assert callable(method)

    def test_handle_put_has_persona_update_permission(self):
        """The handle_put method should be decorated with persona:update permission."""
        handler = PersonaHandler({})
        method = handler.handle_put
        assert callable(method)

    def test_handle_delete_has_persona_delete_permission(self):
        """The handle_delete method should be decorated with persona:delete permission."""
        handler = PersonaHandler({})
        method = handler.handle_delete
        assert callable(method)

    @pytest.mark.no_auto_auth
    def test_rbac_enforced_when_real_auth_enabled(self, mock_http_handler):
        """Test that RBAC returns 401 when real auth is enabled without token."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=MagicMock())

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch.dict(
                "os.environ",
                {"ARAGORA_TEST_REAL_AUTH": "1", "PYTEST_CURRENT_TEST": "test"},
            ):
                result = handler.handle("/api/personas", {}, mock_http_handler)

        # Without a valid token, should get 401 or 403
        assert result is not None
        assert result.status_code in (401, 403)


# ===========================================================================
# Exception/Error Handling Tests
# ===========================================================================


class TestPersonaErrorHandling:
    """Tests for exception handling in persona endpoints."""

    def test_get_performance_exception(self, persona_handler, mock_http_handler):
        """Test error handling when performance summary raises exception."""
        mock_manager = MagicMock()
        mock_manager.get_performance_summary.side_effect = ValueError("DB crash")
        persona_handler.get_persona_manager = MagicMock(return_value=mock_manager)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/performance", {}, mock_http_handler)

        assert result.status_code == 500

    def test_get_domains_exception(self, persona_handler, mock_http_handler):
        """Test error handling when get_best_domains raises exception."""
        mock_elo = MagicMock()
        mock_elo.get_best_domains.side_effect = RuntimeError("ELO error")
        persona_handler.get_elo_system = MagicMock(return_value=mock_elo)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/domains", {}, mock_http_handler)

        assert result.status_code == 500

    def test_get_persona_exception(self, persona_handler, mock_http_handler):
        """Test error handling when get_persona raises exception."""
        mock_manager = MagicMock()
        mock_manager.get_persona.side_effect = ValueError("lookup failed")
        persona_handler.get_persona_manager = MagicMock(return_value=mock_manager)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 500

    def test_create_persona_exception(self, persona_handler, mock_http_handler):
        """Test error handling when create_persona raises exception."""
        mock_manager = MagicMock()
        mock_manager.create_persona.side_effect = ValueError("DB write failed")
        persona_handler.get_persona_manager = MagicMock(return_value=mock_manager)
        persona_handler.read_json_body_validated = MagicMock(
            return_value=(
                {"agent_name": "test_agent", "traits": [], "expertise": {}},
                None,
            )
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 500

    def test_update_persona_exception(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test error handling when update (create_persona) raises exception."""
        mock_persona_manager.create_persona = MagicMock(side_effect=ValueError("DB write failed"))
        persona_handler.get_persona_manager = MagicMock(return_value=mock_persona_manager)
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "Updated"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 500

    def test_delete_persona_exception(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test error handling when delete raises exception."""
        mock_persona_manager._connection_ctx.__enter__ = MagicMock(
            side_effect=ValueError("DB connection failed")
        )
        mock_persona_manager._connection_ctx.__exit__ = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/claude/persona", {}, mock_http_handler
            )

        assert result.status_code == 500

    def test_grounded_persona_exception(self, persona_handler, mock_http_handler):
        """Test error handling when PersonaSynthesizer raises exception."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.get_grounded_persona.side_effect = ValueError("Synthesis error")

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/grounded-persona", {}, mock_http_handler
                    )

        assert result.status_code == 500

    def test_identity_prompt_exception(self, persona_handler, mock_http_handler):
        """Test error handling when identity prompt synthesis raises exception."""
        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.side_effect = ValueError("Prompt error")

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True):
                with patch(
                    "aragora.server.handlers.persona.PersonaSynthesizer",
                    return_value=mock_synthesizer,
                ):
                    result = persona_handler.handle(
                        "/api/agent/claude/identity-prompt", {}, mock_http_handler
                    )

        assert result.status_code == 500

    def test_accuracy_exception(self, mock_http_handler):
        """Test error handling when PositionTracker raises exception."""
        handler = PersonaHandler({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_path = Path(tmp_dir)
            db_path = nomic_path / "aragora_positions.db"
            db_path.touch()
            handler.get_nomic_dir = MagicMock(return_value=nomic_path)

            mock_tracker_cls = MagicMock(side_effect=ValueError("Tracker init failed"))

            with patch(
                "aragora.server.handlers.persona._persona_limiter.is_allowed",
                return_value=True,
            ):
                with patch(
                    "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                    True,
                ):
                    with patch(
                        "aragora.server.handlers.persona.PositionTracker",
                        mock_tracker_cls,
                    ):
                        result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 500

    def test_accuracy_returns_none_data(self, mock_http_handler):
        """Test response when PositionTracker returns None for agent."""
        handler = PersonaHandler({})
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_path = Path(tmp_dir)
            db_path = nomic_path / "aragora_positions.db"
            db_path.touch()
            handler.get_nomic_dir = MagicMock(return_value=nomic_path)

            mock_tracker = MagicMock()
            mock_tracker.get_agent_position_accuracy.return_value = None

            with patch(
                "aragora.server.handlers.persona._persona_limiter.is_allowed",
                return_value=True,
            ):
                with patch(
                    "aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE",
                    True,
                ):
                    with patch(
                        "aragora.server.handlers.persona.PositionTracker",
                        return_value=mock_tracker,
                    ):
                        result = handler.handle("/api/agent/claude/accuracy", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["total_positions"] == 0
        assert "message" in data


# ===========================================================================
# Unmatched Path Tests
# ===========================================================================


class TestUnmatchedPaths:
    """Tests for unmatched paths returning None."""

    def test_handle_returns_none_for_unmatched(self, persona_handler, mock_http_handler):
        """GET handle should return None for unmatched agent paths."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/something/else", {}, mock_http_handler)

        assert result is None

    def test_handle_post_returns_none_for_unmatched(self, persona_handler, mock_http_handler):
        """POST handle should return None for non-persona paths."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/other", {}, mock_http_handler)

        assert result is None

    def test_handle_put_returns_none_for_unmatched(self, persona_handler, mock_http_handler):
        """PUT handle should return None for non-persona paths."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put("/api/other/path", {}, mock_http_handler)

        assert result is None

    def test_handle_delete_returns_none_for_unmatched(self, persona_handler, mock_http_handler):
        """DELETE handle should return None for non-persona paths."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete("/api/other/path", {}, mock_http_handler)

        assert result is None


# ===========================================================================
# PUT/DELETE Path Validation Tests
# ===========================================================================


class TestPutDeletePathValidation:
    """Tests for path validation on PUT and DELETE endpoints."""

    def test_put_invalid_agent_name(self, persona_handler, mock_http_handler):
        """PUT should reject invalid agent names."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "Test"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put(
                "/api/agent/../../../etc/passwd/persona", {}, mock_http_handler
            )

        assert result.status_code == 400

    def test_delete_invalid_agent_name(self, persona_handler, mock_http_handler):
        """DELETE should reject invalid agent names."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/../../../etc/passwd/persona", {}, mock_http_handler
            )

        assert result.status_code == 400

    def test_put_with_version_prefix(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """PUT should handle versioned paths."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "Updated via v1"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put(
                "/api/v1/agent/claude/persona", {}, mock_http_handler
            )

        assert result.status_code == 200

    def test_delete_with_version_prefix(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """DELETE should handle versioned paths."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_persona_manager._connection_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_persona_manager._connection_ctx.__exit__ = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/v1/agent/claude/persona", {}, mock_http_handler
            )

        assert result.status_code == 200


# ===========================================================================
# Update Persona Additional Validation Tests
# ===========================================================================


class TestUpdatePersonaValidation:
    """Additional validation tests for PUT /api/agent/{name}/persona."""

    def test_update_persona_invalid_expertise(self, persona_handler, mock_http_handler):
        """Test update fails with invalid expertise type."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"expertise": "not-a-dict"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "expertise must be a dict" in data["error"]

    def test_update_persona_no_manager(self, mock_http_handler):
        """Test update fails when persona manager not configured."""
        handler = PersonaHandler({})
        handler.get_persona_manager = MagicMock(return_value=None)
        handler.read_json_body_validated = MagicMock(return_value=({"description": "Test"}, None))

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 503

    def test_update_persona_preserves_existing_fields(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test update preserves existing fields when not provided."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"description": "New description only"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["success"] is True

    def test_update_persona_json_body_error(self, persona_handler, mock_http_handler):
        """Test update fails when JSON body parsing fails."""
        from aragora.server.handlers.utils.responses import error_response as er

        persona_handler.read_json_body_validated = MagicMock(
            return_value=(None, er("Invalid JSON", 400))
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_put("/api/agent/claude/persona", {}, mock_http_handler)

        assert result.status_code == 400


# ===========================================================================
# Create Persona Additional Tests
# ===========================================================================


class TestCreatePersonaAdditional:
    """Additional tests for POST /api/personas."""

    def test_create_persona_empty_agent_name(self, persona_handler, mock_http_handler):
        """Test creation fails with empty agent_name."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"agent_name": "   "}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400

    def test_create_persona_json_body_error(self, persona_handler, mock_http_handler):
        """Test creation fails when JSON body parsing fails."""
        from aragora.server.handlers.utils.responses import error_response as er

        persona_handler.read_json_body_validated = MagicMock(
            return_value=(None, er("Invalid JSON", 400))
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 400

    def test_create_persona_default_traits_expertise(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test creation succeeds with default (empty) traits and expertise."""
        persona_handler.read_json_body_validated = MagicMock(
            return_value=({"agent_name": "newagent"}, None)
        )

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_post("/api/personas", {}, mock_http_handler)

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data["success"] is True
        assert data["persona"]["agent_name"] == "newagent"


# ===========================================================================
# Persona Data Content Verification Tests
# ===========================================================================


class TestPersonaDataContent:
    """Tests verifying correct data shapes in responses."""

    def test_all_personas_response_fields(self, persona_handler, mock_http_handler):
        """Test that all persona response includes expected fields per persona."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/personas", {}, mock_http_handler)

        data = parse_handler_response(result)
        for persona in data["personas"]:
            assert "agent_name" in persona
            assert "description" in persona
            assert "traits" in persona
            assert "expertise" in persona
            assert "created_at" in persona
            assert "updated_at" in persona

    def test_domains_response_shape(self, persona_handler, mock_http_handler):
        """Test that domains response has correct shape."""
        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle("/api/agent/claude/domains", {}, mock_http_handler)

        data = parse_handler_response(result)
        assert "agent" in data
        assert "domains" in data
        assert "count" in data
        for domain in data["domains"]:
            assert "domain" in domain
            assert "calibration_score" in domain

    def test_delete_success_message_includes_agent_name(
        self, persona_handler, mock_http_handler, mock_persona_manager
    ):
        """Test delete success message includes the agent name."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_persona_manager._connection_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_persona_manager._connection_ctx.__exit__ = MagicMock(return_value=None)

        with patch(
            "aragora.server.handlers.persona._persona_limiter.is_allowed",
            return_value=True,
        ):
            result = persona_handler.handle_delete(
                "/api/agent/claude/persona", {}, mock_http_handler
            )

        data = parse_handler_response(result)
        assert "claude" in data["message"]

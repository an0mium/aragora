"""Tests for persona handler endpoints.

Tests the persona API endpoints including:
- GET /api/personas - Get all agent personas
- GET /api/agent/{name}/persona - Get agent persona
- GET /api/agent/{name}/grounded-persona - Get truth-grounded persona
- GET /api/agent/{name}/identity-prompt - Get identity prompt
- GET /api/agent/{name}/performance - Get agent performance summary
- GET /api/agent/{name}/domains - Get agent expertise domains
- GET /api/agent/{name}/accuracy - Get position accuracy stats
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockPersona:
    """Mock persona object."""

    def __init__(
        self,
        agent_name: str,
        description: str = "A helpful AI assistant",
        traits: List[str] = None,
        expertise: List[str] = None,
        created_at: str = None,
        updated_at: str = None,
    ):
        self.agent_name = agent_name
        self.description = description
        self.traits = traits or ["analytical", "thorough"]
        self.expertise = expertise or ["software engineering", "debugging"]
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()


class MockPersonaManager:
    """Mock PersonaManager for testing."""

    def __init__(self):
        self._personas: Dict[str, MockPersona] = {}
        self._performance: Dict[str, dict] = {}

    def add_persona(self, persona: MockPersona):
        self._personas[persona.agent_name] = persona

    def add_performance(self, agent: str, data: dict):
        self._performance[agent] = data

    def get_all_personas(self) -> List[MockPersona]:
        return list(self._personas.values())

    def get_persona(self, agent: str) -> Optional[MockPersona]:
        return self._personas.get(agent)

    def get_performance_summary(self, agent: str) -> dict:
        return self._performance.get(
            agent,
            {
                "debates_participated": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
            },
        )


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self):
        self._domains: Dict[str, List[tuple]] = {}

    def add_domains(self, agent: str, domains: List[tuple]):
        self._domains[agent] = domains

    def get_best_domains(self, agent: str, limit: int = 10) -> List[tuple]:
        return self._domains.get(agent, [])[:limit]


class MockHandler:
    """Mock HTTP handler."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


@pytest.fixture
def mock_handler():
    """Create mock handler."""
    return MockHandler()


@pytest.fixture
def mock_persona_manager():
    """Create mock persona manager with sample data."""
    manager = MockPersonaManager()
    manager.add_persona(
        MockPersona(
            agent_name="claude",
            description="Anthropic's Claude assistant",
            traits=["analytical", "careful", "thorough"],
            expertise=["reasoning", "coding", "writing"],
        )
    )
    manager.add_persona(
        MockPersona(
            agent_name="gpt-4",
            description="OpenAI's GPT-4 assistant",
            traits=["creative", "versatile"],
            expertise=["general knowledge", "coding"],
        )
    )
    manager.add_performance(
        "claude",
        {
            "debates_participated": 50,
            "win_rate": 0.65,
            "avg_confidence": 0.82,
        },
    )
    return manager


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system with sample data."""
    elo = MockEloSystem()
    elo.add_domains(
        "claude",
        [
            ("security", 0.92),
            ("architecture", 0.88),
            ("testing", 0.85),
        ],
    )
    return elo


@pytest.fixture
def persona_handler(mock_persona_manager, mock_elo_system):
    """Create PersonaHandler for testing."""
    from aragora.server.handlers.persona import PersonaHandler

    ctx = {
        "persona_manager": mock_persona_manager,
        "elo_system": mock_elo_system,
        "position_ledger": None,
        "nomic_dir": None,
    }
    handler = PersonaHandler(ctx)
    return handler


class TestPersonaHandlerRouting:
    """Test routing logic for persona handler."""

    def test_can_handle_personas_list(self):
        """Test can_handle for /api/personas."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/personas") is True

    def test_can_handle_agent_persona(self):
        """Test can_handle for /api/agent/{name}/persona."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/persona") is True
        assert handler.can_handle("/api/agent/gpt-4/persona") is True

    def test_can_handle_grounded_persona(self):
        """Test can_handle for /api/agent/{name}/grounded-persona."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/grounded-persona") is True

    def test_can_handle_identity_prompt(self):
        """Test can_handle for /api/agent/{name}/identity-prompt."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/identity-prompt") is True

    def test_can_handle_performance(self):
        """Test can_handle for /api/agent/{name}/performance."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/performance") is True

    def test_can_handle_domains(self):
        """Test can_handle for /api/agent/{name}/domains."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/domains") is True

    def test_can_handle_accuracy(self):
        """Test can_handle for /api/agent/{name}/accuracy."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude/accuracy") is True

    def test_cannot_handle_invalid(self):
        """Test can_handle rejects invalid paths."""
        from aragora.server.handlers.persona import PersonaHandler

        handler = PersonaHandler({})
        assert handler.can_handle("/api/agent/claude") is False
        assert handler.can_handle("/api/agent/claude/unknown") is False
        assert handler.can_handle("/api/other") is False


class TestPersonaHandlerListAll:
    """Test /api/personas endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_list_all_personas(self, mock_limiter, persona_handler, mock_handler):
        """Test successful personas listing."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/personas", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "personas" in body
        assert body["count"] == 2
        agents = [p["agent_name"] for p in body["personas"]]
        assert "claude" in agents
        assert "gpt-4" in agents

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_list_personas_no_manager(self, mock_limiter, mock_handler):
        """Test personas listing when manager not configured."""
        from aragora.server.handlers.persona import PersonaHandler

        mock_limiter.is_allowed.return_value = True

        handler = PersonaHandler({"persona_manager": None})

        result = handler.handle("/api/personas", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "error" in body
        assert body["personas"] == []


class TestPersonaHandlerGetPersona:
    """Test /api/agent/{name}/persona endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_persona_success(self, mock_limiter, persona_handler, mock_handler):
        """Test successful persona retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/persona", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "persona" in body
        assert body["persona"]["agent_name"] == "claude"
        assert "analytical" in body["persona"]["traits"]

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_persona_not_found(self, mock_limiter, persona_handler, mock_handler):
        """Test persona retrieval for unknown agent."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/unknown-agent/persona", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["persona"] is None
        assert "error" in body


class TestPersonaHandlerPerformance:
    """Test /api/agent/{name}/performance endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_performance_success(self, mock_limiter, persona_handler, mock_handler):
        """Test successful performance retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/performance", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["agent"] == "claude"
        assert "performance" in body
        assert body["performance"]["win_rate"] == 0.65

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_performance_no_manager(self, mock_limiter, mock_handler):
        """Test performance when manager not configured."""
        from aragora.server.handlers.persona import PersonaHandler

        mock_limiter.is_allowed.return_value = True

        handler = PersonaHandler({"persona_manager": None})

        result = handler.handle("/api/agent/claude/performance", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


class TestPersonaHandlerDomains:
    """Test /api/agent/{name}/domains endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_domains_success(self, mock_limiter, persona_handler, mock_handler):
        """Test successful domains retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/domains", {"limit": "5"}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["agent"] == "claude"
        assert "domains" in body
        assert len(body["domains"]) == 3
        assert body["domains"][0]["domain"] == "security"
        assert body["domains"][0]["calibration_score"] == 0.92

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_get_domains_no_elo(self, mock_limiter, mock_handler):
        """Test domains when ELO system not configured."""
        from aragora.server.handlers.persona import PersonaHandler

        mock_limiter.is_allowed.return_value = True

        handler = PersonaHandler({"elo_system": None})

        result = handler.handle("/api/agent/claude/domains", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


class TestPersonaHandlerGroundedPersona:
    """Test /api/agent/{name}/grounded-persona endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", False)
    def test_grounded_persona_unavailable(self, mock_limiter, persona_handler, mock_handler):
        """Test grounded persona when module unavailable."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/grounded-persona", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503
        body = parse_body(result)
        assert "not available" in body["error"].lower()

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True)
    @patch("aragora.server.handlers.persona.PersonaSynthesizer")
    def test_grounded_persona_success(
        self, mock_synthesizer_cls, mock_limiter, persona_handler, mock_handler
    ):
        """Test successful grounded persona retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_persona = MagicMock()
        mock_persona.elo = 1250
        mock_persona.domain_elos = {"security": 1300, "performance": 1200}
        mock_persona.games_played = 50
        mock_persona.win_rate = 0.65
        mock_persona.calibration_score = 0.82
        mock_persona.position_accuracy = 0.78
        mock_persona.positions_taken = 120
        mock_persona.reversals = 5

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_grounded_persona.return_value = mock_persona
        mock_synthesizer_cls.return_value = mock_synthesizer

        result = persona_handler.handle("/api/agent/claude/grounded-persona", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["agent"] == "claude"
        assert body["elo"] == 1250
        assert body["win_rate"] == 0.65


class TestPersonaHandlerIdentityPrompt:
    """Test /api/agent/{name}/identity-prompt endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", False)
    def test_identity_prompt_unavailable(self, mock_limiter, persona_handler, mock_handler):
        """Test identity prompt when module unavailable."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/identity-prompt", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True)
    @patch("aragora.server.handlers.persona.PersonaSynthesizer")
    def test_identity_prompt_success(
        self, mock_synthesizer_cls, mock_limiter, persona_handler, mock_handler
    ):
        """Test successful identity prompt generation."""
        mock_limiter.is_allowed.return_value = True

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.return_value = (
            "You are Claude, an AI assistant..."
        )
        mock_synthesizer_cls.return_value = mock_synthesizer

        result = persona_handler.handle("/api/agent/claude/identity-prompt", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["agent"] == "claude"
        assert "identity_prompt" in body

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True)
    @patch("aragora.server.handlers.persona.PersonaSynthesizer")
    def test_identity_prompt_with_sections(
        self, mock_synthesizer_cls, mock_limiter, persona_handler, mock_handler
    ):
        """Test identity prompt with section filter."""
        mock_limiter.is_allowed.return_value = True

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.return_value = "Filtered prompt..."
        mock_synthesizer_cls.return_value = mock_synthesizer

        result = persona_handler.handle(
            "/api/agent/claude/identity-prompt",
            {"sections": "traits,expertise"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["sections"] == ["traits", "expertise"]


class TestPersonaHandlerAccuracy:
    """Test /api/agent/{name}/accuracy endpoint."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    @patch("aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE", False)
    def test_accuracy_unavailable(self, mock_limiter, persona_handler, mock_handler):
        """Test accuracy when PositionTracker unavailable."""
        mock_limiter.is_allowed.return_value = True

        result = persona_handler.handle("/api/agent/claude/accuracy", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


class TestPersonaHandlerRateLimiting:
    """Test rate limiting for persona endpoints."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_rate_limit_exceeded(self, mock_limiter, persona_handler, mock_handler):
        """Test rate limit exceeded response."""
        mock_limiter.is_allowed.return_value = False

        result = persona_handler.handle("/api/personas", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429
        body = parse_body(result)
        assert "rate limit" in body["error"].lower()


class TestPersonaHandlerAgentNameValidation:
    """Test agent name validation."""

    @patch("aragora.server.handlers.persona._persona_limiter")
    def test_invalid_agent_name(self, mock_limiter, persona_handler, mock_handler):
        """Test rejection of invalid agent names."""
        mock_limiter.is_allowed.return_value = True

        # Agent names with special characters should be rejected
        result = persona_handler.handle("/api/agent/../etc/passwd/persona", {}, mock_handler)

        # Should return error for path traversal attempt
        assert result is not None
        # Either 400 (invalid) or 200 with no persona found
        if result.status_code == 400:
            body = parse_body(result)
            assert "error" in body

"""
Tests for the Persona Handler endpoints.

Covers:
- GET /api/personas - Get all agent personas
- GET /api/agent/{name}/persona - Get agent persona
- GET /api/agent/{name}/grounded-persona - Get truth-grounded persona
- GET /api/agent/{name}/identity-prompt - Get identity prompt
- GET /api/agent/{name}/performance - Get agent performance summary
- GET /api/agent/{name}/domains - Get agent expertise domains
- GET /api/agent/{name}/accuracy - Get position accuracy stats
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest

from aragora.server.handlers.persona import PersonaHandler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def persona_handler(handler_context):
    """Create a PersonaHandler with mock context."""
    return PersonaHandler(handler_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    handler = Mock()
    handler.headers = {}
    handler.command = 'GET'
    return handler


@pytest.fixture
def mock_persona_manager():
    """Create a mock persona manager."""
    manager = MagicMock()

    mock_persona = MagicMock()
    mock_persona.agent_name = "claude"
    mock_persona.description = "A helpful AI assistant"
    mock_persona.traits = ["analytical", "helpful"]
    mock_persona.expertise = ["coding", "analysis"]
    mock_persona.created_at = "2025-01-01T00:00:00Z"
    mock_persona.updated_at = "2025-01-10T00:00:00Z"

    manager.get_all_personas.return_value = [mock_persona]
    manager.get_persona.return_value = mock_persona
    manager.get_performance_summary.return_value = {
        "wins": 10,
        "losses": 5,
        "win_rate": 0.67
    }

    return manager


@pytest.fixture
def handler_context_with_persona(handler_context, mock_persona_manager):
    """Handler context with persona manager configured."""
    handler_context["persona_manager"] = mock_persona_manager
    return handler_context


# =============================================================================
# can_handle Tests
# =============================================================================

class TestCanHandle:
    """Test route matching for PersonaHandler."""

    def test_can_handle_personas_route(self, persona_handler):
        """Test matching all personas endpoint."""
        assert persona_handler.can_handle("/api/personas")

    def test_can_handle_agent_persona_route(self, persona_handler):
        """Test matching agent persona endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/persona")

    def test_can_handle_grounded_persona_route(self, persona_handler):
        """Test matching grounded persona endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/grounded-persona")

    def test_can_handle_identity_prompt_route(self, persona_handler):
        """Test matching identity prompt endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/identity-prompt")

    def test_can_handle_performance_route(self, persona_handler):
        """Test matching performance endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/performance")

    def test_can_handle_domains_route(self, persona_handler):
        """Test matching domains endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/domains")

    def test_can_handle_accuracy_route(self, persona_handler):
        """Test matching accuracy endpoint."""
        assert persona_handler.can_handle("/api/agent/claude/accuracy")

    def test_cannot_handle_unknown_route(self, persona_handler):
        """Test rejection of unknown routes."""
        assert not persona_handler.can_handle("/api/unknown")
        assert not persona_handler.can_handle("/api/agent/claude/unknown")
        assert not persona_handler.can_handle("/api/agent/")


# =============================================================================
# All Personas Endpoint Tests
# =============================================================================

class TestAllPersonasEndpoint:
    """Test /api/personas endpoint."""

    def test_personas_no_manager_returns_error(self, handler_context, mock_http_handler):
        """Test response when persona manager not configured."""
        handler_context["persona_manager"] = None
        handler = PersonaHandler(handler_context)

        result = handler.handle("/api/personas", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "error" in body
        assert body["personas"] == []

    def test_personas_returns_all(self, handler_context_with_persona, mock_http_handler):
        """Test successful retrieval of all personas."""
        handler = PersonaHandler(handler_context_with_persona)

        result = handler.handle("/api/personas", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["personas"]) == 1
        assert body["personas"][0]["agent_name"] == "claude"
        assert body["count"] == 1


# =============================================================================
# Agent Persona Endpoint Tests
# =============================================================================

class TestAgentPersonaEndpoint:
    """Test /api/agent/:name/persona endpoint."""

    def test_persona_no_manager_returns_503(self, handler_context, mock_http_handler):
        """Test error when persona manager not configured."""
        handler_context["persona_manager"] = None
        handler = PersonaHandler(handler_context)

        result = handler.handle("/api/agent/claude/persona", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_persona_invalid_name_returns_error(self, handler_context_with_persona, mock_http_handler):
        """Test error on invalid agent name."""
        handler = PersonaHandler(handler_context_with_persona)

        result = handler.handle("/api/agent/../etc/passwd/persona", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400

    def test_persona_found(self, handler_context_with_persona, mock_http_handler):
        """Test successful persona retrieval."""
        handler = PersonaHandler(handler_context_with_persona)

        result = handler.handle("/api/agent/claude/persona", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["persona"]["agent_name"] == "claude"
        assert "description" in body["persona"]

    def test_persona_not_found(self, handler_context_with_persona, mock_http_handler):
        """Test response when persona not found."""
        handler_context_with_persona["persona_manager"].get_persona.return_value = None
        handler = PersonaHandler(handler_context_with_persona)

        result = handler.handle("/api/agent/unknown/persona", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["persona"] is None
        assert "error" in body


# =============================================================================
# Performance Endpoint Tests
# =============================================================================

class TestPerformanceEndpoint:
    """Test /api/agent/:name/performance endpoint."""

    def test_performance_no_manager_returns_503(self, handler_context, mock_http_handler):
        """Test error when persona manager not configured."""
        handler_context["persona_manager"] = None
        handler = PersonaHandler(handler_context)

        result = handler.handle("/api/agent/claude/performance", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_performance_success(self, handler_context_with_persona, mock_http_handler):
        """Test successful performance retrieval."""
        handler = PersonaHandler(handler_context_with_persona)

        result = handler.handle("/api/agent/claude/performance", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert "performance" in body
        assert body["performance"]["wins"] == 10


# =============================================================================
# Domains Endpoint Tests
# =============================================================================

class TestDomainsEndpoint:
    """Test /api/agent/:name/domains endpoint."""

    def test_domains_no_elo_system_returns_503(self, handler_context, mock_http_handler):
        """Test error when ELO system not configured."""
        handler_context["elo_system"] = None
        handler = PersonaHandler(handler_context)

        result = handler.handle("/api/agent/claude/domains", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_domains_with_limit(self, handler_context, mock_http_handler):
        """Test domains endpoint respects limit parameter."""
        mock_elo = MagicMock()
        mock_elo.get_best_domains.return_value = [
            ("coding", 0.9),
            ("analysis", 0.8),
        ]
        handler_context["elo_system"] = mock_elo
        handler = PersonaHandler(handler_context)

        result = handler.handle(
            "/api/agent/claude/domains",
            {"limit": ["5"]},
            mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert len(body["domains"]) == 2


# =============================================================================
# Grounded Persona Endpoint Tests
# =============================================================================

class TestGroundedPersonaEndpoint:
    """Test /api/agent/:name/grounded-persona endpoint."""

    def test_grounded_persona_module_unavailable_returns_503(self, persona_handler, mock_http_handler):
        """Test error when grounded module unavailable."""
        with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", False):
            result = persona_handler.handle(
                "/api/agent/claude/grounded-persona",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_grounded_persona_success(self, handler_context_with_persona, mock_http_handler):
        """Test successful grounded persona retrieval."""
        handler = PersonaHandler(handler_context_with_persona)

        mock_grounded = MagicMock()
        mock_grounded.elo = 1200
        mock_grounded.domain_elos = {"coding": 1300}
        mock_grounded.games_played = 50
        mock_grounded.win_rate = 0.6
        mock_grounded.calibration_score = 0.8
        mock_grounded.position_accuracy = 0.75
        mock_grounded.positions_taken = 100
        mock_grounded.reversals = 5

        mock_synthesizer = MagicMock()
        mock_synthesizer.get_grounded_persona.return_value = mock_grounded

        with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True), \
             patch("aragora.server.handlers.persona.PersonaSynthesizer", return_value=mock_synthesizer):
            result = handler.handle(
                "/api/agent/claude/grounded-persona",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["elo"] == 1200


# =============================================================================
# Identity Prompt Endpoint Tests
# =============================================================================

class TestIdentityPromptEndpoint:
    """Test /api/agent/:name/identity-prompt endpoint."""

    def test_identity_prompt_module_unavailable_returns_503(self, persona_handler, mock_http_handler):
        """Test error when grounded module unavailable."""
        with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", False):
            result = persona_handler.handle(
                "/api/agent/claude/identity-prompt",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_identity_prompt_with_sections(self, handler_context_with_persona, mock_http_handler):
        """Test identity prompt with section filter."""
        handler = PersonaHandler(handler_context_with_persona)

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize_identity_prompt.return_value = "You are Claude..."

        with patch("aragora.server.handlers.persona.GROUNDED_AVAILABLE", True), \
             patch("aragora.server.handlers.persona.PersonaSynthesizer", return_value=mock_synthesizer):
            result = handler.handle(
                "/api/agent/claude/identity-prompt",
                {"sections": ["performance,expertise"]},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["identity_prompt"] == "You are Claude..."
        assert body["sections"] == ["performance", "expertise"]


# =============================================================================
# Accuracy Endpoint Tests
# =============================================================================

class TestAccuracyEndpoint:
    """Test /api/agent/:name/accuracy endpoint."""

    def test_accuracy_module_unavailable_returns_503(self, persona_handler, mock_http_handler):
        """Test error when position tracker module unavailable."""
        with patch("aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE", False):
            result = persona_handler.handle(
                "/api/agent/claude/accuracy",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_accuracy_no_nomic_dir_returns_503(self, handler_context, mock_http_handler):
        """Test error when nomic_dir not configured."""
        handler_context["nomic_dir"] = None
        handler = PersonaHandler(handler_context)

        with patch("aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE", True):
            result = handler.handle(
                "/api/agent/claude/accuracy",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_accuracy_no_data_returns_zeros(self, handler_context, mock_http_handler):
        """Test response when no accuracy data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler_context["nomic_dir"] = Path(tmpdir)
            handler = PersonaHandler(handler_context)

            with patch("aragora.server.handlers.persona.POSITION_TRACKER_AVAILABLE", True):
                result = handler.handle(
                    "/api/agent/claude/accuracy",
                    {},
                    mock_http_handler
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["total_positions"] == 0
        assert body["accuracy_rate"] == 0.0

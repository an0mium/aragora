"""
Tests for PersonaHandler endpoints.

Endpoints tested:
- GET /api/personas - List all personas
- GET /api/agent/{name}/persona - Get agent persona
- GET /api/agent/{name}/grounded-persona - Get truth-grounded persona
- GET /api/agent/{name}/identity-prompt - Get identity prompt
- GET /api/agent/{name}/performance - Get agent performance summary
- GET /api/agent/{name}/domains - Get agent expertise domains
- GET /api/agent/{name}/accuracy - Get position accuracy stats
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers import PersonaHandler, HandlerResult
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_persona():
    """Create a mock persona object."""
    persona = Mock()
    persona.agent_name = "claude"
    persona.description = "A helpful AI assistant"
    persona.traits = ["helpful", "analytical", "thorough"]
    persona.expertise = ["coding", "reasoning", "writing"]
    persona.created_at = "2024-01-01T00:00:00Z"
    persona.updated_at = "2024-01-15T00:00:00Z"
    return persona


@pytest.fixture
def mock_persona_manager(mock_persona):
    """Create a mock persona manager."""
    manager = Mock()
    manager.get_all_personas.return_value = [mock_persona]
    manager.get_persona.return_value = mock_persona
    manager.get_performance_summary.return_value = {
        "total_debates": 50,
        "wins": 30,
        "win_rate": 0.6,
        "avg_calibration": 0.75,
    }
    return manager


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_best_domains.return_value = [
        ("coding", 0.85),
        ("reasoning", 0.78),
        ("writing", 0.72),
    ]
    return elo


@pytest.fixture
def persona_handler(mock_persona_manager, mock_elo_system):
    """Create a PersonaHandler with mock dependencies."""
    ctx = {
        "persona_manager": mock_persona_manager,
        "elo_system": mock_elo_system,
        "position_ledger": None,
        "nomic_dir": None,
    }
    return PersonaHandler(ctx)


@pytest.fixture
def handler_no_persona_manager(mock_elo_system):
    """Create a PersonaHandler without persona manager."""
    ctx = {
        "persona_manager": None,
        "elo_system": mock_elo_system,
        "nomic_dir": None,
    }
    return PersonaHandler(ctx)


@pytest.fixture
def handler_no_elo(mock_persona_manager):
    """Create a PersonaHandler without ELO system."""
    ctx = {
        "persona_manager": mock_persona_manager,
        "elo_system": None,
        "nomic_dir": None,
    }
    return PersonaHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestPersonaRouting:
    """Tests for route matching."""

    def test_can_handle_personas_list(self, persona_handler):
        assert persona_handler.can_handle("/api/personas") is True

    def test_can_handle_agent_persona(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/persona") is True

    def test_can_handle_grounded_persona(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/grounded-persona") is True

    def test_can_handle_identity_prompt(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/identity-prompt") is True

    def test_can_handle_performance(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/performance") is True

    def test_can_handle_domains(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/domains") is True

    def test_can_handle_accuracy(self, persona_handler):
        assert persona_handler.can_handle("/api/agent/claude/accuracy") is True

    def test_cannot_handle_unrelated_routes(self, persona_handler):
        assert persona_handler.can_handle("/api/debates") is False
        assert persona_handler.can_handle("/api/agents") is False
        assert persona_handler.can_handle("/api/agent/claude/unknown") is False

    def test_cannot_handle_partial_paths(self, persona_handler):
        assert persona_handler.can_handle("/api/agent") is False
        assert persona_handler.can_handle("/api/agent/") is False


# ============================================================================
# GET /api/personas Tests
# ============================================================================

class TestListPersonas:
    """Tests for GET /api/personas endpoint."""

    def test_list_personas_success(self, persona_handler, mock_persona_manager):
        result = persona_handler.handle("/api/personas", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "personas" in data
        assert len(data["personas"]) == 1
        assert data["count"] == 1
        assert data["personas"][0]["agent_name"] == "claude"

    def test_list_personas_no_manager(self, handler_no_persona_manager):
        result = handler_no_persona_manager.handle("/api/personas", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "error" in data
        assert data["personas"] == []

    def test_list_personas_empty(self, persona_handler, mock_persona_manager):
        mock_persona_manager.get_all_personas.return_value = []
        result = persona_handler.handle("/api/personas", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["personas"] == []
        assert data["count"] == 0

    def test_list_personas_exception(self, persona_handler, mock_persona_manager):
        mock_persona_manager.get_all_personas.side_effect = Exception("Database error")
        result = persona_handler.handle("/api/personas", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# GET /api/agent/{name}/persona Tests
# ============================================================================

class TestGetAgentPersona:
    """Tests for GET /api/agent/{name}/persona endpoint."""

    def test_get_persona_success(self, persona_handler):
        result = persona_handler.handle("/api/agent/claude/persona", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "persona" in data
        assert data["persona"]["agent_name"] == "claude"
        assert "traits" in data["persona"]

    def test_get_persona_not_found(self, persona_handler, mock_persona_manager):
        mock_persona_manager.get_persona.return_value = None
        result = persona_handler.handle("/api/agent/unknown/persona", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["persona"] is None
        assert "error" in data

    def test_get_persona_no_manager(self, handler_no_persona_manager):
        result = handler_no_persona_manager.handle("/api/agent/claude/persona", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "error" in data

    def test_get_persona_invalid_name(self, persona_handler):
        result = persona_handler.handle("/api/agent/../etc/passwd/persona", {}, None)

        assert result is not None
        assert result.status_code == 400

    def test_get_persona_exception(self, persona_handler, mock_persona_manager):
        mock_persona_manager.get_persona.side_effect = Exception("Database error")
        result = persona_handler.handle("/api/agent/claude/persona", {}, None)

        assert result is not None
        assert result.status_code == 500


# ============================================================================
# GET /api/agent/{name}/performance Tests
# ============================================================================

class TestGetAgentPerformance:
    """Tests for GET /api/agent/{name}/performance endpoint."""

    def test_get_performance_success(self, persona_handler):
        result = persona_handler.handle("/api/agent/claude/performance", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "performance" in data
        assert data["performance"]["total_debates"] == 50

    def test_get_performance_no_manager(self, handler_no_persona_manager):
        result = handler_no_persona_manager.handle("/api/agent/claude/performance", {}, None)

        assert result is not None
        assert result.status_code == 503

    def test_get_performance_exception(self, persona_handler, mock_persona_manager):
        mock_persona_manager.get_performance_summary.side_effect = Exception("Error")
        result = persona_handler.handle("/api/agent/claude/performance", {}, None)

        assert result is not None
        assert result.status_code == 500


# ============================================================================
# GET /api/agent/{name}/domains Tests
# ============================================================================

class TestGetAgentDomains:
    """Tests for GET /api/agent/{name}/domains endpoint."""

    def test_get_domains_success(self, persona_handler):
        result = persona_handler.handle("/api/agent/claude/domains", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert "domains" in data
        assert len(data["domains"]) == 3
        assert data["domains"][0]["domain"] == "coding"

    def test_get_domains_with_limit(self, persona_handler, mock_elo_system):
        mock_elo_system.get_best_domains.return_value = [("coding", 0.85)]
        result = persona_handler.handle("/api/agent/claude/domains", {"limit": "1"}, None)

        assert result is not None
        assert result.status_code == 200
        mock_elo_system.get_best_domains.assert_called_with("claude", limit=1)

    def test_get_domains_no_elo(self, handler_no_elo):
        result = handler_no_elo.handle("/api/agent/claude/domains", {}, None)

        assert result is not None
        assert result.status_code == 503

    def test_get_domains_exception(self, persona_handler, mock_elo_system):
        mock_elo_system.get_best_domains.side_effect = Exception("Error")
        result = persona_handler.handle("/api/agent/claude/domains", {}, None)

        assert result is not None
        assert result.status_code == 500


# ============================================================================
# GET /api/agent/{name}/grounded-persona Tests
# ============================================================================

class TestGetGroundedPersona:
    """Tests for GET /api/agent/{name}/grounded-persona endpoint."""

    def test_grounded_persona_returns_result(self, persona_handler):
        # Should return some result (either 503 if module not available, 200 success, or 500 error)
        result = persona_handler.handle("/api/agent/claude/grounded-persona", {}, None)
        assert result is not None
        # Valid responses: 200 (success), 503 (module not available), 500 (internal error)
        assert result.status_code in (200, 500, 503)

    def test_grounded_persona_invalid_agent(self, persona_handler):
        result = persona_handler.handle("/api/agent/../test/grounded-persona", {}, None)

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# GET /api/agent/{name}/identity-prompt Tests
# ============================================================================

class TestGetIdentityPrompt:
    """Tests for GET /api/agent/{name}/identity-prompt endpoint."""

    def test_identity_prompt_invalid_agent(self, persona_handler):
        result = persona_handler.handle("/api/agent/../../etc/identity-prompt", {}, None)

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# GET /api/agent/{name}/accuracy Tests
# ============================================================================

class TestGetAgentAccuracy:
    """Tests for GET /api/agent/{name}/accuracy endpoint."""

    def test_accuracy_no_nomic_dir(self, persona_handler):
        result = persona_handler.handle("/api/agent/claude/accuracy", {}, None)

        # Without nomic_dir, should return 503
        assert result is not None
        # Either 503 (module not available) or returns empty data
        data = json.loads(result.body)
        assert "error" in data or "total_positions" in data

    def test_accuracy_invalid_agent(self, persona_handler):
        result = persona_handler.handle("/api/agent/../test/accuracy", {}, None)

        assert result is not None
        assert result.status_code == 400


# ============================================================================
# Security Tests
# ============================================================================

class TestPersonaSecurity:
    """Security tests for persona endpoints."""

    def test_path_traversal_blocked_persona(self, persona_handler):
        result = persona_handler.handle("/api/agent/../../../etc/passwd/persona", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_performance(self, persona_handler):
        result = persona_handler.handle("/api/agent/..%2F..%2Fetc/performance", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_domains(self, persona_handler):
        result = persona_handler.handle("/api/agent/test..admin/domains", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, persona_handler):
        result = persona_handler.handle("/api/agent/'; DROP TABLE agents;--/persona", {}, None)
        assert result.status_code == 400

    def test_valid_agent_names_accepted(self, persona_handler):
        # Valid agent names should work
        valid_names = ["claude", "gpt-4", "gemini_pro", "agent123"]
        for name in valid_names:
            result = persona_handler.handle(f"/api/agent/{name}/persona", {}, None)
            # Should not be 400 (validation error)
            assert result.status_code != 400 or "invalid" not in json.loads(result.body).get("error", "").lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestPersonaErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, persona_handler):
        result = persona_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_handle_returns_none_for_incomplete_agent_path(self, persona_handler):
        # Path with only /api/agent/ should not match any endpoint
        result = persona_handler.handle("/api/agent/claude", {}, None)
        # Without a valid suffix like /persona, should return None
        assert result is None


# ============================================================================
# Edge Cases
# ============================================================================

class TestPersonaEdgeCases:
    """Tests for edge cases."""

    def test_empty_agent_name(self, persona_handler):
        # Empty agent name in path
        result = persona_handler.handle("/api/agent//persona", {}, None)
        # Should fail validation or return None
        assert result is None or result.status_code == 400

    def test_very_long_agent_name(self, persona_handler):
        long_name = "a" * 1000
        result = persona_handler.handle(f"/api/agent/{long_name}/persona", {}, None)
        # Should handle gracefully
        assert result is not None

    def test_special_characters_in_agent_name(self, persona_handler):
        result = persona_handler.handle("/api/agent/test<script>/persona", {}, None)
        assert result.status_code == 400

    def test_unicode_agent_name(self, persona_handler):
        result = persona_handler.handle("/api/agent/测试/persona", {}, None)
        # Should either accept or reject gracefully
        assert result is not None

"""
Tests for EvolutionHandler endpoints.

Endpoints tested:
- GET /api/evolution/{agent}/history - Get prompt evolution history for an agent
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.evolution import EvolutionHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_prompt_evolver():
    """Create a mock PromptEvolver."""
    evolver = Mock()
    evolver.get_evolution_history.return_value = [
        {
            "version": 3,
            "prompt": "You are an expert AI assistant...",
            "score": 0.85,
            "created_at": "2024-01-15T00:00:00Z",
            "parent_version": 2,
        },
        {
            "version": 2,
            "prompt": "You are a helpful assistant...",
            "score": 0.78,
            "created_at": "2024-01-10T00:00:00Z",
            "parent_version": 1,
        },
        {
            "version": 1,
            "prompt": "You are an assistant...",
            "score": 0.72,
            "created_at": "2024-01-05T00:00:00Z",
            "parent_version": None,
        },
    ]
    return evolver


@pytest.fixture
def evolution_handler(tmp_path):
    """Create an EvolutionHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": tmp_path,
    }
    return EvolutionHandler(ctx)


@pytest.fixture
def handler_no_nomic_dir():
    """Create an EvolutionHandler without nomic_dir."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return EvolutionHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestEvolutionRouting:
    """Tests for route matching."""

    def test_can_handle_evolution_history(self, evolution_handler):
        assert evolution_handler.can_handle("/api/evolution/claude/history") is True

    def test_can_handle_hyphenated_agent(self, evolution_handler):
        assert evolution_handler.can_handle("/api/evolution/gpt-4/history") is True

    def test_cannot_handle_base_route(self, evolution_handler):
        assert evolution_handler.can_handle("/api/evolution") is False

    def test_cannot_handle_agent_without_history(self, evolution_handler):
        assert evolution_handler.can_handle("/api/evolution/claude") is False

    def test_cannot_handle_unrelated_routes(self, evolution_handler):
        assert evolution_handler.can_handle("/api/agents") is False
        assert evolution_handler.can_handle("/api/evolution/claude/other") is False


# ============================================================================
# GET /api/evolution/{agent}/history Tests
# ============================================================================

class TestEvolutionHistory:
    """Tests for GET /api/evolution/{agent}/history endpoint."""

    def test_evolution_module_unavailable(self, evolution_handler):
        import aragora.server.handlers.evolution as mod
        original = mod.EVOLUTION_AVAILABLE
        mod.EVOLUTION_AVAILABLE = False
        try:
            result = evolution_handler.handle("/api/evolution/claude/history", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.EVOLUTION_AVAILABLE = original

    def test_evolution_no_nomic_dir(self, handler_no_nomic_dir):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        result = handler_no_nomic_dir.handle("/api/evolution/claude/history", {}, None)
        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "nomic" in data["error"].lower() or "not configured" in data["error"].lower()

    def test_evolution_history_success(self, evolution_handler, mock_prompt_evolver):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        with patch.object(mod, 'PromptEvolver', return_value=mock_prompt_evolver):
            result = evolution_handler.handle("/api/evolution/claude/history", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["agent"] == "claude"
            assert "history" in data
            assert data["count"] == 3
            assert len(data["history"]) == 3

    def test_evolution_history_with_limit(self, evolution_handler, mock_prompt_evolver):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        with patch.object(mod, 'PromptEvolver', return_value=mock_prompt_evolver):
            result = evolution_handler.handle("/api/evolution/claude/history", {"limit": "5"}, None)

            assert result is not None
            assert result.status_code == 200
            # Verify the limit was passed to the evolver
            mock_prompt_evolver.get_evolution_history.assert_called_with("claude", limit=5)

    def test_evolution_history_limit_clamped(self, evolution_handler, mock_prompt_evolver):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        with patch.object(mod, 'PromptEvolver', return_value=mock_prompt_evolver):
            # Request 0, should be clamped to 1
            result = evolution_handler.handle("/api/evolution/claude/history", {"limit": "0"}, None)

            assert result is not None
            assert result.status_code == 200
            mock_prompt_evolver.get_evolution_history.assert_called_with("claude", limit=1)

    def test_evolution_history_invalid_agent(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution/test..admin/history", {}, None)
        assert result is not None
        assert result.status_code == 400


# ============================================================================
# Security Tests
# ============================================================================

class TestEvolutionSecurity:
    """Security tests for evolution endpoints."""

    def test_path_traversal_blocked(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution/test..admin/history", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution/'; DROP TABLE--/history", {}, None)
        assert result.status_code == 400

    def test_xss_blocked(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution/<script>/history", {}, None)
        assert result.status_code == 400


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestEvolutionErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, evolution_handler):
        result = evolution_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_evolution_history_exception(self, evolution_handler):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        mock_evolver = Mock()
        mock_evolver.get_evolution_history.side_effect = Exception("Database error")

        with patch.object(mod, 'PromptEvolver', return_value=mock_evolver):
            result = evolution_handler.handle("/api/evolution/claude/history", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================

class TestEvolutionEdgeCases:
    """Tests for edge cases."""

    def test_empty_agent_name(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution//history", {}, None)
        # Empty name should fail validation or return error
        assert result is None or result.status_code == 400

    def test_very_long_agent_name(self, evolution_handler):
        long_name = "a" * 1000
        result = evolution_handler.handle(f"/api/evolution/{long_name}/history", {}, None)
        # Should handle gracefully
        assert result is not None

    def test_unicode_agent_name(self, evolution_handler):
        result = evolution_handler.handle("/api/evolution/测试/history", {}, None)
        # Should either accept or reject gracefully
        assert result is not None

    def test_evolution_history_empty(self, evolution_handler, mock_prompt_evolver):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        mock_prompt_evolver.get_evolution_history.return_value = []

        with patch.object(mod, 'PromptEvolver', return_value=mock_prompt_evolver):
            result = evolution_handler.handle("/api/evolution/claude/history", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["history"] == []
            assert data["count"] == 0

    def test_invalid_limit_param(self, evolution_handler, mock_prompt_evolver):
        import aragora.server.handlers.evolution as mod

        if not mod.EVOLUTION_AVAILABLE:
            pytest.skip("Evolution module not available")

        with patch.object(mod, 'PromptEvolver', return_value=mock_prompt_evolver):
            # Invalid param should use default
            result = evolution_handler.handle("/api/evolution/claude/history", {"limit": "invalid"}, None)

            assert result is not None
            assert result.status_code == 200

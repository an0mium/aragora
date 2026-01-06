"""
Tests for ProbesHandler endpoints.

Endpoints tested:
- POST /api/probes/capability - Run capability probes on an agent
- POST /api/probes/run - Legacy route for capability probes
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from aragora.server.handlers.probes import ProbesHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.record_redteam_result = Mock()
    return elo


@pytest.fixture
def mock_nomic_dir(tmp_path):
    """Create a mock nomic directory."""
    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir()
    return nomic_dir


@pytest.fixture
def probes_handler(mock_elo_system, mock_nomic_dir):
    """Create a ProbesHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": mock_elo_system,
        "nomic_dir": mock_nomic_dir,
    }
    return ProbesHandler(ctx)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with request body."""
    handler = Mock()
    handler.headers = {"Content-Length": "100"}
    handler.server = Mock()
    handler.server.stream_server = None
    return handler


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestProbesRouting:
    """Tests for route matching."""

    def test_can_handle_capability(self, probes_handler):
        assert probes_handler.can_handle("/api/probes/capability") is True

    def test_can_handle_run(self, probes_handler):
        # Legacy route
        assert probes_handler.can_handle("/api/probes/run") is True

    def test_cannot_handle_unrelated_routes(self, probes_handler):
        assert probes_handler.can_handle("/api/probes") is False
        assert probes_handler.can_handle("/api/probes/other") is False
        assert probes_handler.can_handle("/api/agents") is False


# ============================================================================
# POST /api/probes/capability Tests
# ============================================================================

class TestCapabilityProbe:
    """Tests for POST /api/probes/capability endpoint."""

    def test_probe_prober_unavailable(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        original = mod.PROBER_AVAILABLE
        mod.PROBER_AVAILABLE = False
        try:
            mock_handler.rfile = Mock()
            mock_handler.rfile.read.return_value = b'{"agent_name": "test"}'

            result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.PROBER_AVAILABLE = original

    def test_probe_agent_unavailable(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        original_prober = mod.PROBER_AVAILABLE
        original_agent = mod.AGENT_AVAILABLE
        mod.PROBER_AVAILABLE = True
        mod.AGENT_AVAILABLE = False
        try:
            mock_handler.rfile = Mock()
            mock_handler.rfile.read.return_value = b'{"agent_name": "test"}'

            result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.PROBER_AVAILABLE = original_prober
            mod.AGENT_AVAILABLE = original_agent

    def test_probe_missing_agent_name(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        if not mod.PROBER_AVAILABLE or not mod.AGENT_AVAILABLE:
            pytest.skip("Prober or agent module not available")

        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{}'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agent_name" in data["error"].lower()

    def test_probe_invalid_agent_name(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        if not mod.PROBER_AVAILABLE or not mod.AGENT_AVAILABLE:
            pytest.skip("Prober or agent module not available")

        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"agent_name": "../etc/passwd"}'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower()

    def test_probe_invalid_json(self, probes_handler, mock_handler):
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'not json'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result is not None
        assert result.status_code == 400

    def test_probe_empty_probe_types(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        if not mod.PROBER_AVAILABLE or not mod.AGENT_AVAILABLE:
            pytest.skip("Prober or agent module not available")

        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"agent_name": "test", "probe_types": ["invalid_type"]}'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "probe types" in data["error"].lower()

    def test_probe_legacy_route(self, probes_handler, mock_handler):
        """Test that legacy /api/probes/run route works."""
        import aragora.server.handlers.probes as mod

        original = mod.PROBER_AVAILABLE
        mod.PROBER_AVAILABLE = False
        try:
            mock_handler.rfile = Mock()
            mock_handler.rfile.read.return_value = b'{"agent_name": "test"}'

            # Should handle legacy route the same way
            result = probes_handler.handle_post("/api/probes/run", {}, mock_handler)
            assert result is not None
            assert result.status_code == 503  # Service unavailable
        finally:
            mod.PROBER_AVAILABLE = original


# ============================================================================
# Security Tests
# ============================================================================

class TestProbesSecurity:
    """Security tests for probes endpoints."""

    def test_path_traversal_blocked(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        if not mod.PROBER_AVAILABLE or not mod.AGENT_AVAILABLE:
            pytest.skip("Prober or agent module not available")

        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"agent_name": "..%2F..%2Fetc"}'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, probes_handler, mock_handler):
        import aragora.server.handlers.probes as mod

        if not mod.PROBER_AVAILABLE or not mod.AGENT_AVAILABLE:
            pytest.skip("Prober or agent module not available")

        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"agent_name": "\'; DROP TABLE agents;--"}'

        result = probes_handler.handle_post("/api/probes/capability", {}, mock_handler)
        assert result.status_code == 400


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestProbesErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_get(self, probes_handler):
        # ProbesHandler only handles POST
        result = probes_handler.handle("/api/probes/capability", {}, None)
        assert result is None

    def test_handle_returns_none_for_unhandled_route(self, probes_handler, mock_handler):
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{}'

        result = probes_handler.handle_post("/api/other/endpoint", {}, mock_handler)
        assert result is None


# ============================================================================
# Handler Import Tests
# ============================================================================

class TestProbesHandlerImport:
    """Test ProbesHandler import and export."""

    def test_handler_importable(self):
        """ProbesHandler can be imported from handlers package."""
        from aragora.server.handlers import ProbesHandler
        assert ProbesHandler is not None

    def test_handler_in_all_exports(self):
        """ProbesHandler is in __all__ exports."""
        from aragora.server.handlers import __all__
        assert "ProbesHandler" in __all__

"""Tests for the LaboratoryHandler class."""

import json
import pytest
from unittest.mock import Mock, patch


class TestLaboratoryHandlerRouting:
    """Test route matching for LaboratoryHandler."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.laboratory import LaboratoryHandler

        ctx = {}
        return LaboratoryHandler(ctx)

    def test_can_handle_emergent_traits(self, handler):
        assert handler.can_handle("/api/laboratory/emergent-traits") is True

    def test_can_handle_cross_pollinations(self, handler):
        assert handler.can_handle("/api/laboratory/cross-pollinations/suggest") is True

    def test_cannot_handle_unknown_route(self, handler):
        assert handler.can_handle("/api/laboratory/unknown") is False
        assert handler.can_handle("/api/other") is False


class TestEmergentTraitsEndpoint:
    """Test /api/laboratory/emergent-traits GET endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.laboratory import LaboratoryHandler

        ctx = {}
        return LaboratoryHandler(ctx)

    def test_emergent_traits_unavailable(self, handler):
        """Returns 503 when laboratory not available."""
        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False):
            result = handler.handle("/api/laboratory/emergent-traits", {})
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "error" in data

    def test_emergent_traits_success(self, handler):
        """Returns emergent traits on success."""
        mock_trait = Mock()
        mock_trait.agent_name = "test_agent"
        mock_trait.trait_name = "analytical"
        mock_trait.domain = "reasoning"
        mock_trait.confidence = 0.8
        mock_trait.evidence = ["won 5 debates"]
        mock_trait.detected_at = "2025-01-01"

        mock_lab = Mock()
        mock_lab.detect_emergent_traits.return_value = [mock_trait]

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory", return_value=mock_lab
            ):
                result = handler.handle("/api/laboratory/emergent-traits", {})
                assert result.status_code == 200
                data = json.loads(result.body)
                assert "emergent_traits" in data
                assert data["count"] == 1


class TestCrossPollinationsEndpoint:
    """Test /api/laboratory/cross-pollinations/suggest POST endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.laboratory import LaboratoryHandler

        ctx = {}
        return LaboratoryHandler(ctx)

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HTTP handler with request body."""
        handler = Mock()
        handler.headers = {"Content-Length": "50"}
        return handler

    def test_cross_pollinations_unavailable(self, handler, mock_handler):
        """Returns 503 when laboratory not available."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"target_agent": "test"}'

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False):
            result = handler.handle_post(
                "/api/laboratory/cross-pollinations/suggest", {}, mock_handler
            )
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "error" in data

    def test_cross_pollinations_missing_target(self, handler, mock_handler):
        """Returns 400 when target_agent is missing."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b"{}"

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch("aragora.server.handlers.laboratory.PersonaLaboratory"):
                result = handler.handle_post(
                    "/api/laboratory/cross-pollinations/suggest", {}, mock_handler
                )
                assert result.status_code == 400
                data = json.loads(result.body)
                assert "error" in data

    def test_cross_pollinations_success(self, handler, mock_handler):
        """Returns suggestions on success."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"target_agent": "claude"}'

        mock_lab = Mock()
        mock_lab.suggest_cross_pollinations.return_value = [
            ("gemini", "reasoning", "High performance in logic debates"),
        ]

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory", return_value=mock_lab
            ):
                result = handler.handle_post(
                    "/api/laboratory/cross-pollinations/suggest", {}, mock_handler
                )
                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["target_agent"] == "claude"
                assert data["count"] == 1
                assert data["suggestions"][0]["source_agent"] == "gemini"


class TestLaboratoryHandlerImport:
    """Test LaboratoryHandler import and export."""

    def test_handler_importable(self):
        """LaboratoryHandler can be imported from handlers package."""
        from aragora.server.handlers import LaboratoryHandler

        assert LaboratoryHandler is not None

    def test_handler_in_all_exports(self):
        """LaboratoryHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "LaboratoryHandler" in __all__

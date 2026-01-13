"""Tests for the VerificationHandler class."""

import json
import pytest
from unittest.mock import Mock, patch


class TestVerificationHandlerRouting:
    """Test route matching for VerificationHandler."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.verification import VerificationHandler

        ctx = {}
        return VerificationHandler(ctx)

    def test_can_handle_verification_status(self, handler):
        assert handler.can_handle("/api/verification/status") is True

    def test_cannot_handle_unknown_route(self, handler):
        assert handler.can_handle("/api/other") is False
        assert handler.can_handle("/api/verification/verify") is False


class TestVerificationStatusEndpoint:
    """Test /api/verification/status endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.verification import VerificationHandler

        ctx = {}
        return VerificationHandler(ctx)

    def test_status_when_unavailable(self, handler):
        """Returns unavailable status when formal verification not installed."""
        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", False):
            result = handler.handle("/api/verification/status", {}, None)
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["available"] is False
            assert "hint" in data
            assert data["backends"] == []

    def test_status_when_available(self, handler):
        """Returns status from verification manager when available."""
        mock_manager = Mock()
        mock_manager.status_report.return_value = {
            "any_available": True,
            "backends": [
                {"name": "z3", "available": True},
                {"name": "lean", "available": False},
            ],
        }

        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler.handle("/api/verification/status", {}, None)
                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["available"] is True
                assert len(data["backends"]) == 2

    def test_status_handles_exception(self, handler):
        """Returns error when exception occurs."""

        def raise_error():
            raise RuntimeError("Backend error")

        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.verification.get_formal_verification_manager",
                side_effect=raise_error,
            ):
                result = handler.handle("/api/verification/status", {}, None)
                assert result.status_code == 500
                data = json.loads(result.body)
                assert "error" in data


class TestVerificationFormalVerifyEndpoint:
    """Test /api/verification/formal-verify POST endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.verification import VerificationHandler

        ctx = {}
        return VerificationHandler(ctx)

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HTTP handler with request body."""
        handler = Mock()
        handler.headers = {"Content-Length": "50"}
        return handler

    def test_can_handle_formal_verify(self, handler):
        """Handler can handle formal-verify route."""
        assert handler.can_handle("/api/verification/formal-verify") is True

    def test_formal_verify_unavailable(self, handler, mock_handler):
        """Returns 503 when verification not available."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b'{"claim": "test"}'

        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", False):
            result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "error" in data

    def test_formal_verify_missing_claim(self, handler, mock_handler):
        """Returns 400 when claim is missing."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = b"{}"

        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", True):
            result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)
            assert result.status_code == 400
            data = json.loads(result.body)
            assert "error" in data
            assert "claim" in data["error"].lower()

    def test_formal_verify_success(self, handler, mock_handler):
        """Returns verification result on success."""
        mock_handler.rfile = Mock()
        mock_handler.rfile.read.return_value = (
            b'{"claim": "All integers greater than 0 are positive"}'
        )

        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "status": "proof_found",
            "is_verified": True,
        }

        mock_manager = Mock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification = Mock(return_value=mock_result)

        import asyncio

        async def mock_verify(*args, **kwargs):
            return mock_result

        mock_manager.attempt_formal_verification = mock_verify

        with patch("aragora.server.handlers.verification.FORMAL_VERIFICATION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler.handle_post("/api/verification/formal-verify", {}, mock_handler)
                assert result.status_code == 200
                data = json.loads(result.body)
                assert data["status"] == "proof_found"


class TestVerificationHandlerImport:
    """Test VerificationHandler import and export."""

    def test_handler_importable(self):
        """VerificationHandler can be imported from handlers package."""
        from aragora.server.handlers import VerificationHandler

        assert VerificationHandler is not None

    def test_handler_in_all_exports(self):
        """VerificationHandler is in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "VerificationHandler" in __all__

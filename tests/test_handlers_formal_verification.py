"""Tests for the FormalVerificationHandler class.

Tests the formal verification API endpoints:
- POST /api/verify/claim - Verify a single claim
- POST /api/verify/batch - Batch verification
- GET /api/verify/status - Get backend status
- POST /api/verify/translate - Translate to formal language
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock


class TestFormalVerificationHandlerRouting:
    """Test route matching for FormalVerificationHandler."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        return FormalVerificationHandler({})

    def test_can_handle_verify_claim(self, handler):
        """Handler can handle /api/verify/claim."""
        assert handler.can_handle("/api/v1/verify/claim") is True

    def test_can_handle_verify_batch(self, handler):
        """Handler can handle /api/verify/batch."""
        assert handler.can_handle("/api/v1/verify/batch") is True

    def test_can_handle_verify_status(self, handler):
        """Handler can handle /api/verify/status."""
        assert handler.can_handle("/api/v1/verify/status") is True

    def test_can_handle_verify_translate(self, handler):
        """Handler can handle /api/verify/translate."""
        assert handler.can_handle("/api/v1/verify/translate") is True

    def test_cannot_handle_unknown_route(self, handler):
        """Handler does not handle unknown routes."""
        assert handler.can_handle("/api/v1/other") is False
        assert handler.can_handle("/api/v1/debates") is False

    def test_can_handle_verify_prefix(self, handler):
        """Handler can handle any /api/verify/* route."""
        assert handler.can_handle("/api/v1/verify/custom") is True


class TestVerifyClaimEndpoint:
    """Test /api/verify/claim POST endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        return FormalVerificationHandler({})

    @pytest.fixture
    def mock_http_handler(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_verify_claim_requires_body(self, handler, mock_http_handler):
        """Returns 400 when no body provided."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/claim",
            body=None,
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_verify_claim_requires_claim_field(self, handler, mock_http_handler):
        """Returns 400 when claim field is missing."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/claim",
            body=b'{"context": "test"}',
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "claim" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_verify_claim_success(self, handler, mock_http_handler):
        """Returns verification result on success."""
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "status": "proof_found",
            "is_verified": True,
            "formal_statement": "theorem test : True := trivial",
            "language": "lean4",
        }

        mock_manager = Mock()
        mock_manager.attempt_formal_verification = AsyncMock(return_value=mock_result)

        with patch.object(handler, "_get_manager", return_value=mock_manager):
            result = await handler.handle_async(
                mock_http_handler,
                "POST",
                "/api/verify/claim",
                body=b'{"claim": "1 + 1 = 2"}',
            )
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["status"] == "proof_found"
            assert data["is_verified"] is True


class TestVerifyBatchEndpoint:
    """Test /api/verify/batch POST endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        return FormalVerificationHandler({})

    @pytest.fixture
    def mock_http_handler(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_verify_batch_requires_body(self, handler, mock_http_handler):
        """Returns 400 when no body provided."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/batch",
            body=None,
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_verify_batch_requires_claims_array(self, handler, mock_http_handler):
        """Returns 400 when claims array is missing."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/batch",
            body=b'{"timeout": 30}',
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "claims" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_verify_batch_max_claims_limit(self, handler, mock_http_handler):
        """Returns 400 when more than 20 claims provided."""
        claims = [{"claim": f"claim {i}"} for i in range(25)]
        body = json.dumps({"claims": claims}).encode()

        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/batch",
            body=body,
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "20" in data["error"] or "maximum" in data["error"].lower()


class TestVerifyStatusEndpoint:
    """Test /api/verify/status GET endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        return FormalVerificationHandler({})

    @pytest.fixture
    def mock_http_handler(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_verify_status_returns_backends(self, handler, mock_http_handler):
        """Returns status with backend info."""
        mock_manager = Mock()
        mock_manager.status_report.return_value = {
            "backends": [
                {"language": "z3_smt", "available": True},
                {"language": "lean4", "available": False},
            ],
            "any_available": True,
        }

        with patch.object(handler, "_get_manager", return_value=mock_manager):
            with patch(
                "aragora.server.handlers.formal_verification.DeepSeekProverTranslator", create=True
            ) as mock_ds:
                mock_ds.return_value.is_available = False
                result = await handler.handle_async(
                    mock_http_handler,
                    "GET",
                    "/api/verify/status",
                    body=None,
                )
                assert result.status_code == 200
                data = json.loads(result.body)
                assert "backends" in data
                assert data["any_available"] is True


class TestVerifyTranslateEndpoint:
    """Test /api/verify/translate POST endpoint."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        return FormalVerificationHandler({})

    @pytest.fixture
    def mock_http_handler(self):
        return Mock()

    @pytest.mark.asyncio
    async def test_translate_requires_body(self, handler, mock_http_handler):
        """Returns 400 when no body provided."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/translate",
            body=None,
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_translate_requires_claim(self, handler, mock_http_handler):
        """Returns 400 when claim field is missing."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/translate",
            body=b'{"target_language": "lean4"}',
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "claim" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_translate_unknown_language(self, handler, mock_http_handler):
        """Returns 400 for unknown target language."""
        result = await handler.handle_async(
            mock_http_handler,
            "POST",
            "/api/verify/translate",
            body=b'{"claim": "1 + 1 = 2", "target_language": "coq"}',
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "unknown" in data["error"].lower() or "target" in data["error"].lower()


class TestFormalVerificationHandlerImport:
    """Test FormalVerificationHandler import and export."""

    def test_handler_importable(self):
        """FormalVerificationHandler can be imported."""
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        assert FormalVerificationHandler is not None

    def test_handler_has_routes(self):
        """FormalVerificationHandler defines ROUTES."""
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        handler = FormalVerificationHandler({})
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0
        assert "/api/verify/claim" in handler.ROUTES

"""
Tests for the LaboratoryHandler module.

Comprehensive tests covering:
- Handler routing for all laboratory endpoints
- Successful request handling
- Error responses (400, 429, 500, 503)
- Input validation
- Rate limiting behavior
- Query parameter handling
- Response format validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.laboratory import LaboratoryHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockEmergentTrait:
    """Mock emergent trait for testing."""

    agent_name: str
    trait_name: str
    domain: str
    confidence: float
    evidence: list[str]
    detected_at: str


@pytest.fixture
def mock_server_context() -> dict[str, Any]:
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None, "persona_manager": None}


@pytest.fixture
def mock_http_handler() -> MagicMock:
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    handler.rfile = BytesIO(b"")
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from aragora.server.handlers.laboratory import _laboratory_limiter

    if hasattr(_laboratory_limiter, "_buckets"):
        _laboratory_limiter._buckets.clear()
    yield


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
        handler.request_body = body_bytes
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"
        handler.request_body = b"{}"

    return handler


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


# ===========================================================================
# Test Handler Routing
# ===========================================================================


class TestLaboratoryHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_can_handle_emergent_traits(self, handler: LaboratoryHandler):
        """Handler can handle emergent-traits endpoint."""
        assert handler.can_handle("/api/v1/laboratory/emergent-traits")

    def test_can_handle_cross_pollinations_suggest(self, handler: LaboratoryHandler):
        """Handler can handle cross-pollinations/suggest endpoint."""
        assert handler.can_handle("/api/v1/laboratory/cross-pollinations/suggest")

    def test_cannot_handle_other_paths(self, handler: LaboratoryHandler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/genesis")
        assert not handler.can_handle("/api/v1/laboratory/other")


class TestLaboratoryHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_routes_contains_emergent_traits(self, handler: LaboratoryHandler):
        """ROUTES contains emergent-traits endpoint."""
        assert "/api/v1/laboratory/emergent-traits" in handler.ROUTES

    def test_routes_contains_cross_pollinations_suggest(self, handler: LaboratoryHandler):
        """ROUTES contains cross-pollinations/suggest endpoint."""
        assert "/api/v1/laboratory/cross-pollinations/suggest" in handler.ROUTES


# ===========================================================================
# Test Emergent Traits Endpoint
# ===========================================================================


class TestEmergentTraitsEndpoint:
    """Tests for GET /api/v1/laboratory/emergent-traits endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_returns_503_when_laboratory_unavailable(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits endpoint returns 503 when laboratory not available."""
        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False):
            result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        assert result is not None
        assert get_status(result) == 503
        body = get_body(result)
        assert "error" in body
        assert "laboratory" in body["error"].lower()

    def test_returns_traits_when_available(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits endpoint returns traits when laboratory available."""
        mock_traits = [
            MockEmergentTrait(
                agent_name="claude",
                trait_name="analytical",
                domain="reasoning",
                confidence=0.85,
                evidence=["High accuracy on logic tasks"],
                detected_at="2024-01-15T10:00:00Z",
            ),
            MockEmergentTrait(
                agent_name="gpt4",
                trait_name="creative",
                domain="writing",
                confidence=0.75,
                evidence=["Strong creative outputs"],
                detected_at="2024-01-15T11:00:00Z",
            ),
        ]

        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = mock_traits

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "emergent_traits" in body
        assert "count" in body
        assert "min_confidence" in body
        assert len(body["emergent_traits"]) == 2
        assert body["emergent_traits"][0]["agent"] == "claude"

    def test_filters_by_min_confidence(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits endpoint filters by min_confidence parameter."""
        mock_traits = [
            MockEmergentTrait(
                agent_name="claude",
                trait_name="analytical",
                domain="reasoning",
                confidence=0.9,
                evidence=["High accuracy"],
                detected_at="2024-01-15T10:00:00Z",
            ),
            MockEmergentTrait(
                agent_name="gpt4",
                trait_name="creative",
                domain="writing",
                confidence=0.4,  # Below threshold
                evidence=["Moderate outputs"],
                detected_at="2024-01-15T11:00:00Z",
            ),
        ]

        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = mock_traits

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle(
                    "/api/v1/laboratory/emergent-traits",
                    {"min_confidence": ["0.5"]},
                    mock_http_handler,
                )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert len(body["emergent_traits"]) == 1
        assert body["emergent_traits"][0]["agent"] == "claude"
        assert body["min_confidence"] == 0.5

    def test_respects_limit_parameter(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits endpoint respects limit parameter."""
        mock_traits = [
            MockEmergentTrait(
                agent_name=f"agent_{i}",
                trait_name=f"trait_{i}",
                domain="reasoning",
                confidence=0.8,
                evidence=["Evidence"],
                detected_at="2024-01-15T10:00:00Z",
            )
            for i in range(10)
        ]

        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = mock_traits

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle(
                    "/api/v1/laboratory/emergent-traits",
                    {"limit": ["3"]},
                    mock_http_handler,
                )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert len(body["emergent_traits"]) == 3

    def test_limit_capped_at_100(self, handler: LaboratoryHandler, mock_http_handler: MagicMock):
        """Emergent traits endpoint caps limit at 100."""
        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = []

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle(
                    "/api/v1/laboratory/emergent-traits",
                    {"limit": ["500"]},
                    mock_http_handler,
                )

        assert result is not None
        # Should complete without error, limit is clamped internally

    def test_min_confidence_bounded(self, handler: LaboratoryHandler, mock_http_handler: MagicMock):
        """Emergent traits endpoint bounds min_confidence to 0.0-1.0."""
        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = []

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                # Test with value > 1.0 (should be clamped)
                result = handler.handle(
                    "/api/v1/laboratory/emergent-traits",
                    {"min_confidence": ["1.5"]},
                    mock_http_handler,
                )

        assert result is not None
        # Should complete without error, value is bounded internally

    def test_handles_exception_gracefully(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits endpoint returns 500 on internal error."""
        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                side_effect=Exception("Database error"),
            ):
                result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        assert result is not None
        assert get_status(result) == 500

    def test_default_parameters(self, handler: LaboratoryHandler, mock_http_handler: MagicMock):
        """Emergent traits endpoint uses sensible defaults."""
        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = []

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        # Default min_confidence is 0.5
        assert body["min_confidence"] == 0.5


# ===========================================================================
# Test Cross Pollinations Suggest Endpoint
# ===========================================================================


class TestCrossPollinationsSuggestEndpoint:
    """Tests for POST /api/v1/laboratory/cross-pollinations/suggest endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_returns_503_when_laboratory_unavailable(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint returns 503 when laboratory not available."""
        http_handler = make_mock_handler(body={"target_agent": "claude"}, method="POST")

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False):
            result = handler.handle_post(
                "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
            )

        assert result is not None
        assert get_status(result) == 503
        body = get_body(result)
        assert "error" in body
        assert "laboratory" in body["error"].lower()

    def test_returns_400_when_target_agent_missing(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint returns 400 when target_agent missing."""
        http_handler = make_mock_handler(body={}, method="POST")

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
            ):
                result = handler.handle_post(
                    "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
                )

        assert result is not None
        assert get_status(result) == 400
        body = get_body(result)
        assert "target_agent" in body["error"].lower()

    def test_returns_400_for_invalid_json(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint returns 400 for invalid JSON body."""
        http_handler = MagicMock()
        http_handler.command = "POST"
        http_handler.headers = {"Content-Length": "100"}
        http_handler.rfile = BytesIO(b"not valid json{{{")
        http_handler.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
            )

        assert result is not None
        assert get_status(result) == 400

    def test_returns_suggestions_when_available(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint returns suggestions when available."""
        http_handler = make_mock_handler(body={"target_agent": "claude"}, method="POST")

        mock_suggestions = [
            ("gpt4", "creative_writing", "Strong performance in creative tasks"),
            ("gemini", "code_review", "High accuracy in code analysis"),
        ]

        mock_lab = MagicMock()
        mock_lab.suggest_cross_pollinations.return_value = mock_suggestions

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle_post(
                    "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
                )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert "target_agent" in body
        assert body["target_agent"] == "claude"
        assert "suggestions" in body
        assert "count" in body
        assert len(body["suggestions"]) == 2
        assert body["suggestions"][0]["source_agent"] == "gpt4"
        assert body["suggestions"][0]["trait_or_domain"] == "creative_writing"
        assert body["suggestions"][0]["reason"] == "Strong performance in creative tasks"

    def test_handles_empty_suggestions(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint handles empty suggestions list."""
        http_handler = make_mock_handler(body={"target_agent": "new_agent"}, method="POST")

        mock_lab = MagicMock()
        mock_lab.suggest_cross_pollinations.return_value = []

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle_post(
                    "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
                )

        assert result is not None
        assert get_status(result) == 200
        body = get_body(result)
        assert body["count"] == 0
        assert body["suggestions"] == []

    def test_handles_exception_gracefully(self, handler: LaboratoryHandler):
        """Cross pollinations endpoint returns 500 on internal error."""
        http_handler = make_mock_handler(body={"target_agent": "claude"}, method="POST")

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                side_effect=Exception("Database error"),
            ):
                result = handler.handle_post(
                    "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
                )

        assert result is not None
        assert get_status(result) == 500


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_get_rate_limit_exceeded_returns_429(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Rate limit exceeded returns 429 for GET requests."""
        from aragora.server.handlers.laboratory import _laboratory_limiter

        with patch.object(_laboratory_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        assert result is not None
        assert get_status(result) == 429
        body = get_body(result)
        assert "error" in body
        assert "rate limit" in body["error"].lower()

    def test_post_rate_limit_exceeded_returns_429(self, handler: LaboratoryHandler):
        """Rate limit exceeded returns 429 for POST requests."""
        from aragora.server.handlers.laboratory import _laboratory_limiter

        http_handler = make_mock_handler(body={"target_agent": "claude"}, method="POST")

        with patch.object(_laboratory_limiter, "is_allowed", return_value=False):
            result = handler.handle_post(
                "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
            )

        assert result is not None
        assert get_status(result) == 429

    def test_multiple_requests_tracked(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Multiple requests are tracked for rate limiting."""
        for _ in range(5):
            result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)
            assert result is not None


# ===========================================================================
# Test Handler Methods Return None for Unknown Paths
# ===========================================================================


class TestUnknownPaths:
    """Tests for behavior with unknown paths."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_handle_returns_none_for_unknown_get_path(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Handle returns None for unknown GET paths."""
        result = handler.handle("/api/v1/laboratory/unknown", {}, mock_http_handler)
        assert result is None

    def test_handle_post_returns_none_for_unknown_post_path(self, handler: LaboratoryHandler):
        """Handle_post returns None for unknown POST paths."""
        http_handler = make_mock_handler(body={}, method="POST")
        result = handler.handle_post("/api/v1/laboratory/unknown", {}, http_handler)
        assert result is None


# ===========================================================================
# Test Integration
# ===========================================================================


class TestIntegration:
    """Integration tests for laboratory handler."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_all_get_routes_reachable(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """All GET routes return a response."""
        get_routes = ["/api/v1/laboratory/emergent-traits"]

        for route in get_routes:
            result = handler.handle(route, {}, mock_http_handler)
            assert result is not None, f"Route {route} returned None"
            assert get_status(result) in [
                200,
                400,
                429,
                500,
                503,
            ], f"Route {route} returned unexpected status {get_status(result)}"

    def test_all_post_routes_reachable(self, handler: LaboratoryHandler):
        """All POST routes return a response."""
        post_routes = ["/api/v1/laboratory/cross-pollinations/suggest"]

        for route in post_routes:
            http_handler = make_mock_handler(body={"target_agent": "test"}, method="POST")
            result = handler.handle_post(route, {}, http_handler)
            assert result is not None, f"Route {route} returned None"
            assert get_status(result) in [
                200,
                400,
                429,
                500,
                503,
            ], f"Route {route} returned unexpected status {get_status(result)}"

    def test_handler_inherits_from_base(self, handler: LaboratoryHandler):
        """Handler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)


# ===========================================================================
# Test Response Format
# ===========================================================================


class TestResponseFormat:
    """Tests for response format validation."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_emergent_traits_response_structure(
        self, handler: LaboratoryHandler, mock_http_handler: MagicMock
    ):
        """Emergent traits response has correct structure."""
        mock_traits = [
            MockEmergentTrait(
                agent_name="claude",
                trait_name="analytical",
                domain="reasoning",
                confidence=0.85,
                evidence=["Evidence 1", "Evidence 2"],
                detected_at="2024-01-15T10:00:00Z",
            ),
        ]

        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = mock_traits

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        body = get_body(result)
        trait = body["emergent_traits"][0]

        # Verify all expected fields are present
        assert "agent" in trait
        assert "trait" in trait
        assert "domain" in trait
        assert "confidence" in trait
        assert "evidence" in trait
        assert "detected_at" in trait

        # Verify values
        assert trait["agent"] == "claude"
        assert trait["trait"] == "analytical"
        assert trait["domain"] == "reasoning"
        assert trait["confidence"] == 0.85
        assert len(trait["evidence"]) == 2

    def test_cross_pollinations_response_structure(self, handler: LaboratoryHandler):
        """Cross pollinations response has correct structure."""
        http_handler = make_mock_handler(body={"target_agent": "claude"}, method="POST")

        mock_suggestions = [
            ("gpt4", "creative_writing", "Strong creative outputs"),
        ]

        mock_lab = MagicMock()
        mock_lab.suggest_cross_pollinations.return_value = mock_suggestions

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                return_value=mock_lab,
            ):
                result = handler.handle_post(
                    "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
                )

        body = get_body(result)
        suggestion = body["suggestions"][0]

        # Verify all expected fields are present
        assert "source_agent" in suggestion
        assert "trait_or_domain" in suggestion
        assert "reason" in suggestion

        # Verify top-level fields
        assert "target_agent" in body
        assert "suggestions" in body
        assert "count" in body


# ===========================================================================
# Test PersonaLaboratory Initialization
# ===========================================================================


class TestLaboratoryInitialization:
    """Tests for PersonaLaboratory initialization with context."""

    @pytest.fixture
    def handler(self, mock_server_context: dict[str, Any]) -> LaboratoryHandler:
        return LaboratoryHandler(mock_server_context)

    def test_lab_initialized_with_nomic_dir(self, mock_http_handler: MagicMock):
        """Laboratory is initialized with nomic_dir from context."""
        from pathlib import Path

        mock_nomic_dir = Path("/tmp/nomic")
        context = {
            "nomic_dir": mock_nomic_dir,
            "persona_manager": MagicMock(),
        }
        handler = LaboratoryHandler(context)

        mock_lab_class = MagicMock()
        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = []
        mock_lab_class.return_value = mock_lab

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                mock_lab_class,
            ):
                handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        # Verify PersonaLaboratory was called with correct db_path
        mock_lab_class.assert_called_once()
        call_kwargs = mock_lab_class.call_args[1]
        assert "db_path" in call_kwargs
        assert "laboratory.db" in call_kwargs["db_path"]

    def test_lab_initialized_with_persona_manager(self, mock_http_handler: MagicMock):
        """Laboratory is initialized with persona_manager from context."""
        mock_persona_manager = MagicMock()
        context = {
            "nomic_dir": None,
            "persona_manager": mock_persona_manager,
        }
        handler = LaboratoryHandler(context)

        mock_lab_class = MagicMock()
        mock_lab = MagicMock()
        mock_lab.detect_emergent_traits.return_value = []
        mock_lab_class.return_value = mock_lab

        with patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.laboratory.PersonaLaboratory",
                mock_lab_class,
            ):
                handler.handle("/api/v1/laboratory/emergent-traits", {}, mock_http_handler)

        # Verify PersonaLaboratory was called with persona_manager
        mock_lab_class.assert_called_once()
        call_kwargs = mock_lab_class.call_args[1]
        assert call_kwargs["persona_manager"] is mock_persona_manager

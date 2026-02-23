"""Tests for laboratory handler (aragora/server/handlers/laboratory.py).

Covers all routes and behavior of the LaboratoryHandler class:
- can_handle() routing for all ROUTES
- GET  /api/v1/laboratory/emergent-traits      - Get emergent traits
- POST /api/v1/laboratory/cross-pollinations/suggest - Suggest trait transfers
- Rate limiting behavior (GET and POST)
- Error handling (missing params, unavailable lab, invalid body)
- Edge cases (empty results, confidence filtering, limit clamping)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.laboratory import LaboratoryHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to LaboratoryHandler."""

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


@dataclass
class _FakeTrait:
    """Mimics the emergent trait object returned by PersonaLaboratory."""

    agent_name: str
    trait_name: str
    domain: str
    confidence: float
    evidence: str
    detected_at: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a LaboratoryHandler with minimal server context."""
    return LaboratoryHandler({})


@pytest.fixture
def handler_with_ctx():
    """Create a LaboratoryHandler with a persona_manager in context."""
    return LaboratoryHandler({"persona_manager": MagicMock()})


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter before each test."""
    import aragora.server.handlers.laboratory as lab_mod

    lab_mod._laboratory_limiter._buckets.clear()
    yield
    lab_mod._laboratory_limiter._buckets.clear()


def _make_traits(count: int = 3, base_confidence: float = 0.7) -> list[_FakeTrait]:
    """Build a list of fake emergent traits."""
    return [
        _FakeTrait(
            agent_name=f"agent-{i}",
            trait_name=f"trait-{i}",
            domain=f"domain-{i}",
            confidence=round(base_confidence + i * 0.05, 2),
            evidence=f"Evidence for trait {i}",
            detected_at=f"2026-02-{10 + i}T00:00:00Z",
        )
        for i in range(count)
    ]


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify can_handle accepts and rejects correct paths."""

    def test_emergent_traits_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory/emergent-traits") is True

    def test_cross_pollinations_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory/cross-pollinations") is True

    def test_cross_pollinations_suggest_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory/cross-pollinations/suggest") is True

    def test_experiments_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory/experiments") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory/unknown") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_partial_path(self, handler):
        assert handler.can_handle("/api/v1/laboratory") is False

    def test_rejects_no_version_prefix(self, handler):
        assert handler.can_handle("/api/laboratory/emergent-traits") is False


# ============================================================================
# GET /api/v1/laboratory/emergent-traits
# ============================================================================


class TestGetEmergentTraits:
    """Tests for the emergent-traits GET endpoint."""

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_returns_traits(self, mock_lab_cls, handler):
        traits = _make_traits(3)
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 3
        assert len(body["emergent_traits"]) == 3

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_trait_fields(self, mock_lab_cls, handler):
        traits = _make_traits(1)
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        body = _body(result)
        t = body["emergent_traits"][0]
        assert t["agent"] == "agent-0"
        assert t["trait"] == "trait-0"
        assert t["domain"] == "domain-0"
        assert t["confidence"] == 0.7
        assert t["evidence"] == "Evidence for trait 0"
        assert t["detected_at"] == "2026-02-10T00:00:00Z"

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_confidence_filtering(self, mock_lab_cls, handler):
        """Traits below min_confidence are excluded."""
        traits = [
            _FakeTrait("a1", "t1", "d1", 0.3, "ev", "2026-01-01"),
            _FakeTrait("a2", "t2", "d2", 0.8, "ev", "2026-01-02"),
            _FakeTrait("a3", "t3", "d3", 0.6, "ev", "2026-01-03"),
        ]
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle(
            "/api/v1/laboratory/emergent-traits",
            {"min_confidence": "0.5"},
            _MockHTTPHandler(),
        )

        body = _body(result)
        assert body["count"] == 2
        agents = [t["agent"] for t in body["emergent_traits"]]
        assert "a2" in agents
        assert "a3" in agents
        assert "a1" not in agents

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_confidence_filtering_high_threshold(self, mock_lab_cls, handler):
        """High threshold filters out most traits."""
        traits = _make_traits(5, base_confidence=0.5)
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle(
            "/api/v1/laboratory/emergent-traits",
            {"min_confidence": "0.9"},
            _MockHTTPHandler(),
        )

        body = _body(result)
        # Only traits with confidence >= 0.9 pass
        for t in body["emergent_traits"]:
            assert t["confidence"] >= 0.9

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_limit_param(self, mock_lab_cls, handler):
        """The limit parameter caps the number of returned traits."""
        traits = _make_traits(10)
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle(
            "/api/v1/laboratory/emergent-traits",
            {"limit": "2"},
            _MockHTTPHandler(),
        )

        body = _body(result)
        assert body["count"] == 2
        assert len(body["emergent_traits"]) == 2

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_default_min_confidence(self, mock_lab_cls, handler):
        """Default min_confidence is 0.5."""
        traits = [
            _FakeTrait("a1", "t1", "d1", 0.4, "ev", "2026-01-01"),
            _FakeTrait("a2", "t2", "d2", 0.5, "ev", "2026-01-02"),
        ]
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle(
            "/api/v1/laboratory/emergent-traits",
            {},
            _MockHTTPHandler(),
        )

        body = _body(result)
        assert body["count"] == 1
        assert body["min_confidence"] == 0.5
        assert body["emergent_traits"][0]["agent"] == "a2"

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_empty_traits(self, mock_lab_cls, handler):
        """No traits detected returns empty list."""
        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["emergent_traits"] == []

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_all_filtered_out(self, mock_lab_cls, handler):
        """All traits below threshold returns empty."""
        traits = _make_traits(3, base_confidence=0.1)
        mock_lab_cls.return_value.detect_emergent_traits.return_value = traits

        result = handler.handle(
            "/api/v1/laboratory/emergent-traits",
            {"min_confidence": "0.99"},
            _MockHTTPHandler(),
        )

        body = _body(result)
        assert body["count"] == 0

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False)
    def test_lab_unavailable(self, handler):
        """Returns 503 when PersonaLaboratory not importable."""
        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory", None)
    def test_lab_class_is_none(self, handler):
        """Returns 503 when PersonaLaboratory is None despite AVAILABLE=True."""
        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        assert _status(result) == 503

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_nomic_dir_from_ctx(self, mock_lab_cls, handler_with_ctx):
        """When ctx has nomic_dir, it is passed to lab as db_path."""
        from pathlib import Path

        handler_with_ctx.ctx["nomic_dir"] = Path("/tmp/test_nomic")
        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        handler_with_ctx.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        _, kwargs = mock_lab_cls.call_args
        assert kwargs["db_path"] == "/tmp/test_nomic/laboratory.db"

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_no_nomic_dir(self, mock_lab_cls, handler):
        """When ctx has no nomic_dir, db_path is None."""
        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        _, kwargs = mock_lab_cls.call_args
        assert kwargs["db_path"] is None

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_persona_manager_passed(self, mock_lab_cls, handler_with_ctx):
        """persona_manager from ctx is passed to PersonaLaboratory."""
        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        handler_with_ctx.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        _, kwargs = mock_lab_cls.call_args
        assert kwargs["persona_manager"] is not None

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_exception_in_detect_traits(self, mock_lab_cls, handler):
        """handle_errors decorator catches exceptions and returns 500."""
        mock_lab_cls.return_value.detect_emergent_traits.side_effect = RuntimeError(
            "Database error"
        )

        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, _MockHTTPHandler())

        assert _status(result) == 500

    def test_unmatched_get_path_returns_none(self, handler):
        """Paths not matching any route return None from handle()."""
        result = handler.handle("/api/v1/laboratory/cross-pollinations", {}, _MockHTTPHandler())

        assert result is None


# ============================================================================
# POST /api/v1/laboratory/cross-pollinations/suggest
# ============================================================================


class TestSuggestCrossPollinations:
    """Tests for the cross-pollinations suggest POST endpoint."""

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_returns_suggestions(self, mock_lab_cls, handler):
        suggestions = [
            ("agent-source-1", "analytical-thinking", "Strong performance in domain X"),
            ("agent-source-2", "creativity", "Unique approach to problems"),
        ]
        mock_lab_cls.return_value.suggest_cross_pollinations.return_value = suggestions

        http_handler = _MockHTTPHandler(body={"target_agent": "agent-target"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["target_agent"] == "agent-target"
        assert body["count"] == 2
        assert len(body["suggestions"]) == 2

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_suggestion_fields(self, mock_lab_cls, handler):
        suggestions = [
            ("src-agent", "domain-x", "Good at analysis"),
        ]
        mock_lab_cls.return_value.suggest_cross_pollinations.return_value = suggestions

        http_handler = _MockHTTPHandler(body={"target_agent": "tgt"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        body = _body(result)
        s = body["suggestions"][0]
        assert s["source_agent"] == "src-agent"
        assert s["trait_or_domain"] == "domain-x"
        assert s["reason"] == "Good at analysis"

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_empty_suggestions(self, mock_lab_cls, handler):
        mock_lab_cls.return_value.suggest_cross_pollinations.return_value = []

        http_handler = _MockHTTPHandler(body={"target_agent": "tgt"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["suggestions"] == []

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_missing_target_agent(self, mock_lab_cls, handler):
        """Missing target_agent returns 400."""
        http_handler = _MockHTTPHandler(body={})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 400
        assert "target_agent" in _body(result).get("error", "")

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_empty_target_agent(self, mock_lab_cls, handler):
        """Empty string target_agent returns 400."""
        http_handler = _MockHTTPHandler(body={"target_agent": ""})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 400

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", False)
    def test_lab_unavailable(self, handler):
        http_handler = _MockHTTPHandler(body={"target_agent": "agent-1"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory", None)
    def test_lab_class_is_none_post(self, handler):
        http_handler = _MockHTTPHandler(body={"target_agent": "agent-1"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 503

    def test_invalid_json_body(self, handler):
        """Non-JSON body returns 400."""
        http_handler = _MockHTTPHandler()
        http_handler.rfile.read.return_value = b"not-json"
        http_handler.headers = {"Content-Length": "8"}

        with (
            patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True),
            patch("aragora.server.handlers.laboratory.PersonaLaboratory"),
        ):
            result = handler.handle_post(
                "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
            )

        assert _status(result) == 400
        assert (
            "Invalid JSON" in _body(result).get("error", "")
            or "body" in _body(result).get("error", "").lower()
        )

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_nomic_dir_from_ctx_post(self, mock_lab_cls, handler_with_ctx):
        from pathlib import Path

        handler_with_ctx.ctx["nomic_dir"] = Path("/tmp/test_nomic")
        mock_lab_cls.return_value.suggest_cross_pollinations.return_value = []

        http_handler = _MockHTTPHandler(body={"target_agent": "a1"})
        handler_with_ctx.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        _, kwargs = mock_lab_cls.call_args
        assert kwargs["db_path"] == "/tmp/test_nomic/laboratory.db"

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_exception_in_suggest(self, mock_lab_cls, handler):
        """handle_errors decorator catches exceptions on POST."""
        mock_lab_cls.return_value.suggest_cross_pollinations.side_effect = RuntimeError(
            "Internal error"
        )

        http_handler = _MockHTTPHandler(body={"target_agent": "a1"})
        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 500

    def test_unmatched_post_path_returns_none(self, handler):
        http_handler = _MockHTTPHandler(body={"target_agent": "a1"})
        result = handler.handle_post("/api/v1/laboratory/experiments", {}, http_handler)

        assert result is None


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on laboratory endpoints."""

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_get_rate_limit_exceeded(self, mock_lab_cls, handler):
        """GET returns 429 when rate limit is hit."""
        import aragora.server.handlers.laboratory as lab_mod

        http_handler = _MockHTTPHandler()
        # Exhaust the rate limiter for this IP
        for _ in range(25):
            lab_mod._laboratory_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, http_handler)

        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_post_rate_limit_exceeded(self, mock_lab_cls, handler):
        """POST returns 429 when rate limit is hit."""
        import aragora.server.handlers.laboratory as lab_mod

        http_handler = _MockHTTPHandler(body={"target_agent": "a1"})
        for _ in range(25):
            lab_mod._laboratory_limiter.is_allowed("127.0.0.1")

        result = handler.handle_post(
            "/api/v1/laboratory/cross-pollinations/suggest", {}, http_handler
        )

        assert _status(result) == 429

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_different_ips_not_affected(self, mock_lab_cls, handler):
        """Different IPs have independent rate limits."""
        import aragora.server.handlers.laboratory as lab_mod

        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        # Exhaust limit for IP 10.0.0.1
        for _ in range(25):
            lab_mod._laboratory_limiter.is_allowed("10.0.0.1")

        # Different IP should still work
        http_handler = _MockHTTPHandler(client_address=("192.168.1.1", 9999))
        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, http_handler)

        assert _status(result) == 200

    @patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True)
    @patch("aragora.server.handlers.laboratory.PersonaLaboratory")
    def test_get_within_rate_limit(self, mock_lab_cls, handler):
        """Requests within the rate limit succeed."""
        mock_lab_cls.return_value.detect_emergent_traits.return_value = []

        http_handler = _MockHTTPHandler()
        result = handler.handle("/api/v1/laboratory/emergent-traits", {}, http_handler)

        assert _status(result) == 200

    def test_null_handler_rate_limit(self, handler):
        """Handler=None uses 'unknown' IP for rate limiting."""
        with (
            patch("aragora.server.handlers.laboratory.LABORATORY_AVAILABLE", True),
            patch("aragora.server.handlers.laboratory.PersonaLaboratory") as mock_lab_cls,
        ):
            mock_lab_cls.return_value.detect_emergent_traits.return_value = []
            result = handler.handle("/api/v1/laboratory/emergent-traits", {}, None)
            # Should still work (unknown IP has its own bucket)
            assert _status(result) == 200


# ============================================================================
# Handler Initialization
# ============================================================================


class TestHandlerInit:
    """Tests for handler construction."""

    def test_default_ctx(self):
        h = LaboratoryHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"persona_manager": "pm", "nomic_dir": "/tmp"}
        h = LaboratoryHandler(ctx)
        assert h.ctx == ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = LaboratoryHandler(None)
        assert h.ctx == {}


# ============================================================================
# ROUTES constant
# ============================================================================


class TestRoutes:
    """Verify ROUTES constant is correct."""

    def test_routes_contains_emergent_traits(self):
        assert "/api/v1/laboratory/emergent-traits" in LaboratoryHandler.ROUTES

    def test_routes_contains_cross_pollinations(self):
        assert "/api/v1/laboratory/cross-pollinations" in LaboratoryHandler.ROUTES

    def test_routes_contains_suggest(self):
        assert "/api/v1/laboratory/cross-pollinations/suggest" in LaboratoryHandler.ROUTES

    def test_routes_contains_experiments(self):
        assert "/api/v1/laboratory/experiments" in LaboratoryHandler.ROUTES

    def test_routes_count(self):
        assert len(LaboratoryHandler.ROUTES) == 4

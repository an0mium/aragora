"""Comprehensive tests for VerificationHandler.

Tests cover:
- Route matching (can_handle) for all ROUTES
- GET /api/v1/verification/status - backend status endpoint
  - When formal verification is NOT available (module not installed)
  - When formal verification IS available (with mocked manager)
  - Manager status_report returning various shapes
- POST /api/v1/verification/formal-verify - claim verification
  - Formal verification unavailable returns 503
  - Invalid/missing JSON body returns 400
  - Schema validation failures (missing claim, empty claim, too-long claim, too-long context)
  - Successful verification with proof_found status
  - Various failure statuses: translation_failed, not_supported, proof_failed, timeout
  - proof_failed with counterexample (proof_text present)
  - Timeout capping at 120
  - _safe_float utility
  - claim_type passthrough
  - context default
  - Backend unavailable (any_available=False) returns 503
- handle() routing: dispatches GET to _get_status, returns None for unknown
- handle_post() routing: dispatches POST to _verify_claim, returns None for unknown
- Handler construction: __init__ with/without ctx
- RBAC permission decorators are present (verified via conftest auto-auth bypass)
- Edge cases: empty body, whitespace claim, non-string status on result
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed into handle/handle_post."""

    def __init__(self, method="GET", body=None, client_address=None):
        self.command = method
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


class _MockStatus(Enum):
    """Mock verification status enum."""

    PROOF_FOUND = "proof_found"
    PROOF_FAILED = "proof_failed"
    TRANSLATION_FAILED = "translation_failed"
    NOT_SUPPORTED = "not_supported"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a VerificationHandler instance."""
    from aragora.server.handlers.verification.verification import VerificationHandler

    return VerificationHandler(ctx={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler with no body."""
    return _MockHTTPHandler()


@pytest.fixture
def mock_http_with_body():
    """Factory for mock HTTP handlers with a JSON body."""

    def _create(body: dict[str, Any]) -> _MockHTTPHandler:
        return _MockHTTPHandler(method="POST", body=body)

    return _create


# =============================================================================
# _safe_float utility
# =============================================================================


class TestSafeFloat:
    """Test the _safe_float utility function."""

    def test_valid_int(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(42) == 42.0

    def test_valid_float(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(3.14) == 3.14

    def test_valid_string_number(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float("99.5") == 99.5

    def test_invalid_string(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float("not_a_number", 5.0) == 5.0

    def test_none_value(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(None, 7.0) == 7.0

    def test_default_is_zero(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float("bad") == 0.0

    def test_empty_string(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float("", 10.0) == 10.0

    def test_boolean_true(self):
        from aragora.server.handlers.verification.verification import _safe_float

        # bool is subclass of int in Python, float(True) == 1.0
        assert _safe_float(True) == 1.0

    def test_boolean_false(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(False) == 0.0

    def test_list_returns_default(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float([1, 2], 99.0) == 99.0

    def test_dict_returns_default(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float({"a": 1}, 42.0) == 42.0

    def test_negative_number(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(-15.5) == -15.5

    def test_zero(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float(0) == 0.0

    def test_string_negative(self):
        from aragora.server.handlers.verification.verification import _safe_float

        assert _safe_float("-3.7") == -3.7


# =============================================================================
# Route Matching (can_handle)
# =============================================================================


class TestCanHandle:
    """Test route matching logic for all declared ROUTES."""

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/verification/status") is True

    def test_formal_verify_route(self, handler):
        assert handler.can_handle("/api/v1/verification/formal-verify") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/verification/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/verification") is False

    def test_extra_suffix(self, handler):
        assert handler.can_handle("/api/v1/verification/status/extra") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/verification/status") is False

    def test_no_api_prefix(self, handler):
        assert handler.can_handle("/verification/status") is False

    def test_routes_list(self, handler):
        from aragora.server.handlers.verification.verification import VerificationHandler

        assert "/api/v1/verification/status" in VerificationHandler.ROUTES
        assert "/api/v1/verification/formal-verify" in VerificationHandler.ROUTES
        assert len(VerificationHandler.ROUTES) == 2


# =============================================================================
# Handler Construction
# =============================================================================


class TestHandlerConstruction:
    """Test handler instantiation patterns."""

    def test_init_with_ctx(self):
        from aragora.server.handlers.verification.verification import VerificationHandler

        ctx = {"key": "value"}
        h = VerificationHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none_ctx(self):
        from aragora.server.handlers.verification.verification import VerificationHandler

        h = VerificationHandler(ctx=None)
        assert h.ctx == {}

    def test_init_default_ctx(self):
        from aragora.server.handlers.verification.verification import VerificationHandler

        h = VerificationHandler()
        assert h.ctx == {}


# =============================================================================
# GET /api/v1/verification/status - handle() routing
# =============================================================================


class TestHandleGetRouting:
    """Test handle() GET request routing."""

    def test_status_route_dispatches(self, handler):
        """GET to /api/v1/verification/status returns a result."""
        with patch.object(handler, "_get_status", return_value=MagicMock(status_code=200)) as mock_fn:
            result = handler.handle("/api/v1/verification/status", {}, MagicMock())
            mock_fn.assert_called_once()

    def test_unknown_route_returns_none(self, handler):
        """GET to unknown route returns None."""
        result = handler.handle("/api/v1/verification/unknown", {}, MagicMock())
        assert result is None

    def test_formal_verify_route_not_handled_by_get(self, handler):
        """GET to formal-verify route returns None (POST only)."""
        result = handler.handle("/api/v1/verification/formal-verify", {}, MagicMock())
        assert result is None


# =============================================================================
# POST /api/v1/verification/formal-verify - handle_post() routing
# =============================================================================


class TestHandlePostRouting:
    """Test handle_post() POST request routing."""

    def test_formal_verify_route_dispatches(self, handler, mock_http_with_body):
        """POST to /api/v1/verification/formal-verify dispatches to _verify_claim."""
        mock_http = mock_http_with_body({"claim": "test"})
        with patch.object(handler, "_verify_claim", return_value=MagicMock(status_code=200)) as mock_fn:
            result = handler.handle_post("/api/v1/verification/formal-verify", {}, mock_http)
            mock_fn.assert_called_once_with(mock_http)

    def test_unknown_route_returns_none(self, handler, mock_http):
        """POST to unknown route returns None."""
        result = handler.handle_post("/api/v1/verification/unknown", {}, mock_http)
        assert result is None

    def test_status_route_not_handled_by_post(self, handler, mock_http):
        """POST to status route returns None (GET only)."""
        result = handler.handle_post("/api/v1/verification/status", {}, mock_http)
        assert result is None


# =============================================================================
# GET /api/v1/verification/status - _get_status()
# =============================================================================


class TestGetStatus:
    """Test the _get_status method for backend availability."""

    def test_formal_verification_not_available(self, handler):
        """When FORMAL_VERIFICATION_AVAILABLE is False, return unavailable status."""
        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            False,
        ):
            result = handler._get_status()
            assert _status(result) == 200
            data = _body(result)
            assert data["available"] is False
            assert "hint" in data
            assert "z3-solver" in data["hint"]
            assert data["backends"] == []

    def test_formal_verification_available_with_backends(self, handler):
        """When formal verification IS available, return manager status."""
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "any_available": True,
            "backends": [
                {"name": "z3", "available": True},
                {"name": "lean", "available": False},
            ],
        }

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler._get_status()
                assert _status(result) == 200
                data = _body(result)
                assert data["available"] is True
                assert len(data["backends"]) == 2

    def test_formal_verification_available_no_backends(self, handler):
        """When formal verification module exists but no backends available."""
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "any_available": False,
            "backends": [],
        }

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler._get_status()
                assert _status(result) == 200
                data = _body(result)
                assert data["available"] is False
                assert data["backends"] == []

    def test_status_report_missing_keys_use_defaults(self, handler):
        """When status_report returns dict with missing keys, defaults used."""
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler._get_status()
                assert _status(result) == 200
                data = _body(result)
                assert data["available"] is False
                assert data["backends"] == []


# =============================================================================
# POST /api/v1/verification/formal-verify - _verify_claim()
# =============================================================================


class TestVerifyClaim:
    """Test the _verify_claim method for formal verification."""

    def test_not_available_returns_503(self, handler, mock_http_with_body):
        """When FORMAL_VERIFICATION_AVAILABLE is False, return 503."""
        mock_http = mock_http_with_body({"claim": "test"})
        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            False,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 503
            data = _body(result)
            assert "not available" in data["error"].lower()
            assert "z3-solver" in data["hint"]

    def test_invalid_json_body_returns_400(self, handler):
        """When JSON body is invalid, return 400."""
        mock_http = _MockHTTPHandler(method="POST")
        mock_http.headers = {"Content-Length": "10"}
        mock_http.rfile.read.return_value = b"not json!!"

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400
            data = _body(result)
            assert "invalid" in data["error"].lower() or "json" in data["error"].lower()

    def test_body_too_large_returns_400(self, handler):
        """When body exceeds MAX_BODY_SIZE, read_json_body returns None."""
        mock_http = _MockHTTPHandler(method="POST")
        # Set extremely large content-length
        mock_http.headers = {"Content-Length": str(100_000_000)}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_empty_body_fails_schema_validation(self, handler):
        """When body is empty dict, schema validation fails (claim required)."""
        mock_http = _MockHTTPHandler(method="POST")
        # Empty body -> read_json_body returns {} -> schema validates claim as required

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_missing_claim_fails_schema_validation(self, handler, mock_http_with_body):
        """When claim field is missing, schema validation fails."""
        mock_http = mock_http_with_body({"context": "some context"})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_empty_claim_fails_schema_validation(self, handler, mock_http_with_body):
        """When claim is empty string, schema validation fails (min_length=1)."""
        mock_http = mock_http_with_body({"claim": ""})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_claim_too_long_fails_schema_validation(self, handler):
        """When claim exceeds max_length=5000, schema validation fails."""
        long_claim = "x" * 5001
        mock_http = _MockHTTPHandler(method="POST", body={"claim": long_claim})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_context_too_long_fails_schema_validation(self, handler):
        """When context exceeds max_length=10000, schema validation fails."""
        long_context = "y" * 10001
        mock_http = _MockHTTPHandler(method="POST", body={"claim": "valid claim", "context": long_context})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            assert _status(result) == 400

    def test_backends_not_available_returns_503(self, handler, mock_http_with_body):
        """When no backends are available, return 503."""
        mock_http = mock_http_with_body({"claim": "1+1=2"})
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "any_available": False,
            "backends": [{"name": "z3", "available": False}],
        }

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                result = handler._verify_claim(mock_http)
                assert _status(result) == 503
                data = _body(result)
                assert "no formal verification backends" in data["error"].lower()
                assert "backends" in data

    def test_successful_proof_found(self, handler, mock_http_with_body):
        """Successful verification with proof_found status."""
        mock_http = mock_http_with_body({"claim": "1+1=2"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "status": "proof_found",
            "is_verified": True,
            "formal_statement": "(assert (= (+ 1 1) 2))",
            "proof_hash": "abc123",
        }
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    assert _status(result) == 200
                    data = _body(result)
                    assert data["status"] == "proof_found"
                    assert data["is_verified"] is True
                    assert data["claim"] == "1+1=2"
                    # No hint for proof_found
                    assert "hint" not in data

    def test_claim_is_stripped(self, handler, mock_http_with_body):
        """Claim text is stripped of leading/trailing whitespace."""
        mock_http = mock_http_with_body({"claim": "  test claim  "})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    data = _body(result)
                    assert data["claim"] == "test claim"

    def test_claim_type_included_when_present(self, handler, mock_http_with_body):
        """claim_type is included in response when provided."""
        mock_http = mock_http_with_body({"claim": "test", "claim_type": "arithmetic"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    data = _body(result)
                    assert data["claim_type"] == "arithmetic"

    def test_claim_type_omitted_when_none(self, handler, mock_http_with_body):
        """claim_type is NOT included in response when not provided."""
        mock_http = mock_http_with_body({"claim": "test"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    data = _body(result)
                    assert "claim_type" not in data

    def test_context_defaults_to_empty_string(self, handler, mock_http_with_body):
        """When context is not provided, default is empty string."""
        mock_http = mock_http_with_body({"claim": "test"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ) as mock_run:
                    handler._verify_claim(mock_http)
                    # Verify run_async was called with the manager's coroutine
                    mock_manager.attempt_formal_verification.assert_called_once()
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["context"] == ""

    def test_timeout_default_is_30(self, handler, mock_http_with_body):
        """When timeout is not provided, default is 30."""
        mock_http = mock_http_with_body({"claim": "test"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == 30.0

    def test_timeout_capped_at_120(self, handler, mock_http_with_body):
        """Timeout is capped at 120 seconds."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": 500})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == 120.0

    def test_timeout_normal_value_passes_through(self, handler, mock_http_with_body):
        """Normal timeout values pass through unchanged."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": 60})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == 60.0

    def test_timeout_invalid_defaults_to_30(self, handler, mock_http_with_body):
        """When timeout is invalid (non-numeric string), defaults to 30."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": "bad"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    # _safe_float("bad", 30.0) -> 30.0, min(30.0, 120.0) -> 30.0
                    assert call_kwargs["timeout_seconds"] == 30.0

    def test_timeout_none_defaults_to_30(self, handler, mock_http_with_body):
        """When timeout is None, default of 30 is used."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": None})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    # _safe_float(None, 30.0) -> 30.0
                    assert call_kwargs["timeout_seconds"] == 30.0


# =============================================================================
# Response Hints by Status
# =============================================================================


class TestResponseHints:
    """Test user-friendly hints added for various verification statuses."""

    def _run_verify_with_status(self, handler, status_enum, result_dict):
        """Helper to run _verify_claim with a given status."""
        mock_http = _MockHTTPHandler(method="POST", body={"claim": "test claim"})
        mock_result = MagicMock()
        mock_result.to_dict.return_value = result_dict
        mock_result.status = status_enum

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    return handler._verify_claim(mock_http)

    def test_translation_failed_hint(self, handler):
        """translation_failed status includes hint, examples, and suggestions."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.TRANSLATION_FAILED,
            {"status": "translation_failed", "is_verified": False},
        )
        data = _body(result)
        assert "hint" in data
        assert "rephrasing" in data["hint"].lower()
        assert "examples" in data
        assert len(data["examples"]) == 4
        assert "suggestions" in data
        assert len(data["suggestions"]) == 4

    def test_not_supported_hint(self, handler):
        """not_supported status includes hint and supported_types."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.NOT_SUPPORTED,
            {"status": "not_supported", "is_verified": False},
        )
        data = _body(result)
        assert "hint" in data
        assert "not suitable" in data["hint"].lower()
        assert "supported_types" in data
        assert len(data["supported_types"]) == 4

    def test_proof_failed_hint(self, handler):
        """proof_failed status includes hint."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.PROOF_FAILED,
            {"status": "proof_failed", "is_verified": False},
        )
        data = _body(result)
        assert "hint" in data
        assert "false or unprovable" in data["hint"].lower()
        # No counterexample note when no proof_text
        assert "counterexample_note" not in data

    def test_proof_failed_with_counterexample(self, handler):
        """proof_failed with proof_text includes counterexample_note."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.PROOF_FAILED,
            {
                "status": "proof_failed",
                "is_verified": False,
                "proof_text": "counterexample: x = 0",
            },
        )
        data = _body(result)
        assert "counterexample_note" in data
        assert "counterexample was found" in data["counterexample_note"].lower()

    def test_timeout_hint(self, handler):
        """timeout status includes hint and max_timeout."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.TIMEOUT,
            {"status": "timeout", "is_verified": False},
        )
        data = _body(result)
        assert "hint" in data
        assert "time limit" in data["hint"].lower()
        assert data["max_timeout"] == 120

    def test_proof_found_no_hint(self, handler):
        """proof_found status does NOT include hint."""
        result = self._run_verify_with_status(
            handler,
            _MockStatus.PROOF_FOUND,
            {"status": "proof_found", "is_verified": True},
        )
        data = _body(result)
        assert "hint" not in data

    def test_status_without_value_attribute(self, handler):
        """When status has no .value attribute, str() is used."""
        mock_http = _MockHTTPHandler(method="POST", body={"claim": "test"})
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "custom_status", "is_verified": False}
        # status without .value -- remove the attribute
        mock_result.status = "custom_string_status"

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    # Should not raise and should return a valid response
                    assert _status(result) == 200


# =============================================================================
# Verification Manager Invocation
# =============================================================================


class TestVerificationManagerInvocation:
    """Test that the manager is invoked correctly with the right arguments."""

    def test_claim_passed_to_manager(self, handler, mock_http_with_body):
        """Claim text is passed to manager.attempt_formal_verification."""
        mock_http = mock_http_with_body({"claim": "For all x, x + 0 = x"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["claim"] == "For all x, x + 0 = x"

    def test_all_params_passed_to_manager(self, handler, mock_http_with_body):
        """All parameters are correctly passed to manager."""
        mock_http = mock_http_with_body({
            "claim": "A implies B",
            "claim_type": "logical",
            "context": "boolean logic",
            "timeout": 45,
        })

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["claim"] == "A implies B"
                    assert call_kwargs["claim_type"] == "logical"
                    assert call_kwargs["context"] == "boolean logic"
                    assert call_kwargs["timeout_seconds"] == 45.0

    def test_run_async_is_used(self, handler, mock_http_with_body):
        """run_async is called to bridge sync/async."""
        mock_http = mock_http_with_body({"claim": "test"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ) as mock_run:
                    handler._verify_claim(mock_http)
                    mock_run.assert_called_once()


# =============================================================================
# Integration: handle() and handle_post() end-to-end
# =============================================================================


class TestEndToEnd:
    """End-to-end tests through handle/handle_post."""

    def test_get_status_through_handle(self, handler):
        """GET /api/v1/verification/status through handle() returns response."""
        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            False,
        ):
            result = handler.handle("/api/v1/verification/status", {}, MagicMock())
            assert _status(result) == 200
            data = _body(result)
            assert data["available"] is False

    def test_post_formal_verify_through_handle_post(self, handler, mock_http_with_body):
        """POST /api/v1/verification/formal-verify through handle_post()."""
        mock_http = mock_http_with_body({"claim": "1+1=2"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler.handle_post("/api/v1/verification/formal-verify", {}, mock_http)
                    assert _status(result) == 200
                    data = _body(result)
                    assert data["is_verified"] is True

    def test_post_unavailable_through_handle_post(self, handler, mock_http_with_body):
        """POST when verification unavailable returns 503 through handle_post()."""
        mock_http = mock_http_with_body({"claim": "test"})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            False,
        ):
            result = handler.handle_post("/api/v1/verification/formal-verify", {}, mock_http)
            assert _status(result) == 503


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_handler_inherits_base_handler(self, handler):
        """VerificationHandler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_handler_has_read_json_body(self, handler):
        """Handler has read_json_body method from BaseHandler."""
        assert hasattr(handler, "read_json_body")

    def test_timeout_zero_passes_through(self, handler, mock_http_with_body):
        """Timeout of 0 passes through as 0.0."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": 0})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == 0.0

    def test_timeout_exactly_120(self, handler, mock_http_with_body):
        """Timeout of exactly 120 passes through unchanged."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": 120})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == 120.0

    def test_timeout_negative_passes_through(self, handler, mock_http_with_body):
        """Negative timeout passes through (min with 120 still negative)."""
        mock_http = mock_http_with_body({"claim": "test", "timeout": -5})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}
        mock_manager.attempt_formal_verification.return_value = mock_result

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    handler._verify_claim(mock_http)
                    call_kwargs = mock_manager.attempt_formal_verification.call_args[1]
                    assert call_kwargs["timeout_seconds"] == -5.0

    def test_claim_max_length_exactly_5000(self, handler, mock_http_with_body):
        """Claim of exactly 5000 chars passes schema validation."""
        claim = "x" * 5000

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        mock_http = _MockHTTPHandler(method="POST", body={"claim": claim})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    assert _status(result) == 200

    def test_context_max_length_exactly_10000(self, handler, mock_http_with_body):
        """Context of exactly 10000 chars passes schema validation."""
        context = "y" * 10000

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        mock_http = _MockHTTPHandler(method="POST", body={"claim": "test", "context": context})

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    assert _status(result) == 200

    def test_unicode_claim(self, handler, mock_http_with_body):
        """Unicode claims are handled correctly."""
        mock_http = mock_http_with_body({"claim": "Pour tout x, x + 0 = x"})

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    assert _status(result) == 200
                    data = _body(result)
                    assert "Pour tout" in data["claim"]

    def test_no_content_length_returns_empty_body(self, handler):
        """When Content-Length is 0, read_json_body returns empty dict."""
        mock_http = _MockHTTPHandler(method="POST")
        mock_http.headers = {"Content-Length": "0"}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            result = handler._verify_claim(mock_http)
            # Empty body -> schema validation fails (claim required)
            assert _status(result) == 400

    def test_extra_fields_in_body_ignored(self, handler, mock_http_with_body):
        """Extra fields not in schema are ignored gracefully."""
        mock_http = mock_http_with_body({
            "claim": "test",
            "extra_field": "should be ignored",
            "another": 123,
        })

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockStatus.PROOF_FOUND

        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": True}

        with patch(
            "aragora.server.handlers.verification.verification.FORMAL_VERIFICATION_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.server.handlers.verification.verification.get_formal_verification_manager",
                return_value=mock_manager,
            ):
                with patch(
                    "aragora.server.handlers.verification.verification.run_async",
                    return_value=mock_result,
                ):
                    result = handler._verify_claim(mock_http)
                    assert _status(result) == 200


# =============================================================================
# Module-level Constants
# =============================================================================


class TestModuleConstants:
    """Test module-level constants and imports."""

    def test_formal_verification_available_is_bool(self):
        from aragora.server.handlers.verification.verification import FORMAL_VERIFICATION_AVAILABLE

        assert isinstance(FORMAL_VERIFICATION_AVAILABLE, bool)

    def test_get_formal_verification_manager_exists(self):
        from aragora.server.handlers.verification.verification import get_formal_verification_manager

        # It may be None if the module is not available, or a function
        assert get_formal_verification_manager is None or callable(get_formal_verification_manager)

    def test_verification_schema_structure(self):
        from aragora.server.validation.schema import VERIFICATION_SCHEMA

        assert "claim" in VERIFICATION_SCHEMA
        assert VERIFICATION_SCHEMA["claim"]["type"] == "string"
        assert VERIFICATION_SCHEMA["claim"]["required"] is True
        assert VERIFICATION_SCHEMA["claim"]["min_length"] == 1
        assert VERIFICATION_SCHEMA["claim"]["max_length"] == 5000
        assert "context" in VERIFICATION_SCHEMA
        assert VERIFICATION_SCHEMA["context"]["required"] is False

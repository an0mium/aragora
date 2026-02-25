"""
Tests for aragora.server.handlers.security_debate - Security Debate HTTP Handlers.

Tests cover:
- SecurityDebateHandler: instantiation, ROUTES, can_handle
- GET /api/v1/audit/security/debate/{id}: returns not_found status
- POST /api/v1/audit/security/debate: body validation, success, module unavailable
- handle routing: returns None for unmatched paths
- handle_post routing: returns None for unmatched paths
"""

from __future__ import annotations

import functools
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# We need to patch decorators before importing the handler class, because
# @rate_limit and @require_permission are applied at class definition time.
def _passthrough_decorator(*args, **kwargs):
    """A no-op decorator that passes through the function unchanged.

    Uses functools.wraps to preserve __wrapped__ attribute, preventing
    test pollution when other tests check for decorator metadata.
    """
    if len(args) == 1 and callable(args[0]):

        @functools.wraps(args[0])
        def passthrough(*a, **kw):
            return args[0](*a, **kw)

        return passthrough

    def wrapper(func):
        @functools.wraps(func)
        def inner(*a, **kw):
            return func(*a, **kw)

        return inner

    return wrapper


with patch("aragora.server.middleware.rate_limit.rate_limit", _passthrough_decorator):
    with patch("aragora.rbac.decorators.require_permission", _passthrough_decorator):
        # Force re-import with patched decorators
        import importlib

        import aragora.server.handlers.security_debate as _sd_module

        importlib.reload(_sd_module)
        SecurityDebateHandler = _sd_module.SecurityDebateHandler

from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
        "Authorization": "Bearer test-token",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Security Objects
# ===========================================================================


class MockDebateResult:
    """Mock security debate result."""

    def __init__(self):
        self.debate_id = "debate-sec-001"
        self.consensus_reached = True
        self.confidence = 0.85
        self.final_answer = "Patch the vulnerability immediately"
        self.rounds_used = 3
        self.votes = []


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a SecurityDebateHandler with mocked dependencies."""
    h = SecurityDebateHandler(server_context={})
    return h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestSecurityDebateHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, SecurityDebateHandler)

    def test_routes(self, handler):
        assert "/api/v1/audit/security/debate" in handler.ROUTES
        assert "/api/v1/audit/security/debate/:id" in handler.ROUTES

    def test_can_handle_debate(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate") is True

    def test_can_handle_debate_id(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate/abc-123") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/v1/audit/findings") is False

    def test_prefix(self, handler):
        assert handler._PREFIX == "/api/v1/audit/security/debate"


# ===========================================================================
# Test GET /api/v1/audit/security/debate/{id}
# ===========================================================================


class TestGetDebateStatus:
    """Tests for getting debate status."""

    def test_get_status_returns_not_found(self, handler):
        result = handler.get_api_v1_audit_security_debate_id("debate-sec-001")
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "debate-sec-001"
        assert data["status"] == "not_found"
        assert "not persisted" in data["message"].lower()

    def test_get_status_arbitrary_id(self, handler):
        result = handler.get_api_v1_audit_security_debate_id("any-id-works")
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "any-id-works"


# ===========================================================================
# Test POST /api/v1/audit/security/debate
# ===========================================================================


class TestPostSecurityDebate:
    """Tests for triggering a security debate."""

    def test_post_no_body(self, handler):
        with patch.object(handler, "get_json_body", return_value=None):
            result = handler.post_api_v1_audit_security_debate()
            assert result.status_code == 400

    def test_post_empty_findings(self, handler):
        with patch.object(handler, "get_json_body", return_value={"findings": []}):
            result = handler.post_api_v1_audit_security_debate()
            assert result.status_code == 400

    def test_post_findings_not_array(self, handler):
        with patch.object(handler, "get_json_body", return_value={"findings": "not-an-array"}):
            result = handler.post_api_v1_audit_security_debate()
            assert result.status_code == 400

    def test_post_module_unavailable(self, handler):
        findings = [
            {
                "severity": "high",
                "title": "SQL Injection",
                "description": "User input not sanitized",
            }
        ]
        with patch.object(handler, "get_json_body", return_value={"findings": findings}):
            with patch(
                "aragora.server.handlers.security_debate.SecurityDebateHandler.post_api_v1_audit_security_debate",
                wraps=handler.post_api_v1_audit_security_debate,
            ):
                # Simulate ImportError by patching the import within the method
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.debate.security_debate": None,
                        "aragora.events.security_events": None,
                    },
                ):
                    result = handler.post_api_v1_audit_security_debate()
                    assert result.status_code == 500

    def test_post_success(self, handler):
        findings = [
            {
                "severity": "critical",
                "title": "RCE Vulnerability",
                "description": "Remote code execution via deserialization",
                "file_path": "app/utils.py",
                "line_number": 42,
            }
        ]

        mock_result = MockDebateResult()

        # We need to mock both the security event types and the debate runner
        mock_security_finding = MagicMock()
        mock_security_event = MagicMock()
        mock_security_event.id = "evt-001"

        mock_severity_cls = MagicMock()
        mock_severity_cls.CRITICAL = MagicMock()
        mock_severity_cls.HIGH = MagicMock()
        mock_severity_cls.MEDIUM = MagicMock()
        mock_severity_cls.return_value = mock_severity_cls.CRITICAL

        mock_event_type_cls = MagicMock()
        mock_event_type_cls.SAST_CRITICAL = MagicMock()
        mock_event_type_cls.VULNERABILITY_DETECTED = MagicMock()

        mock_finding_obj = MagicMock()
        mock_finding_obj.severity = mock_severity_cls.CRITICAL

        with patch.object(handler, "get_json_body", return_value={"findings": findings}):
            with patch.dict("sys.modules", {}):
                with patch(
                    "aragora.server.handlers.security_debate.run_async",
                    return_value=mock_result,
                ) as mock_run_async:
                    # Patch the imports inside the method
                    mock_sec_debate = MagicMock()
                    mock_sec_events = MagicMock()
                    mock_sec_events.SecuritySeverity = mock_severity_cls
                    mock_sec_events.SecurityEventType = mock_event_type_cls
                    mock_sec_events.SecurityFinding = MagicMock(return_value=mock_finding_obj)
                    mock_sec_events.SecurityEvent = MagicMock(return_value=mock_security_event)

                    with patch.dict(
                        "sys.modules",
                        {
                            "aragora.debate.security_debate": mock_sec_debate,
                            "aragora.events.security_events": mock_sec_events,
                        },
                    ):
                        result = handler.post_api_v1_audit_security_debate()
                        assert result.status_code == 200
                        data = _parse_body(result)
                        assert data["status"] == "completed"
                        assert data["consensus_reached"] is True
                        assert data["findings_analyzed"] == 1


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_get_debate_id(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/audit/security/debate/abc-123", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["debate_id"] == "abc-123"

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        # The base path alone doesn't match GET (no ID segment)
        result = handler.handle("/api/v1/audit/security/debate", {}, mock_handler)
        assert result is None

    def test_handle_too_short_path(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/audit/security", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() method routing."""

    def test_handle_post_debate(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.object(handler, "get_json_body", return_value=None):
            result = handler.handle_post("/api/v1/audit/security/debate", {}, mock_handler)
            assert result is not None
            assert result.status_code == 400  # No body

    def test_handle_post_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("POST")
        result = handler.handle_post("/api/v1/audit/security/debate/abc-123", {}, mock_handler)
        assert result is None

    def test_handle_post_trailing_slash(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.object(handler, "get_json_body", return_value=None):
            result = handler.handle_post("/api/v1/audit/security/debate/", {}, mock_handler)
            # Trailing slash stripped should still match
            assert result is not None

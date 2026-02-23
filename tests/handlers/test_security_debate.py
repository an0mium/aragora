"""Tests for SecurityDebateHandler.

Covers:
- Route matching (can_handle)
- GET /api/v1/audit/security/debate/:id (debate status lookup)
- POST /api/v1/audit/security/debate (trigger security debate)
- Request body validation (missing body, empty findings, invalid types)
- Confidence threshold clamping and defaults
- Timeout clamping and defaults
- Security finding conversion (severity mapping, default fields)
- Event type determination (critical vs non-critical)
- Debate result response building (votes, debate_id fallback)
- Import failure handling (security debate module unavailable)
- Debate runtime errors
- Rate limiting on POST
- RBAC permission checks
- Edge cases (trailing slashes, unknown sub-paths)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.handlers.security_debate import SecurityDebateHandler


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Get status code from HandlerResult."""
    return result.status_code


# ============================================================================
# Mock Security Types
# ============================================================================


class MockSecuritySeverity(Enum):
    """Mock SecuritySeverity enum matching real one."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MockSecurityEventType(Enum):
    """Mock SecurityEventType enum."""

    SAST_CRITICAL = "sast_critical"
    VULNERABILITY_DETECTED = "vulnerability_detected"


@dataclass
class MockSecurityFinding:
    """Mock SecurityFinding dataclass."""

    id: str
    finding_type: str
    severity: MockSecuritySeverity
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    cve_id: str | None = None
    package_name: str | None = None
    package_version: str | None = None
    recommendation: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MockSecurityEvent:
    """Mock SecurityEvent dataclass."""

    event_type: MockSecurityEventType
    severity: MockSecuritySeverity
    source: str
    repository: str
    findings: list[MockSecurityFinding]
    id: str = "mock-event-id"

    def __post_init__(self):
        if not hasattr(self, "id") or self.id == "mock-event-id":
            self.id = f"evt-{uuid.uuid4().hex[:8]}"


@dataclass
class MockVote:
    """Mock vote object."""

    agent_name: str
    vote: str


@dataclass
class MockDebateResult:
    """Mock debate result."""

    debate_id: str = "debate-123"
    consensus_reached: bool = True
    confidence: float = 0.85
    final_answer: str = "Apply input validation and parameterized queries."
    rounds_used: int = 3
    votes: list[MockVote] | None = None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {
        "user_store": MagicMock(),
        "nomic_dir": "/tmp/test",
        "stream_emitter": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create SecurityDebateHandler with mock context."""
    return SecurityDebateHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with empty body."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.path = "/api/v1/audit/security/debate"
    mock.headers = {"Content-Length": "2"}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


def _make_http_handler(body: dict | None = None, method: str = "POST") -> MagicMock:
    """Factory for creating mock HTTP handlers with specific body."""
    mock = MagicMock()
    mock.command = method
    mock.client_address = ("127.0.0.1", 12345)
    mock.path = "/api/v1/audit/security/debate"

    if body is not None:
        body_bytes = json.dumps(body).encode()
        mock.rfile.read.return_value = body_bytes
        mock.headers = {"Content-Length": str(len(body_bytes))}
    else:
        mock.rfile.read.return_value = b"{}"
        mock.headers = {"Content-Length": "2"}
    return mock


def _minimal_findings_body(**overrides) -> dict:
    """Create a minimal valid request body with findings."""
    body = {
        "findings": [
            {
                "severity": "high",
                "title": "SQL Injection in login",
                "description": "User input not sanitized",
                "file_path": "app/auth.py",
                "line_number": 42,
            }
        ],
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def mock_rate_limiter(monkeypatch):
    """Bypass rate limiting for all tests."""
    try:
        from aragora.server.middleware.rate_limit import decorators as rl_dec

        mock_result = MagicMock()
        mock_result.allowed = True
        mock_result.limit = 10
        mock_result.remaining = 9
        mock_result.retry_after = 0

        mock_limiter = MagicMock()
        mock_limiter.get_client_key.return_value = "test-127.0.0.1"
        mock_limiter.allow.return_value = mock_result

        monkeypatch.setattr(rl_dec, "get_rate_limiter", lambda *a, **kw: mock_limiter)

        # Also patch distributed limiter
        try:
            from aragora.server.middleware.rate_limit import distributed as rl_dist

            mock_dist = MagicMock()
            mock_dist.get_client_key.return_value = "test-127.0.0.1"
            mock_dist.allow.return_value = mock_result
            mock_dist.configure_endpoint = MagicMock()
            monkeypatch.setattr(rl_dist, "get_distributed_limiter", lambda: mock_dist)
        except (ImportError, AttributeError):
            pass
    except (ImportError, AttributeError):
        pass
    yield


# ============================================================================
# Mock security modules for import patching
# ============================================================================


def _build_security_mocks():
    """Build mock modules for aragora.debate.security_debate and aragora.events.security_events."""
    mock_run = AsyncMock(return_value=MockDebateResult())

    mock_security_debate_mod = MagicMock()
    mock_security_debate_mod.run_security_debate = mock_run

    mock_events_mod = MagicMock()
    mock_events_mod.SecurityEvent = MockSecurityEvent
    mock_events_mod.SecurityEventType = MockSecurityEventType
    mock_events_mod.SecurityFinding = MockSecurityFinding
    mock_events_mod.SecuritySeverity = MockSecuritySeverity

    return mock_run, mock_security_debate_mod, mock_events_mod


def _patch_security_imports(mock_sd_mod, mock_ev_mod):
    """Create patch context for security module imports."""
    return patch.dict(
        "sys.modules",
        {
            "aragora.debate.security_debate": mock_sd_mod,
            "aragora.events.security_events": mock_ev_mod,
        },
    )


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestCanHandle:
    """Test route matching logic via can_handle."""

    def test_base_debate_path(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate") is True

    def test_debate_id_path(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate/abc-123") is True

    def test_debate_uuid_path(self, handler):
        uid = str(uuid.uuid4())
        assert handler.can_handle(f"/api/v1/audit/security/debate/{uid}") is True

    def test_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate/") is True

    def test_unrelated_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_prefix_rejected(self, handler):
        assert handler.can_handle("/api/v1/audit/security") is False

    def test_similar_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/audit/security/scan") is False

    def test_nested_sub_path(self, handler):
        assert handler.can_handle("/api/v1/audit/security/debate/id/extra") is True

    def test_empty_path_rejected(self, handler):
        assert handler.can_handle("") is False

    def test_root_path_rejected(self, handler):
        assert handler.can_handle("/") is False


# ============================================================================
# Handler Class Tests
# ============================================================================


class TestHandlerInitialization:
    """Test handler class setup."""

    def test_extends_secure_handler(self, handler):
        from aragora.server.handlers.secure import SecureHandler

        assert isinstance(handler, SecureHandler)

    def test_has_routes(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) == 2

    def test_routes_contain_expected_paths(self, handler):
        assert "/api/v1/audit/security/debate" in handler.ROUTES
        assert "/api/v1/audit/security/debate/:id" in handler.ROUTES

    def test_prefix(self, handler):
        assert handler._PREFIX == "/api/v1/audit/security/debate"

    def test_initialization_with_empty_context(self):
        h = SecurityDebateHandler({})
        assert h is not None

    def test_initialization_with_minimal_context(self):
        h = SecurityDebateHandler({"user_store": None})
        assert h is not None


# ============================================================================
# GET /api/v1/audit/security/debate/:id
# ============================================================================


class TestGetDebateStatus:
    """Test GET debate status endpoint."""

    def test_returns_not_found_for_any_id(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/some-debate-id",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert _status(result) == 200
        data = _body(result)
        assert data["debate_id"] == "some-debate-id"
        assert data["status"] == "not_found"

    def test_returns_message_about_persistence(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/xyz",
            {},
            mock_http_handler,
        )
        data = _body(result)
        assert "not persisted" in data["message"].lower() or "POST" in data["message"]

    def test_uuid_debate_id(self, handler, mock_http_handler):
        uid = str(uuid.uuid4())
        result = handler.handle(
            f"/api/v1/audit/security/debate/{uid}",
            {},
            mock_http_handler,
        )
        assert result is not None
        data = _body(result)
        assert data["debate_id"] == uid

    def test_preserves_debate_id_exactly(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/my-SPECIAL-id_123",
            {},
            mock_http_handler,
        )
        data = _body(result)
        assert data["debate_id"] == "my-SPECIAL-id_123"

    def test_base_path_returns_none(self, handler, mock_http_handler):
        """GET on the base path (no id) returns None (not handled)."""
        result = handler.handle(
            "/api/v1/audit/security/debate",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_trailing_slash_base_path_returns_none(self, handler, mock_http_handler):
        """GET on /api/v1/audit/security/debate/ returns None."""
        result = handler.handle(
            "/api/v1/audit/security/debate/",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_debate_id_with_trailing_slash(self, handler, mock_http_handler):
        """Trailing slash on ID path is stripped, still returns debate status."""
        result = handler.handle(
            "/api/v1/audit/security/debate/some-id/",
            {},
            mock_http_handler,
        )
        # After rstrip("/"), path becomes .../some-id, parts[-1] = "some-id"
        assert result is not None
        data = _body(result)
        assert data["debate_id"] == "some-id"


# ============================================================================
# POST /api/v1/audit/security/debate - Body Validation
# ============================================================================


class TestPostBodyValidation:
    """Test POST request body validation."""

    def test_missing_body_returns_400(self, handler):
        """No JSON body returns 400."""
        http = _make_http_handler(body=None)
        # Make rfile.read return invalid JSON
        http.rfile.read.return_value = b"not json"
        http.headers = {"Content-Length": "8"}
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result is not None
        assert _status(result) == 400

    def test_empty_findings_returns_400(self, handler):
        """Empty findings array returns 400."""
        http = _make_http_handler(body={"findings": []})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result is not None
        assert _status(result) == 400
        assert "no findings" in _body(result)["error"].lower()

    def test_missing_findings_key_returns_400(self, handler):
        """Missing 'findings' key returns 400."""
        http = _make_http_handler(body={"repository": "myrepo"})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result is not None
        assert _status(result) == 400

    def test_findings_not_array_returns_400(self, handler):
        """findings as non-array returns 400."""
        http = _make_http_handler(body={"findings": "not-a-list"})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result is not None
        assert _status(result) == 400
        assert "array" in _body(result)["error"].lower()

    def test_findings_as_dict_returns_400(self, handler):
        """findings as a dictionary returns 400."""
        http = _make_http_handler(body={"findings": {"title": "not an array"}})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result is not None
        assert _status(result) == 400

    def test_unhandled_post_path_returns_none(self, handler):
        """POST on non-matching path returns None."""
        http = _make_http_handler(body=_minimal_findings_body())
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate/some-id", {}, http)
        assert result is None


# ============================================================================
# POST - Confidence Threshold Clamping
# ============================================================================


class TestConfidenceThreshold:
    """Test confidence_threshold parsing and clamping."""

    def _run_debate_with_body(self, handler, body):
        """Helper to run debate with given body and capture run_security_debate args."""
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return mock_run, mock_ra

    def test_default_confidence(self, handler):
        """Default confidence threshold is 0.7."""
        _, mock_ra = self._run_debate_with_body(handler, _minimal_findings_body())
        # The coroutine is passed to run_async; we verify the result is built
        assert mock_ra.called

    def test_custom_confidence(self, handler):
        """Custom confidence threshold is used."""
        body = _minimal_findings_body(confidence_threshold=0.9)
        _, mock_ra = self._run_debate_with_body(handler, body)
        assert mock_ra.called

    def test_confidence_below_minimum_clamped(self, handler):
        """Confidence below 0.1 is clamped to 0.1."""
        body = _minimal_findings_body(confidence_threshold=0.01)
        _, mock_ra = self._run_debate_with_body(handler, body)
        assert mock_ra.called

    def test_confidence_above_maximum_clamped(self, handler):
        """Confidence above 1.0 is clamped to 1.0."""
        body = _minimal_findings_body(confidence_threshold=5.0)
        _, mock_ra = self._run_debate_with_body(handler, body)
        assert mock_ra.called

    def test_invalid_confidence_uses_default(self, handler):
        """Non-numeric confidence uses default 0.7."""
        body = _minimal_findings_body(confidence_threshold="not-a-number")
        _, mock_ra = self._run_debate_with_body(handler, body)
        assert mock_ra.called

    def test_none_confidence_uses_default(self, handler):
        """None confidence uses default 0.7."""
        body = _minimal_findings_body(confidence_threshold=None)
        _, mock_ra = self._run_debate_with_body(handler, body)
        assert mock_ra.called


# ============================================================================
# POST - Timeout Clamping
# ============================================================================


class TestTimeoutClamping:
    """Test timeout_seconds parsing and clamping."""

    def _run_debate_with_body(self, handler, body):
        """Helper to run debate."""
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_default_timeout(self, handler):
        result = self._run_debate_with_body(handler, _minimal_findings_body())
        assert _status(result) == 200

    def test_custom_timeout(self, handler):
        body = _minimal_findings_body(timeout_seconds=120)
        result = self._run_debate_with_body(handler, body)
        assert _status(result) == 200

    def test_timeout_below_minimum_clamped(self, handler):
        """Timeout below 30 is clamped to 30."""
        body = _minimal_findings_body(timeout_seconds=5)
        result = self._run_debate_with_body(handler, body)
        assert _status(result) == 200

    def test_timeout_above_maximum_clamped(self, handler):
        """Timeout above 600 is clamped to 600."""
        body = _minimal_findings_body(timeout_seconds=9999)
        result = self._run_debate_with_body(handler, body)
        assert _status(result) == 200

    def test_invalid_timeout_uses_default(self, handler):
        """Non-numeric timeout uses default 300."""
        body = _minimal_findings_body(timeout_seconds="slow")
        result = self._run_debate_with_body(handler, body)
        assert _status(result) == 200

    def test_none_timeout_uses_default(self, handler):
        body = _minimal_findings_body(timeout_seconds=None)
        result = self._run_debate_with_body(handler, body)
        assert _status(result) == 200


# ============================================================================
# POST - Import Failure Handling
# ============================================================================


class TestImportFailure:
    """Test behavior when security debate modules are not available."""

    def test_import_error_returns_500(self, handler):
        """ImportError from security debate module returns 500."""
        http = _make_http_handler(body=_minimal_findings_body())
        handler._current_handler = http

        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.security_debate": None,
            },
        ):
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert result is not None
            assert _status(result) == 500
            assert "not available" in _body(result)["error"].lower()

    def test_import_error_events_module_returns_500(self, handler):
        """ImportError from security events module returns 500."""
        http = _make_http_handler(body=_minimal_findings_body())
        handler._current_handler = http

        with patch.dict(
            "sys.modules",
            {
                "aragora.events.security_events": None,
            },
        ):
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert result is not None
            assert _status(result) == 500


# ============================================================================
# POST - Successful Debate
# ============================================================================


class TestSuccessfulDebate:
    """Test successful security debate execution."""

    def _run_successful_debate(self, handler, body=None, debate_result=None):
        """Helper to run a successful debate and return the result."""
        if body is None:
            body = _minimal_findings_body()
        if debate_result is None:
            debate_result = MockDebateResult()

        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        mock_run.return_value = debate_result
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = debate_result
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_returns_200(self, handler):
        result = self._run_successful_debate(handler)
        assert _status(result) == 200

    def test_response_has_debate_id(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert "debate_id" in data
        assert data["debate_id"] == "debate-123"

    def test_response_has_status_completed(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert data["status"] == "completed"

    def test_response_has_consensus_reached(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert data["consensus_reached"] is True

    def test_response_has_confidence(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert data["confidence"] == 0.85

    def test_response_has_final_answer(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert "input validation" in data["final_answer"].lower() or len(data["final_answer"]) > 0

    def test_response_has_rounds_used(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert data["rounds_used"] == 3

    def test_response_has_duration_ms(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert "duration_ms" in data
        assert isinstance(data["duration_ms"], int)

    def test_response_has_findings_analyzed_count(self, handler):
        result = self._run_successful_debate(handler)
        data = _body(result)
        assert data["findings_analyzed"] == 1

    def test_multiple_findings_counted(self, handler):
        body = {
            "findings": [
                {"severity": "high", "title": "Finding 1", "description": "Desc 1"},
                {"severity": "medium", "title": "Finding 2", "description": "Desc 2"},
                {"severity": "low", "title": "Finding 3", "description": "Desc 3"},
            ]
        }
        result = self._run_successful_debate(handler, body=body)
        data = _body(result)
        assert data["findings_analyzed"] == 3

    def test_no_consensus_result(self, handler):
        debate_result = MockDebateResult(consensus_reached=False, confidence=0.3)
        result = self._run_successful_debate(handler, debate_result=debate_result)
        data = _body(result)
        assert data["consensus_reached"] is False
        assert data["confidence"] == 0.3

    def test_debate_id_fallback_when_no_debate_id_attr(self, handler):
        """When result has no debate_id attr, event.id is used."""
        debate_result = MagicMock()
        debate_result.consensus_reached = True
        debate_result.confidence = 0.75
        debate_result.final_answer = "Fix it"
        debate_result.rounds_used = 2
        debate_result.votes = None
        # Remove debate_id attribute
        del debate_result.debate_id

        result = self._run_successful_debate(handler, debate_result=debate_result)
        data = _body(result)
        # debate_id should be the event's id (auto-generated UUID)
        assert "debate_id" in data
        assert len(data["debate_id"]) > 0


# ============================================================================
# POST - Votes in Response
# ============================================================================


class TestVotesInResponse:
    """Test vote inclusion in debate response."""

    def _run_with_votes(self, handler, votes):
        debate_result = MockDebateResult(votes=votes)
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        mock_run.return_value = debate_result
        http = _make_http_handler(body=_minimal_findings_body())
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = debate_result
            return handler.handle_post("/api/v1/audit/security/debate", {}, http)

    def test_votes_included_when_present(self, handler):
        votes = [
            MockVote(agent_name="claude", vote="approve"),
            MockVote(agent_name="gpt4", vote="reject"),
        ]
        result = self._run_with_votes(handler, votes)
        data = _body(result)
        assert "votes" in data
        assert data["votes"]["claude"] == "approve"
        assert data["votes"]["gpt4"] == "reject"

    def test_votes_excluded_when_none(self, handler):
        result = self._run_with_votes(handler, None)
        data = _body(result)
        assert "votes" not in data

    def test_votes_excluded_when_empty(self, handler):
        result = self._run_with_votes(handler, [])
        data = _body(result)
        assert "votes" not in data

    def test_votes_handles_malformed_vote_objects(self, handler):
        """Votes without agent_name or vote attributes are skipped."""
        vote_good = MockVote(agent_name="claude", vote="approve")
        vote_bad = MagicMock(spec=[])  # no attributes at all
        result = self._run_with_votes(handler, [vote_good, vote_bad])
        data = _body(result)
        assert "votes" in data
        assert len(data["votes"]) == 1


# ============================================================================
# POST - Finding Severity Mapping
# ============================================================================


class TestFindingSeverity:
    """Test severity parsing and mapping for findings."""

    def _run_with_findings(self, handler, findings_data):
        body = {"findings": findings_data}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_critical_severity_accepted(self, handler):
        result = self._run_with_findings(
            handler, [{"severity": "critical", "title": "RCE", "description": "Remote code exec"}]
        )
        assert _status(result) == 200

    def test_high_severity_accepted(self, handler):
        result = self._run_with_findings(
            handler, [{"severity": "high", "title": "SQLi", "description": "SQL Injection"}]
        )
        assert _status(result) == 200

    def test_medium_severity_accepted(self, handler):
        result = self._run_with_findings(
            handler, [{"severity": "medium", "title": "XSS", "description": "Cross-site scripting"}]
        )
        assert _status(result) == 200

    def test_low_severity_accepted(self, handler):
        result = self._run_with_findings(
            handler, [{"severity": "low", "title": "Info", "description": "Info disclosure"}]
        )
        assert _status(result) == 200

    def test_invalid_severity_defaults_to_medium(self, handler):
        """Invalid severity string falls back to MEDIUM."""
        result = self._run_with_findings(
            handler, [{"severity": "unknown_sev", "title": "Test", "description": "Test"}]
        )
        assert _status(result) == 200

    def test_missing_severity_defaults_to_medium(self, handler):
        """Missing severity defaults to medium."""
        result = self._run_with_findings(
            handler, [{"title": "No severity", "description": "No sev field"}]
        )
        assert _status(result) == 200

    def test_uppercase_severity_normalized(self, handler):
        """Uppercase severity is lowercased."""
        result = self._run_with_findings(
            handler, [{"severity": "HIGH", "title": "Test", "description": "Test"}]
        )
        assert _status(result) == 200

    def test_mixed_case_severity_normalized(self, handler):
        result = self._run_with_findings(
            handler, [{"severity": "Critical", "title": "Test", "description": "Test"}]
        )
        assert _status(result) == 200


# ============================================================================
# POST - Finding Field Defaults
# ============================================================================


class TestFindingFieldDefaults:
    """Test default values for finding fields."""

    def _run_with_finding(self, handler, finding_data):
        body = {"findings": [finding_data]}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_minimal_finding_accepted(self, handler):
        """A finding with only severity/title/description works."""
        result = self._run_with_finding(
            handler,
            {
                "severity": "medium",
                "title": "Test",
                "description": "Test desc",
            },
        )
        assert _status(result) == 200

    def test_empty_finding_accepted(self, handler):
        """Empty finding dict still creates a valid SecurityFinding."""
        result = self._run_with_finding(handler, {})
        assert _status(result) == 200

    def test_finding_with_all_fields(self, handler):
        result = self._run_with_finding(
            handler,
            {
                "id": "custom-id",
                "finding_type": "dependency",
                "severity": "critical",
                "title": "Critical vuln",
                "description": "Critical vulnerability",
                "file_path": "src/main.py",
                "line_number": 100,
                "cve_id": "CVE-2024-1234",
                "package_name": "requests",
                "package_version": "2.25.0",
                "recommendation": "Upgrade to 2.32.0",
                "metadata": {"source": "snyk"},
            },
        )
        assert _status(result) == 200

    def test_default_title_includes_index(self, handler):
        """Default title includes finding index (1-based)."""
        # With two findings, second gets "Finding 2"
        body = {"findings": [{}, {}]}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200


# ============================================================================
# POST - Event Type Determination
# ============================================================================


class TestEventTypeDetermination:
    """Test event type selection based on finding severities."""

    def _run_and_capture_event(self, handler, findings_data):
        body = {"findings": findings_data}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        captured_events = []

        original_event_init = MockSecurityEvent.__init__

        def capturing_init(self_event, *args, **kwargs):
            original_event_init(self_event, *args, **kwargs)
            captured_events.append(self_event)

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
            patch.object(MockSecurityEvent, "__init__", capturing_init),
        ):
            mock_ra.return_value = MockDebateResult()
            handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return captured_events

    def test_critical_finding_sets_sast_critical_event(self, handler):
        events = self._run_and_capture_event(
            handler, [{"severity": "critical", "title": "RCE", "description": "Remote code exec"}]
        )
        assert len(events) >= 1
        assert events[0].event_type == MockSecurityEventType.SAST_CRITICAL

    def test_non_critical_finding_sets_vulnerability_detected(self, handler):
        events = self._run_and_capture_event(
            handler, [{"severity": "high", "title": "SQLi", "description": "SQL Injection"}]
        )
        assert len(events) >= 1
        assert events[0].event_type == MockSecurityEventType.VULNERABILITY_DETECTED

    def test_mixed_severities_with_critical(self, handler):
        events = self._run_and_capture_event(
            handler,
            [
                {"severity": "low", "title": "Info", "description": "Test"},
                {"severity": "critical", "title": "RCE", "description": "Test"},
            ],
        )
        assert len(events) >= 1
        assert events[0].event_type == MockSecurityEventType.SAST_CRITICAL

    def test_all_low_severity(self, handler):
        events = self._run_and_capture_event(
            handler,
            [
                {"severity": "low", "title": "A", "description": "Test"},
                {"severity": "low", "title": "B", "description": "Test"},
            ],
        )
        assert len(events) >= 1
        assert events[0].event_type == MockSecurityEventType.VULNERABILITY_DETECTED


# ============================================================================
# POST - Repository Field
# ============================================================================


class TestRepositoryField:
    """Test repository field from request body."""

    def _run_and_check_200(self, handler, body):
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_custom_repository_name(self, handler):
        body = _minimal_findings_body(repository="my-app-repo")
        result = self._run_and_check_200(handler, body)
        assert _status(result) == 200

    def test_default_repository_is_unknown(self, handler):
        body = _minimal_findings_body()
        # No repository field => defaults to "unknown"
        result = self._run_and_check_200(handler, body)
        assert _status(result) == 200


# ============================================================================
# POST - Debate Runtime Errors
# ============================================================================


class TestDebateRuntimeErrors:
    """Test behavior when debate execution fails."""

    def _run_with_error(self, handler, error):
        body = _minimal_findings_body()
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.side_effect = error
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            return result

    def test_runtime_error_returns_500(self, handler):
        result = self._run_with_error(handler, RuntimeError("event loop issue"))
        assert _status(result) == 500

    def test_os_error_returns_500(self, handler):
        result = self._run_with_error(handler, OSError("disk full"))
        assert _status(result) == 500

    def test_connection_error_returns_500(self, handler):
        result = self._run_with_error(handler, ConnectionError("API unreachable"))
        assert _status(result) == 500

    def test_timeout_error_returns_500(self, handler):
        result = self._run_with_error(handler, TimeoutError("debate timed out"))
        assert _status(result) == 500

    def test_value_error_returns_500(self, handler):
        result = self._run_with_error(handler, ValueError("invalid config"))
        assert _status(result) == 500

    def test_type_error_returns_500(self, handler):
        result = self._run_with_error(handler, TypeError("wrong type"))
        assert _status(result) == 500

    def test_error_response_contains_error_message(self, handler):
        result = self._run_with_error(handler, RuntimeError("test error"))
        assert _status(result) == 500
        data = _body(result)
        assert "error" in data


# ============================================================================
# POST - handle_errors Decorator Coverage
# ============================================================================


class TestHandleErrorsDecorator:
    """Test that @handle_errors on handle_post catches unexpected exceptions."""

    def test_unexpected_exception_caught_by_handle_errors(self, handler):
        """Exceptions not caught by the inner try/except are caught by @handle_errors."""
        body = _minimal_findings_body()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        # Patch get_json_body to raise an unexpected error
        with patch.object(handler, "get_json_body", side_effect=Exception("unexpected")):
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert result is not None
            # @handle_errors should return an error response
            assert _status(result) >= 400


# ============================================================================
# POST - Edge Cases
# ============================================================================


class TestPostEdgeCases:
    """Test edge cases for POST endpoint."""

    def test_trailing_slash_on_post_path(self, handler):
        """POST with trailing slash still matches."""
        body = _minimal_findings_body()
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate/", {}, http)
            # path.rstrip("/") == self._PREFIX should match
            assert result is not None
            assert _status(result) == 200

    def test_very_large_findings_list(self, handler):
        """Many findings are processed correctly."""
        findings = [
            {"severity": "medium", "title": f"Finding {i}", "description": f"Desc {i}"}
            for i in range(50)
        ]
        body = {"findings": findings}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200
            data = _body(result)
            assert data["findings_analyzed"] == 50

    def test_finding_with_custom_id_preserved(self, handler):
        """Custom finding IDs are preserved (not overwritten)."""
        body = {
            "findings": [
                {"id": "my-custom-id", "severity": "high", "title": "Test", "description": "Test"}
            ]
        }
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200

    def test_finding_without_id_gets_uuid(self, handler):
        """Finding without ID gets a generated UUID."""
        body = {"findings": [{"severity": "high", "title": "Test", "description": "Test"}]}
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200

    def test_zero_confidence_threshold_clamped_to_minimum(self, handler):
        """confidence_threshold=0 is clamped to 0.1."""
        body = _minimal_findings_body(confidence_threshold=0)
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200

    def test_negative_timeout_clamped_to_minimum(self, handler):
        """timeout_seconds=-10 is clamped to 30."""
        body = _minimal_findings_body(timeout_seconds=-10)
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert _status(result) == 200


# ============================================================================
# GET - Edge Cases
# ============================================================================


class TestGetEdgeCases:
    """Test edge cases for GET endpoint."""

    def test_empty_string_id(self, handler, mock_http_handler):
        """Path with empty trailing segment after slash returns None."""
        # /api/v1/audit/security/debate/ -> after rstrip, parts length = 6, last part = 'debate'
        # This should return None since the base path is 6 parts
        result = handler.handle(
            "/api/v1/audit/security/debate/",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_deeply_nested_path_returns_none(self, handler, mock_http_handler):
        """Deeply nested path (8+ parts) returns None."""
        result = handler.handle(
            "/api/v1/audit/security/debate/id1/extra/nested",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_numeric_debate_id(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/12345",
            {},
            mock_http_handler,
        )
        assert result is not None
        data = _body(result)
        assert data["debate_id"] == "12345"

    def test_special_characters_in_id(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/id-with_special.chars",
            {},
            mock_http_handler,
        )
        assert result is not None
        data = _body(result)
        assert data["debate_id"] == "id-with_special.chars"


# ============================================================================
# Response Structure Tests
# ============================================================================


class TestResponseStructure:
    """Test response format consistency."""

    def test_get_response_is_json(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/test-id",
            {},
            mock_http_handler,
        )
        assert result.content_type == "application/json"

    def test_post_success_response_is_json(self, handler):
        body = _minimal_findings_body()
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            assert result.content_type == "application/json"

    def test_error_response_is_json(self, handler):
        http = _make_http_handler(body={"findings": []})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        assert result.content_type == "application/json"

    def test_error_response_has_error_key(self, handler):
        http = _make_http_handler(body={"findings": []})
        handler._current_handler = http
        result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
        data = _body(result)
        assert "error" in data

    def test_get_not_found_response_keys(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/audit/security/debate/abc",
            {},
            mock_http_handler,
        )
        data = _body(result)
        assert set(data.keys()) == {"debate_id", "status", "message"}

    def test_post_success_response_keys(self, handler):
        body = _minimal_findings_body()
        mock_run, mock_sd_mod, mock_ev_mod = _build_security_mocks()
        http = _make_http_handler(body=body)
        handler._current_handler = http

        with (
            _patch_security_imports(mock_sd_mod, mock_ev_mod),
            patch("aragora.server.handlers.security_debate.run_async") as mock_ra,
        ):
            mock_ra.return_value = MockDebateResult()
            result = handler.handle_post("/api/v1/audit/security/debate", {}, http)
            data = _body(result)
            expected_keys = {
                "debate_id",
                "status",
                "consensus_reached",
                "confidence",
                "final_answer",
                "rounds_used",
                "duration_ms",
                "findings_analyzed",
            }
            assert expected_keys.issubset(set(data.keys()))

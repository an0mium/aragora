"""Tests for evolution A/B testing handler.

Covers all routes and behavior of EvolutionABTestingHandler:
- GET  /api/evolution/ab-tests             - List all A/B tests
- GET  /api/evolution/ab-tests/{agent}/active - Get active test for agent
- GET  /api/evolution/ab-tests/{id}        - Get specific test
- POST /api/evolution/ab-tests             - Start new A/B test
- POST /api/evolution/ab-tests/{id}/record - Record debate result
- POST /api/evolution/ab-tests/{id}/conclude - Conclude test
- DELETE /api/evolution/ab-tests/{id}      - Cancel test
- can_handle() routing
- Module unavailable (503) responses
- Manager not configured (503) responses
- Input validation and error paths
- Rate limiting on DELETE
- Authentication on DELETE
- RBAC permission checks (via conftest auto-bypass)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.evolution.ab_testing as _ab_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw)
        return raw
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return 0


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockTestStatus(Enum):
    """Mock A/B test status enum."""

    ACTIVE = "active"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


class MockABTest:
    """Mock ABTest object returned by the manager."""

    def __init__(
        self,
        test_id: str = "test-001",
        agent: str = "claude",
        baseline_version: int = 1,
        evolved_version: int = 2,
        status: MockTestStatus = MockTestStatus.ACTIVE,
    ):
        self.test_id = test_id
        self.agent = agent
        self.baseline_version = baseline_version
        self.evolved_version = evolved_version
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "agent": self.agent,
            "baseline_version": self.baseline_version,
            "evolved_version": self.evolved_version,
            "status": self.status.value,
        }


class MockABTestResult:
    """Mock A/B test conclusion result."""

    def __init__(
        self,
        test_id: str = "test-001",
        winner: str = "evolved",
        confidence: float = 0.95,
        recommendation: str = "Promote evolved version",
        stats: dict | None = None,
    ):
        self.test_id = test_id
        self.winner = winner
        self.confidence = confidence
        self.recommendation = recommendation
        self.stats = stats or {"baseline_wins": 3, "evolved_wins": 7}


class MockHTTPHandler:
    """Mock HTTP handler for BaseHandler and rate limiting."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Patch target constants
# ---------------------------------------------------------------------------

_GET_ALL_TESTS_PATCH = (
    "aragora.server.handlers.evolution.ab_testing.EvolutionABTestingHandler._get_all_tests"
)
# Local imports in handle_delete - must patch at source
_EXTRACT_USER_PATCH = "aragora.billing.jwt_auth.extract_user_from_request"
_GET_CLIENT_IP_PATCH = "aragora.server.handlers.utils.rate_limit.get_client_ip"

# Save original value to restore
_ORIG_AB_AVAILABLE = _ab_mod.AB_TESTING_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_ab_available():
    """Restore AB_TESTING_AVAILABLE after each test."""
    yield
    _ab_mod.AB_TESTING_AVAILABLE = _ORIG_AB_AVAILABLE


@pytest.fixture
def mock_manager():
    """Create a mock ABTestManager."""
    mgr = MagicMock()
    mgr.db_path = "ab_tests.db"
    return mgr


@pytest.fixture
def handler(mock_manager):
    """Create EvolutionABTestingHandler with AB_TESTING_AVAILABLE=True."""
    _ab_mod.AB_TESTING_AVAILABLE = True
    h = _ab_mod.EvolutionABTestingHandler(ctx={})
    h._manager = mock_manager
    return h


@pytest.fixture
def handler_unavailable():
    """Handler with AB_TESTING_AVAILABLE = False."""
    _ab_mod.AB_TESTING_AVAILABLE = False
    return _ab_mod.EvolutionABTestingHandler(ctx={})


@pytest.fixture
def handler_no_manager():
    """Handler with AB_TESTING_AVAILABLE True but manager always returns None.

    Uses a throwaway subclass to avoid polluting the real class with a
    patched property descriptor.
    """
    _ab_mod.AB_TESTING_AVAILABLE = True

    class _NoManagerHandler(_ab_mod.EvolutionABTestingHandler):
        @property
        def manager(self):
            return None

    return _NoManagerHandler(ctx={})


@pytest.fixture
def http():
    """Create a default mock HTTP handler."""
    return MockHTTPHandler()


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Test can_handle routing."""

    def test_can_handle_base_path(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests") is True

    def test_can_handle_base_path_trailing_slash(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests/") is True

    def test_can_handle_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/evolution/ab-tests") is True

    def test_can_handle_versioned_path_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/evolution/ab-tests/") is True

    def test_can_handle_with_id(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests/test-001") is True

    def test_can_handle_with_active_suffix(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests/claude/active") is True

    def test_can_handle_with_record_suffix(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests/test-001/record") is True

    def test_can_handle_with_conclude_suffix(self, handler):
        assert handler.can_handle("/api/evolution/ab-tests/test-001/conclude") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/evolution/something-else") is False

    def test_cannot_handle_partial_path(self, handler):
        assert handler.can_handle("/api/evolution") is False

    def test_cannot_handle_random_path(self, handler):
        assert handler.can_handle("/api/debates") is False


# ============================================================================
# GET /api/evolution/ab-tests (list tests)
# ============================================================================


class TestListTests:
    """Test listing A/B tests."""

    def test_list_all_tests(self, handler, mock_manager, http):
        """List all tests without filters."""
        test1 = MockABTest(test_id="t1", agent="claude")
        test2 = MockABTest(test_id="t2", agent="gpt4")

        with patch(_GET_ALL_TESTS_PATCH, return_value=[test1, test2]):
            result = handler.handle("/api/evolution/ab-tests", {}, http)

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["tests"]) == 2

    def test_list_tests_with_agent_filter(self, handler, mock_manager, http):
        """List tests filtered by agent."""
        test1 = MockABTest(test_id="t1", agent="claude")
        mock_manager.get_agent_tests.return_value = [test1]

        result = handler.handle("/api/evolution/ab-tests", {"agent": "claude"}, http)

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        # get_agent_tests is called with positional agent arg
        mock_manager.get_agent_tests.assert_called_once_with("claude", limit=50)

    def test_list_tests_with_limit(self, handler, mock_manager, http):
        """Custom limit parameter is respected."""
        mock_manager.get_agent_tests.return_value = []

        result = handler.handle("/api/evolution/ab-tests", {"agent": "claude", "limit": "10"}, http)

        assert _status(result) == 200
        mock_manager.get_agent_tests.assert_called_once_with("claude", limit=10)

    def test_list_tests_invalid_status_filter(self, handler, mock_manager, http):
        """Invalid status filter returns 400."""
        result = handler.handle("/api/evolution/ab-tests", {"status": "bogus"}, http)

        assert _status(result) == 400
        body = _body(result)
        assert "Invalid status" in body.get("error", "")

    def test_list_tests_valid_status_active(self, handler, mock_manager, http):
        """Valid status 'active' is accepted."""
        t = MockABTest()
        with patch(_GET_ALL_TESTS_PATCH, return_value=[t]):
            result = handler.handle("/api/evolution/ab-tests", {"status": "active"}, http)

        assert _status(result) == 200

    def test_list_tests_valid_status_concluded(self, handler, mock_manager, http):
        """Valid status 'concluded' is accepted."""
        with patch(_GET_ALL_TESTS_PATCH, return_value=[]):
            result = handler.handle("/api/evolution/ab-tests", {"status": "concluded"}, http)

        assert _status(result) == 200

    def test_list_tests_valid_status_cancelled(self, handler, mock_manager, http):
        """Valid status 'cancelled' is accepted."""
        with patch(_GET_ALL_TESTS_PATCH, return_value=[]):
            result = handler.handle("/api/evolution/ab-tests", {"status": "cancelled"}, http)

        assert _status(result) == 200

    def test_list_tests_versioned_path(self, handler, mock_manager, http):
        """Versioned API path is supported."""
        with patch(_GET_ALL_TESTS_PATCH, return_value=[]):
            result = handler.handle("/api/v1/evolution/ab-tests", {}, http)

        assert _status(result) == 200

    def test_list_tests_trailing_slash(self, handler, mock_manager, http):
        """Trailing slash is handled."""
        with patch(_GET_ALL_TESTS_PATCH, return_value=[]):
            result = handler.handle("/api/evolution/ab-tests/", {}, http)

        assert _status(result) == 200


# ============================================================================
# GET /api/evolution/ab-tests/{id} (get specific test)
# ============================================================================


class TestGetTest:
    """Test getting a specific A/B test."""

    def test_get_test_found(self, handler, mock_manager, http):
        """Get an existing test by ID."""
        test = MockABTest(test_id="test-001", agent="claude")
        mock_manager.get_test.return_value = test

        result = handler.handle("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["test_id"] == "test-001"
        assert body["agent"] == "claude"
        mock_manager.get_test.assert_called_once_with("test-001")

    def test_get_test_not_found(self, handler, mock_manager, http):
        """Get a nonexistent test returns 404."""
        mock_manager.get_test.return_value = None

        result = handler.handle("/api/evolution/ab-tests/nonexistent", {}, http)

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_get_test_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for get test."""
        test = MockABTest(test_id="test-002")
        mock_manager.get_test.return_value = test

        result = handler.handle("/api/v1/evolution/ab-tests/test-002", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["test_id"] == "test-002"


# ============================================================================
# GET /api/evolution/ab-tests/{agent}/active (get active test for agent)
# ============================================================================


class TestGetActiveTest:
    """Test getting active A/B test for an agent."""

    def test_active_test_found(self, handler, mock_manager, http):
        """Active test exists for agent."""
        test = MockABTest(test_id="test-active", agent="claude")
        mock_manager.get_active_test.return_value = test

        result = handler.handle("/api/evolution/ab-tests/claude/active", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["has_active_test"] is True
        assert body["test"] is not None
        assert body["test"]["test_id"] == "test-active"
        mock_manager.get_active_test.assert_called_once_with("claude")

    def test_no_active_test(self, handler, mock_manager, http):
        """No active test for agent returns has_active_test=False."""
        mock_manager.get_active_test.return_value = None

        result = handler.handle("/api/evolution/ab-tests/gpt4/active", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "gpt4"
        assert body["has_active_test"] is False
        assert body["test"] is None

    def test_active_test_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for active test lookup."""
        mock_manager.get_active_test.return_value = None

        result = handler.handle("/api/v1/evolution/ab-tests/claude/active", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["has_active_test"] is False


# ============================================================================
# POST /api/evolution/ab-tests (create test)
# ============================================================================


class TestCreateTest:
    """Test creating A/B tests."""

    def test_create_test_success(self, handler, mock_manager, http):
        """Successfully create a new A/B test."""
        created = MockABTest(test_id="new-test", agent="claude")
        mock_manager.start_test.return_value = created

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 201
        resp = _body(result)
        assert resp["message"] == "A/B test created"
        assert resp["test"]["test_id"] == "new-test"
        mock_manager.start_test.assert_called_once_with(
            agent="claude",
            baseline_version=1,
            evolved_version=2,
            metadata=None,
        )

    def test_create_test_with_metadata(self, handler, mock_manager, http):
        """Create with optional metadata."""
        created = MockABTest(test_id="meta-test")
        mock_manager.start_test.return_value = created

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
            "metadata": {"description": "Testing new prompt"},
        }
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 201
        mock_manager.start_test.assert_called_once_with(
            agent="claude",
            baseline_version=1,
            evolved_version=2,
            metadata={"description": "Testing new prompt"},
        )

    def test_create_test_missing_agent(self, handler, mock_manager, http):
        """Missing agent field returns 400."""
        body = {"baseline_version": 1, "evolved_version": 2}
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 400
        assert "agent" in _body(result).get("error", "").lower()

    def test_create_test_agent_not_string(self, handler, mock_manager, http):
        """Non-string agent returns 400."""
        body = {"agent": 123, "baseline_version": 1, "evolved_version": 2}
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 400
        assert "string" in _body(result).get("error", "").lower()

    def test_create_test_missing_versions(self, handler, mock_manager, http):
        """Missing version fields return 400."""
        body = {"agent": "claude"}
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 400
        assert "version" in _body(result).get("error", "").lower()

    def test_create_test_missing_evolved_version(self, handler, mock_manager, http):
        """Missing evolved_version returns 400."""
        body = {"agent": "claude", "baseline_version": 1}
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 400

    def test_create_test_non_integer_versions(self, handler, mock_manager, http):
        """Non-integer versions return 400."""
        body = {
            "agent": "claude",
            "baseline_version": "abc",
            "evolved_version": "xyz",
        }
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 400
        assert "integer" in _body(result).get("error", "").lower()

    def test_create_test_string_integer_versions(self, handler, mock_manager, http):
        """String-encoded integer versions are accepted."""
        created = MockABTest(test_id="str-ver")
        mock_manager.start_test.return_value = created

        body = {
            "agent": "claude",
            "baseline_version": "1",
            "evolved_version": "2",
        }
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 201

    def test_create_test_conflict(self, handler, mock_manager, http):
        """ValueError from manager returns 409 (conflict)."""
        mock_manager.start_test.side_effect = ValueError("Duplicate test")

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = handler.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 409
        assert "conflict" in _body(result).get("error", "").lower()

    def test_create_test_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for creation."""
        created = MockABTest(test_id="ver-test")
        mock_manager.start_test.return_value = created

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = handler.handle_post("/api/v1/evolution/ab-tests", body, http)

        assert _status(result) == 201

    def test_create_test_trailing_slash(self, handler, mock_manager, http):
        """Trailing slash path works for creation."""
        created = MockABTest(test_id="slash-test")
        mock_manager.start_test.return_value = created

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = handler.handle_post("/api/evolution/ab-tests/", body, http)

        assert _status(result) == 201


# ============================================================================
# POST /api/evolution/ab-tests/{id}/record (record debate result)
# ============================================================================


class TestRecordResult:
    """Test recording A/B test results."""

    def test_record_result_success(self, handler, mock_manager, http):
        """Successfully record a debate result."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "debate-abc123",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200
        resp = _body(result)
        assert resp["message"] == "Result recorded"
        assert resp["test"]["test_id"] == "test-001"
        mock_manager.record_result.assert_called_once_with(
            agent="claude",
            debate_id="debate-abc123",
            variant="baseline",
            won=True,
        )

    def test_record_result_evolved_variant(self, handler, mock_manager, http):
        """Record result for evolved variant."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "debate-xyz789",
            "variant": "evolved",
            "won": False,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200
        mock_manager.record_result.assert_called_once_with(
            agent="claude",
            debate_id="debate-xyz789",
            variant="evolved",
            won=False,
        )

    def test_record_result_test_not_found(self, handler, mock_manager, http):
        """Record on nonexistent test returns 404."""
        mock_manager.get_test.return_value = None

        body = {
            "debate_id": "debate-123",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/missing/record", body, http)

        assert _status(result) == 404

    def test_record_result_concluded_test(self, handler, mock_manager, http):
        """Cannot record results on concluded test."""
        concluded_test = MockABTest(test_id="test-done", status=MockTestStatus.CONCLUDED)
        mock_manager.get_test.return_value = concluded_test

        body = {
            "debate_id": "debate-123",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-done/record", body, http)

        assert _status(result) == 400
        assert "concluded" in _body(result).get("error", "").lower()

    def test_record_result_missing_debate_id(self, handler, mock_manager, http):
        """Missing debate_id returns 400."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {"variant": "baseline", "won": True}
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400
        assert "debate_id" in _body(result).get("error", "").lower()

    def test_record_result_invalid_debate_id(self, handler, mock_manager, http):
        """Invalid debate_id format returns 400."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {
            "debate_id": "../../etc/passwd",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400
        assert "debate_id" in _body(result).get("error", "").lower()

    def test_record_result_non_string_debate_id(self, handler, mock_manager, http):
        """Non-string debate_id returns 400."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {"debate_id": 12345, "variant": "baseline", "won": True}
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400

    def test_record_result_invalid_variant(self, handler, mock_manager, http):
        """Invalid variant value returns 400."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {
            "debate_id": "debate-123",
            "variant": "control",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400
        assert "variant" in _body(result).get("error", "").lower()

    def test_record_result_missing_won(self, handler, mock_manager, http):
        """Missing won field returns 400."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {"debate_id": "debate-123", "variant": "baseline"}
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400
        assert "won" in _body(result).get("error", "").lower()

    def test_record_result_failed(self, handler, mock_manager, http):
        """Manager returns None on failure -> 500."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = None

        body = {
            "debate_id": "debate-123",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 500

    def test_record_result_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for recording results."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "debate-123",
            "variant": "evolved",
            "won": True,
        }
        result = handler.handle_post("/api/v1/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200


# ============================================================================
# POST /api/evolution/ab-tests/{id}/conclude (conclude test)
# ============================================================================


class TestConcludeTest:
    """Test concluding A/B tests."""

    def test_conclude_test_success(self, handler, mock_manager, http):
        """Successfully conclude a test."""
        mock_result = MockABTestResult(
            test_id="test-001",
            winner="evolved",
            confidence=0.95,
        )
        mock_manager.conclude_test.return_value = mock_result

        result = handler.handle_post("/api/evolution/ab-tests/test-001/conclude", {}, http)

        assert _status(result) == 200
        resp = _body(result)
        assert resp["message"] == "A/B test concluded"
        assert resp["result"]["test_id"] == "test-001"
        assert resp["result"]["winner"] == "evolved"
        assert resp["result"]["confidence"] == 0.95
        mock_manager.conclude_test.assert_called_once_with("test-001", force=False)

    def test_conclude_test_with_force(self, handler, mock_manager, http):
        """Conclude with force=True."""
        mock_result = MockABTestResult(test_id="test-001")
        mock_manager.conclude_test.return_value = mock_result

        result = handler.handle_post(
            "/api/evolution/ab-tests/test-001/conclude", {"force": True}, http
        )

        assert _status(result) == 200
        mock_manager.conclude_test.assert_called_once_with("test-001", force=True)

    def test_conclude_test_value_error(self, handler, mock_manager, http):
        """ValueError from manager returns 400."""
        mock_manager.conclude_test.side_effect = ValueError("Not enough data")

        result = handler.handle_post("/api/evolution/ab-tests/test-001/conclude", {}, http)

        assert _status(result) == 400

    def test_conclude_test_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for concluding."""
        mock_result = MockABTestResult(test_id="test-002")
        mock_manager.conclude_test.return_value = mock_result

        result = handler.handle_post("/api/v1/evolution/ab-tests/test-002/conclude", {}, http)

        assert _status(result) == 200

    def test_conclude_test_result_fields(self, handler, mock_manager, http):
        """Conclusion result includes all expected fields."""
        mock_result = MockABTestResult(
            test_id="test-001",
            winner="baseline",
            confidence=0.8,
            recommendation="Keep baseline",
            stats={"baseline_wins": 7, "evolved_wins": 3},
        )
        mock_manager.conclude_test.return_value = mock_result

        result = handler.handle_post("/api/evolution/ab-tests/test-001/conclude", {}, http)

        assert _status(result) == 200
        resp = _body(result)
        r = resp["result"]
        assert r["test_id"] == "test-001"
        assert r["winner"] == "baseline"
        assert r["confidence"] == 0.8
        assert r["recommendation"] == "Keep baseline"
        assert r["stats"]["baseline_wins"] == 7


# ============================================================================
# DELETE /api/evolution/ab-tests/{id} (cancel test)
# ============================================================================


class TestDeleteCancelTest:
    """Test cancelling A/B tests via DELETE."""

    def _make_auth_ctx(self, authenticated: bool = True):
        """Create a mock UserAuthContext."""
        ctx = MagicMock()
        ctx.is_authenticated = authenticated
        ctx.authenticated = authenticated
        return ctx

    def test_cancel_test_success(self, handler, mock_manager, http):
        """Successfully cancel a test."""
        mock_manager.cancel_test.return_value = True
        auth_ctx = self._make_auth_ctx(authenticated=True)

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=auth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            handler._delete_limiter = MagicMock()
            handler._delete_limiter.is_allowed.return_value = True

            result = handler.handle_delete("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "A/B test cancelled"
        assert body["test_id"] == "test-001"
        mock_manager.cancel_test.assert_called_once_with("test-001")

    def test_cancel_test_not_found(self, handler, mock_manager, http):
        """Cancel nonexistent test returns 404."""
        mock_manager.cancel_test.return_value = False
        auth_ctx = self._make_auth_ctx(authenticated=True)

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=auth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            handler._delete_limiter = MagicMock()
            handler._delete_limiter.is_allowed.return_value = True

            result = handler.handle_delete("/api/evolution/ab-tests/missing-test", {}, http)

        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_cancel_test_unauthenticated(self, handler, mock_manager, http):
        """Unauthenticated delete returns 401."""
        unauth_ctx = self._make_auth_ctx(authenticated=False)

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=unauth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            result = handler.handle_delete("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 401
        assert "authentication" in _body(result).get("error", "").lower()

    def test_cancel_test_rate_limited(self, handler, mock_manager, http):
        """Rate limited delete returns 429."""
        auth_ctx = self._make_auth_ctx(authenticated=True)

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=auth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            handler._delete_limiter = MagicMock()
            handler._delete_limiter.is_allowed.return_value = False

            result = handler.handle_delete("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 429
        assert "rate limit" in _body(result).get("error", "").lower()

    def test_cancel_test_versioned_path(self, handler, mock_manager, http):
        """Versioned path works for DELETE."""
        mock_manager.cancel_test.return_value = True
        auth_ctx = self._make_auth_ctx(authenticated=True)

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=auth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            handler._delete_limiter = MagicMock()
            handler._delete_limiter.is_allowed.return_value = True

            result = handler.handle_delete("/api/v1/evolution/ab-tests/test-002", {}, http)

        assert _status(result) == 200

    def test_cancel_test_wrong_path_length(self, handler, mock_manager, http):
        """DELETE on path with extra segments returns None."""
        with (
            patch(
                _EXTRACT_USER_PATCH,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            result = handler.handle_delete("/api/evolution/ab-tests/test-001/extra", {}, http)

        assert result is None


# ============================================================================
# Module unavailable (503) tests
# ============================================================================


class TestModuleUnavailable:
    """Test behavior when A/B testing module is not available."""

    def test_handle_returns_503(self, handler_unavailable, http):
        """GET returns 503 when module unavailable."""
        result = handler_unavailable.handle("/api/evolution/ab-tests", {}, http)

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_handle_post_returns_503(self, handler_unavailable, http):
        """POST returns 503 when module unavailable."""
        result = handler_unavailable.handle_post(
            "/api/evolution/ab-tests", {"agent": "claude"}, http
        )

        assert _status(result) == 503

    def test_handle_delete_returns_503(self, handler_unavailable, http):
        """DELETE returns 503 when module unavailable."""
        with (
            patch(
                _EXTRACT_USER_PATCH,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            result = handler_unavailable.handle_delete("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 503


# ============================================================================
# Manager not configured (503) tests
# ============================================================================


class TestManagerNotConfigured:
    """Test behavior when manager is not configured."""

    def test_list_tests_503(self, handler_no_manager, http):
        """Listing tests returns 503 when manager not configured."""
        result = handler_no_manager.handle("/api/evolution/ab-tests", {}, http)

        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    def test_get_test_503(self, handler_no_manager, http):
        """Getting test returns 503 when manager not configured."""
        result = handler_no_manager.handle("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 503

    def test_get_active_test_503(self, handler_no_manager, http):
        """Getting active test returns 503 when manager not configured."""
        result = handler_no_manager.handle("/api/evolution/ab-tests/claude/active", {}, http)

        assert _status(result) == 503

    def test_create_test_503(self, handler_no_manager, http):
        """Creating test returns 503 when manager not configured."""
        body = {"agent": "claude", "baseline_version": 1, "evolved_version": 2}
        result = handler_no_manager.handle_post("/api/evolution/ab-tests", body, http)

        assert _status(result) == 503

    def test_record_result_503(self, handler_no_manager, http):
        """Recording result returns 503 when manager not configured."""
        body = {"debate_id": "d1", "variant": "baseline", "won": True}
        result = handler_no_manager.handle_post(
            "/api/evolution/ab-tests/test-001/record", body, http
        )

        assert _status(result) == 503

    def test_conclude_test_503(self, handler_no_manager, http):
        """Concluding test returns 503 when manager not configured."""
        result = handler_no_manager.handle_post(
            "/api/evolution/ab-tests/test-001/conclude", {}, http
        )

        assert _status(result) == 503

    def test_cancel_test_503(self, handler_no_manager, http):
        """Cancelling test returns 503 when manager not configured."""
        auth_ctx = MagicMock()
        auth_ctx.is_authenticated = True

        with (
            patch(
                _EXTRACT_USER_PATCH,
                return_value=auth_ctx,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            handler_no_manager._delete_limiter = MagicMock()
            handler_no_manager._delete_limiter.is_allowed.return_value = True

            result = handler_no_manager.handle_delete("/api/evolution/ab-tests/test-001", {}, http)

        assert _status(result) == 503


# ============================================================================
# Routing edge cases / returns None
# ============================================================================


class TestRoutingEdgeCases:
    """Test edge cases in route dispatch."""

    def test_handle_returns_none_for_unknown_subpath(self, handler, http):
        """Unknown subpath returns None (not handled)."""
        result = handler.handle("/api/evolution/ab-tests/test-001/unknown/extra", {}, http)

        assert result is None

    def test_handle_post_returns_none_for_unknown_action(self, handler, http):
        """POST on unknown action returns None."""
        result = handler.handle_post("/api/evolution/ab-tests/test-001/unknown", {}, http)

        assert result is None

    def test_handle_post_returns_none_short_path(self, handler, http):
        """POST on too-short subpath with id returns None."""
        # POST /api/evolution/ab-tests/{id} (no action) - not handled by handle_post
        result = handler.handle_post("/api/evolution/ab-tests/test-001", {}, http)

        assert result is None

    def test_handle_returns_none_too_many_segments(self, handler, http):
        """Path with too many segments returns None."""
        result = handler.handle("/api/evolution/ab-tests/agent1/active/extra", {}, http)

        assert result is None

    def test_handle_delete_returns_none_for_base_path(self, handler, http):
        """DELETE on base path returns None (no id)."""
        with (
            patch(
                _EXTRACT_USER_PATCH,
            ),
            patch(
                _GET_CLIENT_IP_PATCH,
                return_value="127.0.0.1",
            ),
        ):
            result = handler.handle_delete("/api/evolution/ab-tests", {}, http)

        assert result is None


# ============================================================================
# Input validation tests
# ============================================================================


class TestInputValidation:
    """Test input validation for path segments and body fields."""

    def test_debate_id_with_special_chars_rejected(self, handler, mock_manager, http):
        """debate_id with path traversal chars is rejected."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {
            "debate_id": "id with spaces",
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400

    def test_debate_id_too_long_rejected(self, handler, mock_manager, http):
        """debate_id over 128 chars is rejected."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        mock_manager.get_test.return_value = active_test

        body = {
            "debate_id": "a" * 129,
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 400

    def test_debate_id_max_length_accepted(self, handler, mock_manager, http):
        """debate_id at exactly 128 chars is accepted."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "a" * 128,
            "variant": "baseline",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200

    def test_debate_id_with_hyphens_underscores(self, handler, mock_manager, http):
        """debate_id with hyphens and underscores is accepted."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "debate_2024-01-15_run-1",
            "variant": "evolved",
            "won": True,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200

    def test_won_field_truthy_conversion(self, handler, mock_manager, http):
        """Non-boolean truthy 'won' value is coerced to bool."""
        active_test = MockABTest(test_id="test-001", status=MockTestStatus.ACTIVE)
        updated_test = MockABTest(test_id="test-001")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "debate-123",
            "variant": "baseline",
            "won": 1,
        }
        result = handler.handle_post("/api/evolution/ab-tests/test-001/record", body, http)

        assert _status(result) == 200
        mock_manager.record_result.assert_called_once_with(
            agent="claude",
            debate_id="debate-123",
            variant="baseline",
            won=True,
        )

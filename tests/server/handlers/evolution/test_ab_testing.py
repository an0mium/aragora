"""Tests for the evolution A/B testing endpoint handler.

Tests cover:
- Experiment CRUD operations (create, read, list, delete)
- Traffic allocation / recording results
- Statistical significance analysis (conclude)
- Validation of request bodies and path segments
- Graceful degradation when AB_TESTING_AVAILABLE=False
- Manager not configured (503)
- Auth and rate limiting for delete
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

# Pre-mock broken transitive imports to allow importing evolution handler
# without triggering the full handlers.__init__ chain
if "aragora.server.handlers.social._slack_impl" not in sys.modules:
    sys.modules["aragora.server.handlers.social._slack_impl"] = MagicMock()

import pytest


@pytest.fixture(autouse=True)
def _clear_global_state():
    """Reset module-level globals and bypass RBAC between tests."""
    import aragora.server.handlers.evolution.ab_testing as mod

    # Bypass RBAC permission checks by patching _get_context_from_args
    # to return a context with wildcard permissions
    from aragora.rbac.models import AuthorizationContext

    mock_ctx = AuthorizationContext(
        user_id="test-user",
        permissions={"*"},
    )

    with patch("aragora.rbac.decorators._get_context_from_args", return_value=mock_ctx):
        yield


def _make_handler_instance(ctx=None):
    """Create an EvolutionABTestingHandler with a mock context."""
    from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

    if ctx is None:
        ctx = {"ab_tests_db": ":memory:"}
    return EvolutionABTestingHandler(ctx)


def _mock_http_handler(client_ip="127.0.0.1"):
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {}
    return handler


def _parse_result(result):
    """Parse a HandlerResult into (status_code, data_dict)."""
    return result.status_code, json.loads(result.body)


def _make_test_obj(test_id="test-1", agent="claude", status_value="active"):
    """Create a mock ABTest object."""
    test = MagicMock()
    test.test_id = test_id
    test.agent = agent
    test.status = MagicMock()
    test.status.value = status_value
    test.to_dict.return_value = {
        "test_id": test_id,
        "agent": agent,
        "status": status_value,
        "baseline_version": 1,
        "evolved_version": 2,
    }
    return test


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_ab_tests_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/ab-tests") is True

    def test_ab_tests_trailing_slash(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/ab-tests/") is True

    def test_ab_tests_with_id(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/ab-tests/test-1") is True

    def test_ab_tests_active(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/ab-tests/claude/active") is True

    def test_v1_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/v1/evolution/ab-tests") is True

    def test_unrelated_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/patterns") is False


# ---------------------------------------------------------------------------
# AB_TESTING_AVAILABLE=False degradation
# ---------------------------------------------------------------------------


class TestABTestingUnavailable:
    def test_handle_returns_503(self):
        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            h = _make_handler_instance()
            result = h.handle("/api/evolution/ab-tests", {}, _mock_http_handler())
            status, data = _parse_result(result)
            assert status == 503
            assert "not available" in data["error"]
        finally:
            mod.AB_TESTING_AVAILABLE = original

    def test_handle_post_returns_503(self):
        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            h = _make_handler_instance()
            result = h.handle_post("/api/evolution/ab-tests", {}, _mock_http_handler())
            status, data = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original

    def test_handle_delete_returns_503(self):
        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            h = _make_handler_instance()
            result = h.handle_delete("/api/evolution/ab-tests/test-1", {}, _mock_http_handler())
            status, data = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# List tests
# ---------------------------------------------------------------------------


class TestListTests:
    def test_list_all_tests(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        test1 = _make_test_obj("t1", "claude")
        test2 = _make_test_obj("t2", "gpt4")

        with patch.object(h, "_get_all_tests", return_value=[test1, test2]):
            result = h._list_tests({})

        status, data = _parse_result(result)
        assert status == 200
        assert data["count"] == 2

    def test_list_tests_by_agent(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        test1 = _make_test_obj("t1", "claude")
        mock_manager.get_agent_tests.return_value = [test1]

        result = h._list_tests({"agent": "claude"})

        status, data = _parse_result(result)
        assert status == 200
        assert data["count"] == 1
        mock_manager.get_agent_tests.assert_called_once()

    def test_list_tests_no_manager_returns_503(self):
        h = _make_handler_instance()
        h._manager = None

        # Make the property return None by mocking AB_TESTING_AVAILABLE
        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._list_tests({})
            status, data = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# Get single test
# ---------------------------------------------------------------------------


class TestGetTest:
    def test_get_existing_test(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.get_test.return_value = _make_test_obj("t1", "claude")

        result = h._get_test("t1")
        status, data = _parse_result(result)
        assert status == 200
        assert data["test_id"] == "t1"

    def test_get_nonexistent_test(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.get_test.return_value = None

        result = h._get_test("nonexistent")
        status, data = _parse_result(result)
        assert status == 404

    def test_get_test_no_manager(self):
        h = _make_handler_instance()
        h._manager = None

        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._get_test("t1")
            status, _ = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# Get active test for agent
# ---------------------------------------------------------------------------


class TestGetActiveTest:
    def test_active_test_exists(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.get_active_test.return_value = _make_test_obj("t1", "claude")

        result = h._get_active_test("claude")
        status, data = _parse_result(result)
        assert status == 200
        assert data["has_active_test"] is True
        assert data["test"]["test_id"] == "t1"

    def test_no_active_test(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.get_active_test.return_value = None

        result = h._get_active_test("claude")
        status, data = _parse_result(result)
        assert status == 200
        assert data["has_active_test"] is False
        assert data["test"] is None

    def test_active_test_no_manager(self):
        h = _make_handler_instance()
        h._manager = None

        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._get_active_test("claude")
            status, _ = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# Create test
# ---------------------------------------------------------------------------


class TestCreateTest:
    def test_create_success(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.start_test.return_value = _make_test_obj("t1", "claude")

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = h._create_test(body)
        status, data = _parse_result(result)
        assert status == 201
        assert data["message"] == "A/B test created"

    def test_create_missing_agent(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        result = h._create_test({"baseline_version": 1, "evolved_version": 2})
        status, data = _parse_result(result)
        assert status == 400
        assert "agent" in data["error"]

    def test_create_missing_versions(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        result = h._create_test({"agent": "claude"})
        status, data = _parse_result(result)
        assert status == 400
        assert "version" in data["error"].lower()

    def test_create_invalid_version_type(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        result = h._create_test(
            {
                "agent": "claude",
                "baseline_version": "abc",
                "evolved_version": 2,
            }
        )
        status, data = _parse_result(result)
        assert status == 400
        assert "integer" in data["error"].lower()

    def test_create_conflict(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.start_test.side_effect = ValueError("duplicate test")

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
        }
        result = h._create_test(body)
        status, data = _parse_result(result)
        assert status == 409

    def test_create_no_manager(self):
        h = _make_handler_instance()
        h._manager = None

        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._create_test(
                {"agent": "claude", "baseline_version": 1, "evolved_version": 2}
            )
            status, _ = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original

    def test_create_with_metadata(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.start_test.return_value = _make_test_obj("t1", "claude")

        body = {
            "agent": "claude",
            "baseline_version": 1,
            "evolved_version": 2,
            "metadata": {"description": "testing chain-of-thought"},
        }
        result = h._create_test(body)
        status, _ = _parse_result(result)
        assert status == 201
        mock_manager.start_test.assert_called_once_with(
            agent="claude",
            baseline_version=1,
            evolved_version=2,
            metadata={"description": "testing chain-of-thought"},
        )


# ---------------------------------------------------------------------------
# Record result
# ---------------------------------------------------------------------------


class TestRecordResult:
    def test_record_success(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        active_test = _make_test_obj("t1", "claude", "active")
        mock_manager.get_test.return_value = active_test

        updated_test = _make_test_obj("t1", "claude", "active")
        mock_manager.record_result.return_value = updated_test

        body = {
            "debate_id": "d1",
            "variant": "baseline",
            "won": True,
        }
        result = h._record_result("t1", body)
        status, data = _parse_result(result)
        assert status == 200
        assert data["message"] == "Result recorded"

    def test_record_test_not_found(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.get_test.return_value = None

        body = {"debate_id": "d1", "variant": "baseline", "won": True}
        result = h._record_result("nonexistent", body)
        status, _ = _parse_result(result)
        assert status == 404

    def test_record_concluded_test(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        concluded_test = _make_test_obj("t1", "claude", "concluded")
        mock_manager.get_test.return_value = concluded_test

        body = {"debate_id": "d1", "variant": "baseline", "won": True}
        result = h._record_result("t1", body)
        status, data = _parse_result(result)
        assert status == 400
        assert "concluded" in data["error"].lower()

    def test_record_missing_debate_id(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        active_test = _make_test_obj("t1", "claude", "active")
        mock_manager.get_test.return_value = active_test

        body = {"variant": "baseline", "won": True}
        result = h._record_result("t1", body)
        status, data = _parse_result(result)
        assert status == 400
        assert "debate_id" in data["error"]

    def test_record_invalid_variant(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        active_test = _make_test_obj("t1", "claude", "active")
        mock_manager.get_test.return_value = active_test

        body = {"debate_id": "d1", "variant": "invalid", "won": True}
        result = h._record_result("t1", body)
        status, data = _parse_result(result)
        assert status == 400
        assert "variant" in data["error"]

    def test_record_missing_won(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        active_test = _make_test_obj("t1", "claude", "active")
        mock_manager.get_test.return_value = active_test

        body = {"debate_id": "d1", "variant": "baseline"}
        result = h._record_result("t1", body)
        status, data = _parse_result(result)
        assert status == 400
        assert "won" in data["error"]

    def test_record_failed_returns_500(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        active_test = _make_test_obj("t1", "claude", "active")
        mock_manager.get_test.return_value = active_test
        mock_manager.record_result.return_value = None

        body = {"debate_id": "d1", "variant": "baseline", "won": True}
        result = h._record_result("t1", body)
        status, _ = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Conclude test
# ---------------------------------------------------------------------------


class TestConcludeTest:
    def test_conclude_success(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        mock_result = SimpleNamespace(
            test_id="t1",
            winner="evolved",
            confidence=0.95,
            recommendation="adopt evolved prompt",
            stats={"baseline_wins": 3, "evolved_wins": 7},
        )
        mock_manager.conclude_test.return_value = mock_result

        result = h._conclude_test("t1", {})
        status, data = _parse_result(result)
        assert status == 200
        assert data["message"] == "A/B test concluded"
        assert data["result"]["winner"] == "evolved"
        assert data["result"]["confidence"] == 0.95

    def test_conclude_with_force(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager

        mock_result = SimpleNamespace(
            test_id="t1",
            winner="baseline",
            confidence=0.6,
            recommendation="keep baseline",
            stats={},
        )
        mock_manager.conclude_test.return_value = mock_result

        result = h._conclude_test("t1", {"force": True})
        status, _ = _parse_result(result)
        assert status == 200
        mock_manager.conclude_test.assert_called_once_with("t1", force=True)

    def test_conclude_value_error(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.conclude_test.side_effect = ValueError("not enough data")

        result = h._conclude_test("t1", {})
        status, data = _parse_result(result)
        assert status == 400
        assert "failed" in data["error"].lower()

    def test_conclude_no_manager(self):
        h = _make_handler_instance()
        h._manager = None

        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._conclude_test("t1", {})
            status, _ = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# Cancel test
# ---------------------------------------------------------------------------


class TestCancelTest:
    def test_cancel_success(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.cancel_test.return_value = True

        result = h._cancel_test("t1")
        status, data = _parse_result(result)
        assert status == 200
        assert data["message"] == "A/B test cancelled"
        assert data["test_id"] == "t1"

    def test_cancel_not_found(self):
        h = _make_handler_instance()
        mock_manager = MagicMock()
        h._manager = mock_manager
        mock_manager.cancel_test.return_value = False

        result = h._cancel_test("nonexistent")
        status, data = _parse_result(result)
        assert status == 404

    def test_cancel_no_manager(self):
        h = _make_handler_instance()
        h._manager = None

        import aragora.server.handlers.evolution.ab_testing as mod

        original = mod.AB_TESTING_AVAILABLE
        try:
            mod.AB_TESTING_AVAILABLE = False
            result = h._cancel_test("t1")
            status, _ = _parse_result(result)
            assert status == 503
        finally:
            mod.AB_TESTING_AVAILABLE = original


# ---------------------------------------------------------------------------
# Handle routing for GET
# ---------------------------------------------------------------------------


class TestHandleGetRouting:
    def test_list_route(self):
        h = _make_handler_instance()
        with patch.object(h, "_list_tests", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/ab-tests", {}, _mock_http_handler())
            mock.assert_called_once()

    def test_get_test_route(self):
        h = _make_handler_instance()
        with patch.object(h, "_get_test", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/ab-tests/test-1", {}, _mock_http_handler())
            mock.assert_called_once_with("test-1")

    def test_active_test_route(self):
        h = _make_handler_instance()
        with patch.object(h, "_get_active_test", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/ab-tests/claude/active", {}, _mock_http_handler())
            mock.assert_called_once_with("claude")


# ---------------------------------------------------------------------------
# Handle routing for POST
# ---------------------------------------------------------------------------


class TestHandlePostRouting:
    def test_create_route(self):
        h = _make_handler_instance()
        body = {"agent": "claude", "baseline_version": 1, "evolved_version": 2}
        with patch.object(h, "_create_test", return_value=MagicMock(status_code=201)) as mock:
            h.handle_post("/api/evolution/ab-tests", body, _mock_http_handler())
            mock.assert_called_once_with(body)

    def test_record_route(self):
        h = _make_handler_instance()
        body = {"debate_id": "d1", "variant": "baseline", "won": True}
        with patch.object(h, "_record_result", return_value=MagicMock(status_code=200)) as mock:
            h.handle_post("/api/evolution/ab-tests/t1/record", body, _mock_http_handler())
            mock.assert_called_once_with("t1", body)

    def test_conclude_route(self):
        h = _make_handler_instance()
        body = {}
        with patch.object(h, "_conclude_test", return_value=MagicMock(status_code=200)) as mock:
            h.handle_post("/api/evolution/ab-tests/t1/conclude", body, _mock_http_handler())
            mock.assert_called_once_with("t1", body)

    def test_unknown_post_action_returns_none(self):
        h = _make_handler_instance()
        result = h.handle_post("/api/evolution/ab-tests/t1/unknown", {}, _mock_http_handler())
        assert result is None

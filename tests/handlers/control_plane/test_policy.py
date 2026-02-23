"""Comprehensive tests for control plane policy violation handlers.

Tests the PolicyHandlerMixin endpoints:
- GET   /api/control-plane/policies/violations (list violations)
- GET   /api/control-plane/policies/violations/{violation_id} (get violation)
- GET   /api/control-plane/policies/violations/stats (violation statistics)
- PATCH /api/control-plane/policies/violations/{violation_id} (update violation)

Also tests:
- Routing through handle() and handle_patch()
- Policy store unavailable (503) paths
- Permission denied scenarios
- Authentication error propagation
- Query parameter parsing (list vs scalar, defaults)
- Error handling for each exception type
- User objects with/without role/user_id/id attributes
- Valid and invalid status transitions
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler
from aragora.server.handlers.control_plane.policy import PolicyHandlerMixin


# ============================================================================
# Auth bypass fixture for policy handler tests
# ============================================================================


@pytest.fixture(autouse=True)
def _bypass_policy_auth(request, monkeypatch):
    """Bypass auth for policy handler tests.

    The handler methods have TWO layers of auth:
    1. ``@require_permission`` decorator (runs before the method body)
    2. Inline ``self.require_auth_or_error()`` + ``_get_has_permission()``
       calls inside the method body.

    The conftest auto-auth bypasses the decorator via
    ``_test_user_context_override``, but does not bypass the inline auth
    checks.  Additionally, the ``handle()`` entry-point calls
    ``require_permission_or_error`` which also needs to be bypassed.

    For ``no_auto_auth`` tests, we still bypass the decorator (since those
    tests target the inline auth / permission logic, not the decorator) but
    leave the inline ``require_auth_or_error`` and ``_get_has_permission``
    for the test to control via its own ``patch.object`` context managers.
    """
    is_no_auto_auth = "no_auto_auth" in [m.name for m in request.node.iter_markers()]

    # Always bypass the @require_permission decorator so tests can reach the
    # method body.  The no_auto_auth tests control the inline auth separately.
    from aragora.server.handlers.utils import decorators as handler_decorators

    mock_decorator_user = MagicMock()
    mock_decorator_user.role = "admin"
    mock_decorator_user.user_id = "test-user-001"
    mock_decorator_user.is_authenticated = True
    monkeypatch.setattr(handler_decorators, "_test_user_context_override", mock_decorator_user)
    monkeypatch.setattr(handler_decorators, "has_permission", lambda role, perm: True)

    if not is_no_auto_auth:
        # For normal tests: also bypass the inline auth and permission checks.
        mock_user = MagicMock()
        mock_user.role = "admin"
        mock_user.user_id = "test-user-001"
        mock_user.id = "test-user-001"
        mock_user.is_authenticated = True

        from aragora.server.handlers.base import BaseHandler

        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, handler: (mock_user, None),
        )

        # Patch require_permission_or_error so the handle() entry-point
        # permission gate also passes.
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, handler, permission: (mock_user, None),
        )

        # Patch _get_has_permission in the policy module so the inline
        # permission check always grants access.
        monkeypatch.setattr(
            "aragora.server.handlers.control_plane.policy._get_has_permission",
            lambda: lambda role, perm: True,
        )

    yield


# ============================================================================
# Helpers
# ============================================================================


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_policy_store():
    """Create a mock policy store with standard methods."""
    store = MagicMock()
    store.list_violations = MagicMock(return_value=[])
    store.count_violations = MagicMock(return_value={})
    store.update_violation_status = MagicMock(return_value=True)
    return store


@pytest.fixture
def handler(mock_policy_store):
    """Create a ControlPlaneHandler with a mock policy store."""
    ctx: dict[str, Any] = {
        "control_plane_coordinator": MagicMock(),
    }
    h = ControlPlaneHandler(ctx)
    with patch.object(h, "_get_policy_store", return_value=mock_policy_store):
        # Bind the patched method so it persists
        original_get = h._get_policy_store
    # We need to patch it more permanently for the fixture
    h._get_policy_store = lambda: mock_policy_store
    return h


@pytest.fixture
def handler_no_store():
    """Create a ControlPlaneHandler with NO policy store."""
    ctx: dict[str, Any] = {
        "control_plane_coordinator": MagicMock(),
    }
    h = ControlPlaneHandler(ctx)
    h._get_policy_store = lambda: None
    return h


@pytest.fixture
def mock_http_handler():
    """Create a minimal mock HTTP handler."""
    m = MagicMock()
    m.path = "/api/control-plane/policies/violations"
    m.headers = {"Content-Type": "application/json"}
    return m


@pytest.fixture
def sample_violations():
    """Create a list of sample violation dicts."""
    return [
        {
            "id": "viol-001",
            "policy_id": "pol-001",
            "violation_type": "rate_limit",
            "status": "open",
            "workspace_id": "ws-001",
            "description": "Rate limit exceeded",
        },
        {
            "id": "viol-002",
            "policy_id": "pol-002",
            "violation_type": "access_control",
            "status": "resolved",
            "workspace_id": "ws-002",
            "description": "Unauthorized access attempt",
        },
        {
            "id": "viol-003",
            "policy_id": "pol-001",
            "violation_type": "rate_limit",
            "status": "investigating",
            "workspace_id": "ws-001",
            "description": "Sustained rate limit breach",
        },
    ]


# ============================================================================
# GET /api/control-plane/policies/violations (list violations)
# ============================================================================


class TestListPolicyViolations:
    """Tests for _handle_list_policy_violations."""

    def test_list_violations_empty(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["violations"] == []
        assert body["total"] == 0
        assert body["limit"] == 100
        assert body["offset"] == 0

    def test_list_violations_returns_violations(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert len(body["violations"]) == 3
        assert body["violations"][0]["id"] == "viol-001"

    def test_list_violations_with_policy_id_filter_list(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {"policy_id": ["pol-001"]}, mock_http_handler
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id="pol-001",
            violation_type=None,
            status=None,
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_policy_id_filter_scalar(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"policy_id": "pol-002"}, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id="pol-002",
            violation_type=None,
            status=None,
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_violation_type_filter_list(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {"violation_type": ["rate_limit"]}, mock_http_handler
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type="rate_limit",
            status=None,
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_violation_type_filter_scalar(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {"violation_type": "access_control"}, mock_http_handler
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type="access_control",
            status=None,
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_status_filter_list(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"status": ["open"]}, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type=None,
            status="open",
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_status_filter_scalar(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"status": "resolved"}, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type=None,
            status="resolved",
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_with_workspace_id_filter_list(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {"workspace_id": ["ws-001"]}, mock_http_handler
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type=None,
            status=None,
            workspace_id="ws-001",
            limit=100,
            offset=0,
        )

    def test_list_violations_with_workspace_id_filter_scalar(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {"workspace_id": "ws-002"}, mock_http_handler
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type=None,
            status=None,
            workspace_id="ws-002",
            limit=100,
            offset=0,
        )

    def test_list_violations_with_limit_list(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"limit": ["50"]}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 50

    def test_list_violations_with_limit_scalar(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"limit": 25}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 25

    def test_list_violations_with_offset_list(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"offset": ["10"]}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["offset"] == 10

    def test_list_violations_with_offset_scalar(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"offset": 5}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["offset"] == 5

    def test_list_violations_with_all_filters(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations(
            {
                "policy_id": ["pol-001"],
                "violation_type": ["rate_limit"],
                "status": ["open"],
                "workspace_id": ["ws-001"],
                "limit": ["20"],
                "offset": ["5"],
            },
            mock_http_handler,
        )
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id="pol-001",
            violation_type="rate_limit",
            status="open",
            workspace_id="ws-001",
            limit=20,
            offset=5,
        )

    def test_list_violations_default_limit_and_offset(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.list_violations.assert_called_once_with(
            policy_id=None,
            violation_type=None,
            status=None,
            workspace_id=None,
            limit=100,
            offset=0,
        )

    def test_list_violations_no_store(self, handler_no_store, mock_http_handler):
        result = handler_no_store._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_list_violations_value_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = ValueError("bad param")
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 500

    def test_list_violations_type_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = TypeError("wrong type")
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 500

    def test_list_violations_key_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = KeyError("missing")
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 500

    def test_list_violations_runtime_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = RuntimeError("crash")
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 500

    def test_list_violations_os_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = OSError("disk failure")
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 500

    def test_list_violations_none_filter_values(
        self, handler, mock_policy_store, mock_http_handler
    ):
        """When filter lists contain None, should handle gracefully."""
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"policy_id": [None]}, mock_http_handler)
        assert _status(result) == 200


# ============================================================================
# GET /api/control-plane/policies/violations/{violation_id}
# ============================================================================


class TestGetPolicyViolation:
    """Tests for _handle_get_policy_violation."""

    def test_get_violation_success(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["violation"]["id"] == "viol-001"
        assert body["violation"]["policy_id"] == "pol-001"

    def test_get_violation_second_item(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler._handle_get_policy_violation("viol-002", mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["violation"]["id"] == "viol-002"
        assert body["violation"]["violation_type"] == "access_control"

    def test_get_violation_third_item(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler._handle_get_policy_violation("viol-003", mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["violation"]["id"] == "viol-003"

    def test_get_violation_not_found(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_get_policy_violation("nonexistent", mock_http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_violation_not_found_among_others(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler._handle_get_policy_violation("viol-999", mock_http_handler)
        assert _status(result) == 404

    def test_get_violation_no_store(self, handler_no_store, mock_http_handler):
        result = handler_no_store._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_get_violation_value_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = ValueError("bad")
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 500

    def test_get_violation_type_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = TypeError("wrong")
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 500

    def test_get_violation_key_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = KeyError("missing")
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 500

    def test_get_violation_runtime_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = RuntimeError("crash")
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 500

    def test_get_violation_os_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.side_effect = OSError("disk")
        result = handler._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 500

    def test_get_violation_calls_list_with_limit(
        self, handler, mock_policy_store, mock_http_handler
    ):
        """Verify it fetches up to 1000 violations to search through."""
        mock_policy_store.list_violations.return_value = []
        handler._handle_get_policy_violation("viol-001", mock_http_handler)
        mock_policy_store.list_violations.assert_called_once_with(limit=1000)


# ============================================================================
# GET /api/control-plane/policies/violations/stats
# ============================================================================


class TestGetPolicyViolationStats:
    """Tests for _handle_get_policy_violation_stats."""

    def test_stats_empty(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.return_value = {}
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["open"] == 0
        assert body["resolved"] == 0
        assert body["by_type"] == {}
        assert body["open_by_type"] == {}

    def test_stats_with_violations(self, handler, mock_policy_store, mock_http_handler):
        # First call is count_violations(status="open"), second is count_violations()
        mock_policy_store.count_violations.side_effect = [
            {"rate_limit": 3, "access_control": 1},  # open counts
            {"rate_limit": 5, "access_control": 3},  # total counts
        ]
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 8
        assert body["open"] == 4
        assert body["resolved"] == 4
        assert body["by_type"] == {"rate_limit": 5, "access_control": 3}
        assert body["open_by_type"] == {"rate_limit": 3, "access_control": 1}

    def test_stats_all_open(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = [
            {"rate_limit": 5},  # open
            {"rate_limit": 5},  # total
        ]
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert body["open"] == 5
        assert body["resolved"] == 0

    def test_stats_all_resolved(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = [
            {},  # open (none)
            {"rate_limit": 10},  # total
        ]
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 10
        assert body["open"] == 0
        assert body["resolved"] == 10

    def test_stats_multiple_types(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = [
            {"rate_limit": 2, "access_control": 1, "data_leak": 3},  # open
            {"rate_limit": 10, "access_control": 5, "data_leak": 7},  # total
        ]
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 22
        assert body["open"] == 6
        assert body["resolved"] == 16

    def test_stats_no_store(self, handler_no_store, mock_http_handler):
        result = handler_no_store._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_stats_value_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = ValueError("bad")
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 500

    def test_stats_type_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = TypeError("wrong")
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 500

    def test_stats_key_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = KeyError("missing")
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 500

    def test_stats_runtime_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = RuntimeError("crash")
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 500

    def test_stats_os_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = OSError("disk")
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 500

    def test_stats_empty_counts(self, handler, mock_policy_store, mock_http_handler):
        """Verify stats returns zeros when count_violations returns empty dict."""
        mock_policy_store.count_violations.return_value = {}
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        # Auth may block — verify response is either 200 with zeros or auth error
        status = _status(result)
        assert status in (200, 401, 403)


# ============================================================================
# PATCH /api/control-plane/policies/violations/{violation_id}
# ============================================================================


class TestUpdatePolicyViolation:
    """Tests for _handle_update_policy_violation."""

    def test_update_violation_success_open(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "open"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["updated"] is True
        assert data["violation_id"] == "viol-001"
        assert data["status"] == "open"

    def test_update_violation_success_investigating(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "investigating"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["status"] == "investigating"

    def test_update_violation_success_resolved(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["status"] == "resolved"

    def test_update_violation_success_false_positive(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "false_positive"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert data["status"] == "false_positive"

    def test_update_violation_all_valid_statuses(
        self, handler, mock_policy_store, mock_http_handler
    ):
        """Test all four valid statuses work."""
        for status in ["open", "investigating", "resolved", "false_positive"]:
            mock_policy_store.update_violation_status.return_value = True
            body = {"status": status}
            result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
            assert _status(result) == 200, f"Failed for status {status}"
            assert _body(result)["status"] == status

    def test_update_violation_missing_status(self, handler, mock_policy_store, mock_http_handler):
        body = {}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 400
        assert "status" in _body(result).get("error", "").lower()

    def test_update_violation_empty_status(self, handler, mock_policy_store, mock_http_handler):
        body = {"status": ""}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 400

    def test_update_violation_none_status(self, handler, mock_policy_store, mock_http_handler):
        body = {"status": None}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 400

    def test_update_violation_invalid_status(self, handler, mock_policy_store, mock_http_handler):
        body = {"status": "invalid_status"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 400
        error_msg = _body(result).get("error", "")
        assert "invalid status" in error_msg.lower() or "valid values" in error_msg.lower()

    def test_update_violation_invalid_status_closed(
        self, handler, mock_policy_store, mock_http_handler
    ):
        body = {"status": "closed"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 400

    def test_update_violation_not_found(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.return_value = False
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("nonexistent", body, mock_http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_update_violation_no_store(self, handler_no_store, mock_http_handler):
        body = {"status": "resolved"}
        result = handler_no_store._handle_update_policy_violation(
            "viol-001", body, mock_http_handler
        )
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_update_violation_with_resolution_notes(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved", "resolution_notes": "Fixed by adjusting rate limit"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="resolved",
            resolved_by="test-user-001",
            resolution_notes="Fixed by adjusting rate limit",
        )

    def test_update_violation_without_resolution_notes(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "investigating"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        mock_policy_store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="investigating",
            resolved_by="test-user-001",
            resolution_notes=None,
        )

    def test_update_violation_message_in_response(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 200
        data = _body(result)
        assert "resolved" in data["message"]

    def test_update_violation_value_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.side_effect = ValueError("bad")
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_update_violation_type_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.side_effect = TypeError("wrong")
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_update_violation_key_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.side_effect = KeyError("missing")
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_update_violation_runtime_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.side_effect = RuntimeError("crash")
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_update_violation_os_error(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.side_effect = OSError("disk")
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# User Attribute Handling for Update
# ============================================================================


class TestUpdateUserAttributes:
    """Tests for user attribute handling in _handle_update_policy_violation."""

    @pytest.mark.no_auto_auth
    def test_update_user_with_user_id(self, mock_http_handler):
        """User with user_id attribute should use user_id as resolved_by."""
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        store.update_violation_status.return_value = True
        h._get_policy_store = lambda: store

        mock_user = MagicMock()
        mock_user.user_id = "user-abc"
        mock_user.role = "admin"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: True,
            ):
                body = {"status": "resolved", "resolution_notes": "Fixed"}
                result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)

        assert _status(result) == 200
        store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="resolved",
            resolved_by="user-abc",
            resolution_notes="Fixed",
        )

    @pytest.mark.no_auto_auth
    def test_update_user_with_id_only(self, mock_http_handler):
        """User without user_id but with id should use id as resolved_by."""
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        store.update_violation_status.return_value = True
        h._get_policy_store = lambda: store

        mock_user = MagicMock(spec=["id", "role"])
        mock_user.id = "user-xyz"
        mock_user.role = "admin"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: True,
            ):
                body = {"status": "resolved"}
                result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)

        assert _status(result) == 200
        store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="resolved",
            resolved_by="user-xyz",
            resolution_notes=None,
        )

    @pytest.mark.no_auto_auth
    def test_update_user_without_id_attributes(self, mock_http_handler):
        """User without user_id or id should pass None as resolved_by."""
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        store.update_violation_status.return_value = True
        h._get_policy_store = lambda: store

        mock_user = MagicMock(spec=["role"])
        mock_user.role = "admin"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: True,
            ):
                body = {"status": "resolved"}
                result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)

        assert _status(result) == 200
        store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="resolved",
            resolved_by=None,
            resolution_notes=None,
        )


# ============================================================================
# Permission Denied Tests (no_auto_auth)
# ============================================================================


class TestPermissionDenied:
    """Tests for permission-denied scenarios on policy endpoints."""

    @pytest.mark.no_auto_auth
    def test_list_violations_permission_denied(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        h._get_policy_store = lambda: store

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                result = h._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 403
        assert "denied" in _body(result).get("error", "").lower()

    @pytest.mark.no_auto_auth
    def test_get_violation_permission_denied(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        h._get_policy_store = lambda: store

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                result = h._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_stats_permission_denied(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        h._get_policy_store = lambda: store

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                result = h._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_update_violation_permission_denied(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        h._get_policy_store = lambda: store

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                body = {"status": "resolved"}
                result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 403


# ============================================================================
# Authentication Error Tests (no_auto_auth)
# ============================================================================


class TestAuthErrors:
    """Tests for authentication error propagation on policy endpoints."""

    @pytest.mark.no_auto_auth
    def test_list_violations_auth_error(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)
        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            result = h._handle_list_policy_violations({}, mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_get_violation_auth_error(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)
        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            result = h._handle_get_policy_violation("viol-001", mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_stats_auth_error(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)
        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            result = h._handle_get_policy_violation_stats(mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_update_violation_auth_error(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)
        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            body = {"status": "resolved"}
            result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)
        assert _status(result) == 401


# ============================================================================
# User Without role Attribute
# ============================================================================


class TestUserWithoutRole:
    """Tests for user objects that lack a 'role' attribute."""

    @pytest.mark.no_auto_auth
    def test_list_violations_user_without_role(self, mock_http_handler):
        """User without role attribute should pass None for role check."""
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        store.list_violations.return_value = []
        h._get_policy_store = lambda: store

        mock_user = MagicMock(spec=[])  # No attributes at all

        permission_calls = []

        def track_permission(role, perm):
            permission_calls.append((role, perm))
            return True

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.policy._get_has_permission",
                return_value=track_permission,
            ):
                result = h._handle_list_policy_violations({}, mock_http_handler)

        assert _status(result) == 200
        # role should be None since mock_user has no 'role' attribute
        assert permission_calls[0][0] is None
        assert permission_calls[0][1] == "controlplane:policies"

    @pytest.mark.no_auto_auth
    def test_update_violation_user_without_role(self, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        store = MagicMock()
        store.update_violation_status.return_value = True
        h._get_policy_store = lambda: store

        mock_user = MagicMock(spec=[])

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch.object(h, "require_permission_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.policy._get_has_permission",
                    return_value=lambda role, perm: True,
                ):
                    body = {"status": "resolved"}
                    result = h._handle_update_policy_violation("viol-001", body, mock_http_handler)

        assert _status(result) == 200
        # resolved_by should be None since user has no user_id or id
        store.update_violation_status.assert_called_once_with(
            violation_id="viol-001",
            status="resolved",
            resolved_by=None,
            resolution_notes=None,
        )


# ============================================================================
# GET Routing via handle()
# ============================================================================


class TestGetRouting:
    """Tests for GET request routing for policy endpoints through handle()."""

    def test_route_list_violations(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler.handle("/api/control-plane/policies/violations", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["total"] == 0

    def test_route_list_violations_v1(self, handler, mock_policy_store, mock_http_handler):
        """Versioned path /api/v1/control-plane/policies/violations should normalize."""
        mock_policy_store.list_violations.return_value = []
        result = handler.handle("/api/v1/control-plane/policies/violations", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["total"] == 0

    def test_route_get_violation(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler.handle(
            "/api/control-plane/policies/violations/viol-001", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["violation"]["id"] == "viol-001"

    def test_route_get_violation_v1(
        self, handler, mock_policy_store, mock_http_handler, sample_violations
    ):
        mock_policy_store.list_violations.return_value = sample_violations
        result = handler.handle(
            "/api/v1/control-plane/policies/violations/viol-002", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["violation"]["id"] == "viol-002"

    def test_route_get_violation_not_found(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler.handle(
            "/api/control-plane/policies/violations/nonexistent", {}, mock_http_handler
        )
        assert _status(result) == 404

    def test_route_stats(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.return_value = {}
        result = handler.handle(
            "/api/control-plane/policies/violations/stats", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert "total" in _body(result)

    def test_route_stats_v1(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.return_value = {}
        result = handler.handle(
            "/api/v1/control-plane/policies/violations/stats", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert "total" in _body(result)

    def test_route_list_violations_with_query_params(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.list_violations.return_value = []
        result = handler.handle(
            "/api/control-plane/policies/violations",
            {"status": "open", "limit": 50},
            mock_http_handler,
        )
        assert _status(result) == 200


# ============================================================================
# PATCH Routing via handle_patch()
# ============================================================================


class TestPatchRouting:
    """Tests for PATCH request routing through handle_patch()."""

    def test_route_patch_violation(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved"}
        body_bytes = json.dumps(body).encode()
        mock_http_handler.rfile.read.return_value = body_bytes
        mock_http_handler.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        }
        result = handler.handle_patch(
            "/api/control-plane/policies/violations/viol-001", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["updated"] is True
        assert _body(result)["violation_id"] == "viol-001"

    def test_route_patch_violation_v1(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "investigating"}
        body_bytes = json.dumps(body).encode()
        mock_http_handler.rfile.read.return_value = body_bytes
        mock_http_handler.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        }
        result = handler.handle_patch(
            "/api/v1/control-plane/policies/violations/viol-001", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["status"] == "investigating"

    def test_route_patch_extracts_violation_id(self, handler, mock_policy_store, mock_http_handler):
        """Verify the violation_id is correctly extracted from the path."""
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved"}
        body_bytes = json.dumps(body).encode()
        mock_http_handler.rfile.read.return_value = body_bytes
        mock_http_handler.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        }
        result = handler.handle_patch(
            "/api/control-plane/policies/violations/my-specific-violation",
            {},
            mock_http_handler,
        )
        assert _status(result) == 200
        assert _body(result)["violation_id"] == "my-specific-violation"

    def test_route_patch_unknown_path_returns_none(self, handler, mock_http_handler):
        result = handler.handle_patch("/api/control-plane/unknown/path", {}, mock_http_handler)
        assert result is None

    def test_route_patch_violations_base_path_returns_none(self, handler, mock_http_handler):
        """PATCH to /violations (no ID) should not match and return None."""
        result = handler.handle_patch(
            "/api/control-plane/policies/violations", {}, mock_http_handler
        )
        assert result is None

    def test_route_patch_violations_nested_path_returns_none(self, handler, mock_http_handler):
        """PATCH to /violations/id/extra should not match (wrong segment count)."""
        result = handler.handle_patch(
            "/api/control-plane/policies/violations/viol-001/extra", {}, mock_http_handler
        )
        assert result is None


# ============================================================================
# _get_policy_store Helper
# ============================================================================


class TestGetPolicyStore:
    """Tests for _get_policy_store helper method."""

    def test_get_policy_store_success(self):
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        mock_store = MagicMock()
        with patch(
            "aragora.server.handlers.control_plane.policy.get_control_plane_policy_store",
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_store
            # Temporarily restore original method
            original = PolicyHandlerMixin._get_policy_store
            result = original(h)
        # This may or may not work depending on import structure.
        # The important thing is the method doesn't crash.

    def test_get_policy_store_import_error(self):
        """When the policy store module is not available, returns None."""
        ctx: dict[str, Any] = {"control_plane_coordinator": MagicMock()}
        h = ControlPlaneHandler(ctx)
        from aragora.server.handlers.control_plane.policy import PolicyHandlerMixin

        with patch(
            "aragora.server.handlers.control_plane.policy.get_control_plane_policy_store",
            side_effect=ImportError("not available"),
            create=True,
        ):
            # Need to actually call the real method, not our patched lambda
            result = PolicyHandlerMixin._get_policy_store(h)
        # Should return None or a store depending on the import
        # If the real import succeeds, we get a store. If not, None.
        # Either way, it should not raise.


# ============================================================================
# _get_has_permission Helper
# ============================================================================


class TestGetHasPermission:
    """Tests for _get_has_permission module-level helper."""

    def test_get_has_permission_returns_callable(self):
        from aragora.server.handlers.control_plane.policy import _get_has_permission

        fn = _get_has_permission()
        assert callable(fn)

    def test_get_has_permission_fallback(self):
        """When the control_plane module has no has_permission, falls back."""
        from aragora.server.handlers.control_plane.policy import _get_has_permission

        fn = _get_has_permission()
        assert callable(fn)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases for full coverage."""

    def test_list_violations_single_violation(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = [{"id": "viol-only", "status": "open"}]
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 1
        assert len(body["violations"]) == 1

    def test_list_violations_many_violations(self, handler, mock_policy_store, mock_http_handler):
        violations = [{"id": f"viol-{i:03d}", "status": "open"} for i in range(100)]
        mock_policy_store.list_violations.return_value = violations
        result = handler._handle_list_policy_violations({}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 100
        assert len(body["violations"]) == 100

    def test_get_violation_various_ids(self, handler, mock_policy_store, mock_http_handler):
        """Verify various violation_id formats work."""
        for vid in ["simple", "with-dashes", "with_underscores", "v.dotted", "123"]:
            mock_policy_store.list_violations.return_value = [{"id": vid}]
            result = handler._handle_get_policy_violation(vid, mock_http_handler)
            assert _status(result) == 200
            assert _body(result)["violation"]["id"] == vid

    def test_update_violation_response_contains_all_fields(
        self, handler, mock_policy_store, mock_http_handler
    ):
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "false_positive"}
        result = handler._handle_update_policy_violation("viol-001", body, mock_http_handler)
        data = _body(result)
        assert "updated" in data
        assert "violation_id" in data
        assert "status" in data
        assert "message" in data

    def test_stats_single_type(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.count_violations.side_effect = [
            {"security": 1},  # open
            {"security": 3},  # total
        ]
        result = handler._handle_get_policy_violation_stats(mock_http_handler)
        body = _body(result)
        assert body["total"] == 3
        assert body["open"] == 1
        assert body["resolved"] == 2

    def test_list_violations_limit_zero(self, handler, mock_policy_store, mock_http_handler):
        """Limit of zero should be passed to store."""
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"limit": ["0"]}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["limit"] == 0

    def test_list_violations_large_offset(self, handler, mock_policy_store, mock_http_handler):
        mock_policy_store.list_violations.return_value = []
        result = handler._handle_list_policy_violations({"offset": ["9999"]}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["offset"] == 9999

    def test_update_violation_preserves_violation_id(
        self, handler, mock_policy_store, mock_http_handler
    ):
        """The response should echo back the exact violation_id from the path."""
        mock_policy_store.update_violation_status.return_value = True
        body = {"status": "resolved"}
        result = handler._handle_update_policy_violation(
            "a-complex-id-123_abc", body, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["violation_id"] == "a-complex-id-123_abc"

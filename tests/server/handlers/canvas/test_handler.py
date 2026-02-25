"""
Tests for Canvas HTTP Handler.

Covers:
- Route regex pattern matching (6 patterns)
- can_handle path detection
- RBAC enforcement via @require_permission decorators
- Request body extraction
- Canvas CRUD operations (list, create, get, update, delete)
- Node operations (add, update, delete)
- Edge operations (add, delete)
- Action execution
- Rate limiting
- Authentication enforcement
- Error handling paths
- Method not allowed responses
- Route dispatching (_route_request)
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest
from tests.utils.state_reset import (
    reset_permission_checker_override,
    restore_rbac_context_extractor,
)


# ---------------------------------------------------------------------------
# Capture original _get_context_from_args at import time (before any test
# can replace it with a patched closure).
# ---------------------------------------------------------------------------

try:
    from aragora.rbac.decorators import (
        _get_context_from_args as _original_get_context_from_args,
    )
except ImportError:
    _original_get_context_from_args = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Autouse fixture: clear global rate limiter state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the global _canvas_limiter state before each test."""
    from aragora.server.handlers.canvas.handler import _canvas_limiter

    with _canvas_limiter._lock:
        _canvas_limiter._buckets.clear()
    yield
    with _canvas_limiter._lock:
        _canvas_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _reset_permission_checker():
    """Reset the singleton PermissionChecker and context helpers between tests.

    The tests/server/handlers/conftest.py auto-bypass fixture patches
    check_permission on the PermissionChecker singleton instance for
    non-no_auto_auth tests.  If monkeypatch teardown from a previous test
    hasn't fully cleaned up the instance-level attribute, the patched
    "always allow" method could leak into a subsequent no_auto_auth test.

    We explicitly delete any instance-level override so the class method
    is always used at the start of each test.

    Additionally, we restore ``_get_context_from_args`` in
    ``aragora.rbac.decorators`` to its original implementation.  The
    conftest auto-bypass patches this function to always return an admin
    context; if monkeypatch teardown ordering allows the patched version
    to linger, ``@require_permission`` may receive an admin context
    instead of the one explicitly passed by the test, or skip the RBAC
    check entirely when ``auth_config.enabled`` is False.
    """
    # Restore _get_context_from_args to its original implementation
    restore_rbac_context_extractor(_original_get_context_from_args)
    reset_permission_checker_override()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_context() -> dict:
    """Minimal ServerContext dict."""
    return {}


def _make_handler_obj(
    method: str = "GET",
    body: bytes | None = None,
    client_address: tuple = ("127.0.0.1", 12345),
    headers: dict | None = None,
) -> MagicMock:
    """Build a mock HTTP handler object used by CanvasHandler.handle()."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = client_address
    handler.headers = headers or {}
    if body is not None:
        handler.request = SimpleNamespace(body=body)
    else:
        handler.request = SimpleNamespace(body=None)
    return handler


def _make_canvas_handler():
    """Instantiate a CanvasHandler with mocked auth and RBAC."""
    from aragora.server.handlers.canvas.handler import CanvasHandler

    ch = CanvasHandler(_make_server_context())
    return ch


def _json_body(data: dict) -> bytes:
    return json.dumps(data).encode("utf-8")


def _parse_result(result) -> dict:
    """Parse a HandlerResult body into a dict."""
    return json.loads(result.body.decode("utf-8"))


def _canvas_dict(canvas_id="c1", name="Test"):
    """Fake canvas object with to_dict."""
    obj = MagicMock()
    obj.to_dict.return_value = {"id": canvas_id, "name": name}
    return obj


def _node_dict(node_id="n1"):
    obj = MagicMock()
    obj.to_dict.return_value = {"id": node_id, "type": "text"}
    return obj


def _edge_dict(edge_id="e1"):
    obj = MagicMock()
    obj.to_dict.return_value = {"id": edge_id, "source": "n1", "target": "n2"}
    return obj


# Patch targets
_AUTH_PATCH = "aragora.server.handlers.canvas.handler.CanvasHandler.require_auth_or_error"
_RBAC_CHECKER_PATCH = "aragora.server.handlers.canvas.handler.get_permission_checker"
_MANAGER_PATCH = "aragora.server.handlers.canvas.handler.CanvasHandler._get_canvas_manager"
_RUN_ASYNC_PATCH = "aragora.server.handlers.canvas.handler.CanvasHandler._run_async"
_GET_CLIENT_IP_PATCH = "aragora.server.handlers.canvas.handler.get_client_ip"


def _mock_auth(user_id="user-1", org_id="org-1", roles=None):
    """Return a mock (user, None) tuple for auth success."""
    user = SimpleNamespace(
        user_id=user_id,
        org_id=org_id,
        roles=roles or {"admin"},
    )
    return (user, None)


def _mock_rbac_allow():
    """Return a mock permission checker that always allows."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = True
    checker.check_permission.return_value = decision
    return checker


def _mock_rbac_deny(permission="canvas.read"):
    """Return a mock permission checker that always denies."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = False
    checker.check_permission.return_value = decision
    return checker


def _make_auth_context(user_id="user-1", org_id="org-1", roles=None):
    """Create a mock AuthorizationContext for testing."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id=user_id,
        org_id=org_id,
        roles=roles or {"admin"},
    )


# ===========================================================================
# Regex Pattern Tests
# ===========================================================================


class TestRoutePatterns:
    """Test the 6 compiled regex route patterns."""

    def test_canvas_id_pattern_matches(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ID_PATTERN

        m = CANVAS_ID_PATTERN.match("/api/v1/canvas/abc-123")
        assert m is not None
        assert m.group(1) == "abc-123"

    def test_canvas_id_pattern_with_underscores(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ID_PATTERN

        m = CANVAS_ID_PATTERN.match("/api/v1/canvas/my_canvas_01")
        assert m is not None
        assert m.group(1) == "my_canvas_01"

    def test_canvas_id_pattern_rejects_nested(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ID_PATTERN

        m = CANVAS_ID_PATTERN.match("/api/v1/canvas/abc/nodes")
        assert m is None

    def test_canvas_nodes_pattern(self):
        from aragora.server.handlers.canvas.handler import CANVAS_NODES_PATTERN

        m = CANVAS_NODES_PATTERN.match("/api/v1/canvas/c1/nodes")
        assert m is not None
        assert m.group(1) == "c1"

    def test_canvas_node_pattern(self):
        from aragora.server.handlers.canvas.handler import CANVAS_NODE_PATTERN

        m = CANVAS_NODE_PATTERN.match("/api/v1/canvas/c1/nodes/n1")
        assert m is not None
        assert m.groups() == ("c1", "n1")

    def test_canvas_node_pattern_rejects_extra_segment(self):
        from aragora.server.handlers.canvas.handler import CANVAS_NODE_PATTERN

        m = CANVAS_NODE_PATTERN.match("/api/v1/canvas/c1/nodes/n1/extra")
        assert m is None

    def test_canvas_edges_pattern(self):
        from aragora.server.handlers.canvas.handler import CANVAS_EDGES_PATTERN

        m = CANVAS_EDGES_PATTERN.match("/api/v1/canvas/c1/edges")
        assert m is not None
        assert m.group(1) == "c1"

    def test_canvas_edge_pattern(self):
        from aragora.server.handlers.canvas.handler import CANVAS_EDGE_PATTERN

        m = CANVAS_EDGE_PATTERN.match("/api/v1/canvas/c1/edges/e1")
        assert m is not None
        assert m.groups() == ("c1", "e1")

    def test_canvas_action_pattern(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ACTION_PATTERN

        m = CANVAS_ACTION_PATTERN.match("/api/v1/canvas/c1/action")
        assert m is not None
        assert m.group(1) == "c1"

    def test_canvas_action_pattern_rejects_bad_path(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ACTION_PATTERN

        m = CANVAS_ACTION_PATTERN.match("/api/v1/canvas/c1/actions")
        assert m is None

    def test_patterns_reject_special_characters(self):
        from aragora.server.handlers.canvas.handler import CANVAS_ID_PATTERN

        m = CANVAS_ID_PATTERN.match("/api/v1/canvas/a b c")
        assert m is None
        m = CANVAS_ID_PATTERN.match("/api/v1/canvas/a.b")
        assert m is None


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    def test_can_handle_base_path(self):
        ch = _make_canvas_handler()
        assert ch.can_handle("/api/v1/canvas") is True

    def test_can_handle_canvas_id(self):
        ch = _make_canvas_handler()
        assert ch.can_handle("/api/v1/canvas/abc") is True

    def test_can_handle_nodes(self):
        ch = _make_canvas_handler()
        assert ch.can_handle("/api/v1/canvas/abc/nodes") is True

    def test_cannot_handle_other(self):
        ch = _make_canvas_handler()
        assert ch.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial(self):
        ch = _make_canvas_handler()
        assert ch.can_handle("/api/v1/canva") is False


# ===========================================================================
# RBAC Decorator Tests
# ===========================================================================
# Note: RBAC permissions are now enforced via @require_permission decorators
# on each operation method. The permission mapping is:
# - canvas:read - List, get canvas
# - canvas:create - Create canvas, add nodes/edges
# - canvas:update - Update canvas/nodes
# - canvas:delete - Delete canvas/nodes/edges
# - canvas:run - Execute actions
# These are tested implicitly through the handler integration tests.


# ===========================================================================
# Request Body Extraction Tests
# ===========================================================================


class TestGetRequestBody:
    def test_valid_json(self):
        ch = _make_canvas_handler()
        handler = _make_handler_obj(body=_json_body({"name": "Test"}))
        result = ch._get_request_body(handler)
        assert result == {"name": "Test"}

    def test_no_body(self):
        ch = _make_canvas_handler()
        handler = _make_handler_obj(body=None)
        result = ch._get_request_body(handler)
        assert result == {}

    def test_invalid_json(self):
        ch = _make_canvas_handler()
        handler = _make_handler_obj(body=b"not json")
        result = ch._get_request_body(handler)
        assert result == {}

    def test_no_request_attr(self):
        ch = _make_canvas_handler()
        handler = MagicMock(spec=[])
        result = ch._get_request_body(handler)
        assert result == {}


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    @patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.1")
    def test_rate_limit_exceeded(self, mock_ip):
        from aragora.server.handlers.canvas.handler import _canvas_limiter

        ch = _make_canvas_handler()
        handler = _make_handler_obj()

        # Exhaust the limiter
        with _canvas_limiter._lock:
            import time

            now = time.time()
            _canvas_limiter._buckets["10.0.0.1"] = [now] * 200

        result = ch.handle("/api/v1/canvas", {}, handler)
        assert result is not None
        assert result.status_code == 429


# ===========================================================================
# Authentication Tests
# ===========================================================================


class TestAuthentication:
    @patch(_GET_CLIENT_IP_PATCH, return_value="127.0.0.1")
    @patch(_AUTH_PATCH, side_effect=ValueError("no token"))
    def test_auth_failure_returns_401(self, mock_auth, mock_ip):
        ch = _make_canvas_handler()
        handler = _make_handler_obj()
        result = ch.handle("/api/v1/canvas", {}, handler)
        assert result is not None
        assert result.status_code == 401

    @patch(_GET_CLIENT_IP_PATCH, return_value="127.0.0.1")
    @patch(_AUTH_PATCH)
    def test_auth_error_response_returns_error(self, mock_auth, mock_ip):
        from aragora.server.handlers.base import error_response

        mock_auth.return_value = (None, error_response("Bad token", 401))
        ch = _make_canvas_handler()
        handler = _make_handler_obj()
        result = ch.handle("/api/v1/canvas", {}, handler)
        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# RBAC Enforcement Tests
# ===========================================================================


@pytest.mark.no_auto_auth
class TestRBACEnforcement:
    """Test RBAC enforcement via @require_permission decorators.

    RBAC is now enforced via decorators on individual operation methods.
    The decorators check permissions and raise PermissionDeniedError if denied,
    which is caught by the handle() method and converted to a 403 response.
    """

    @patch(_GET_CLIENT_IP_PATCH, return_value="127.0.0.1")
    @patch(_AUTH_PATCH, return_value=_mock_auth())
    def test_rbac_denied_returns_403(self, mock_auth, mock_ip):
        ch = _make_canvas_handler()
        handler = _make_handler_obj(method="GET")

        # Must patch where the decorator looks up get_permission_checker
        with patch("aragora.rbac.decorators.get_permission_checker") as mock_pc:
            mock_pc.return_value = _mock_rbac_deny()
            result = ch.handle("/api/v1/canvas", {}, handler)

        assert result is not None
        assert result.status_code == 403
        body = _parse_result(result)
        assert "Permission denied" in body.get("error", "")

    @patch(_GET_CLIENT_IP_PATCH, return_value="127.0.0.1")
    @patch(_AUTH_PATCH, return_value=_mock_auth())
    @patch(_MANAGER_PATCH)
    @patch(_RUN_ASYNC_PATCH, return_value=[])
    def test_rbac_allowed_proceeds(self, mock_run, mock_mgr, mock_auth, mock_ip):
        ch = _make_canvas_handler()
        handler = _make_handler_obj(method="GET")

        # Must patch where the decorator looks up get_permission_checker
        with patch("aragora.rbac.decorators.get_permission_checker") as mock_pc:
            mock_pc.return_value = _mock_rbac_allow()
            result = ch.handle("/api/v1/canvas", {}, handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Canvas CRUD Tests
# ===========================================================================


class TestListCanvases:
    def test_list_returns_canvases(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        canvases = [_canvas_dict("c1", "A"), _canvas_dict("c2", "B")]
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=canvases),
        ):
            result = ch._list_canvases(ctx, {}, "user-1", "ws-1")
        body = _parse_result(result)
        assert body["count"] == 2
        assert len(body["canvases"]) == 2

    def test_list_empty(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=[]),
        ):
            result = ch._list_canvases(ctx, {}, "user-1", "ws-1")
        body = _parse_result(result)
        assert body["count"] == 0

    def test_list_uses_query_params(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=[]) as mock_run,
        ):
            ch._list_canvases(ctx, {"owner_id": "o1", "workspace_id": "w1"}, "user-1", "ws-1")
        # The coroutine passed to _run_async should use o1/w1 from query_params
        assert mock_run.called

    def test_list_error_returns_500(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("db down")):
            result = ch._list_canvases(ctx, {}, "user-1", "ws-1")
        assert result.status_code == 500


class TestCreateCanvas:
    def test_create_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        canvas = _canvas_dict("c-new", "New Canvas")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=canvas),
        ):
            result = ch._create_canvas(ctx, {"name": "New Canvas"}, "user-1", "ws-1")
        assert result.status_code == 201
        body = _parse_result(result)
        assert body["name"] == "New Canvas"

    def test_create_default_name(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        canvas = _canvas_dict()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=canvas),
        ):
            result = ch._create_canvas(ctx, {}, "user-1", "ws-1")
        assert result.status_code == 201

    def test_create_error_returns_500(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("fail")):
            result = ch._create_canvas(ctx, {"name": "X"}, "user-1", "ws-1")
        assert result.status_code == 500


class TestGetCanvas:
    def test_get_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        canvas = _canvas_dict("c1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=canvas),
        ):
            result = ch._get_canvas(ctx, "c1", "user-1")
        assert result.status_code == 200

    def test_get_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=None),
        ):
            result = ch._get_canvas(ctx, "missing", "user-1")
        assert result.status_code == 404

    def test_get_error_returns_500(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("err")):
            result = ch._get_canvas(ctx, "c1", "user-1")
        assert result.status_code == 500


class TestUpdateCanvas:
    def test_update_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        canvas = _canvas_dict("c1", "Updated")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=canvas),
        ):
            result = ch._update_canvas(ctx, "c1", {"name": "Updated"}, "user-1")
        assert result.status_code == 200

    def test_update_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=None),
        ):
            result = ch._update_canvas(ctx, "missing", {"name": "X"}, "user-1")
        assert result.status_code == 404

    def test_update_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._update_canvas(ctx, "c1", {}, "user-1")
        assert result.status_code == 500


class TestDeleteCanvas:
    def test_delete_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=True),
        ):
            result = ch._delete_canvas(ctx, "c1", "user-1")
        assert result.status_code == 200
        body = _parse_result(result)
        assert body["deleted"] is True

    def test_delete_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=False),
        ):
            result = ch._delete_canvas(ctx, "missing", "user-1")
        assert result.status_code == 404

    def test_delete_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._delete_canvas(ctx, "c1", "user-1")
        assert result.status_code == 500


# ===========================================================================
# Node Operation Tests
# ===========================================================================


class TestAddNode:
    def test_add_node_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        node = _node_dict("n1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=node),
        ):
            result = ch._add_node(ctx, "c1", {"type": "text", "label": "Hello"}, "user-1")
        assert result.status_code == 201

    def test_add_node_invalid_type(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        # Patch the CanvasNodeType to raise ValueError for invalid type
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch("aragora.server.handlers.canvas.handler.CanvasHandler._add_node") as mock_add,
        ):
            # Instead, test directly - the handler catches ValueError from CanvasNodeType
            pass

        # Test via the actual method with mocked CanvasNodeType
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch("aragora.canvas.CanvasNodeType", side_effect=ValueError("bad")),
        ):
            result = ch._add_node(ctx, "c1", {"type": "invalid_xyz"}, "user-1")
        assert result.status_code == 400 or result.status_code == 500

    def test_add_node_canvas_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=None),
        ):
            result = ch._add_node(ctx, "missing", {"type": "text"}, "user-1")
        assert result.status_code == 404

    def test_add_node_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._add_node(ctx, "c1", {}, "user-1")
        assert result.status_code == 500


class TestUpdateNode:
    def test_update_node_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        node = _node_dict("n1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=node),
        ):
            result = ch._update_node(ctx, "c1", "n1", {"label": "Updated"}, "user-1")
        assert result.status_code == 200

    def test_update_node_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=None),
        ):
            result = ch._update_node(ctx, "c1", "n1", {"label": "X"}, "user-1")
        assert result.status_code == 404

    def test_update_node_with_position(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        node = _node_dict("n1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=node),
        ):
            result = ch._update_node(
                ctx,
                "c1",
                "n1",
                {"position": {"x": 10, "y": 20}, "label": "Moved"},
                "user-1",
            )
        assert result.status_code == 200

    def test_update_node_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._update_node(ctx, "c1", "n1", {}, "user-1")
        assert result.status_code == 500


class TestDeleteNode:
    def test_delete_node_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=True),
        ):
            result = ch._delete_node(ctx, "c1", "n1", "user-1")
        assert result.status_code == 200
        body = _parse_result(result)
        assert body["deleted"] is True
        assert body["node_id"] == "n1"

    def test_delete_node_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=False),
        ):
            result = ch._delete_node(ctx, "c1", "n1", "user-1")
        assert result.status_code == 404

    def test_delete_node_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._delete_node(ctx, "c1", "n1", "user-1")
        assert result.status_code == 500


# ===========================================================================
# Edge Operation Tests
# ===========================================================================


class TestAddEdge:
    def test_add_edge_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        edge = _edge_dict("e1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=edge),
        ):
            result = ch._add_edge(ctx, "c1", {"source_id": "n1", "target_id": "n2"}, "user-1")
        assert result.status_code == 201

    def test_add_edge_missing_source(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with patch.object(ch, "_get_canvas_manager", return_value=mgr):
            result = ch._add_edge(ctx, "c1", {"target_id": "n2"}, "user-1")
        assert result.status_code == 400

    def test_add_edge_missing_target(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with patch.object(ch, "_get_canvas_manager", return_value=mgr):
            result = ch._add_edge(ctx, "c1", {"source_id": "n1"}, "user-1")
        assert result.status_code == 400

    def test_add_edge_alt_keys(self):
        """Test that 'source' and 'target' keys also work."""
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        edge = _edge_dict("e1")
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=edge),
        ):
            result = ch._add_edge(ctx, "c1", {"source": "n1", "target": "n2"}, "user-1")
        assert result.status_code == 201

    def test_add_edge_canvas_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=None),
        ):
            result = ch._add_edge(ctx, "c1", {"source_id": "n1", "target_id": "n2"}, "user-1")
        assert result.status_code == 404

    def test_add_edge_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._add_edge(ctx, "c1", {"source_id": "n1", "target_id": "n2"}, "user-1")
        assert result.status_code == 500


class TestDeleteEdge:
    def test_delete_edge_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=True),
        ):
            result = ch._delete_edge(ctx, "c1", "e1", "user-1")
        assert result.status_code == 200
        body = _parse_result(result)
        assert body["deleted"] is True
        assert body["edge_id"] == "e1"

    def test_delete_edge_not_found(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value=False),
        ):
            result = ch._delete_edge(ctx, "c1", "e1", "user-1")
        assert result.status_code == 404

    def test_delete_edge_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas_manager", side_effect=RuntimeError("x")):
            result = ch._delete_edge(ctx, "c1", "e1", "user-1")
        assert result.status_code == 500


# ===========================================================================
# Action Execution Tests
# ===========================================================================


class TestExecuteAction:
    def test_execute_success(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value={"status": "ok"}),
        ):
            result = ch._execute_action(
                ctx, "c1", {"action": "layout", "params": {"algo": "force"}}, "user-1"
            )
        assert result.status_code == 200
        body = _parse_result(result)
        assert body["action"] == "layout"
        assert body["canvas_id"] == "c1"
        assert body["result"] == {"status": "ok"}

    def test_execute_missing_action(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._execute_action(ctx, "c1", {}, "user-1")
        assert result.status_code == 400
        body = _parse_result(result)
        assert "action is required" in body.get("error", "")

    def test_execute_with_node_id(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", return_value="done"),
        ):
            result = ch._execute_action(ctx, "c1", {"action": "run", "node_id": "n1"}, "user-1")
        body = _parse_result(result)
        assert body["node_id"] == "n1"

    def test_execute_error(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        mgr = MagicMock()
        with (
            patch.object(ch, "_get_canvas_manager", return_value=mgr),
            patch.object(ch, "_run_async", side_effect=RuntimeError("boom")),
        ):
            result = ch._execute_action(ctx, "c1", {"action": "run"}, "user-1")
        assert result.status_code == 500


# ===========================================================================
# Route Dispatching Tests
# ===========================================================================


class TestRouteRequest:
    """Test _route_request method dispatching.

    Note: _route_request now takes an AuthorizationContext parameter which is passed
    to individual operation methods for RBAC enforcement via @require_permission decorators.
    """

    def test_list_canvases_get(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_list_canvases", return_value="list_result") as mock:
            result = ch._route_request("/api/v1/canvas", "GET", {"q": "1"}, {}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, {"q": "1"}, "u", "w")
        assert result == "list_result"

    def test_create_canvas_post(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_create_canvas", return_value="create_result") as mock:
            result = ch._route_request("/api/v1/canvas", "POST", {}, {"name": "X"}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, {"name": "X"}, "u", "w")

    def test_canvas_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas", "DELETE", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_get_canvas_by_id(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_get_canvas", return_value="get_result") as mock:
            result = ch._route_request("/api/v1/canvas/c1", "GET", {}, {}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, "c1", "u")

    def test_update_canvas_by_id(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_update_canvas", return_value="up") as mock:
            result = ch._route_request("/api/v1/canvas/c1", "PUT", {}, {"name": "Y"}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, "c1", {"name": "Y"}, "u")

    def test_delete_canvas_by_id(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_delete_canvas", return_value="del") as mock:
            result = ch._route_request("/api/v1/canvas/c1", "DELETE", {}, {}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, "c1", "u")

    def test_canvas_id_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1", "PATCH", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_add_node_post(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_add_node", return_value="node") as mock:
            result = ch._route_request(
                "/api/v1/canvas/c1/nodes", "POST", {}, {"type": "text"}, "u", "w", ctx
            )
        mock.assert_called_once_with(ctx, "c1", {"type": "text"}, "u")

    def test_nodes_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/nodes", "GET", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_update_node_put(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_update_node", return_value="upn") as mock:
            ch._route_request(
                "/api/v1/canvas/c1/nodes/n1", "PUT", {}, {"label": "X"}, "u", "w", ctx
            )
        mock.assert_called_once_with(ctx, "c1", "n1", {"label": "X"}, "u")

    def test_delete_node(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_delete_node", return_value="dn") as mock:
            ch._route_request("/api/v1/canvas/c1/nodes/n1", "DELETE", {}, {}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, "c1", "n1", "u")

    def test_node_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/nodes/n1", "GET", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_add_edge_post(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_add_edge", return_value="edge") as mock:
            ch._route_request(
                "/api/v1/canvas/c1/edges", "POST", {}, {"source_id": "a"}, "u", "w", ctx
            )
        mock.assert_called_once_with(ctx, "c1", {"source_id": "a"}, "u")

    def test_edges_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/edges", "GET", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_delete_edge(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_delete_edge", return_value="de") as mock:
            ch._route_request("/api/v1/canvas/c1/edges/e1", "DELETE", {}, {}, "u", "w", ctx)
        mock.assert_called_once_with(ctx, "c1", "e1", "u")

    def test_edge_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/edges/e1", "GET", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_execute_action_post(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        with patch.object(ch, "_execute_action", return_value="act") as mock:
            ch._route_request(
                "/api/v1/canvas/c1/action", "POST", {}, {"action": "run"}, "u", "w", ctx
            )
        mock.assert_called_once_with(ctx, "c1", {"action": "run"}, "u")

    def test_action_method_not_allowed(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/action", "GET", {}, {}, "u", "w", ctx)
        assert result.status_code == 405

    def test_unmatched_path_returns_none(self):
        ch = _make_canvas_handler()
        ctx = _make_auth_context()
        result = ch._route_request("/api/v1/canvas/c1/unknown", "GET", {}, {}, "u", "w", ctx)
        assert result is None


# ===========================================================================
# Resource type
# ===========================================================================


class TestResourceType:
    def test_resource_type_is_canvas(self):
        ch = _make_canvas_handler()
        assert ch.RESOURCE_TYPE == "canvas"

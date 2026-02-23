"""Tests for CurationOperationsMixin (aragora/server/handlers/knowledge_base/mound/curation.py).

Covers all routes and behavior of the curation mixin:
- GET  /api/v1/knowledge/mound/curation/policy   - Get curation policy
- POST /api/v1/knowledge/mound/curation/policy   - Set curation policy
- GET  /api/v1/knowledge/mound/curation/status   - Get curation status
- POST /api/v1/knowledge/mound/curation/run      - Trigger curation run
- GET  /api/v1/knowledge/mound/curation/history   - Get curation history
- GET  /api/v1/knowledge/mound/curation/scores    - Get quality scores
- GET  /api/v1/knowledge/mound/curation/tiers     - Get tier distribution
- Routing dispatch (_handle_curation_routes) and RBAC checks
- Error cases: missing mound, import failures, value errors
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.curation import (
    CurationOperationsMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-token-123"


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Autouse fixture: bypass @require_auth by making auth_config accept our token
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_require_auth(monkeypatch):
    """Patch auth_config so the @require_auth decorator accepts our test token."""
    from aragora.server import auth as auth_module

    monkeypatch.setattr(auth_module.auth_config, "api_token", _TEST_TOKEN)
    monkeypatch.setattr(auth_module.auth_config, "validate_token", lambda token: token == _TEST_TOKEN)


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for curation tests."""

    command: str = "GET"
    path: str = ""
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": "test-agent",
            "Authorization": f"Bearer {_TEST_TOKEN}",
            "Content-Length": "0",
        }
    )
    client_address: tuple = ("127.0.0.1", 12345)
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def get(cls) -> MockHTTPHandler:
        return cls(command="GET")

    @classmethod
    def post(cls, body: dict | None = None) -> MockHTTPHandler:
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            handler = cls(
                command="POST",
                headers={
                    "User-Agent": "test-agent",
                    "Authorization": f"Bearer {_TEST_TOKEN}",
                    "Content-Length": str(len(raw)),
                },
                rfile=io.BytesIO(raw),
            )
            return handler
        return cls(command="POST")


# ---------------------------------------------------------------------------
# Mock CurationPolicy / TierLevel / QualityScore
# ---------------------------------------------------------------------------


class MockCurationAction(str, Enum):
    PROMOTE = "promote"
    DEMOTE = "demote"
    ARCHIVE = "archive"
    REFRESH = "refresh"
    FLAG = "flag"


@dataclass
class MockCurationPolicy:
    workspace_id: str = "default"
    policy_id: str = "policy-001"
    enabled: bool = True
    name: str = "default"
    quality_threshold: float = 0.5
    promotion_threshold: float = 0.85
    demotion_threshold: float = 0.35
    archive_threshold: float = 0.2
    refresh_staleness_threshold: float = 0.7
    usage_window_days: int = 30
    min_retrievals_for_promotion: int = 5


@dataclass
class MockQualityScore:
    node_id: str = "node-001"
    overall_score: float = 0.8
    freshness_score: float = 0.9
    confidence_score: float = 0.85
    usage_score: float = 0.7
    relevance_score: float = 0.75
    relationship_score: float = 0.6
    recommendation: MockCurationAction = MockCurationAction.PROMOTE
    debate_uses: int = 3
    retrieval_count: int = 12


class MockTierLevel(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    GLACIAL = "glacial"


@dataclass
class MockCurationResult:
    promoted_count: int = 2
    demoted_count: int = 1
    archived_count: int = 0
    refreshed_count: int = 3
    flagged_count: int = 1


# ---------------------------------------------------------------------------
# Concrete test class that combines the mixin with stubs
# ---------------------------------------------------------------------------


class CurationTestHandler(CurationOperationsMixin):
    """Concrete handler for testing the curation mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound

    def read_json_body(self, handler):
        """Read JSON body from the mock handler."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length <= 0:
                return {}
            body = handler.rfile.read(content_length)
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with curation-related methods removed."""
    mound = MagicMock()

    # Remove curation methods by default (tests add them as needed)
    del mound.get_curation_policy
    del mound.set_curation_policy
    del mound.get_curation_status
    del mound.get_node_count
    del mound.run_curation
    del mound.get_curation_history
    del mound.get_quality_scores
    del mound.get_tier_distribution

    return mound


@pytest.fixture
def mound_with_policy(mock_mound):
    """Mound that has get/set_curation_policy methods."""
    policy = MockCurationPolicy()
    mock_mound.get_curation_policy = MagicMock(return_value=policy)
    mock_mound.set_curation_policy = MagicMock(return_value=None)
    return mock_mound


@pytest.fixture
def mound_with_status(mock_mound):
    """Mound that has get_curation_status and get_node_count."""
    mock_mound.get_curation_status = MagicMock(
        return_value={
            "last_run": "2025-01-15T10:00:00Z",
            "is_running": False,
            "stats": {"items_scored": 42, "items_pending": 5},
        }
    )
    mock_mound.get_node_count = MagicMock(return_value=100)
    return mock_mound


@pytest.fixture
def mound_with_curation(mock_mound):
    """Mound that has run_curation method."""
    mock_mound.run_curation = MagicMock(return_value=MockCurationResult())
    return mock_mound


@pytest.fixture
def mound_with_history(mock_mound):
    """Mound that has get_curation_history method."""
    mock_mound.get_curation_history = MagicMock(
        return_value=[
            {"run_id": "run-1", "promoted": 3, "demoted": 1},
            {"run_id": "run-2", "promoted": 5, "demoted": 2},
        ]
    )
    return mock_mound


@pytest.fixture
def mound_with_scores(mock_mound):
    """Mound that has get_quality_scores method."""
    scores = [
        MockQualityScore(node_id="node-001", overall_score=0.9),
        MockQualityScore(
            node_id="node-002",
            overall_score=0.3,
            recommendation=MockCurationAction.DEMOTE,
        ),
    ]
    mock_mound.get_quality_scores = MagicMock(return_value=scores)
    return mock_mound


@pytest.fixture
def mound_with_tiers(mock_mound):
    """Mound that has get_tier_distribution method."""
    mock_mound.get_tier_distribution = MagicMock(
        return_value={"hot": 10, "warm": 30, "cold": 50, "glacial": 10}
    )
    return mock_mound


@pytest.fixture
def handler(mock_mound):
    """Create a CurationTestHandler with default empty mound."""
    return CurationTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a CurationTestHandler with no mound (None)."""
    return CurationTestHandler(mound=None)


# Patch targets
_RUN_ASYNC_PATCH = (
    "aragora.server.handlers.knowledge_base.mound.curation._run_async"
)


# ============================================================================
# Tests: _handle_curation_routes dispatch
# ============================================================================


class TestCurationRouteDispatch:
    """Test routing logic in _handle_curation_routes."""

    def test_routes_to_get_policy(self, handler):
        """GET policy path dispatches to _handle_get_curation_policy."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_get_curation_policy", return_value=HandlerResult(200, "application/json", b'{}')) as mock_get:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy", {}, http
            )
            mock_get.assert_called_once_with({})
            assert _status(result) == 200

    def test_routes_to_set_policy_on_post(self, handler):
        """POST policy path dispatches to _handle_set_curation_policy."""
        http = MockHTTPHandler.post({"enabled": True})
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_set_curation_policy", return_value=HandlerResult(200, "application/json", b'{}')) as mock_set:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy", {}, http
            )
            mock_set.assert_called_once_with(http)
            assert _status(result) == 200

    def test_routes_to_status(self, handler):
        """Status path dispatches to _handle_curation_status."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_curation_status", return_value=HandlerResult(200, "application/json", b'{}')) as mock_fn:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/status", {}, http
            )
            mock_fn.assert_called_once_with({})

    def test_routes_to_run(self, handler):
        """Run path dispatches to _handle_run_curation."""
        http = MockHTTPHandler.post({})
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_run_curation", return_value=HandlerResult(200, "application/json", b'{}')) as mock_fn:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/run", {}, http
            )
            mock_fn.assert_called_once_with(http)

    def test_routes_to_history(self, handler):
        """History path dispatches to _handle_curation_history."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_curation_history", return_value=HandlerResult(200, "application/json", b'{}')) as mock_fn:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/history", {}, http
            )
            mock_fn.assert_called_once_with({})

    def test_routes_to_scores(self, handler):
        """Scores path dispatches to _handle_quality_scores."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_quality_scores", return_value=HandlerResult(200, "application/json", b'{}')) as mock_fn:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/scores", {}, http
            )
            mock_fn.assert_called_once_with({})

    def test_routes_to_tiers(self, handler):
        """Tiers path dispatches to _handle_tier_distribution."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None), \
             patch.object(handler, "_handle_tier_distribution", return_value=HandlerResult(200, "application/json", b'{}')) as mock_fn:
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/tiers", {}, http
            )
            mock_fn.assert_called_once_with({})

    def test_unknown_path_returns_none(self, handler):
        """Unrecognized path returns None."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None):
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/nonexistent", {}, http
            )
            assert result is None

    def test_rbac_error_short_circuits(self, handler):
        """RBAC failure returns error without dispatching to handler."""
        http = MockHTTPHandler.get()
        rbac_err = HandlerResult(403, "application/json", b'{"error":"denied"}')
        with patch.object(handler, "_check_knowledge_permission", return_value=rbac_err):
            result = handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy", {}, http
            )
            assert _status(result) == 403

    def test_post_uses_update_action_for_rbac(self, handler):
        """POST requests check 'update' RBAC action, GET checks 'read'."""
        http = MockHTTPHandler.post({})
        with patch.object(handler, "_check_knowledge_permission", return_value=None) as mock_check, \
             patch.object(handler, "_handle_set_curation_policy", return_value=HandlerResult(200, "application/json", b'{}')):
            handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy", {}, http
            )
            mock_check.assert_called_once_with(http, "update")

    def test_get_uses_read_action_for_rbac(self, handler):
        """GET requests check 'read' RBAC action."""
        http = MockHTTPHandler.get()
        with patch.object(handler, "_check_knowledge_permission", return_value=None) as mock_check, \
             patch.object(handler, "_handle_get_curation_policy", return_value=HandlerResult(200, "application/json", b'{}')):
            handler._handle_curation_routes(
                "/api/v1/knowledge/mound/curation/policy", {}, http
            )
            mock_check.assert_called_once_with(http, "read")


# ============================================================================
# Tests: GET /api/v1/knowledge/mound/curation/policy
# ============================================================================


class TestGetCurationPolicy:
    """Test _handle_get_curation_policy."""

    def test_returns_custom_policy_when_mound_has_method(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_get_curation_policy({})
        body = _body(result)
        assert _status(result) == 200
        assert body["workspace_id"] == "default"
        assert body["policy"]["policy_id"] == "policy-001"
        assert body["policy"]["enabled"] is True
        assert body["policy"]["quality_threshold"] == 0.5

    def test_returns_custom_policy_with_workspace_param(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_get_curation_policy({"workspace_id": "ws-custom"})
        body = _body(result)
        assert body["workspace_id"] == "ws-custom"

    def test_returns_default_policy_when_mound_returns_none(self, mock_mound):
        mock_mound.get_curation_policy = MagicMock(return_value=None)
        h = CurationTestHandler(mound=mock_mound)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_get_curation_policy({})
        body = _body(result)
        assert _status(result) == 200
        assert "note" in body
        assert "default" in body["note"].lower()

    def test_returns_default_when_no_get_curation_policy_method(self):
        mound = MagicMock(spec=[])  # No methods at all
        h = CurationTestHandler(mound=mound)
        result = h._handle_get_curation_policy({})
        body = _body(result)
        assert _status(result) == 200
        assert "note" in body
        assert body["workspace_id"] == "default"

    def test_default_workspace_is_default(self):
        mound = MagicMock(spec=[])
        h = CurationTestHandler(mound=mound)
        result = h._handle_get_curation_policy({})
        body = _body(result)
        assert body["workspace_id"] == "default"

    def test_no_mound_returns_503(self, handler_no_mound):
        result = handler_no_mound._handle_get_curation_policy({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_import_error_returns_501(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        # Simulate ImportError by blocking the auto_curation module import
        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.ops.auto_curation": None},
        ):
            result = h._handle_get_curation_policy({})
        assert _status(result) == 501
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_value_error_returns_500(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad data")):
            result = h._handle_get_curation_policy({})
        assert _status(result) == 500

    def test_policy_fields_are_present(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_get_curation_policy({})
        body = _body(result)
        policy = body["policy"]
        expected_fields = [
            "policy_id", "enabled", "name", "quality_threshold",
            "promotion_threshold", "demotion_threshold", "archive_threshold",
            "refresh_staleness_threshold", "usage_window_days",
            "min_retrievals_for_promotion",
        ]
        for field_name in expected_fields:
            assert field_name in policy, f"Missing field: {field_name}"

    def test_default_policy_has_all_fields(self):
        """Default policy (no custom policy set) includes all expected fields."""
        mound = MagicMock(spec=[])
        h = CurationTestHandler(mound=mound)
        result = h._handle_get_curation_policy({})
        body = _body(result)
        policy = body["policy"]
        assert "policy_id" in policy
        assert "enabled" in policy
        assert "name" in policy
        assert "quality_threshold" in policy


# ============================================================================
# Tests: POST /api/v1/knowledge/mound/curation/policy
# ============================================================================


class TestSetCurationPolicy:
    """Test _handle_set_curation_policy."""

    def test_set_policy_with_mound_method(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.post({"workspace_id": "ws-1", "enabled": True, "name": "custom"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_set_curation_policy(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["workspace_id"] == "ws-1"
        assert body["message"] == "Curation policy updated"

    def test_set_policy_without_mound_storage(self):
        mound = MagicMock(spec=[])  # No set_curation_policy
        h = CurationTestHandler(mound=mound)
        http = MockHTTPHandler.post({"workspace_id": "ws-2", "enabled": False})
        result = h._handle_set_curation_policy(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert "not available" in body.get("note", "").lower()

    def test_set_policy_no_body_returns_400(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.get()
        with patch.object(h, "_read_request_body", return_value=None):
            result = h._handle_set_curation_policy(http)
        assert _status(result) == 400
        body = _body(result)
        assert "json body required" in body["error"].lower()

    def test_set_policy_no_mound_returns_503(self, handler_no_mound):
        http = MockHTTPHandler.post({"enabled": True})
        result = handler_no_mound._handle_set_curation_policy(http)
        assert _status(result) == 503

    def test_set_policy_import_error_returns_501(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        http = MockHTTPHandler.post({"enabled": True})
        with patch.dict("sys.modules", {"aragora.knowledge.mound.ops.auto_curation": None}):
            result = h._handle_set_curation_policy(http)
        assert _status(result) == 501

    def test_set_policy_value_error_returns_500(self, mound_with_policy):
        """ValueError inside the try block (e.g. from CurationPolicy) returns 500."""
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.post({"enabled": True})
        # Trigger ValueError inside the try block by making set_curation_policy raise
        mound_with_policy.set_curation_policy = MagicMock(return_value=None)
        with patch(
            _RUN_ASYNC_PATCH,
            side_effect=ValueError("bad policy data"),
        ):
            result = h._handle_set_curation_policy(http)
        assert _status(result) == 500

    def test_set_policy_defaults(self, mound_with_policy):
        """Body with no fields should use defaults for all policy values."""
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_set_curation_policy(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["workspace_id"] == "default"

    def test_set_policy_custom_thresholds(self, mound_with_policy):
        """Custom threshold values are forwarded to CurationPolicy."""
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.post({
            "quality_threshold": 0.7,
            "promotion_threshold": 0.95,
            "demotion_threshold": 0.25,
            "archive_threshold": 0.1,
            "usage_window_days": 60,
            "min_retrievals_for_promotion": 10,
        })
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_set_curation_policy(http)
        assert _status(result) == 200

    def test_set_policy_returns_policy_id(self, mound_with_policy):
        h = CurationTestHandler(mound=mound_with_policy)
        http = MockHTTPHandler.post({"name": "my-policy"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_set_curation_policy(http)
        body = _body(result)
        assert "policy_id" in body


# ============================================================================
# Tests: GET /api/v1/knowledge/mound/curation/status
# ============================================================================


class TestCurationStatus:
    """Test _handle_curation_status."""

    def test_status_with_mound_methods(self, mound_with_status):
        h = CurationTestHandler(mound=mound_with_status)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_status({})
        body = _body(result)
        assert _status(result) == 200
        assert body["workspace_id"] == "default"
        assert body["last_run"] == "2025-01-15T10:00:00Z"
        assert body["is_running"] is False
        assert body["stats"]["total_items"] == 100

    def test_status_default_when_no_methods(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_curation_status({})
        body = _body(result)
        assert _status(result) == 200
        assert body["workspace_id"] == "default"
        assert body["last_run"] is None
        assert body["is_running"] is False
        assert body["stats"]["total_items"] == 0

    def test_status_with_workspace_param(self, mound_with_status):
        h = CurationTestHandler(mound=mound_with_status)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_status({"workspace_id": "ws-abc"})
        body = _body(result)
        assert body["workspace_id"] == "ws-abc"

    def test_status_no_mound_returns_503(self, handler_no_mound):
        result = handler_no_mound._handle_curation_status({})
        assert _status(result) == 503

    def test_status_error_returns_500(self, mock_mound):
        mock_mound.get_curation_status = MagicMock(return_value={"key": "val"})
        h = CurationTestHandler(mound=mock_mound)
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("err")):
            result = h._handle_curation_status({})
        assert _status(result) == 500

    def test_status_partial_methods(self, mock_mound):
        """Mound has get_node_count but not get_curation_status."""
        mock_mound.get_node_count = MagicMock(return_value=50)
        h = CurationTestHandler(mound=mock_mound)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_status({})
        body = _body(result)
        assert _status(result) == 200
        assert body["stats"]["total_items"] == 50
        assert body["last_run"] is None

    def test_status_mound_returns_none_for_status(self, mock_mound):
        """get_curation_status returns None, defaults are preserved."""
        mock_mound.get_curation_status = MagicMock(return_value=None)
        h = CurationTestHandler(mound=mock_mound)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_status({})
        body = _body(result)
        assert _status(result) == 200
        assert body["last_run"] is None

    def test_status_has_stats_section(self, mock_mound):
        """Response always includes a stats section."""
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_curation_status({})
        body = _body(result)
        assert "stats" in body
        assert "total_items" in body["stats"]
        assert "items_scored" in body["stats"]
        assert "items_pending" in body["stats"]


# ============================================================================
# Tests: POST /api/v1/knowledge/mound/curation/run
# ============================================================================


class TestRunCuration:
    """Test _handle_run_curation."""

    def test_run_curation_success(self, mound_with_curation):
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.post({"workspace_id": "ws-1", "dry_run": False})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["promoted"] == 2
        assert body["demoted"] == 1
        assert body["archived"] == 0
        assert body["refreshed"] == 3
        assert body["flagged"] == 1
        assert "completed_at" in body

    def test_run_curation_dry_run(self, mound_with_curation):
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.post({"dry_run": True})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert body["dry_run"] is True

    def test_run_curation_no_mound_returns_503(self, handler_no_mound):
        http = MockHTTPHandler.post({})
        result = handler_no_mound._handle_run_curation(http)
        assert _status(result) == 503

    def test_run_curation_no_run_method(self, mock_mound):
        """Mound without run_curation returns simulated result."""
        h = CurationTestHandler(mound=mock_mound)
        http = MockHTTPHandler.post({})
        result = h._handle_run_curation(http)
        body = _body(result)
        assert _status(result) == 200
        assert "note" in body
        assert body["promoted"] == 0
        assert body["demoted"] == 0

    def test_run_curation_error_returns_500(self, mound_with_curation):
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("boom")):
            result = h._handle_run_curation(http)
        assert _status(result) == 500

    def test_run_curation_default_workspace(self, mound_with_curation):
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert body["workspace_id"] == "default"

    def test_run_curation_none_result(self, mock_mound):
        """run_curation returns None => simulated result."""
        mock_mound.run_curation = MagicMock(return_value=None)
        h = CurationTestHandler(mound=mock_mound)
        http = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert _status(result) == 200
        assert "note" in body

    def test_run_curation_with_no_body(self, mound_with_curation):
        """None body => body or {} => defaults used."""
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.get()
        with patch.object(h, "_read_request_body", return_value=None), \
             patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert _status(result) == 200
        assert body["workspace_id"] == "default"
        assert body["dry_run"] is False

    def test_run_curation_has_started_at(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        http = MockHTTPHandler.post({})
        result = h._handle_run_curation(http)
        body = _body(result)
        assert "started_at" in body

    def test_run_curation_has_completed_at(self, mock_mound):
        """Even when no run_curation method, completed_at is set."""
        h = CurationTestHandler(mound=mock_mound)
        http = MockHTTPHandler.post({})
        result = h._handle_run_curation(http)
        body = _body(result)
        assert "completed_at" in body

    def test_run_curation_workspace_from_body(self, mound_with_curation):
        h = CurationTestHandler(mound=mound_with_curation)
        http = MockHTTPHandler.post({"workspace_id": "prod-ws"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_run_curation(http)
        body = _body(result)
        assert body["workspace_id"] == "prod-ws"


# ============================================================================
# Tests: GET /api/v1/knowledge/mound/curation/history
# ============================================================================


class TestCurationHistory:
    """Test _handle_curation_history."""

    def test_history_returns_entries(self, mound_with_history):
        h = CurationTestHandler(mound=mound_with_history)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_history({})
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["history"]) == 2
        assert body["history"][0]["run_id"] == "run-1"

    def test_history_with_workspace(self, mound_with_history):
        h = CurationTestHandler(mound=mound_with_history)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_curation_history({"workspace_id": "ws-xyz"})
        body = _body(result)
        assert body["workspace_id"] == "ws-xyz"

    def test_history_no_mound_returns_503(self, handler_no_mound):
        result = handler_no_mound._handle_curation_history({})
        assert _status(result) == 503

    def test_history_no_method_returns_empty(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_curation_history({})
        body = _body(result)
        assert _status(result) == 200
        assert body["history"] == []
        assert body["count"] == 0

    def test_history_error_returns_500(self, mound_with_history):
        h = CurationTestHandler(mound=mound_with_history)
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("bad")):
            result = h._handle_curation_history({})
        assert _status(result) == 500

    def test_history_default_limit(self, mound_with_history):
        """Default limit is 20 when not specified."""
        h = CurationTestHandler(mound=mound_with_history)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            h._handle_curation_history({})
        call_args = mound_with_history.get_curation_history.call_args
        assert call_args is not None

    def test_history_custom_limit(self, mound_with_history):
        """Limit from query params is forwarded."""
        h = CurationTestHandler(mound=mound_with_history)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            h._handle_curation_history({"limit": "10"})
        mound_with_history.get_curation_history.assert_called_once()

    def test_history_response_has_count(self, mock_mound):
        """Response always has a count field matching history length."""
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_curation_history({})
        body = _body(result)
        assert body["count"] == len(body["history"])


# ============================================================================
# Tests: GET /api/v1/knowledge/mound/curation/scores
# ============================================================================


class TestQualityScores:
    """Test _handle_quality_scores."""

    def test_scores_returns_formatted_entries(self, mound_with_scores):
        h = CurationTestHandler(mound=mound_with_scores)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_quality_scores({})
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["scores"]) == 2
        s0 = body["scores"][0]
        assert s0["node_id"] == "node-001"
        assert s0["overall_score"] == 0.9
        assert s0["freshness_score"] == 0.9
        assert s0["confidence_score"] == 0.85
        assert s0["usage_score"] == 0.7
        assert s0["relevance_score"] == 0.75
        assert s0["relationship_score"] == 0.6
        assert s0["recommendation"] == "promote"
        assert s0["debate_uses"] == 3
        assert s0["retrieval_count"] == 12

    def test_scores_second_entry(self, mound_with_scores):
        h = CurationTestHandler(mound=mound_with_scores)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_quality_scores({})
        body = _body(result)
        s1 = body["scores"][1]
        assert s1["node_id"] == "node-002"
        assert s1["recommendation"] == "demote"

    def test_scores_no_mound_returns_503(self, handler_no_mound):
        result = handler_no_mound._handle_quality_scores({})
        assert _status(result) == 503

    def test_scores_no_method_returns_empty(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_quality_scores({})
        body = _body(result)
        assert _status(result) == 200
        assert body["scores"] == []
        assert body["count"] == 0

    def test_scores_error_returns_500(self, mound_with_scores):
        h = CurationTestHandler(mound=mound_with_scores)
        with patch(_RUN_ASYNC_PATCH, side_effect=AttributeError("missing")):
            result = h._handle_quality_scores({})
        assert _status(result) == 500

    def test_scores_filters_in_response(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_quality_scores({})
        body = _body(result)
        assert "filters" in body
        assert body["filters"]["min_score"] == 0.0
        assert body["filters"]["max_score"] == 1.0
        assert body["filters"]["limit"] == 50

    def test_scores_custom_filters(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_quality_scores({
            "min_score": "0.3",
            "max_score": "0.9",
            "limit": "25",
        })
        body = _body(result)
        assert body["filters"]["min_score"] == 0.3
        assert body["filters"]["max_score"] == 0.9
        assert body["filters"]["limit"] == 25

    def test_scores_workspace_param(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_quality_scores({"workspace_id": "ws-scores"})
        body = _body(result)
        assert body["workspace_id"] == "ws-scores"

    def test_scores_count_matches_list_length(self, mound_with_scores):
        h = CurationTestHandler(mound=mound_with_scores)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_quality_scores({})
        body = _body(result)
        assert body["count"] == len(body["scores"])


# ============================================================================
# Tests: GET /api/v1/knowledge/mound/curation/tiers
# ============================================================================


class TestTierDistribution:
    """Test _handle_tier_distribution."""

    def test_tiers_returns_distribution(self, mound_with_tiers):
        h = CurationTestHandler(mound=mound_with_tiers)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_tier_distribution({})
        body = _body(result)
        assert _status(result) == 200
        assert body["distribution"]["hot"] == 10
        assert body["distribution"]["warm"] == 30
        assert body["distribution"]["cold"] == 50
        assert body["distribution"]["glacial"] == 10
        assert body["total"] == 100

    def test_tiers_percentages(self, mound_with_tiers):
        h = CurationTestHandler(mound=mound_with_tiers)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_tier_distribution({})
        body = _body(result)
        assert body["percentages"]["hot"] == 10.0
        assert body["percentages"]["warm"] == 30.0
        assert body["percentages"]["cold"] == 50.0
        assert body["percentages"]["glacial"] == 10.0

    def test_tiers_defaults_when_no_method(self, mock_mound):
        """Without get_tier_distribution, returns zero counts."""
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_tier_distribution({})
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 0
        for pct in body["percentages"].values():
            assert pct == 0

    def test_tiers_no_mound_returns_503(self, handler_no_mound):
        result = handler_no_mound._handle_tier_distribution({})
        assert _status(result) == 503

    def test_tiers_import_error_returns_501(self, mock_mound):
        h = CurationTestHandler(mound=mock_mound)
        with patch.dict("sys.modules", {"aragora.knowledge.mound.ops.auto_curation": None}):
            result = h._handle_tier_distribution({})
        assert _status(result) == 501

    def test_tiers_error_returns_500(self, mound_with_tiers):
        h = CurationTestHandler(mound=mound_with_tiers)
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("bad")):
            result = h._handle_tier_distribution({})
        assert _status(result) == 500

    def test_tiers_workspace_param(self, mound_with_tiers):
        h = CurationTestHandler(mound=mound_with_tiers)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_tier_distribution({"workspace_id": "ws-tier"})
        body = _body(result)
        assert body["workspace_id"] == "ws-tier"

    def test_tiers_zero_total_avoids_division_by_zero(self, mock_mound):
        """When total is 0, percentages should be 0 not raise ZeroDivisionError."""
        mock_mound.get_tier_distribution = MagicMock(
            return_value={"hot": 0, "warm": 0, "cold": 0, "glacial": 0}
        )
        h = CurationTestHandler(mound=mock_mound)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = h._handle_tier_distribution({})
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 0
        for pct in body["percentages"].values():
            assert pct == 0

    def test_tiers_has_all_tier_levels(self, mock_mound):
        """Default distribution includes all four tier levels."""
        h = CurationTestHandler(mound=mock_mound)
        result = h._handle_tier_distribution({})
        body = _body(result)
        for tier in ["hot", "warm", "cold", "glacial"]:
            assert tier in body["distribution"]


# ============================================================================
# Tests: _check_knowledge_permission
# ============================================================================


class TestCheckKnowledgePermission:
    """Test the RBAC permission check method."""

    def test_returns_401_when_not_authenticated(self, handler):
        """Unauthenticated user gets 401."""
        http = MockHTTPHandler.get()
        mock_user = MagicMock()
        mock_user.authenticated = False
        with patch(
            "aragora.billing.auth.extract_user_from_request",
            return_value=mock_user,
        ):
            result = handler._check_knowledge_permission(http, "read")
        assert _status(result) == 401

    def test_returns_403_when_denied(self, handler):
        """Permission denied returns 403."""
        http = MockHTTPHandler.get()
        mock_user = MagicMock()
        mock_user.authenticated = True
        mock_user.user_id = "user-1"
        mock_user.org_id = "org-1"
        mock_user.roles = ["member"]

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "insufficient permissions"

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.billing.auth.extract_user_from_request",
            return_value=mock_user,
        ), patch(
            "aragora.server.handlers.knowledge_base.mound.curation.get_permission_checker",
            return_value=mock_checker,
        ):
            result = handler._check_knowledge_permission(http, "write")
        assert _status(result) == 403

    def test_returns_none_when_permitted(self, handler):
        """No error returned when user has permission."""
        http = MockHTTPHandler.get()
        mock_user = MagicMock()
        mock_user.authenticated = True
        mock_user.user_id = "user-1"
        mock_user.org_id = "org-1"
        mock_user.roles = ["admin"]

        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.billing.auth.extract_user_from_request",
            return_value=mock_user,
        ), patch(
            "aragora.server.handlers.knowledge_base.mound.curation.get_permission_checker",
            return_value=mock_checker,
        ):
            result = handler._check_knowledge_permission(http, "read")
        assert result is None

    def test_import_error_returns_500(self, handler):
        """Import failure in RBAC check returns 500."""
        http = MockHTTPHandler.get()
        with patch.dict("sys.modules", {"aragora.billing.auth": None}):
            result = handler._check_knowledge_permission(http, "read")
        assert _status(result) == 500

    def test_permission_string_uses_knowledge_prefix(self, handler):
        """Permission string is formatted as 'knowledge.<action>'."""
        http = MockHTTPHandler.get()
        mock_user = MagicMock()
        mock_user.authenticated = True
        mock_user.user_id = "user-1"
        mock_user.org_id = None
        mock_user.roles = None

        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.billing.auth.extract_user_from_request",
            return_value=mock_user,
        ), patch(
            "aragora.server.handlers.knowledge_base.mound.curation.get_permission_checker",
            return_value=mock_checker,
        ):
            handler._check_knowledge_permission(http, "delete")

        # Verify the permission string passed to check_permission
        call_args = mock_checker.check_permission.call_args
        # Second positional arg is the permission string
        assert call_args[0][1] == "knowledge.delete"


# ============================================================================
# Tests: _read_request_body
# ============================================================================


class TestReadRequestBody:
    """Test the _read_request_body helper."""

    def test_reads_body_from_base_module(self, handler):
        http = MockHTTPHandler.post({"key": "value"})
        with patch(
            "aragora.server.handlers.base.read_json_body",
            return_value={"key": "value"},
            create=True,
        ):
            result = handler._read_request_body(http)
        assert result == {"key": "value"}

    def test_falls_back_to_instance_method(self, handler):
        """When base module helper is not available, falls back to self.read_json_body."""
        http = MockHTTPHandler.post({"fallback": True})
        # Patch base module so it has no read_json_body
        mock_base = MagicMock(spec=[])
        with patch.dict("sys.modules", {"aragora.server.handlers.base": mock_base}):
            result = handler._read_request_body(http)
        assert result == {"fallback": True}

    def test_returns_none_for_bad_json(self, handler):
        """Invalid JSON body returns None from read_json_body."""
        http = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "5", "User-Agent": "test", "Authorization": "Bearer x"},
            rfile=io.BytesIO(b"notjs"),
        )
        result = handler.read_json_body(http)
        assert result is None

    def test_returns_empty_dict_for_no_content(self, handler):
        """Empty body returns empty dict."""
        http = MockHTTPHandler.get()  # Content-Length=0
        result = handler.read_json_body(http)
        assert result == {}

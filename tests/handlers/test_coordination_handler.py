"""Tests for the CoordinationHandler (cross-workspace coordination REST handler).

Covers all 12 endpoints:
- POST   /api/v1/coordination/workspaces       - register workspace
- GET    /api/v1/coordination/workspaces        - list registered workspaces
- DELETE /api/v1/coordination/workspaces/{id}   - unregister workspace
- POST   /api/v1/coordination/federation        - create federation policy
- GET    /api/v1/coordination/federation        - list federation policies
- POST   /api/v1/coordination/execute           - cross-workspace execution
- GET    /api/v1/coordination/executions        - list executions
- POST   /api/v1/coordination/consent           - grant consent
- DELETE /api/v1/coordination/consent/{id}      - revoke consent
- GET    /api/v1/coordination/consent           - list consents
- POST   /api/v1/coordination/approve/{id}      - approve pending execution
- GET    /api/v1/coordination/stats             - coordination stats
- GET    /api/v1/coordination/health            - health check

Test structure:
  TestCanHandle               - Route matching
  TestRegisterWorkspace       - POST /workspaces (happy + errors)
  TestListWorkspaces          - GET /workspaces
  TestUnregisterWorkspace     - DELETE /workspaces/{id}
  TestCreatePolicy            - POST /federation (happy + errors)
  TestListPolicies            - GET /federation
  TestExecute                 - POST /execute (happy + errors)
  TestListExecutions          - GET /executions
  TestGrantConsent            - POST /consent (happy + errors)
  TestRevokeConsent           - DELETE /consent/{id}
  TestListConsents            - GET /consent
  TestApprove                 - POST /approve/{id}
  TestStats                   - GET /stats
  TestHealth                  - GET /health
  TestCoordinationUnavailable - 501 when module not available
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict[str, Any]:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    raw = result.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    return result.status_code


def _make_http_handler(
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with JSON body support."""
    handler = MagicMock()
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
    else:
        raw = b""
    handler.headers = headers or {"Content-Length": str(len(raw))}
    if "Content-Length" not in handler.headers:
        handler.headers["Content-Length"] = str(len(raw))
    handler.rfile = BytesIO(raw)
    return handler


# ---------------------------------------------------------------------------
# Mock coordinator and dataclasses
# ---------------------------------------------------------------------------


class _MockWorkspace:
    """Lightweight stand-in for FederatedWorkspace.to_dict()."""

    def __init__(self, ws_id: str = "ws-1", name: str = "Test WS"):
        self.id = ws_id
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name}


class _MockPolicy:
    """Lightweight stand-in for FederationPolicy.to_dict()."""

    def __init__(self, name: str = "default"):
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


class _MockConsent:
    """Lightweight stand-in for DataSharingConsent.to_dict()."""

    def __init__(self, consent_id: str = "consent-1"):
        self.id = consent_id

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id}


class _MockRequest:
    """Lightweight stand-in for CrossWorkspaceRequest.to_dict()."""

    def __init__(self, request_id: str = "req-1"):
        self.id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id}


def _make_coordinator(**overrides: Any) -> MagicMock:
    """Create a mock CrossWorkspaceCoordinator."""
    coord = MagicMock()
    coord.register_workspace = MagicMock()
    coord.unregister_workspace = MagicMock()
    coord.list_workspaces = MagicMock(return_value=[])
    coord.set_policy = MagicMock()
    coord.list_consents = MagicMock(return_value=[])
    coord.list_pending_requests = MagicMock(return_value=[])
    coord.grant_consent = MagicMock(return_value=_MockConsent())
    coord.revoke_consent = MagicMock(return_value=True)
    coord.approve_request = MagicMock(return_value=True)
    coord.get_stats = MagicMock(
        return_value={
            "total_workspaces": 2,
            "total_consents": 1,
            "valid_consents": 1,
            "pending_requests": 0,
            "registered_handlers": [],
        }
    )

    # Default policy for list_policies
    default_pol = _MockPolicy("default")
    coord._default_policy = default_pol
    coord._workspace_policies = {}
    coord._pair_policies = {}

    for k, v in overrides.items():
        setattr(coord, k, v)
    return coord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator():
    """Create a mock coordinator."""
    return _make_coordinator()


@pytest.fixture
def handler(coordinator):
    """Create CoordinationHandler with coordination module mocked as available."""
    with (
        patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
        patch("aragora.server.handlers.coordination.get_coordinator", return_value=coordinator),
    ):
        from aragora.server.handlers.coordination import CoordinationHandler

        h = CoordinationHandler(ctx={})
        h._coordinator = coordinator
        return h


@pytest.fixture
def handler_no_coord():
    """Create CoordinationHandler with coordination module unavailable."""
    with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
        from aragora.server.handlers.coordination import CoordinationHandler

        h = CoordinationHandler(ctx={})
        h._coordinator = None
        return h


# ===================================================================
# Route matching
# ===================================================================


@pytest.mark.no_auto_auth
class TestCanHandle:
    """Test can_handle route matching."""

    def test_matches_versioned_prefix(self, handler):
        assert handler.can_handle("/api/v1/coordination/workspaces") is True

    def test_matches_unversioned_prefix(self, handler):
        assert handler.can_handle("/api/coordination/workspaces") is True

    def test_matches_with_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/coordination/workspaces/") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/coord") is False


# ===================================================================
# Workspace endpoints
# ===================================================================


@pytest.mark.no_auto_auth
class TestRegisterWorkspace:
    """POST /api/v1/coordination/workspaces -- register workspace."""

    def test_register_success(self, handler, coordinator):
        """Happy path: register a workspace with valid body."""
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch("aragora.server.handlers.coordination.FederatedWorkspace") as MockWS,
            patch("aragora.server.handlers.coordination.FederationMode") as MockFM,
        ):
            mock_ws_instance = MagicMock()
            mock_ws_instance.to_dict.return_value = {"id": "ws-1", "name": "Alpha"}
            MockWS.return_value = mock_ws_instance
            MockFM.return_value = "readonly"
            # FederationMode is iterable for error message
            MockFM.__iter__ = MagicMock(return_value=iter([]))

            mock_handler = _make_http_handler(
                body={
                    "id": "ws-1",
                    "name": "Alpha",
                    "org_id": "org-1",
                    "federation_mode": "readonly",
                }
            )
            result = handler.handle_post("/api/v1/coordination/workspaces", {}, mock_handler)

            assert _status(result) == 201
            body = _body(result)
            assert body["registered"] is True

    def test_register_missing_id(self, handler):
        """Error: missing workspace id."""
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", True):
            mock_handler = _make_http_handler(body={"name": "Alpha"})
            result = handler.handle_post("/api/v1/coordination/workspaces", {}, mock_handler)
            assert _status(result) == 400
            assert "id" in _body(result).get("error", "").lower()

    def test_register_invalid_json(self, handler):
        """Error: invalid JSON body."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "5"}
        mock_handler.rfile = BytesIO(b"notjs")
        result = handler.handle_post("/api/v1/coordination/workspaces", {}, mock_handler)
        assert _status(result) == 400
        assert "json" in _body(result).get("error", "").lower()

    def test_register_invalid_federation_mode(self, handler, coordinator):
        """Error: invalid federation_mode value."""
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch("aragora.server.handlers.coordination._parse_federation_mode", return_value=None),
            patch("aragora.server.handlers.coordination.FederationMode") as MockFM,
        ):
            MockFM.__iter__ = MagicMock(return_value=iter([MagicMock(value="readonly")]))
            mock_handler = _make_http_handler(
                body={
                    "id": "ws-1",
                    "federation_mode": "invalid_mode",
                }
            )
            result = handler.handle_post("/api/v1/coordination/workspaces", {}, mock_handler)
            assert _status(result) == 400
            assert "federation_mode" in _body(result).get("error", "").lower()


@pytest.mark.no_auto_auth
class TestListWorkspaces:
    """GET /api/v1/coordination/workspaces -- list workspaces."""

    def test_list_empty(self, handler, coordinator):
        coordinator.list_workspaces.return_value = []
        result = handler.handle("/api/v1/coordination/workspaces", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_list_with_workspaces(self, handler, coordinator):
        coordinator.list_workspaces.return_value = [
            _MockWorkspace("ws-1", "Alpha"),
            _MockWorkspace("ws-2", "Beta"),
        ]
        result = handler.handle("/api/v1/coordination/workspaces", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert body["workspaces"][0]["id"] == "ws-1"


@pytest.mark.no_auto_auth
class TestUnregisterWorkspace:
    """DELETE /api/v1/coordination/workspaces/{id} -- unregister workspace."""

    def test_unregister_success(self, handler, coordinator):
        result = handler.handle_delete("/api/v1/coordination/workspaces/ws-1", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["unregistered"] is True
        assert body["workspace_id"] == "ws-1"
        coordinator.unregister_workspace.assert_called_once_with("ws-1")

    def test_unregister_no_id(self, handler):
        """DELETE /api/v1/coordination/workspaces/ -- missing ID returns None (no match)."""
        result = handler.handle_delete("/api/v1/coordination/workspaces", {}, None)
        # Path doesn't match the delete pattern since no /{id} segment
        assert result is None


# ===================================================================
# Federation policy endpoints
# ===================================================================


@pytest.mark.no_auto_auth
class TestCreatePolicy:
    """POST /api/v1/coordination/federation -- create federation policy."""

    def test_create_policy_success(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch(
                "aragora.server.handlers.coordination._parse_federation_mode",
                return_value="isolated",
            ),
            patch("aragora.server.handlers.coordination._parse_sharing_scope", return_value="none"),
            patch("aragora.server.handlers.coordination.FederationPolicy") as MockFP,
        ):
            mock_policy = MagicMock()
            mock_policy.to_dict.return_value = {"name": "strict-policy"}
            MockFP.return_value = mock_policy

            mock_handler = _make_http_handler(
                body={
                    "name": "strict-policy",
                    "mode": "isolated",
                    "sharing_scope": "none",
                }
            )
            result = handler.handle_post("/api/v1/coordination/federation", {}, mock_handler)
            assert _status(result) == 201
            body = _body(result)
            assert body["created"] is True

    def test_create_policy_missing_name(self, handler, coordinator):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", True):
            mock_handler = _make_http_handler(body={"mode": "isolated"})
            result = handler.handle_post("/api/v1/coordination/federation", {}, mock_handler)
            assert _status(result) == 400
            assert "name" in _body(result).get("error", "").lower()

    def test_create_policy_invalid_mode(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch("aragora.server.handlers.coordination._parse_federation_mode", return_value=None),
        ):
            mock_handler = _make_http_handler(
                body={
                    "name": "bad-mode-policy",
                    "mode": "wrong",
                }
            )
            result = handler.handle_post("/api/v1/coordination/federation", {}, mock_handler)
            assert _status(result) == 400
            assert "mode" in _body(result).get("error", "").lower()

    def test_create_policy_invalid_sharing_scope(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch(
                "aragora.server.handlers.coordination._parse_federation_mode",
                return_value="isolated",
            ),
            patch("aragora.server.handlers.coordination._parse_sharing_scope", return_value=None),
        ):
            mock_handler = _make_http_handler(
                body={
                    "name": "bad-scope-policy",
                    "mode": "isolated",
                    "sharing_scope": "wrong",
                }
            )
            result = handler.handle_post("/api/v1/coordination/federation", {}, mock_handler)
            assert _status(result) == 400
            assert "sharing_scope" in _body(result).get("error", "").lower()

    def test_create_policy_invalid_json(self, handler):
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "3"}
        mock_handler.rfile = BytesIO(b"bad")
        result = handler.handle_post("/api/v1/coordination/federation", {}, mock_handler)
        assert _status(result) == 400


@pytest.mark.no_auto_auth
class TestListPolicies:
    """GET /api/v1/coordination/federation -- list federation policies."""

    def test_list_policies_default_only(self, handler, coordinator):
        result = handler.handle("/api/v1/coordination/federation", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1
        # At minimum the default policy is listed
        assert any(p.get("scope") == "default" for p in body["policies"])

    def test_list_policies_with_workspace_and_pair(self, handler, coordinator):
        ws_pol = _MockPolicy("ws-pol")
        pair_pol = _MockPolicy("pair-pol")
        coordinator._workspace_policies = {"ws-1": ws_pol}
        coordinator._pair_policies = {("ws-1", "ws-2"): pair_pol}

        result = handler.handle("/api/v1/coordination/federation", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3  # default + workspace + pair
        scopes = [p["scope"] for p in body["policies"]]
        assert "workspace" in scopes
        assert "pair" in scopes


# ===================================================================
# Execution endpoints
# ===================================================================


@pytest.mark.no_auto_auth
class TestExecute:
    """POST /api/v1/coordination/execute -- cross-workspace execution."""

    def test_execute_missing_operation(self, handler, coordinator):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", True):
            mock_handler = _make_http_handler(
                body={
                    "source_workspace_id": "ws-1",
                    "target_workspace_id": "ws-2",
                }
            )
            result = handler.handle_post("/api/v1/coordination/execute", {}, mock_handler)
            assert _status(result) == 400
            assert "operation" in _body(result).get("error", "").lower()

    def test_execute_invalid_operation(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch("aragora.server.handlers.coordination._parse_operation_type", return_value=None),
        ):
            mock_handler = _make_http_handler(
                body={
                    "operation": "invalid_op",
                    "source_workspace_id": "ws-1",
                    "target_workspace_id": "ws-2",
                }
            )
            result = handler.handle_post("/api/v1/coordination/execute", {}, mock_handler)
            assert _status(result) == 400
            assert "operation" in _body(result).get("error", "").lower()

    def test_execute_missing_workspace_ids(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch(
                "aragora.server.handlers.coordination._parse_operation_type",
                return_value="read_knowledge",
            ),
        ):
            mock_handler = _make_http_handler(
                body={
                    "operation": "read_knowledge",
                }
            )
            result = handler.handle_post("/api/v1/coordination/execute", {}, mock_handler)
            assert _status(result) == 400
            assert "workspace" in _body(result).get("error", "").lower()

    def test_execute_invalid_json(self, handler):
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "3"}
        mock_handler.rfile = BytesIO(b"bad")
        result = handler.handle_post("/api/v1/coordination/execute", {}, mock_handler)
        assert _status(result) == 400


@pytest.mark.no_auto_auth
class TestListExecutions:
    """GET /api/v1/coordination/executions -- list pending executions."""

    def test_list_empty(self, handler, coordinator):
        coordinator.list_pending_requests.return_value = []
        result = handler.handle("/api/v1/coordination/executions", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["executions"] == []
        assert body["total"] == 0

    def test_list_with_filter(self, handler, coordinator):
        coordinator.list_pending_requests.return_value = [_MockRequest("req-1")]
        result = handler.handle(
            "/api/v1/coordination/executions",
            {"workspace_id": "ws-1"},
            None,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        coordinator.list_pending_requests.assert_called_once_with("ws-1")


# ===================================================================
# Consent endpoints
# ===================================================================


@pytest.mark.no_auto_auth
class TestGrantConsent:
    """POST /api/v1/coordination/consent -- grant consent."""

    def test_grant_success(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch(
                "aragora.server.handlers.coordination._parse_sharing_scope", return_value="metadata"
            ),
        ):
            mock_handler = _make_http_handler(
                body={
                    "source_workspace_id": "ws-1",
                    "target_workspace_id": "ws-2",
                    "scope": "metadata",
                    "granted_by": "admin",
                }
            )
            result = handler.handle_post("/api/v1/coordination/consent", {}, mock_handler)
            assert _status(result) == 201
            body = _body(result)
            assert body["granted"] is True

    def test_grant_missing_workspace_ids(self, handler, coordinator):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", True):
            mock_handler = _make_http_handler(body={"scope": "metadata"})
            result = handler.handle_post("/api/v1/coordination/consent", {}, mock_handler)
            assert _status(result) == 400
            assert "workspace" in _body(result).get("error", "").lower()

    def test_grant_invalid_scope(self, handler, coordinator):
        with (
            patch("aragora.server.handlers.coordination._HAS_COORDINATION", True),
            patch("aragora.server.handlers.coordination._parse_sharing_scope", return_value=None),
        ):
            mock_handler = _make_http_handler(
                body={
                    "source_workspace_id": "ws-1",
                    "target_workspace_id": "ws-2",
                    "scope": "invalid",
                }
            )
            result = handler.handle_post("/api/v1/coordination/consent", {}, mock_handler)
            assert _status(result) == 400
            assert "scope" in _body(result).get("error", "").lower()

    def test_grant_invalid_json(self, handler):
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "3"}
        mock_handler.rfile = BytesIO(b"bad")
        result = handler.handle_post("/api/v1/coordination/consent", {}, mock_handler)
        assert _status(result) == 400


@pytest.mark.no_auto_auth
class TestRevokeConsent:
    """DELETE /api/v1/coordination/consent/{id} -- revoke consent."""

    def test_revoke_success(self, handler, coordinator):
        coordinator.revoke_consent.return_value = True
        result = handler.handle_delete("/api/v1/coordination/consent/consent-42", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["revoked"] is True
        assert body["consent_id"] == "consent-42"
        coordinator.revoke_consent.assert_called_once_with("consent-42", revoked_by="api")

    def test_revoke_not_found(self, handler, coordinator):
        coordinator.revoke_consent.return_value = False
        result = handler.handle_delete("/api/v1/coordination/consent/missing-id", {}, None)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()


@pytest.mark.no_auto_auth
class TestListConsents:
    """GET /api/v1/coordination/consent -- list consents."""

    def test_list_all_consents(self, handler, coordinator):
        coordinator.list_consents.return_value = [
            _MockConsent("c-1"),
            _MockConsent("c-2"),
        ]
        result = handler.handle("/api/v1/coordination/consent", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    def test_list_consents_with_filter(self, handler, coordinator):
        coordinator.list_consents.return_value = [_MockConsent("c-1")]
        result = handler.handle(
            "/api/v1/coordination/consent",
            {"workspace_id": "ws-1"},
            None,
        )
        assert _status(result) == 200
        coordinator.list_consents.assert_called_once_with("ws-1")


# ===================================================================
# Approval endpoint
# ===================================================================


@pytest.mark.no_auto_auth
class TestApprove:
    """POST /api/v1/coordination/approve/{id} -- approve pending execution."""

    def test_approve_success(self, handler, coordinator):
        coordinator.approve_request.return_value = True
        mock_handler = _make_http_handler(body={"approved_by": "admin-user"})
        result = handler.handle_post("/api/v1/coordination/approve/req-42", {}, mock_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["approved"] is True
        assert body["request_id"] == "req-42"
        coordinator.approve_request.assert_called_once_with("req-42", "admin-user")

    def test_approve_not_found(self, handler, coordinator):
        coordinator.approve_request.return_value = False
        mock_handler = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/coordination/approve/missing-req", {}, mock_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_approve_default_approved_by(self, handler, coordinator):
        """When no approved_by in body, defaults to 'api'."""
        coordinator.approve_request.return_value = True
        mock_handler = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/coordination/approve/req-55", {}, mock_handler)
        assert _status(result) == 200
        coordinator.approve_request.assert_called_once_with("req-55", "api")

    def test_approve_with_empty_body(self, handler, coordinator):
        """Approve endpoint accepts empty body (body defaults to {})."""
        coordinator.approve_request.return_value = True
        # Handler with no Content-Length / empty body
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "0"}
        mock_handler.rfile = BytesIO(b"")
        result = handler.handle_post("/api/v1/coordination/approve/req-77", {}, mock_handler)
        assert _status(result) == 200


# ===================================================================
# Stats endpoint
# ===================================================================


@pytest.mark.no_auto_auth
class TestStats:
    """GET /api/v1/coordination/stats -- coordination statistics."""

    def test_stats_happy_path(self, handler, coordinator):
        result = handler.handle("/api/v1/coordination/stats", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert "total_workspaces" in body
        assert body["total_workspaces"] == 2


# ===================================================================
# Health endpoint
# ===================================================================


@pytest.mark.no_auto_auth
class TestHealth:
    """GET /api/v1/coordination/health -- health check."""

    def test_health_healthy(self, handler, coordinator):
        result = handler.handle("/api/v1/coordination/health", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert "total_workspaces" in body

    def test_health_idle_when_no_workspaces(self, handler, coordinator):
        coordinator.get_stats.return_value = {
            "total_workspaces": 0,
            "total_consents": 0,
            "valid_consents": 0,
            "pending_requests": 0,
            "registered_handlers": [],
        }
        result = handler.handle("/api/v1/coordination/health", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "idle"

    def test_health_degraded_when_coordinator_none(self, handler):
        handler._coordinator = None
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", True):
            result = handler._handle_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"

    def test_health_unavailable_when_module_missing(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_health()
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"


# ===================================================================
# Fleet monitor endpoints
# ===================================================================


@pytest.mark.no_auto_auth
class TestFleetEndpoints:
    def test_get_fleet_status(self, handler):
        mock_fleet = MagicMock()
        mock_fleet.fleet_status.return_value = {
            "generated_at": "2026-02-26T00:00:00+00:00",
            "active_sessions": 1,
            "total_sessions": 1,
            "sessions": [],
            "claims": {"conflict_count": 0},
            "merge_queue": {"total": 0, "blocked": 0, "items": []},
            "actionable_failures": [],
        }
        with patch(
            "aragora.server.handlers.coordination.create_fleet_coordinator",
            return_value=mock_fleet,
        ):
            result = handler.handle("/api/v1/coordination/fleet/status", {"tail": "500"}, None)

        assert _status(result) == 200
        body = _body(result)
        assert body["active_sessions"] == 1
        mock_fleet.fleet_status.assert_called_once_with(tail_lines=500)

    def test_get_fleet_logs(self, handler):
        mock_fleet = MagicMock()
        mock_fleet.fleet_logs.return_value = {
            "generated_at": "2026-02-26T00:00:00+00:00",
            "tail_lines": 200,
            "sessions": [],
        }
        with patch(
            "aragora.server.handlers.coordination.create_fleet_coordinator",
            return_value=mock_fleet,
        ):
            result = handler.handle("/api/v1/coordination/fleet/logs", {}, None)

        assert _status(result) == 200
        body = _body(result)
        assert body["tail_lines"] == 200

    def test_post_fleet_claim_conflict_returns_409(self, handler):
        mock_fleet = MagicMock()
        mock_fleet.claim_paths.return_value = {
            "ok": False,
            "applied": False,
            "owner": "track:qa",
            "conflicts": [{"path": "x.py", "owner": "track:core"}],
        }
        with patch(
            "aragora.server.handlers.coordination.create_fleet_coordinator",
            return_value=mock_fleet,
        ):
            req = _make_http_handler(
                body={
                    "owner": "track:qa",
                    "paths": ["x.py"],
                    "override": False,
                }
            )
            result = handler.handle_post("/api/v1/coordination/fleet/claims", {}, req)

        assert _status(result) == 409
        body = _body(result)
        assert body["ok"] is False

    def test_post_fleet_merge_queue_enqueue(self, handler):
        mock_fleet = MagicMock()
        mock_fleet.enqueue_merge.return_value = {
            "ok": True,
            "enqueued": True,
            "item": {"id": "mq-1"},
        }
        with patch(
            "aragora.server.handlers.coordination.create_fleet_coordinator",
            return_value=mock_fleet,
        ):
            req = _make_http_handler(
                body={
                    "action": "enqueue",
                    "owner": "track:core",
                    "branch": "codex/core-123",
                }
            )
            result = handler.handle_post("/api/v1/coordination/fleet/merge-queue", {}, req)

        assert _status(result) == 201
        body = _body(result)
        assert body["enqueued"] is True


# ===================================================================
# Coordination unavailable (501)
# ===================================================================


@pytest.mark.no_auto_auth
class TestCoordinationUnavailable:
    """All write endpoints return 501 when module is unavailable."""

    def test_list_workspaces_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_list_workspaces({})
        assert _status(result) == 501

    def test_register_workspace_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_register_workspace({"id": "ws-1"})
        assert _status(result) == 501

    def test_unregister_workspace_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_unregister_workspace("ws-1")
        assert _status(result) == 501

    def test_create_policy_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_create_policy({"name": "test"})
        assert _status(result) == 501

    def test_list_policies_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_list_policies({})
        assert _status(result) == 501

    def test_execute_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_execute({"operation": "read_knowledge"})
        assert _status(result) == 501

    def test_list_executions_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_list_executions({})
        assert _status(result) == 501

    def test_grant_consent_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_grant_consent(
                {
                    "source_workspace_id": "ws-1",
                    "target_workspace_id": "ws-2",
                }
            )
        assert _status(result) == 501

    def test_revoke_consent_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_revoke_consent("c-1")
        assert _status(result) == 501

    def test_list_consents_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_list_consents({})
        assert _status(result) == 501

    def test_approve_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_approve("req-1", {})
        assert _status(result) == 501

    def test_stats_unavailable(self, handler_no_coord):
        with patch("aragora.server.handlers.coordination._HAS_COORDINATION", False):
            result = handler_no_coord._handle_stats()
        assert _status(result) == 501


# ===================================================================
# GET dispatch returns None for unmatched paths
# ===================================================================


@pytest.mark.no_auto_auth
class TestDispatchReturnsNone:
    """Handler returns None for paths it cannot route."""

    def test_get_unmatched(self, handler):
        result = handler.handle("/api/v1/coordination/unknown", {}, None)
        assert result is None

    def test_post_unmatched(self, handler):
        mock_handler = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/coordination/unknown", {}, mock_handler)
        assert result is None

    def test_delete_unmatched(self, handler):
        result = handler.handle_delete("/api/v1/coordination/unknown", {}, None)
        assert result is None

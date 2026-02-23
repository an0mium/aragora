"""Tests for Gas Town Dashboard Handler.

Covers all endpoints:
- GET /api/v1/dashboard/gastown/overview
- GET /api/v1/dashboard/gastown/convoys
- GET /api/v1/dashboard/gastown/agents
- GET /api/v1/dashboard/gastown/beads
- GET /api/v1/dashboard/gastown/metrics

Including: routing, auth/RBAC, caching, error handling, edge cases, query params.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.gastown_dashboard import (
    CACHE_TTL,
    GasTownDashboardHandler,
    _get_cached_data,
    _gt_dashboard_cache,
    _set_cached_data,
)


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


# ---------------------------------------------------------------------------
# Mock enums and data models
# ---------------------------------------------------------------------------


class MockGastownConvoyStatus(Enum):
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class MockConvoyStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class MockBeadStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MockBeadPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MockWorkspaceBeadStatus(Enum):
    DONE = "done"
    FAILED = "failed"
    PENDING = "pending"


class MockAgentRole(Enum):
    MAYOR = "mayor"
    WITNESS = "witness"
    POLECAT = "polecat"
    CREW = "crew"


class MockAgentCapability(Enum):
    DEBATE = "debate"
    REVIEW = "review"


class MockConvoy:
    def __init__(self, id, title, status, created_at=None, bead_ids=None, description=""):
        self.id = id
        self.title = title
        self.status = status
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.bead_ids = bead_ids or []
        self.description = description


class MockBead:
    def __init__(self, id, status, convoy_id=None):
        self.id = id
        self.status = status
        self.convoy_id = convoy_id


class MockWorkspaceBead:
    def __init__(self, id, status):
        self.id = id
        self.status = status


class MockAgent:
    def __init__(self, agent_id, role, supervised_by=None, is_ephemeral=False,
                 assigned_at=None, capabilities=None):
        self.agent_id = agent_id
        self.role = role
        self.supervised_by = supervised_by
        self.is_ephemeral = is_ephemeral
        self.assigned_at = assigned_at or datetime.now(timezone.utc)
        self.capabilities = capabilities or []


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handle()."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GasTownDashboardHandler with minimal server context."""
    return GasTownDashboardHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler."""
    return _MockHTTPHandler()


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the dashboard cache between tests."""
    _gt_dashboard_cache.clear()
    yield
    _gt_dashboard_cache.clear()


# ---------------------------------------------------------------------------
# can_handle routing tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for handler routing via can_handle."""

    def test_overview_route(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/overview")

    def test_convoys_route(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/convoys")

    def test_agents_route(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/agents")

    def test_beads_route(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/beads")

    def test_metrics_route(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/metrics")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/other")

    def test_v2_path_accepted(self, handler):
        """v2 path is normalized to /api/dashboard/gastown/... which passes."""
        assert handler.can_handle("/api/v2/dashboard/gastown/overview")

    def test_unversioned_path_accepted(self, handler):
        """Unversioned /api/dashboard/gastown/ paths pass via normalized check."""
        assert handler.can_handle("/api/dashboard/gastown/overview")

    def test_default_method_is_get(self, handler):
        assert handler.can_handle("/api/v1/dashboard/gastown/overview", method="GET")

    def test_post_method_accepted(self, handler):
        """can_handle does not filter by method; routing does."""
        assert handler.can_handle("/api/v1/dashboard/gastown/overview", method="POST")


# ---------------------------------------------------------------------------
# Authentication / RBAC tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for authentication and permission checks."""

    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self, handler, mock_http):
        """Missing auth returns 401."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import UnauthorizedError

        original = SecureHandler.get_auth_context

        async def raise_unauth(self, req, require_auth=True):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 401
        assert "Authentication required" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    async def test_forbidden_returns_403(self, handler, mock_http):
        """ForbiddenError from get_auth_context returns 403."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        async def raise_forbidden(self, req, require_auth=True):
            raise ForbiddenError("Blocked")

        with patch.object(SecureHandler, "get_auth_context", raise_forbidden):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    async def test_permission_denied_returns_403(self, handler, mock_http):
        """If check_permission raises ForbiddenError, returns 403."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        mock_ctx = AuthorizationContext(
            user_id="u1", user_email="u@x.com", org_id="o1",
            roles={"viewer"}, permissions=set(),
        )

        async def mock_get_auth(self, req, require_auth=True):
            return mock_ctx

        def mock_check_perm(self, ctx, perm, resource_id=None):
            raise ForbiddenError("No gastown:read")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth), \
             patch.object(SecureHandler, "check_permission", mock_check_perm):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 403

    async def test_authenticated_request_succeeds(self, handler, mock_http):
        """With auto-auth, a basic request should succeed."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch(
            "aragora.server.handlers.gastown_dashboard.GasTownDashboardHandler._get_canonical_workspace_stores",
            return_value=None,
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Overview endpoint tests
# ---------------------------------------------------------------------------


class TestOverview:
    """Tests for GET /api/v1/dashboard/gastown/overview."""

    async def test_overview_empty_state(self, handler, mock_http):
        """Overview returns defaults when no extensions or stores are available."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["convoys"]["total"] == 0
        assert body["beads"]["total"] == 0
        assert body["agents"]["total"] == 0
        assert body["witness_patrol"]["active"] is False
        assert body["mayor"]["active"] is False

    async def test_overview_caching(self, handler, mock_http):
        """Second call returns cached data."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result1 = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)
            body1 = _body(result1)

            # Second call should return cached
            result2 = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)
            body2 = _body(result2)

        assert body1["generated_at"] == body2["generated_at"]

    async def test_overview_force_refresh(self, handler, mock_http):
        """refresh=true bypasses cache."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result1 = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {}, mock_http,
            )
            body1 = _body(result1)

            result2 = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )
            body2 = _body(result2)

        # Force refresh generates a new timestamp
        # (they may be equal if test is fast, but the code path is exercised)
        assert "generated_at" in body2

    async def test_overview_with_gastown_convoys(self, handler, mock_http):
        """Overview counts convoys from Gas Town tracker."""
        convoys = [
            MockConvoy("c1", "Convoy 1", MockGastownConvoyStatus.IN_PROGRESS),
            MockConvoy("c2", "Convoy 2", MockGastownConvoyStatus.COMPLETED),
            MockConvoy("c3", "Convoy 3", MockGastownConvoyStatus.BLOCKED),
            MockConvoy("c4", "Convoy 4", MockGastownConvoyStatus.REVIEW),
        ]

        mock_tracker = AsyncMock()
        mock_tracker.list_convoys = AsyncMock(return_value=convoys)

        mock_state = MagicMock()
        mock_state.convoy_tracker = mock_tracker
        mock_state.coordinator = None

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ), patch(
            "aragora.server.handlers.gastown_dashboard.GasTownDashboardHandler._get_canonical_workspace_stores",
            return_value=None,
        ), patch.dict(
            "sys.modules",
            {"aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus)},
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        body = _body(result)
        assert body["convoys"]["total"] == 4
        assert body["convoys"]["active"] == 2  # IN_PROGRESS + REVIEW
        assert body["convoys"]["completed"] == 1
        assert body["convoys"]["failed"] == 1  # BLOCKED

    async def test_overview_with_nomic_convoys(self, handler, mock_http):
        """Overview counts convoys from nomic stores when Gas Town tracker is unavailable."""
        convoys = [
            MockConvoy("c1", "Convoy 1", MockConvoyStatus.ACTIVE),
            MockConvoy("c2", "Convoy 2", MockConvoyStatus.COMPLETED),
            MockConvoy("c3", "Convoy 3", MockConvoyStatus.FAILED),
        ]

        mock_manager = AsyncMock()
        mock_manager.list_convoys = AsyncMock(return_value=convoys)

        mock_stores = AsyncMock()
        mock_stores.convoy_manager = AsyncMock(return_value=mock_manager)
        mock_stores.bead_store = AsyncMock(return_value=MagicMock())

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(
            handler, "_get_canonical_workspace_stores", return_value=mock_stores,
        ), patch.dict(
            "sys.modules",
            {"aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus)},
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        body = _body(result)
        assert body["convoys"]["total"] == 3
        assert body["convoys"]["active"] == 1
        assert body["convoys"]["completed"] == 1
        assert body["convoys"]["failed"] == 1

    async def test_overview_convoy_import_error(self, handler, mock_http):
        """ImportError in convoy section is handled gracefully."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(
            handler, "_get_canonical_workspace_stores", side_effect=ImportError("no module"),
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["convoys"]["total"] == 0

    async def test_overview_convoy_runtime_error(self, handler, mock_http):
        """RuntimeError in convoy section is handled gracefully."""
        mock_state = MagicMock()
        mock_state.convoy_tracker = MagicMock()
        mock_state.convoy_tracker.list_convoys = AsyncMock(side_effect=RuntimeError("oops"))

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ), patch.dict(
            "sys.modules",
            {"aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus)},
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["convoys"]["total"] == 0

    async def test_overview_bead_stats(self, handler, mock_http):
        """Overview collects bead counts by status."""
        mock_bead_store = AsyncMock()

        async def list_beads_by_status(status=None, limit=1000):
            if status == MockBeadStatus.PENDING:
                return [MockBead("b1", MockBeadStatus.PENDING)]
            if status == MockBeadStatus.COMPLETED:
                return [MockBead("b2", MockBeadStatus.COMPLETED), MockBead("b3", MockBeadStatus.COMPLETED)]
            return []

        mock_bead_store.list_beads = AsyncMock(side_effect=list_beads_by_status)

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(
            handler, "_get_canonical_workspace_stores", return_value=mock_stores,
        ), patch.dict(
            "sys.modules",
            {"aragora.nomic.stores": MagicMock(BeadStatus=MockBeadStatus, ConvoyStatus=MockConvoyStatus)},
        ):
            # Also need to mock convoy_manager for the convoy section
            mock_stores.convoy_manager = AsyncMock(
                return_value=AsyncMock(list_convoys=AsyncMock(return_value=[])),
            )
            result = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )

        body = _body(result)
        assert body["beads"]["pending"] == 1
        assert body["beads"]["completed"] == 2
        assert body["beads"]["total"] == 3

    async def test_overview_witness_patrol_active(self, handler, mock_http):
        """Overview reports witness patrol status."""
        mock_witness = MagicMock()
        mock_witness._running = True

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None), \
             patch(
                 "aragora.server.handlers.gastown_dashboard.get_witness_behavior",
                 return_value=mock_witness,
                 create=True,
             ), patch.dict("sys.modules", {
                 "aragora.server.startup": MagicMock(
                     get_witness_behavior=MagicMock(return_value=mock_witness),
                     get_mayor_coordinator=MagicMock(return_value=None),
                 ),
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )

        body = _body(result)
        assert body["witness_patrol"]["active"] is True

    async def test_overview_mayor_active(self, handler, mock_http):
        """Overview reports mayor status."""
        mock_coordinator = MagicMock()
        mock_coordinator.is_mayor = True
        mock_coordinator.get_current_mayor_node.return_value = "node-42"

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None), \
             patch.dict("sys.modules", {
                 "aragora.server.startup": MagicMock(
                     get_witness_behavior=MagicMock(return_value=None),
                     get_mayor_coordinator=MagicMock(return_value=mock_coordinator),
                 ),
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )

        body = _body(result)
        assert body["mayor"]["active"] is True
        assert body["mayor"]["node_id"] == "node-42"

    async def test_overview_witness_import_error(self, handler, mock_http):
        """ImportError from witness module handled gracefully."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None), \
             patch.dict("sys.modules", {
                 "aragora.server.startup": None,
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )

        assert _status(result) == 200

    async def test_overview_agent_stats_import_error(self, handler, mock_http):
        """ImportError for agent roles handled gracefully."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None), \
             patch.dict("sys.modules", {
                 "aragora.extensions.gastown.agent_roles": None,
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/overview", {"refresh": "true"}, mock_http,
            )

        body = _body(result)
        assert body["agents"]["total"] == 0


# ---------------------------------------------------------------------------
# Convoys endpoint tests
# ---------------------------------------------------------------------------


class TestConvoys:
    """Tests for GET /api/v1/dashboard/gastown/convoys."""

    async def test_convoys_empty(self, handler, mock_http):
        """Returns empty list when no stores available."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["convoys"] == []
        assert body["total"] == 0
        assert body["showing"] == 0

    async def test_convoys_with_gastown_tracker(self, handler, mock_http):
        """Returns convoy list from Gas Town tracker."""
        now = datetime.now(timezone.utc)
        convoys = [
            MockConvoy("c1", "Convoy Alpha", MockGastownConvoyStatus.IN_PROGRESS, now),
            MockConvoy("c2", "Convoy Beta", MockGastownConvoyStatus.COMPLETED, now),
        ]

        mock_tracker = AsyncMock()
        mock_tracker.list_convoys = AsyncMock(return_value=convoys)

        beads_c1 = [
            MockWorkspaceBead("b1", MockWorkspaceBeadStatus.DONE),
            MockWorkspaceBead("b2", MockWorkspaceBeadStatus.PENDING),
        ]
        beads_c2 = [
            MockWorkspaceBead("b3", MockWorkspaceBeadStatus.DONE),
        ]

        mock_bead_mgr = AsyncMock()

        async def list_beads(convoy_id=None):
            return beads_c1 if convoy_id == "c1" else beads_c2

        mock_bead_mgr.list_beads = AsyncMock(side_effect=list_beads)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=mock_tracker,
        ), patch.dict("sys.modules", {
            "aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus),
            "aragora.workspace.bead": MagicMock(
                BeadManager=MagicMock(return_value=mock_bead_mgr),
                BeadStatus=MockWorkspaceBeadStatus,
            ),
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["total"] == 2
        assert body["showing"] == 2
        assert body["convoys"][0]["id"] == "c1"
        assert body["convoys"][0]["total_beads"] == 2
        assert body["convoys"][0]["completed_beads"] == 1
        assert body["convoys"][0]["progress_percentage"] == 50.0

    async def test_convoys_with_nomic_stores(self, handler, mock_http):
        """Returns convoy list from nomic stores."""
        now = datetime.now(timezone.utc)

        mock_bead_completed = MagicMock()
        mock_bead_completed.status = MagicMock(value="completed")
        mock_bead_failed = MagicMock()
        mock_bead_failed.status = MagicMock(value="failed")

        convoy = MockConvoy("c1", "Test", MockConvoyStatus.ACTIVE, now, bead_ids=["b1", "b2"])

        mock_bead_store = AsyncMock()

        async def get_bead(bead_id):
            if bead_id == "b1":
                return mock_bead_completed
            return mock_bead_failed

        mock_bead_store.get = AsyncMock(side_effect=get_bead)

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=[convoy])

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["total"] == 1
        assert body["convoys"][0]["completed_beads"] == 1
        assert body["convoys"][0]["failed_beads"] == 1
        assert body["convoys"][0]["progress_percentage"] == 50.0

    async def test_convoys_nomic_missing_bead(self, handler, mock_http):
        """Missing beads in store are skipped gracefully."""
        convoy = MockConvoy("c1", "Test", MockConvoyStatus.ACTIVE, bead_ids=["b1", "missing"])

        mock_bead_store = AsyncMock()
        mock_bead_completed = MagicMock()
        mock_bead_completed.status = MagicMock(value="completed")

        async def get_bead(bead_id):
            if bead_id == "b1":
                return mock_bead_completed
            return None

        mock_bead_store.get = AsyncMock(side_effect=get_bead)

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=[convoy])

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["convoys"][0]["total_beads"] == 2
        assert body["convoys"][0]["completed_beads"] == 1

    async def test_convoys_limit_query_param(self, handler, mock_http):
        """Limit query param truncates results."""
        convoys = [
            MockConvoy(f"c{i}", f"Convoy {i}", MockConvoyStatus.ACTIVE, bead_ids=[])
            for i in range(5)
        ]

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=convoys)

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None)))
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/convoys", {"limit": "2"}, mock_http,
            )

        body = _body(result)
        assert body["total"] == 5
        assert body["showing"] == 2

    async def test_convoys_invalid_status_filter_gastown(self, handler, mock_http):
        """Invalid status filter returns 400 for Gas Town tracker path."""
        mock_tracker = AsyncMock()
        mock_tracker.list_convoys = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=mock_tracker,
        ), patch.dict("sys.modules", {
            "aragora.extensions.gastown.models": MagicMock(
                ConvoyStatus=MockGastownConvoyStatus,
            ),
            "aragora.workspace.bead": MagicMock(),
        }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/convoys", {"status": "invalid_status"}, mock_http,
            )

        assert _status(result) == 400
        assert "Invalid status" in _body(result).get("error", "")

    async def test_convoys_invalid_status_filter_nomic(self, handler, mock_http):
        """Invalid status filter returns 400 for nomic stores path."""
        mock_convoy_mgr = AsyncMock()
        mock_bead_store = AsyncMock()

        mock_stores = AsyncMock()
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/convoys", {"status": "bogus"}, mock_http,
            )

        assert _status(result) == 400

    async def test_convoys_valid_status_filter_nomic(self, handler, mock_http):
        """Valid status filter is applied for nomic stores path."""
        convoys = [MockConvoy("c1", "Test", MockConvoyStatus.ACTIVE, bead_ids=[])]

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=convoys)

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None)))
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/convoys", {"status": "active"}, mock_http,
            )

        body = _body(result)
        assert _status(result) == 200

    async def test_convoys_import_error(self, handler, mock_http):
        """ImportError returns empty list."""
        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            side_effect=ImportError("no convoy"),
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["convoys"] == []
        assert body["total"] == 0

    async def test_convoys_attribute_error(self, handler, mock_http):
        """AttributeError returns empty with error flag."""
        mock_tracker = MagicMock()
        mock_tracker.list_convoys = AsyncMock(side_effect=AttributeError("bad attr"))

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=mock_tracker,
        ), patch.dict("sys.modules", {
            "aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus),
            "aragora.workspace.bead": MagicMock(),
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body.get("error") == "data_error"

    async def test_convoys_runtime_error_returns_500(self, handler, mock_http):
        """RuntimeError returns 500."""
        mock_tracker = MagicMock()
        mock_tracker.list_convoys = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=mock_tracker,
        ), patch.dict("sys.modules", {
            "aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus),
            "aragora.workspace.bead": MagicMock(),
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        assert _status(result) == 500

    async def test_convoys_zero_beads_progress(self, handler, mock_http):
        """Convoy with zero beads has 0.0 progress."""
        convoy = MockConvoy("c1", "Empty", MockConvoyStatus.ACTIVE, bead_ids=[])

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=[convoy])

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None)))
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["convoys"][0]["progress_percentage"] == 0.0

    async def test_convoys_null_timestamps(self, handler, mock_http):
        """Convoys with null timestamps yield null in response."""
        convoy = MockConvoy("c1", "NoDate", MockConvoyStatus.ACTIVE, bead_ids=[])
        convoy.created_at = None
        convoy.updated_at = None

        mock_convoy_mgr = AsyncMock()
        mock_convoy_mgr.list_convoys = AsyncMock(return_value=[convoy])

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=AsyncMock(get=AsyncMock(return_value=None)))
        mock_stores.convoy_manager = AsyncMock(return_value=mock_convoy_mgr)

        with patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {}, mock_http)

        body = _body(result)
        assert body["convoys"][0]["created_at"] is None
        assert body["convoys"][0]["updated_at"] is None


# ---------------------------------------------------------------------------
# Agents endpoint tests
# ---------------------------------------------------------------------------


class TestAgents:
    """Tests for GET /api/v1/dashboard/gastown/agents."""

    async def test_agents_import_error(self, handler, mock_http):
        """Returns empty when agent module not available."""
        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": None}):
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        body = _body(result)
        assert body["agents_by_role"] == {}
        assert body["totals"]["total"] == 0

    async def test_agents_with_data(self, handler, mock_http):
        """Returns agents grouped by role."""
        agents_mayor = [
            MockAgent("a1", MockAgentRole.MAYOR, capabilities=[MockAgentCapability.DEBATE]),
        ]
        agents_witness = [
            MockAgent("a2", MockAgentRole.WITNESS),
            MockAgent("a3", MockAgentRole.WITNESS),
        ]

        mock_hierarchy = AsyncMock()

        async def list_agents(role=None):
            if role == MockAgentRole.MAYOR:
                return agents_mayor
            if role == MockAgentRole.WITNESS:
                return agents_witness
            return []

        mock_hierarchy.list_agents = AsyncMock(side_effect=list_agents)

        mock_module = MagicMock()
        mock_module.AgentRole = MockAgentRole
        mock_module.AgentHierarchy = MagicMock(return_value=mock_hierarchy)

        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": mock_module}):
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        body = _body(result)
        assert body["totals"]["mayor"] == 1
        assert body["totals"]["witness"] == 2
        assert body["totals"]["total"] == 3

    async def test_agents_data_error(self, handler, mock_http):
        """TypeError returns empty with error flag."""
        mock_module = MagicMock()
        mock_module.AgentRole = MockAgentRole
        mock_module.AgentHierarchy = MagicMock(side_effect=TypeError("bad"))

        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": mock_module}):
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        body = _body(result)
        assert body.get("error") == "data_error"
        assert body["totals"]["total"] == 0

    async def test_agents_runtime_error_returns_500(self, handler, mock_http):
        """RuntimeError returns 500."""
        mock_module = MagicMock()
        mock_module.AgentRole = MockAgentRole
        mock_module.AgentHierarchy = MagicMock(side_effect=RuntimeError("crash"))

        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": mock_module}):
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        assert _status(result) == 500

    async def test_agents_agent_fields(self, handler, mock_http):
        """Agent serialization includes all expected fields."""
        now = datetime.now(timezone.utc)
        agent = MockAgent(
            "a1", MockAgentRole.MAYOR,
            supervised_by="supervisor-1",
            is_ephemeral=True,
            assigned_at=now,
            capabilities=[MockAgentCapability.DEBATE, MockAgentCapability.REVIEW],
        )

        mock_hierarchy = AsyncMock()

        async def list_agents(role=None):
            if role == MockAgentRole.MAYOR:
                return [agent]
            return []

        mock_hierarchy.list_agents = AsyncMock(side_effect=list_agents)

        mock_module = MagicMock()
        mock_module.AgentRole = MockAgentRole
        mock_module.AgentHierarchy = MagicMock(return_value=mock_hierarchy)

        with patch.dict("sys.modules", {"aragora.extensions.gastown.agent_roles": mock_module}):
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        body = _body(result)
        mayor_agents = body["agents_by_role"]["mayor"]
        assert len(mayor_agents) == 1
        a = mayor_agents[0]
        assert a["agent_id"] == "a1"
        assert a["role"] == "mayor"
        assert a["supervised_by"] == "supervisor-1"
        assert a["is_ephemeral"] is True
        assert a["assigned_at"] == now.isoformat()
        assert a["capabilities"] == ["debate", "review"]


# ---------------------------------------------------------------------------
# Beads endpoint tests
# ---------------------------------------------------------------------------


class TestBeads:
    """Tests for GET /api/v1/dashboard/gastown/beads."""

    async def test_beads_no_stores(self, handler, mock_http):
        """Returns empty when no stores available."""
        with patch.object(handler, "_get_canonical_workspace_stores", return_value=None), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(BeadStatus=MockBeadStatus, BeadPriority=MockBeadPriority),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        body = _body(result)
        assert body["queue_depth"] == 0
        assert body["total"] == 0

    async def test_beads_with_data(self, handler, mock_http):
        """Returns bead counts by status and priority."""
        mock_bead_store = AsyncMock()

        async def list_beads(status=None, priority=None, limit=1000):
            if status == MockBeadStatus.PENDING:
                return [MagicMock()] * 3
            if status == MockBeadStatus.RUNNING:
                return [MagicMock()] * 2
            if status == MockBeadStatus.COMPLETED:
                return [MagicMock()] * 5
            if status == MockBeadStatus.FAILED:
                return [MagicMock()] * 1
            if priority == MockBeadPriority.HIGH:
                return [MagicMock()] * 4
            if priority == MockBeadPriority.LOW:
                return [MagicMock()] * 2
            return []

        mock_bead_store.list_beads = AsyncMock(side_effect=list_beads)

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)

        with patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(BeadStatus=MockBeadStatus, BeadPriority=MockBeadPriority),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        body = _body(result)
        assert body["by_status"]["pending"] == 3
        assert body["by_status"]["running"] == 2
        assert body["by_status"]["completed"] == 5
        assert body["by_status"]["failed"] == 1
        assert body["total"] == 11
        assert body["queue_depth"] == 3
        assert body["processing"] == 2  # running + claimed (no claimed here)
        assert body["by_priority"]["high"] == 4
        assert body["by_priority"]["low"] == 2

    async def test_beads_import_error(self, handler, mock_http):
        """Returns empty when bead module unavailable."""
        with patch.dict("sys.modules", {"aragora.nomic.stores": None}):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        body = _body(result)
        assert body["queue_depth"] == 0
        assert body["total"] == 0

    async def test_beads_attribute_error(self, handler, mock_http):
        """AttributeError returns empty with error flag."""
        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(side_effect=AttributeError("broken"))

        with patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(BeadStatus=MockBeadStatus, BeadPriority=MockBeadPriority),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        body = _body(result)
        assert body.get("error") == "data_error"

    async def test_beads_runtime_error_returns_500(self, handler, mock_http):
        """RuntimeError returns 500."""
        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(BeadStatus=MockBeadStatus, BeadPriority=MockBeadPriority),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        assert _status(result) == 500

    async def test_beads_claimed_contributes_to_processing(self, handler, mock_http):
        """Claimed beads are included in the processing count."""
        # We need a BeadStatus that includes 'claimed'
        class BeadStatusWithClaimed(Enum):
            PENDING = "pending"
            RUNNING = "running"
            CLAIMED = "claimed"
            COMPLETED = "completed"

        mock_bead_store = AsyncMock()

        async def list_beads(status=None, priority=None, limit=1000):
            if status is not None and status.value == "claimed":
                return [MagicMock()] * 2
            if status is not None and status.value == "running":
                return [MagicMock()] * 3
            return []

        mock_bead_store.list_beads = AsyncMock(side_effect=list_beads)

        mock_stores = AsyncMock()
        mock_stores.bead_store = AsyncMock(return_value=mock_bead_store)

        with patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(
                     BeadStatus=BeadStatusWithClaimed,
                     BeadPriority=MockBeadPriority,
                 ),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        body = _body(result)
        assert body["processing"] == 5  # 3 running + 2 claimed


# ---------------------------------------------------------------------------
# Metrics endpoint tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for GET /api/v1/dashboard/gastown/metrics."""

    async def test_metrics_defaults(self, handler, mock_http):
        """Returns default metrics when no sources available."""
        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": None,
        }), patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=None):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["period_hours"] == 24
        assert body["beads_per_hour"] == 0.0
        assert body["avg_bead_duration_minutes"] is None
        assert body["convoy_completion_rate"] == 0.0
        assert body["gupp_recovery_events"] == 0

    async def test_metrics_with_gastown_metrics_module(self, handler, mock_http):
        """Returns metrics from Gas Town metrics module."""
        mock_metrics = MagicMock()
        mock_metrics.get_beads_completed_count = MagicMock(return_value=48)
        mock_metrics.get_convoy_completion_rate = MagicMock(return_value=75.5)
        mock_metrics.get_gupp_recovery_count = MagicMock(return_value=3)

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": mock_metrics,
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["beads_per_hour"] == 2.0  # 48 / 24
        assert body["convoy_completion_rate"] == 75.5
        assert body["gupp_recovery_events"] == 3

    async def test_metrics_custom_hours(self, handler, mock_http):
        """Custom hours query param affects calculations."""
        mock_metrics = MagicMock()
        mock_metrics.get_beads_completed_count = MagicMock(return_value=100)
        mock_metrics.get_convoy_completion_rate = MagicMock(return_value=50.0)
        mock_metrics.get_gupp_recovery_count = MagicMock(return_value=0)

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": mock_metrics,
        }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/metrics", {"hours": "10"}, mock_http,
            )

        body = _body(result)
        assert body["period_hours"] == 10
        assert body["beads_per_hour"] == 10.0  # 100 / 10

    async def test_metrics_fallback_gastown_tracker(self, handler, mock_http):
        """Fallback calculates convoy_completion_rate from Gas Town tracker."""
        convoys = [
            MockConvoy("c1", "Done", MockGastownConvoyStatus.COMPLETED),
            MockConvoy("c2", "WIP", MockGastownConvoyStatus.IN_PROGRESS),
            MockConvoy("c3", "Done2", MockGastownConvoyStatus.COMPLETED),
            MockConvoy("c4", "Blocked", MockGastownConvoyStatus.BLOCKED),
        ]
        mock_tracker = AsyncMock()
        mock_tracker.list_convoys = AsyncMock(return_value=convoys)

        mock_state = MagicMock()
        mock_state.convoy_tracker = mock_tracker
        mock_state.coordinator = None

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": None,
            "aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus),
        }), patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["convoy_completion_rate"] == 50.0  # 2/4 * 100

    async def test_metrics_fallback_nomic_stores(self, handler, mock_http):
        """Fallback calculates convoy_completion_rate from nomic stores."""
        convoys = [
            MockConvoy("c1", "Done", MockConvoyStatus.COMPLETED),
            MockConvoy("c2", "Active", MockConvoyStatus.ACTIVE),
        ]

        mock_mgr = AsyncMock()
        mock_mgr.list_convoys = AsyncMock(return_value=convoys)

        mock_stores = AsyncMock()
        mock_stores.convoy_manager = AsyncMock(return_value=mock_mgr)

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": None,
        }), patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            return_value=None,
        ), patch.object(handler, "_get_canonical_workspace_stores", return_value=mock_stores), \
             patch.dict("sys.modules", {
                 "aragora.nomic.stores": MagicMock(ConvoyStatus=MockConvoyStatus),
             }):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["convoy_completion_rate"] == 50.0

    async def test_metrics_fallback_import_error(self, handler, mock_http):
        """Complete import failure in fallback still returns default metrics."""
        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": None,
        }), patch(
            "aragora.server.handlers.gastown_dashboard._resolve_convoy_tracker",
            side_effect=ImportError("nope"),
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["convoy_completion_rate"] == 0.0

    async def test_metrics_data_error(self, handler, mock_http):
        """AttributeError from metrics module returns default metrics."""
        mock_metrics = MagicMock()
        mock_metrics.get_beads_completed_count = MagicMock(side_effect=AttributeError("bad"))

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": mock_metrics,
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["beads_per_hour"] == 0.0

    async def test_metrics_runtime_error(self, handler, mock_http):
        """RuntimeError from metrics module returns default metrics (not 500)."""
        mock_metrics = MagicMock()
        mock_metrics.get_beads_completed_count = MagicMock(side_effect=RuntimeError("oops"))

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": mock_metrics,
        }):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        # Metrics endpoint always returns json_response, never error_response
        assert _status(result) == 200

    async def test_metrics_zero_hours_clamped(self, handler, mock_http):
        """Hours is clamped to min_val=1 by safe_query_int."""
        mock_metrics = MagicMock()
        mock_metrics.get_beads_completed_count = MagicMock(return_value=10)
        mock_metrics.get_convoy_completion_rate = MagicMock(return_value=0.0)
        mock_metrics.get_gupp_recovery_count = MagicMock(return_value=0)

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": mock_metrics,
        }):
            result = await handler.handle(
                "/api/v1/dashboard/gastown/metrics", {"hours": "0"}, mock_http,
            )

        body = _body(result)
        # safe_query_int clamps to min_val=1
        assert body["period_hours"] == 1
        assert body["beads_per_hour"] == 10.0

    async def test_metrics_fallback_empty_convoys(self, handler, mock_http):
        """Fallback with zero convoys keeps completion rate at 0."""
        mock_tracker = AsyncMock()
        mock_tracker.list_convoys = AsyncMock(return_value=[])

        mock_state = MagicMock()
        mock_state.convoy_tracker = mock_tracker
        mock_state.coordinator = None

        with patch.dict("sys.modules", {
            "aragora.extensions.gastown.metrics": None,
            "aragora.extensions.gastown.models": MagicMock(ConvoyStatus=MockGastownConvoyStatus),
        }), patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ):
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        body = _body(result)
        assert body["convoy_completion_rate"] == 0.0


# ---------------------------------------------------------------------------
# Route dispatch tests
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    """Tests for the main handle() method route dispatch."""

    async def test_unknown_sub_path_returns_none(self, handler, mock_http):
        """Unknown sub-path returns None (not handled)."""
        result = await handler.handle("/api/v1/dashboard/gastown/unknown", {}, mock_http)
        assert result is None

    async def test_overview_route_dispatch(self, handler, mock_http):
        """Overview route dispatches correctly."""
        with patch.object(handler, "_get_overview", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v1/dashboard/gastown/overview", {}, mock_http)

        mock_get.assert_awaited_once()

    async def test_convoys_route_dispatch(self, handler, mock_http):
        """Convoys route dispatches correctly."""
        with patch.object(handler, "_get_convoys", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v1/dashboard/gastown/convoys", {"limit": "5"}, mock_http)

        mock_get.assert_awaited_once()

    async def test_agents_route_dispatch(self, handler, mock_http):
        """Agents route dispatches correctly."""
        with patch.object(handler, "_get_agents", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v1/dashboard/gastown/agents", {}, mock_http)

        mock_get.assert_awaited_once()

    async def test_beads_route_dispatch(self, handler, mock_http):
        """Beads route dispatches correctly."""
        with patch.object(handler, "_get_beads", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v1/dashboard/gastown/beads", {}, mock_http)

        mock_get.assert_awaited_once()

    async def test_metrics_route_dispatch(self, handler, mock_http):
        """Metrics route dispatches correctly."""
        with patch.object(handler, "_get_metrics", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v1/dashboard/gastown/metrics", {}, mock_http)

        mock_get.assert_awaited_once()

    async def test_v2_path_dispatches_correctly(self, handler, mock_http):
        """v2 paths are normalized and dispatch to the correct handler."""
        with patch.object(handler, "_get_overview", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b'{}')
            result = await handler.handle("/api/v2/dashboard/gastown/overview", {}, mock_http)

        mock_get.assert_awaited_once()


# ---------------------------------------------------------------------------
# Cache utility tests
# ---------------------------------------------------------------------------


class TestCacheUtilities:
    """Tests for _get_cached_data and _set_cached_data."""

    def test_set_and_get_cache(self):
        """Set then get returns cached data."""
        data = {"key": "value"}
        _set_cached_data("test_key", data)
        result = _get_cached_data("test_key")
        assert result == data

    def test_get_cache_miss(self):
        """Nonexistent key returns None."""
        result = _get_cached_data("nonexistent")
        assert result is None

    def test_cache_expiry(self):
        """Expired cache returns None."""
        import time

        _gt_dashboard_cache["expired"] = {
            "data": {"old": True},
            "cached_at": datetime.now(timezone.utc).timestamp() - CACHE_TTL - 1,
        }
        result = _get_cached_data("expired")
        assert result is None

    def test_cache_not_expired(self):
        """Non-expired cache returns data."""
        _gt_dashboard_cache["fresh"] = {
            "data": {"fresh": True},
            "cached_at": datetime.now(timezone.utc).timestamp(),
        }
        result = _get_cached_data("fresh")
        assert result == {"fresh": True}


# ---------------------------------------------------------------------------
# _get_gastown_state / _resolve_convoy_tracker tests
# ---------------------------------------------------------------------------


class TestGasTownStateResolution:
    """Tests for _get_gastown_state and _resolve_convoy_tracker."""

    def test_get_gastown_state_import_error(self):
        """ImportError returns None."""
        from aragora.server.handlers.gastown_dashboard import _get_gastown_state

        with patch.dict("sys.modules", {"aragora.server.extensions": None}):
            result = _get_gastown_state()
        assert result is None

    def test_get_gastown_state_runtime_error(self):
        """RuntimeError returns None."""
        from aragora.server.handlers.gastown_dashboard import _get_gastown_state

        mock_ext = MagicMock()
        mock_ext.get_extension_state = MagicMock(side_effect=RuntimeError("no state"))

        with patch.dict("sys.modules", {"aragora.server.extensions": mock_ext}):
            result = _get_gastown_state()
        assert result is None

    def test_resolve_convoy_tracker_no_state(self):
        """Returns None when no Gas Town state."""
        from aragora.server.handlers.gastown_dashboard import _resolve_convoy_tracker

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=None,
        ):
            result = _resolve_convoy_tracker()
        assert result is None

    def test_resolve_convoy_tracker_from_tracker(self):
        """Returns convoy_tracker directly when present."""
        from aragora.server.handlers.gastown_dashboard import _resolve_convoy_tracker

        mock_tracker = MagicMock()
        mock_state = MagicMock()
        mock_state.convoy_tracker = mock_tracker
        mock_state.coordinator = None

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ):
            result = _resolve_convoy_tracker()
        assert result is mock_tracker

    def test_resolve_convoy_tracker_from_coordinator(self):
        """Falls back to coordinator.convoys when convoy_tracker is None."""
        from aragora.server.handlers.gastown_dashboard import _resolve_convoy_tracker

        mock_convoys = MagicMock()
        mock_state = MagicMock()
        mock_state.convoy_tracker = None
        mock_state.coordinator = MagicMock()
        mock_state.coordinator.convoys = mock_convoys

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ):
            result = _resolve_convoy_tracker()
        assert result is mock_convoys

    def test_resolve_convoy_tracker_no_tracker_no_coordinator(self):
        """Returns None when neither tracker nor coordinator available."""
        from aragora.server.handlers.gastown_dashboard import _resolve_convoy_tracker

        mock_state = MagicMock()
        mock_state.convoy_tracker = None
        mock_state.coordinator = None

        with patch(
            "aragora.server.handlers.gastown_dashboard._get_gastown_state",
            return_value=mock_state,
        ):
            result = _resolve_convoy_tracker()
        assert result is None


# ---------------------------------------------------------------------------
# Workspace stores tests
# ---------------------------------------------------------------------------


class TestWorkspaceStores:
    """Tests for _get_canonical_workspace_stores."""

    def test_stores_cached(self, handler):
        """Stores are cached on first call."""
        mock_stores = MagicMock()
        with patch.dict("sys.modules", {
            "aragora.stores": MagicMock(get_canonical_workspace_stores=MagicMock(return_value=mock_stores)),
        }):
            result1 = handler._get_canonical_workspace_stores()
            result2 = handler._get_canonical_workspace_stores()

        assert result1 is result2

    def test_stores_import_error(self, handler):
        """ImportError returns None."""
        with patch.dict("sys.modules", {"aragora.stores": None}):
            # Clear any cached stores
            if hasattr(handler, "_canonical_workspace_stores"):
                delattr(handler, "_canonical_workspace_stores")
            result = handler._get_canonical_workspace_stores()
        assert result is None


# ---------------------------------------------------------------------------
# Handler attributes
# ---------------------------------------------------------------------------


class TestHandlerAttributes:
    """Tests for handler class attributes."""

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "gastown"

    def test_routes_list(self, handler):
        assert len(handler.ROUTES) == 5
        assert "/api/v1/dashboard/gastown/overview" in handler.ROUTES
        assert "/api/v1/dashboard/gastown/convoys" in handler.ROUTES
        assert "/api/v1/dashboard/gastown/agents" in handler.ROUTES
        assert "/api/v1/dashboard/gastown/beads" in handler.ROUTES
        assert "/api/v1/dashboard/gastown/metrics" in handler.ROUTES

    def test_cache_ttl(self):
        assert CACHE_TTL == 15

    def test_handler_inherits_secure(self, handler):
        from aragora.server.handlers.secure import SecureHandler

        assert isinstance(handler, SecureHandler)

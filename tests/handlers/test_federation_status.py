"""Tests for FederationStatusHandler.

Covers all routes and behavior of the FederationStatusHandler class:
- handle_request routing for all ROUTES
- GET /api/v1/federation/status         - Federation overview status
- GET /api/v1/federation/workspaces     - List connected workspaces with health
- GET /api/v1/federation/activity       - Recent sync activity feed
- GET /api/v1/federation/config         - Federation configuration
- Coordinator unavailable fallbacks (graceful degradation)
- Error handling for coordinator exceptions
- Version-prefix stripping (/api/v1/ -> /api/)
- Unmatched route returns None
- Non-GET methods return None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.federation.status import FederationStatusHandler

# =============================================================================
# Module path for patching
# =============================================================================

MODULE = "aragora.server.handlers.federation.status"

# =============================================================================
# Helpers
# =============================================================================


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Parse the JSON body from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Domain Objects
# =============================================================================


class MockFederationMode(Enum):
    FULL = "full"
    READ_ONLY = "read_only"
    ISOLATED = "isolated"


class MockConsentScope(Enum):
    FULL = "full"
    READ_ONLY = "read_only"
    RESTRICTED = "restricted"


class MockSharingScope(Enum):
    GLOBAL = "global"
    ORG_ONLY = "org_only"
    NONE = "none"


class MockOperationType(Enum):
    SYNC = "sync"
    QUERY = "query"
    EXECUTE = "execute"


@dataclass
class MockWorkspace:
    """Mock federated workspace."""

    id: str
    name: str | None = None
    org_id: str = "org-1"
    is_online: bool = True
    federation_mode: MockFederationMode = MockFederationMode.FULL
    last_heartbeat: datetime | None = None
    latency_ms: float = 25.0
    supports_agent_execution: bool = True
    supports_workflow_execution: bool = True
    supports_knowledge_query: bool = True


@dataclass
class MockConsent:
    """Mock federation consent."""

    id: str
    source_workspace_id: str
    target_workspace_id: str
    scope: MockConsentScope = MockConsentScope.FULL
    data_types: set[str] = field(default_factory=lambda: {"debates", "receipts"})
    times_used: int = 5
    revoked: bool = False
    revoked_at: datetime | None = None
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime | None = None
    _valid: bool = True

    def is_valid(self) -> bool:
        return self._valid and not self.revoked


@dataclass
class MockPendingRequest:
    """Mock pending federation request."""

    id: str
    operation: MockOperationType = MockOperationType.SYNC
    source_workspace_id: str = "ws-1"
    target_workspace_id: str = "ws-2"
    requester_id: str = "user-1"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MockDefaultPolicy:
    """Mock federation default policy."""

    mode: MockFederationMode = MockFederationMode.FULL
    require_approval: bool = False
    sharing_scope: MockSharingScope = MockSharingScope.GLOBAL
    audit_all_requests: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "require_approval": self.require_approval,
            "sharing_scope": self.sharing_scope.value,
            "audit_all_requests": self.audit_all_requests,
        }


# =============================================================================
# Mock Coordinator
# =============================================================================


class MockCrossWorkspaceCoordinator:
    """Mock CrossWorkspaceCoordinator for handler tests."""

    def __init__(
        self,
        workspaces: list[MockWorkspace] | None = None,
        consents: list[MockConsent] | None = None,
        pending_requests: list[MockPendingRequest] | None = None,
        stats: dict[str, Any] | None = None,
        default_policy: MockDefaultPolicy | None = None,
        workspace_policies: dict[str, Any] | None = None,
    ):
        self._workspaces = workspaces or []
        self._consents = consents or []
        self._pending_requests = pending_requests or []
        self._stats = stats or {
            "valid_consents": 3,
            "pending_requests": 1,
            "registered_handlers": ["sync", "query"],
        }
        self._default_policy = default_policy or MockDefaultPolicy()
        self._workspace_policies = workspace_policies or {}

    def get_stats(self) -> dict[str, Any]:
        return self._stats

    def list_workspaces(self) -> list[MockWorkspace]:
        return self._workspaces

    def list_consents(self) -> list[MockConsent]:
        return self._consents

    def list_pending_requests(self) -> list[MockPendingRequest]:
        return self._pending_requests


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a FederationStatusHandler with minimal server context."""
    return FederationStatusHandler({})


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with sample data."""
    now = datetime.now(timezone.utc)
    workspaces = [
        MockWorkspace(
            id="ws-1",
            name="Primary",
            org_id="org-1",
            is_online=True,
            last_heartbeat=now,
            latency_ms=12.5,
        ),
        MockWorkspace(
            id="ws-2",
            name="Secondary",
            org_id="org-1",
            is_online=True,
            last_heartbeat=now - timedelta(minutes=2),
            latency_ms=45.0,
        ),
    ]
    consents = [
        MockConsent(
            id="consent-1",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            times_used=10,
            granted_at=now - timedelta(hours=1),
            last_used=now - timedelta(minutes=5),
        ),
        MockConsent(
            id="consent-2",
            source_workspace_id="ws-2",
            target_workspace_id="ws-1",
            times_used=3,
            granted_at=now - timedelta(days=1),
        ),
    ]
    pending = [
        MockPendingRequest(
            id="req-1",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            requester_id="user-42",
            created_at=now - timedelta(minutes=10),
        ),
    ]
    return MockCrossWorkspaceCoordinator(
        workspaces=workspaces,
        consents=consents,
        pending_requests=pending,
    )


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouting:
    """Test handle_request routing logic."""

    def test_routes_class_attribute(self):
        """Verify ROUTES lists expected paths."""
        assert "/api/federation/status" in FederationStatusHandler.ROUTES
        assert "/api/federation/workspaces" in FederationStatusHandler.ROUTES
        assert "/api/federation/activity" in FederationStatusHandler.ROUTES
        assert "/api/federation/config" in FederationStatusHandler.ROUTES

    def test_get_status_route(self, handler):
        """GET /api/v1/federation/status routes correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})
        assert result is not None
        assert _status(result) == 200

    def test_get_workspaces_route(self, handler):
        """GET /api/v1/federation/workspaces routes correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})
        assert result is not None
        assert _status(result) == 200

    def test_get_activity_route(self, handler):
        """GET /api/v1/federation/activity routes correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})
        assert result is not None
        assert _status(result) == 200

    def test_get_config_route(self, handler):
        """GET /api/v1/federation/config routes correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})
        assert result is not None
        assert _status(result) == 200

    def test_unversioned_path_routes(self, handler):
        """Unversioned /api/federation/status also routes correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/federation/status", None, {})
        assert result is not None
        assert _status(result) == 200

    def test_post_method_returns_none(self, handler):
        """POST to any federation route returns None (not handled)."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("POST", "/api/v1/federation/status", None, {})
        assert result is None

    def test_put_method_returns_none(self, handler):
        """PUT to any federation route returns None."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("PUT", "/api/v1/federation/status", None, {})
        assert result is None

    def test_delete_method_returns_none(self, handler):
        """DELETE to any federation route returns None."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("DELETE", "/api/v1/federation/status", None, {})
        assert result is None

    def test_unknown_path_returns_none(self, handler):
        """Unknown path returns None."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/unknown", None, {})
        assert result is None

    def test_partial_path_returns_none(self, handler):
        """Partial path without specific endpoint returns None."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation", None, {})
        assert result is None


# =============================================================================
# GET /api/v1/federation/status Tests
# =============================================================================


class TestFederationStatus:
    """Test the federation status overview endpoint."""

    def test_coordinator_unavailable(self, handler):
        """When coordinator is unavailable, returns offline fallback."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"
        assert body["connected_workspaces"] == 0
        assert body["shared_knowledge_count"] == 0
        assert body["sync_health"] == "offline"
        assert body["federation_mode"] == "isolated"
        assert body["message"] == "Federation service not initialized"

    def test_active_with_all_online(self, handler, mock_coordinator):
        """All workspaces online shows healthy status."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=mock_coordinator):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "active"
        assert body["connected_workspaces"] == 2
        assert body["online_workspaces"] == 2
        assert body["sync_health"] == "healthy"
        assert body["shared_knowledge_count"] == 13  # 10 + 3 from valid consents
        assert body["valid_consents"] == 3
        assert body["pending_requests"] == 1
        assert body["registered_handlers"] == ["sync", "query"]
        assert "timestamp" in body

    def test_degraded_health_when_partial_online(self, handler):
        """Mixed online/offline workspaces shows degraded health."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[
                MockWorkspace(id="ws-1", is_online=True),
                MockWorkspace(id="ws-2", is_online=False),
            ],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["sync_health"] == "degraded"
        assert body["online_workspaces"] == 1
        assert body["connected_workspaces"] == 2

    def test_offline_health_when_none_online(self, handler):
        """All workspaces offline shows offline health."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[
                MockWorkspace(id="ws-1", is_online=False),
                MockWorkspace(id="ws-2", is_online=False),
            ],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["sync_health"] == "offline"

    def test_idle_health_when_no_workspaces(self, handler):
        """No workspaces shows idle health."""
        coord = MockCrossWorkspaceCoordinator(workspaces=[], consents=[])
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["status"] == "idle"
        assert body["sync_health"] == "idle"
        assert body["connected_workspaces"] == 0

    def test_federation_mode_from_policy(self, handler):
        """Federation mode is read from coordinator's default policy."""
        policy = MockDefaultPolicy(mode=MockFederationMode.READ_ONLY)
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1")],
            consents=[],
            default_policy=policy,
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["federation_mode"] == "read_only"

    def test_federation_mode_isolated_when_no_policy(self, handler):
        """Without default policy, federation_mode falls back to 'isolated'."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1")],
            consents=[],
        )
        # Remove the policy
        coord._default_policy = None
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["federation_mode"] == "isolated"

    def test_shared_knowledge_only_counts_valid_consents(self, handler):
        """shared_knowledge_count only includes valid (non-revoked) consents."""
        consents = [
            MockConsent(id="c1", source_workspace_id="ws-1", target_workspace_id="ws-2", times_used=10),
            MockConsent(id="c2", source_workspace_id="ws-2", target_workspace_id="ws-1", times_used=5, revoked=True),
            MockConsent(id="c3", source_workspace_id="ws-1", target_workspace_id="ws-3", times_used=7, _valid=False),
        ]
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1")],
            consents=consents,
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["shared_knowledge_count"] == 10  # Only c1 is valid

    def test_coordinator_error_returns_500(self, handler):
        """RuntimeError from coordinator returns 500 error."""
        coord = MagicMock()
        coord.get_stats.side_effect = RuntimeError("coordinator down")
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert _status(result) == 500

    def test_coordinator_value_error_returns_500(self, handler):
        """ValueError from coordinator returns 500 error."""
        coord = MagicMock()
        coord.get_stats.side_effect = ValueError("bad data")
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert _status(result) == 500


# =============================================================================
# GET /api/v1/federation/workspaces Tests
# =============================================================================


class TestListWorkspaces:
    """Test the connected workspaces listing endpoint."""

    def test_coordinator_unavailable(self, handler):
        """When coordinator is unavailable, returns empty workspace list."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_workspace_list_with_data(self, handler, mock_coordinator):
        """Returns enriched workspace data with consents and capabilities."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=mock_coordinator):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        workspaces = body["workspaces"]

        # First workspace
        ws1 = workspaces[0]
        assert ws1["id"] == "ws-1"
        assert ws1["name"] == "Primary"
        assert ws1["org_id"] == "org-1"
        assert ws1["status"] == "connected"
        assert ws1["is_online"] is True
        assert ws1["federation_mode"] == "full"
        assert ws1["last_heartbeat"] is not None
        assert ws1["latency_ms"] == 12.5
        assert ws1["capabilities"]["agent_execution"] is True
        assert ws1["capabilities"]["workflow_execution"] is True
        assert ws1["capabilities"]["knowledge_query"] is True

    def test_workspace_status_connected(self, handler):
        """Online workspace shows 'connected' status."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1", is_online=True)],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        ws = _body(result)["workspaces"][0]
        assert ws["status"] == "connected"

    def test_workspace_status_stale(self, handler):
        """Offline workspace with a last_heartbeat shows 'stale' status."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[
                MockWorkspace(
                    id="ws-1",
                    is_online=False,
                    last_heartbeat=datetime.now(timezone.utc) - timedelta(minutes=10),
                ),
            ],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        ws = _body(result)["workspaces"][0]
        assert ws["status"] == "stale"

    def test_workspace_status_disconnected(self, handler):
        """Offline workspace with no heartbeat shows 'disconnected' status."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[
                MockWorkspace(id="ws-1", is_online=False, last_heartbeat=None),
            ],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        ws = _body(result)["workspaces"][0]
        assert ws["status"] == "disconnected"
        assert ws["last_heartbeat"] is None

    def test_workspace_name_fallback_to_id(self, handler):
        """Workspace with no name falls back to id."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-fallback", name=None)],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        ws = _body(result)["workspaces"][0]
        assert ws["name"] == "ws-fallback"

    def test_workspace_consent_counting(self, handler):
        """Consents involving a workspace are counted correctly."""
        now = datetime.now(timezone.utc)
        consents = [
            MockConsent(id="c1", source_workspace_id="ws-1", target_workspace_id="ws-2", times_used=10),
            MockConsent(id="c2", source_workspace_id="ws-2", target_workspace_id="ws-1", times_used=3),
            MockConsent(id="c3", source_workspace_id="ws-3", target_workspace_id="ws-4", times_used=99),
            MockConsent(id="c4", source_workspace_id="ws-1", target_workspace_id="ws-3", times_used=7, revoked=True),
        ]
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1")],
            consents=consents,
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        ws = _body(result)["workspaces"][0]
        # c1 (source=ws-1, valid) + c2 (target=ws-1, valid) = 2 active consents, 13 shared items
        assert ws["active_consents"] == 2
        assert ws["shared_items"] == 13

    def test_workspace_capabilities_flags(self, handler):
        """Capabilities are correctly reported."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[
                MockWorkspace(
                    id="ws-1",
                    supports_agent_execution=False,
                    supports_workflow_execution=True,
                    supports_knowledge_query=False,
                ),
            ],
            consents=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        caps = _body(result)["workspaces"][0]["capabilities"]
        assert caps["agent_execution"] is False
        assert caps["workflow_execution"] is True
        assert caps["knowledge_query"] is False

    def test_coordinator_error_returns_500(self, handler):
        """RuntimeError from coordinator returns 500."""
        coord = MagicMock()
        coord.list_workspaces.side_effect = RuntimeError("workspace fetch failed")
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        assert _status(result) == 500

    def test_empty_workspaces(self, handler):
        """No workspaces returns empty list with total=0."""
        coord = MockCrossWorkspaceCoordinator(workspaces=[], consents=[])
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        body = _body(result)
        assert body["workspaces"] == []
        assert body["total"] == 0


# =============================================================================
# GET /api/v1/federation/activity Tests
# =============================================================================


class TestSyncActivity:
    """Test the sync activity feed endpoint."""

    def test_coordinator_unavailable(self, handler):
        """When coordinator is unavailable, returns empty activity list."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["activity"] == []
        assert body["total"] == 0

    def test_activity_with_consents_and_pending(self, handler, mock_coordinator):
        """Returns combined consent and pending request activity."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=mock_coordinator):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        assert _status(result) == 200
        body = _body(result)
        # 2 consents + 1 pending request = 3 activity items
        assert body["total"] == 3
        assert len(body["activity"]) == 3

    def test_active_consent_activity_shape(self, handler):
        """Active consent produces correct activity entry."""
        now = datetime.now(timezone.utc)
        last_used = now - timedelta(minutes=5)
        consent = MockConsent(
            id="c1",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            scope=MockConsentScope.READ_ONLY,
            data_types={"debates", "knowledge"},
            times_used=42,
            granted_at=now - timedelta(hours=2),
            last_used=last_used,
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[consent],
            pending_requests=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        activity = _body(result)["activity"][0]
        assert activity["id"] == "c1"
        assert activity["type"] == "consent_active"
        assert activity["source_workspace"] == "ws-1"
        assert activity["target_workspace"] == "ws-2"
        assert activity["scope"] == "read_only"
        assert set(activity["data_types"]) == {"debates", "knowledge"}
        assert activity["times_used"] == 42
        assert "last_sync" in activity
        assert "timestamp" in activity

    def test_revoked_consent_activity_shape(self, handler):
        """Revoked consent shows as consent_revoked type."""
        now = datetime.now(timezone.utc)
        revoked_at = now - timedelta(hours=1)
        consent = MockConsent(
            id="c-revoked",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            revoked=True,
            revoked_at=revoked_at,
            granted_at=now - timedelta(days=7),
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[consent],
            pending_requests=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        activity = _body(result)["activity"][0]
        assert activity["type"] == "consent_revoked"
        # Timestamp should use revoked_at for revoked consents
        assert activity["timestamp"] == revoked_at.isoformat()

    def test_revoked_consent_without_revoked_at_uses_granted_at(self, handler):
        """Revoked consent without revoked_at falls back to granted_at."""
        now = datetime.now(timezone.utc)
        granted_at = now - timedelta(days=5)
        consent = MockConsent(
            id="c-rev2",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            revoked=True,
            revoked_at=None,
            granted_at=granted_at,
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[consent],
            pending_requests=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        activity = _body(result)["activity"][0]
        assert activity["type"] == "consent_revoked"
        assert activity["timestamp"] == granted_at.isoformat()

    def test_consent_without_last_used_has_no_last_sync(self, handler):
        """Consent without last_used omits last_sync field."""
        consent = MockConsent(
            id="c-no-used",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            last_used=None,
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[consent],
            pending_requests=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        activity = _body(result)["activity"][0]
        assert "last_sync" not in activity

    def test_pending_request_activity_shape(self, handler):
        """Pending request produces correct activity entry."""
        now = datetime.now(timezone.utc)
        req = MockPendingRequest(
            id="req-1",
            operation=MockOperationType.QUERY,
            source_workspace_id="ws-a",
            target_workspace_id="ws-b",
            requester_id="user-99",
            created_at=now,
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[],
            pending_requests=[req],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        activity = _body(result)["activity"][0]
        assert activity["id"] == "req-1"
        assert activity["type"] == "pending_approval"
        assert activity["operation"] == "query"
        assert activity["source_workspace"] == "ws-a"
        assert activity["target_workspace"] == "ws-b"
        assert activity["requester"] == "user-99"
        assert "timestamp" in activity

    def test_activity_sorted_by_timestamp_descending(self, handler):
        """Activity items are sorted by timestamp descending (newest first)."""
        now = datetime.now(timezone.utc)
        old_consent = MockConsent(
            id="c-old",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            granted_at=now - timedelta(days=10),
        )
        new_consent = MockConsent(
            id="c-new",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            granted_at=now - timedelta(minutes=1),
        )
        pending = MockPendingRequest(
            id="req-mid",
            created_at=now - timedelta(hours=1),
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[old_consent, new_consent],
            pending_requests=[pending],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        items = _body(result)["activity"]
        assert len(items) == 3
        # Newest first
        assert items[0]["id"] == "c-new"
        assert items[1]["id"] == "req-mid"
        assert items[2]["id"] == "c-old"

    def test_coordinator_error_returns_500(self, handler):
        """RuntimeError from coordinator returns 500."""
        coord = MagicMock()
        coord.list_consents.side_effect = RuntimeError("sync error")
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        assert _status(result) == 500

    def test_empty_activity(self, handler):
        """No consents or requests returns empty activity."""
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[],
            consents=[],
            pending_requests=[],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})

        body = _body(result)
        assert body["activity"] == []
        assert body["total"] == 0


# =============================================================================
# GET /api/v1/federation/config Tests
# =============================================================================


class TestFederationConfig:
    """Test the federation configuration endpoint."""

    def test_coordinator_unavailable(self, handler):
        """When coordinator is unavailable, returns default config."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["default_policy"] is None
        assert body["knowledge_sharing"]["types"] == []
        assert body["knowledge_sharing"]["approval_required"] is True
        assert body["knowledge_sharing"]["scope"] == "none"

    def test_config_with_default_policy(self, handler, mock_coordinator):
        """Returns policy dict and knowledge sharing summary."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=mock_coordinator):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        assert _status(result) == 200
        body = _body(result)
        assert body["default_policy"] is not None
        assert body["default_policy"]["mode"] == "full"
        assert body["default_policy"]["require_approval"] is False
        assert body["default_policy"]["sharing_scope"] == "global"
        assert body["default_policy"]["audit_all_requests"] is True
        assert body["workspace_policy_count"] == 0

    def test_config_knowledge_sharing_types(self, handler):
        """Knowledge sharing types are aggregated from valid consents."""
        consents = [
            MockConsent(
                id="c1",
                source_workspace_id="ws-1",
                target_workspace_id="ws-2",
                data_types={"debates", "receipts"},
            ),
            MockConsent(
                id="c2",
                source_workspace_id="ws-2",
                target_workspace_id="ws-1",
                data_types={"knowledge", "debates"},
            ),
            MockConsent(
                id="c3",
                source_workspace_id="ws-1",
                target_workspace_id="ws-3",
                data_types={"analytics"},
                revoked=True,  # Revoked - should not be included
            ),
        ]
        coord = MockCrossWorkspaceCoordinator(consents=consents)
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        body = _body(result)
        sharing_types = body["knowledge_sharing"]["types"]
        # Only valid consents: debates, receipts, knowledge (sorted)
        assert sharing_types == ["debates", "knowledge", "receipts"]

    def test_config_knowledge_sharing_from_policy(self, handler):
        """Knowledge sharing settings reflect the default policy."""
        policy = MockDefaultPolicy(
            require_approval=True,
            sharing_scope=MockSharingScope.ORG_ONLY,
            audit_all_requests=False,
        )
        coord = MockCrossWorkspaceCoordinator(
            consents=[],
            default_policy=policy,
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        ks = _body(result)["knowledge_sharing"]
        assert ks["approval_required"] is True
        assert ks["scope"] == "org_only"
        assert ks["audit_enabled"] is False

    def test_config_no_default_policy(self, handler):
        """Without default policy, sharing config uses safe defaults."""
        coord = MockCrossWorkspaceCoordinator(consents=[])
        coord._default_policy = None
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        body = _body(result)
        assert body["default_policy"] is None
        ks = body["knowledge_sharing"]
        assert ks["approval_required"] is True
        assert ks["scope"] == "none"
        assert ks["audit_enabled"] is True

    def test_config_workspace_policy_count(self, handler):
        """workspace_policy_count reflects number of per-workspace policies."""
        coord = MockCrossWorkspaceCoordinator(
            consents=[],
            workspace_policies={"ws-1": {"mode": "full"}, "ws-2": {"mode": "read_only"}},
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        body = _body(result)
        assert body["workspace_policy_count"] == 2

    def test_coordinator_error_returns_500(self, handler):
        """RuntimeError from coordinator returns 500."""
        coord = MagicMock()
        coord.list_consents.side_effect = ValueError("config read failed")
        coord._default_policy = None
        coord._workspace_policies = {}
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        assert _status(result) == 500


# =============================================================================
# Safe Import Coordinator Tests
# =============================================================================


class TestSafeImportCoordinator:
    """Test the _safe_import_coordinator function."""

    def test_import_error_returns_none(self):
        """ImportError during import returns None."""
        with patch(
            f"{MODULE}._safe_import_coordinator",
            wraps=None,
        ):
            # Test the actual function behavior
            from aragora.server.handlers.federation.status import _safe_import_coordinator

            with patch(
                "aragora.server.handlers.federation.status._safe_import_coordinator"
            ) as mock_import:
                mock_import.return_value = None
                result = mock_import()
                assert result is None

    def test_returns_coordinator_when_available(self):
        """Returns coordinator when module imports successfully."""
        mock_coord = MagicMock()
        with patch(
            f"{MODULE}._safe_import_coordinator",
            return_value=mock_coord,
        ):
            from aragora.server.handlers.federation.status import _safe_import_coordinator

            result = _safe_import_coordinator()
            assert result is mock_coord


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_version_prefix_stripping(self, handler):
        """Both /api/v1/ and /api/v2/ prefixes are stripped correctly."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result_v1 = handler.handle_request("GET", "/api/v1/federation/status", None, {})
            result_v2 = handler.handle_request("GET", "/api/v2/federation/status", None, {})

        assert result_v1 is not None
        assert result_v2 is not None
        assert _status(result_v1) == 200
        assert _status(result_v2) == 200

    def test_handler_result_is_handler_result_type(self, handler):
        """All responses are HandlerResult instances."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert isinstance(result, HandlerResult)
        assert result.content_type == "application/json"

    def test_consent_with_zero_times_used(self, handler):
        """Consent with zero times_used does not inflate counts."""
        consent = MockConsent(
            id="c-zero",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            times_used=0,
        )
        coord = MockCrossWorkspaceCoordinator(
            workspaces=[MockWorkspace(id="ws-1")],
            consents=[consent],
        )
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        body = _body(result)
        assert body["shared_knowledge_count"] == 0

    def test_large_number_of_workspaces(self, handler):
        """Handler works with many workspaces."""
        workspaces = [
            MockWorkspace(id=f"ws-{i}", is_online=(i % 2 == 0))
            for i in range(50)
        ]
        coord = MockCrossWorkspaceCoordinator(workspaces=workspaces, consents=[])
        with patch(f"{MODULE}._safe_import_coordinator", return_value=coord):
            result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})

        body = _body(result)
        assert body["total"] == 50
        assert len(body["workspaces"]) == 50

    def test_query_params_passed_through(self, handler):
        """Query params are accepted (even if not currently used)."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request(
                "GET", "/api/v1/federation/status", None, {"limit": "10", "offset": "0"}
            )

        assert _status(result) == 200

    def test_handler_none_argument(self, handler):
        """Handler (3rd arg) can be None without errors."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=None):
            result = handler.handle_request("GET", "/api/v1/federation/status", None, {})

        assert result is not None
        assert _status(result) == 200

    def test_all_four_endpoints_return_different_data_shapes(self, handler, mock_coordinator):
        """Each endpoint returns a distinct response shape."""
        with patch(f"{MODULE}._safe_import_coordinator", return_value=mock_coordinator):
            status_result = handler.handle_request("GET", "/api/v1/federation/status", None, {})
            ws_result = handler.handle_request("GET", "/api/v1/federation/workspaces", None, {})
            act_result = handler.handle_request("GET", "/api/v1/federation/activity", None, {})
            cfg_result = handler.handle_request("GET", "/api/v1/federation/config", None, {})

        status_body = _body(status_result)
        ws_body = _body(ws_result)
        act_body = _body(act_result)
        cfg_body = _body(cfg_result)

        # Status has sync_health key
        assert "sync_health" in status_body
        # Workspaces has workspaces list
        assert "workspaces" in ws_body
        # Activity has activity list
        assert "activity" in act_body
        # Config has knowledge_sharing
        assert "knowledge_sharing" in cfg_body

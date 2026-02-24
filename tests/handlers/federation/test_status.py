"""Tests for FederationStatusHandler.

Covers all routes and behaviour of the FederationStatusHandler class:
- GET /api/federation/status      - Federation overview
- GET /api/federation/workspaces  - Connected workspace list
- GET /api/federation/activity    - Recent sync activity feed
- GET /api/federation/config      - Federation configuration

Tests cover both the coordinator-available and coordinator-unavailable
code paths, plus error handling and edge cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.federation.status import FederationStatusHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _get_status_code(result) -> int:
    """Extract HTTP status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return 200


# ---------------------------------------------------------------------------
# Mock enums — match the production enum values
# ---------------------------------------------------------------------------


class MockSharingScope(str, Enum):
    NONE = "none"
    METADATA = "metadata"
    SUMMARY = "summary"
    FULL = "full"


class MockFederationMode(str, Enum):
    ISOLATED = "isolated"
    READONLY = "readonly"
    BIDIRECTIONAL = "bidirectional"


class MockOperationType(str, Enum):
    READ_KNOWLEDGE = "read_knowledge"
    QUERY_MOUND = "query_mound"
    EXECUTE_AGENT = "execute_agent"


# ---------------------------------------------------------------------------
# Mock data classes
# ---------------------------------------------------------------------------


@dataclass
class MockWorkspace:
    id: str = "ws-1"
    name: str = "Workspace One"
    org_id: str = "org-1"
    is_online: bool = True
    federation_mode: MockFederationMode = MockFederationMode.BIDIRECTIONAL
    last_heartbeat: datetime | None = None
    latency_ms: float = 42.0
    supports_agent_execution: bool = True
    supports_workflow_execution: bool = True
    supports_knowledge_query: bool = True


@dataclass
class MockConsent:
    id: str = "consent-1"
    source_workspace_id: str = "ws-1"
    target_workspace_id: str = "ws-2"
    scope: MockSharingScope = MockSharingScope.FULL
    data_types: set = field(default_factory=lambda: {"debates", "findings"})
    revoked: bool = False
    revoked_at: datetime | None = None
    granted_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    times_used: int = 10
    last_used: datetime | None = None

    def is_valid(self) -> bool:
        return not self.revoked


@dataclass
class MockPendingRequest:
    id: str = "req-1"
    operation: MockOperationType = MockOperationType.READ_KNOWLEDGE
    source_workspace_id: str = "ws-1"
    target_workspace_id: str = "ws-2"
    requester_id: str = "user-1"
    created_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 20, tzinfo=timezone.utc)
    )


@dataclass
class MockPolicy:
    mode: MockFederationMode = MockFederationMode.BIDIRECTIONAL
    sharing_scope: MockSharingScope = MockSharingScope.FULL
    require_approval: bool = False
    audit_all_requests: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "sharing_scope": self.sharing_scope.value,
            "require_approval": self.require_approval,
            "audit_all_requests": self.audit_all_requests,
        }


# ---------------------------------------------------------------------------
# Coordinator factory helper
# ---------------------------------------------------------------------------


def _make_coordinator(
    workspaces: list | None = None,
    consents: list | None = None,
    pending_requests: list | None = None,
    policy: MockPolicy | None = None,
    stats: dict | None = None,
) -> MagicMock:
    """Create a mock coordinator with sensible defaults."""
    coord = MagicMock()
    coord.list_workspaces.return_value = workspaces or []
    coord.list_consents.return_value = consents or []
    coord.list_pending_requests.return_value = pending_requests or []
    coord._default_policy = policy
    coord._workspace_policies = {}
    coord.get_stats.return_value = stats or {
        "total_workspaces": len(workspaces or []),
        "total_consents": len(consents or []),
        "valid_consents": sum(1 for c in (consents or []) if c.is_valid()),
        "pending_requests": len(pending_requests or []),
        "registered_handlers": [],
    }
    return coord


def _patch_coordinator(coordinator):
    """Patch the safe import to return the given coordinator."""
    return patch(
        "aragora.server.handlers.federation.status._safe_import_coordinator",
        return_value=coordinator,
    )


def _patch_no_coordinator():
    """Patch the safe import to return None (coordinator unavailable)."""
    return patch(
        "aragora.server.handlers.federation.status._safe_import_coordinator",
        return_value=None,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return FederationStatusHandler({})


# ===========================================================================
# GET /api/federation/status
# ===========================================================================


class TestFederationStatus:
    """Test _handle_federation_status endpoint."""

    def test_unavailable_when_no_coordinator(self, handler):
        with _patch_no_coordinator():
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["status"] == "unavailable"
        assert body["connected_workspaces"] == 0
        assert body["sync_health"] == "offline"
        assert body["federation_mode"] == "isolated"

    def test_idle_when_no_workspaces(self, handler):
        coord = _make_coordinator()
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["status"] == "idle"
        assert body["connected_workspaces"] == 0
        assert body["sync_health"] == "idle"

    def test_active_with_online_workspaces(self, handler):
        ws1 = MockWorkspace(id="ws-1", is_online=True)
        ws2 = MockWorkspace(id="ws-2", is_online=True)
        consent = MockConsent(times_used=5)
        policy = MockPolicy(mode=MockFederationMode.BIDIRECTIONAL)

        coord = _make_coordinator(
            workspaces=[ws1, ws2],
            consents=[consent],
            policy=policy,
        )
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["status"] == "active"
        assert body["connected_workspaces"] == 2
        assert body["online_workspaces"] == 2
        assert body["sync_health"] == "healthy"
        assert body["shared_knowledge_count"] == 5
        assert body["federation_mode"] == "bidirectional"
        assert "timestamp" in body

    def test_degraded_health_when_some_offline(self, handler):
        ws1 = MockWorkspace(id="ws-1", is_online=True)
        ws2 = MockWorkspace(id="ws-2", is_online=False)
        coord = _make_coordinator(workspaces=[ws1, ws2])
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["sync_health"] == "degraded"
        assert body["online_workspaces"] == 1

    def test_offline_health_when_all_offline(self, handler):
        ws1 = MockWorkspace(id="ws-1", is_online=False)
        ws2 = MockWorkspace(id="ws-2", is_online=False)
        coord = _make_coordinator(workspaces=[ws1, ws2])
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["sync_health"] == "offline"
        assert body["online_workspaces"] == 0

    def test_revoked_consents_excluded_from_shared_count(self, handler):
        c1 = MockConsent(id="c1", times_used=10, revoked=False)
        c2 = MockConsent(id="c2", times_used=20, revoked=True)
        coord = _make_coordinator(consents=[c1, c2])
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        # Only c1 is valid, so shared_knowledge_count = 10
        assert body["shared_knowledge_count"] == 10

    def test_no_policy_defaults_to_isolated(self, handler):
        ws1 = MockWorkspace(id="ws-1")
        coord = _make_coordinator(workspaces=[ws1], policy=None)
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["federation_mode"] == "isolated"

    def test_stats_fields_forwarded(self, handler):
        coord = _make_coordinator(
            stats={
                "total_workspaces": 3,
                "total_consents": 5,
                "valid_consents": 4,
                "pending_requests": 2,
                "registered_handlers": ["read_knowledge", "query_mound"],
            }
        )
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        body = _parse_response(result)
        assert body["valid_consents"] == 4
        assert body["pending_requests"] == 2
        assert body["registered_handlers"] == ["read_knowledge", "query_mound"]

    def test_runtime_error_returns_500(self, handler):
        coord = MagicMock()
        coord.get_stats.side_effect = RuntimeError("db down")
        with _patch_coordinator(coord):
            result = handler._handle_federation_status({})

        assert _get_status_code(result) == 500


# ===========================================================================
# GET /api/federation/workspaces
# ===========================================================================


class TestListWorkspaces:
    """Test _handle_list_workspaces endpoint."""

    def test_empty_when_no_coordinator(self, handler):
        with _patch_no_coordinator():
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_single_online_workspace(self, handler):
        ws = MockWorkspace(
            id="ws-1",
            name="Dev Workspace",
            org_id="org-dev",
            is_online=True,
            latency_ms=15.5,
        )
        coord = _make_coordinator(workspaces=[ws])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        assert body["total"] == 1
        ws_data = body["workspaces"][0]
        assert ws_data["id"] == "ws-1"
        assert ws_data["name"] == "Dev Workspace"
        assert ws_data["org_id"] == "org-dev"
        assert ws_data["status"] == "connected"
        assert ws_data["is_online"] is True
        assert ws_data["latency_ms"] == 15.5
        assert ws_data["capabilities"]["agent_execution"] is True
        assert ws_data["capabilities"]["workflow_execution"] is True
        assert ws_data["capabilities"]["knowledge_query"] is True

    def test_offline_workspace_with_heartbeat_shows_stale(self, handler):
        ws = MockWorkspace(
            id="ws-1",
            is_online=False,
            last_heartbeat=datetime(2026, 2, 20, tzinfo=timezone.utc),
        )
        coord = _make_coordinator(workspaces=[ws])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        ws_data = body["workspaces"][0]
        assert ws_data["status"] == "stale"
        assert ws_data["last_heartbeat"] is not None

    def test_offline_workspace_no_heartbeat_shows_disconnected(self, handler):
        ws = MockWorkspace(id="ws-1", is_online=False, last_heartbeat=None)
        coord = _make_coordinator(workspaces=[ws])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        assert body["workspaces"][0]["status"] == "disconnected"
        assert body["workspaces"][0]["last_heartbeat"] is None

    def test_shared_items_counted_from_consents(self, handler):
        ws = MockWorkspace(id="ws-1")
        c1 = MockConsent(
            id="c1",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            times_used=10,
        )
        c2 = MockConsent(
            id="c2",
            source_workspace_id="ws-3",
            target_workspace_id="ws-1",
            times_used=5,
        )
        # c3 involves different workspaces — should not count for ws-1
        c3 = MockConsent(
            id="c3",
            source_workspace_id="ws-3",
            target_workspace_id="ws-4",
            times_used=99,
        )
        coord = _make_coordinator(workspaces=[ws], consents=[c1, c2, c3])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        ws_data = body["workspaces"][0]
        assert ws_data["shared_items"] == 15  # 10 + 5
        assert ws_data["active_consents"] == 2

    def test_revoked_consents_excluded_from_workspace_counts(self, handler):
        ws = MockWorkspace(id="ws-1")
        c_valid = MockConsent(
            id="c1",
            source_workspace_id="ws-1",
            target_workspace_id="ws-2",
            times_used=10,
        )
        c_revoked = MockConsent(
            id="c2",
            source_workspace_id="ws-1",
            target_workspace_id="ws-3",
            times_used=50,
            revoked=True,
        )
        coord = _make_coordinator(workspaces=[ws], consents=[c_valid, c_revoked])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        ws_data = body["workspaces"][0]
        assert ws_data["shared_items"] == 10  # Only valid consent
        assert ws_data["active_consents"] == 1

    def test_multiple_workspaces(self, handler):
        ws1 = MockWorkspace(id="ws-1", name="WS1", is_online=True)
        ws2 = MockWorkspace(id="ws-2", name="WS2", is_online=False)
        coord = _make_coordinator(workspaces=[ws1, ws2])
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        body = _parse_response(result)
        assert body["total"] == 2
        ids = {w["id"] for w in body["workspaces"]}
        assert ids == {"ws-1", "ws-2"}

    def test_runtime_error_returns_500(self, handler):
        coord = MagicMock()
        coord.list_workspaces.side_effect = RuntimeError("network error")
        with _patch_coordinator(coord):
            result = handler._handle_list_workspaces({})

        assert _get_status_code(result) == 500


# ===========================================================================
# GET /api/federation/activity
# ===========================================================================


class TestSyncActivity:
    """Test _handle_sync_activity endpoint."""

    def test_empty_when_no_coordinator(self, handler):
        with _patch_no_coordinator():
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        assert body["activity"] == []
        assert body["total"] == 0

    def test_active_consent_activity(self, handler):
        consent = MockConsent(
            id="c1",
            revoked=False,
            scope=MockSharingScope.FULL,
            data_types={"debates", "findings"},
            times_used=15,
            granted_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
            last_used=datetime(2026, 2, 20, tzinfo=timezone.utc),
        )
        coord = _make_coordinator(consents=[consent])
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        assert body["total"] == 1
        event = body["activity"][0]
        assert event["id"] == "c1"
        assert event["type"] == "consent_active"
        assert event["scope"] == "full"
        assert set(event["data_types"]) == {"debates", "findings"}
        assert event["times_used"] == 15
        assert "last_sync" in event
        assert event["timestamp"] == "2026-01-10T00:00:00+00:00"

    def test_revoked_consent_activity(self, handler):
        consent = MockConsent(
            id="c2",
            revoked=True,
            revoked_at=datetime(2026, 2, 15, tzinfo=timezone.utc),
            granted_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        )
        coord = _make_coordinator(consents=[consent])
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        event = body["activity"][0]
        assert event["type"] == "consent_revoked"
        # Timestamp uses revoked_at when revoked
        assert event["timestamp"] == "2026-02-15T00:00:00+00:00"

    def test_consent_no_last_used_omits_last_sync(self, handler):
        consent = MockConsent(id="c1", last_used=None)
        coord = _make_coordinator(consents=[consent])
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        assert "last_sync" not in body["activity"][0]

    def test_pending_request_activity(self, handler):
        req = MockPendingRequest(
            id="req-1",
            operation=MockOperationType.READ_KNOWLEDGE,
            source_workspace_id="ws-a",
            target_workspace_id="ws-b",
            requester_id="admin-user",
            created_at=datetime(2026, 2, 22, tzinfo=timezone.utc),
        )
        coord = _make_coordinator(pending_requests=[req])
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        assert body["total"] == 1
        event = body["activity"][0]
        assert event["id"] == "req-1"
        assert event["type"] == "pending_approval"
        assert event["operation"] == "read_knowledge"
        assert event["source_workspace"] == "ws-a"
        assert event["target_workspace"] == "ws-b"
        assert event["requester"] == "admin-user"

    def test_mixed_activity_sorted_by_timestamp_desc(self, handler):
        early = MockConsent(
            id="early",
            granted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        late = MockConsent(
            id="late",
            granted_at=datetime(2026, 2, 28, tzinfo=timezone.utc),
        )
        mid = MockPendingRequest(
            id="mid",
            created_at=datetime(2026, 2, 10, tzinfo=timezone.utc),
        )
        coord = _make_coordinator(
            consents=[early, late],
            pending_requests=[mid],
        )
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        body = _parse_response(result)
        assert body["total"] == 3
        ids = [e["id"] for e in body["activity"]]
        assert ids == ["late", "mid", "early"]

    def test_runtime_error_returns_500(self, handler):
        coord = MagicMock()
        coord.list_consents.side_effect = RuntimeError("storage failure")
        with _patch_coordinator(coord):
            result = handler._handle_sync_activity({})

        assert _get_status_code(result) == 500


# ===========================================================================
# GET /api/federation/config
# ===========================================================================


class TestFederationConfig:
    """Test _handle_federation_config endpoint."""

    def test_defaults_when_no_coordinator(self, handler):
        with _patch_no_coordinator():
            result = handler._handle_federation_config({})

        body = _parse_response(result)
        assert body["default_policy"] is None
        assert body["knowledge_sharing"]["types"] == []
        assert body["knowledge_sharing"]["approval_required"] is True
        assert body["knowledge_sharing"]["scope"] == "none"

    def test_config_with_policy(self, handler):
        policy = MockPolicy(
            mode=MockFederationMode.BIDIRECTIONAL,
            sharing_scope=MockSharingScope.FULL,
            require_approval=False,
            audit_all_requests=True,
        )
        consent = MockConsent(
            data_types={"debates", "findings"},
            revoked=False,
        )
        coord = _make_coordinator(policy=policy, consents=[consent])
        with _patch_coordinator(coord):
            result = handler._handle_federation_config({})

        body = _parse_response(result)
        assert body["default_policy"] is not None
        assert body["default_policy"]["mode"] == "bidirectional"
        assert body["default_policy"]["sharing_scope"] == "full"
        assert body["knowledge_sharing"]["approval_required"] is False
        assert body["knowledge_sharing"]["scope"] == "full"
        assert body["knowledge_sharing"]["audit_enabled"] is True
        assert set(body["knowledge_sharing"]["types"]) == {"debates", "findings"}

    def test_revoked_consents_excluded_from_sharing_types(self, handler):
        policy = MockPolicy()
        c_valid = MockConsent(data_types={"debates"}, revoked=False)
        c_revoked = MockConsent(data_types={"secrets"}, revoked=True)
        coord = _make_coordinator(
            policy=policy, consents=[c_valid, c_revoked]
        )
        with _patch_coordinator(coord):
            result = handler._handle_federation_config({})

        body = _parse_response(result)
        assert "secrets" not in body["knowledge_sharing"]["types"]
        assert "debates" in body["knowledge_sharing"]["types"]

    def test_workspace_policy_count(self, handler):
        policy = MockPolicy()
        coord = _make_coordinator(policy=policy)
        coord._workspace_policies = {"ws-1": MockPolicy(), "ws-2": MockPolicy()}
        with _patch_coordinator(coord):
            result = handler._handle_federation_config({})

        body = _parse_response(result)
        assert body["workspace_policy_count"] == 2

    def test_no_policy_approval_defaults_true(self, handler):
        coord = _make_coordinator(policy=None)
        with _patch_coordinator(coord):
            result = handler._handle_federation_config({})

        body = _parse_response(result)
        assert body["knowledge_sharing"]["approval_required"] is True
        assert body["knowledge_sharing"]["scope"] == "none"

    def test_runtime_error_returns_500(self, handler):
        coord = MagicMock()
        coord.list_consents.side_effect = RuntimeError("config read failed")
        coord._default_policy = MockPolicy()
        with _patch_coordinator(coord):
            result = handler._handle_federation_config({})

        assert _get_status_code(result) == 500


# ===========================================================================
# Routing
# ===========================================================================


class TestRouting:
    """Test handle_request routing logic."""

    def test_routes_federation_status(self, handler):
        with _patch_no_coordinator():
            result = handler.handle_request(
                "GET", "/api/v1/federation/status", MagicMock(), {}
            )

        body = _parse_response(result)
        assert body["status"] == "unavailable"

    def test_routes_federation_workspaces(self, handler):
        with _patch_no_coordinator():
            result = handler.handle_request(
                "GET", "/api/v1/federation/workspaces", MagicMock(), {}
            )

        body = _parse_response(result)
        assert body["workspaces"] == []

    def test_routes_federation_activity(self, handler):
        with _patch_no_coordinator():
            result = handler.handle_request(
                "GET", "/api/v1/federation/activity", MagicMock(), {}
            )

        body = _parse_response(result)
        assert body["activity"] == []

    def test_routes_federation_config(self, handler):
        with _patch_no_coordinator():
            result = handler.handle_request(
                "GET", "/api/v1/federation/config", MagicMock(), {}
            )

        body = _parse_response(result)
        assert body["default_policy"] is None

    def test_unversioned_path_also_routes(self, handler):
        with _patch_no_coordinator():
            result = handler.handle_request(
                "GET", "/api/federation/status", MagicMock(), {}
            )

        body = _parse_response(result)
        assert body["status"] == "unavailable"

    def test_unknown_path_returns_none(self, handler):
        result = handler.handle_request(
            "GET", "/api/v1/federation/unknown", MagicMock(), {}
        )
        assert result is None

    def test_post_method_returns_none(self, handler):
        result = handler.handle_request(
            "POST", "/api/v1/federation/status", MagicMock(), {}
        )
        assert result is None

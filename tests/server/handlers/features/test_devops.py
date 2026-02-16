"""
Tests for DevOps Incident Management Handler.

Tests cover:
- can_handle routing
- Permission checks (auth, RBAC)
- PagerDuty webhook handling
- Incident lifecycle (create, list, get, acknowledge, resolve, reassign, merge)
- Notes (list, add)
- On-call schedules
- Services (list, get)
- Status endpoint
- Error handling / connector unavailable
- Module-level state cleanup
- Utility methods
"""

import sys
import types as _types_mod

# Pre-stub Slack modules to avoid circular ImportError
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from aragora.server.handlers.features.devops import (
    DevOpsHandler,
    DevOpsCircuitBreaker,
    create_devops_handler,
    _connector_instances,
    _active_contexts,
    get_pagerduty_connector,
    get_devops_circuit_breaker,
    get_devops_circuit_breaker_status,
    _clear_devops_components,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result):
    """Parse a HandlerResult dataclass into (body_dict, status_code)."""
    body = json.loads(result.body) if result.body else {}
    return body, result.status_code


def _make_request(
    *,
    tenant_id="test-tenant",
    query=None,
    json_body=None,
    raw_body=b"",
    headers=None,
):
    """Build a fake request object."""
    req = SimpleNamespace()
    req.tenant_id = tenant_id
    req.query = query or {}
    req.headers = headers or {}
    req.body = raw_body

    if json_body is not None:

        async def _json():
            return json_body

        req.json = _json
    else:

        async def _json():
            return {}

        req.json = _json

    return req


def _make_incident(
    id="PDINC001234",  # Valid PagerDuty ID format (7-15 uppercase alphanumeric)
    title="Test Incident",
    status_val="triggered",
    urgency_val="high",
    service_id="PDSVC001234",  # Valid PagerDuty ID format
    service_name="Test Service",
    incident_number=1,
    created_at=None,
    html_url="https://pd.example.com/incidents/PDINC001234",
    description="A test incident",
    assignees=None,
    priority=None,
):
    inc = SimpleNamespace()
    inc.id = id
    inc.title = title
    inc.status = SimpleNamespace(value=status_val)
    inc.urgency = SimpleNamespace(value=urgency_val)
    inc.service_id = service_id
    inc.service_name = service_name
    inc.incident_number = incident_number
    inc.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    inc.html_url = html_url
    inc.description = description
    inc.assignees = assignees or []
    inc.priority = priority
    return inc


def _make_note(id="NOTE001", content="investigation note", created_at=None, user=None):
    note = SimpleNamespace()
    note.id = id
    note.content = content
    note.created_at = created_at or datetime(2025, 1, 2, tzinfo=timezone.utc)
    note.user = user
    return note


def _make_service(
    id="PDSVC001234",  # Valid PagerDuty ID format
    name="Test Service",
    description="A test service",
    status_val="active",
    html_url="https://pd.example.com/services/PDSVC001234",
    escalation_policy_id="PDESCPOL001",  # Valid PagerDuty ID format
    created_at=None,
):
    svc = SimpleNamespace()
    svc.id = id
    svc.name = name
    svc.description = description
    svc.status = SimpleNamespace(value=status_val)
    svc.html_url = html_url
    svc.escalation_policy_id = escalation_policy_id
    svc.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    return svc


def _make_oncall(
    schedule_id="PDSCHED01234",  # Valid PagerDuty ID format
    schedule_name="Primary",
    user_id="PDUSR001234",  # Valid PagerDuty ID format
    user_name="Alice",
    user_email="alice@example.com",
    start=None,
    end=None,
    escalation_level=1,
):
    sched = SimpleNamespace()
    sched.schedule_id = schedule_id
    sched.schedule_name = schedule_name
    sched.user = SimpleNamespace(id=user_id, name=user_name, email=user_email)
    sched.start = start or datetime(2025, 1, 1, tzinfo=timezone.utc)
    sched.end = end or datetime(2025, 1, 8, tzinfo=timezone.utc)
    sched.escalation_level = escalation_level
    return sched


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Clear module-level connector caches and circuit breaker between tests."""
    _clear_devops_components()
    yield
    _clear_devops_components()


@pytest.fixture()
def handler():
    """Create a DevOpsHandler with mocked auth that always succeeds."""
    h = DevOpsHandler(server_context={})
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True

    async def _get_auth(request, require_auth=True):
        return auth_ctx

    h.get_auth_context = _get_auth
    h.check_permission = MagicMock(return_value=True)
    return h


@pytest.fixture()
def mock_connector():
    """Create a mock PagerDuty connector."""
    conn = AsyncMock()
    conn.verify_webhook_signature = MagicMock(return_value=True)
    conn.parse_webhook = MagicMock(return_value=None)
    return conn


def _patch_connector(mock_conn):
    """Shorthand to patch the connector lookup."""
    return patch(
        "aragora.server.handlers.features.devops.get_pagerduty_connector",
        new_callable=AsyncMock,
        return_value=mock_conn,
    )


# ---------------------------------------------------------------------------
# Tests: can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_incidents_path(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/incidents") is True

    def test_incidents_with_id(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/incidents/INC123") is True

    def test_incidents_action(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/incidents/INC123/acknowledge") is True

    def test_oncall_path(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/oncall") is True

    def test_oncall_service_path(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/oncall/services/SVC1") is True

    def test_services_path(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/services") is True

    def test_services_with_id(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/services/SVC1") is True

    def test_webhooks_pagerduty(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/webhooks/pagerduty") is True

    def test_devops_status(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/devops/status") is True

    def test_unrelated_path(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/api/v1/debates") is False

    def test_unrelated_root(self):
        h = DevOpsHandler(server_context={})
        assert h.can_handle("/health") is False


class TestRoutes:
    def test_routes_defined(self):
        assert len(DevOpsHandler.ROUTES) > 0

    def test_expected_routes_present(self):
        expected = [
            "/api/v1/incidents",
            "/api/v1/webhooks/pagerduty",
            "/api/v1/oncall",
            "/api/v1/services",
            "/api/v1/devops/status",
        ]
        for route in expected:
            assert route in DevOpsHandler.ROUTES, f"Expected route: {route}"


# ---------------------------------------------------------------------------
# Tests: Authentication & Permissions
# ---------------------------------------------------------------------------


class TestAuthPermissions:
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        h = DevOpsHandler(server_context={})

        async def _fail_auth(request, require_auth=True):
            raise UnauthorizedError("no token")

        h.get_auth_context = _fail_auth
        req = _make_request()
        result = await h.handle(req, "/api/v1/incidents", "GET")
        _, status = _parse_result(result)
        assert status == 401

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = DevOpsHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        def _deny(ctx, perm, resource_id=None):
            raise ForbiddenError(f"Permission denied: {perm}", permission=perm)

        h.check_permission = _deny
        req = _make_request()
        result = await h.handle(req, "/api/v1/incidents", "GET")
        _, status = _parse_result(result)
        assert status == 403

    @pytest.mark.asyncio
    async def test_webhook_requires_webhook_permission(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = DevOpsHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        checked_permissions = []

        def _check(ctx, perm, resource_id=None):
            checked_permissions.append(perm)
            raise ForbiddenError(f"denied: {perm}", permission=perm)

        h.check_permission = _check
        req = _make_request()
        await h.handle(req, "/api/v1/webhooks/pagerduty", "POST")
        assert "devops:webhook" in checked_permissions

    @pytest.mark.asyncio
    async def test_post_requires_write_permission(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = DevOpsHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        checked_permissions = []

        def _check(ctx, perm, resource_id=None):
            checked_permissions.append(perm)
            raise ForbiddenError(f"denied: {perm}", permission=perm)

        h.check_permission = _check
        req = _make_request()
        await h.handle(req, "/api/v1/incidents", "POST")
        assert "devops:write" in checked_permissions

    @pytest.mark.asyncio
    async def test_get_requires_read_permission(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = DevOpsHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        checked_permissions = []

        def _check(ctx, perm, resource_id=None):
            checked_permissions.append(perm)
            raise ForbiddenError(f"denied: {perm}", permission=perm)

        h.check_permission = _check
        req = _make_request()
        await h.handle(req, "/api/v1/incidents", "GET")
        assert "devops:read" in checked_permissions


# ---------------------------------------------------------------------------
# Tests: Status endpoint (via handle)
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_configured(self, handler):
        req = _make_request()
        with patch.dict(
            "os.environ",
            {
                "PAGERDUTY_API_KEY": "key123",
                "PAGERDUTY_EMAIL": "ops@co.com",
                "PAGERDUTY_WEBHOOK_SECRET": "sec",
            },
        ):
            result = await handler.handle(req, "/api/v1/devops/status", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["configured"] is True
        assert body["data"]["api_key_set"] is True
        assert body["data"]["email_set"] is True
        assert body["data"]["webhook_secret_set"] is True

    @pytest.mark.asyncio
    async def test_status_not_configured(self, handler):
        req = _make_request()
        with patch.dict("os.environ", {}, clear=True):
            result = await handler.handle(req, "/api/v1/devops/status", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["configured"] is False


# ---------------------------------------------------------------------------
# Tests: Connector unavailable (503) via handle for top-level paths
# ---------------------------------------------------------------------------


class TestConnectorUnavailable:
    @pytest.mark.asyncio
    async def test_list_incidents_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler.handle(req, "/api/v1/incidents", "GET")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_create_incident_503(self, handler):
        req = _make_request(json_body={"title": "t", "service_id": "s"})
        with _patch_connector(None):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_get_oncall_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler.handle(req, "/api/v1/oncall", "GET")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_list_services_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler.handle(req, "/api/v1/services", "GET")
        _, status = _parse_result(result)
        assert status == 503


# ---------------------------------------------------------------------------
# Tests: List & Create incidents (via handle -- exact path match)
# ---------------------------------------------------------------------------


class TestListIncidents:
    @pytest.mark.asyncio
    async def test_list_incidents_success(self, handler, mock_connector):
        inc = _make_incident()
        mock_connector.list_incidents = AsyncMock(return_value=([inc], False))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "GET")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["count"] == 1
        assert body["data"]["incidents"][0]["id"] == "PDINC001234"
        assert body["data"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_incidents_with_filters(self, handler, mock_connector):
        mock_connector.list_incidents = AsyncMock(return_value=([], False))

        req = _make_request(
            query={
                "status": "triggered,acknowledged",
                "service_ids": "PDSVC000001,PDSVC000002",
                "urgency": "high",
                "limit": "10",
                "offset": "5",
            }
        )
        with _patch_connector(mock_connector):
            await handler.handle(req, "/api/v1/incidents", "GET")

        mock_connector.list_incidents.assert_called_once_with(
            statuses=["triggered", "acknowledged"],
            service_ids=["PDSVC000001", "PDSVC000002"],
            urgencies=["high"],
            limit=10,
            offset=5,
        )

    @pytest.mark.asyncio
    async def test_list_incidents_error(self, handler, mock_connector):
        mock_connector.list_incidents = AsyncMock(side_effect=RuntimeError("PD down"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "GET")
        _, status = _parse_result(result)
        assert status == 500


class TestCreateIncident:
    @pytest.mark.asyncio
    async def test_create_incident_success(self, handler, mock_connector):
        created = _make_incident(id="PDINC00NEW1", title="DB down")
        mock_connector.create_incident = AsyncMock(return_value=created)

        req = _make_request(
            json_body={
                "title": "DB down",
                "service_id": "PDSVC001234",  # Valid PagerDuty ID format
                "urgency": "high",
                "body": "Database is unreachable",
            }
        )

        with (
            _patch_connector(mock_connector),
            patch("aragora.connectors.devops.pagerduty.IncidentCreateRequest") as mock_req_cls,
            patch("aragora.connectors.devops.pagerduty.IncidentUrgency") as mock_urgency_cls,
        ):
            mock_urgency_cls.return_value = "high"
            mock_req_cls.return_value = MagicMock()
            result = await handler.handle(req, "/api/v1/incidents", "POST")

        body, status = _parse_result(result)
        assert status == 201
        assert body["incident"]["id"] == "PDINC00NEW1"

    @pytest.mark.asyncio
    async def test_create_incident_missing_title(self, handler, mock_connector):
        req = _make_request(json_body={"service_id": "PDSVC001234"})
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_create_incident_missing_service_id(self, handler, mock_connector):
        req = _make_request(json_body={"title": "Something broke"})
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_create_incident_error(self, handler, mock_connector):
        mock_connector.create_incident = AsyncMock(side_effect=RuntimeError("fail"))

        req = _make_request(json_body={"title": "X", "service_id": "PDSVC001234"})  # Valid ID
        with (
            _patch_connector(mock_connector),
            patch("aragora.connectors.devops.pagerduty.IncidentCreateRequest"),
            patch("aragora.connectors.devops.pagerduty.IncidentUrgency"),
        ):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        _, status = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Tests: Incident sub-handlers (direct method calls)
#
# The private handler methods are tested directly to exercise incident
# lifecycle logic (get, acknowledge, resolve, reassign, merge, notes).
# ---------------------------------------------------------------------------


class TestGetIncident:
    @pytest.mark.asyncio
    async def test_get_incident_success(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000042")
        mock_connector.get_incident = AsyncMock(return_value=inc)

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_incident(req, "test-tenant", "PDINC000042")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["incident"]["id"] == "PDINC000042"
        assert body["data"]["incident"]["description"] == "A test incident"

    @pytest.mark.asyncio
    async def test_get_incident_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler._handle_get_incident(req, "test-tenant", "PDINC000042")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_get_incident_error(self, handler, mock_connector):
        mock_connector.get_incident = AsyncMock(side_effect=RuntimeError("not found"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_incident(req, "test-tenant", "PDINC000099")
        _, status = _parse_result(result)
        assert status == 500


class TestAcknowledgeIncident:
    @pytest.mark.asyncio
    async def test_acknowledge_success(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000010", status_val="acknowledged")
        mock_connector.acknowledge_incident = AsyncMock(return_value=inc)

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_acknowledge_incident(req, "test-tenant", "PDINC000010")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["incident"]["status"] == "acknowledged"
        assert body["data"]["message"] == "Incident acknowledged"

    @pytest.mark.asyncio
    async def test_acknowledge_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler._handle_acknowledge_incident(req, "test-tenant", "PDINC000010")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_acknowledge_error(self, handler, mock_connector):
        mock_connector.acknowledge_incident = AsyncMock(side_effect=ConnectionError("err"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_acknowledge_incident(req, "test-tenant", "PDINC000010")
        _, status = _parse_result(result)
        assert status == 500


class TestResolveIncident:
    @pytest.mark.asyncio
    async def test_resolve_success(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000020", status_val="resolved")
        mock_connector.resolve_incident = AsyncMock(return_value=inc)

        req = _make_request(json_body={"resolution": "Restarted the service"})
        with _patch_connector(mock_connector):
            result = await handler._handle_resolve_incident(req, "test-tenant", "PDINC000020")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["incident"]["status"] == "resolved"
        assert body["data"]["message"] == "Incident resolved"
        mock_connector.resolve_incident.assert_called_once_with(
            "PDINC000020", "Restarted the service"
        )

    @pytest.mark.asyncio
    async def test_resolve_without_resolution(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000020", status_val="resolved")
        mock_connector.resolve_incident = AsyncMock(return_value=inc)

        req = _make_request(json_body={})
        with _patch_connector(mock_connector):
            result = await handler._handle_resolve_incident(req, "test-tenant", "PDINC000020")

        body, _ = _parse_result(result)
        assert body["data"]["incident"]["status"] == "resolved"
        mock_connector.resolve_incident.assert_called_once_with("PDINC000020", None)

    @pytest.mark.asyncio
    async def test_resolve_503(self, handler):
        req = _make_request(json_body={})
        with _patch_connector(None):
            result = await handler._handle_resolve_incident(req, "test-tenant", "PDINC000020")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_resolve_error(self, handler, mock_connector):
        mock_connector.resolve_incident = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request(json_body={})
        with _patch_connector(mock_connector):
            result = await handler._handle_resolve_incident(req, "test-tenant", "PDINC000020")
        _, status = _parse_result(result)
        assert status == 500


class TestReassignIncident:
    @pytest.mark.asyncio
    async def test_reassign_success_user_ids(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000030")
        inc.assignees = ["PDUSR000001"]
        mock_connector.reassign_incident = AsyncMock(return_value=inc)

        req = _make_request(json_body={"user_ids": ["PDUSR000001"]})
        with _patch_connector(mock_connector):
            result = await handler._handle_reassign_incident(req, "test-tenant", "PDINC000030")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["message"] == "Incident reassigned"

    @pytest.mark.asyncio
    async def test_reassign_success_escalation_policy(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000030")
        mock_connector.reassign_incident = AsyncMock(return_value=inc)

        req = _make_request(json_body={"escalation_policy_id": "PDESCPOL001"})
        with _patch_connector(mock_connector):
            result = await handler._handle_reassign_incident(req, "test-tenant", "PDINC000030")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["message"] == "Incident reassigned"

    @pytest.mark.asyncio
    async def test_reassign_missing_params(self, handler, mock_connector):
        req = _make_request(json_body={})
        with _patch_connector(mock_connector):
            result = await handler._handle_reassign_incident(req, "test-tenant", "PDINC000030")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_reassign_error(self, handler, mock_connector):
        mock_connector.reassign_incident = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request(json_body={"user_ids": ["PDUSR000001"]})
        with _patch_connector(mock_connector):
            result = await handler._handle_reassign_incident(req, "test-tenant", "PDINC000030")
        _, status = _parse_result(result)
        assert status == 500


class TestMergeIncidents:
    @pytest.mark.asyncio
    async def test_merge_success(self, handler, mock_connector):
        inc = _make_incident(id="PDINC000050", title="Merged")
        mock_connector.merge_incidents = AsyncMock(return_value=inc)

        req = _make_request(json_body={"source_incident_ids": ["PDINC000051", "PDINC000052"]})
        with _patch_connector(mock_connector):
            result = await handler._handle_merge_incidents(req, "test-tenant", "PDINC000050")

        body, status = _parse_result(result)
        assert status == 200
        assert "Merged 2 incidents" in body["data"]["message"]

    @pytest.mark.asyncio
    async def test_merge_missing_source_ids(self, handler, mock_connector):
        req = _make_request(json_body={})
        with _patch_connector(mock_connector):
            result = await handler._handle_merge_incidents(req, "test-tenant", "PDINC000050")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_merge_empty_source_ids(self, handler, mock_connector):
        req = _make_request(json_body={"source_incident_ids": []})
        with _patch_connector(mock_connector):
            result = await handler._handle_merge_incidents(req, "test-tenant", "PDINC000050")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_merge_error(self, handler, mock_connector):
        mock_connector.merge_incidents = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request(json_body={"source_incident_ids": ["PDINC000051"]})
        with _patch_connector(mock_connector):
            result = await handler._handle_merge_incidents(req, "test-tenant", "PDINC000050")
        _, status = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Tests: Notes (direct method calls)
# ---------------------------------------------------------------------------


class TestListNotes:
    @pytest.mark.asyncio
    async def test_list_notes_success(self, handler, mock_connector):
        note = _make_note(user=SimpleNamespace(id="U1", name="Bob"))
        mock_connector.list_notes = AsyncMock(return_value=[note])

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_list_notes(req, "test-tenant", "PDINC000001")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["count"] == 1
        assert body["data"]["notes"][0]["id"] == "NOTE001"
        assert body["data"]["notes"][0]["user"]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_list_notes_no_user(self, handler, mock_connector):
        note = _make_note(user=None)
        mock_connector.list_notes = AsyncMock(return_value=[note])

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_list_notes(req, "test-tenant", "PDINC000001")

        body, _ = _parse_result(result)
        assert body["data"]["notes"][0]["user"] is None

    @pytest.mark.asyncio
    async def test_list_notes_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler._handle_list_notes(req, "test-tenant", "PDINC000001")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_list_notes_error(self, handler, mock_connector):
        mock_connector.list_notes = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_list_notes(req, "test-tenant", "PDINC000001")
        _, status = _parse_result(result)
        assert status == 500


class TestAddNote:
    @pytest.mark.asyncio
    async def test_add_note_success(self, handler, mock_connector):
        note = _make_note(id="NOTE_NEW", content="root cause found")
        mock_connector.add_note = AsyncMock(return_value=note)

        req = _make_request(json_body={"content": "root cause found"})
        with _patch_connector(mock_connector):
            result = await handler._handle_add_note(req, "test-tenant", "PDINC000001")

        body, status = _parse_result(result)
        assert status == 201
        assert body["note"]["id"] == "NOTE_NEW"

    @pytest.mark.asyncio
    async def test_add_note_missing_content(self, handler, mock_connector):
        req = _make_request(json_body={})
        with _patch_connector(mock_connector):
            result = await handler._handle_add_note(req, "test-tenant", "PDINC000001")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_add_note_error(self, handler, mock_connector):
        mock_connector.add_note = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request(json_body={"content": "text"})
        with _patch_connector(mock_connector):
            result = await handler._handle_add_note(req, "test-tenant", "PDINC000001")
        _, status = _parse_result(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_add_note_503(self, handler):
        req = _make_request(json_body={"content": "text"})
        with _patch_connector(None):
            result = await handler._handle_add_note(req, "test-tenant", "PDINC000001")
        _, status = _parse_result(result)
        assert status == 503


# ---------------------------------------------------------------------------
# Tests: On-Call (via handle for top-level, direct for service-specific)
# ---------------------------------------------------------------------------


class TestGetOncall:
    @pytest.mark.asyncio
    async def test_get_oncall_success(self, handler, mock_connector):
        sched = _make_oncall()
        mock_connector.get_on_call = AsyncMock(return_value=[sched])

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/oncall", "GET")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["count"] == 1
        assert body["data"]["oncall"][0]["user"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_oncall_with_schedule_filter(self, handler, mock_connector):
        mock_connector.get_on_call = AsyncMock(return_value=[])

        req = _make_request(query={"schedule_ids": "PDSCHED0001,PDSCHED0002"})
        with _patch_connector(mock_connector):
            await handler.handle(req, "/api/v1/oncall", "GET")

        mock_connector.get_on_call.assert_called_once_with(
            schedule_ids=["PDSCHED0001", "PDSCHED0002"]
        )

    @pytest.mark.asyncio
    async def test_get_oncall_error(self, handler, mock_connector):
        mock_connector.get_on_call = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/oncall", "GET")
        _, status = _parse_result(result)
        assert status == 500


class TestGetOncallForService:
    @pytest.mark.asyncio
    async def test_get_oncall_for_service_success(self, handler, mock_connector):
        sched = _make_oncall()
        mock_connector.get_current_on_call_for_service = AsyncMock(return_value=[sched])

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_oncall_for_service(req, "test-tenant", "PDSVC000001")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["service_id"] == "PDSVC000001"
        assert len(body["data"]["oncall"]) == 1

    @pytest.mark.asyncio
    async def test_get_oncall_for_service_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler._handle_get_oncall_for_service(req, "test-tenant", "PDSVC000001")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_get_oncall_for_service_error(self, handler, mock_connector):
        mock_connector.get_current_on_call_for_service = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_oncall_for_service(req, "test-tenant", "PDSVC000001")
        _, status = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Tests: Services (via handle for list, direct for get)
# ---------------------------------------------------------------------------


class TestListServices:
    @pytest.mark.asyncio
    async def test_list_services_success(self, handler, mock_connector):
        svc = _make_service()
        mock_connector.list_services = AsyncMock(return_value=([svc], False))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/services", "GET")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["count"] == 1
        assert body["data"]["services"][0]["id"] == "PDSVC001234"

    @pytest.mark.asyncio
    async def test_list_services_with_pagination(self, handler, mock_connector):
        mock_connector.list_services = AsyncMock(return_value=([], True))

        req = _make_request(query={"limit": "5", "offset": "10"})
        with _patch_connector(mock_connector):
            await handler.handle(req, "/api/v1/services", "GET")

        mock_connector.list_services.assert_called_once_with(limit=5, offset=10)

    @pytest.mark.asyncio
    async def test_list_services_error(self, handler, mock_connector):
        mock_connector.list_services = AsyncMock(side_effect=RuntimeError("err"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/services", "GET")
        _, status = _parse_result(result)
        assert status == 500


class TestGetService:
    @pytest.mark.asyncio
    async def test_get_service_success(self, handler, mock_connector):
        svc = _make_service(id="PDSVC000042")
        mock_connector.get_service = AsyncMock(return_value=svc)

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_service(req, "test-tenant", "PDSVC000042")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["service"]["id"] == "PDSVC000042"
        assert body["data"]["service"]["escalation_policy_id"] == "PDESCPOL001"

    @pytest.mark.asyncio
    async def test_get_service_503(self, handler):
        req = _make_request()
        with _patch_connector(None):
            result = await handler._handle_get_service(req, "test-tenant", "PDSVC000099")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_get_service_error(self, handler, mock_connector):
        mock_connector.get_service = AsyncMock(side_effect=RuntimeError("no service"))

        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_service(req, "test-tenant", "PDSVC000099")
        _, status = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Tests: Webhooks (via handle)
# ---------------------------------------------------------------------------


class TestPagerDutyWebhook:
    @pytest.mark.asyncio
    async def test_webhook_success_with_connector(self, handler, mock_connector):
        mock_connector.parse_webhook = MagicMock(return_value=None)

        req = _make_request(
            json_body={"event": {"event_type": "incident.triggered"}},
            raw_body=b'{"event": {"event_type": "incident.triggered"}}',
            headers={"X-PagerDuty-Signature": "sig123"},
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/webhooks/pagerduty", "POST")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["received"] is True
        assert body["data"]["event_type"] == "incident.triggered"

    @pytest.mark.asyncio
    async def test_webhook_with_parsed_payload(self, handler, mock_connector):
        payload = SimpleNamespace(event_type="incident.resolved")
        payload.to_dict = MagicMock(return_value={"event_type": "incident.resolved"})
        mock_connector.parse_webhook = MagicMock(return_value=payload)

        req = _make_request(
            json_body={"event": {}},
            raw_body=b'{"event": {}}',
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/webhooks/pagerduty", "POST")

        body, _ = _parse_result(result)
        assert body["data"]["event_type"] == "incident.resolved"

    @pytest.mark.asyncio
    async def test_webhook_no_connector(self, handler):
        req = _make_request(
            json_body={"event": {"event_type": "incident.triggered"}},
            raw_body=b'{"event": {"event_type": "incident.triggered"}}',
        )
        with _patch_connector(None):
            result = await handler.handle(req, "/api/v1/webhooks/pagerduty", "POST")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["received"] is True

    @pytest.mark.asyncio
    async def test_webhook_error_still_returns_received(self, handler, mock_connector):
        mock_connector.verify_webhook_signature = MagicMock(side_effect=RuntimeError("boom"))

        req = _make_request(
            json_body={"event": {}},
            raw_body=b"bad",
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/webhooks/pagerduty", "POST")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["received"] is True

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature_still_processes(self, handler, mock_connector):
        mock_connector.verify_webhook_signature = MagicMock(return_value=False)
        mock_connector.parse_webhook = MagicMock(return_value=None)

        req = _make_request(
            json_body={"event": {"event_type": "incident.acknowledged"}},
            raw_body=b"{}",
            headers={"X-PagerDuty-Signature": "bad_sig"},
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/webhooks/pagerduty", "POST")

        body, status = _parse_result(result)
        assert status == 200
        assert body["data"]["received"] is True


# ---------------------------------------------------------------------------
# Tests: Not found / unknown routes (via handle)
# ---------------------------------------------------------------------------


class TestNotFound:
    @pytest.mark.asyncio
    async def test_put_incidents_not_routed(self, handler):
        req = _make_request()
        result = await handler.handle(req, "/api/v1/incidents", "PUT")
        _, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_delete_incidents_not_routed(self, handler):
        req = _make_request()
        result = await handler.handle(req, "/api/v1/incidents", "DELETE")
        _, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_unknown_devops_subpath(self, handler):
        req = _make_request()
        result = await handler.handle(req, "/api/v1/devops/nonexistent", "GET")
        _, status = _parse_result(result)
        assert status == 404


# ---------------------------------------------------------------------------
# Tests: Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_handler(self):
        h = create_devops_handler()
        assert isinstance(h, DevOpsHandler)

    def test_creates_handler_with_context(self):
        ctx = {"key": "value"}
        h = create_devops_handler(server_context=ctx)
        assert isinstance(h, DevOpsHandler)


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector
# ---------------------------------------------------------------------------


class TestGetPagerDutyConnector:
    @pytest.mark.asyncio
    async def test_returns_none_without_env_vars(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await get_pagerduty_connector("tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_missing_email(self):
        with patch.dict("os.environ", {"PAGERDUTY_API_KEY": "key"}, clear=True):
            result = await get_pagerduty_connector("tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_missing_api_key(self):
        with patch.dict("os.environ", {"PAGERDUTY_EMAIL": "e@x.com"}, clear=True):
            result = await get_pagerduty_connector("tenant1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_connector(self):
        mock_conn = MagicMock()
        _connector_instances["cached_tenant"] = mock_conn
        result = await get_pagerduty_connector("cached_tenant")
        assert result is mock_conn

    @pytest.mark.asyncio
    async def test_handles_import_error(self):
        """When the pagerduty module cannot be imported, returns None."""
        with (
            patch.dict(
                "os.environ",
                {"PAGERDUTY_API_KEY": "k", "PAGERDUTY_EMAIL": "e@x.com"},
                clear=True,
            ),
            patch.dict(sys.modules, {"aragora.connectors.devops.pagerduty": None}),
        ):
            result = await get_pagerduty_connector("tenant_imp")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Tenant ID extraction
# ---------------------------------------------------------------------------


class TestTenantId:
    def test_extracts_from_request(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace(tenant_id="my-tenant")
        assert h._get_tenant_id(req) == "my-tenant"

    def test_defaults_to_default(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace()
        assert h._get_tenant_id(req) == "default"


# ---------------------------------------------------------------------------
# Tests: Utility methods
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_get_query_params_from_query(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace(query={"a": "1", "b": "2"})
        assert h._get_query_params(req) == {"a": "1", "b": "2"}

    def test_get_query_params_empty(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace()
        assert h._get_query_params(req) == {}

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self):
        h = DevOpsHandler(server_context={})

        async def _json():
            return {"key": "val"}

        req = SimpleNamespace(json=_json)
        result = await h._get_json_body(req)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_get_json_body_property(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace(json={"key": "val"})
        result = await h._get_json_body(req)
        assert result == {"key": "val"}

    @pytest.mark.asyncio
    async def test_get_json_body_none(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace()
        result = await h._get_json_body(req)
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_raw_body_callable(self):
        h = DevOpsHandler(server_context={})

        async def _body():
            return b"raw"

        req = SimpleNamespace(body=_body)
        result = await h._get_raw_body(req)
        assert result == b"raw"

    @pytest.mark.asyncio
    async def test_get_raw_body_property(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace(body=b"raw")
        result = await h._get_raw_body(req)
        assert result == b"raw"

    @pytest.mark.asyncio
    async def test_get_raw_body_read(self):
        h = DevOpsHandler(server_context={})

        async def _read():
            return b"read_data"

        req = SimpleNamespace(read=_read)
        result = await h._get_raw_body(req)
        assert result == b"read_data"

    @pytest.mark.asyncio
    async def test_get_raw_body_nothing(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace()
        result = await h._get_raw_body(req)
        assert result == b""

    def test_get_header(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace(headers={"X-Custom": "value"})
        assert h._get_header(req, "X-Custom") == "value"

    def test_get_header_missing(self):
        h = DevOpsHandler(server_context={})
        req = SimpleNamespace()
        assert h._get_header(req, "X-Custom") is None


# ---------------------------------------------------------------------------
# Tests: _emit_connector_event
# ---------------------------------------------------------------------------


class TestEmitConnectorEvent:
    @pytest.mark.asyncio
    async def test_emit_event_without_emitter(self, handler):
        """When no emitter in context, no error raised."""
        handler.ctx = {}
        await handler._emit_connector_event(
            event_type="incident.resolved",
            tenant_id="t1",
            data={},
        )

    @pytest.mark.asyncio
    async def test_emit_event_with_no_context(self, handler):
        """When ctx is None, no error raised."""
        handler.ctx = None
        await handler._emit_connector_event(
            event_type="test",
            tenant_id="t1",
            data={},
        )


# ---------------------------------------------------------------------------
# Tests: Circuit Breaker
# ---------------------------------------------------------------------------


class TestDevOpsCircuitBreaker:
    """Tests for the DevOpsCircuitBreaker class."""

    def test_initial_state_closed(self):
        cb = DevOpsCircuitBreaker()
        assert cb.state == "closed"
        assert cb.is_allowed() is True

    def test_opens_after_threshold_failures(self):
        cb = DevOpsCircuitBreaker(failure_threshold=3)
        assert cb.state == "closed"

        # Record failures until threshold
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.is_allowed() is False

    def test_success_resets_failure_count(self):
        cb = DevOpsCircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2

        cb.record_success()
        assert cb._failure_count == 0

    def test_half_open_state_after_cooldown(self):
        import time

        cb = DevOpsCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to half_open
        assert cb.state == "half_open"
        assert cb.is_allowed() is True

    def test_half_open_closes_on_success(self):
        import time

        cb = DevOpsCircuitBreaker(failure_threshold=2, cooldown_seconds=0.05, half_open_max_calls=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)

        # Transition to half_open
        assert cb.state == "half_open"

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.state == "closed"

    def test_half_open_reopens_on_failure(self):
        import time

        cb = DevOpsCircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)

        # Transition to half_open
        assert cb.state == "half_open"

        # Fail during half_open
        cb.record_failure()

        assert cb.state == "open"

    def test_reset(self):
        cb = DevOpsCircuitBreaker(failure_threshold=2)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_get_status(self):
        cb = DevOpsCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        status = cb.get_status()

        assert status["state"] == "closed"
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0
        assert "failure_count" in status


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with handler methods."""

    @pytest.fixture(autouse=True)
    def clear_circuit_breaker(self):
        """Reset the circuit breaker before each test."""
        from aragora.server.handlers.features.devops import _clear_devops_components

        _clear_devops_components()
        yield
        _clear_devops_components()

    @pytest.mark.asyncio
    async def test_list_incidents_returns_503_when_circuit_open(self, handler):
        cb = get_devops_circuit_breaker()
        # Open the circuit
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"

        req = _make_request()
        result = await handler._handle_list_incidents(req, "test-tenant")
        _, status = _parse_result(result)
        assert status == 503

    @pytest.mark.asyncio
    async def test_status_endpoint_includes_circuit_breaker_status(self, handler):
        req = _make_request()
        with patch.dict(
            "os.environ",
            {"PAGERDUTY_API_KEY": "key", "PAGERDUTY_EMAIL": "e@co.com"},
        ):
            result = await handler.handle(req, "/api/v1/devops/status", "GET")
        body, status = _parse_result(result)
        assert status == 200
        assert "circuit_breaker" in body["data"]
        assert body["data"]["circuit_breaker"]["state"] == "closed"


# ---------------------------------------------------------------------------
# Tests: Input Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_pagerduty_id_valid(self):
        from aragora.server.handlers.features.devops import _validate_pagerduty_id

        is_valid, err = _validate_pagerduty_id("ABCD123", "test_id")
        assert is_valid is True
        assert err is None

    def test_validate_pagerduty_id_empty(self):
        from aragora.server.handlers.features.devops import _validate_pagerduty_id

        is_valid, err = _validate_pagerduty_id("", "test_id")
        assert is_valid is False
        assert "required" in err

    def test_validate_pagerduty_id_too_long(self):
        from aragora.server.handlers.features.devops import _validate_pagerduty_id

        is_valid, err = _validate_pagerduty_id("A" * 25, "test_id")
        assert is_valid is False
        assert "too long" in err

    def test_validate_pagerduty_id_invalid_format(self):
        from aragora.server.handlers.features.devops import _validate_pagerduty_id

        is_valid, err = _validate_pagerduty_id("abc-123!", "test_id")
        assert is_valid is False
        assert "invalid format" in err

    def test_validate_urgency_valid(self):
        from aragora.server.handlers.features.devops import _validate_urgency

        assert _validate_urgency("high") == "high"
        assert _validate_urgency("low") == "low"
        assert _validate_urgency("HIGH") == "high"

    def test_validate_urgency_invalid_defaults_high(self):
        from aragora.server.handlers.features.devops import _validate_urgency

        assert _validate_urgency("invalid") == "high"
        assert _validate_urgency(None) == "high"

    def test_validate_string_field_valid(self):
        from aragora.server.handlers.features.devops import _validate_string_field

        val, err = _validate_string_field("test value", "field", required=True)
        assert val == "test value"
        assert err is None

    def test_validate_string_field_required_missing(self):
        from aragora.server.handlers.features.devops import _validate_string_field

        val, err = _validate_string_field(None, "field", required=True)
        assert val is None
        assert "required" in err

    def test_validate_string_field_exceeds_max_length(self):
        from aragora.server.handlers.features.devops import _validate_string_field

        val, err = _validate_string_field("x" * 100, "field", max_length=50)
        assert val is None
        assert "exceeds maximum" in err

    def test_validate_string_field_strips_whitespace(self):
        from aragora.server.handlers.features.devops import _validate_string_field

        val, err = _validate_string_field("  test  ", "field")
        assert val == "test"
        assert err is None

    def test_validate_id_list_valid(self):
        from aragora.server.handlers.features.devops import _validate_id_list

        val, err = _validate_id_list(["ABCD123", "EFGH456"], "ids")
        assert val == ["ABCD123", "EFGH456"]
        assert err is None

    def test_validate_id_list_exceeds_max(self):
        from aragora.server.handlers.features.devops import _validate_id_list

        ids = ["ABC1234"] * 30
        val, err = _validate_id_list(ids, "ids", max_items=20)
        assert val is None
        assert "exceeds maximum" in err

    def test_validate_id_list_invalid_item(self):
        from aragora.server.handlers.features.devops import _validate_id_list

        val, err = _validate_id_list(["ABCD123", "invalid!"], "ids")
        assert val is None
        assert "invalid format" in err

    def test_validate_id_list_not_list(self):
        from aragora.server.handlers.features.devops import _validate_id_list

        val, err = _validate_id_list("not-a-list", "ids")
        assert val is None
        assert "must be a list" in err


class TestInputValidationIntegration:
    """Tests for input validation integration with handler methods."""

    @pytest.fixture(autouse=True)
    def clear_state(self):
        """Reset components before each test."""
        from aragora.server.handlers.features.devops import _clear_devops_components

        _clear_devops_components()
        yield
        _clear_devops_components()

    @pytest.mark.asyncio
    async def test_create_incident_invalid_service_id(self, handler, mock_connector):
        req = _make_request(
            json_body={
                "title": "Valid Title",
                "service_id": "invalid!@#",  # Invalid format
            }
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "service_id" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_create_incident_title_too_long(self, handler, mock_connector):
        req = _make_request(
            json_body={
                "title": "X" * 600,  # Exceeds MAX_TITLE_LENGTH
                "service_id": "ABCD1234567",
            }
        )
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "POST")
        body, status = _parse_result(result)
        assert status == 400
        assert "title" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_list_incidents_invalid_status_filter(self, handler, mock_connector):
        req = _make_request(query={"status": "invalid_status"})
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "GET")
        body, status = _parse_result(result)
        assert status == 400
        assert "Invalid status" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_list_incidents_invalid_urgency_filter(self, handler, mock_connector):
        req = _make_request(query={"urgency": "critical"})  # Not valid
        with _patch_connector(mock_connector):
            result = await handler.handle(req, "/api/v1/incidents", "GET")
        body, status = _parse_result(result)
        assert status == 400
        assert "Invalid urgency" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_get_incident_invalid_id_format(self, handler, mock_connector):
        req = _make_request()
        with _patch_connector(mock_connector):
            result = await handler._handle_get_incident(req, "test-tenant", "invalid!id")
        _, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_add_note_content_too_long(self, handler, mock_connector):
        req = _make_request(json_body={"content": "X" * 6000})  # Exceeds limit
        with _patch_connector(mock_connector):
            result = await handler._handle_add_note(req, "test-tenant", "ABCD1234567")
        body, status = _parse_result(result)
        assert status == 400
        assert "content" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_reassign_invalid_user_ids(self, handler, mock_connector):
        req = _make_request(json_body={"user_ids": ["invalid!"]})
        with _patch_connector(mock_connector):
            result = await handler._handle_reassign_incident(req, "test-tenant", "ABCD1234567")
        body, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_merge_too_many_source_ids(self, handler, mock_connector):
        # Create 60 valid IDs (exceeds MAX_SOURCE_INCIDENT_IDS of 50)
        source_ids = [f"INC{i:010d}" for i in range(60)]
        req = _make_request(json_body={"source_incident_ids": source_ids})
        with _patch_connector(mock_connector):
            result = await handler._handle_merge_incidents(req, "test-tenant", "INC0000000001")
        body, status = _parse_result(result)
        assert status == 400
        assert "exceeds maximum" in body.get("error", "")


# ---------------------------------------------------------------------------
# Tests: Clear Components
# ---------------------------------------------------------------------------


class TestClearComponents:
    """Test the _clear_devops_components function."""

    def test_clears_all_state(self):
        from aragora.server.handlers.features.devops import (
            _clear_devops_components,
            _connector_instances,
        )

        # Set up some state
        _connector_instances["test"] = MagicMock()
        _active_contexts["test"] = MagicMock()
        cb = get_devops_circuit_breaker()
        cb.record_failure()

        # Clear
        _clear_devops_components()

        # Verify cleared
        assert len(_connector_instances) == 0
        assert len(_active_contexts) == 0


# ---------------------------------------------------------------------------
# Tests: Module Exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test that required exports are available."""

    def test_circuit_breaker_exports(self):
        from aragora.server.handlers.features.devops import (
            _clear_devops_components,
        )

        assert DevOpsCircuitBreaker is not None
        assert callable(get_devops_circuit_breaker)
        assert callable(get_devops_circuit_breaker_status)
        assert callable(_clear_devops_components)

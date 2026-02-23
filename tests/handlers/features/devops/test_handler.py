"""Tests for the DevOps incident management handler.

Covers all routes and behavior of the DevOpsHandler class:
- can_handle() routing for all defined paths
- GET /api/v1/incidents - list incidents with filtering
- GET /api/v1/incidents/{id} - get incident details
- POST /api/v1/incidents - create incident
- POST /api/v1/incidents/{id}/acknowledge - acknowledge incident
- POST /api/v1/incidents/{id}/resolve - resolve incident
- POST /api/v1/incidents/{id}/reassign - reassign incident
- POST /api/v1/incidents/{id}/merge - merge incidents
- GET /api/v1/incidents/{id}/notes - list notes
- POST /api/v1/incidents/{id}/notes - add note
- GET /api/v1/oncall - get on-call schedules
- GET /api/v1/oncall/services/{id} - get on-call for service
- GET /api/v1/services - list services
- GET /api/v1/services/{id} - get service details
- POST /api/v1/webhooks/pagerduty - webhook handler
- GET /api/v1/devops/status - connection status
- Error handling (circuit breaker, connector failures, validation)
- Path parameter extraction
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.devops.handler import (
    DevOpsHandler,
    create_devops_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockIncidentStatus(Enum):
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class MockUrgency(Enum):
    HIGH = "high"
    LOW = "low"


class MockServiceStatus(Enum):
    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass
class MockIncident:
    id: str = "PINC001"
    title: str = "Test Incident"
    status: MockIncidentStatus = MockIncidentStatus.TRIGGERED
    urgency: MockUrgency = MockUrgency.HIGH
    service_id: str = "PSVC001"
    service_name: str = "Test Service"
    incident_number: int = 42
    created_at: datetime | None = None
    html_url: str = "https://pd.example.com/incidents/PINC001"
    description: str | None = "A test incident"
    assignees: list[str] | None = None
    priority: str | None = "P1"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        if self.assignees is None:
            self.assignees = ["user1"]


@dataclass
class MockNoteUser:
    id: str = "PUSR001"
    name: str = "Test User"
    email: str = "test@example.com"


@dataclass
class MockNote:
    id: str = "PNOTE001"
    content: str = "Investigation note"
    created_at: datetime | None = None
    user: MockNoteUser | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        if self.user is None:
            self.user = MockNoteUser()


@dataclass
class MockOnCallSchedule:
    schedule_id: str = "PSCHED001"
    schedule_name: str = "Primary Schedule"
    user: MockNoteUser = None
    start: datetime = None
    end: datetime = None
    escalation_level: int = 1

    def __post_init__(self):
        if self.user is None:
            self.user = MockNoteUser()
        if self.start is None:
            self.start = datetime(2026, 2, 23, 0, 0, 0, tzinfo=timezone.utc)
        if self.end is None:
            self.end = datetime(2026, 2, 24, 0, 0, 0, tzinfo=timezone.utc)


@dataclass
class MockService:
    id: str = "PSVC001"
    name: str = "Test Service"
    description: str = "A test service"
    status: MockServiceStatus = MockServiceStatus.ACTIVE
    html_url: str = "https://pd.example.com/services/PSVC001"
    escalation_policy_id: str = "PESCPOL001"
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class MockWebhookPayload:
    event_type: str = "incident.triggered"

    def to_dict(self) -> dict:
        return {"event_type": self.event_type}


@dataclass
class MockRequest:
    """Mock async HTTP request for DevOpsHandler."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = field(default_factory=dict)
    _body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tenant_id: str = "default"

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def body(self) -> bytes:
        return json.dumps(self._body or {}).encode()

    async def read(self) -> bytes:
        return json.dumps(self._body or {}).encode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a DevOpsHandler instance."""
    return DevOpsHandler({})


@pytest.fixture
def mock_connector():
    """Create a mock PagerDuty connector."""
    connector = AsyncMock()
    connector.list_incidents = AsyncMock(return_value=([MockIncident()], False))
    connector.get_incident = AsyncMock(return_value=MockIncident())
    connector.create_incident = AsyncMock(return_value=MockIncident())
    connector.acknowledge_incident = AsyncMock(
        return_value=MockIncident(status=MockIncidentStatus.ACKNOWLEDGED)
    )
    connector.resolve_incident = AsyncMock(
        return_value=MockIncident(status=MockIncidentStatus.RESOLVED)
    )
    connector.reassign_incident = AsyncMock(return_value=MockIncident())
    connector.merge_incidents = AsyncMock(return_value=MockIncident())
    connector.list_notes = AsyncMock(return_value=[MockNote()])
    connector.add_note = AsyncMock(return_value=MockNote())
    connector.get_on_call = AsyncMock(return_value=[MockOnCallSchedule()])
    connector.get_current_on_call_for_service = AsyncMock(
        return_value=[MockOnCallSchedule()]
    )
    connector.list_services = AsyncMock(return_value=([MockService()], False))
    connector.get_service = AsyncMock(return_value=MockService())
    connector.verify_webhook_signature = MagicMock(return_value=True)
    connector.parse_webhook = MagicMock(return_value=MockWebhookPayload())
    return connector


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker that allows all requests."""
    cb = MagicMock()
    cb.is_allowed.return_value = True
    cb.record_success = MagicMock()
    cb.record_failure = MagicMock()
    return cb


@pytest.fixture
def mock_circuit_breaker_open():
    """Create a mock circuit breaker that blocks all requests."""
    cb = MagicMock()
    cb.is_allowed.return_value = False
    return cb


def _patch_connector(connector):
    """Return a patch context for get_pagerduty_connector."""
    return patch(
        "aragora.server.handlers.features.devops.connector.get_pagerduty_connector",
        new_callable=AsyncMock,
        return_value=connector,
    )


def _patch_cb(cb):
    """Return a patch context for get_devops_circuit_breaker."""
    return patch(
        "aragora.server.handlers.features.devops.handler.get_devops_circuit_breaker",
        return_value=cb,
    )


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestFactory:
    """Tests for the create_devops_handler factory."""

    def test_create_devops_handler_returns_instance(self):
        h = create_devops_handler()
        assert isinstance(h, DevOpsHandler)

    def test_create_devops_handler_with_context(self):
        ctx = {"emitter": MagicMock()}
        h = create_devops_handler(ctx)
        assert isinstance(h, DevOpsHandler)

    def test_create_devops_handler_none_context(self):
        h = create_devops_handler(None)
        assert isinstance(h, DevOpsHandler)


# ---------------------------------------------------------------------------
# can_handle Tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_handles_incidents_path(self, handler):
        assert handler.can_handle("/api/v1/incidents") is True

    def test_handles_incident_by_id(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001") is True

    def test_handles_incident_acknowledge(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001/acknowledge", "POST") is True

    def test_handles_incident_resolve(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001/resolve", "POST") is True

    def test_handles_incident_reassign(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001/reassign", "POST") is True

    def test_handles_incident_merge(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001/merge", "POST") is True

    def test_handles_incident_notes(self, handler):
        assert handler.can_handle("/api/v1/incidents/PINC001/notes") is True

    def test_handles_oncall(self, handler):
        assert handler.can_handle("/api/v1/oncall") is True

    def test_handles_oncall_services(self, handler):
        assert handler.can_handle("/api/v1/oncall/services/PSVC001") is True

    def test_handles_services(self, handler):
        assert handler.can_handle("/api/v1/services") is True

    def test_handles_service_by_id(self, handler):
        assert handler.can_handle("/api/v1/services/PSVC001") is True

    def test_handles_webhooks_pagerduty(self, handler):
        assert handler.can_handle("/api/v1/webhooks/pagerduty", "POST") is True

    def test_handles_devops_status(self, handler):
        assert handler.can_handle("/api/v1/devops/status") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_does_not_handle_health(self, handler):
        assert handler.can_handle("/api/v1/health") is False

    def test_does_not_handle_users(self, handler):
        assert handler.can_handle("/api/v1/users") is False


# ---------------------------------------------------------------------------
# ROUTES list Tests
# ---------------------------------------------------------------------------


class TestRoutesDefinition:
    """Tests for the ROUTES class attribute."""

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 13

    def test_routes_contain_expected_paths(self, handler):
        expected = [
            "/api/v1/incidents",
            "/api/v1/incidents/{incident_id}",
            "/api/v1/oncall",
            "/api/v1/services",
            "/api/v1/webhooks/pagerduty",
            "/api/v1/devops/status",
        ]
        for path in expected:
            assert path in handler.ROUTES, f"Missing route: {path}"


# ---------------------------------------------------------------------------
# GET /api/v1/devops/status
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for GET /api/v1/devops/status."""

    @pytest.mark.asyncio
    async def test_status_configured(self, handler):
        request = MockRequest()
        with patch.dict("os.environ", {
            "PAGERDUTY_API_KEY": "test-key",
            "PAGERDUTY_EMAIL": "test@example.com",
            "PAGERDUTY_WEBHOOK_SECRET": "secret",
        }), patch(
            "aragora.server.handlers.features.devops.handler.get_devops_circuit_breaker_status",
            return_value={"state": "closed"},
        ):
            result = await handler.handle(request, "/api/v1/devops/status", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["configured"] is True
        assert data["api_key_set"] is True
        assert data["email_set"] is True
        assert data["webhook_secret_set"] is True
        assert data["circuit_breaker"] == {"state": "closed"}

    @pytest.mark.asyncio
    async def test_status_not_configured(self, handler):
        request = MockRequest()
        with patch.dict("os.environ", {}, clear=True), patch(
            "aragora.server.handlers.features.devops.handler.get_devops_circuit_breaker_status",
            return_value={"state": "closed"},
        ):
            result = await handler.handle(request, "/api/v1/devops/status", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["configured"] is False
        assert data["api_key_set"] is False

    @pytest.mark.asyncio
    async def test_status_partial_config(self, handler):
        request = MockRequest()
        with patch.dict("os.environ", {
            "PAGERDUTY_API_KEY": "test-key",
        }, clear=True), patch(
            "aragora.server.handlers.features.devops.handler.get_devops_circuit_breaker_status",
            return_value={"state": "closed"},
        ):
            result = await handler.handle(request, "/api/v1/devops/status", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["configured"] is False
        assert data["api_key_set"] is True
        assert data["email_set"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/incidents
# ---------------------------------------------------------------------------


class TestListIncidents:
    """Tests for GET /api/v1/incidents."""

    @pytest.mark.asyncio
    async def test_list_incidents_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 1
        assert data["has_more"] is False
        assert data["incidents"][0]["id"] == "PINC001"
        assert data["incidents"][0]["title"] == "Test Incident"
        assert data["incidents"][0]["status"] == "triggered"
        mock_circuit_breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_incidents_with_status_filter(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"status": "triggered,acknowledged"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.list_incidents.call_args[1]
        assert call_kwargs["statuses"] == ["triggered", "acknowledged"]

    @pytest.mark.asyncio
    async def test_list_incidents_invalid_status(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"status": "invalid_status"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_incidents_with_urgency_filter(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"urgency": "high"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.list_incidents.call_args[1]
        assert call_kwargs["urgencies"] == ["high"]

    @pytest.mark.asyncio
    async def test_list_incidents_invalid_urgency(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"urgency": "critical"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_incidents_with_service_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"service_ids": "PSVC001,PSVC002"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.list_incidents.call_args[1]
        assert call_kwargs["service_ids"] == ["PSVC001", "PSVC002"]

    @pytest.mark.asyncio
    async def test_list_incidents_with_pagination(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"limit": "10", "offset": "20"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.list_incidents.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_list_incidents_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(query={})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_incidents_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 503
        assert "not configured" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_incidents_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.list_incidents.side_effect = ConnectionError("connection lost")
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()
        mock_circuit_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_incidents_timeout_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.list_incidents.side_effect = TimeoutError("timed out")
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/incidents
# ---------------------------------------------------------------------------


class TestCreateIncident:
    """Tests for POST /api/v1/incidents."""

    @pytest.mark.asyncio
    async def test_create_incident_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Critical DB failure",
            "service_id": "PSVC001",
            "urgency": "high",
            "body": "Database is not responding",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker), patch(
            "aragora.server.handlers.features.devops.handler.IncidentCreateRequest",
            create=True,
        ) as mock_req_cls, patch(
            "aragora.server.handlers.features.devops.handler.IncidentUrgency",
            create=True,
        ):
            # The handler imports IncidentCreateRequest and IncidentUrgency inside the method,
            # so we patch the connector's create_incident directly
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 201
        body = _body(result)
        assert body["message"] == "Incident created successfully"
        assert body["incident"]["id"] == "PINC001"
        mock_circuit_breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_incident_missing_title(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "service_id": "PSVC001",
            "urgency": "high",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400
        assert "title" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_incident_missing_service_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test Incident",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400
        assert "service_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_incident_invalid_service_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test Incident",
            "service_id": "invalid id with spaces!",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_incident_too_long_title(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "x" * 501,
            "service_id": "PSVC001",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400
        assert "length" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_incident_invalid_escalation_policy_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
            "escalation_policy_id": "invalid id!!!",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_incident_invalid_priority_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
            "priority_id": "bad format!!",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_incident_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
        })
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_incident_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
        })
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_incident_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.create_incident.side_effect = ConnectionError("failed")
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
            "urgency": "high",
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_incident_too_long_body(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={
            "title": "Test",
            "service_id": "PSVC001",
            "body": "x" * 10001,
        })
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "POST")
        assert _status(result) == 400
        assert "length" in _body(result)["error"].lower()


# ---------------------------------------------------------------------------
# GET /api/v1/incidents/{id}
# ---------------------------------------------------------------------------


class TestGetIncident:
    """Tests for GET /api/v1/incidents/{id}."""

    @pytest.mark.asyncio
    async def test_get_incident_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents/PINC001", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["incident"]["id"] == "PINC001"
        assert data["incident"]["description"] == "A test incident"
        assert data["incident"]["assignees"] == ["user1"]
        mock_connector.get_incident.assert_awaited_once_with("PINC001")

    @pytest.mark.asyncio
    async def test_get_incident_invalid_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents/bad id!!", "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_incident_id_too_long(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, f"/api/v1/incidents/{'A' * 21}", "GET"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_incident_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest()
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(request, "/api/v1/incidents/PINC001", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_incident_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.get_incident.side_effect = OSError("network error")
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents/PINC001", "GET")
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_incident_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents/PINC001", "GET")
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# POST /api/v1/incidents/{id}/acknowledge
# ---------------------------------------------------------------------------


class TestAcknowledgeIncident:
    """Tests for POST /api/v1/incidents/{id}/acknowledge."""

    @pytest.mark.asyncio
    async def test_acknowledge_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/acknowledge", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["message"] == "Incident acknowledged"
        assert data["incident"]["status"] == "acknowledged"
        mock_connector.acknowledge_incident.assert_awaited_once_with("PINC001")
        mock_circuit_breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_acknowledge_invalid_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/acknowledge", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_acknowledge_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest()
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/acknowledge", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_acknowledge_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/acknowledge", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_acknowledge_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.acknowledge_incident.side_effect = ValueError("invalid state")
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/acknowledge", "POST"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/incidents/{id}/resolve
# ---------------------------------------------------------------------------


class TestResolveIncident:
    """Tests for POST /api/v1/incidents/{id}/resolve."""

    @pytest.mark.asyncio
    async def test_resolve_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"resolution": "Fixed the issue"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/resolve", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["message"] == "Incident resolved"
        assert data["incident"]["status"] == "resolved"
        mock_connector.resolve_incident.assert_awaited_once_with("PINC001", "Fixed the issue")

    @pytest.mark.asyncio
    async def test_resolve_without_resolution(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/resolve", "POST"
            )
        assert _status(result) == 200
        mock_connector.resolve_incident.assert_awaited_once_with("PINC001", None)

    @pytest.mark.asyncio
    async def test_resolve_too_long_resolution(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"resolution": "x" * 2001})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/resolve", "POST"
            )
        assert _status(result) == 400
        assert "length" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_resolve_invalid_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/resolve", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resolve_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(_body={})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/resolve", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_resolve_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.resolve_incident.side_effect = TimeoutError("timeout")
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/resolve", "POST"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/incidents/{id}/reassign
# ---------------------------------------------------------------------------


class TestReassignIncident:
    """Tests for POST /api/v1/incidents/{id}/reassign."""

    @pytest.mark.asyncio
    async def test_reassign_with_user_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"user_ids": ["PUSR001"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["message"] == "Incident reassigned"

    @pytest.mark.asyncio
    async def test_reassign_with_escalation_policy(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"escalation_policy_id": "PESCPOL001"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reassign_missing_both(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_reassign_invalid_user_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"user_ids": "not-a-list"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_reassign_invalid_escalation_policy_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"escalation_policy_id": "bad id!!"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_reassign_invalid_incident_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"user_ids": ["PUSR001"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/reassign", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_reassign_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(_body={"user_ids": ["PUSR001"]})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_reassign_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.reassign_incident.side_effect = OSError("fail")
        request = MockRequest(_body={"user_ids": ["PUSR001"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/reassign", "POST"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/incidents/{id}/merge
# ---------------------------------------------------------------------------


class TestMergeIncidents:
    """Tests for POST /api/v1/incidents/{id}/merge."""

    @pytest.mark.asyncio
    async def test_merge_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"source_incident_ids": ["PINC002", "PINC003"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert "merged" in data["message"].lower()
        mock_connector.merge_incidents.assert_awaited_once()
        mock_circuit_breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_empty_source_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"source_incident_ids": []})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_merge_missing_source_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_merge_invalid_incident_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"source_incident_ids": ["PINC002"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/merge", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_merge_invalid_source_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"source_incident_ids": ["bad id!!"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_merge_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(_body={"source_incident_ids": ["PINC002"]})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_merge_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.merge_incidents.side_effect = AttributeError("merge failed")
        request = MockRequest(_body={"source_incident_ids": ["PINC002"]})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/merge", "POST"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/v1/incidents/{id}/notes
# ---------------------------------------------------------------------------


class TestListNotes:
    """Tests for GET /api/v1/incidents/{id}/notes."""

    @pytest.mark.asyncio
    async def test_list_notes_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "GET"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 1
        assert data["notes"][0]["id"] == "PNOTE001"
        assert data["notes"][0]["content"] == "Investigation note"
        assert data["notes"][0]["user"]["id"] == "PUSR001"
        mock_connector.list_notes.assert_awaited_once_with("PINC001")

    @pytest.mark.asyncio
    async def test_list_notes_invalid_incident_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/notes", "GET"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_notes_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest()
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "GET"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_notes_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.list_notes.side_effect = ConnectionError("fail")
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "GET"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/incidents/{id}/notes
# ---------------------------------------------------------------------------


class TestAddNote:
    """Tests for POST /api/v1/incidents/{id}/notes."""

    @pytest.mark.asyncio
    async def test_add_note_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"content": "Found root cause"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "POST"
            )
        assert _status(result) == 201
        body = _body(result)
        assert body["message"] == "Note added"
        assert body["note"]["id"] == "PNOTE001"
        mock_connector.add_note.assert_awaited_once_with("PINC001", "Found root cause")

    @pytest.mark.asyncio
    async def test_add_note_missing_content(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "POST"
            )
        assert _status(result) == 400
        assert "content" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_add_note_too_long_content(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"content": "x" * 5001})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "POST"
            )
        assert _status(result) == 400
        assert "length" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_add_note_invalid_incident_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={"content": "Note"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/bad!id/notes", "POST"
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_add_note_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(_body={"content": "Note"})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "POST"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_add_note_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.add_note.side_effect = TimeoutError("timeout")
        request = MockRequest(_body={"content": "Note"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "POST"
            )
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/v1/oncall
# ---------------------------------------------------------------------------


class TestGetOnCall:
    """Tests for GET /api/v1/oncall."""

    @pytest.mark.asyncio
    async def test_get_oncall_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 1
        assert data["oncall"][0]["schedule_id"] == "PSCHED001"
        assert data["oncall"][0]["user"]["id"] == "PUSR001"
        assert data["oncall"][0]["escalation_level"] == 1

    @pytest.mark.asyncio
    async def test_get_oncall_with_schedule_ids(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"schedule_ids": "PSCHED001,PSCHED002"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.get_on_call.call_args[1]
        assert call_kwargs["schedule_ids"] == ["PSCHED001", "PSCHED002"]

    @pytest.mark.asyncio
    async def test_get_oncall_invalid_schedule_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"schedule_ids": "bad id!!"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_oncall_circuit_breaker_open(self, handler, mock_circuit_breaker_open):
        request = MockRequest(query={})
        with _patch_cb(mock_circuit_breaker_open):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_oncall_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_oncall_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.get_on_call.side_effect = ConnectionError("fail")
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        assert _status(result) == 500
        mock_circuit_breaker.record_failure.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/v1/oncall/services/{id}
# ---------------------------------------------------------------------------


class TestGetOnCallForService:
    """Tests for GET /api/v1/oncall/services/{id}."""

    @pytest.mark.asyncio
    async def test_oncall_for_service_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/oncall/services/PSVC001", "GET"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["service_id"] == "PSVC001"
        assert len(data["oncall"]) == 1
        mock_connector.get_current_on_call_for_service.assert_awaited_once_with("PSVC001")

    @pytest.mark.asyncio
    async def test_oncall_for_service_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/oncall/services/PSVC001", "GET"
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_oncall_for_service_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.get_current_on_call_for_service.side_effect = ValueError("fail")
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/oncall/services/PSVC001", "GET"
            )
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/services
# ---------------------------------------------------------------------------


class TestListServices:
    """Tests for GET /api/v1/services."""

    @pytest.mark.asyncio
    async def test_list_services_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["count"] == 1
        assert data["has_more"] is False
        assert data["services"][0]["id"] == "PSVC001"
        assert data["services"][0]["name"] == "Test Service"
        assert data["services"][0]["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_services_with_pagination(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={"limit": "5", "offset": "10"})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 200
        call_kwargs = mock_connector.list_services.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["offset"] == 10

    @pytest.mark.asyncio
    async def test_list_services_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_services_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.list_services.side_effect = OSError("network error")
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/services/{id}
# ---------------------------------------------------------------------------


class TestGetService:
    """Tests for GET /api/v1/services/{id}."""

    @pytest.mark.asyncio
    async def test_get_service_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services/PSVC001", "GET")
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["service"]["id"] == "PSVC001"
        assert data["service"]["name"] == "Test Service"
        assert data["service"]["escalation_policy_id"] == "PESCPOL001"
        mock_connector.get_service.assert_awaited_once_with("PSVC001")

    @pytest.mark.asyncio
    async def test_get_service_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services/PSVC001", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_service_connector_error(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.get_service.side_effect = AttributeError("missing attribute")
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services/PSVC001", "GET")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/webhooks/pagerduty
# ---------------------------------------------------------------------------


class TestPagerDutyWebhook:
    """Tests for POST /api/v1/webhooks/pagerduty."""

    @pytest.mark.asyncio
    async def test_webhook_success(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(
            _body={
                "event": {
                    "event_type": "incident.triggered",
                    "data": {"id": "PINC001"},
                },
            },
            headers={"X-PagerDuty-Signature": "valid-sig"},
        )
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/webhooks/pagerduty", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["received"] is True
        assert data["event_type"] == "incident.triggered"

    @pytest.mark.asyncio
    async def test_webhook_no_connector(self, handler, mock_circuit_breaker):
        request = MockRequest(
            _body={
                "event": {
                    "event_type": "incident.acknowledged",
                    "data": {},
                },
            },
        )
        with _patch_connector(None), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/webhooks/pagerduty", "POST"
            )
        # Webhook still returns 200 even without connector (fallback parsing)
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["received"] is True
        assert data["event_type"] == "incident.acknowledged"

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.verify_webhook_signature.return_value = False
        request = MockRequest(
            _body={"event": {"event_type": "incident.triggered"}},
            headers={"X-PagerDuty-Signature": "bad-sig"},
        )
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/webhooks/pagerduty", "POST"
            )
        # Handler logs warning but does not reject - still processes
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_webhook_with_emitter(self, handler, mock_connector, mock_circuit_breaker):
        emitter = MagicMock()
        handler_with_emitter = DevOpsHandler({"emitter": emitter})
        request = MockRequest(
            _body={"event": {"event_type": "incident.triggered"}},
        )
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler_with_emitter.handle(
                request, "/api/v1/webhooks/pagerduty", "POST"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_webhook_error_still_returns_success(self, handler, mock_connector, mock_circuit_breaker):
        """Webhook errors return 200 with error info to avoid PagerDuty retries."""
        mock_connector.parse_webhook.side_effect = ValueError("parse error")
        request = MockRequest(
            _body={"event": {"event_type": "incident.triggered"}},
        )
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/webhooks/pagerduty", "POST"
            )
        assert _status(result) == 200
        data = _body(result)["data"]
        assert data["received"] is True


# ---------------------------------------------------------------------------
# Not Found / Routing Edge Cases
# ---------------------------------------------------------------------------


class TestRoutingEdgeCases:
    """Tests for routing edge cases and 404 responses."""

    @pytest.mark.asyncio
    async def test_unknown_incident_action(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/unknown_action", "GET"
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_to_list_incidents_wrong_method_on_action(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            # GET on acknowledge should 404
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/acknowledge", "GET"
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_put_on_incidents_returns_404(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "PUT")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_on_incidents_returns_404(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "DELETE")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_on_oncall_returns_404(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_on_services_returns_404(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "POST")
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Path Parameter Extraction
# ---------------------------------------------------------------------------


class TestPathParameterExtraction:
    """Tests for correct path parameter extraction from URLs."""

    @pytest.mark.asyncio
    async def test_incident_id_extracted_correctly(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(request, "/api/v1/incidents/ABCDEF123", "GET")
        mock_connector.get_incident.assert_awaited_once_with("ABCDEF123")

    @pytest.mark.asyncio
    async def test_service_id_extracted_from_services_path(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(request, "/api/v1/services/SVCXYZ789", "GET")
        mock_connector.get_service.assert_awaited_once_with("SVCXYZ789")

    @pytest.mark.asyncio
    async def test_service_id_extracted_from_oncall_services_path(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(
                request, "/api/v1/oncall/services/SVCABC123", "GET"
            )
        mock_connector.get_current_on_call_for_service.assert_awaited_once_with("SVCABC123")

    @pytest.mark.asyncio
    async def test_incident_id_for_acknowledge(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(
                request, "/api/v1/incidents/INC999/acknowledge", "POST"
            )
        mock_connector.acknowledge_incident.assert_awaited_once_with("INC999")

    @pytest.mark.asyncio
    async def test_incident_id_for_resolve(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(_body={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(
                request, "/api/v1/incidents/INC888/resolve", "POST"
            )
        mock_connector.resolve_incident.assert_awaited_once_with("INC888", None)

    @pytest.mark.asyncio
    async def test_incident_id_for_notes_list(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            await handler.handle(
                request, "/api/v1/incidents/INC777/notes", "GET"
            )
        mock_connector.list_notes.assert_awaited_once_with("INC777")


# ---------------------------------------------------------------------------
# Tenant ID Extraction
# ---------------------------------------------------------------------------


class TestTenantIdExtraction:
    """Tests for tenant ID extraction from request."""

    @pytest.mark.asyncio
    async def test_tenant_id_from_request(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(tenant_id="tenant-42", query={})
        with _patch_connector(mock_connector) as mock_get_connector, _patch_cb(mock_circuit_breaker):
            await handler.handle(request, "/api/v1/services", "GET")
        mock_get_connector.assert_awaited_once_with("tenant-42")

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        # Remove tenant_id attribute
        del request.tenant_id
        with _patch_connector(mock_connector) as mock_get_connector, _patch_cb(mock_circuit_breaker):
            await handler.handle(request, "/api/v1/services", "GET")
        mock_get_connector.assert_awaited_once_with("default")


# ---------------------------------------------------------------------------
# Handle Signature Tests
# ---------------------------------------------------------------------------


class TestHandleSignature:
    """Tests for the handle method's flexible signature."""

    @pytest.mark.asyncio
    async def test_handle_with_string_path(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_legacy_pattern_raises(self, handler):
        with pytest.raises(TypeError, match="expects.*request.*path.*method"):
            await handler.handle("/api/v1/services", {"key": "value"}, MagicMock())

    @pytest.mark.asyncio
    async def test_handle_default_method_is_get(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            # Pass None for method -- handler defaults to "GET"
            result = await handler.handle(request, "/api/v1/services", None)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Error Handling in handle() outer try/except
# ---------------------------------------------------------------------------


class TestOuterErrorHandling:
    """Tests for the outer error handling in handle()."""

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_cb(mock_circuit_breaker), patch.object(
            handler, "_get_tenant_id", side_effect=RuntimeError("unexpected")
        ):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 500
        assert "internal server error" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_cb(mock_circuit_breaker), patch.object(
            handler, "_get_tenant_id", side_effect=KeyError("missing key")
        ):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_cb(mock_circuit_breaker), patch.object(
            handler, "_get_tenant_id", side_effect=OSError("disk full")
        ):
            result = await handler.handle(request, "/api/v1/services", "GET")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Response Format Verification
# ---------------------------------------------------------------------------


class TestResponseFormat:
    """Tests verifying correct response format for various endpoints."""

    @pytest.mark.asyncio
    async def test_list_incidents_response_shape(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        incident = MockIncident(created_at=None)
        mock_connector.list_incidents.return_value = ([incident], True)
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents", "GET")
        data = _body(result)["data"]
        inc = data["incidents"][0]
        assert "id" in inc
        assert "title" in inc
        assert "status" in inc
        assert "urgency" in inc
        assert "service_id" in inc
        assert "service_name" in inc
        assert "incident_number" in inc
        assert "created_at" in inc
        assert "html_url" in inc
        assert inc["created_at"] is None
        assert data["has_more"] is True

    @pytest.mark.asyncio
    async def test_get_incident_response_has_extra_fields(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/incidents/PINC001", "GET")
        inc = _body(result)["data"]["incident"]
        # Get incident includes extra fields not in list
        assert "description" in inc
        assert "assignees" in inc
        assert "priority" in inc

    @pytest.mark.asyncio
    async def test_list_notes_response_with_user(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "GET"
            )
        note = _body(result)["data"]["notes"][0]
        assert note["user"]["id"] == "PUSR001"
        assert note["user"]["name"] == "Test User"

    @pytest.mark.asyncio
    async def test_list_notes_response_without_user(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.list_notes.return_value = [MockNote(user=None)]
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(
                request, "/api/v1/incidents/PINC001/notes", "GET"
            )
        note = _body(result)["data"]["notes"][0]
        assert note["user"] is None

    @pytest.mark.asyncio
    async def test_get_service_response_shape(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services/PSVC001", "GET")
        svc = _body(result)["data"]["service"]
        assert "id" in svc
        assert "name" in svc
        assert "description" in svc
        assert "status" in svc
        assert "escalation_policy_id" in svc
        assert "html_url" in svc
        assert "created_at" in svc

    @pytest.mark.asyncio
    async def test_get_service_response_without_created_at(self, handler, mock_connector, mock_circuit_breaker):
        mock_connector.get_service.return_value = MockService(created_at=None)
        request = MockRequest()
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/services/PSVC001", "GET")
        svc = _body(result)["data"]["service"]
        assert svc["created_at"] is None

    @pytest.mark.asyncio
    async def test_oncall_response_has_start_end(self, handler, mock_connector, mock_circuit_breaker):
        request = MockRequest(query={})
        with _patch_connector(mock_connector), _patch_cb(mock_circuit_breaker):
            result = await handler.handle(request, "/api/v1/oncall", "GET")
        sched = _body(result)["data"]["oncall"][0]
        assert "start" in sched
        assert "end" in sched
        assert "escalation_level" in sched


# ---------------------------------------------------------------------------
# Utility method tests
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for internal utility methods."""

    def test_get_query_params_from_query_attr(self, handler):
        request = MockRequest(query={"key": "val"})
        params = handler._get_query_params(request)
        assert params["key"] == "val"

    def test_get_query_params_from_query_string(self, handler):
        request = MagicMock(spec=[])
        request.query_string = "key=val&other=123"
        params = handler._get_query_params(request)
        assert params["key"] == "val"
        assert params["other"] == "123"

    def test_get_query_params_no_query(self, handler):
        request = MagicMock(spec=[])
        params = handler._get_query_params(request)
        assert params == {}

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self, handler):
        request = MockRequest(_body={"key": "value"})
        body = await handler._get_json_body(request)
        assert body == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_property(self, handler):
        request = MagicMock(spec=[])
        request.json = {"key": "value"}
        body = await handler._get_json_body(request)
        assert body == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_no_json(self, handler):
        request = MagicMock(spec=[])
        body = await handler._get_json_body(request)
        assert body == {}

    @pytest.mark.asyncio
    async def test_get_raw_body_callable(self, handler):
        request = MockRequest()
        raw = await handler._get_raw_body(request)
        assert isinstance(raw, bytes)

    @pytest.mark.asyncio
    async def test_get_raw_body_read(self, handler):
        request = MagicMock(spec=[])
        request.read = AsyncMock(return_value=b"raw data")
        raw = await handler._get_raw_body(request)
        assert raw == b"raw data"

    @pytest.mark.asyncio
    async def test_get_raw_body_no_method(self, handler):
        request = MagicMock(spec=[])
        raw = await handler._get_raw_body(request)
        assert raw == b""

    def test_get_header(self, handler):
        request = MockRequest(headers={"X-Custom": "test"})
        assert handler._get_header(request, "X-Custom") == "test"

    def test_get_header_missing(self, handler):
        request = MockRequest(headers={})
        assert handler._get_header(request, "X-Custom") is None

    def test_get_header_no_headers(self, handler):
        request = MagicMock(spec=[])
        assert handler._get_header(request, "X-Custom") is None

    def test_get_tenant_id_from_request(self, handler):
        request = MockRequest(tenant_id="t-123")
        assert handler._get_tenant_id(request) == "t-123"

    def test_get_tenant_id_default(self, handler):
        request = MagicMock(spec=[])
        assert handler._get_tenant_id(request) == "default"

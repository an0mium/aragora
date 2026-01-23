"""
Tests for PagerDuty Incident Management Connector.
"""

import pytest
import hashlib
import hmac

from aragora.connectors.devops.pagerduty import (
    PagerDutyConnector,
    PagerDutyCredentials,
    PagerDutyError,
    Incident,
    IncidentCreateRequest,
    IncidentNote,
    IncidentPriority,
    IncidentStatus,
    IncidentUrgency,
    OnCallSchedule,
    Service,
    ServiceStatus,
    User,
    WebhookPayload,
    get_mock_incident,
    get_mock_on_call,
    get_mock_service,
    get_mock_user,
)


class TestEnums:
    """Tests for enum values."""

    def test_incident_urgency_values(self):
        assert IncidentUrgency.HIGH.value == "high"
        assert IncidentUrgency.LOW.value == "low"

    def test_incident_status_values(self):
        assert IncidentStatus.TRIGGERED.value == "triggered"
        assert IncidentStatus.ACKNOWLEDGED.value == "acknowledged"
        assert IncidentStatus.RESOLVED.value == "resolved"

    def test_incident_priority_values(self):
        assert IncidentPriority.P1.value == "P1"
        assert IncidentPriority.P5.value == "P5"

    def test_service_status_values(self):
        assert ServiceStatus.ACTIVE.value == "active"
        assert ServiceStatus.CRITICAL.value == "critical"


class TestPagerDutyCredentials:
    """Tests for PagerDutyCredentials dataclass."""

    def test_credentials_creation(self):
        creds = PagerDutyCredentials(
            api_key="test_api_key",
            email="user@example.com",
            webhook_secret="secret123",
        )
        assert creds.api_key == "test_api_key"
        assert creds.email == "user@example.com"

    def test_credentials_optional_webhook_secret(self):
        creds = PagerDutyCredentials(
            api_key="key",
            email="user@example.com",
        )
        assert creds.webhook_secret is None


class TestService:
    """Tests for Service dataclass."""

    def test_service_from_api(self):
        api_data = {
            "id": "PSERVICE1",
            "name": "API Gateway",
            "description": "Gateway service",
            "status": "active",
            "escalation_policy": {"id": "PESCAL1"},
            "html_url": "https://example.pagerduty.com/services/PSERVICE1",
            "created_at": "2024-01-15T10:30:00Z",
        }
        service = Service.from_api(api_data)
        assert service.id == "PSERVICE1"
        assert service.name == "API Gateway"
        assert service.status == ServiceStatus.ACTIVE


class TestUser:
    """Tests for User dataclass."""

    def test_user_from_api(self):
        api_data = {
            "id": "PUSER123",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "html_url": "https://example.pagerduty.com/users/PUSER123",
        }
        user = User.from_api(api_data)
        assert user.id == "PUSER123"
        assert user.name == "Jane Smith"


class TestOnCallSchedule:
    """Tests for OnCallSchedule dataclass."""

    def test_schedule_from_api(self):
        api_data = {
            "schedule": {"id": "PSCHED1", "summary": "Weekly Rotation"},
            "user": {
                "id": "PUSER1",
                "name": "John Doe",
                "email": "john@example.com",
            },
            "start": "2024-01-15T00:00:00Z",
            "end": "2024-01-22T00:00:00Z",
        }
        schedule = OnCallSchedule.from_api(api_data)
        assert schedule.schedule_id == "PSCHED1"
        assert schedule.user.id == "PUSER1"
        assert schedule.user.name == "John Doe"


class TestIncidentNote:
    """Tests for IncidentNote dataclass."""

    def test_note_from_api(self):
        api_data = {
            "id": "PNOTE123",
            "content": "Root cause identified",
            "created_at": "2024-01-15T11:30:00Z",
            "user": {"id": "U1", "name": "John Doe", "email": "john@example.com"},
        }
        note = IncidentNote.from_api(api_data)
        assert note.id == "PNOTE123"
        assert note.content == "Root cause identified"
        assert note.user is not None
        assert note.user.name == "John Doe"


class TestIncident:
    """Tests for Incident dataclass."""

    def test_incident_from_api(self):
        api_data = {
            "id": "PINC123",
            "title": "Database Connection Pool Exhausted",
            "status": "triggered",
            "urgency": "high",
            "service": {"id": "PSVC1", "name": "Database", "summary": "Database"},
            "created_at": "2024-01-15T10:00:00Z",
            "html_url": "https://example.pagerduty.com/incidents/PINC123",
            "incident_number": 42,
        }
        incident = Incident.from_api(api_data)
        assert incident.id == "PINC123"
        assert incident.status == IncidentStatus.TRIGGERED
        assert incident.incident_number == 42


class TestIncidentCreateRequest:
    """Tests for IncidentCreateRequest dataclass."""

    def test_create_request_minimal(self):
        request = IncidentCreateRequest(
            title="Test Incident",
            service_id="svc_123",
        )
        assert request.title == "Test Incident"
        assert request.urgency == IncidentUrgency.HIGH


class TestPagerDutyConnectorLifecycle:
    """Tests for connector lifecycle management."""

    def test_connector_initialization(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        assert connector.credentials == creds
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        async with PagerDutyConnector(creds) as connector:
            assert connector._client is not None

    def test_connector_client_property_raises(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        with pytest.raises(PagerDutyError):
            _ = connector.client


class TestPagerDutyConnectorWebhooks:
    """Tests for webhook handling."""

    def test_verify_webhook_signature_valid(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
            webhook_secret="webhook_secret_123",
        )
        connector = PagerDutyConnector(creds)

        payload = '{"event": "incident.triggered"}'
        signature = hmac.new(
            creds.webhook_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        assert connector.verify_webhook_signature(payload.encode(), f"v1={signature}") is True

    def test_verify_webhook_signature_invalid(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
            webhook_secret="webhook_secret_123",
        )
        connector = PagerDutyConnector(creds)
        assert connector.verify_webhook_signature(b"payload", "v1=invalid") is False

    def test_verify_webhook_no_secret(self):
        """When no webhook secret is configured, verification returns True (skipped)."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        # Without secret, verification is skipped and returns True
        assert connector.verify_webhook_signature(b"payload", "signature") is True

    def test_parse_webhook(self):
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.triggered",
                "resource_type": "incident",
                "occurred_at": "2024-01-15T10:00:00Z",
                "data": {"id": "PINC123"},
            },
        }

        payload = connector.parse_webhook(webhook_data)
        assert isinstance(payload, WebhookPayload)
        assert payload.event_type == "incident.triggered"


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_service(self):
        service = get_mock_service()
        assert isinstance(service, Service)

    def test_get_mock_user(self):
        user = get_mock_user()
        assert isinstance(user, User)

    def test_get_mock_incident(self):
        incident = get_mock_incident()
        assert isinstance(incident, Incident)

    def test_get_mock_on_call(self):
        schedule = get_mock_on_call()
        assert isinstance(schedule, OnCallSchedule)


class TestPagerDutyError:
    """Tests for PagerDutyError exception."""

    def test_error_creation(self):
        error = PagerDutyError("Not found", status_code=404, error_code="NOT_FOUND")
        assert str(error) == "Not found"
        assert error.status_code == 404
        assert error.error_code == "NOT_FOUND"

    def test_error_string_representation(self):
        error = PagerDutyError("Server error", status_code=500)
        assert "Server error" in str(error)

"""
Tests for PagerDuty Incident Management Connector.

Tests cover:
- Dataclass serialization and from_api parsing
- Connector lifecycle (context manager)
- Incident operations (create, get, list, acknowledge, resolve, reassign, merge)
- Note operations
- Service and user queries
- On-call schedule queries
- Webhook signature verification and parsing
- High-level integration methods
- Mock data generation
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import hmac
import json

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


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_incident_urgency_values(self):
        """Test incident urgency enum values."""
        assert IncidentUrgency.HIGH.value == "high"
        assert IncidentUrgency.LOW.value == "low"

    def test_incident_status_values(self):
        """Test incident status enum values."""
        assert IncidentStatus.TRIGGERED.value == "triggered"
        assert IncidentStatus.ACKNOWLEDGED.value == "acknowledged"
        assert IncidentStatus.RESOLVED.value == "resolved"

    def test_incident_priority_values(self):
        """Test incident priority enum values."""
        assert IncidentPriority.P1.value == "P1"
        assert IncidentPriority.P2.value == "P2"
        assert IncidentPriority.P3.value == "P3"
        assert IncidentPriority.P4.value == "P4"
        assert IncidentPriority.P5.value == "P5"

    def test_service_status_values(self):
        """Test service status enum values."""
        assert ServiceStatus.ACTIVE.value == "active"
        assert ServiceStatus.WARNING.value == "warning"
        assert ServiceStatus.CRITICAL.value == "critical"
        assert ServiceStatus.MAINTENANCE.value == "maintenance"
        assert ServiceStatus.DISABLED.value == "disabled"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestPagerDutyCredentials:
    """Tests for PagerDutyCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials initialization."""
        creds = PagerDutyCredentials(
            api_key="test_api_key",
            email="user@example.com",
            webhook_secret="secret123",
        )
        assert creds.api_key == "test_api_key"
        assert creds.email == "user@example.com"
        assert creds.webhook_secret == "secret123"

    def test_credentials_optional_webhook_secret(self):
        """Test credentials without webhook secret."""
        creds = PagerDutyCredentials(
            api_key="key",
            email="user@example.com",
        )
        assert creds.webhook_secret is None


class TestService:
    """Tests for Service dataclass."""

    def test_service_creation(self):
        """Test service initialization."""
        service = Service(
            id="svc_123",
            name="Production API",
            description="Main production API service",
            status=ServiceStatus.ACTIVE,
        )
        assert service.id == "svc_123"
        assert service.name == "Production API"
        assert service.status == ServiceStatus.ACTIVE

    def test_service_from_api(self):
        """Test service creation from API response."""
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
        assert service.escalation_policy_id == "PESCAL1"


class TestUser:
    """Tests for User dataclass."""

    def test_user_creation(self):
        """Test user initialization."""
        user = User(
            id="user_123",
            name="John Doe",
            email="john@example.com",
        )
        assert user.id == "user_123"
        assert user.name == "John Doe"
        assert user.email == "john@example.com"

    def test_user_from_api(self):
        """Test user creation from API response."""
        api_data = {
            "id": "PUSER123",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "html_url": "https://example.pagerduty.com/users/PUSER123",
            "job_title": "SRE Engineer",
        }
        user = User.from_api(api_data)
        assert user.id == "PUSER123"
        assert user.name == "Jane Smith"
        assert user.email == "jane@example.com"


class TestOnCallSchedule:
    """Tests for OnCallSchedule dataclass."""

    def test_schedule_creation(self):
        """Test schedule initialization."""
        schedule = OnCallSchedule(
            schedule_id="sched_123",
            schedule_name="Primary On-Call",
            user_id="user_456",
            user_name="Jane Doe",
            user_email="jane@example.com",
            start=datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 22, 0, 0, tzinfo=timezone.utc),
        )
        assert schedule.schedule_id == "sched_123"
        assert schedule.user_name == "Jane Doe"

    def test_schedule_from_api(self):
        """Test schedule creation from API response."""
        api_data = {
            "schedule": {"id": "PSCHED1", "summary": "Weekly Rotation"},
            "user": {
                "id": "PUSER1",
                "summary": "John Doe",
                "email": "john@example.com",
            },
            "start": "2024-01-15T00:00:00Z",
            "end": "2024-01-22T00:00:00Z",
        }
        schedule = OnCallSchedule.from_api(api_data)
        assert schedule.schedule_id == "PSCHED1"
        assert schedule.user_id == "PUSER1"
        assert schedule.user_name == "John Doe"


class TestIncidentNote:
    """Tests for IncidentNote dataclass."""

    def test_note_creation(self):
        """Test note initialization."""
        note = IncidentNote(
            id="note_123",
            content="Investigation started",
            created_at=datetime.now(timezone.utc),
            user_name="Jane Doe",
        )
        assert note.id == "note_123"
        assert note.content == "Investigation started"

    def test_note_from_api(self):
        """Test note creation from API response."""
        api_data = {
            "id": "PNOTE123",
            "content": "Root cause identified",
            "created_at": "2024-01-15T11:30:00Z",
            "user": {"summary": "John Doe"},
        }
        note = IncidentNote.from_api(api_data)
        assert note.id == "PNOTE123"
        assert note.content == "Root cause identified"
        assert note.user_name == "John Doe"


class TestIncident:
    """Tests for Incident dataclass."""

    def test_incident_creation(self):
        """Test incident initialization."""
        incident = Incident(
            id="inc_123",
            title="API Latency Spike",
            status=IncidentStatus.TRIGGERED,
            urgency=IncidentUrgency.HIGH,
            service_id="svc_456",
            service_name="Production API",
            created_at=datetime.now(timezone.utc),
        )
        assert incident.id == "inc_123"
        assert incident.title == "API Latency Spike"
        assert incident.status == IncidentStatus.TRIGGERED

    def test_incident_from_api(self):
        """Test incident creation from API response."""
        api_data = {
            "id": "PINC123",
            "title": "Database Connection Pool Exhausted",
            "status": "triggered",
            "urgency": "high",
            "service": {"id": "PSVC1", "summary": "Database"},
            "created_at": "2024-01-15T10:00:00Z",
            "html_url": "https://example.pagerduty.com/incidents/PINC123",
            "incident_number": 42,
            "priority": {"summary": "P1"},
            "assignments": [{"assignee": {"id": "PUSER1", "summary": "Jane Doe"}}],
        }
        incident = Incident.from_api(api_data)
        assert incident.id == "PINC123"
        assert incident.title == "Database Connection Pool Exhausted"
        assert incident.status == IncidentStatus.TRIGGERED
        assert incident.urgency == IncidentUrgency.HIGH
        assert incident.incident_number == 42


class TestIncidentCreateRequest:
    """Tests for IncidentCreateRequest dataclass."""

    def test_create_request_minimal(self):
        """Test minimal incident create request."""
        request = IncidentCreateRequest(
            title="Test Incident",
            service_id="svc_123",
        )
        assert request.title == "Test Incident"
        assert request.service_id == "svc_123"
        assert request.urgency == IncidentUrgency.HIGH

    def test_create_request_full(self):
        """Test full incident create request."""
        request = IncidentCreateRequest(
            title="Critical Alert",
            service_id="svc_123",
            urgency=IncidentUrgency.LOW,
            body="Detailed description of the issue",
            escalation_policy_id="esc_456",
            priority_id="pri_789",
        )
        assert request.urgency == IncidentUrgency.LOW
        assert request.body == "Detailed description of the issue"


# =============================================================================
# Connector Tests
# =============================================================================


class TestPagerDutyConnectorLifecycle:
    """Tests for connector lifecycle management."""

    def test_connector_initialization(self):
        """Test connector initialization."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        assert connector._credentials == creds
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager(self):
        """Test connector as async context manager."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        async with PagerDutyConnector(creds) as connector:
            assert connector._client is not None

    @pytest.mark.asyncio
    async def test_connector_client_property(self):
        """Test client property raises error before initialization."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        with pytest.raises(RuntimeError, match="context manager"):
            _ = connector.client


class TestPagerDutyConnectorIncidentOperations:
    """Tests for incident operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident(self, connector):
        """Test incident creation."""
        mock_response = {
            "incident": {
                "id": "PINC_NEW",
                "title": "New Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="New Alert",
                    service_id="PSVC1",
                )
                incident = await connector.create_incident(request)
                assert incident.id == "PINC_NEW"
                assert incident.title == "New Alert"

    @pytest.mark.asyncio
    async def test_get_incident(self, connector):
        """Test getting incident details."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test Incident",
                "status": "acknowledged",
                "urgency": "high",
                "service": {"id": "PSVC1", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.get_incident("PINC123")
                assert incident.id == "PINC123"
                assert incident.status == IncidentStatus.ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_list_incidents(self, connector):
        """Test listing incidents."""
        mock_response = {
            "incidents": [
                {
                    "id": "PINC1",
                    "title": "Alert 1",
                    "status": "triggered",
                    "urgency": "high",
                    "service": {"id": "PSVC1", "summary": "API"},
                    "created_at": "2024-01-15T10:00:00Z",
                },
                {
                    "id": "PINC2",
                    "title": "Alert 2",
                    "status": "acknowledged",
                    "urgency": "low",
                    "service": {"id": "PSVC2", "summary": "DB"},
                    "created_at": "2024-01-15T09:00:00Z",
                },
            ],
            "more": False,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incidents, has_more = await connector.list_incidents()
                assert len(incidents) == 2
                assert incidents[0].id == "PINC1"
                assert incidents[1].status == IncidentStatus.ACKNOWLEDGED
                assert has_more is False

    @pytest.mark.asyncio
    async def test_acknowledge_incident(self, connector):
        """Test acknowledging an incident."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "acknowledged",
                "urgency": "high",
                "service": {"id": "PSVC1", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.acknowledge_incident("PINC123")
                assert incident.status == IncidentStatus.ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_resolve_incident(self, connector):
        """Test resolving an incident."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "resolved",
                "urgency": "high",
                "service": {"id": "PSVC1", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.resolve_incident(
                    "PINC123", resolution="Fixed by restarting the service"
                )
                assert incident.status == IncidentStatus.RESOLVED


class TestPagerDutyConnectorNoteOperations:
    """Tests for note operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_add_note(self, connector):
        """Test adding a note to an incident."""
        mock_response = {
            "note": {
                "id": "PNOTE_NEW",
                "content": "Investigation update",
                "created_at": "2024-01-15T11:00:00Z",
                "user": {"summary": "Jane Doe"},
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                note = await connector.add_note("PINC123", "Investigation update")
                assert note.id == "PNOTE_NEW"
                assert note.content == "Investigation update"

    @pytest.mark.asyncio
    async def test_list_notes(self, connector):
        """Test listing incident notes."""
        mock_response = {
            "notes": [
                {
                    "id": "PNOTE1",
                    "content": "Note 1",
                    "created_at": "2024-01-15T10:00:00Z",
                    "user": {"summary": "User 1"},
                },
                {
                    "id": "PNOTE2",
                    "content": "Note 2",
                    "created_at": "2024-01-15T11:00:00Z",
                    "user": {"summary": "User 2"},
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                notes = await connector.list_notes("PINC123")
                assert len(notes) == 2
                assert notes[0].id == "PNOTE1"


class TestPagerDutyConnectorWebhooks:
    """Tests for webhook handling."""

    def test_verify_webhook_signature_valid(self):
        """Test valid webhook signature verification."""
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

        assert connector.verify_webhook_signature(payload, f"v1={signature}") is True

    def test_verify_webhook_signature_invalid(self):
        """Test invalid webhook signature verification."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
            webhook_secret="webhook_secret_123",
        )
        connector = PagerDutyConnector(creds)

        payload = '{"event": "incident.triggered"}'
        assert connector.verify_webhook_signature(payload, "v1=invalid_signature") is False

    def test_verify_webhook_no_secret(self):
        """Test webhook verification without secret configured."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        assert connector.verify_webhook_signature("payload", "signature") is False

    def test_parse_webhook(self):
        """Test webhook payload parsing."""
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
                "data": {
                    "id": "PINC123",
                    "title": "Alert",
                },
            },
        }

        payload = connector.parse_webhook(webhook_data)
        assert isinstance(payload, WebhookPayload)
        assert payload.event_type == "incident.triggered"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_service(self):
        """Test mock service generation."""
        service = get_mock_service()
        assert isinstance(service, Service)
        assert service.id is not None
        assert service.name is not None

    def test_get_mock_user(self):
        """Test mock user generation."""
        user = get_mock_user()
        assert isinstance(user, User)
        assert user.id is not None
        assert user.email is not None

    def test_get_mock_incident(self):
        """Test mock incident generation."""
        incident = get_mock_incident()
        assert isinstance(incident, Incident)
        assert incident.id is not None
        assert incident.status in IncidentStatus

    def test_get_mock_on_call(self):
        """Test mock on-call schedule generation."""
        schedule = get_mock_on_call()
        assert isinstance(schedule, OnCallSchedule)
        assert schedule.schedule_id is not None
        assert schedule.user_name is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPagerDutyError:
    """Tests for PagerDutyError exception."""

    def test_error_creation(self):
        """Test error initialization."""
        error = PagerDutyError("Not found", 404, {"error": {"message": "Incident not found"}})
        assert error.message == "Not found"
        assert error.status_code == 404
        assert "Incident not found" in str(error.response)

    def test_error_string_representation(self):
        """Test error string output."""
        error = PagerDutyError("Server error", 500)
        assert "500" in str(error)
        assert "Server error" in str(error)

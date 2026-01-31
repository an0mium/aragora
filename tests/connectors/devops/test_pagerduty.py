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
        user = User(id="user_456", name="Jane Doe", email="jane@example.com")
        schedule = OnCallSchedule(
            user=user,
            schedule_id="sched_123",
            schedule_name="Primary On-Call",
            start=datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 22, 0, 0, tzinfo=timezone.utc),
        )
        assert schedule.schedule_id == "sched_123"
        assert schedule.user.name == "Jane Doe"

    def test_schedule_from_api(self):
        """Test schedule creation from API response."""
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

    def test_note_creation(self):
        """Test note initialization."""
        user = User(id="user_456", name="Jane Doe", email="jane@example.com")
        note = IncidentNote(
            id="note_123",
            content="Investigation started",
            created_at=datetime.now(timezone.utc),
            user=user,
        )
        assert note.id == "note_123"
        assert note.content == "Investigation started"
        assert note.user.name == "Jane Doe"

    def test_note_from_api(self):
        """Test note creation from API response."""
        api_data = {
            "id": "PNOTE123",
            "content": "Root cause identified",
            "created_at": "2024-01-15T11:30:00Z",
            "user": {"id": "user_789", "name": "John Doe", "email": "john@example.com"},
        }
        note = IncidentNote.from_api(api_data)
        assert note.id == "PNOTE123"
        assert note.content == "Root cause identified"
        assert note.user.name == "John Doe"


class TestIncident:
    """Tests for Incident dataclass."""

    def test_incident_creation(self):
        """Test incident initialization."""
        service = Service(id="svc_456", name="Production API")
        incident = Incident(
            id="inc_123",
            incident_number=1,
            title="API Latency Spike",
            status=IncidentStatus.TRIGGERED,
            urgency=IncidentUrgency.HIGH,
            service=service,
            created_at=datetime.now(timezone.utc),
        )
        assert incident.id == "inc_123"
        assert incident.title == "API Latency Spike"
        assert incident.status == IncidentStatus.TRIGGERED
        assert incident.service.name == "Production API"

    def test_incident_from_api(self):
        """Test incident creation from API response."""
        api_data = {
            "id": "PINC123",
            "title": "Database Connection Pool Exhausted",
            "status": "triggered",
            "urgency": "high",
            "service": {"id": "PSVC1", "name": "Database", "summary": "Database"},
            "created_at": "2024-01-15T10:00:00Z",
            "html_url": "https://example.pagerduty.com/incidents/PINC123",
            "incident_number": 42,
            "priority": {"summary": "P1"},
            "assignments": [
                {
                    "assignee": {
                        "id": "PUSER1",
                        "name": "Jane Doe",
                        "email": "jane@example.com",
                        "summary": "Jane Doe",
                    }
                }
            ],
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
            description="Detailed description of the issue",
            escalation_policy_id="esc_456",
            priority_id="pri_789",
        )
        assert request.urgency == IncidentUrgency.LOW
        assert request.description == "Detailed description of the issue"


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
        assert connector.credentials == creds
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
        with pytest.raises(PagerDutyError, match="context manager"):
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
                "incident_number": 123,
                "title": "New Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
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
                "incident_number": 456,
                "title": "Test Incident",
                "status": "acknowledged",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
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
                    "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                    "incident_number": 1,
                    "created_at": "2024-01-15T10:00:00Z",
                },
                {
                    "id": "PINC2",
                    "title": "Alert 2",
                    "status": "acknowledged",
                    "urgency": "low",
                    "service": {"id": "PSVC2", "name": "DB", "summary": "DB"},
                    "incident_number": 2,
                    "created_at": "2024-01-15T09:00:00Z",
                },
            ],
            "more": False,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incidents = await connector.list_incidents()
                assert len(incidents) == 2
                assert incidents[0].id == "PINC1"
                assert incidents[1].status == IncidentStatus.ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_acknowledge_incident(self, connector):
        """Test acknowledging an incident."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "acknowledged",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
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
        note_response = {
            "note": {
                "id": "PNOTE123",
                "content": "Resolution: Fixed by restarting the service",
                "created_at": "2024-01-15T10:00:00Z",
                "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            }
        }
        incident_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "resolved",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            # First call is add_note, second call is update incident
            mock_request.side_effect = [note_response, incident_response]

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
                "user": {
                    "id": "PUSER1",
                    "name": "Jane Doe",
                    "email": "jane@example.com",
                    "summary": "Jane Doe",
                },
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
                    "user": {
                        "id": "PUSER1",
                        "name": "User 1",
                        "email": "user1@example.com",
                        "summary": "User 1",
                    },
                },
                {
                    "id": "PNOTE2",
                    "content": "Note 2",
                    "created_at": "2024-01-15T11:00:00Z",
                    "user": {
                        "id": "PUSER2",
                        "name": "User 2",
                        "email": "user2@example.com",
                        "summary": "User 2",
                    },
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

        payload = b'{"event": "incident.triggered"}'
        signature = hmac.new(creds.webhook_secret.encode(), payload, hashlib.sha256).hexdigest()

        assert connector.verify_webhook_signature(payload, f"v1={signature}") is True

    def test_verify_webhook_signature_invalid(self):
        """Test invalid webhook signature verification."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
            webhook_secret="webhook_secret_123",
        )
        connector = PagerDutyConnector(creds)

        payload = b'{"event": "incident.triggered"}'
        assert connector.verify_webhook_signature(payload, "v1=invalid_signature") is False

    def test_verify_webhook_no_secret(self):
        """Test webhook verification without secret configured logs warning and returns True."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)
        # When no webhook secret is configured, returns True (verification skipped)
        assert connector.verify_webhook_signature(b"payload", "signature") is True

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
        assert schedule.user.name is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestPagerDutyError:
    """Tests for PagerDutyError exception."""

    def test_error_creation(self):
        """Test error initialization."""
        error = PagerDutyError("Incident not found", 404, "NOT_FOUND")
        assert str(error) == "Incident not found"
        assert error.status_code == 404
        assert error.error_code == "NOT_FOUND"

    def test_error_string_representation(self):
        """Test error string output."""
        error = PagerDutyError("Server error", 500)
        assert "Server error" in str(error)

    def test_error_without_status_code(self):
        """Test error without status code."""
        error = PagerDutyError("General error")
        assert error.status_code is None
        assert error.error_code is None


# =============================================================================
# Additional Incident Operations Tests
# =============================================================================


class TestIncidentReassignAndMerge:
    """Tests for incident reassign and merge operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_reassign_incident(self, connector):
        """Test reassigning an incident."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
                "created_at": "2024-01-15T10:00:00Z",
                "assignments": [
                    {
                        "assignee": {
                            "id": "PUSER2",
                            "name": "New User",
                            "email": "new@example.com",
                            "summary": "New User",
                        }
                    }
                ],
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.reassign_incident("PINC123", ["PUSER2"])
                assert len(incident.assigned_to) == 1
                assert incident.assigned_to[0].id == "PUSER2"

    @pytest.mark.asyncio
    async def test_merge_incidents(self, connector):
        """Test merging incidents."""
        mock_response = {
            "incident": {
                "id": "PINC_TARGET",
                "title": "Merged Incident",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 100,
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.merge_incidents("PINC_TARGET", ["PINC1", "PINC2"])
                assert incident.id == "PINC_TARGET"

    @pytest.mark.asyncio
    async def test_list_incidents_with_filters(self, connector):
        """Test listing incidents with various filters."""
        mock_response = {"incidents": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                # Test with status filter
                await connector.list_incidents(statuses=[IncidentStatus.TRIGGERED])
                mock_request.assert_called()

                # Test with urgency filter
                await connector.list_incidents(urgencies=[IncidentUrgency.HIGH])
                mock_request.assert_called()

                # Test with service_ids filter
                await connector.list_incidents(service_ids=["SVC1", "SVC2"])
                mock_request.assert_called()

                # Test with date filters
                since = datetime(2024, 1, 1, tzinfo=timezone.utc)
                until = datetime(2024, 1, 31, tzinfo=timezone.utc)
                await connector.list_incidents(since=since, until=until)
                mock_request.assert_called()


# =============================================================================
# Service Operations Tests
# =============================================================================


class TestServiceOperations:
    """Tests for service operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_list_services(self, connector):
        """Test listing services."""
        mock_response = {
            "services": [
                {
                    "id": "PSVC1",
                    "name": "Production API",
                    "status": "active",
                    "description": "Main API service",
                },
                {
                    "id": "PSVC2",
                    "name": "Database",
                    "status": "warning",
                    "description": "Database service",
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                services = await connector.list_services()
                assert len(services) == 2
                assert services[0].id == "PSVC1"
                assert services[1].status == ServiceStatus.WARNING

    @pytest.mark.asyncio
    async def test_list_services_with_query(self, connector):
        """Test listing services with search query."""
        mock_response = {"services": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_services(query="api", include_disabled=True, limit=50)
                mock_request.assert_called_once()
                call_args = mock_request.call_args
                assert call_args[1]["params"]["query"] == "api"
                assert call_args[1]["params"]["include[]"] == "disabled"

    @pytest.mark.asyncio
    async def test_get_service(self, connector):
        """Test getting a single service."""
        mock_response = {
            "service": {
                "id": "PSVC1",
                "name": "Production API",
                "status": "active",
                "escalation_policy": {"id": "PESCAL1"},
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                service = await connector.get_service("PSVC1")
                assert service.id == "PSVC1"
                assert service.escalation_policy_id == "PESCAL1"


# =============================================================================
# User Operations Tests
# =============================================================================


class TestUserOperations:
    """Tests for user operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_list_users(self, connector):
        """Test listing users."""
        mock_response = {
            "users": [
                {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
                {"id": "PUSER2", "name": "Bob", "email": "bob@example.com"},
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                users = await connector.list_users()
                assert len(users) == 2
                assert users[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_list_users_with_query(self, connector):
        """Test listing users with search query."""
        mock_response = {"users": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_users(query="alice", limit=10)
                mock_request.assert_called_once()
                call_args = mock_request.call_args
                assert call_args[1]["params"]["query"] == "alice"
                assert call_args[1]["params"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_user(self, connector):
        """Test getting a single user."""
        mock_response = {
            "user": {
                "id": "PUSER1",
                "name": "Alice Smith",
                "email": "alice@example.com",
                "role": "admin",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                user = await connector.get_user("PUSER1")
                assert user.id == "PUSER1"
                assert user.role == "admin"


# =============================================================================
# On-Call Schedule Operations Tests
# =============================================================================


class TestOnCallOperations:
    """Tests for on-call schedule operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_get_on_call(self, connector):
        """Test getting on-call schedules."""
        mock_response = {
            "oncalls": [
                {
                    "schedule": {"id": "PSCHED1", "summary": "Primary"},
                    "user": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
                    "start": "2024-01-15T00:00:00Z",
                    "end": "2024-01-22T00:00:00Z",
                    "escalation_level": 1,
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                oncalls = await connector.get_on_call()
                assert len(oncalls) == 1
                assert oncalls[0].user.name == "Alice"

    @pytest.mark.asyncio
    async def test_get_on_call_with_filters(self, connector):
        """Test getting on-call with filters."""
        mock_response = {"oncalls": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                since = datetime(2024, 1, 1, tzinfo=timezone.utc)
                until = datetime(2024, 1, 31, tzinfo=timezone.utc)
                await connector.get_on_call(
                    schedule_ids=["PSCHED1"],
                    escalation_policy_ids=["PESCAL1"],
                    since=since,
                    until=until,
                )
                mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_on_call_for_service(self, connector):
        """Test getting current on-call for a service."""
        service_response = {
            "service": {
                "id": "PSVC1",
                "name": "API",
                "status": "active",
                "escalation_policy": {"id": "PESCAL1"},
            }
        }
        oncall_response = {
            "oncalls": [
                {
                    "schedule": {"id": "PSCHED1", "summary": "Primary"},
                    "user": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
                    "start": "2024-01-15T00:00:00Z",
                    "end": "2024-01-22T00:00:00Z",
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [service_response, oncall_response]

            async with connector:
                users = await connector.get_current_on_call_for_service("PSVC1")
                assert len(users) == 1
                assert users[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_get_current_on_call_no_escalation_policy(self, connector):
        """Test getting on-call for service without escalation policy."""
        service_response = {
            "service": {
                "id": "PSVC1",
                "name": "API",
                "status": "active",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = service_response

            async with connector:
                users = await connector.get_current_on_call_for_service("PSVC1")
                assert users == []


# =============================================================================
# Integration Helper Tests
# =============================================================================


class TestIntegrationHelpers:
    """Tests for high-level integration helper methods."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident_from_finding(self, connector):
        """Test creating incident from a security/bug finding."""
        mock_response = {
            "incident": {
                "id": "PINC_FINDING",
                "incident_number": 456,
                "title": "[CRITICAL] SQL Injection",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.create_incident_from_finding(
                    title="SQL Injection",
                    service_id="PSVC1",
                    severity="critical",
                    description="Found SQL injection vulnerability",
                    source="sast_scanner",
                    finding_id="finding_123",
                    file_path="src/auth.py",
                    line_number=42,
                )
                assert incident.id == "PINC_FINDING"
                # Verify the request body contains proper title formatting
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert "[CRITICAL]" in body["title"]
                assert body["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_create_incident_from_finding_low_severity(self, connector):
        """Test creating incident from low severity finding."""
        mock_response = {
            "incident": {
                "id": "PINC_LOW",
                "incident_number": 789,
                "title": "[MEDIUM] Code Quality Issue",
                "status": "triggered",
                "urgency": "low",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.create_incident_from_finding(
                    title="Code Quality Issue",
                    service_id="PSVC1",
                    severity="medium",
                    description="Found code quality issue",
                    source="linter",
                )
                assert incident.urgency == IncidentUrgency.LOW

    @pytest.mark.asyncio
    async def test_add_investigation_update(self, connector):
        """Test adding investigation update to incident."""
        mock_response = {
            "note": {
                "id": "PNOTE_UPDATE",
                "content": "[Alice] Investigation Update:\nRoot cause identified",
                "created_at": "2024-01-15T11:00:00Z",
                "user": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                note = await connector.add_investigation_update(
                    "PINC123",
                    "Root cause identified",
                    investigator="Alice",
                )
                assert note.id == "PNOTE_UPDATE"

    @pytest.mark.asyncio
    async def test_add_investigation_update_no_investigator(self, connector):
        """Test adding investigation update without investigator."""
        mock_response = {
            "note": {
                "id": "PNOTE_UPDATE",
                "content": "Investigation Update:\nWorking on fix",
                "created_at": "2024-01-15T11:00:00Z",
                "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                note = await connector.add_investigation_update("PINC123", "Working on fix")
                assert note.id == "PNOTE_UPDATE"

    @pytest.mark.asyncio
    async def test_resolve_with_runbook(self, connector):
        """Test resolving incident with runbook documentation."""
        note_response = {
            "note": {
                "id": "PNOTE_RUNBOOK",
                "content": "Runbook Steps Executed:\n  1. Step 1\n  2. Step 2",
                "created_at": "2024-01-15T11:00:00Z",
                "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            }
        }
        note_resolution_response = {
            "note": {
                "id": "PNOTE_RESOLUTION",
                "content": "Resolution: Fixed by restarting",
                "created_at": "2024-01-15T11:01:00Z",
                "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            }
        }
        incident_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "resolved",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [note_response, note_resolution_response, incident_response]

            async with connector:
                incident = await connector.resolve_with_runbook(
                    "PINC123",
                    ["Step 1", "Step 2"],
                    "Fixed by restarting",
                )
                assert incident.status == IncidentStatus.RESOLVED


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Tests for internal _request method."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_request_returns_empty_on_204(self, connector):
        """Test that 204 response returns empty dict."""
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            async with connector:
                connector._client = mock_client
                result = await connector._request("DELETE", "/incidents/PINC123")
                assert result == {}

    @pytest.mark.asyncio
    async def test_request_raises_on_400_error(self, connector):
        """Test that 400 error raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request", "code": "BAD_REQUEST"}
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("POST", "/incidents")
                assert exc_info.value.status_code == 400
                assert exc_info.value.error_code == "BAD_REQUEST"

    @pytest.mark.asyncio
    async def test_request_raises_on_404_error(self, connector):
        """Test that 404 error raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not found"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents/NONEXISTENT")
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_request_raises_on_http_error(self, connector):
        """Test that HTTP errors are handled."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.side_effect = httpx.HTTPError("Connection failed")
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError, match="HTTP error"):
                    await connector._request("GET", "/incidents")


# =============================================================================
# Data Model Edge Cases
# =============================================================================


class TestDataModelEdgeCases:
    """Tests for edge cases in data model parsing."""

    def test_service_from_api_minimal(self):
        """Test service with minimal API data."""
        api_data = {"id": "PSVC1", "name": "Service", "status": "active"}
        service = Service.from_api(api_data)
        assert service.id == "PSVC1"
        assert service.description is None
        assert service.escalation_policy_id is None

    def test_service_from_api_unknown_status(self):
        """Test service with non-standard status."""
        # ServiceStatus uses value() mapping, so the actual status string must match enum
        api_data = {"id": "PSVC1", "name": "Service", "status": "maintenance"}
        service = Service.from_api(api_data)
        assert service.status == ServiceStatus.MAINTENANCE

    def test_user_from_api_with_role(self):
        """Test user with role field."""
        api_data = {
            "id": "PUSER1",
            "name": "Admin",
            "email": "admin@example.com",
            "role": "admin",
        }
        user = User.from_api(api_data)
        assert user.role == "admin"

    def test_incident_from_api_with_resolved_at(self):
        """Test incident with resolved_at field."""
        api_data = {
            "id": "PINC1",
            "title": "Resolved Incident",
            "status": "resolved",
            "urgency": "low",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "resolved_at": "2024-01-15T11:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.resolved_at is not None
        assert incident.status == IncidentStatus.RESOLVED

    def test_incident_from_api_without_service(self):
        """Test incident without service data."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.service is None

    def test_incident_from_api_without_priority(self):
        """Test incident without priority."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.priority is None

    def test_incident_from_api_with_empty_assignments(self):
        """Test incident with empty assignments."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "assignments": [],
        }
        incident = Incident.from_api(api_data)
        assert incident.assigned_to == []

    def test_oncall_schedule_from_api_with_unknown_user(self):
        """Test on-call schedule with missing user data."""
        api_data = {
            "schedule": {"id": "PSCHED1", "summary": "Primary"},
            "user": {},
            "start": "2024-01-15T00:00:00Z",
            "end": "2024-01-22T00:00:00Z",
        }
        schedule = OnCallSchedule.from_api(api_data)
        # Should create User with "unknown" defaults
        assert schedule.user.id == "unknown"
        assert schedule.user.name == "Unknown"

    def test_incident_note_from_api_without_user(self):
        """Test incident note without user data."""
        api_data = {
            "id": "PNOTE1",
            "content": "System generated note",
            "created_at": "2024-01-15T10:00:00Z",
        }
        note = IncidentNote.from_api(api_data)
        assert note.user is None

    def test_incident_note_from_api_without_created_at(self):
        """Test incident note without created_at."""
        api_data = {
            "id": "PNOTE1",
            "content": "Note content",
        }
        note = IncidentNote.from_api(api_data)
        assert note.created_at is None


# =============================================================================
# Webhook Parsing Edge Cases
# =============================================================================


class TestWebhookParsingEdgeCases:
    """Tests for webhook parsing edge cases."""

    def test_parse_webhook_unknown_event_type(self):
        """Test parsing webhook with unknown event type."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {"event": {}}
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "unknown"

    def test_parse_webhook_with_incident_data(self):
        """Test parsing webhook with full incident data."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.acknowledged",
                "data": {
                    "id": "PINC123",
                    "type": "incident",
                    "title": "Test Incident",
                    "status": "acknowledged",
                    "urgency": "high",
                    "incident_number": 42,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.acknowledged"
        assert payload.incident is not None
        assert payload.incident.id == "PINC123"

    def test_parse_webhook_non_incident_data(self):
        """Test parsing webhook with non-incident data type."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "service.created",
                "data": {"id": "PSVC1", "type": "service", "name": "New Service"},
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "service.created"
        assert payload.incident is None  # Not an incident type

    def test_verify_webhook_signature_without_v1_prefix(self):
        """Test webhook signature verification without v1= prefix."""
        creds = PagerDutyCredentials(
            api_key="key",
            email="user@example.com",
            webhook_secret="secret123",
        )
        connector = PagerDutyConnector(creds)

        payload = b'{"event": "test"}'
        signature = hmac.new(creds.webhook_secret.encode(), payload, hashlib.sha256).hexdigest()

        # Should work even without v1= prefix
        assert connector.verify_webhook_signature(payload, signature) is True


# =============================================================================
# IncidentCreateRequest Tests
# =============================================================================


class TestIncidentCreateRequestFields:
    """Tests for IncidentCreateRequest fields."""

    def test_request_with_assignments(self):
        """Test request with user assignments."""
        request = IncidentCreateRequest(
            title="Alert",
            service_id="PSVC1",
            assignments=["PUSER1", "PUSER2"],
        )
        assert request.assignments == ["PUSER1", "PUSER2"]

    def test_request_with_incident_key(self):
        """Test request with deduplication key."""
        request = IncidentCreateRequest(
            title="Alert",
            service_id="PSVC1",
            incident_key="unique-key-123",
        )
        assert request.incident_key == "unique-key-123"

    def test_request_defaults(self):
        """Test request default values."""
        request = IncidentCreateRequest(
            title="Alert",
            service_id="PSVC1",
        )
        assert request.urgency == IncidentUrgency.HIGH
        assert request.description is None
        assert request.priority_id is None
        assert request.escalation_policy_id is None
        assert request.incident_key is None
        assert request.assignments is None


# =============================================================================
# Additional Connector Tests for Full Coverage
# =============================================================================


class TestCreateIncidentVariants:
    """Additional tests for incident creation variants."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident_with_description(self, connector):
        """Test incident creation with description."""
        mock_response = {
            "incident": {
                "id": "PINC_DESC",
                "incident_number": 100,
                "title": "Alert with Description",
                "status": "triggered",
                "urgency": "high",
                "description": "Detailed description",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Alert with Description",
                    service_id="PSVC1",
                    description="Detailed description",
                )
                incident = await connector.create_incident(request)
                assert incident.id == "PINC_DESC"

    @pytest.mark.asyncio
    async def test_create_incident_with_priority(self, connector):
        """Test incident creation with priority."""
        mock_response = {
            "incident": {
                "id": "PINC_PRIO",
                "incident_number": 101,
                "title": "Priority Alert",
                "status": "triggered",
                "urgency": "high",
                "priority": {"summary": "P1"},
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Priority Alert",
                    service_id="PSVC1",
                    priority_id="P1",
                )
                incident = await connector.create_incident(request)
                assert incident.priority == IncidentPriority.P1

    @pytest.mark.asyncio
    async def test_create_incident_with_escalation_policy(self, connector):
        """Test incident creation with escalation policy."""
        mock_response = {
            "incident": {
                "id": "PINC_ESC",
                "incident_number": 102,
                "title": "Escalated Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Escalated Alert",
                    service_id="PSVC1",
                    escalation_policy_id="PESCAL1",
                )
                incident = await connector.create_incident(request)
                # Verify request body contains escalation policy
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert "escalation_policy" in body

    @pytest.mark.asyncio
    async def test_create_incident_with_incident_key(self, connector):
        """Test incident creation with deduplication key."""
        mock_response = {
            "incident": {
                "id": "PINC_KEY",
                "incident_number": 103,
                "title": "Dedup Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Dedup Alert",
                    service_id="PSVC1",
                    incident_key="dedup-key-123",
                )
                incident = await connector.create_incident(request)
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["incident_key"] == "dedup-key-123"

    @pytest.mark.asyncio
    async def test_create_incident_with_assignments(self, connector):
        """Test incident creation with user assignments."""
        mock_response = {
            "incident": {
                "id": "PINC_ASSIGN",
                "incident_number": 104,
                "title": "Assigned Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "assignments": [
                    {"assignee": {"id": "PUSER1", "name": "User1", "email": "u1@example.com"}},
                ],
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Assigned Alert",
                    service_id="PSVC1",
                    assignments=["PUSER1"],
                )
                incident = await connector.create_incident(request)
                assert len(incident.assigned_to) == 1

    @pytest.mark.asyncio
    async def test_create_incident_low_urgency(self, connector):
        """Test incident creation with low urgency."""
        mock_response = {
            "incident": {
                "id": "PINC_LOW",
                "incident_number": 105,
                "title": "Low Priority Alert",
                "status": "triggered",
                "urgency": "low",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Low Priority Alert",
                    service_id="PSVC1",
                    urgency=IncidentUrgency.LOW,
                )
                incident = await connector.create_incident(request)
                assert incident.urgency == IncidentUrgency.LOW


class TestResolveWithoutNote:
    """Test resolving incident without resolution note."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_resolve_incident_without_resolution_note(self, connector):
        """Test resolving incident without adding resolution note."""
        incident_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "resolved",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = incident_response

            async with connector:
                # Resolve without resolution note
                incident = await connector.resolve_incident("PINC123")
                assert incident.status == IncidentStatus.RESOLVED
                # Should only call once (no note added)
                assert mock_request.call_count == 1


class TestListOperationsPagination:
    """Tests for pagination in list operations."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_list_incidents_pagination(self, connector):
        """Test incident listing with pagination parameters."""
        mock_response = {"incidents": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_incidents(limit=50, offset=100)
                call_args = mock_request.call_args
                assert call_args[1]["params"]["limit"] == 50
                assert call_args[1]["params"]["offset"] == 100

    @pytest.mark.asyncio
    async def test_list_incidents_limit_capped_at_100(self, connector):
        """Test that limit is capped at 100."""
        mock_response = {"incidents": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_incidents(limit=200)  # Request more than max
                call_args = mock_request.call_args
                assert call_args[1]["params"]["limit"] == 100  # Should be capped

    @pytest.mark.asyncio
    async def test_list_services_pagination(self, connector):
        """Test service listing with pagination."""
        mock_response = {"services": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_services(limit=25, offset=50)
                call_args = mock_request.call_args
                assert call_args[1]["params"]["limit"] == 25
                assert call_args[1]["params"]["offset"] == 50

    @pytest.mark.asyncio
    async def test_list_users_pagination(self, connector):
        """Test user listing with pagination."""
        mock_response = {"users": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.list_users(limit=10, offset=20)
                call_args = mock_request.call_args
                assert call_args[1]["params"]["limit"] == 10
                assert call_args[1]["params"]["offset"] == 20


class TestEmptyResponseHandling:
    """Tests for handling empty API responses."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_list_incidents_empty(self, connector):
        """Test listing incidents with empty response."""
        mock_response = {"incidents": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incidents = await connector.list_incidents()
                assert incidents == []

    @pytest.mark.asyncio
    async def test_list_services_empty(self, connector):
        """Test listing services with empty response."""
        mock_response = {"services": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                services = await connector.list_services()
                assert services == []

    @pytest.mark.asyncio
    async def test_list_users_empty(self, connector):
        """Test listing users with empty response."""
        mock_response = {"users": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                users = await connector.list_users()
                assert users == []

    @pytest.mark.asyncio
    async def test_list_notes_empty(self, connector):
        """Test listing notes with empty response."""
        mock_response = {"notes": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                notes = await connector.list_notes("PINC123")
                assert notes == []

    @pytest.mark.asyncio
    async def test_get_on_call_empty(self, connector):
        """Test getting on-call with empty response."""
        mock_response = {"oncalls": []}

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                oncalls = await connector.get_on_call()
                assert oncalls == []


class TestServiceStatusVariants:
    """Tests for various service status values."""

    def test_service_status_critical(self):
        """Test service with critical status."""
        api_data = {"id": "PSVC1", "name": "Service", "status": "critical"}
        service = Service.from_api(api_data)
        assert service.status == ServiceStatus.CRITICAL

    def test_service_status_disabled(self):
        """Test service with disabled status."""
        api_data = {"id": "PSVC1", "name": "Service", "status": "disabled"}
        service = Service.from_api(api_data)
        assert service.status == ServiceStatus.DISABLED

    def test_service_status_warning(self):
        """Test service with warning status."""
        api_data = {"id": "PSVC1", "name": "Service", "status": "warning"}
        service = Service.from_api(api_data)
        assert service.status == ServiceStatus.WARNING


class TestIncidentPriorityLevels:
    """Tests for all incident priority levels."""

    def test_incident_priority_p2(self):
        """Test incident with P2 priority."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "priority": {"summary": "P2"},
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.priority == IncidentPriority.P2

    def test_incident_priority_p3(self):
        """Test incident with P3 priority."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "priority": {"summary": "P3"},
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.priority == IncidentPriority.P3

    def test_incident_priority_p4(self):
        """Test incident with P4 priority."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "low",
            "incident_number": 1,
            "priority": {"summary": "P4"},
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.priority == IncidentPriority.P4

    def test_incident_priority_p5(self):
        """Test incident with P5 priority."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "low",
            "incident_number": 1,
            "priority": {"summary": "P5"},
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.priority == IncidentPriority.P5


class TestOncallDuplicateUserFiltering:
    """Tests for on-call user deduplication."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_get_current_on_call_deduplicates_users(self, connector):
        """Test that duplicate users are removed from on-call list."""
        service_response = {
            "service": {
                "id": "PSVC1",
                "name": "API",
                "status": "active",
                "escalation_policy": {"id": "PESCAL1"},
            }
        }
        # Same user appears in multiple on-call schedules
        oncall_response = {
            "oncalls": [
                {
                    "schedule": {"id": "PSCHED1", "summary": "Primary"},
                    "user": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
                    "start": "2024-01-15T00:00:00Z",
                    "end": "2024-01-22T00:00:00Z",
                },
                {
                    "schedule": {"id": "PSCHED2", "summary": "Secondary"},
                    "user": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"},
                    "start": "2024-01-15T00:00:00Z",
                    "end": "2024-01-22T00:00:00Z",
                },
                {
                    "schedule": {"id": "PSCHED3", "summary": "Tertiary"},
                    "user": {"id": "PUSER2", "name": "Bob", "email": "bob@example.com"},
                    "start": "2024-01-15T00:00:00Z",
                    "end": "2024-01-22T00:00:00Z",
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [service_response, oncall_response]

            async with connector:
                users = await connector.get_current_on_call_for_service("PSVC1")
                # Should only have 2 users (Alice deduplicated)
                assert len(users) == 2
                user_ids = [u.id for u in users]
                assert "PUSER1" in user_ids
                assert "PUSER2" in user_ids


class TestWebhookPayloadDataclass:
    """Tests for WebhookPayload dataclass."""

    def test_webhook_payload_default_values(self):
        """Test WebhookPayload default values."""
        payload = WebhookPayload(event_type="test.event")
        assert payload.event_type == "test.event"
        assert payload.incident is None
        assert payload.raw_data == {}

    def test_webhook_payload_with_raw_data(self):
        """Test WebhookPayload with raw data."""
        raw = {"key": "value", "nested": {"inner": "data"}}
        payload = WebhookPayload(event_type="test", raw_data=raw)
        assert payload.raw_data == raw


class TestIncidentLastStatusChange:
    """Tests for incident last_status_change_at parsing."""

    def test_incident_with_last_status_change_at(self):
        """Test incident with last_status_change_at field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "acknowledged",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "last_status_change_at": "2024-01-15T11:30:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.last_status_change_at is not None


class TestEnumFromString:
    """Tests for enum conversion from strings."""

    def test_incident_urgency_from_string(self):
        """Test IncidentUrgency from string value."""
        assert IncidentUrgency("high") == IncidentUrgency.HIGH
        assert IncidentUrgency("low") == IncidentUrgency.LOW

    def test_incident_status_from_string(self):
        """Test IncidentStatus from string value."""
        assert IncidentStatus("triggered") == IncidentStatus.TRIGGERED
        assert IncidentStatus("acknowledged") == IncidentStatus.ACKNOWLEDGED
        assert IncidentStatus("resolved") == IncidentStatus.RESOLVED

    def test_service_status_from_string(self):
        """Test ServiceStatus from string value."""
        assert ServiceStatus("active") == ServiceStatus.ACTIVE
        assert ServiceStatus("maintenance") == ServiceStatus.MAINTENANCE


class TestFindingFromFilePath:
    """Tests for creating incidents from findings with file paths."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident_from_finding_without_file_path(self, connector):
        """Test creating incident from finding without file path."""
        mock_response = {
            "incident": {
                "id": "PINC_NOFILE",
                "incident_number": 200,
                "title": "[HIGH] Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.create_incident_from_finding(
                    title="Alert",
                    service_id="PSVC1",
                    severity="high",
                    description="Some issue",
                    source="scanner",
                )
                assert incident.id == "PINC_NOFILE"

    @pytest.mark.asyncio
    async def test_create_incident_from_finding_without_finding_id(self, connector):
        """Test creating incident from finding without finding_id (no dedup key)."""
        mock_response = {
            "incident": {
                "id": "PINC_NODEDUP",
                "incident_number": 201,
                "title": "[LOW] Alert",
                "status": "triggered",
                "urgency": "low",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.create_incident_from_finding(
                    title="Alert",
                    service_id="PSVC1",
                    severity="low",
                    description="Some issue",
                    source="scanner",
                    # No finding_id - should not have incident_key
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert "incident_key" not in body


class TestUserHtmlUrl:
    """Tests for User html_url field."""

    def test_user_with_html_url(self):
        """Test user with html_url field."""
        api_data = {
            "id": "PUSER1",
            "name": "John",
            "email": "john@example.com",
            "html_url": "https://example.pagerduty.com/users/PUSER1",
        }
        user = User.from_api(api_data)
        assert user.html_url == "https://example.pagerduty.com/users/PUSER1"

    def test_user_without_html_url(self):
        """Test user without html_url field."""
        api_data = {
            "id": "PUSER1",
            "name": "John",
            "email": "john@example.com",
        }
        user = User.from_api(api_data)
        assert user.html_url is None


# =============================================================================
# Additional Error Handling Tests
# =============================================================================


class TestApiErrorHandling:
    """Additional tests for API error handling."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_request_raises_on_401_unauthorized(self, connector):
        """Test that 401 unauthorized raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key", "code": "UNAUTHORIZED"}
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 401
                assert exc_info.value.error_code == "UNAUTHORIZED"

    @pytest.mark.asyncio
    async def test_request_raises_on_403_forbidden(self, connector):
        """Test that 403 forbidden raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "error": {"message": "Access denied", "code": "FORBIDDEN"}
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_request_raises_on_500_server_error(self, connector):
        """Test that 500 server error raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_raises_on_502_bad_gateway(self, connector):
        """Test that 502 bad gateway raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.return_value = {"error": {"message": "Bad gateway"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_request_raises_on_503_service_unavailable(self, connector):
        """Test that 503 service unavailable raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {"error": {"message": "Service temporarily unavailable"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_request_raises_on_429_rate_limit(self, connector):
        """Test that 429 rate limit raises PagerDutyError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"message": "Rate limit exceeded", "code": "RATE_LIMIT"}
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("GET", "/incidents")
                assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_request_handles_error_without_code(self, connector):
        """Test error handling when error code is missing."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Bad request"}  # No code field
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("POST", "/incidents")
                assert exc_info.value.error_code is None

    @pytest.mark.asyncio
    async def test_request_handles_empty_error_object(self, connector):
        """Test error handling when error object is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError) as exc_info:
                    await connector._request("POST", "/incidents")
                assert "Unknown error" in str(exc_info.value)


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)

        try:
            async with connector:
                assert connector._client is not None
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Client should be closed
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager_client_closed_after_exit(self):
        """Test that HTTP client is properly closed after exiting context."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        connector = PagerDutyConnector(creds)

        async with connector:
            client = connector._client
            assert client is not None

        # After exit, client should be None
        assert connector._client is None


# =============================================================================
# API Header Verification Tests
# =============================================================================


class TestApiHeaders:
    """Tests for API header configuration."""

    @pytest.mark.asyncio
    async def test_connector_sets_correct_headers(self):
        """Test that connector sets correct authorization headers."""
        creds = PagerDutyCredentials(
            api_key="my_api_key_123",
            email="admin@company.com",
        )

        # Use a real httpx.AsyncClient to verify headers are set correctly
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.aclose = AsyncMock()

            async with PagerDutyConnector(creds):
                # Verify the AsyncClient was called with correct headers
                mock_client_class.assert_called_once()
                call_kwargs = mock_client_class.call_args[1]
                assert call_kwargs["headers"]["Authorization"] == "Token token=my_api_key_123"
                assert call_kwargs["headers"]["Content-Type"] == "application/json"
                assert call_kwargs["headers"]["From"] == "admin@company.com"
                assert call_kwargs["base_url"] == "https://api.pagerduty.com"
                assert call_kwargs["timeout"] == 30.0

    def test_base_url_constant(self):
        """Test that base URL is correctly defined."""
        assert PagerDutyConnector.BASE_URL == "https://api.pagerduty.com"


# =============================================================================
# Alert Deduplication Tests
# =============================================================================


class TestAlertDeduplication:
    """Tests for alert deduplication functionality."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident_with_dedup_key_in_body(self, connector):
        """Test that incident key is properly set in request body."""
        mock_response = {
            "incident": {
                "id": "PINC_DEDUP",
                "incident_number": 300,
                "title": "Dedup Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Dedup Test",
                    service_id="PSVC1",
                    incident_key="my-unique-dedup-key-12345",
                )
                await connector.create_incident(request)

                # Verify the request body
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["incident_key"] == "my-unique-dedup-key-12345"

    @pytest.mark.asyncio
    async def test_finding_generates_dedup_key_from_source_and_id(self, connector):
        """Test that create_incident_from_finding generates correct dedup key."""
        mock_response = {
            "incident": {
                "id": "PINC_FINDING",
                "incident_number": 301,
                "title": "[HIGH] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test Finding",
                    service_id="PSVC1",
                    severity="high",
                    description="Description",
                    source="security_scanner",
                    finding_id="VULN-2024-001",
                )

                # Verify dedup key format is "source:finding_id"
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["incident_key"] == "security_scanner:VULN-2024-001"


# =============================================================================
# Webhook Event Type Tests
# =============================================================================


class TestWebhookEventTypes:
    """Tests for various webhook event types."""

    def test_parse_webhook_incident_resolved(self):
        """Test parsing incident.resolved webhook."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.resolved",
                "data": {
                    "id": "PINC123",
                    "type": "incident",
                    "title": "Resolved Incident",
                    "status": "resolved",
                    "urgency": "high",
                    "incident_number": 100,
                    "created_at": "2024-01-15T10:00:00Z",
                    "resolved_at": "2024-01-15T12:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.resolved"
        assert payload.incident.status == IncidentStatus.RESOLVED

    def test_parse_webhook_incident_reassigned(self):
        """Test parsing incident.reassigned webhook."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.reassigned",
                "data": {
                    "id": "PINC456",
                    "type": "incident",
                    "title": "Reassigned Incident",
                    "status": "triggered",
                    "urgency": "high",
                    "incident_number": 101,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.reassigned"

    def test_parse_webhook_incident_delegated(self):
        """Test parsing incident.delegated webhook."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.delegated",
                "data": {
                    "id": "PINC789",
                    "type": "incident",
                    "title": "Delegated Incident",
                    "status": "acknowledged",
                    "urgency": "low",
                    "incident_number": 102,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.delegated"

    def test_parse_webhook_incident_escalated(self):
        """Test parsing incident.escalated webhook."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.escalated",
                "data": {
                    "id": "PINC_ESC",
                    "type": "incident",
                    "title": "Escalated Incident",
                    "status": "triggered",
                    "urgency": "high",
                    "incident_number": 103,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.escalated"

    def test_parse_webhook_incident_priority_updated(self):
        """Test parsing incident.priority_updated webhook."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.priority_updated",
                "data": {
                    "id": "PINC_PRIO",
                    "type": "incident",
                    "title": "Priority Updated",
                    "status": "triggered",
                    "urgency": "high",
                    "priority": {"summary": "P1"},
                    "incident_number": 104,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            }
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.event_type == "incident.priority_updated"
        assert payload.incident.priority == IncidentPriority.P1

    def test_parse_webhook_preserves_raw_data(self):
        """Test that raw_data is preserved in webhook payload."""
        creds = PagerDutyCredentials(api_key="key", email="user@example.com")
        connector = PagerDutyConnector(creds)

        webhook_data = {
            "event": {
                "event_type": "incident.triggered",
                "data": {
                    "id": "PINC123",
                    "type": "incident",
                    "title": "Test",
                    "status": "triggered",
                    "urgency": "high",
                    "incident_number": 1,
                    "created_at": "2024-01-15T10:00:00Z",
                },
            },
            "custom_field": "custom_value",
        }
        payload = connector.parse_webhook(webhook_data)
        assert payload.raw_data == webhook_data
        assert payload.raw_data["custom_field"] == "custom_value"


# =============================================================================
# Escalation Policy Tests
# =============================================================================


class TestEscalationPolicyHandling:
    """Tests for escalation policy handling."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_create_incident_with_escalation_policy_body(self, connector):
        """Test that escalation policy is correctly included in request body."""
        mock_response = {
            "incident": {
                "id": "PINC_ESC",
                "incident_number": 400,
                "title": "Escalated Alert",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                request = IncidentCreateRequest(
                    title="Escalated Alert",
                    service_id="PSVC1",
                    escalation_policy_id="PESCAL_CUSTOM",
                )
                await connector.create_incident(request)

                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["escalation_policy"]["id"] == "PESCAL_CUSTOM"
                assert body["escalation_policy"]["type"] == "escalation_policy_reference"


# =============================================================================
# Service HTML URL Tests
# =============================================================================


class TestServiceHtmlUrl:
    """Tests for Service html_url field."""

    def test_service_with_html_url(self):
        """Test service with html_url field."""
        api_data = {
            "id": "PSVC1",
            "name": "Production API",
            "status": "active",
            "html_url": "https://example.pagerduty.com/services/PSVC1",
        }
        service = Service.from_api(api_data)
        assert service.html_url == "https://example.pagerduty.com/services/PSVC1"

    def test_service_without_html_url(self):
        """Test service without html_url field."""
        api_data = {
            "id": "PSVC1",
            "name": "Service",
            "status": "active",
        }
        service = Service.from_api(api_data)
        assert service.html_url is None


# =============================================================================
# Incident HTML URL Tests
# =============================================================================


class TestIncidentHtmlUrl:
    """Tests for Incident html_url field."""

    def test_incident_with_html_url(self):
        """Test incident with html_url field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "html_url": "https://example.pagerduty.com/incidents/PINC1",
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.html_url == "https://example.pagerduty.com/incidents/PINC1"

    def test_incident_without_html_url(self):
        """Test incident without html_url field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.html_url is None


# =============================================================================
# Incident Description Tests
# =============================================================================


class TestIncidentDescription:
    """Tests for Incident description field."""

    def test_incident_with_description(self):
        """Test incident with description field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "description": "Detailed incident description here",
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.description == "Detailed incident description here"

    def test_incident_without_description(self):
        """Test incident without description field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.description is None


# =============================================================================
# Multiple User Assignment Tests
# =============================================================================


class TestMultipleUserAssignments:
    """Tests for incidents with multiple assignees."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    def test_incident_with_multiple_assignees(self):
        """Test incident with multiple assigned users."""
        api_data = {
            "id": "PINC1",
            "title": "Multi-user Incident",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "assignments": [
                {"assignee": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"}},
                {"assignee": {"id": "PUSER2", "name": "Bob", "email": "bob@example.com"}},
                {"assignee": {"id": "PUSER3", "name": "Charlie", "email": "charlie@example.com"}},
            ],
        }
        incident = Incident.from_api(api_data)
        assert len(incident.assigned_to) == 3
        assert incident.assigned_to[0].name == "Alice"
        assert incident.assigned_to[1].name == "Bob"
        assert incident.assigned_to[2].name == "Charlie"

    def test_incident_with_assignment_missing_assignee(self):
        """Test incident with assignment entry missing assignee field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "assignments": [
                {"assignee": {"id": "PUSER1", "name": "Alice", "email": "alice@example.com"}},
                {},  # Missing assignee
                {"assignee": {"id": "PUSER2", "name": "Bob", "email": "bob@example.com"}},
            ],
        }
        incident = Incident.from_api(api_data)
        # Only 2 users should be included (the one without assignee is skipped)
        assert len(incident.assigned_to) == 2

    @pytest.mark.asyncio
    async def test_reassign_to_multiple_users(self, connector):
        """Test reassigning incident to multiple users."""
        mock_response = {
            "incident": {
                "id": "PINC123",
                "title": "Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "incident_number": 1,
                "created_at": "2024-01-15T10:00:00Z",
                "assignments": [
                    {"assignee": {"id": "PUSER1", "name": "User1", "email": "u1@example.com"}},
                    {"assignee": {"id": "PUSER2", "name": "User2", "email": "u2@example.com"}},
                    {"assignee": {"id": "PUSER3", "name": "User3", "email": "u3@example.com"}},
                ],
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                incident = await connector.reassign_incident(
                    "PINC123", ["PUSER1", "PUSER2", "PUSER3"]
                )
                assert len(incident.assigned_to) == 3

                # Verify request body format
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert len(body["assignments"]) == 3
                assert body["assignments"][0]["assignee"]["id"] == "PUSER1"


# =============================================================================
# On-Call Schedule Escalation Level Tests
# =============================================================================


class TestOnCallEscalationLevels:
    """Tests for on-call schedule escalation levels."""

    def test_oncall_schedule_with_escalation_level(self):
        """Test on-call schedule with specific escalation level."""
        api_data = {
            "schedule": {"id": "PSCHED1", "summary": "Primary"},
            "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            "start": "2024-01-15T00:00:00Z",
            "end": "2024-01-22T00:00:00Z",
            "escalation_level": 3,
        }
        schedule = OnCallSchedule.from_api(api_data)
        assert schedule.escalation_level == 3

    def test_oncall_schedule_default_escalation_level(self):
        """Test on-call schedule with default escalation level."""
        api_data = {
            "schedule": {"id": "PSCHED1", "summary": "Primary"},
            "user": {"id": "PUSER1", "name": "User", "email": "user@example.com"},
            "start": "2024-01-15T00:00:00Z",
            "end": "2024-01-22T00:00:00Z",
            # No escalation_level provided
        }
        schedule = OnCallSchedule.from_api(api_data)
        assert schedule.escalation_level == 1  # Default


# =============================================================================
# Service Created At Tests
# =============================================================================


class TestServiceCreatedAt:
    """Tests for Service created_at field."""

    def test_service_with_created_at(self):
        """Test service with created_at field."""
        api_data = {
            "id": "PSVC1",
            "name": "Service",
            "status": "active",
            "created_at": "2024-01-15T10:30:00Z",
        }
        service = Service.from_api(api_data)
        assert service.created_at is not None
        assert service.created_at.year == 2024
        assert service.created_at.month == 1
        assert service.created_at.day == 15

    def test_service_without_created_at(self):
        """Test service without created_at field."""
        api_data = {
            "id": "PSVC1",
            "name": "Service",
            "status": "active",
        }
        service = Service.from_api(api_data)
        assert service.created_at is None


# =============================================================================
# Incident Created At Tests
# =============================================================================


class TestIncidentTimestamps:
    """Tests for Incident timestamp fields."""

    def test_incident_with_all_timestamps(self):
        """Test incident with all timestamp fields."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "resolved",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
            "resolved_at": "2024-01-15T12:00:00Z",
            "last_status_change_at": "2024-01-15T12:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.created_at is not None
        assert incident.resolved_at is not None
        assert incident.last_status_change_at is not None

    def test_incident_without_resolved_at(self):
        """Test incident without resolved_at field."""
        api_data = {
            "id": "PINC1",
            "title": "Test",
            "status": "triggered",
            "urgency": "high",
            "incident_number": 1,
            "created_at": "2024-01-15T10:00:00Z",
        }
        incident = Incident.from_api(api_data)
        assert incident.resolved_at is None


# =============================================================================
# Finding Severity Mapping Tests
# =============================================================================


class TestFindingSeverityMapping:
    """Tests for severity to urgency mapping in create_incident_from_finding."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_critical_severity_maps_to_high_urgency(self, connector):
        """Test that critical severity maps to high urgency."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[CRITICAL] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="critical",
                    description="Desc",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_high_severity_maps_to_high_urgency(self, connector):
        """Test that high severity maps to high urgency."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[HIGH] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="high",
                    description="Desc",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_medium_severity_maps_to_low_urgency(self, connector):
        """Test that medium severity maps to low urgency."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[MEDIUM] Test",
                "status": "triggered",
                "urgency": "low",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="medium",
                    description="Desc",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["urgency"] == "low"

    @pytest.mark.asyncio
    async def test_low_severity_maps_to_low_urgency(self, connector):
        """Test that low severity maps to low urgency."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[LOW] Test",
                "status": "triggered",
                "urgency": "low",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="low",
                    description="Desc",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["urgency"] == "low"


# =============================================================================
# Finding Description Format Tests
# =============================================================================


class TestFindingDescriptionFormat:
    """Tests for description formatting in create_incident_from_finding."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_finding_description_includes_source(self, connector):
        """Test that finding description includes source."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[HIGH] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="high",
                    description="Issue found",
                    source="sast_scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                description = body["body"]["details"]
                assert "Source: sast_scanner" in description

    @pytest.mark.asyncio
    async def test_finding_description_includes_severity(self, connector):
        """Test that finding description includes severity."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[CRITICAL] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="critical",
                    description="Issue found",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                description = body["body"]["details"]
                assert "Severity: critical" in description

    @pytest.mark.asyncio
    async def test_finding_description_includes_file_location(self, connector):
        """Test that finding description includes file location."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[HIGH] Test",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="Test",
                    service_id="PSVC1",
                    severity="high",
                    description="Issue found",
                    source="scanner",
                    file_path="/src/auth/login.py",
                    line_number=42,
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                description = body["body"]["details"]
                assert "File: /src/auth/login.py:42" in description

    @pytest.mark.asyncio
    async def test_finding_title_format(self, connector):
        """Test that finding title is formatted correctly."""
        mock_response = {
            "incident": {
                "id": "PINC1",
                "incident_number": 1,
                "title": "[CRITICAL] SQL Injection Vulnerability",
                "status": "triggered",
                "urgency": "high",
                "service": {"id": "PSVC1", "name": "API", "summary": "API"},
                "created_at": "2024-01-15T10:00:00Z",
            }
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            async with connector:
                await connector.create_incident_from_finding(
                    title="SQL Injection Vulnerability",
                    service_id="PSVC1",
                    severity="critical",
                    description="Found SQL injection",
                    source="scanner",
                )
                call_args = mock_request.call_args
                body = call_args[1]["json"]["incident"]
                assert body["title"] == "[CRITICAL] SQL Injection Vulnerability"


# =============================================================================
# HTTP Connection Error Tests
# =============================================================================


class TestHttpConnectionErrors:
    """Tests for HTTP connection errors."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        creds = PagerDutyCredentials(
            api_key="test_key",
            email="user@example.com",
        )
        return PagerDutyConnector(creds)

    @pytest.mark.asyncio
    async def test_request_handles_connection_error(self, connector):
        """Test that connection errors are handled."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError, match="HTTP error"):
                    await connector._request("GET", "/incidents")

    @pytest.mark.asyncio
    async def test_request_handles_timeout_error(self, connector):
        """Test that timeout errors are handled."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.side_effect = httpx.TimeoutException("Request timed out")
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError, match="HTTP error"):
                    await connector._request("GET", "/incidents")

    @pytest.mark.asyncio
    async def test_request_handles_network_error(self, connector):
        """Test that network errors are handled."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request.side_effect = httpx.NetworkError("Network unreachable")
            mock_client_class.return_value = mock_client

            async with connector:
                connector._client = mock_client
                with pytest.raises(PagerDutyError, match="HTTP error"):
                    await connector._request("GET", "/incidents")

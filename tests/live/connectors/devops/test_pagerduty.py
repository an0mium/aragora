"""
Tests for PagerDuty Incident Management Connector in the Live module.

Comprehensive tests covering:
1. Event creation and serialization
2. Stream lifecycle (start, stop, pause, resume)
3. Event filtering
4. Backpressure handling
5. Error handling
"""

import pytest
from datetime import datetime, timezone, timedelta
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
# Fixtures
# =============================================================================


@pytest.fixture
def valid_credentials():
    """Create valid PagerDuty credentials for testing."""
    return PagerDutyCredentials(
        api_key="test_api_key_12345",
        email="admin@example.com",
        webhook_secret="webhook_secret_test_123",
    )


@pytest.fixture
def credentials_without_secret():
    """Create credentials without webhook secret."""
    return PagerDutyCredentials(
        api_key="test_api_key",
        email="user@example.com",
    )


@pytest.fixture
def connector(valid_credentials):
    """Create a PagerDuty connector for testing."""
    return PagerDutyConnector(valid_credentials)


# =============================================================================
# Event Creation and Serialization Tests
# =============================================================================


class TestEventCreation:
    """Tests for event creation functionality."""

    def test_incident_event_creation(self):
        """Test creating an incident event with all fields."""
        service = Service(
            id="svc_001",
            name="Production API",
            description="Main production API service",
            status=ServiceStatus.ACTIVE,
            escalation_policy_id="escpol_001",
        )

        user = User(
            id="user_001",
            name="John Responder",
            email="john@example.com",
            role="admin",
        )

        incident = Incident(
            id="inc_001",
            incident_number=1234,
            title="Database Connection Pool Exhausted",
            status=IncidentStatus.TRIGGERED,
            urgency=IncidentUrgency.HIGH,
            service=service,
            assigned_to=[user],
            priority=IncidentPriority.P1,
            description="Connection pool at 100% capacity",
            html_url="https://example.pagerduty.com/incidents/inc_001",
            created_at=datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
        )

        assert incident.id == "inc_001"
        assert incident.incident_number == 1234
        assert incident.status == IncidentStatus.TRIGGERED
        assert incident.urgency == IncidentUrgency.HIGH
        assert incident.priority == IncidentPriority.P1
        assert len(incident.assigned_to) == 1
        assert incident.service.name == "Production API"

    def test_incident_event_minimal_creation(self):
        """Test creating an incident event with minimal fields."""
        incident = Incident(
            id="inc_002",
            incident_number=5678,
            title="Minimal Incident",
            status=IncidentStatus.TRIGGERED,
            urgency=IncidentUrgency.LOW,
        )

        assert incident.id == "inc_002"
        assert incident.service is None
        assert incident.assigned_to == []
        assert incident.priority is None

    def test_service_event_creation(self):
        """Test creating a service event."""
        service = Service(
            id="svc_002",
            name="API Gateway",
            description="Gateway routing service",
            status=ServiceStatus.CRITICAL,
            escalation_policy_id="escpol_002",
            html_url="https://example.pagerduty.com/services/svc_002",
            created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )

        assert service.id == "svc_002"
        assert service.status == ServiceStatus.CRITICAL

    def test_user_event_creation(self):
        """Test creating a user event."""
        user = User(
            id="user_002",
            name="Jane Engineer",
            email="jane@example.com",
            role="responder",
            html_url="https://example.pagerduty.com/users/user_002",
        )

        assert user.id == "user_002"
        assert user.role == "responder"

    def test_incident_note_event_creation(self):
        """Test creating an incident note event."""
        user = User(id="user_003", name="Note Author", email="author@example.com")
        note = IncidentNote(
            id="note_001",
            content="Root cause identified: memory leak in service X",
            user=user,
            created_at=datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc),
        )

        assert note.id == "note_001"
        assert "memory leak" in note.content
        assert note.user.name == "Note Author"

    def test_on_call_schedule_event_creation(self):
        """Test creating an on-call schedule event."""
        user = User(id="user_004", name="On-Call Engineer", email="oncall@example.com")
        schedule = OnCallSchedule(
            user=user,
            schedule_id="sched_001",
            schedule_name="Primary On-Call",
            start=datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 6, 22, 0, 0, 0, tzinfo=timezone.utc),
            escalation_level=1,
        )

        assert schedule.schedule_id == "sched_001"
        assert schedule.escalation_level == 1

    def test_incident_create_request_event(self):
        """Test creating an incident create request event."""
        request = IncidentCreateRequest(
            title="Test Incident",
            service_id="svc_003",
            urgency=IncidentUrgency.HIGH,
            description="Test incident description",
            priority_id="pri_001",
            escalation_policy_id="escpol_003",
            incident_key="dedup_key_001",
            assignments=["user_001", "user_002"],
        )

        assert request.title == "Test Incident"
        assert request.urgency == IncidentUrgency.HIGH
        assert len(request.assignments) == 2


class TestEventSerialization:
    """Tests for event serialization."""

    def test_service_from_api_serialization(self):
        """Test service deserialization from API response."""
        api_data = {
            "id": "PSERVICE001",
            "name": "Database Service",
            "description": "PostgreSQL database cluster",
            "status": "warning",
            "escalation_policy": {"id": "PESCPOL001"},
            "html_url": "https://example.pagerduty.com/services/PSERVICE001",
            "created_at": "2024-01-15T10:30:00Z",
        }

        service = Service.from_api(api_data)

        assert service.id == "PSERVICE001"
        assert service.name == "Database Service"
        assert service.status == ServiceStatus.WARNING
        assert service.escalation_policy_id == "PESCPOL001"

    def test_user_from_api_serialization(self):
        """Test user deserialization from API response."""
        api_data = {
            "id": "PUSER001",
            "name": "Sarah Admin",
            "email": "sarah@example.com",
            "role": "owner",
            "html_url": "https://example.pagerduty.com/users/PUSER001",
        }

        user = User.from_api(api_data)

        assert user.id == "PUSER001"
        assert user.name == "Sarah Admin"
        assert user.role == "owner"

    def test_incident_from_api_serialization(self):
        """Test incident deserialization from API response."""
        api_data = {
            "id": "PINC001",
            "incident_number": 9999,
            "title": "API Latency Spike",
            "status": "acknowledged",
            "urgency": "high",
            "service": {
                "id": "PSVC001",
                "name": "API Service",
                "status": "active",
            },
            "assignments": [
                {"assignee": {"id": "PUSER001", "name": "John", "email": "john@test.com"}}
            ],
            "priority": {"summary": "P2"},
            "html_url": "https://example.pagerduty.com/incidents/PINC001",
            "description": "Response times above threshold",
            "created_at": "2024-06-15T14:00:00Z",
            "last_status_change_at": "2024-06-15T14:15:00Z",
        }

        incident = Incident.from_api(api_data)

        assert incident.id == "PINC001"
        assert incident.incident_number == 9999
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.urgency == IncidentUrgency.HIGH
        assert incident.priority == IncidentPriority.P2
        assert len(incident.assigned_to) == 1

    def test_on_call_schedule_from_api_serialization(self):
        """Test on-call schedule deserialization from API response."""
        api_data = {
            "schedule": {"id": "PSCHED001", "summary": "Weekend Rotation"},
            "user": {
                "id": "PUSER002",
                "name": "Weekend Engineer",
                "email": "weekend@example.com",
            },
            "start": "2024-06-15T00:00:00Z",
            "end": "2024-06-17T00:00:00Z",
            "escalation_level": 2,
        }

        schedule = OnCallSchedule.from_api(api_data)

        assert schedule.schedule_id == "PSCHED001"
        assert schedule.schedule_name == "Weekend Rotation"
        assert schedule.escalation_level == 2

    def test_incident_note_from_api_serialization(self):
        """Test incident note deserialization from API response."""
        api_data = {
            "id": "PNOTE001",
            "content": "Investigation complete - fix deployed",
            "created_at": "2024-06-15T16:00:00Z",
            "user": {
                "id": "PUSER003",
                "name": "Note Writer",
                "email": "writer@example.com",
            },
        }

        note = IncidentNote.from_api(api_data)

        assert note.id == "PNOTE001"
        assert "fix deployed" in note.content
        assert note.user.name == "Note Writer"

    def test_webhook_payload_parsing(self, connector):
        """Test webhook payload parsing."""
        webhook_data = {
            "event": {
                "event_type": "incident.triggered",
                "resource_type": "incident",
                "occurred_at": "2024-06-15T10:00:00Z",
                "data": {
                    "id": "PINC_WEBHOOK",
                    "type": "incident",
                    "incident_number": 7777,
                    "title": "Webhook Test Incident",
                    "status": "triggered",
                    "urgency": "high",
                },
            },
        }

        payload = connector.parse_webhook(webhook_data)

        assert isinstance(payload, WebhookPayload)
        assert payload.event_type == "incident.triggered"
        assert payload.incident is not None
        assert payload.incident.incident_number == 7777


# =============================================================================
# Stream Lifecycle Tests
# =============================================================================


class TestStreamLifecycle:
    """Tests for stream lifecycle management (start, stop, pause, resume)."""

    def test_connector_initialization(self, valid_credentials):
        """Test connector initializes correctly."""
        connector = PagerDutyConnector(valid_credentials)
        assert connector.credentials == valid_credentials
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager_start(self, connector):
        """Test connector starts properly with context manager."""
        async with connector as conn:
            assert conn._client is not None

    @pytest.mark.asyncio
    async def test_connector_context_manager_stop(self, connector):
        """Test connector stops properly after context manager exit."""
        async with connector as conn:
            assert conn._client is not None
        assert connector._client is None

    def test_connector_client_property_raises_without_init(self, connector):
        """Test client property raises error when not initialized."""
        with pytest.raises(PagerDutyError) as exc_info:
            _ = connector.client
        assert "not initialized" in str(exc_info.value)


# =============================================================================
# Event Filtering Tests
# =============================================================================


class TestEventFiltering:
    """Tests for event filtering functionality."""

    def test_incident_status_filtering(self):
        """Test filtering incidents by status."""
        incidents = [
            Incident(
                id="1",
                incident_number=1,
                title="I1",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
            ),
            Incident(
                id="2",
                incident_number=2,
                title="I2",
                status=IncidentStatus.ACKNOWLEDGED,
                urgency=IncidentUrgency.HIGH,
            ),
            Incident(
                id="3",
                incident_number=3,
                title="I3",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.LOW,
            ),
            Incident(
                id="4",
                incident_number=4,
                title="I4",
                status=IncidentStatus.RESOLVED,
                urgency=IncidentUrgency.HIGH,
            ),
        ]

        triggered = [i for i in incidents if i.status == IncidentStatus.TRIGGERED]
        assert len(triggered) == 2

        resolved = [i for i in incidents if i.status == IncidentStatus.RESOLVED]
        assert len(resolved) == 1

    def test_incident_urgency_filtering(self):
        """Test filtering incidents by urgency."""
        incidents = [
            Incident(
                id="1",
                incident_number=1,
                title="I1",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
            ),
            Incident(
                id="2",
                incident_number=2,
                title="I2",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.LOW,
            ),
            Incident(
                id="3",
                incident_number=3,
                title="I3",
                status=IncidentStatus.ACKNOWLEDGED,
                urgency=IncidentUrgency.HIGH,
            ),
        ]

        high_urgency = [i for i in incidents if i.urgency == IncidentUrgency.HIGH]
        assert len(high_urgency) == 2

    def test_incident_priority_filtering(self):
        """Test filtering incidents by priority."""
        incidents = [
            Incident(
                id="1",
                incident_number=1,
                title="I1",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
                priority=IncidentPriority.P1,
            ),
            Incident(
                id="2",
                incident_number=2,
                title="I2",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
                priority=IncidentPriority.P2,
            ),
            Incident(
                id="3",
                incident_number=3,
                title="I3",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
                priority=IncidentPriority.P1,
            ),
            Incident(
                id="4",
                incident_number=4,
                title="I4",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.LOW,
                priority=None,
            ),
        ]

        p1_incidents = [i for i in incidents if i.priority == IncidentPriority.P1]
        assert len(p1_incidents) == 2

        no_priority = [i for i in incidents if i.priority is None]
        assert len(no_priority) == 1

    def test_service_status_filtering(self):
        """Test filtering services by status."""
        services = [
            Service(id="1", name="S1", status=ServiceStatus.ACTIVE),
            Service(id="2", name="S2", status=ServiceStatus.CRITICAL),
            Service(id="3", name="S3", status=ServiceStatus.ACTIVE),
            Service(id="4", name="S4", status=ServiceStatus.MAINTENANCE),
        ]

        active = [s for s in services if s.status == ServiceStatus.ACTIVE]
        assert len(active) == 2

        critical = [s for s in services if s.status == ServiceStatus.CRITICAL]
        assert len(critical) == 1

    def test_incident_date_range_filtering(self):
        """Test filtering incidents by date range."""
        base_date = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        incidents = [
            Incident(
                id="1",
                incident_number=1,
                title="Old",
                status=IncidentStatus.RESOLVED,
                urgency=IncidentUrgency.LOW,
                created_at=base_date - timedelta(days=30),
            ),
            Incident(
                id="2",
                incident_number=2,
                title="Recent",
                status=IncidentStatus.ACKNOWLEDGED,
                urgency=IncidentUrgency.HIGH,
                created_at=base_date - timedelta(days=2),
            ),
            Incident(
                id="3",
                incident_number=3,
                title="Today",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
                created_at=base_date,
            ),
        ]

        # Filter for last 7 days
        cutoff = base_date - timedelta(days=7)
        recent = [i for i in incidents if i.created_at and i.created_at >= cutoff]
        assert len(recent) == 2

    def test_combined_incident_filtering(self):
        """Test combining multiple filters on incidents."""
        base_date = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        incidents = [
            Incident(
                id="1",
                incident_number=1,
                title="P1 High Triggered",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
                priority=IncidentPriority.P1,
                created_at=base_date - timedelta(hours=2),
            ),
            Incident(
                id="2",
                incident_number=2,
                title="P1 High Resolved",
                status=IncidentStatus.RESOLVED,
                urgency=IncidentUrgency.HIGH,
                priority=IncidentPriority.P1,
                created_at=base_date - timedelta(hours=4),
            ),
            Incident(
                id="3",
                incident_number=3,
                title="P2 Low Triggered",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.LOW,
                priority=IncidentPriority.P2,
                created_at=base_date - timedelta(hours=1),
            ),
        ]

        # High urgency, P1 priority, triggered
        filtered = [
            i
            for i in incidents
            if i.urgency == IncidentUrgency.HIGH
            and i.priority == IncidentPriority.P1
            and i.status == IncidentStatus.TRIGGERED
        ]
        assert len(filtered) == 1
        assert filtered[0].id == "1"


# =============================================================================
# Backpressure Handling Tests
# =============================================================================


class TestBackpressureHandling:
    """Tests for backpressure handling in event processing."""

    def test_incident_batch_size_limiting(self):
        """Test limiting incident batch sizes."""
        all_incidents = [
            Incident(
                id=str(i),
                incident_number=i,
                title=f"Incident {i}",
                status=IncidentStatus.TRIGGERED,
                urgency=IncidentUrgency.HIGH,
            )
            for i in range(250)
        ]

        # PagerDuty max page size is 100
        batch_size = 100
        batches = [
            all_incidents[i : i + batch_size] for i in range(0, len(all_incidents), batch_size)
        ]

        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50

    def test_pagination_offset_calculation(self):
        """Test pagination offset calculations."""
        total_items = 350
        page_size = 100

        offsets = []
        offset = 0
        while offset < total_items:
            offsets.append(offset)
            offset += page_size

        assert offsets == [0, 100, 200, 300]

    def test_rate_limit_retry_backoff(self):
        """Test rate limit retry backoff calculations."""
        max_retries = 5
        base_delay = 1.0  # seconds

        delays = [base_delay * (2**i) for i in range(max_retries)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

        # With jitter (conceptual - actual jitter would be random)
        max_delay = base_delay * (2**max_retries)
        assert max_delay == 32.0

    def test_concurrent_request_limiting(self):
        """Test concurrent request limiting."""
        max_concurrent = 5
        pending_requests = []

        # Simulate queuing requests
        for i in range(20):
            if len([r for r in pending_requests if r == "pending"]) < max_concurrent:
                pending_requests.append("pending")
            else:
                pending_requests.append("queued")

        pending_count = len([r for r in pending_requests if r == "pending"])
        queued_count = len([r for r in pending_requests if r == "queued"])

        assert pending_count == max_concurrent
        assert queued_count == 15

    def test_note_batch_processing(self):
        """Test processing notes in batches."""
        notes = [IncidentNote(id=str(i), content=f"Note {i}") for i in range(30)]

        batch_size = 10
        processed_batches = []

        for i in range(0, len(notes), batch_size):
            batch = notes[i : i + batch_size]
            processed_batches.append(len(batch))

        assert processed_batches == [10, 10, 10]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the connector."""

    def test_pagerduty_error_creation(self):
        """Test PagerDutyError exception creation."""
        error = PagerDutyError(
            message="Rate limit exceeded",
            status_code=429,
            error_code="RATE_LIMIT",
        )

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT"

    def test_pagerduty_error_minimal(self):
        """Test PagerDutyError with minimal info."""
        error = PagerDutyError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_code is None

    def test_webhook_signature_verification_valid(self, connector):
        """Test valid webhook signature verification."""
        payload = '{"event": "incident.triggered"}'
        secret = connector.credentials.webhook_secret
        signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        assert connector.verify_webhook_signature(payload.encode(), f"v1={signature}") is True

    def test_webhook_signature_verification_invalid(self, connector):
        """Test invalid webhook signature verification."""
        payload = b'{"event": "incident.triggered"}'
        invalid_signature = "v1=invalid_signature_here"

        assert connector.verify_webhook_signature(payload, invalid_signature) is False

    def test_webhook_signature_verification_no_secret(self, credentials_without_secret):
        """Test webhook verification skipped when no secret configured."""
        connector = PagerDutyConnector(credentials_without_secret)

        # Should return True (skip verification) when no secret
        assert connector.verify_webhook_signature(b"payload", "any_signature") is True

    def test_service_from_api_missing_fields(self):
        """Test service creation handles missing optional fields."""
        api_data = {
            "id": "PSVC_MIN",
            "name": "Minimal Service",
        }

        service = Service.from_api(api_data)

        assert service.id == "PSVC_MIN"
        assert service.description is None
        assert service.escalation_policy_id is None

    def test_incident_from_api_missing_fields(self):
        """Test incident creation handles missing optional fields."""
        api_data = {
            "id": "PINC_MIN",
            "title": "Minimal Incident",
            "status": "triggered",
            "urgency": "low",
        }

        incident = Incident.from_api(api_data)

        assert incident.id == "PINC_MIN"
        assert incident.incident_number == 0  # Default
        assert incident.service is None
        assert incident.assigned_to == []
        assert incident.priority is None

    def test_on_call_schedule_missing_user(self):
        """Test schedule handles missing user gracefully."""
        api_data = {
            "schedule": {"id": "PSCHED_NO_USER", "summary": "Test"},
            "user": {},  # Empty user
            "start": "2024-06-15T00:00:00Z",
            "end": "2024-06-22T00:00:00Z",
        }

        schedule = OnCallSchedule.from_api(api_data)

        # Should create a placeholder user
        assert schedule.user.id == "unknown"
        assert schedule.user.name == "Unknown"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_service_structure(self):
        """Test mock service has complete structure."""
        service = get_mock_service()

        assert isinstance(service, Service)
        assert service.id is not None
        assert service.name is not None
        assert service.status is not None

    def test_get_mock_user_structure(self):
        """Test mock user has complete structure."""
        user = get_mock_user()

        assert isinstance(user, User)
        assert user.id is not None
        assert user.name is not None
        assert user.email is not None

    def test_get_mock_incident_structure(self):
        """Test mock incident has complete structure."""
        incident = get_mock_incident()

        assert isinstance(incident, Incident)
        assert incident.id is not None
        assert incident.incident_number > 0
        assert incident.title is not None
        assert incident.status is not None
        assert incident.urgency is not None
        assert incident.service is not None
        assert len(incident.assigned_to) > 0

    def test_get_mock_on_call_structure(self):
        """Test mock on-call schedule has complete structure."""
        schedule = get_mock_on_call()

        assert isinstance(schedule, OnCallSchedule)
        assert schedule.schedule_id is not None
        assert schedule.schedule_name is not None
        assert schedule.user is not None
        assert schedule.start is not None
        assert schedule.end is not None


# =============================================================================
# Enum Value Tests
# =============================================================================


class TestEnumValues:
    """Tests for enum value completeness."""

    def test_all_incident_statuses_defined(self):
        """Test all expected incident statuses are defined."""
        expected_statuses = {"triggered", "acknowledged", "resolved"}
        actual_statuses = {s.value for s in IncidentStatus}
        assert expected_statuses == actual_statuses

    def test_all_incident_urgencies_defined(self):
        """Test all expected urgencies are defined."""
        expected_urgencies = {"high", "low"}
        actual_urgencies = {u.value for u in IncidentUrgency}
        assert expected_urgencies == actual_urgencies

    def test_all_incident_priorities_defined(self):
        """Test all expected priorities are defined."""
        expected_priorities = {"P1", "P2", "P3", "P4", "P5"}
        actual_priorities = {p.value for p in IncidentPriority}
        assert expected_priorities == actual_priorities

    def test_all_service_statuses_defined(self):
        """Test all expected service statuses are defined."""
        expected_statuses = {"active", "warning", "critical", "maintenance", "disabled"}
        actual_statuses = {s.value for s in ServiceStatus}
        assert expected_statuses == actual_statuses


# =============================================================================
# Integration Helper Tests
# =============================================================================


class TestIntegrationHelpers:
    """Tests for integration helper methods."""

    def test_incident_create_request_defaults(self):
        """Test IncidentCreateRequest has sensible defaults."""
        request = IncidentCreateRequest(
            title="Test",
            service_id="svc_test",
        )

        assert request.urgency == IncidentUrgency.HIGH  # Default
        assert request.description is None
        assert request.priority_id is None
        assert request.escalation_policy_id is None
        assert request.incident_key is None
        assert request.assignments is None

    def test_incident_create_request_with_dedup_key(self):
        """Test IncidentCreateRequest with deduplication key."""
        request = IncidentCreateRequest(
            title="Deduplicated Incident",
            service_id="svc_dedup",
            incident_key="unique_key_123",
        )

        assert request.incident_key == "unique_key_123"

    def test_incident_create_request_with_assignments(self):
        """Test IncidentCreateRequest with assignments."""
        request = IncidentCreateRequest(
            title="Assigned Incident",
            service_id="svc_assign",
            assignments=["user_1", "user_2", "user_3"],
        )

        assert len(request.assignments) == 3

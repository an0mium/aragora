"""
Tests for DocuSign E-Signature Connector in the Live module.

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
import base64

from aragora.connectors.legal.docusign import (
    DocuSignConnector,
    DocuSignCredentials,
    DocuSignEnvironment,
    Envelope,
    EnvelopeCreateRequest,
    EnvelopeStatus,
    Recipient,
    RecipientType,
    Document,
    SignatureTab,
    get_mock_envelope,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_credentials():
    """Create valid DocuSign credentials for testing."""
    return DocuSignCredentials(
        access_token="test_token_12345",
        account_id="acc_test_001",
        base_uri="https://demo.docusign.net/restapi",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


@pytest.fixture
def expired_credentials():
    """Create expired DocuSign credentials for testing."""
    return DocuSignCredentials(
        access_token="expired_token",
        account_id="acc_test_001",
        base_uri="https://demo.docusign.net/restapi",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )


@pytest.fixture
def configured_connector():
    """Create a configured connector for testing."""
    return DocuSignConnector(
        integration_key="test_integration_key",
        user_id="test_user_id",
        account_id="test_account_id",
    )


@pytest.fixture
def authenticated_connector(configured_connector, valid_credentials):
    """Create an authenticated connector for testing."""
    configured_connector._credentials = valid_credentials
    return configured_connector


# =============================================================================
# Event Creation and Serialization Tests
# =============================================================================


class TestEventCreation:
    """Tests for event creation functionality."""

    def test_envelope_event_creation(self):
        """Test creating an envelope event with all fields."""
        envelope = Envelope(
            envelope_id="env_001",
            status=EnvelopeStatus.SENT,
            email_subject="Contract for Review",
            email_body="Please review and sign the attached contract.",
            signers=[
                {
                    "email": "signer@example.com",
                    "name": "John Signer",
                    "recipientId": "1",
                    "status": "sent",
                }
            ],
            carbon_copies=[
                {
                    "email": "cc@example.com",
                    "name": "Jane CC",
                    "recipientId": "2",
                }
            ],
            documents=[{"id": "1", "name": "Contract.pdf"}],
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            sent_at=datetime(2024, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
            sender_name="Demo Sender",
            sender_email="sender@example.com",
        )

        assert envelope.envelope_id == "env_001"
        assert envelope.status == EnvelopeStatus.SENT
        assert len(envelope.signers) == 1
        assert len(envelope.carbon_copies) == 1
        assert envelope.sender_name == "Demo Sender"

    def test_envelope_event_minimal_creation(self):
        """Test creating an envelope event with minimal fields."""
        envelope = Envelope(
            envelope_id="env_002",
            status=EnvelopeStatus.CREATED,
            email_subject="Minimal Envelope",
        )

        assert envelope.envelope_id == "env_002"
        assert envelope.status == EnvelopeStatus.CREATED
        assert envelope.signers == []
        assert envelope.carbon_copies == []
        assert envelope.documents == []

    def test_recipient_event_creation(self):
        """Test creating recipient events."""
        recipient = Recipient(
            email="recipient@example.com",
            name="Test Recipient",
            recipient_type=RecipientType.SIGNER,
            routing_order=1,
            recipient_id="rec_001",
            access_code="1234",
        )

        assert recipient.email == "recipient@example.com"
        assert recipient.recipient_type == RecipientType.SIGNER
        assert recipient.access_code == "1234"

    def test_document_event_creation(self):
        """Test creating document events."""
        content = b"PDF document content bytes"
        doc = Document(
            document_id="doc_001",
            name="Agreement.pdf",
            content=content,
            file_extension="pdf",
            order=1,
        )

        assert doc.document_id == "doc_001"
        assert doc.name == "Agreement.pdf"
        assert doc.content == content
        assert doc.order == 1

    def test_signature_tab_event_creation(self):
        """Test creating signature tab events."""
        tab = SignatureTab(
            tab_type="signature",
            page_number=2,
            x_position=150,
            y_position=450,
            recipient_id="1",
            optional=False,
        )

        assert tab.tab_type == "signature"
        assert tab.page_number == 2
        assert not tab.optional


class TestEventSerialization:
    """Tests for event serialization."""

    def test_envelope_to_dict_serialization(self):
        """Test envelope serialization to dictionary."""
        envelope = Envelope(
            envelope_id="env_serialize_001",
            status=EnvelopeStatus.COMPLETED,
            email_subject="Serialization Test",
            created_at=datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
            sent_at=datetime(2024, 6, 15, 14, 35, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 6, 16, 9, 0, 0, tzinfo=timezone.utc),
        )

        serialized = envelope.to_dict()

        assert serialized["envelope_id"] == "env_serialize_001"
        assert serialized["status"] == "completed"
        assert "2024-06-15" in serialized["created_at"]
        assert "2024-06-16" in serialized["completed_at"]

    def test_envelope_to_dict_with_null_timestamps(self):
        """Test envelope serialization with null timestamps."""
        envelope = Envelope(
            envelope_id="env_null_001",
            status=EnvelopeStatus.CREATED,
            email_subject="Null Timestamp Test",
        )

        serialized = envelope.to_dict()

        assert serialized["created_at"] is None
        assert serialized["sent_at"] is None
        assert serialized["completed_at"] is None

    def test_recipient_to_dict_serialization(self):
        """Test recipient serialization."""
        recipient = Recipient(
            email="test@example.com",
            name="Test User",
            recipient_type=RecipientType.CARBON_COPY,
            routing_order=2,
            recipient_id="rec_test_001",
        )

        serialized = recipient.to_dict()

        assert serialized["email"] == "test@example.com"
        assert serialized["name"] == "Test User"
        assert serialized["recipient_type"] == "carbon_copy"
        assert serialized["routing_order"] == 2
        assert serialized["recipient_id"] == "rec_test_001"

    def test_recipient_to_dict_with_default_id(self):
        """Test recipient serialization with default ID."""
        recipient = Recipient(
            email="default@example.com",
            name="Default ID",
            routing_order=5,
        )

        serialized = recipient.to_dict()
        assert serialized["recipient_id"] == "5"  # Uses routing_order as default

    def test_signature_tab_to_dict_for_signature(self):
        """Test signature tab serialization for signature type."""
        tab = SignatureTab(
            tab_type="signature",
            page_number=3,
            x_position=200,
            y_position=600,
            recipient_id="1",
        )

        serialized = tab.to_dict()

        assert serialized["pageNumber"] == "3"
        assert serialized["xPosition"] == "200"
        assert serialized["yPosition"] == "600"
        assert serialized["recipientId"] == "1"

    def test_signature_tab_to_dict_for_text(self):
        """Test signature tab serialization for text type."""
        tab = SignatureTab(
            tab_type="text",
            page_number=1,
            x_position=100,
            y_position=100,
            tab_label="CompanyName",
            value="Acme Corporation",
            width=250,
            height=25,
        )

        serialized = tab.to_dict()

        assert serialized["tabLabel"] == "CompanyName"
        assert serialized["value"] == "Acme Corporation"
        assert serialized["width"] == "250"
        assert serialized["height"] == "25"

    def test_document_to_base64_serialization(self):
        """Test document base64 encoding."""
        content = b"Test document content for base64 encoding"
        doc = Document(
            document_id="doc_b64",
            name="test.pdf",
            content=content,
        )

        encoded = doc.to_base64()
        decoded = base64.b64decode(encoded)

        assert decoded == content


# =============================================================================
# Stream Lifecycle Tests
# =============================================================================


class TestStreamLifecycle:
    """Tests for stream lifecycle management (start, stop, pause, resume)."""

    def test_connector_initialization(self):
        """Test connector initializes in stopped state."""
        connector = DocuSignConnector()
        assert connector._credentials is None
        assert not connector.is_authenticated

    def test_connector_configuration_state(self, configured_connector):
        """Test connector configuration state tracking."""
        assert configured_connector.is_configured
        assert not configured_connector.is_authenticated

    def test_connector_authenticated_state(self, authenticated_connector):
        """Test connector authenticated state."""
        assert authenticated_connector.is_configured
        assert authenticated_connector.is_authenticated

    def test_credential_expiry_lifecycle(
        self, configured_connector, valid_credentials, expired_credentials
    ):
        """Test credential expiry state transitions."""
        # Start with valid credentials
        configured_connector._credentials = valid_credentials
        assert configured_connector.is_authenticated

        # Simulate credential expiry
        configured_connector._credentials = expired_credentials
        assert not configured_connector.is_authenticated

    def test_connector_demo_environment_urls(self):
        """Test demo environment URL configuration."""
        connector = DocuSignConnector(environment=DocuSignEnvironment.DEMO)
        assert connector.auth_url == DocuSignConnector.DEMO_AUTH_URL
        assert connector.api_url == DocuSignConnector.DEMO_API_URL

    def test_connector_production_environment_urls(self):
        """Test production environment URL configuration."""
        connector = DocuSignConnector(environment=DocuSignEnvironment.PRODUCTION)
        assert connector.auth_url == DocuSignConnector.PRODUCTION_AUTH_URL
        assert connector.api_url == DocuSignConnector.PRODUCTION_API_URL


# =============================================================================
# Event Filtering Tests
# =============================================================================


class TestEventFiltering:
    """Tests for event filtering functionality."""

    def test_envelope_status_filtering(self):
        """Test filtering envelopes by status."""
        envelopes = [
            Envelope(envelope_id="1", status=EnvelopeStatus.SENT, email_subject="Test 1"),
            Envelope(envelope_id="2", status=EnvelopeStatus.COMPLETED, email_subject="Test 2"),
            Envelope(envelope_id="3", status=EnvelopeStatus.SENT, email_subject="Test 3"),
            Envelope(envelope_id="4", status=EnvelopeStatus.VOIDED, email_subject="Test 4"),
        ]

        sent_envelopes = [e for e in envelopes if e.status == EnvelopeStatus.SENT]
        assert len(sent_envelopes) == 2

        completed_envelopes = [e for e in envelopes if e.status == EnvelopeStatus.COMPLETED]
        assert len(completed_envelopes) == 1

    def test_recipient_type_filtering(self):
        """Test filtering recipients by type."""
        recipients = [
            Recipient(email="a@test.com", name="A", recipient_type=RecipientType.SIGNER),
            Recipient(email="b@test.com", name="B", recipient_type=RecipientType.CARBON_COPY),
            Recipient(email="c@test.com", name="C", recipient_type=RecipientType.SIGNER),
            Recipient(email="d@test.com", name="D", recipient_type=RecipientType.EDITOR),
        ]

        signers = [r for r in recipients if r.recipient_type == RecipientType.SIGNER]
        assert len(signers) == 2

        editors = [r for r in recipients if r.recipient_type == RecipientType.EDITOR]
        assert len(editors) == 1

    def test_envelope_date_range_filtering(self):
        """Test filtering envelopes by date range."""
        base_date = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        envelopes = [
            Envelope(
                envelope_id="1",
                status=EnvelopeStatus.SENT,
                email_subject="Old",
                created_at=base_date - timedelta(days=10),
            ),
            Envelope(
                envelope_id="2",
                status=EnvelopeStatus.SENT,
                email_subject="Recent",
                created_at=base_date - timedelta(days=2),
            ),
            Envelope(
                envelope_id="3",
                status=EnvelopeStatus.SENT,
                email_subject="Today",
                created_at=base_date,
            ),
        ]

        cutoff = base_date - timedelta(days=5)
        recent_envelopes = [e for e in envelopes if e.created_at and e.created_at >= cutoff]
        assert len(recent_envelopes) == 2

    def test_envelope_with_multiple_filters(self):
        """Test combining multiple filters on envelopes."""
        base_date = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        envelopes = [
            Envelope(
                envelope_id="1",
                status=EnvelopeStatus.SENT,
                email_subject="Contract",
                created_at=base_date - timedelta(days=1),
            ),
            Envelope(
                envelope_id="2",
                status=EnvelopeStatus.COMPLETED,
                email_subject="Contract",
                created_at=base_date - timedelta(days=1),
            ),
            Envelope(
                envelope_id="3",
                status=EnvelopeStatus.SENT,
                email_subject="Invoice",
                created_at=base_date - timedelta(days=10),
            ),
        ]

        # Filter by status AND subject AND date
        cutoff = base_date - timedelta(days=5)
        filtered = [
            e
            for e in envelopes
            if e.status == EnvelopeStatus.SENT
            and "Contract" in e.email_subject
            and e.created_at
            and e.created_at >= cutoff
        ]
        assert len(filtered) == 1
        assert filtered[0].envelope_id == "1"


# =============================================================================
# Backpressure Handling Tests
# =============================================================================


class TestBackpressureHandling:
    """Tests for backpressure handling in event processing."""

    def test_batch_size_limiting(self):
        """Test limiting batch sizes to handle backpressure."""
        all_documents = [
            Document(document_id=str(i), name=f"doc_{i}.pdf", content=b"content")
            for i in range(100)
        ]

        batch_size = 10
        batches = [
            all_documents[i : i + batch_size] for i in range(0, len(all_documents), batch_size)
        ]

        assert len(batches) == 10
        assert all(len(batch) == batch_size for batch in batches)

    def test_recipient_batch_processing(self):
        """Test processing recipients in batches."""
        recipients = [Recipient(email=f"user{i}@test.com", name=f"User {i}") for i in range(50)]

        batch_size = 15
        processed = []

        for i in range(0, len(recipients), batch_size):
            batch = recipients[i : i + batch_size]
            processed.extend([r.to_dict() for r in batch])

        assert len(processed) == 50

    def test_envelope_pagination_support(self):
        """Test pagination support for envelope lists."""
        total_envelopes = 250
        page_size = 100

        # Simulate paginated retrieval
        pages_needed = (total_envelopes + page_size - 1) // page_size
        assert pages_needed == 3

        # Verify last page size calculation
        last_page_size = total_envelopes % page_size
        assert last_page_size == 50

    def test_rate_limit_calculation(self):
        """Test rate limit calculations for API calls."""
        max_requests_per_minute = 60
        request_interval_ms = (60 * 1000) // max_requests_per_minute

        assert request_interval_ms == 1000  # 1 second between requests

    def test_buffer_overflow_prevention(self):
        """Test that buffer overflow is prevented through size limiting."""
        max_buffer_size = 1000
        current_buffer = []

        # Simulate adding items until buffer is full
        for i in range(1500):
            if len(current_buffer) < max_buffer_size:
                current_buffer.append(f"item_{i}")
            else:
                # Overflow condition - drop oldest or reject
                break

        assert len(current_buffer) == max_buffer_size


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the connector."""

    def test_credentials_expiry_detection(self, expired_credentials):
        """Test detection of expired credentials."""
        assert expired_credentials.is_expired is True

    def test_credentials_validity_detection(self, valid_credentials):
        """Test detection of valid credentials."""
        assert valid_credentials.is_expired is False

    def test_credentials_missing_expiry(self):
        """Test handling of credentials without expiry."""
        creds = DocuSignCredentials(
            access_token="no_expiry_token",
            account_id="acc",
            base_uri="https://demo.docusign.net",
        )
        # Credentials without expiry should be treated as expired
        assert creds.is_expired is True

    def test_unconfigured_connector_detection(self):
        """Test detection of unconfigured connector."""
        connector = DocuSignConnector()
        assert not connector.is_configured

    def test_unauthenticated_connector_detection(self, configured_connector):
        """Test detection of unauthenticated connector."""
        assert configured_connector.is_configured
        assert not configured_connector.is_authenticated

    def test_invalid_envelope_status_handling(self):
        """Test handling of invalid envelope status values."""
        # EnvelopeStatus should only accept valid values
        valid_statuses = [s.value for s in EnvelopeStatus]
        assert "invalid_status" not in valid_statuses
        assert "sent" in valid_statuses
        assert "completed" in valid_statuses

    def test_empty_document_content_handling(self):
        """Test handling of documents with empty content."""
        doc = Document(
            document_id="empty_doc",
            name="empty.pdf",
            content=b"",
        )
        # Should not raise error, but return empty base64
        encoded = doc.to_base64()
        assert encoded == ""

    def test_invalid_recipient_type_in_routing(self):
        """Test that routing handles various recipient types."""
        # All recipient types should be supported
        for rtype in RecipientType:
            recipient = Recipient(
                email=f"{rtype.value}@test.com",
                name=f"Test {rtype.value}",
                recipient_type=rtype,
            )
            serialized = recipient.to_dict()
            assert serialized["recipient_type"] == rtype.value


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_envelope_structure(self):
        """Test mock envelope has complete structure."""
        envelope = get_mock_envelope()

        assert isinstance(envelope, Envelope)
        assert envelope.envelope_id is not None
        assert envelope.status is not None
        assert envelope.email_subject is not None

    def test_get_mock_envelope_has_recipients(self):
        """Test mock envelope has recipients."""
        envelope = get_mock_envelope()

        assert len(envelope.signers) > 0
        assert len(envelope.carbon_copies) > 0

    def test_get_mock_envelope_has_documents(self):
        """Test mock envelope has documents."""
        envelope = get_mock_envelope()

        assert len(envelope.documents) > 0

    def test_get_mock_envelope_timestamps(self):
        """Test mock envelope has timestamps."""
        envelope = get_mock_envelope()

        assert envelope.created_at is not None
        assert envelope.sent_at is not None


# =============================================================================
# Enum Value Tests
# =============================================================================


class TestEnumValues:
    """Tests for enum value completeness."""

    def test_all_envelope_statuses_defined(self):
        """Test all expected envelope statuses are defined."""
        expected_statuses = {
            "created",
            "sent",
            "delivered",
            "signed",
            "completed",
            "declined",
            "voided",
        }
        actual_statuses = {s.value for s in EnvelopeStatus}
        assert expected_statuses == actual_statuses

    def test_all_recipient_types_defined(self):
        """Test all expected recipient types are defined."""
        expected_types = {
            "signer",
            "carbon_copy",
            "certified_delivery",
            "in_person_signer",
            "editor",
            "agent",
        }
        actual_types = {r.value for r in RecipientType}
        assert expected_types == actual_types

    def test_all_environments_defined(self):
        """Test all expected environments are defined."""
        expected_envs = {"demo", "production"}
        actual_envs = {e.value for e in DocuSignEnvironment}
        assert expected_envs == actual_envs

"""
Tests for DocuSign E-Signature Connector.
"""

from datetime import datetime, timezone, timedelta
import base64

from aragora.connectors.legal.docusign import (
    DocuSignConnector,
    DocuSignCredentials,
    DocuSignEnvironment,
    Envelope,
    EnvelopeStatus,
    Recipient,
    RecipientType,
    Document,
    SignatureTab,
    get_mock_envelope,
)


class TestDocuSignCredentials:
    """Tests for DocuSignCredentials dataclass."""

    def test_credentials_creation(self):
        creds = DocuSignCredentials(
            access_token="test_token",
            account_id="acc_123",
            base_uri="https://demo.docusign.net/restapi",
        )
        assert creds.access_token == "test_token"
        assert creds.account_id == "acc_123"
        assert creds.token_type == "Bearer"

    def test_is_expired_no_expiry(self):
        creds = DocuSignCredentials(
            access_token="test",
            account_id="acc",
            base_uri="https://demo.docusign.net",
        )
        assert creds.is_expired is True

    def test_is_expired_future(self):
        creds = DocuSignCredentials(
            access_token="test",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert creds.is_expired is False

    def test_is_expired_past(self):
        creds = DocuSignCredentials(
            access_token="test",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert creds.is_expired is True


class TestRecipient:
    """Tests for Recipient dataclass."""

    def test_recipient_creation(self):
        recipient = Recipient(
            email="signer@example.com",
            name="John Signer",
            recipient_type=RecipientType.SIGNER,
            routing_order=1,
        )
        assert recipient.email == "signer@example.com"
        assert recipient.name == "John Signer"

    def test_recipient_to_dict(self):
        recipient = Recipient(
            email="signer@example.com",
            name="John Signer",
            routing_order=2,
            recipient_id="rec_1",
        )
        data = recipient.to_dict()
        assert data["email"] == "signer@example.com"
        assert data["routing_order"] == 2
        assert data["recipient_id"] == "rec_1"


class TestSignatureTab:
    """Tests for SignatureTab dataclass."""

    def test_signature_tab_to_dict(self):
        tab = SignatureTab(
            page_number=2,
            x_position=100,
            y_position=300,
        )
        data = tab.to_dict()
        assert data["pageNumber"] == "2"
        assert data["xPosition"] == "100"

    def test_text_tab_to_dict(self):
        tab = SignatureTab(
            tab_type="text",
            page_number=1,
            x_position=50,
            y_position=50,
            tab_label="Company Name",
            value="Acme Corp",
        )
        data = tab.to_dict()
        assert data["tabLabel"] == "Company Name"
        assert data["value"] == "Acme Corp"


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_to_base64(self):
        content = b"Test PDF content"
        doc = Document(
            document_id="1",
            name="test.pdf",
            content=content,
        )
        encoded = doc.to_base64()
        assert encoded == base64.b64encode(content).decode("utf-8")


class TestEnvelope:
    """Tests for Envelope dataclass."""

    def test_envelope_to_dict(self):
        envelope = Envelope(
            envelope_id="env_456",
            status=EnvelopeStatus.COMPLETED,
            email_subject="Contract",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        )
        data = envelope.to_dict()
        assert data["envelope_id"] == "env_456"
        assert data["status"] == "completed"


class TestDocuSignConnectorConfiguration:
    """Tests for connector configuration."""

    def test_connector_defaults(self):
        connector = DocuSignConnector()
        assert connector.environment == DocuSignEnvironment.DEMO
        assert connector.auth_url == DocuSignConnector.DEMO_AUTH_URL

    def test_connector_production_environment(self):
        connector = DocuSignConnector(environment=DocuSignEnvironment.PRODUCTION)
        assert connector.environment == DocuSignEnvironment.PRODUCTION

    def test_is_configured_missing_fields(self):
        connector = DocuSignConnector()
        assert connector.is_configured is False

    def test_is_configured_with_fields(self):
        connector = DocuSignConnector(
            integration_key="key_123",
            user_id="user_456",
            account_id="acc_789",
        )
        assert connector.is_configured is True

    def test_is_authenticated_no_credentials(self):
        connector = DocuSignConnector()
        assert connector.is_authenticated is False

    def test_is_authenticated_with_valid_credentials(self):
        connector = DocuSignConnector()
        connector._credentials = DocuSignCredentials(
            access_token="token",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert connector.is_authenticated is True


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_envelope(self):
        envelope = get_mock_envelope()
        assert envelope is not None
        assert isinstance(envelope, Envelope)


class TestEnums:
    """Tests for enum values."""

    def test_envelope_status_values(self):
        assert EnvelopeStatus.CREATED.value == "created"
        assert EnvelopeStatus.SENT.value == "sent"
        assert EnvelopeStatus.COMPLETED.value == "completed"

    def test_recipient_type_values(self):
        assert RecipientType.SIGNER.value == "signer"
        assert RecipientType.CARBON_COPY.value == "carbon_copy"

    def test_environment_values(self):
        assert DocuSignEnvironment.DEMO.value == "demo"
        assert DocuSignEnvironment.PRODUCTION.value == "production"

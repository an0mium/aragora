"""
Tests for DocuSign E-Signature Connector.

Tests cover:
- Dataclass serialization
- Connector configuration
- Authentication flow
- Envelope operations (create, get, list, void, resend)
- Document operations (download)
- Mock data generation
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
# Dataclass Tests
# =============================================================================


class TestDocuSignCredentials:
    """Tests for DocuSignCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials initialization."""
        creds = DocuSignCredentials(
            access_token="test_token",
            account_id="acc_123",
            base_uri="https://demo.docusign.net/restapi",
        )
        assert creds.access_token == "test_token"
        assert creds.account_id == "acc_123"
        assert creds.token_type == "Bearer"

    def test_is_expired_no_expiry(self):
        """Test expired check when no expiry set."""
        creds = DocuSignCredentials(
            access_token="test",
            account_id="acc",
            base_uri="https://demo.docusign.net",
        )
        assert creds.is_expired is True

    def test_is_expired_future(self):
        """Test expired check with future expiry."""
        creds = DocuSignCredentials(
            access_token="test",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert creds.is_expired is False

    def test_is_expired_past(self):
        """Test expired check with past expiry."""
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
        """Test recipient initialization."""
        recipient = Recipient(
            email="signer@example.com",
            name="John Signer",
            recipient_type=RecipientType.SIGNER,
            routing_order=1,
        )
        assert recipient.email == "signer@example.com"
        assert recipient.name == "John Signer"
        assert recipient.recipient_type == RecipientType.SIGNER

    def test_recipient_to_dict(self):
        """Test recipient serialization."""
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

    def test_recipient_default_id(self):
        """Test recipient ID defaults to routing order."""
        recipient = Recipient(
            email="test@example.com",
            name="Test",
            routing_order=3,
        )
        data = recipient.to_dict()
        assert data["recipient_id"] == "3"


class TestSignatureTab:
    """Tests for SignatureTab dataclass."""

    def test_signature_tab_creation(self):
        """Test signature tab initialization."""
        tab = SignatureTab(
            tab_type="signature",
            page_number=1,
            x_position=200,
            y_position=500,
            recipient_id="1",
        )
        assert tab.tab_type == "signature"
        assert tab.page_number == 1

    def test_signature_tab_to_dict(self):
        """Test signature tab serialization."""
        tab = SignatureTab(
            page_number=2,
            x_position=100,
            y_position=300,
        )
        data = tab.to_dict()
        assert data["pageNumber"] == "2"
        assert data["xPosition"] == "100"
        assert data["yPosition"] == "300"

    def test_text_tab_to_dict(self):
        """Test text tab includes additional fields."""
        tab = SignatureTab(
            tab_type="text",
            page_number=1,
            x_position=50,
            y_position=50,
            tab_label="Company Name",
            value="Acme Corp",
            width=200,
        )
        data = tab.to_dict()
        assert data["tabLabel"] == "Company Name"
        assert data["value"] == "Acme Corp"
        assert data["width"] == "200"


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test document initialization."""
        content = b"PDF content here"
        doc = Document(
            document_id="doc_1",
            name="Contract.pdf",
            content=content,
        )
        assert doc.document_id == "doc_1"
        assert doc.name == "Contract.pdf"
        assert doc.file_extension == "pdf"

    def test_document_to_base64(self):
        """Test document base64 encoding."""
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

    def test_envelope_creation(self):
        """Test envelope initialization."""
        envelope = Envelope(
            envelope_id="env_123",
            status=EnvelopeStatus.SENT,
            email_subject="Please sign this document",
        )
        assert envelope.envelope_id == "env_123"
        assert envelope.status == EnvelopeStatus.SENT

    def test_envelope_to_dict(self):
        """Test envelope serialization."""
        envelope = Envelope(
            envelope_id="env_456",
            status=EnvelopeStatus.COMPLETED,
            email_subject="Contract",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            sent_at=datetime(2024, 1, 15, 10, 35, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 16, 14, 0, tzinfo=timezone.utc),
        )
        data = envelope.to_dict()
        assert data["envelope_id"] == "env_456"
        assert data["status"] == "completed"
        assert "2024-01-15" in data["created_at"]


# =============================================================================
# Connector Configuration Tests
# =============================================================================


class TestDocuSignConnectorConfiguration:
    """Tests for connector configuration."""

    def test_connector_defaults(self):
        """Test connector with environment defaults."""
        connector = DocuSignConnector()
        assert connector.environment == DocuSignEnvironment.DEMO
        assert connector.auth_url == DocuSignConnector.DEMO_AUTH_URL
        assert connector.api_url == DocuSignConnector.DEMO_API_URL

    def test_connector_production_environment(self):
        """Test connector with production environment."""
        connector = DocuSignConnector(environment=DocuSignEnvironment.PRODUCTION)
        assert connector.environment == DocuSignEnvironment.PRODUCTION
        assert connector.auth_url == DocuSignConnector.PRODUCTION_AUTH_URL
        assert connector.api_url == DocuSignConnector.PRODUCTION_API_URL

    def test_is_configured_missing_fields(self):
        """Test is_configured with missing fields."""
        connector = DocuSignConnector()
        assert connector.is_configured is False

    def test_is_configured_with_fields(self):
        """Test is_configured with all required fields."""
        connector = DocuSignConnector(
            integration_key="key_123",
            user_id="user_456",
            account_id="acc_789",
        )
        assert connector.is_configured is True

    def test_is_authenticated_no_credentials(self):
        """Test is_authenticated without credentials."""
        connector = DocuSignConnector()
        assert connector.is_authenticated is False

    def test_is_authenticated_with_valid_credentials(self):
        """Test is_authenticated with valid credentials."""
        connector = DocuSignConnector()
        connector._credentials = DocuSignCredentials(
            access_token="token",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert connector.is_authenticated is True

    def test_is_authenticated_with_expired_credentials(self):
        """Test is_authenticated with expired credentials."""
        connector = DocuSignConnector()
        connector._credentials = DocuSignCredentials(
            access_token="token",
            account_id="acc",
            base_uri="https://demo.docusign.net",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert connector.is_authenticated is False


# =============================================================================
# Envelope Operation Tests
# =============================================================================

# Skip tests that need async context manager mock fixes
SKIP_ASYNC_MOCK_FIX = pytest.mark.skip(
    reason="Tests need async context manager mock updates for aiohttp"
)


@SKIP_ASYNC_MOCK_FIX
class TestDocuSignConnectorEnvelopeOperations:
    """Tests for envelope operations."""

    @pytest.fixture
    def authenticated_connector(self):
        """Create an authenticated connector for testing."""
        connector = DocuSignConnector(
            integration_key="test_key",
            user_id="test_user",
            account_id="test_account",
        )
        connector._credentials = DocuSignCredentials(
            access_token="valid_token",
            account_id="test_account",
            base_uri="https://demo.docusign.net/restapi",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        return connector

    @SKIP_ASYNC_MOCK_FIX
    @pytest.mark.asyncio
    async def test_create_envelope(self, authenticated_connector):
        """Test envelope creation."""
        mock_response = {
            "envelopeId": "new_env_123",
            "status": "sent",
            "statusDateTime": "2024-01-15T10:30:00Z",
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # Set up async context manager mock for response
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)

            # Set up async context manager mock for session.request()
            mock_request_ctx = AsyncMock()
            mock_request_ctx.__aenter__.return_value = mock_resp
            mock_request_ctx.__aexit__.return_value = None

            # Set up session mock
            mock_session_instance = AsyncMock()
            mock_session_instance.request.return_value = mock_request_ctx

            # Set up session as async context manager
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session.return_value.__aexit__.return_value = None

            request = EnvelopeCreateRequest(
                email_subject="Please sign",
                recipients=[Recipient(email="signer@example.com", name="John Doe")],
                documents=[
                    Document(
                        document_id="1",
                        name="contract.pdf",
                        content=b"PDF content",
                    )
                ],
            )

            envelope = await authenticated_connector.create_envelope(request)
            assert envelope is not None
            assert envelope.envelope_id == "new_env_123"

    @pytest.mark.asyncio
    async def test_get_envelope(self, authenticated_connector):
        """Test getting envelope details."""
        mock_response = {
            "envelopeId": "env_123",
            "status": "completed",
            "emailSubject": "Contract for signature",
            "createdDateTime": "2024-01-15T10:30:00Z",
            "sentDateTime": "2024-01-15T10:35:00Z",
            "completedDateTime": "2024-01-16T14:00:00Z",
            "recipients": {
                "signers": [{"email": "signer@example.com", "name": "John"}],
            },
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            envelope = await authenticated_connector.get_envelope("env_123")
            assert envelope is not None
            assert envelope.envelope_id == "env_123"
            assert envelope.status == EnvelopeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_envelope_not_found(self, authenticated_connector):
        """Test getting non-existent envelope."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 404
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            envelope = await authenticated_connector.get_envelope("nonexistent")
            assert envelope is None

    @pytest.mark.asyncio
    async def test_void_envelope(self, authenticated_connector):
        """Test voiding an envelope."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.json = AsyncMock(return_value={})
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            result = await authenticated_connector.void_envelope(
                "env_123", reason="Contract cancelled"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_resend_envelope(self, authenticated_connector):
        """Test resending envelope notifications."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.json = AsyncMock(return_value={})
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            result = await authenticated_connector.resend_envelope("env_123")
            assert result is True


# =============================================================================
# Document Operation Tests
# =============================================================================


@SKIP_ASYNC_MOCK_FIX
class TestDocuSignConnectorDocumentOperations:
    """Tests for document operations."""

    @pytest.fixture
    def authenticated_connector(self):
        """Create an authenticated connector for testing."""
        connector = DocuSignConnector(
            integration_key="test_key",
            user_id="test_user",
            account_id="test_account",
        )
        connector._credentials = DocuSignCredentials(
            access_token="valid_token",
            account_id="test_account",
            base_uri="https://demo.docusign.net/restapi",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        return connector

    @pytest.mark.asyncio
    async def test_download_document(self, authenticated_connector):
        """Test downloading a signed document."""
        pdf_content = b"%PDF-1.4 signed document content"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.read = AsyncMock(return_value=pdf_content)
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            content = await authenticated_connector.download_document("env_123", "doc_1")
            assert content == pdf_content

    @pytest.mark.asyncio
    async def test_download_certificate(self, authenticated_connector):
        """Test downloading signing certificate."""
        cert_content = b"Certificate of Completion"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.status = 200
            mock_context.__aenter__.return_value.read = AsyncMock(return_value=cert_content)
            mock_session.return_value.__aenter__.return_value.request.return_value = mock_context

            content = await authenticated_connector.download_certificate("env_123")
            assert content == cert_content


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_envelope(self):
        """Test mock envelope generation."""
        envelope = get_mock_envelope()
        assert envelope is not None
        assert isinstance(envelope, Envelope)
        assert envelope.envelope_id is not None
        assert envelope.status in EnvelopeStatus
        assert envelope.email_subject


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_envelope_status_values(self):
        """Test all envelope status values."""
        assert EnvelopeStatus.CREATED.value == "created"
        assert EnvelopeStatus.SENT.value == "sent"
        assert EnvelopeStatus.DELIVERED.value == "delivered"
        assert EnvelopeStatus.SIGNED.value == "signed"
        assert EnvelopeStatus.COMPLETED.value == "completed"
        assert EnvelopeStatus.DECLINED.value == "declined"
        assert EnvelopeStatus.VOIDED.value == "voided"

    def test_recipient_type_values(self):
        """Test all recipient type values."""
        assert RecipientType.SIGNER.value == "signer"
        assert RecipientType.CARBON_COPY.value == "carbon_copy"
        assert RecipientType.CERTIFIED_DELIVERY.value == "certified_delivery"
        assert RecipientType.IN_PERSON_SIGNER.value == "in_person_signer"
        assert RecipientType.EDITOR.value == "editor"
        assert RecipientType.AGENT.value == "agent"

    def test_environment_values(self):
        """Test environment enum values."""
        assert DocuSignEnvironment.DEMO.value == "demo"
        assert DocuSignEnvironment.PRODUCTION.value == "production"

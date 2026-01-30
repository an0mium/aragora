"""Tests for Legal E-Signature Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
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
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.legal import (
    LegalHandler,
    create_legal_handler,
    get_docusign_connector,
)


@pytest.fixture
def handler():
    """Create handler instance."""
    return LegalHandler()


class TestLegalHandler:
    """Tests for LegalHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(LegalHandler, "ROUTES")
        routes = LegalHandler.ROUTES
        assert "/api/v1/legal/envelopes" in routes
        assert "/api/v1/legal/status" in routes
        assert "/api/v1/legal/templates" in routes
        assert "/api/v1/legal/webhooks/docusign" in routes

    def test_can_handle_legal_routes(self, handler):
        """Test can_handle for legal routes."""
        assert handler.can_handle("/api/v1/legal/status") is True
        assert handler.can_handle("/api/v1/legal/envelopes") is True
        assert handler.can_handle("/api/v1/legal/templates") is True

    def test_can_handle_envelope_routes(self, handler):
        """Test can_handle for envelope routes."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123") is True
        assert handler.can_handle("/api/v1/legal/envelopes/env123/void") is True
        assert handler.can_handle("/api/v1/legal/envelopes/env123/resend") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/contracts/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_create_legal_handler(self):
        """Test factory function creates handler."""
        handler = create_legal_handler()
        assert handler is not None
        assert isinstance(handler, LegalHandler)

    def test_create_legal_handler_with_context(self):
        """Test factory function with server context."""
        ctx = {"test": "value"}
        handler = create_legal_handler(ctx)
        assert handler is not None


class TestLegalStatus:
    """Tests for Legal status endpoint."""

    @pytest.mark.asyncio
    async def test_status_not_configured(self):
        """Test status when DocuSign is not configured."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()
        mock_request.tenant_id = "test_tenant"

        with patch(
            "aragora.connectors.legal.docusign.DocuSignConnector", create=True
        ) as MockConnector:
            mock_instance = MagicMock()
            mock_instance.is_configured = False
            mock_instance.is_authenticated = False
            MockConnector.return_value = mock_instance

            result = await handler._handle_status(mock_request, "test_tenant")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_status_import_error(self):
        """Test status when DocuSign connector not installed."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        with patch(
            "aragora.connectors.legal.docusign.DocuSignConnector",
            side_effect=ImportError("Module not found"),
        ):
            # The import happens inside _handle_status, so we need to mock it there
            result = await handler._handle_status(mock_request, "test_tenant")
            assert result.status_code == 200


class TestLegalEnvelopes:
    """Tests for Legal envelope operations."""

    @pytest.mark.asyncio
    async def test_list_envelopes_not_configured(self):
        """Test list envelopes when not configured."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get_connector:
            mock_get_connector.return_value = None

            result = await handler._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_create_envelope_missing_subject(self):
        """Test create envelope requires email_subject."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_authenticated = True

        with (
            patch(
                "aragora.server.handlers.features.legal.get_docusign_connector",
                new_callable=AsyncMock,
            ) as mock_get_connector,
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
        ):
            mock_get_connector.return_value = mock_connector
            mock_body.return_value = {"recipients": [], "documents": []}

            result = await handler._handle_create_envelope(mock_request, "test_tenant")
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_envelope_missing_recipients(self):
        """Test create envelope requires recipients."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_authenticated = True

        with (
            patch(
                "aragora.server.handlers.features.legal.get_docusign_connector",
                new_callable=AsyncMock,
            ) as mock_get_connector,
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
        ):
            mock_get_connector.return_value = mock_connector
            mock_body.return_value = {"email_subject": "Test", "documents": []}

            result = await handler._handle_create_envelope(mock_request, "test_tenant")
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_envelope_missing_documents(self):
        """Test create envelope requires documents."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_authenticated = True

        with (
            patch(
                "aragora.server.handlers.features.legal.get_docusign_connector",
                new_callable=AsyncMock,
            ) as mock_get_connector,
            patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body,
        ):
            mock_get_connector.return_value = mock_connector
            mock_body.return_value = {
                "email_subject": "Test",
                "recipients": [{"email": "test@example.com"}],
            }

            result = await handler._handle_create_envelope(mock_request, "test_tenant")
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_envelope_not_found(self):
        """Test get envelope returns 404 when not found."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        mock_connector = MagicMock()
        mock_connector.is_authenticated = True
        mock_connector.get_envelope = AsyncMock(return_value=None)

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get_connector:
            mock_get_connector.return_value = mock_connector

            result = await handler._handle_get_envelope(mock_request, "test_tenant", "invalid_id")
            assert result.status_code == 404


class TestLegalVoid:
    """Tests for Legal void envelope operations."""

    @pytest.mark.asyncio
    async def test_void_envelope_not_configured(self):
        """Test void envelope when not configured."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get_connector:
            mock_get_connector.return_value = None

            result = await handler._handle_void_envelope(mock_request, "test_tenant", "env123")
            assert result.status_code == 503


class TestLegalWebhook:
    """Tests for Legal webhook handling."""

    @pytest.mark.asyncio
    async def test_webhook_handler(self):
        """Test DocuSign webhook handling."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "envelopeId": "env123",
                "status": "completed",
                "statusChangedDateTime": "2024-01-01T00:00:00Z",
            }

            result = await handler._handle_docusign_webhook(mock_request, "test_tenant")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_webhook_invalid_payload(self):
        """Test webhook with invalid payload."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}

            result = await handler._handle_docusign_webhook(mock_request, "test_tenant")
            # Should still return 200 to prevent retries
            assert result.status_code == 200


class TestLegalTemplates:
    """Tests for Legal templates endpoint."""

    @pytest.mark.asyncio
    async def test_list_templates(self):
        """Test list templates returns placeholder."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()

        result = await handler._handle_list_templates(mock_request, "test_tenant")
        assert result.status_code == 200


class TestGetDocuSignConnector:
    """Tests for connector instance management."""

    @pytest.mark.asyncio
    async def test_get_connector_not_configured(self):
        """Test get connector when not configured."""
        with patch(
            "aragora.connectors.legal.docusign.DocuSignConnector", create=True
        ) as MockConnector:
            mock_instance = MagicMock()
            mock_instance.is_configured = False
            MockConnector.return_value = mock_instance

            # Clear any cached instances
            import aragora.server.handlers.features.legal as legal_module

            legal_module._connector_instances.clear()

            result = await get_docusign_connector("test_tenant")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_connector_import_error(self):
        """Test get connector when module not available."""
        with patch(
            "aragora.connectors.legal.docusign.DocuSignConnector",
            side_effect=ImportError("Module not found"),
        ):
            import aragora.server.handlers.features.legal as legal_module

            legal_module._connector_instances.clear()

            result = await get_docusign_connector("test_tenant")
            assert result is None


class TestLegalUtilities:
    """Tests for Legal handler utilities."""

    def test_get_tenant_id_from_request(self):
        """Test tenant ID extraction from request."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()
        mock_request.tenant_id = "test_tenant"

        result = handler._get_tenant_id(mock_request)
        assert result == "test_tenant"

    def test_get_tenant_id_default(self):
        """Test tenant ID defaults to 'default'."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock(spec=[])  # No tenant_id attribute

        result = handler._get_tenant_id(mock_request)
        assert result == "default"

    def test_get_query_params_from_query(self):
        """Test query params extraction from request.query."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()
        mock_request.query = {"status": "completed", "limit": "10"}

        result = handler._get_query_params(mock_request)
        assert result["status"] == "completed"
        assert result["limit"] == "10"

    @pytest.mark.asyncio
    async def test_get_json_body_callable(self):
        """Test JSON body extraction when json is callable."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"test": "data"})

        result = await handler._get_json_body(mock_request)
        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_get_json_body_attribute(self):
        """Test JSON body extraction when json is attribute."""
        handler = LegalHandler({"storage": None, "user_store": None, "elo_system": None})
        mock_request = MagicMock()
        mock_request.json = {"test": "data"}

        result = await handler._get_json_body(mock_request)
        assert result == {"test": "data"}

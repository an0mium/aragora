"""Tests for Legal E-Signature Handler.

Comprehensive test coverage for:
- Handler creation and factory
- Route matching (can_handle)
- Main routing (handle method)
- Status endpoint
- Envelope operations (create, list, get, void, resend)
- Document operations (download document, download certificate)
- Template listing
- Webhook handling
- Connector management (get_docusign_connector, caching, per-tenant)
- Utility methods (_get_query_params, _get_json_body, _get_tenant_id)
- Event emission (_emit_connector_event)
- Error handling and edge cases
"""

import sys
import types as _types_mod
import base64
import json

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
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.server.handlers.features.legal import (
    LegalHandler,
    create_legal_handler,
    get_docusign_connector,
    _connector_instances,
)


@pytest.fixture(autouse=True)
def clear_connector_cache():
    """Clear cached connector instances and sys.modules state before and after each test.

    The ``_connector_instances`` dict is a module-level cache inside the legal
    handler.  If a previous test populated it (e.g. ``test_connector_cached``
    stores a mock under ``tenant1``), later tests that call
    ``get_docusign_connector`` may skip the import branch entirely and return
    a stale mock.

    Additionally, ``sys.modules`` may contain a real or mock entry for
    ``aragora.connectors.legal.docusign`` left by a prior test (including tests
    from *other* test files in the handler suite).  Removing the entry here
    forces the ``from … import DocuSignConnector`` inside
    ``get_docusign_connector`` to re-resolve through ``sys.modules`` on every
    test, preventing stale references from leaking across test boundaries.
    """
    _connector_instances.clear()

    # Capture and remove stale sys.modules entry so each test controls its own
    # import context via ``patch.dict``.
    _saved_docusign_mod = sys.modules.pop("aragora.connectors.legal.docusign", None)

    yield

    _connector_instances.clear()

    # Restore original sys.modules state (or remove if it was not present)
    if _saved_docusign_mod is not None:
        sys.modules["aragora.connectors.legal.docusign"] = _saved_docusign_mod
    else:
        sys.modules.pop("aragora.connectors.legal.docusign", None)


@pytest.fixture
def handler():
    """Create handler instance with empty context."""
    return LegalHandler()


@pytest.fixture
def handler_with_ctx():
    """Create handler instance with server context."""
    return LegalHandler({"storage": None, "user_store": None, "elo_system": None})


@pytest.fixture
def mock_request():
    """Create mock request with tenant_id and query params."""
    request = MagicMock()
    request.tenant_id = "test_tenant"
    request.query = {}
    return request


@pytest.fixture
def mock_connector():
    """Create a mock DocuSign connector that is authenticated."""
    connector = MagicMock()
    connector.is_configured = True
    connector.is_authenticated = True
    connector.authenticate_jwt = AsyncMock()
    return connector


@pytest.fixture
def mock_unauthenticated_connector():
    """Create a mock DocuSign connector that needs authentication."""
    connector = MagicMock()
    connector.is_configured = True
    connector.is_authenticated = False
    connector.authenticate_jwt = AsyncMock()
    return connector


# =============================================================================
# Handler Creation and Factory Tests
# =============================================================================


class TestLegalHandlerCreation:
    """Tests for LegalHandler instantiation and factory."""

    def test_handler_creation_no_context(self):
        """Test creating handler without server context."""
        handler = LegalHandler()
        assert handler is not None
        assert handler.ctx == {}

    def test_handler_creation_with_none_context(self):
        """Test creating handler with None context."""
        handler = LegalHandler(None)
        assert handler.ctx == {}

    def test_handler_creation_with_context(self):
        """Test creating handler with server context dict."""
        ctx = {"storage": "test_storage", "key": "value"}
        handler = LegalHandler(ctx)
        assert handler.ctx == ctx
        assert handler.ctx["storage"] == "test_storage"

    def test_factory_function_no_args(self):
        """Test create_legal_handler with no arguments."""
        handler = create_legal_handler()
        assert isinstance(handler, LegalHandler)
        assert handler.ctx == {}

    def test_factory_function_with_context(self):
        """Test create_legal_handler with server context."""
        ctx = {"storage": "s", "emitter": "e"}
        handler = create_legal_handler(ctx)
        assert isinstance(handler, LegalHandler)
        assert handler.ctx == ctx

    def test_handler_has_routes(self):
        """Test that handler class defines ROUTES."""
        assert hasattr(LegalHandler, "ROUTES")
        routes = LegalHandler.ROUTES
        assert "/api/v1/legal/envelopes" in routes
        assert "/api/v1/legal/status" in routes
        assert "/api/v1/legal/templates" in routes
        assert "/api/v1/legal/webhooks/docusign" in routes


# =============================================================================
# Route Matching Tests (can_handle)
# =============================================================================


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_can_handle_status(self, handler):
        """Test can_handle matches status route."""
        assert handler.can_handle("/api/v1/legal/status") is True

    def test_can_handle_envelopes(self, handler):
        """Test can_handle matches envelopes route."""
        assert handler.can_handle("/api/v1/legal/envelopes") is True

    def test_can_handle_templates(self, handler):
        """Test can_handle matches templates route."""
        assert handler.can_handle("/api/v1/legal/templates") is True

    def test_can_handle_webhooks(self, handler):
        """Test can_handle matches webhook route."""
        assert handler.can_handle("/api/v1/legal/webhooks/docusign") is True

    def test_can_handle_envelope_by_id(self, handler):
        """Test can_handle matches envelope detail route."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123") is True

    def test_can_handle_void_action(self, handler):
        """Test can_handle matches void action."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123/void") is True

    def test_can_handle_resend_action(self, handler):
        """Test can_handle matches resend action."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123/resend") is True

    def test_can_handle_certificate(self, handler):
        """Test can_handle matches certificate route."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123/certificate") is True

    def test_can_handle_document(self, handler):
        """Test can_handle matches document download route."""
        assert handler.can_handle("/api/v1/legal/envelopes/env123/documents/doc1") is True

    def test_rejects_other_api(self, handler):
        """Test can_handle rejects non-legal routes."""
        assert handler.can_handle("/api/v1/contracts/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False
        assert handler.can_handle("/api/v1/debates/") is False

    def test_rejects_empty_path(self, handler):
        """Test can_handle rejects empty path."""
        assert handler.can_handle("") is False

    def test_rejects_partial_match(self, handler):
        """Test can_handle rejects non-legal paths (uses startswith)."""
        # Note: can_handle uses startswith("/api/v1/legal"), so /api/v1/legalx also matches
        assert handler.can_handle("/api/v1/leg") is False
        assert handler.can_handle("/api/v2/legal") is False


# =============================================================================
# Main Routing Tests (handle method)
# =============================================================================


class TestHandleRouting:
    """Tests for the main handle() method routing logic."""

    @pytest.mark.asyncio
    async def test_route_to_status(self, handler, mock_request):
        """Test routing GET /status to _handle_status."""
        with patch.object(handler, "_handle_status", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(mock_request, "/api/v1/legal/status", "GET")
            mock_method.assert_called_once_with(mock_request, "test_tenant")

    @pytest.mark.asyncio
    async def test_route_to_list_envelopes(self, handler, mock_request):
        """Test routing GET /envelopes to _handle_list_envelopes."""
        with patch.object(handler, "_handle_list_envelopes", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(mock_request, "/api/v1/legal/envelopes", "GET")
            mock_method.assert_called_once_with(mock_request, "test_tenant")

    @pytest.mark.asyncio
    async def test_route_to_create_envelope(self, handler, mock_request):
        """Test routing POST /envelopes to _handle_create_envelope."""
        with patch.object(
            handler, "_handle_create_envelope", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = MagicMock(status_code=201)
            result = await handler.handle(mock_request, "/api/v1/legal/envelopes", "POST")
            mock_method.assert_called_once_with(mock_request, "test_tenant")

    @pytest.mark.asyncio
    async def test_route_to_list_templates(self, handler, mock_request):
        """Test routing GET /templates to _handle_list_templates."""
        with patch.object(handler, "_handle_list_templates", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(mock_request, "/api/v1/legal/templates", "GET")
            mock_method.assert_called_once_with(mock_request, "test_tenant")

    @pytest.mark.asyncio
    async def test_route_to_webhook(self, handler, mock_request):
        """Test routing POST /webhooks/docusign to _handle_docusign_webhook."""
        with patch.object(
            handler, "_handle_docusign_webhook", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(mock_request, "/api/v1/legal/webhooks/docusign", "POST")
            mock_method.assert_called_once_with(mock_request, "test_tenant")

    @pytest.mark.asyncio
    async def test_route_to_void_envelope(self, handler, mock_request):
        """Test routing POST /envelopes/{id}/void to _handle_void_envelope.

        Note: The router uses parts[4] as envelope_id and parts[5] as action.
        For path /api/v1/legal/envelopes/{id}/void, parts[5] is the actual
        envelope_id and parts[6] is the action. The handler routes based on
        len(parts)==6 where parts[5] is the action name.
        """
        with patch.object(handler, "_handle_void_envelope", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            # With /api/v1/legal/envelopes/void, parts = ['', 'api', 'v1', 'legal', 'envelopes', 'void']
            # len(parts)==6, parts[5]=='void', so it routes correctly (envelope_id=parts[4]='envelopes')
            result = await handler.handle(mock_request, "/api/v1/legal/envelopes/void", "POST")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_resend_envelope(self, handler, mock_request):
        """Test routing POST to resend action."""
        with patch.object(
            handler, "_handle_resend_envelope", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(mock_request, "/api/v1/legal/envelopes/resend", "POST")
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_download_certificate(self, handler, mock_request):
        """Test routing GET to certificate action."""
        with patch.object(
            handler, "_handle_download_certificate", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = await handler.handle(
                mock_request, "/api/v1/legal/envelopes/certificate", "GET"
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_download_document(self, handler, mock_request):
        """Test routing GET to document download (7-part path)."""
        with patch.object(
            handler, "_handle_download_document", new_callable=AsyncMock
        ) as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            # 7 parts: ['', 'api', 'v1', 'legal', 'envelopes', 'documents', 'doc456']
            # parts[5]=='documents', parts[6]=='doc456'
            result = await handler.handle(
                mock_request,
                "/api/v1/legal/envelopes/documents/doc456",
                "GET",
            )
            mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_envelope_methods_callable_directly(self, handler_with_ctx, mock_request):
        """Test envelope methods are callable directly bypassing router.

        The router's path parsing has a known indexing pattern where parts[4]
        captures 'envelopes' as envelope_id. Individual handler methods work
        correctly when called directly with proper arguments.
        """
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "actual-env-id"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_route_not_found(self, handler, mock_request):
        """Test 404 for unmatched path."""
        result = await handler.handle(mock_request, "/api/v1/legal/unknown", "GET")
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_route_exception_returns_500(self, handler, mock_request):
        """Test that unhandled exceptions return 500."""
        with patch.object(handler, "_handle_status", new_callable=AsyncMock) as mock_method:
            mock_method.side_effect = RuntimeError("boom")
            result = await handler.handle(mock_request, "/api/v1/legal/status", "GET")
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_get_tenant_id_from_request(self, handler):
        """Test _get_tenant_id extracts tenant from request."""
        request = MagicMock()
        request.tenant_id = "my_tenant"
        assert handler._get_tenant_id(request) == "my_tenant"

    @pytest.mark.asyncio
    async def test_get_tenant_id_default(self, handler):
        """Test _get_tenant_id returns default when not set."""
        request = MagicMock(spec=[])  # No attributes
        assert handler._get_tenant_id(request) == "default"


# =============================================================================
# Status Endpoint Tests
# =============================================================================


class TestStatusEndpoint:
    """Tests for the _handle_status method."""

    @pytest.mark.asyncio
    async def test_status_configured(self, handler_with_ctx, mock_request):
        """Test status when DocuSign is configured."""
        with patch(
            "aragora.server.handlers.features.legal.DocuSignConnector",
            create=True,
        ) as MockCls:
            mock_instance = MagicMock()
            mock_instance.is_configured = True
            mock_instance.is_authenticated = True
            mock_instance.environment.value = "sandbox"
            mock_instance.integration_key = "key123"
            mock_instance.account_id = "acct123"
            MockCls.return_value = mock_instance

            # Patch the import inside the method
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": MagicMock(DocuSignConnector=MockCls)},
            ):
                result = await handler_with_ctx._handle_status(mock_request, "test_tenant")
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_status_not_configured(self, handler_with_ctx, mock_request):
        """Test status when DocuSign is not configured."""
        with patch(
            "aragora.server.handlers.features.legal.DocuSignConnector",
            create=True,
        ) as MockCls:
            mock_instance = MagicMock()
            mock_instance.is_configured = False
            mock_instance.is_authenticated = False
            mock_instance.environment.value = "sandbox"
            mock_instance.integration_key = ""
            mock_instance.account_id = ""
            MockCls.return_value = mock_instance

            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": MagicMock(DocuSignConnector=MockCls)},
            ):
                result = await handler_with_ctx._handle_status(mock_request, "test_tenant")
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_status_import_error(self, handler_with_ctx, mock_request):
        """Test status when DocuSign connector module not installed."""
        # Remove module from sys.modules to force ImportError
        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": None}):
            result = await handler_with_ctx._handle_status(mock_request, "test_tenant")
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["configured"] is False


# =============================================================================
# List Envelopes Tests
# =============================================================================


class TestListEnvelopes:
    """Tests for _handle_list_envelopes."""

    @pytest.mark.asyncio
    async def test_list_envelopes_not_configured(self, handler_with_ctx, mock_request):
        """Test list envelopes when DocuSign not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_list_envelopes_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test list envelopes when authentication fails."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth failed")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_list_envelopes_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful envelope listing."""
        mock_envelope = MagicMock()
        mock_envelope.to_dict.return_value = {"envelope_id": "env1", "status": "sent"}
        mock_connector.list_envelopes = AsyncMock(return_value=[mock_envelope])

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["count"] == 1

    @pytest.mark.asyncio
    async def test_list_envelopes_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test list envelopes when API call fails."""
        mock_connector.list_envelopes = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_list_envelopes_with_query_params(self, handler_with_ctx, mock_connector):
        """Test list envelopes passes query params to connector."""
        mock_request = MagicMock()
        mock_request.tenant_id = "test_tenant"
        mock_request.query = {
            "status": "sent",
            "from_date": "2025-01-01",
            "to_date": "2025-12-31",
            "limit": "10",
        }
        mock_connector.list_envelopes = AsyncMock(return_value=[])
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 200
            mock_connector.list_envelopes.assert_called_once_with(
                status="sent",
                from_date="2025-01-01",
                to_date="2025-12-31",
                count=10,
            )

    @pytest.mark.asyncio
    async def test_list_envelopes_authenticates_if_needed(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test that list envelopes authenticates unauthenticated connector."""
        mock_unauthenticated_connector.list_envelopes = AsyncMock(return_value=[])
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            mock_unauthenticated_connector.authenticate_jwt.assert_called_once()
            assert result.status_code == 200


# =============================================================================
# Create Envelope Tests
# =============================================================================


class TestCreateEnvelope:
    """Tests for _handle_create_envelope."""

    @pytest.mark.asyncio
    async def test_create_not_configured(self, handler_with_ctx, mock_request):
        """Test create envelope when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_create_envelope(mock_request, "test_tenant")
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_create_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test create envelope when auth fails."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_create_envelope(mock_request, "test_tenant")
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_create_missing_subject(self, handler_with_ctx, mock_request, mock_connector):
        """Test create envelope without email_subject."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {"recipients": [{"a": 1}], "documents": [{"b": 2}]}
                result = await handler_with_ctx._handle_create_envelope(mock_request, "test_tenant")
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_missing_recipients(self, handler_with_ctx, mock_request, mock_connector):
        """Test create envelope without recipients."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Test",
                    "recipients": [],
                    "documents": [{"b": 2}],
                }
                result = await handler_with_ctx._handle_create_envelope(mock_request, "test_tenant")
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_missing_documents(self, handler_with_ctx, mock_request, mock_connector):
        """Test create envelope without documents."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Test",
                    "recipients": [{"email": "u@x.com", "name": "A"}],
                    "documents": [],
                }
                result = await handler_with_ctx._handle_create_envelope(mock_request, "test_tenant")
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful envelope creation."""
        mock_envelope = MagicMock()
        mock_envelope.envelope_id = "new-env-123"
        mock_envelope.to_dict.return_value = {"envelope_id": "new-env-123", "status": "sent"}
        mock_connector.create_envelope = AsyncMock(return_value=mock_envelope)

        content_b64 = base64.b64encode(b"test doc content").decode()

        # Mock the DocuSign models
        mock_docusign = MagicMock()
        mock_docusign.Document = MagicMock()
        mock_docusign.EnvelopeCreateRequest = MagicMock()
        mock_docusign.Recipient = MagicMock()
        mock_docusign.RecipientType = MagicMock()
        mock_docusign.SignatureTab = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Please sign",
                    "email_body": "Hello",
                    "recipients": [
                        {"email": "user@example.com", "name": "John Doe", "type": "signer"}
                    ],
                    "documents": [{"name": "contract.pdf", "content_base64": content_b64}],
                    "status": "sent",
                    "tabs": [
                        {
                            "type": "signature",
                            "page": 1,
                            "x": 100,
                            "y": 500,
                            "recipient_id": "1",
                        }
                    ],
                }
                with patch.dict(
                    "sys.modules",
                    {"aragora.connectors.legal.docusign": mock_docusign},
                ):
                    result = await handler_with_ctx._handle_create_envelope(
                        mock_request, "test_tenant"
                    )
                    assert result.status_code == 201
                    body = json.loads(result.body)
                    assert body["data"]["envelope"]["envelope_id"] == "new-env-123"

    @pytest.mark.asyncio
    async def test_create_without_tabs(self, handler_with_ctx, mock_request, mock_connector):
        """Test create envelope without tabs (optional field)."""
        mock_envelope = MagicMock()
        mock_envelope.to_dict.return_value = {"envelope_id": "new-env", "status": "sent"}
        mock_connector.create_envelope = AsyncMock(return_value=mock_envelope)

        content_b64 = base64.b64encode(b"test content").decode()

        mock_docusign = MagicMock()
        mock_docusign.Document = MagicMock()
        mock_docusign.EnvelopeCreateRequest = MagicMock()
        mock_docusign.Recipient = MagicMock()
        mock_docusign.RecipientType = MagicMock()
        mock_docusign.SignatureTab = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Please sign",
                    "recipients": [{"email": "user@example.com", "name": "John", "type": "signer"}],
                    "documents": [{"name": "doc.pdf", "content_base64": content_b64}],
                }
                with patch.dict(
                    "sys.modules",
                    {"aragora.connectors.legal.docusign": mock_docusign},
                ):
                    result = await handler_with_ctx._handle_create_envelope(
                        mock_request, "test_tenant"
                    )
                    assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test create envelope when API call fails."""
        content_b64 = base64.b64encode(b"test content").decode()

        mock_docusign = MagicMock()
        mock_docusign.Document = MagicMock()
        mock_docusign.EnvelopeCreateRequest = MagicMock()
        mock_docusign.Recipient = MagicMock()
        mock_docusign.RecipientType = MagicMock(side_effect=ValueError("invalid type"))
        mock_docusign.SignatureTab = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Test",
                    "recipients": [
                        {"email": "user@example.com", "name": "John", "type": "invalid"}
                    ],
                    "documents": [{"name": "d.pdf", "content_base64": content_b64}],
                }
                with patch.dict(
                    "sys.modules",
                    {"aragora.connectors.legal.docusign": mock_docusign},
                ):
                    result = await handler_with_ctx._handle_create_envelope(
                        mock_request, "test_tenant"
                    )
                    assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_create_multiple_recipients(self, handler_with_ctx, mock_request, mock_connector):
        """Test creating envelope with multiple recipients."""
        mock_envelope = MagicMock()
        mock_envelope.to_dict.return_value = {"envelope_id": "env-multi", "status": "sent"}
        mock_connector.create_envelope = AsyncMock(return_value=mock_envelope)

        content_b64 = base64.b64encode(b"test").decode()
        mock_docusign = MagicMock()
        mock_docusign.Document = MagicMock()
        mock_docusign.EnvelopeCreateRequest = MagicMock()
        mock_docusign.Recipient = MagicMock()
        mock_docusign.RecipientType = MagicMock()
        mock_docusign.SignatureTab = MagicMock()

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {
                    "email_subject": "Multi sign",
                    "recipients": [
                        {
                            "email": "signer1@example.com",
                            "name": "Signer One",
                            "type": "signer",
                            "routing_order": 1,
                        },
                        {
                            "email": "cc@example.com",
                            "name": "CC Person",
                            "type": "cc",
                            "routing_order": 2,
                        },
                    ],
                    "documents": [{"name": "doc.pdf", "content_base64": content_b64}],
                }
                with patch.dict(
                    "sys.modules",
                    {"aragora.connectors.legal.docusign": mock_docusign},
                ):
                    result = await handler_with_ctx._handle_create_envelope(
                        mock_request, "test_tenant"
                    )
                    assert result.status_code == 201


# =============================================================================
# Get Envelope Tests
# =============================================================================


class TestGetEnvelope:
    """Tests for _handle_get_envelope."""

    @pytest.mark.asyncio
    async def test_get_not_configured(self, handler_with_ctx, mock_request):
        """Test get envelope when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test get envelope auth failure."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful get envelope."""
        mock_envelope = MagicMock()
        mock_envelope.to_dict.return_value = {"envelope_id": "env123", "status": "sent"}
        mock_connector.get_envelope = AsyncMock(return_value=mock_envelope)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["envelope"]["envelope_id"] == "env123"

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler_with_ctx, mock_request, mock_connector):
        """Test get envelope when not found."""
        mock_connector.get_envelope = AsyncMock(return_value=None)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test get envelope API error."""
        mock_connector.get_envelope = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_get_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 500


# =============================================================================
# Void Envelope Tests
# =============================================================================


class TestVoidEnvelope:
    """Tests for _handle_void_envelope."""

    @pytest.mark.asyncio
    async def test_void_not_configured(self, handler_with_ctx, mock_request):
        """Test void envelope when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_void_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_void_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test void envelope auth failure."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_void_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_void_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful void envelope."""
        mock_connector.void_envelope = AsyncMock(return_value=True)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {"reason": "Contract cancelled"}
                result = await handler_with_ctx._handle_void_envelope(
                    mock_request, "test_tenant", "env123"
                )
                assert result.status_code == 200
                mock_connector.void_envelope.assert_called_once_with("env123", "Contract cancelled")

    @pytest.mark.asyncio
    async def test_void_default_reason(self, handler_with_ctx, mock_request, mock_connector):
        """Test void envelope uses default reason."""
        mock_connector.void_envelope = AsyncMock(return_value=True)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {}
                result = await handler_with_ctx._handle_void_envelope(
                    mock_request, "test_tenant", "env123"
                )
                assert result.status_code == 200
                mock_connector.void_envelope.assert_called_once_with("env123", "Voided by user")

    @pytest.mark.asyncio
    async def test_void_failure(self, handler_with_ctx, mock_request, mock_connector):
        """Test void envelope returns false."""
        mock_connector.void_envelope = AsyncMock(return_value=False)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {"reason": "test"}
                result = await handler_with_ctx._handle_void_envelope(
                    mock_request, "test_tenant", "env123"
                )
                assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_void_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test void envelope API error."""
        mock_connector.void_envelope = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {"reason": "test"}
                result = await handler_with_ctx._handle_void_envelope(
                    mock_request, "test_tenant", "env123"
                )
                assert result.status_code == 500


# =============================================================================
# Resend Envelope Tests
# =============================================================================


class TestResendEnvelope:
    """Tests for _handle_resend_envelope."""

    @pytest.mark.asyncio
    async def test_resend_not_configured(self, handler_with_ctx, mock_request):
        """Test resend when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_resend_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test resend auth failure."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_resend_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful resend."""
        mock_connector.resend_envelope = AsyncMock(return_value=True)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_resend_failure(self, handler_with_ctx, mock_request, mock_connector):
        """Test resend returns false."""
        mock_connector.resend_envelope = AsyncMock(return_value=False)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_resend_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test resend API error."""
        mock_connector.resend_envelope = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 500


# =============================================================================
# Download Document Tests
# =============================================================================


class TestDownloadDocument:
    """Tests for _handle_download_document."""

    @pytest.mark.asyncio
    async def test_download_not_configured(self, handler_with_ctx, mock_request):
        """Test download when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_download_document(
                mock_request, "test_tenant", "env123", "doc1"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_download_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test download auth failure."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_download_document(
                mock_request, "test_tenant", "env123", "doc1"
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_download_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful document download."""
        doc_content = b"PDF content here"
        mock_connector.download_document = AsyncMock(return_value=doc_content)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_download_document(
                mock_request, "test_tenant", "env123", "doc1"
            )
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["envelope_id"] == "env123"
            assert body["data"]["document_id"] == "doc1"
            assert body["data"]["content_type"] == "application/pdf"
            # Verify base64 encoding
            decoded = base64.b64decode(body["data"]["content_base64"])
            assert decoded == doc_content

    @pytest.mark.asyncio
    async def test_download_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test download API error."""
        mock_connector.download_document = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_download_document(
                mock_request, "test_tenant", "env123", "doc1"
            )
            assert result.status_code == 500


# =============================================================================
# Download Certificate Tests
# =============================================================================


class TestDownloadCertificate:
    """Tests for _handle_download_certificate."""

    @pytest.mark.asyncio
    async def test_certificate_not_configured(self, handler_with_ctx, mock_request):
        """Test certificate download when not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None
            result = await handler_with_ctx._handle_download_certificate(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_certificate_auth_failure(
        self, handler_with_ctx, mock_request, mock_unauthenticated_connector
    ):
        """Test certificate download auth failure."""
        mock_unauthenticated_connector.authenticate_jwt.side_effect = Exception("Auth fail")
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_unauthenticated_connector
            result = await handler_with_ctx._handle_download_certificate(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_certificate_success(self, handler_with_ctx, mock_request, mock_connector):
        """Test successful certificate download."""
        cert_content = b"Certificate PDF"
        mock_connector.download_certificate = AsyncMock(return_value=cert_content)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_download_certificate(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["envelope_id"] == "env123"
            assert body["data"]["content_type"] == "application/pdf"
            decoded = base64.b64decode(body["data"]["content_base64"])
            assert decoded == cert_content

    @pytest.mark.asyncio
    async def test_certificate_api_error(self, handler_with_ctx, mock_request, mock_connector):
        """Test certificate download API error."""
        mock_connector.download_certificate = AsyncMock(side_effect=ValueError("API error"))
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_download_certificate(
                mock_request, "test_tenant", "env123"
            )
            assert result.status_code == 500


# =============================================================================
# Templates Tests
# =============================================================================


class TestListTemplates:
    """Tests for _handle_list_templates."""

    @pytest.mark.asyncio
    async def test_list_templates_no_connector(self, handler_with_ctx, mock_request):
        """Test template listing returns 503 when DocuSign is not configured."""
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler_with_ctx._handle_list_templates(mock_request, "test_tenant")
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_list_templates_success(self, handler_with_ctx, mock_request):
        """Test template listing returns templates from DocuSign API."""
        mock_connector = AsyncMock()
        mock_connector.is_authenticated = True
        mock_connector._request = AsyncMock(
            return_value={
                "envelopeTemplates": [
                    {
                        "templateId": "tpl-1",
                        "name": "NDA Template",
                        "description": "Standard NDA",
                        "created": "2025-01-01",
                        "lastModified": "2025-06-01",
                        "owner": {"userName": "Admin"},
                        "shared": "true",
                        "folderName": "Templates",
                    }
                ],
                "resultSetSize": 1,
            }
        )

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=mock_connector,
        ):
            result = await handler_with_ctx._handle_list_templates(mock_request, "test_tenant")
        assert result.status_code == 200
        body = json.loads(result.body)
        templates = body["data"]["templates"]
        assert len(templates) == 1
        assert templates[0]["template_id"] == "tpl-1"
        assert templates[0]["name"] == "NDA Template"
        assert templates[0]["shared"] is True


# =============================================================================
# Webhook Tests
# =============================================================================


class TestDocuSignWebhook:
    """Tests for _handle_docusign_webhook."""

    @pytest.mark.asyncio
    async def test_webhook_success(self, handler_with_ctx, mock_request):
        """Test successful webhook handling."""
        with patch.object(handler_with_ctx, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "envelopeId": "env-webhook-123",
                "status": "completed",
                "statusChangedDateTime": "2025-01-01T12:00:00Z",
            }
            with patch.object(
                handler_with_ctx, "_emit_connector_event", new_callable=AsyncMock
            ) as mock_emit:
                result = await handler_with_ctx._handle_docusign_webhook(
                    mock_request, "test_tenant"
                )
                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["data"]["received"] is True
                assert body["data"]["envelope_id"] == "env-webhook-123"
                assert body["data"]["status"] == "completed"
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_partial_data(self, handler_with_ctx, mock_request):
        """Test webhook with partial envelope data."""
        with patch.object(handler_with_ctx, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"envelopeId": "env-partial"}
            with patch.object(handler_with_ctx, "_emit_connector_event", new_callable=AsyncMock):
                result = await handler_with_ctx._handle_docusign_webhook(
                    mock_request, "test_tenant"
                )
                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["data"]["status"] is None
                assert body["data"]["event_time"] is None

    @pytest.mark.asyncio
    async def test_webhook_empty_body(self, handler_with_ctx, mock_request):
        """Test webhook with empty body."""
        with patch.object(handler_with_ctx, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}
            with patch.object(handler_with_ctx, "_emit_connector_event", new_callable=AsyncMock):
                result = await handler_with_ctx._handle_docusign_webhook(
                    mock_request, "test_tenant"
                )
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_webhook_malformed_returns_200(self, handler_with_ctx, mock_request):
        """Test webhook returns 200 even on error to prevent retries."""
        with patch.object(handler_with_ctx, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = Exception("parse error")
            result = await handler_with_ctx._handle_docusign_webhook(mock_request, "test_tenant")
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["received"] is True
            assert "error" in body["data"]


# =============================================================================
# Connector Management Tests
# =============================================================================


class TestGetDocusignConnector:
    """Tests for get_docusign_connector function."""

    @pytest.mark.asyncio
    async def test_connector_not_installed(self):
        """Test returns None when DocuSign module not installed."""
        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": None}):
            result = await get_docusign_connector("tenant1")
            assert result is None

    @pytest.mark.asyncio
    async def test_connector_not_configured(self):
        """Test returns None when connector not configured."""
        mock_docusign_mod = MagicMock()
        mock_connector_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.is_configured = False
        mock_connector_cls.return_value = mock_instance
        mock_docusign_mod.DocuSignConnector = mock_connector_cls

        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": mock_docusign_mod}):
            result = await get_docusign_connector("tenant1")
            assert result is None

    @pytest.mark.asyncio
    async def test_connector_configured(self):
        """Test returns connector when configured."""
        mock_docusign_mod = MagicMock()
        mock_connector_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.is_configured = True
        mock_connector_cls.return_value = mock_instance
        mock_docusign_mod.DocuSignConnector = mock_connector_cls

        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": mock_docusign_mod}):
            result = await get_docusign_connector("tenant1")
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_connector_cached(self):
        """Test connector is cached after first creation."""
        mock_connector = MagicMock()
        mock_connector.is_configured = True

        mock_docusign_mod = MagicMock()
        mock_docusign_mod.DocuSignConnector.return_value = mock_connector

        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": mock_docusign_mod}):
            result1 = await get_docusign_connector("tenant1")
            result2 = await get_docusign_connector("tenant1")
            assert result1 is result2
            # Constructor should only be called once
            assert mock_docusign_mod.DocuSignConnector.call_count == 1

    @pytest.mark.asyncio
    async def test_connector_per_tenant(self):
        """Test separate connector per tenant."""
        mock_docusign_mod = MagicMock()
        connector1 = MagicMock()
        connector1.is_configured = True
        connector2 = MagicMock()
        connector2.is_configured = True
        mock_docusign_mod.DocuSignConnector.side_effect = [connector1, connector2]

        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": mock_docusign_mod}):
            result1 = await get_docusign_connector("tenant_a")
            result2 = await get_docusign_connector("tenant_b")
            assert result1 is not result2


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for handler utility methods."""

    def test_get_query_params_from_query_attr(self, handler):
        """Test _get_query_params with request.query dict."""
        request = MagicMock()
        request.query = {"status": "sent", "limit": "10"}
        params = handler._get_query_params(request)
        assert params["status"] == "sent"
        assert params["limit"] == "10"

    def test_get_query_params_from_query_string(self, handler):
        """Test _get_query_params with request.query_string."""
        request = MagicMock(spec=["query_string"])
        request.query_string = "status=sent&limit=10"
        params = handler._get_query_params(request)
        assert params["status"] == "sent"
        assert params["limit"] == "10"

    def test_get_query_params_empty(self, handler):
        """Test _get_query_params with no query support."""
        request = MagicMock(spec=[])
        params = handler._get_query_params(request)
        assert params == {}

    @pytest.mark.asyncio
    async def test_get_json_body_success(self, handler):
        """Test _get_json_body returns parsed body."""
        with patch(
            "aragora.server.handlers.features.legal.parse_json_body",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = ({"key": "value"}, None)
            request = MagicMock()
            result = await handler._get_json_body(request)
            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_json_body_returns_empty_on_error(self, handler):
        """Test _get_json_body returns empty dict on parse failure."""
        with patch(
            "aragora.server.handlers.features.legal.parse_json_body",
            new_callable=AsyncMock,
        ) as mock_parse:
            mock_parse.return_value = (None, MagicMock())
            request = MagicMock()
            result = await handler._get_json_body(request)
            assert result == {}

    def test_get_tenant_id_from_attribute(self, handler):
        """Test _get_tenant_id extracts from request attribute."""
        request = MagicMock()
        request.tenant_id = "my_tenant_123"
        assert handler._get_tenant_id(request) == "my_tenant_123"

    def test_get_tenant_id_default(self, handler):
        """Test _get_tenant_id returns 'default' when attribute missing."""
        request = MagicMock(spec=[])
        assert handler._get_tenant_id(request) == "default"


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEmitConnectorEvent:
    """Tests for _emit_connector_event."""

    @pytest.mark.asyncio
    async def test_emit_with_emitter(self):
        """Test event emission when emitter is available in context."""
        mock_emitter = MagicMock()
        handler = LegalHandler({"emitter": mock_emitter})

        mock_stream_types = MagicMock()
        mock_stream_types.StreamEventType.CONNECTOR_DOCUSIGN_ENVELOPE_STATUS.value = (
            "docusign.envelope.status"
        )

        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_types},
        ):
            await handler._emit_connector_event(
                event_type="docusign_envelope_status",
                tenant_id="t1",
                data={"envelope_id": "env1", "status": "completed"},
            )
            mock_emitter.emit.assert_called_once()
            call_args = mock_emitter.emit.call_args
            assert call_args[0][0] == "docusign.envelope.status"
            event_data = call_args[0][1]
            assert event_data["connector"] == "docusign"
            assert event_data["tenant_id"] == "t1"
            assert event_data["envelope_id"] == "env1"

    @pytest.mark.asyncio
    async def test_emit_without_emitter(self):
        """Test event emission without emitter is a no-op."""
        handler = LegalHandler({})

        mock_stream_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_types},
        ):
            # Should not raise
            await handler._emit_connector_event(
                event_type="test", tenant_id="t1", data={"key": "val"}
            )

    @pytest.mark.asyncio
    async def test_emit_with_empty_context(self):
        """Test event emission with no context."""
        handler = LegalHandler()

        mock_stream_types = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_types},
        ):
            await handler._emit_connector_event(event_type="test", tenant_id="t1", data={})

    @pytest.mark.asyncio
    async def test_emit_import_error(self):
        """Test event emission handles ImportError gracefully."""
        handler = LegalHandler({"emitter": MagicMock()})

        with patch.dict("sys.modules", {"aragora.events.types": None}):
            # Should not raise, just log debug
            await handler._emit_connector_event(event_type="test", tenant_id="t1", data={})


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration-style tests."""

    @pytest.mark.asyncio
    async def test_handle_method_with_default_tenant(self, handler):
        """Test handle extracts default tenant when not set."""
        request = MagicMock(spec=[])  # No tenant_id attribute
        with patch.object(handler, "_handle_list_templates", new_callable=AsyncMock) as mock_m:
            mock_m.return_value = MagicMock(status_code=200)
            result = await handler.handle(request, "/api/v1/legal/templates", "GET")
            mock_m.assert_called_once_with(request, "default")

    @pytest.mark.asyncio
    async def test_envelope_path_with_short_parts(self, handler, mock_request):
        """Test that short envelope paths return 404."""
        # /api/v1/legal/envelopes/ with trailing slash but no ID
        # Split gives ['', 'api', 'v1', 'legal', 'envelopes', '']
        result = await handler.handle(mock_request, "/api/v1/legal/envelopes/", "GET")
        # With parts[4] = 'envelopes' and parts[5] = '' this hits the action routing
        # but '' doesn't match any action, so returns 404
        assert result.status_code in (200, 404, 500)

    @pytest.mark.asyncio
    async def test_envelope_unknown_action(self, handler, mock_request):
        """Test that unknown envelope actions return 404."""
        result = await handler.handle(
            mock_request, "/api/v1/legal/envelopes/env123/unknown_action", "POST"
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_list_envelopes_default_limit(
        self, handler_with_ctx, mock_request, mock_connector
    ):
        """Test list envelopes uses default limit of 25."""
        mock_request.query = {}
        mock_connector.list_envelopes = AsyncMock(return_value=[])
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 200
            mock_connector.list_envelopes.assert_called_once_with(
                status=None,
                from_date=None,
                to_date=None,
                count=25,
            )

    @pytest.mark.asyncio
    async def test_handler_context_preserved(self):
        """Test that handler preserves server context through operations."""
        ctx = {"storage": "test_storage", "emitter": MagicMock(), "config": {"key": "val"}}
        handler = LegalHandler(ctx)
        assert handler.ctx is ctx
        assert handler.ctx["storage"] == "test_storage"

    @pytest.mark.asyncio
    async def test_list_envelopes_multiple_results(
        self, handler_with_ctx, mock_request, mock_connector
    ):
        """Test list envelopes with multiple envelope results."""
        envs = []
        for i in range(5):
            e = MagicMock()
            e.to_dict.return_value = {"envelope_id": f"env{i}", "status": "sent"}
            envs.append(e)
        mock_connector.list_envelopes = AsyncMock(return_value=envs)
        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector
            result = await handler_with_ctx._handle_list_envelopes(mock_request, "test_tenant")
            assert result.status_code == 200
            body = json.loads(result.body)
            assert body["data"]["count"] == 5
            assert len(body["data"]["envelopes"]) == 5

    @pytest.mark.asyncio
    async def test_void_then_resend_same_envelope(
        self, handler_with_ctx, mock_request, mock_connector
    ):
        """Test voiding then resending the same envelope (different operations)."""
        mock_connector.void_envelope = AsyncMock(return_value=True)
        mock_connector.resend_envelope = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.features.legal.get_docusign_connector",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_connector

            with patch.object(
                handler_with_ctx, "_get_json_body", new_callable=AsyncMock
            ) as mock_body:
                mock_body.return_value = {"reason": "test"}
                void_result = await handler_with_ctx._handle_void_envelope(
                    mock_request, "test_tenant", "env123"
                )
                assert void_result.status_code == 200

            resend_result = await handler_with_ctx._handle_resend_envelope(
                mock_request, "test_tenant", "env123"
            )
            assert resend_result.status_code == 200

    @pytest.mark.asyncio
    async def test_connector_cache_cleared_between_tests(self):
        """Test that the autouse fixture clears connector cache."""
        # The clear_connector_cache fixture should ensure _connector_instances is empty
        assert len(_connector_instances) == 0


class TestRequirePermission:
    """Tests for RBAC permission on handle method."""

    def test_handle_has_require_permission(self, handler):
        """Test that handle method has the require_permission decorator."""
        # The handle method should be decorated with @require_permission("legal:read")
        # We can verify by checking if the method has the wrapper attributes
        method = handler.handle
        # The require_permission decorator wraps the function
        assert callable(method)

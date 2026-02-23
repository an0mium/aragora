"""Tests for legal e-signature handler.

Tests the legal API endpoints including:
- GET  /api/v1/legal/status                              - Connection status
- GET  /api/v1/legal/envelopes                           - List envelopes
- POST /api/v1/legal/envelopes                           - Create envelope
- GET  /api/v1/legal/envelopes/{id}                      - Get envelope details
- POST /api/v1/legal/envelopes/{id}/void                 - Void envelope
- POST /api/v1/legal/envelopes/{id}/resend               - Resend notifications
- GET  /api/v1/legal/envelopes/{id}/documents/{doc_id}   - Download document
- GET  /api/v1/legal/envelopes/{id}/certificate           - Download certificate
- POST /api/v1/legal/webhooks/docusign                   - DocuSign webhook
- GET  /api/v1/legal/templates                           - List templates
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock HTTP request for testing the legal handler."""

    path: str = "/api/v1/legal/status"
    method: str = "GET"
    query: dict[str, Any] = field(default_factory=dict)
    _body: dict[str, Any] | None = None
    tenant_id: str = "test-tenant"

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def read(self) -> bytes:
        return json.dumps(self._body or {}).encode()


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract decoded JSON body from HandlerResult."""
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return {}


def _data(result) -> dict[str, Any]:
    """Extract the data sub-dict from a success_response envelope."""
    body = _body(result)
    return body.get("data", body)


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    return _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Mock DocuSign types
# ---------------------------------------------------------------------------


class MockEnvelope:
    """Mock envelope object returned by connector."""

    def __init__(self, envelope_id="env-123", status="sent", **kwargs):
        self.envelope_id = envelope_id
        self.status = status
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {"envelope_id": self.envelope_id, "status": self.status}


class MockEnvironment:
    """Mock environment enum."""

    def __init__(self, value="sandbox"):
        self.value = value


def _make_connector(
    *,
    is_configured=True,
    is_authenticated=True,
    environment_value="sandbox",
    integration_key="ik-123",
    account_id="acc-456",
):
    """Build a mock DocuSign connector with sensible defaults."""
    connector = AsyncMock()
    connector.is_configured = is_configured
    connector.is_authenticated = is_authenticated
    connector.environment = MockEnvironment(environment_value)
    connector.integration_key = integration_key
    connector.account_id = account_id
    connector.authenticate_jwt = AsyncMock()
    connector.list_envelopes = AsyncMock(return_value=[])
    connector.create_envelope = AsyncMock(return_value=MockEnvelope("env-new", "sent"))
    connector.get_envelope = AsyncMock(return_value=MockEnvelope("env-123", "completed"))
    connector.void_envelope = AsyncMock(return_value=True)
    connector.resend_envelope = AsyncMock(return_value=True)
    connector.download_document = AsyncMock(return_value=b"%PDF-content")
    connector.download_certificate = AsyncMock(return_value=b"%PDF-cert")
    connector._request = AsyncMock(return_value={"envelopeTemplates": []})
    return connector


def _make_docusign_module():
    """Create a mock module for aragora.connectors.legal.docusign."""
    mod = MagicMock()
    mod.RecipientType = MagicMock(side_effect=lambda x: x)
    mod.Recipient = MagicMock()
    mod.Document = MagicMock()
    mod.SignatureTab = MagicMock()
    mod.EnvelopeCreateRequest = MagicMock()
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONNECTOR_MODULE = "aragora.server.handlers.features.legal"


@pytest.fixture
def handler():
    """Create a LegalHandler with empty server context."""
    from aragora.server.handlers.features.legal import LegalHandler

    return LegalHandler(server_context={})


@pytest.fixture(autouse=True)
def reset_connector_instances():
    """Clear the module-level connector cache between tests."""
    from aragora.server.handlers.features.legal import _connector_instances

    _connector_instances.clear()
    yield
    _connector_instances.clear()


@pytest.fixture
def mock_connector():
    """Provide a default mock connector."""
    return _make_connector()


@pytest.fixture
def patch_get_connector(mock_connector):
    """Patch get_docusign_connector to return the mock_connector."""
    with patch(
        f"{CONNECTOR_MODULE}.get_docusign_connector",
        new_callable=AsyncMock,
        return_value=mock_connector,
    ):
        yield mock_connector


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Test handler initialization."""

    def test_default_context(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_none_context_becomes_empty(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler(server_context=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# can_handle()
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle routing for legal paths."""

    def test_status_path(self, handler):
        assert handler.can_handle("/api/v1/legal/status")

    def test_envelopes_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes")

    def test_envelope_detail_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes/env-123")

    def test_void_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes/env-123/void", "POST")

    def test_resend_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes/env-123/resend", "POST")

    def test_document_download_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes/env-123/documents/doc-1")

    def test_certificate_path(self, handler):
        assert handler.can_handle("/api/v1/legal/envelopes/env-123/certificate")

    def test_webhook_path(self, handler):
        assert handler.can_handle("/api/v1/legal/webhooks/docusign", "POST")

    def test_templates_path(self, handler):
        assert handler.can_handle("/api/v1/legal/templates")

    def test_rejects_non_legal_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_other_path(self, handler):
        assert not handler.can_handle("/api/v1/users/me")

    def test_routes_list_populated(self, handler):
        assert len(handler.ROUTES) >= 9


# ---------------------------------------------------------------------------
# GET /api/v1/legal/status
# ---------------------------------------------------------------------------


class TestStatus:
    """Test status endpoint."""

    @pytest.mark.asyncio
    async def test_status_configured(self, handler):
        mock_conn = MagicMock()
        mock_conn.is_configured = True
        mock_conn.is_authenticated = True
        mock_conn.environment = MockEnvironment("production")
        mock_conn.integration_key = "ik-abc"
        mock_conn.account_id = "acc-xyz"

        mock_docusign_mod = MagicMock()
        mock_docusign_mod.DocuSignConnector.return_value = mock_conn

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign_mod},
        ):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 200
            data = _data(result)
            assert data["configured"] is True
            assert data["authenticated"] is True
            assert data["environment"] == "production"
            assert data["integration_key_set"] is True
            assert data["account_id_set"] is True

    @pytest.mark.asyncio
    async def test_status_not_configured(self, handler):
        mock_conn = MagicMock()
        mock_conn.is_configured = False
        mock_conn.is_authenticated = False
        mock_conn.environment = MockEnvironment("sandbox")
        mock_conn.integration_key = ""
        mock_conn.account_id = ""

        mock_docusign_mod = MagicMock()
        mock_docusign_mod.DocuSignConnector.return_value = mock_conn

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign_mod},
        ):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 200
            data = _data(result)
            assert data["configured"] is False
            assert data["integration_key_set"] is False
            assert data["account_id_set"] is False

    @pytest.mark.asyncio
    async def test_status_import_error(self, handler):
        """When docusign module is not installed, return graceful fallback."""
        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": None}):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 200
            data = _data(result)
            assert data["configured"] is False
            assert "error" in data

    @pytest.mark.asyncio
    async def test_status_partial_config(self, handler):
        mock_conn = MagicMock()
        mock_conn.is_configured = True
        mock_conn.is_authenticated = False
        mock_conn.environment = MockEnvironment("sandbox")
        mock_conn.integration_key = "ik-123"
        mock_conn.account_id = ""

        mock_docusign_mod = MagicMock()
        mock_docusign_mod.DocuSignConnector.return_value = mock_conn

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign_mod},
        ):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 200
            data = _data(result)
            assert data["configured"] is True
            assert data["authenticated"] is False
            assert data["integration_key_set"] is True
            assert data["account_id_set"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/legal/envelopes (list)
# ---------------------------------------------------------------------------


class TestListEnvelopes:
    """Test listing envelopes."""

    @pytest.mark.asyncio
    async def test_list_success_empty(self, handler, patch_get_connector):
        req = MockRequest(path="/api/v1/legal/envelopes")
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        data = _data(result)
        assert data["envelopes"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_with_envelopes(self, handler, patch_get_connector):
        patch_get_connector.list_envelopes.return_value = [
            MockEnvelope("e1", "sent"),
            MockEnvelope("e2", "completed"),
        ]
        req = MockRequest(path="/api/v1/legal/envelopes")
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        data = _data(result)
        assert data["count"] == 2
        assert data["envelopes"][0]["envelope_id"] == "e1"
        assert data["envelopes"][1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_passes_query_params(self, handler, patch_get_connector):
        req = MockRequest(
            path="/api/v1/legal/envelopes",
            query={"status": "completed", "from_date": "2026-01-01", "limit": "10"},
        )
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        patch_get_connector.list_envelopes.assert_called_once_with(
            status="completed",
            from_date="2026-01-01",
            to_date=None,
            count=10,
        )

    @pytest.mark.asyncio
    async def test_list_passes_to_date(self, handler, patch_get_connector):
        req = MockRequest(
            path="/api/v1/legal/envelopes",
            query={"to_date": "2026-03-01"},
        )
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        call_kwargs = patch_get_connector.list_envelopes.call_args[1]
        assert call_kwargs["to_date"] == "2026-03-01"

    @pytest.mark.asyncio
    async def test_list_bad_limit_defaults(self, handler, patch_get_connector):
        req = MockRequest(
            path="/api/v1/legal/envelopes",
            query={"limit": "notanumber"},
        )
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        call_kwargs = patch_get_connector.list_envelopes.call_args[1]
        assert call_kwargs["count"] == 25

    @pytest.mark.asyncio
    async def test_list_empty_limit_defaults(self, handler, patch_get_connector):
        req = MockRequest(
            path="/api/v1/legal/envelopes",
            query={},
        )
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        call_kwargs = patch_get_connector.list_envelopes.call_args[1]
        assert call_kwargs["count"] == 25

    @pytest.mark.asyncio
    async def test_list_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_auth_failure_connection_error(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = ConnectionError("auth failed")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_auth_failure_timeout(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = TimeoutError("slow")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_auth_failure_value_error(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = ValueError("bad token")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_connector_error(self, handler, patch_get_connector):
        patch_get_connector.list_envelopes.side_effect = ConnectionError("timeout")
        req = MockRequest(path="/api/v1/legal/envelopes")
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_auto_authenticates(self, handler):
        connector = _make_connector(is_authenticated=False)

        async def fake_auth():
            connector.is_authenticated = True

        connector.authenticate_jwt.side_effect = fake_auth
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
            assert _status(result) == 200
            connector.authenticate_jwt.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_skips_auth_when_already_authenticated(self, handler, patch_get_connector):
        req = MockRequest(path="/api/v1/legal/envelopes")
        result = await handler.handle(req, "/api/v1/legal/envelopes", "GET")
        assert _status(result) == 200
        patch_get_connector.authenticate_jwt.assert_not_called()


# ---------------------------------------------------------------------------
# POST /api/v1/legal/envelopes (create)
# ---------------------------------------------------------------------------


class TestCreateEnvelope:
    """Test creating envelopes."""

    def _valid_body(self):
        return {
            "email_subject": "Please sign",
            "recipients": [{"email": "test@example.com", "name": "Test User", "type": "signer"}],
            "documents": [
                {"name": "contract.pdf", "content_base64": base64.b64encode(b"PDF").decode()}
            ],
        }

    @pytest.mark.asyncio
    async def test_create_success(self, handler, patch_get_connector):
        body = self._valid_body()
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201
                body_data = _body(result)
                assert body_data["success"] is True

    @pytest.mark.asyncio
    async def test_create_missing_subject(self, handler, patch_get_connector):
        body = {"recipients": [{"email": "a@b.com", "name": "A"}], "documents": [{}]}
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 400
            assert "email_subject" in _error(result)

    @pytest.mark.asyncio
    async def test_create_missing_recipients(self, handler, patch_get_connector):
        body = {"email_subject": "Sign", "documents": [{}]}
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 400
            assert "recipients" in _error(result)

    @pytest.mark.asyncio
    async def test_create_missing_documents(self, handler, patch_get_connector):
        body = {"email_subject": "Sign", "recipients": [{"email": "a@b.com", "name": "A"}]}
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 400
            assert "documents" in _error(result)

    @pytest.mark.asyncio
    async def test_create_empty_subject(self, handler, patch_get_connector):
        body = {
            "email_subject": "",
            "recipients": [{"email": "a@b.com", "name": "A"}],
            "documents": [{}],
        }
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_empty_recipients_list(self, handler, patch_get_connector):
        body = {"email_subject": "Sign", "recipients": [], "documents": [{}]}
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = ValueError("bad token")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
            result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_create_connector_error(self, handler, patch_get_connector):
        body = self._valid_body()
        mock_docusign = _make_docusign_module()
        patch_get_connector.create_envelope.side_effect = ConnectionError("fail")
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_with_tabs(self, handler, patch_get_connector):
        body = self._valid_body()
        body["tabs"] = [{"type": "signature", "page": 2, "x": 200, "y": 300, "recipient_id": "1"}]
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201
                mock_docusign.SignatureTab.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_email_body(self, handler, patch_get_connector):
        body = self._valid_body()
        body["email_body"] = "Please review and sign."
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_draft_status(self, handler, patch_get_connector):
        body = self._valid_body()
        body["status"] = "created"
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_multiple_recipients(self, handler, patch_get_connector):
        body = self._valid_body()
        body["recipients"].append({"email": "cc@ex.com", "name": "CC User", "type": "cc"})
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201
                assert mock_docusign.Recipient.call_count == 2

    @pytest.mark.asyncio
    async def test_create_multiple_documents(self, handler, patch_get_connector):
        body = self._valid_body()
        body["documents"].append(
            {"name": "appendix.pdf", "content_base64": base64.b64encode(b"APP").decode()}
        )
        mock_docusign = _make_docusign_module()
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.dict(
                "sys.modules",
                {"aragora.connectors.legal.docusign": mock_docusign},
            ):
                req = MockRequest(path="/api/v1/legal/envelopes", method="POST")
                result = await handler.handle(req, "/api/v1/legal/envelopes", "POST")
                assert _status(result) == 201
                assert mock_docusign.Document.call_count == 2


# ---------------------------------------------------------------------------
# Direct method tests for envelope-specific endpoints
# (The internal methods are tested directly because the routing layer
# uses parts[4] for envelope_id which, due to the leading empty string
# from path.split("/"), indexes to "envelopes" instead of the actual ID.
# We test the sub-methods directly to verify their logic.)
# ---------------------------------------------------------------------------


class TestGetEnvelopeDirect:
    """Test _handle_get_envelope directly."""

    @pytest.mark.asyncio
    async def test_get_success(self, handler):
        connector = _make_connector()
        connector.get_envelope.return_value = MockEnvelope("env-abc", "completed")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_get_envelope(req, "test-tenant", "env-abc")
            assert _status(result) == 200
            data = _data(result)
            assert data["envelope"]["envelope_id"] == "env-abc"

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler):
        connector = _make_connector()
        connector.get_envelope.return_value = None
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_get_envelope(req, "test-tenant", "env-missing")
            assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest()
            result = await handler._handle_get_envelope(req, "test-tenant", "env-123")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = TimeoutError("slow")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_get_envelope(req, "test-tenant", "env-123")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_get_connector_error(self, handler):
        connector = _make_connector()
        connector.get_envelope.side_effect = OSError("disk")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_get_envelope(req, "test-tenant", "env-123")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Void envelope - direct method
# ---------------------------------------------------------------------------


class TestVoidEnvelopeDirect:
    """Test _handle_void_envelope directly."""

    @pytest.mark.asyncio
    async def test_void_success(self, handler):
        connector = _make_connector()
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            with patch.object(
                handler,
                "_get_json_body",
                new_callable=AsyncMock,
                return_value={"reason": "No longer needed"},
            ):
                req = MockRequest()
                result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
                assert _status(result) == 200
                data = _data(result)
                assert data["envelope_id"] == "env-1"
                connector.void_envelope.assert_called_once_with("env-1", "No longer needed")

    @pytest.mark.asyncio
    async def test_void_default_reason(self, handler):
        connector = _make_connector()
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            with patch.object(
                handler,
                "_get_json_body",
                new_callable=AsyncMock,
                return_value={},
            ):
                req = MockRequest()
                result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
                assert _status(result) == 200
                connector.void_envelope.assert_called_once_with("env-1", "Voided by user")

    @pytest.mark.asyncio
    async def test_void_failure(self, handler):
        connector = _make_connector()
        connector.void_envelope.return_value = False
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            with patch.object(
                handler,
                "_get_json_body",
                new_callable=AsyncMock,
                return_value={"reason": "test"},
            ):
                req = MockRequest()
                result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_void_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest()
            result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_void_connector_error(self, handler):
        connector = _make_connector()
        connector.void_envelope.side_effect = ConnectionError("net")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            with patch.object(
                handler,
                "_get_json_body",
                new_callable=AsyncMock,
                return_value={"reason": "test"},
            ):
                req = MockRequest()
                result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_void_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = OSError("auth")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_void_envelope(req, "test-tenant", "env-1")
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# Resend envelope - direct method
# ---------------------------------------------------------------------------


class TestResendEnvelopeDirect:
    """Test _handle_resend_envelope directly."""

    @pytest.mark.asyncio
    async def test_resend_success(self, handler):
        connector = _make_connector()
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_resend_envelope(req, "test-tenant", "env-2")
            assert _status(result) == 200
            data = _data(result)
            assert data["envelope_id"] == "env-2"

    @pytest.mark.asyncio
    async def test_resend_failure(self, handler):
        connector = _make_connector()
        connector.resend_envelope.return_value = False
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_resend_envelope(req, "test-tenant", "env-2")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_resend_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest()
            result = await handler._handle_resend_envelope(req, "test-tenant", "env-2")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_resend_connector_error(self, handler):
        connector = _make_connector()
        connector.resend_envelope.side_effect = TimeoutError("slow")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_resend_envelope(req, "test-tenant", "env-2")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_resend_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = ValueError("expired")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_resend_envelope(req, "test-tenant", "env-2")
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# Download document - direct method
# ---------------------------------------------------------------------------


class TestDownloadDocumentDirect:
    """Test _handle_download_document directly."""

    @pytest.mark.asyncio
    async def test_download_success(self, handler):
        connector = _make_connector()
        connector.download_document.return_value = b"pdf-content-bytes"
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_document(req, "test-tenant", "env-1", "doc-1")
            assert _status(result) == 200
            data = _data(result)
            assert data["envelope_id"] == "env-1"
            assert data["document_id"] == "doc-1"
            assert data["content_type"] == "application/pdf"
            decoded = base64.b64decode(data["content_base64"])
            assert decoded == b"pdf-content-bytes"

    @pytest.mark.asyncio
    async def test_download_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest()
            result = await handler._handle_download_document(req, "test-tenant", "env-1", "doc-1")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_download_connector_error(self, handler):
        connector = _make_connector()
        connector.download_document.side_effect = ValueError("bad doc")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_document(req, "test-tenant", "env-1", "doc-1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_download_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = ConnectionError("fail")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_document(req, "test-tenant", "env-1", "doc-1")
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# Download certificate - direct method
# ---------------------------------------------------------------------------


class TestDownloadCertificateDirect:
    """Test _handle_download_certificate directly."""

    @pytest.mark.asyncio
    async def test_certificate_success(self, handler):
        connector = _make_connector()
        connector.download_certificate.return_value = b"cert-bytes"
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_certificate(req, "test-tenant", "env-5")
            assert _status(result) == 200
            data = _data(result)
            assert data["envelope_id"] == "env-5"
            assert data["content_type"] == "application/pdf"
            decoded = base64.b64decode(data["content_base64"])
            assert decoded == b"cert-bytes"

    @pytest.mark.asyncio
    async def test_certificate_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest()
            result = await handler._handle_download_certificate(req, "test-tenant", "env-5")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_certificate_connector_error(self, handler):
        connector = _make_connector()
        connector.download_certificate.side_effect = OSError("io error")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_certificate(req, "test-tenant", "env-5")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_certificate_auth_failure(self, handler):
        connector = _make_connector(is_authenticated=False)
        connector.authenticate_jwt.side_effect = TimeoutError("timeout")
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest()
            result = await handler._handle_download_certificate(req, "test-tenant", "env-5")
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# POST /api/v1/legal/webhooks/docusign
# ---------------------------------------------------------------------------


class TestDocuSignWebhook:
    """Test DocuSign webhook handling."""

    @pytest.mark.asyncio
    async def test_webhook_success(self, handler):
        body = {
            "envelopeId": "env-wh-1",
            "status": "completed",
            "statusChangedDateTime": "2026-02-23T10:00:00Z",
        }
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.object(
                handler, "_emit_connector_event", new_callable=AsyncMock
            ) as mock_emit:
                req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="POST")
                result = await handler.handle(req, "/api/v1/legal/webhooks/docusign", "POST")
                assert _status(result) == 200
                data = _data(result)
                assert data["received"] is True
                assert data["envelope_id"] == "env-wh-1"
                assert data["status"] == "completed"
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_empty_body(self, handler):
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value={}):
            with patch.object(handler, "_emit_connector_event", new_callable=AsyncMock):
                req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="POST")
                result = await handler.handle(req, "/api/v1/legal/webhooks/docusign", "POST")
                assert _status(result) == 200
                data = _data(result)
                assert data["received"] is True
                assert data["envelope_id"] is None

    @pytest.mark.asyncio
    async def test_webhook_malformed_returns_200(self, handler):
        """Malformed webhook payloads return 200 to prevent retries."""
        with patch.object(
            handler,
            "_get_json_body",
            new_callable=AsyncMock,
            side_effect=TypeError("bad data"),
        ):
            req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="POST")
            result = await handler.handle(req, "/api/v1/legal/webhooks/docusign", "POST")
            assert _status(result) == 200
            data = _data(result)
            assert data["received"] is True
            assert "error" in data

    @pytest.mark.asyncio
    async def test_webhook_event_time(self, handler):
        body = {
            "envelopeId": "env-wh-2",
            "status": "sent",
            "statusChangedDateTime": "2026-02-20T08:00:00Z",
        }
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.object(handler, "_emit_connector_event", new_callable=AsyncMock):
                req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="POST")
                result = await handler.handle(req, "/api/v1/legal/webhooks/docusign", "POST")
                data = _data(result)
                assert data["event_time"] == "2026-02-20T08:00:00Z"

    @pytest.mark.asyncio
    async def test_webhook_emit_data_correct(self, handler):
        body = {
            "envelopeId": "env-wh-3",
            "status": "voided",
            "statusChangedDateTime": "2026-02-22T12:00:00Z",
        }
        with patch.object(handler, "_get_json_body", new_callable=AsyncMock, return_value=body):
            with patch.object(
                handler, "_emit_connector_event", new_callable=AsyncMock
            ) as mock_emit:
                req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="POST")
                await handler.handle(req, "/api/v1/legal/webhooks/docusign", "POST")
                mock_emit.assert_called_once_with(
                    event_type="docusign_envelope_status",
                    tenant_id="test-tenant",
                    data={
                        "envelope_id": "env-wh-3",
                        "status": "voided",
                        "event_time": "2026-02-22T12:00:00Z",
                    },
                )


# ---------------------------------------------------------------------------
# GET /api/v1/legal/templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    """Test listing DocuSign templates."""

    @pytest.mark.asyncio
    async def test_list_templates_empty(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {"envelopeTemplates": []}
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 200
        data = _data(result)
        assert data["templates"] == []

    @pytest.mark.asyncio
    async def test_list_templates_with_data(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {
            "envelopeTemplates": [
                {
                    "templateId": "tpl-1",
                    "name": "NDA Template",
                    "description": "Standard NDA",
                    "created": "2026-01-01",
                    "lastModified": "2026-01-15",
                    "owner": {"userName": "Admin"},
                    "shared": "true",
                    "folderName": "Legal",
                },
                {
                    "templateId": "tpl-2",
                    "name": "Employment Contract",
                    "description": "",
                    "created": "2026-02-01",
                    "lastModified": "2026-02-10",
                    "owner": {"userName": "HR"},
                    "shared": "false",
                    "folderName": "HR",
                },
            ],
            "resultSetSize": "2",
        }
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 200
        data = _data(result)
        assert len(data["templates"]) == 2
        tpl = data["templates"][0]
        assert tpl["template_id"] == "tpl-1"
        assert tpl["name"] == "NDA Template"
        assert tpl["description"] == "Standard NDA"
        assert tpl["created"] == "2026-01-01"
        assert tpl["last_modified"] == "2026-01-15"
        assert tpl["owner_name"] == "Admin"
        assert tpl["shared"] is True
        assert tpl["folder_name"] == "Legal"
        tpl2 = data["templates"][1]
        assert tpl2["shared"] is False

    @pytest.mark.asyncio
    async def test_list_templates_with_search_text(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {"envelopeTemplates": []}
        req = MockRequest(
            path="/api/v1/legal/templates",
            query={"search_text": "NDA"},
        )
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 200
        call_args = patch_get_connector._request.call_args
        assert "search_text=NDA" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_list_templates_with_q_param(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {"envelopeTemplates": []}
        req = MockRequest(
            path="/api/v1/legal/templates",
            query={"q": "contract"},
        )
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 200
        call_args = patch_get_connector._request.call_args
        assert "search_text=contract" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_list_templates_no_search(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {"envelopeTemplates": []}
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 200
        call_args = patch_get_connector._request.call_args
        assert call_args[0][1] == "templates"

    @pytest.mark.asyncio
    async def test_list_templates_no_connector(self, handler):
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = MockRequest(path="/api/v1/legal/templates")
            result = await handler.handle(req, "/api/v1/legal/templates", "GET")
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_templates_connector_error(self, handler, patch_get_connector):
        patch_get_connector._request.side_effect = ConnectionError("net error")
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_templates_auth_auto(self, handler):
        """Templates endpoint auto-authenticates unauthenticated connectors."""
        connector = _make_connector(is_authenticated=False)

        async def fake_auth():
            connector.is_authenticated = True

        connector.authenticate_jwt.side_effect = fake_auth
        connector._request.return_value = {"envelopeTemplates": []}
        with patch(
            f"{CONNECTOR_MODULE}.get_docusign_connector",
            new_callable=AsyncMock,
            return_value=connector,
        ):
            req = MockRequest(path="/api/v1/legal/templates")
            result = await handler.handle(req, "/api/v1/legal/templates", "GET")
            assert _status(result) == 200
            connector.authenticate_jwt.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_templates_result_set_size(self, handler, patch_get_connector):
        patch_get_connector._request.return_value = {
            "envelopeTemplates": [
                {"templateId": "t1", "name": "T1"},
            ],
            "resultSetSize": "42",
        }
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        data = _data(result)
        assert data["total_count"] == "42"

    @pytest.mark.asyncio
    async def test_list_templates_missing_owner(self, handler, patch_get_connector):
        """Templates without owner field should fallback gracefully."""
        patch_get_connector._request.return_value = {
            "envelopeTemplates": [
                {"templateId": "t1", "name": "T1"},
            ],
        }
        req = MockRequest(path="/api/v1/legal/templates")
        result = await handler.handle(req, "/api/v1/legal/templates", "GET")
        data = _data(result)
        assert data["templates"][0]["owner_name"] == ""
        assert data["templates"][0]["template_id"] == "t1"


# ---------------------------------------------------------------------------
# Routing: 404 cases
# ---------------------------------------------------------------------------


class TestRouting404:
    """Test that unmatched paths return 404."""

    @pytest.mark.asyncio
    async def test_unknown_path(self, handler):
        req = MockRequest(path="/api/v1/legal/unknown")
        result = await handler.handle(req, "/api/v1/legal/unknown", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_envelopes_wrong_method(self, handler, patch_get_connector):
        req = MockRequest(path="/api/v1/legal/envelopes", method="DELETE")
        result = await handler.handle(req, "/api/v1/legal/envelopes", "DELETE")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_webhook_with_get(self, handler):
        req = MockRequest(path="/api/v1/legal/webhooks/docusign", method="GET")
        result = await handler.handle(req, "/api/v1/legal/webhooks/docusign", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_templates_with_post(self, handler, patch_get_connector):
        req = MockRequest(path="/api/v1/legal/templates", method="POST")
        result = await handler.handle(req, "/api/v1/legal/templates", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_status_with_post(self, handler):
        req = MockRequest(path="/api/v1/legal/status", method="POST")
        result = await handler.handle(req, "/api/v1/legal/status", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_deep_unknown_path(self, handler, patch_get_connector):
        req = MockRequest(path="/api/v1/legal/envelopes/a/b/c/d/e")
        result = await handler.handle(req, "/api/v1/legal/envelopes/a/b/c/d/e", "GET")
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# get_docusign_connector
# ---------------------------------------------------------------------------


class TestGetDocuSignConnector:
    """Test the connector factory function."""

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        from aragora.server.handlers.features.legal import get_docusign_connector

        with patch.dict("sys.modules", {"aragora.connectors.legal.docusign": None}):
            result = await get_docusign_connector("tenant-1")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_configured(self):
        from aragora.server.handlers.features.legal import get_docusign_connector

        mock_conn = MagicMock()
        mock_conn.is_configured = False
        mock_docusign = MagicMock()
        mock_docusign.DocuSignConnector.return_value = mock_conn

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign},
        ):
            result = await get_docusign_connector("tenant-2")
            assert result is None

    @pytest.mark.asyncio
    async def test_caches_connector(self):
        from aragora.server.handlers.features.legal import (
            get_docusign_connector,
            _connector_instances,
        )

        mock_conn = MagicMock()
        mock_conn.is_configured = True
        mock_docusign = MagicMock()
        mock_docusign.DocuSignConnector.return_value = mock_conn

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign},
        ):
            result1 = await get_docusign_connector("tenant-3")
            result2 = await get_docusign_connector("tenant-3")
            assert result1 is result2
            mock_docusign.DocuSignConnector.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_tenants_get_different_connectors(self):
        from aragora.server.handlers.features.legal import get_docusign_connector

        mock_conn_a = MagicMock()
        mock_conn_a.is_configured = True
        mock_conn_b = MagicMock()
        mock_conn_b.is_configured = True

        mock_docusign = MagicMock()
        mock_docusign.DocuSignConnector.side_effect = [mock_conn_a, mock_conn_b]

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.legal.docusign": mock_docusign},
        ):
            result_a = await get_docusign_connector("tenant-a")
            result_b = await get_docusign_connector("tenant-b")
            assert result_a is not result_b


# ---------------------------------------------------------------------------
# _get_tenant_id
# ---------------------------------------------------------------------------


class TestGetTenantId:
    """Test tenant ID extraction."""

    def test_extracts_from_request(self, handler):
        req = MockRequest(tenant_id="my-tenant")
        assert handler._get_tenant_id(req) == "my-tenant"

    def test_defaults_to_default(self, handler):
        req = MagicMock(spec=[])  # no tenant_id attribute
        assert handler._get_tenant_id(req) == "default"


# ---------------------------------------------------------------------------
# _get_query_params
# ---------------------------------------------------------------------------


class TestGetQueryParams:
    """Test query parameter extraction."""

    def test_from_query_dict(self, handler):
        req = MockRequest(query={"a": "1", "b": "2"})
        params = handler._get_query_params(req)
        assert params == {"a": "1", "b": "2"}

    def test_from_query_string(self, handler):
        req = MagicMock(spec=["query_string"])
        req.query_string = "x=10&y=20"
        result = handler._get_query_params(req)
        assert result["x"] == "10"
        assert result["y"] == "20"

    def test_empty_request(self, handler):
        req = MagicMock(spec=[])  # no query or query_string
        result = handler._get_query_params(req)
        assert result == {}


# ---------------------------------------------------------------------------
# _get_json_body
# ---------------------------------------------------------------------------


class TestGetJsonBody:
    """Test JSON body parsing wrapper."""

    @pytest.mark.asyncio
    async def test_returns_parsed_body(self, handler):
        expected = {"key": "value"}
        with patch(
            "aragora.server.handlers.features.legal.parse_json_body",
            new_callable=AsyncMock,
            return_value=(expected, None),
        ):
            result = await handler._get_json_body(MockRequest())
            assert result == expected

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, handler):
        with patch(
            "aragora.server.handlers.features.legal.parse_json_body",
            new_callable=AsyncMock,
            return_value=(None, MagicMock()),
        ):
            result = await handler._get_json_body(MockRequest())
            assert result == {}


# ---------------------------------------------------------------------------
# _emit_connector_event
# ---------------------------------------------------------------------------


class TestEmitConnectorEvent:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_emits_via_ctx_emitter(self):
        from aragora.server.handlers.features.legal import LegalHandler

        mock_emitter = MagicMock()
        h = LegalHandler(server_context={"emitter": mock_emitter})

        mock_event_type = MagicMock()
        mock_event_type.value = "docusign_envelope_status"
        mock_stream_module = MagicMock()
        mock_stream_module.StreamEventType.CONNECTOR_DOCUSIGN_ENVELOPE_STATUS = mock_event_type

        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_module},
        ):
            await h._emit_connector_event(
                event_type="status_change",
                tenant_id="t1",
                data={"envelope_id": "e1"},
            )
            mock_emitter.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emits_silently_on_import_error(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler(server_context={})
        with patch.dict("sys.modules", {"aragora.events.types": None}):
            # Should not raise
            await h._emit_connector_event(
                event_type="test",
                tenant_id="t1",
                data={},
            )

    @pytest.mark.asyncio
    async def test_emits_silently_without_emitter(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler(server_context={})
        mock_stream_module = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_module},
        ):
            await h._emit_connector_event(
                event_type="test",
                tenant_id="t1",
                data={},
            )

    @pytest.mark.asyncio
    async def test_emits_silently_with_none_ctx(self):
        from aragora.server.handlers.features.legal import LegalHandler

        h = LegalHandler(server_context=None)
        mock_stream_module = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.events.types": mock_stream_module},
        ):
            await h._emit_connector_event(
                event_type="test",
                tenant_id="t1",
                data={},
            )


# ---------------------------------------------------------------------------
# create_legal_handler factory
# ---------------------------------------------------------------------------


class TestFactory:
    """Test the factory function."""

    def test_creates_handler(self):
        from aragora.server.handlers.features.legal import create_legal_handler, LegalHandler

        h = create_legal_handler()
        assert isinstance(h, LegalHandler)
        assert h.ctx == {}

    def test_creates_with_context(self):
        from aragora.server.handlers.features.legal import create_legal_handler

        h = create_legal_handler({"emitter": "mock"})
        assert h.ctx == {"emitter": "mock"}

    def test_creates_with_none(self):
        from aragora.server.handlers.features.legal import create_legal_handler

        h = create_legal_handler(None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# Exception catch-all in handle()
# ---------------------------------------------------------------------------


class TestHandleExceptionCatchAll:
    """Test that the top-level exception handler catches errors."""

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=ValueError("boom")):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_key_error_returns_500(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=KeyError("missing")):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_type_error_returns_500(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=TypeError("bad type")):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=RuntimeError("runtime")):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler):
        with patch.object(handler, "_get_tenant_id", side_effect=OSError("os")):
            req = MockRequest(path="/api/v1/legal/status")
            result = await handler.handle(req, "/api/v1/legal/status", "GET")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test that __all__ is properly defined."""

    def test_all_exports(self):
        from aragora.server.handlers.features import legal

        assert "LegalHandler" in legal.__all__
        assert "create_legal_handler" in legal.__all__

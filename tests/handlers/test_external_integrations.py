"""Tests for External Integrations Handler.

Tests the ExternalIntegrationsHandler which provides REST API endpoints
for managing external automation integrations (Zapier, Make, n8n).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.external_integrations import ExternalIntegrationsHandler


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def server_context():
    """Create mock server context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
    }


@pytest.fixture
def handler(server_context):
    """Create external integrations handler with mock context."""
    return ExternalIntegrationsHandler(server_context)


class TestExternalIntegrationsCanHandle:
    """Tests for can_handle method."""

    def test_handles_zapier_path(self, handler):
        """Test that handler can handle Zapier paths."""
        assert handler.can_handle("/api/integrations/zapier/apps") is True
        assert handler.can_handle("/api/v1/integrations/zapier/apps") is True

    def test_handles_make_path(self, handler):
        """Test that handler can handle Make paths."""
        assert handler.can_handle("/api/integrations/make/connections") is True
        assert handler.can_handle("/api/v1/integrations/make/connections") is True

    def test_handles_n8n_path(self, handler):
        """Test that handler can handle n8n paths."""
        assert handler.can_handle("/api/integrations/n8n/credentials") is True
        assert handler.can_handle("/api/v1/integrations/n8n/credentials") is True

    def test_rejects_unknown_platform(self, handler):
        """Test that handler rejects unknown platform paths."""
        assert handler.can_handle("/api/integrations/unknown/apps") is False

    def test_rejects_unrelated_path(self, handler):
        """Test that handler rejects unrelated paths."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False


class TestZapierAppValidation:
    """Tests for Zapier app endpoint validation."""

    def test_create_app_missing_workspace_id(self, handler):
        """Test that creating app without workspace_id returns proper error code."""
        mock_handler = MagicMock()
        body_content = json.dumps({}).encode()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_content)),
        }
        mock_handler.rfile.read.return_value = body_content

        # Mock RBAC to allow
        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_create_zapier_app({}, mock_handler)

        assert result.status_code == 400
        body = parse_body(result)
        assert body["error"] == "workspace_id is required"
        assert body["code"] == "MISSING_WORKSPACE_ID"

    def test_delete_app_not_found(self, handler):
        """Test that deleting non-existent app returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_zapier") as mock_zapier:
                mock_zapier.return_value.delete_app.return_value = False
                result = handler._handle_delete_zapier_app("nonexistent", mock_handler)

        assert result.status_code == 404
        body = parse_body(result)
        assert "ZAPIER_APP_NOT_FOUND" in body["code"]


class TestZapierTriggerValidation:
    """Tests for Zapier trigger endpoint validation."""

    def test_subscribe_trigger_missing_app_id(self, handler):
        """Test that subscribing trigger without app_id returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_subscribe_zapier_trigger(
                {"trigger_type": "debate_complete", "webhook_url": "https://example.com"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_APP_ID"

    def test_subscribe_trigger_missing_trigger_type(self, handler):
        """Test that subscribing trigger without trigger_type returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_subscribe_zapier_trigger(
                {"app_id": "app123", "webhook_url": "https://example.com"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_TRIGGER_TYPE"

    def test_subscribe_trigger_missing_webhook_url(self, handler):
        """Test that subscribing trigger without webhook_url returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_subscribe_zapier_trigger(
                {"app_id": "app123", "trigger_type": "debate_complete"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_WEBHOOK_URL"

    def test_subscribe_trigger_failed(self, handler):
        """Test that failed subscription returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_zapier") as mock_zapier:
                mock_zapier.return_value.subscribe_trigger.return_value = None
                result = handler._handle_subscribe_zapier_trigger(
                    {
                        "app_id": "app123",
                        "trigger_type": "debate_complete",
                        "webhook_url": "https://example.com",
                    },
                    mock_handler,
                )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "TRIGGER_SUBSCRIBE_FAILED"

    def test_unsubscribe_trigger_missing_app_id(self, handler):
        """Test that unsubscribing trigger without app_id returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_unsubscribe_zapier_trigger("", "trigger123", mock_handler)

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_APP_ID"


class TestMakeWebhookValidation:
    """Tests for Make webhook endpoint validation."""

    def test_register_webhook_missing_connection_id(self, handler):
        """Test that registering webhook without connection_id returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_register_make_webhook(
                {"module_type": "debate_start", "webhook_url": "https://example.com"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_CONNECTION_ID"

    def test_register_webhook_missing_module_type(self, handler):
        """Test that registering webhook without module_type returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_register_make_webhook(
                {"connection_id": "conn123", "webhook_url": "https://example.com"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_MODULE_TYPE"


class TestN8nWebhookValidation:
    """Tests for n8n webhook endpoint validation."""

    def test_register_webhook_missing_credential_id(self, handler):
        """Test that registering webhook without credential_id returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_register_n8n_webhook(
                {"events": ["debate_complete"]},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_CREDENTIAL_ID"

    def test_register_webhook_missing_events(self, handler):
        """Test that registering webhook without events returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_register_n8n_webhook(
                {"credential_id": "cred123"},
                mock_handler,
            )

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "MISSING_EVENTS"


class TestTestIntegrationEndpoint:
    """Tests for integration test endpoint."""

    def test_test_unknown_platform(self, handler):
        """Test that testing unknown platform returns proper error code."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            result = handler._handle_test_integration("unknown_platform", mock_handler)

        assert result.status_code == 400
        body = parse_body(result)
        assert body["code"] == "UNKNOWN_PLATFORM"

    def test_test_zapier_success(self, handler):
        """Test that testing Zapier returns success."""
        mock_handler = MagicMock()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_zapier") as mock_zapier:
                mock_instance = MagicMock()
                mock_instance._apps = {}
                mock_instance.TRIGGER_TYPES = {"debate_complete": "Debate Complete"}
                mock_instance.ACTION_TYPES = {"create_debate": "Create Debate"}
                mock_zapier.return_value = mock_instance

                result = handler._handle_test_integration("zapier", mock_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["platform"] == "zapier"
        assert body["status"] == "ok"

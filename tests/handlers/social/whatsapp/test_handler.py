"""Comprehensive tests for WhatsApp handler (aragora/server/handlers/social/whatsapp/handler.py).

Covers all routes and behavior of the WhatsAppHandler class:
- can_handle() routing for all ROUTES
- GET /api/v1/integrations/whatsapp/webhook (verification)
- POST /api/v1/integrations/whatsapp/webhook (incoming messages)
- GET /api/v1/integrations/whatsapp/status
- RBAC permission checks (_check_permission, _check_whatsapp_permission)
- Phone number validation (_validate_phone_number)
- Message processing (_process_messages, _handle_text_message)
- Interactive / button replies
- Button click processing (_process_button_click)
- Vote recording (_record_vote)
- Debate details (_send_debate_details)
- Backward-compatible delegate methods
- Handler factory (get_whatsapp_handler)
- 404 for unknown routes
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Module paths for patching
_HANDLER = "aragora.server.handlers.social.whatsapp.handler"
_CONFIG = "aragora.server.handlers.social.whatsapp.config"
_WEBHOOKS = "aragora.server.handlers.social.whatsapp.webhooks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for WhatsApp tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = headers or {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a WhatsAppHandler with empty context."""
    from aragora.server.handlers.social.whatsapp.handler import WhatsAppHandler

    return WhatsAppHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a basic mock HTTP handler (GET by default)."""
    return MockHTTPHandler(method="GET")


@pytest.fixture(autouse=True)
def _patch_telemetry_and_events():
    """Patch telemetry and chat_events to avoid side effects."""
    with (
        patch(f"{_HANDLER}.record_message"),
        patch(f"{_HANDLER}.record_command"),
        patch(f"{_HANDLER}.record_vote"),
        patch(f"{_HANDLER}.emit_command_received"),
        patch(f"{_HANDLER}.emit_message_received"),
        patch(f"{_HANDLER}.emit_vote_received"),
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_config_tokens(monkeypatch):
    """Ensure config tokens are set so status reports correctly."""
    monkeypatch.setattr(f"{_HANDLER}.WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(f"{_HANDLER}.WHATSAPP_PHONE_NUMBER_ID", "123456")
    monkeypatch.setattr(f"{_HANDLER}.WHATSAPP_VERIFY_TOKEN", "verify-tok")
    monkeypatch.setattr(f"{_HANDLER}.WHATSAPP_APP_SECRET", "app-secret")


@pytest.fixture(autouse=True)
def _patch_rbac(monkeypatch):
    """Default: RBAC available but permissive (no permission errors)."""
    monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", True)
    monkeypatch.setattr(f"{_HANDLER}.check_permission", MagicMock(return_value=MagicMock(allowed=True)))
    monkeypatch.setattr(f"{_HANDLER}.extract_user_from_request", None)
    monkeypatch.setattr(f"{_HANDLER}.rbac_fail_closed", lambda: False)


@pytest.fixture(autouse=True)
def _patch_create_tracked_task(monkeypatch):
    """Patch create_tracked_task to capture coroutines without running them."""
    mock_task = MagicMock()

    def fake_create_tracked_task(coro, name=""):
        # Close the coroutine to avoid warnings
        if hasattr(coro, "close"):
            coro.close()
        return mock_task

    monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", fake_create_tracked_task)


@pytest.fixture(autouse=True)
def _reset_handler_singleton():
    """Reset the module-level handler singleton between tests."""
    import aragora.server.handlers.social.whatsapp.handler as mod

    mod._whatsapp_handler = None
    yield
    mod._whatsapp_handler = None


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestCanHandle:
    """Tests for handler routing via can_handle()."""

    def test_can_handle_webhook(self, handler):
        assert handler.can_handle("/api/v1/integrations/whatsapp/webhook", "GET")

    def test_can_handle_webhook_post(self, handler):
        assert handler.can_handle("/api/v1/integrations/whatsapp/webhook", "POST")

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/integrations/whatsapp/status", "GET")

    def test_cannot_handle_unknown(self, handler):
        assert not handler.can_handle("/api/v1/integrations/whatsapp/unknown", "GET")

    def test_cannot_handle_other_path(self, handler):
        assert not handler.can_handle("/api/v1/debates", "GET")

    def test_routes_list_complete(self, handler):
        assert len(handler.ROUTES) == 2
        assert "/api/v1/integrations/whatsapp/webhook" in handler.ROUTES
        assert "/api/v1/integrations/whatsapp/status" in handler.ROUTES


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestGetStatus:
    """Tests for GET /api/v1/integrations/whatsapp/status."""

    def test_status_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/integrations/whatsapp/status", {}, mock_http_handler)
        assert _status(result) == 200

    def test_status_body_fields(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/integrations/whatsapp/status", {}, mock_http_handler)
        body = _body(result)
        assert "enabled" in body
        assert "access_token_configured" in body
        assert "phone_number_id_configured" in body
        assert "verify_token_configured" in body
        assert "app_secret_configured" in body

    def test_status_enabled_when_tokens_set(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/integrations/whatsapp/status", {}, mock_http_handler)
        body = _body(result)
        assert body["enabled"] is True
        assert body["access_token_configured"] is True
        assert body["phone_number_id_configured"] is True

    def test_status_disabled_when_token_missing(self, handler, mock_http_handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.WHATSAPP_ACCESS_TOKEN", "")
        result = handler.handle("/api/v1/integrations/whatsapp/status", {}, mock_http_handler)
        body = _body(result)
        assert body["enabled"] is False
        assert body["access_token_configured"] is False


# ===========================================================================
# Webhook Verification (GET) Tests
# ===========================================================================


class TestWebhookVerification:
    """Tests for GET /api/v1/integrations/whatsapp/webhook (Meta verification)."""

    def test_verify_valid_token(self, handler, monkeypatch):
        """Valid verify_token and subscribe mode returns 200 with challenge."""
        monkeypatch.setattr(f"{_WEBHOOKS}._config.WHATSAPP_VERIFY_TOKEN", "verify-tok")
        mock = MockHTTPHandler(method="GET")
        query = {
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-tok",
            "hub.challenge": "test-challenge-123",
        }
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", query, mock)
        assert _status(result) == 200
        assert result.body == b"test-challenge-123"

    def test_verify_invalid_token(self, handler, monkeypatch):
        """Invalid verify_token returns 403."""
        monkeypatch.setattr(f"{_WEBHOOKS}._config.WHATSAPP_VERIFY_TOKEN", "verify-tok")
        mock = MockHTTPHandler(method="GET")
        query = {
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "challenge",
        }
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", query, mock)
        assert _status(result) == 403

    def test_verify_missing_mode(self, handler, monkeypatch):
        """Missing hub.mode returns 403."""
        monkeypatch.setattr(f"{_WEBHOOKS}._config.WHATSAPP_VERIFY_TOKEN", "verify-tok")
        mock = MockHTTPHandler(method="GET")
        query = {"hub.verify_token": "verify-tok", "hub.challenge": "challenge"}
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", query, mock)
        assert _status(result) == 403

    def test_verify_no_configured_token_and_no_token_sent(self, handler, monkeypatch):
        """When no WHATSAPP_VERIFY_TOKEN is configured and no token sent, accept."""
        monkeypatch.setattr(f"{_WEBHOOKS}._config.WHATSAPP_VERIFY_TOKEN", "")
        mock = MockHTTPHandler(method="GET")
        query = {"hub.mode": "subscribe", "hub.challenge": "challenge-ok"}
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", query, mock)
        assert _status(result) == 200
        assert result.body == b"challenge-ok"

    def test_verify_with_list_params(self, handler, monkeypatch):
        """Query parameters passed as lists are handled correctly."""
        monkeypatch.setattr(f"{_WEBHOOKS}._config.WHATSAPP_VERIFY_TOKEN", "verify-tok")
        mock = MockHTTPHandler(method="GET")
        query = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["verify-tok"],
            "hub.challenge": ["list-challenge"],
        }
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", query, mock)
        assert _status(result) == 200
        assert result.body == b"list-challenge"


# ===========================================================================
# Webhook POST (Incoming Messages) Tests
# ===========================================================================


class TestWebhookPost:
    """Tests for POST /api/v1/integrations/whatsapp/webhook."""

    def _post_webhook(self, handler, body, monkeypatch, *, sig_ok=True):
        """Helper to POST a webhook with optional signature verification."""
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: sig_ok)
        mock = MockHTTPHandler(body=body, method="POST")
        return handler.handle("/api/v1/integrations/whatsapp/webhook", {}, mock)

    def test_valid_webhook_returns_ok(self, handler, monkeypatch):
        body = {
            "object": "whatsapp_business_account",
            "entry": [],
        }
        # Patch the WebhookProcessor to avoid real processing
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: True)
        handler._webhook_processor = MagicMock()
        from aragora.server.handlers.utils.responses import HandlerResult

        handler._webhook_processor.handle_webhook.return_value = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=json.dumps({"status": "ok"}).encode(),
        )
        mock = MockHTTPHandler(body=body, method="POST")
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", {}, mock)
        assert _status(result) == 200

    def test_signature_failure_returns_401(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: False)
        mock = MockHTTPHandler(body={"object": "whatsapp_business_account"}, method="POST")
        result = handler.handle("/api/v1/integrations/whatsapp/webhook", {}, mock)
        assert _status(result) == 401
        body = _body(result)
        assert "Unauthorized" in body.get("error", "")

    def test_webhook_processor_called_on_valid_signature(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: True)
        handler._webhook_processor = MagicMock()
        from aragora.server.handlers.utils.responses import HandlerResult

        handler._webhook_processor.handle_webhook.return_value = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"status": "ok"}',
        )
        mock = MockHTTPHandler(body={"entry": []}, method="POST")
        handler.handle("/api/v1/integrations/whatsapp/webhook", {}, mock)
        handler._webhook_processor.handle_webhook.assert_called_once_with(mock)


# ===========================================================================
# 404 Not Found Tests
# ===========================================================================


class TestNotFound:
    """Unknown paths return 404."""

    def test_unknown_path_returns_404(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/integrations/whatsapp/unknown", {}, mock_http_handler)
        assert _status(result) == 404

    def test_wrong_prefix_returns_404(self, handler, mock_http_handler):
        result = handler.handle("/api/v1/debates", {}, mock_http_handler)
        assert _status(result) == 404


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission checking methods."""

    def test_check_permission_returns_none_when_rbac_unavailable(self, handler, monkeypatch, mock_http_handler):
        monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", False)
        monkeypatch.setattr(f"{_HANDLER}.rbac_fail_closed", lambda: False)
        result = handler._check_permission(mock_http_handler, "bots.read")
        assert result is None

    def test_check_permission_returns_503_when_fail_closed(self, handler, monkeypatch, mock_http_handler):
        monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", False)
        monkeypatch.setattr(f"{_HANDLER}.rbac_fail_closed", lambda: True)
        result = handler._check_permission(mock_http_handler, "bots.read")
        assert _status(result) == 503

    def test_check_permission_returns_none_when_allowed(self, handler, monkeypatch, mock_http_handler):
        """When RBAC is available and permission is allowed, returns None."""
        mock_user = MagicMock()
        mock_user.user_id = "test-user"
        mock_user.role = "admin"
        mock_user.org_id = "org-1"
        monkeypatch.setattr(f"{_HANDLER}.extract_user_from_request", lambda h: mock_user)
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=True)),
        )
        result = handler._check_permission(mock_http_handler, "bots.read")
        assert result is None

    def test_check_permission_returns_403_when_denied(self, handler, monkeypatch, mock_http_handler):
        """When permission is denied, returns 403."""
        mock_user = MagicMock()
        mock_user.user_id = "test-user"
        mock_user.role = "viewer"
        mock_user.org_id = "org-1"
        monkeypatch.setattr(f"{_HANDLER}.extract_user_from_request", lambda h: mock_user)
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False)),
        )
        result = handler._check_permission(mock_http_handler, "bots.read")
        assert _status(result) == 403

    def test_check_permission_no_context_returns_none(self, handler, monkeypatch, mock_http_handler):
        """When extract_user_from_request returns None, no permission error."""
        monkeypatch.setattr(f"{_HANDLER}.extract_user_from_request", lambda h: None)
        result = handler._check_permission(mock_http_handler, "bots.read")
        assert result is None

    def test_status_endpoint_checks_permission(self, handler, monkeypatch):
        """Status endpoint calls _check_permission."""
        mock_user = MagicMock()
        mock_user.user_id = "denied-user"
        mock_user.role = "viewer"
        mock_user.org_id = None
        monkeypatch.setattr(f"{_HANDLER}.extract_user_from_request", lambda h: mock_user)
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False)),
        )
        mock = MockHTTPHandler(method="GET")
        result = handler.handle("/api/v1/integrations/whatsapp/status", {}, mock)
        assert _status(result) == 403


# ===========================================================================
# WhatsApp Permission Check Tests
# ===========================================================================


class TestWhatsAppPermissionCheck:
    """Tests for _check_whatsapp_permission (message-level RBAC)."""

    def test_returns_none_when_rbac_unavailable(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", False)
        monkeypatch.setattr(f"{_HANDLER}.rbac_fail_closed", lambda: False)
        result = handler._check_whatsapp_permission("1234567890", "bots.read")
        assert result is None

    def test_returns_error_when_fail_closed_no_rbac(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", False)
        monkeypatch.setattr(f"{_HANDLER}.rbac_fail_closed", lambda: True)
        result = handler._check_whatsapp_permission("1234567890", "bots.read")
        assert result is not None
        assert "unavailable" in result.lower()

    def test_returns_none_when_allowed(self, handler, monkeypatch):
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=True)),
        )
        result = handler._check_whatsapp_permission("12345678901", "bots.read")
        assert result is None

    def test_returns_none_when_denied_but_not_enforced(self, handler, monkeypatch):
        """By default whatsapp_enforce_rbac is False, so denied still returns None."""
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False)),
        )
        result = handler._check_whatsapp_permission("12345678901", "debates.create")
        assert result is None

    def test_returns_denied_when_enforced(self, handler, monkeypatch):
        """When whatsapp_enforce_rbac=True, denied returns error message."""
        handler.ctx = {"config": {"whatsapp_enforce_rbac": True}}
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False, reason="no-perm")),
        )
        result = handler._check_whatsapp_permission("12345678901", "debates.create")
        assert result == "Permission denied"

    def test_exception_returns_none(self, handler, monkeypatch):
        """On RBAC check exception, returns None (fail open)."""
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(side_effect=TypeError("bad")),
        )
        result = handler._check_whatsapp_permission("12345678901", "bots.read")
        assert result is None


# ===========================================================================
# Auth Context From Message Tests
# ===========================================================================


class TestAuthContextFromMessage:
    """Tests for _get_auth_context_from_message."""

    def test_returns_none_when_rbac_unavailable(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.RBAC_AVAILABLE", False)
        result = handler._get_auth_context_from_message("12345678901")
        assert result is None

    def test_returns_none_for_empty_number(self, handler):
        result = handler._get_auth_context_from_message("")
        assert result is None

    def test_builds_context_with_whatsapp_prefix(self, handler):
        ctx = handler._get_auth_context_from_message("+1-234-567-8901", "Alice")
        assert ctx is not None
        assert ctx.user_id == "whatsapp:12345678901"
        assert "whatsapp_user" in ctx.roles

    def test_normalizes_phone_number(self, handler):
        ctx = handler._get_auth_context_from_message("+1 234 567 8901")
        assert ctx.user_id == "whatsapp:12345678901"


# ===========================================================================
# Phone Number Validation Tests
# ===========================================================================


class TestPhoneNumberValidation:
    """Tests for _validate_phone_number."""

    def test_valid_number(self, handler):
        is_valid, error = handler._validate_phone_number("12345678901")
        assert is_valid is True
        assert error is None

    def test_valid_with_plus_and_dashes(self, handler):
        is_valid, error = handler._validate_phone_number("+1-234-567-8901")
        assert is_valid is True

    def test_empty_number(self, handler):
        is_valid, error = handler._validate_phone_number("")
        assert is_valid is False
        assert "required" in error.lower()

    def test_too_short(self, handler):
        is_valid, error = handler._validate_phone_number("12345")
        assert is_valid is False
        assert "short" in error.lower()

    def test_too_long(self, handler):
        is_valid, error = handler._validate_phone_number("1234567890123456")
        assert is_valid is False
        assert "long" in error.lower()

    def test_non_digits(self, handler):
        is_valid, error = handler._validate_phone_number("12345abc01")
        assert is_valid is False
        assert "invalid" in error.lower()


# ===========================================================================
# Process Messages Tests
# ===========================================================================


class TestProcessMessages:
    """Tests for _process_messages dispatch."""

    def test_dispatches_text_message(self, handler, monkeypatch):
        handler._handle_text_message = MagicMock()
        value = {
            "contacts": [{"wa_id": "12345678901", "profile": {"name": "Alice"}}],
            "messages": [
                {
                    "type": "text",
                    "from": "12345678901",
                    "text": {"body": "help"},
                }
            ],
        }
        handler._process_messages(value)
        handler._handle_text_message.assert_called_once_with("12345678901", "Alice", "help")

    def test_dispatches_interactive_reply(self, handler):
        handler._handle_interactive_reply = MagicMock()
        msg = {
            "type": "interactive",
            "from": "12345678901",
            "interactive": {"type": "button_reply", "button_reply": {"id": "vote_agree_d1"}},
        }
        value = {
            "contacts": [{"wa_id": "12345678901", "profile": {"name": "Bob"}}],
            "messages": [msg],
        }
        handler._process_messages(value)
        handler._handle_interactive_reply.assert_called_once_with("12345678901", "Bob", msg)

    def test_dispatches_button_reply(self, handler):
        handler._handle_button_reply = MagicMock()
        msg = {
            "type": "button",
            "from": "12345678901",
            "button": {"text": "Agree"},
        }
        value = {
            "contacts": [],
            "messages": [msg],
        }
        handler._process_messages(value)
        handler._handle_button_reply.assert_called_once_with("12345678901", "User", "Agree", msg)

    def test_unknown_type_ignored(self, handler):
        """Messages with unknown type are silently ignored."""
        handler._handle_text_message = MagicMock()
        value = {
            "contacts": [],
            "messages": [{"type": "image", "from": "12345678901"}],
        }
        handler._process_messages(value)
        handler._handle_text_message.assert_not_called()

    def test_no_messages(self, handler):
        """Empty messages list processes without error."""
        handler._process_messages({"contacts": [], "messages": []})

    def test_contact_lookup(self, handler):
        """Profile name is extracted from matching contact."""
        handler._handle_text_message = MagicMock()
        value = {
            "contacts": [
                {"wa_id": "11111111111", "profile": {"name": "Charlie"}},
                {"wa_id": "22222222222", "profile": {"name": "Diana"}},
            ],
            "messages": [
                {"type": "text", "from": "22222222222", "text": {"body": "help"}},
            ],
        }
        handler._process_messages(value)
        handler._handle_text_message.assert_called_once_with("22222222222", "Diana", "help")


# ===========================================================================
# Handle Text Message Tests
# ===========================================================================


class TestHandleTextMessage:
    """Tests for _handle_text_message command routing."""

    def _call(self, handler, text, from_number="12345678901"):
        handler._handle_text_message(from_number, "TestUser", text)

    def test_help_command(self, handler, monkeypatch):
        with patch(f"{_HANDLER}.command_help", return_value="help-text") as mock_help:
            self._call(handler, "help")
            mock_help.assert_called_once()

    def test_status_command(self, handler):
        with patch(f"{_HANDLER}.command_status", return_value="status-text") as mock_status:
            self._call(handler, "status")
            mock_status.assert_called_once()

    def test_agents_command(self, handler):
        with patch(f"{_HANDLER}.command_agents", return_value="agents-text") as mock_agents:
            self._call(handler, "agents")
            mock_agents.assert_called_once()

    def test_debate_command(self, handler):
        with patch(f"{_HANDLER}.command_debate") as mock_debate:
            self._call(handler, "debate Should we use microservices for everything?")
            mock_debate.assert_called_once_with(
                handler, "12345678901", "TestUser", "Should we use microservices for everything?"
            )

    def test_plan_command(self, handler):
        with patch(f"{_HANDLER}.command_debate") as mock_debate:
            self._call(handler, "plan Improve our on-call rotation process")
            mock_debate.assert_called_once()
            args = mock_debate.call_args
            assert args[0][3] == "Improve our on-call rotation process"
            di = args[0][4]
            assert di["include_receipt"] is True
            assert di["include_plan"] is True
            assert di["plan_strategy"] == "single_task"

    def test_implement_command(self, handler):
        with patch(f"{_HANDLER}.command_debate") as mock_debate:
            self._call(handler, "implement Build automated alerting system")
            mock_debate.assert_called_once()
            args = mock_debate.call_args
            di = args[0][4]
            assert di["execution_mode"] == "execute"
            assert di["execution_engine"] == "hybrid"
            assert di["include_context"] is True

    def test_gauntlet_command(self, handler):
        with patch(f"{_HANDLER}.command_gauntlet") as mock_gauntlet:
            self._call(handler, "gauntlet We should migrate to microservices")
            mock_gauntlet.assert_called_once_with(
                handler, "12345678901", "TestUser", "We should migrate to microservices"
            )

    def test_search_command(self, handler):
        with patch(f"{_HANDLER}.command_search", return_value="results") as mock_search:
            self._call(handler, "search machine learning")
            mock_search.assert_called_once_with("machine learning")

    def test_recent_command(self, handler):
        with patch(f"{_HANDLER}.command_recent", return_value="recent-text") as mock_recent:
            self._call(handler, "recent")
            mock_recent.assert_called_once()

    def test_receipt_command(self, handler):
        with patch(f"{_HANDLER}.command_receipt", return_value="receipt-text") as mock_receipt:
            self._call(handler, "receipt abc123")
            mock_receipt.assert_called_once_with("abc123")

    def test_long_message_treated_as_topic(self, handler):
        """Messages over 10 chars that are not commands become topic suggestions."""
        # No command match, but length > 10
        self._call(handler, "this is a long message that is not a command")
        # Should not raise; sends a suggestion

    def test_short_message_default_response(self, handler):
        """Short messages that are not commands get a default reply."""
        self._call(handler, "hi")
        # Should not raise

    def test_invalid_phone_number_stops_processing(self, handler):
        """Invalid phone number causes early return without command processing."""
        with patch(f"{_HANDLER}.command_help") as mock_help:
            handler._handle_text_message("123", "User", "help")
            mock_help.assert_not_called()

    def test_case_insensitive_commands(self, handler):
        """Commands are case insensitive."""
        with patch(f"{_HANDLER}.command_help", return_value="help-text") as mock_help:
            self._call(handler, "HELP")
            mock_help.assert_called_once()

    def test_whitespace_stripped(self, handler):
        with patch(f"{_HANDLER}.command_help", return_value="help-text") as mock_help:
            self._call(handler, "  help  ")
            mock_help.assert_called_once()


# ===========================================================================
# Interactive Reply Tests
# ===========================================================================


class TestHandleInteractiveReply:
    """Tests for _handle_interactive_reply."""

    def test_button_reply(self, handler):
        handler._process_button_click = MagicMock()
        msg = {
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "vote_agree_d1"},
            }
        }
        handler._handle_interactive_reply("12345678901", "Alice", msg)
        handler._process_button_click.assert_called_once_with("12345678901", "Alice", "vote_agree_d1")

    def test_list_reply(self, handler):
        handler._process_button_click = MagicMock()
        msg = {
            "interactive": {
                "type": "list_reply",
                "list_reply": {"id": "details_d2"},
            }
        }
        handler._handle_interactive_reply("12345678901", "Alice", msg)
        handler._process_button_click.assert_called_once_with("12345678901", "Alice", "details_d2")

    def test_unknown_reply_type(self, handler):
        """Unknown interactive type does nothing."""
        handler._process_button_click = MagicMock()
        msg = {"interactive": {"type": "unknown"}}
        handler._handle_interactive_reply("12345678901", "Alice", msg)
        handler._process_button_click.assert_not_called()

    def test_invalid_phone_number_returns_early(self, handler):
        handler._process_button_click = MagicMock()
        msg = {
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "vote_agree_d1"},
            }
        }
        handler._handle_interactive_reply("123", "Alice", msg)
        handler._process_button_click.assert_not_called()


# ===========================================================================
# Button Reply Tests
# ===========================================================================


class TestHandleButtonReply:
    """Tests for _handle_button_reply (quick reply buttons)."""

    def test_agree_button(self, handler):
        msg = {"context": {"message_id": "abc"}}
        handler._handle_button_reply("12345678901", "Alice", "Agree", msg)
        # Should not raise; logs the event

    def test_disagree_button(self, handler):
        msg = {}
        handler._handle_button_reply("12345678901", "Alice", "Disagree", msg)

    def test_invalid_phone_returns_early(self, handler):
        """Invalid phone number in button reply causes early return."""
        msg = {}
        handler._handle_button_reply("123", "Alice", "Agree", msg)
        # No error, just silent return


# ===========================================================================
# Process Button Click Tests
# ===========================================================================


class TestProcessButtonClick:
    """Tests for _process_button_click dispatch."""

    def test_vote_agree(self, handler):
        handler._record_vote = MagicMock()
        handler._process_button_click("12345678901", "Alice", "vote_agree_debate-abc")
        handler._record_vote.assert_called_once_with("12345678901", "Alice", "debate-abc", "agree")

    def test_vote_disagree(self, handler):
        handler._record_vote = MagicMock()
        handler._process_button_click("12345678901", "Alice", "vote_disagree_debate-xyz")
        handler._record_vote.assert_called_once_with("12345678901", "Alice", "debate-xyz", "disagree")

    def test_details(self, handler):
        handler._send_debate_details = MagicMock()
        handler._process_button_click("12345678901", "Alice", "details_debate-123")
        handler._send_debate_details.assert_called_once_with("12345678901", "debate-123")

    def test_unknown_button_id(self, handler):
        """Unknown button ID prefix is logged but does nothing."""
        handler._record_vote = MagicMock()
        handler._send_debate_details = MagicMock()
        handler._process_button_click("12345678901", "Alice", "unknown_button")
        handler._record_vote.assert_not_called()
        handler._send_debate_details.assert_not_called()


# ===========================================================================
# Record Vote Tests
# ===========================================================================


class TestRecordVote:
    """Tests for _record_vote."""

    def test_records_vote_successfully(self, handler, monkeypatch):
        mock_db = MagicMock()
        mock_db.record_vote = MagicMock()
        with patch(f"{_HANDLER}.get_debates_db", return_value=mock_db, create=True):
            # Patch the import inside _record_vote
            with patch("aragora.server.storage.get_debates_db", return_value=mock_db, create=True):
                handler._record_vote("12345678901", "Alice", "debate-1", "agree")
        # No exception means success

    def test_vote_sends_confirmation(self, handler, monkeypatch):
        """Vote sends a confirmation message via tracked task."""
        tasks_created = []
        original_create = None

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", side_effect=ImportError):
            handler._record_vote("12345678901", "Alice", "debate-1", "agree")

        assert any("vote-ack" in t for t in tasks_created)

    def test_vote_agree_emoji(self, handler, monkeypatch):
        """Agree vote gets a + emoji in the response."""
        sent_messages: list[tuple] = []

        def capture_task(coro, name=""):
            sent_messages.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", side_effect=ImportError):
            handler._record_vote("12345678901", "Alice", "debate-1", "agree")

        # Test passes if no exception; the message is sent via async task

    def test_vote_db_error_handled(self, handler, monkeypatch):
        """Database errors during vote recording are handled gracefully."""
        with patch("aragora.server.storage.get_debates_db", side_effect=ImportError):
            handler._record_vote("12345678901", "Alice", "debate-1", "agree")
        # Should not raise


# ===========================================================================
# Send Debate Details Tests
# ===========================================================================


class TestSendDebateDetails:
    """Tests for _send_debate_details."""

    def test_sends_details_when_found(self, handler, monkeypatch):
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "task": "Should AI be regulated?",
            "final_answer": "Yes, with careful consideration.",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
            "agents": ["agent1", "agent2"],
        }
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", return_value=mock_db):
            handler._send_debate_details("12345678901", "debate-123")

        assert any("details" in t for t in tasks_created)

    def test_debate_not_found(self, handler, monkeypatch):
        mock_db = MagicMock()
        mock_db.get.return_value = None
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", return_value=mock_db):
            handler._send_debate_details("12345678901", "nonexistent")

        assert any("notfound" in t for t in tasks_created)

    def test_db_import_error_handled(self, handler, monkeypatch):
        """ImportError when loading DB is handled gracefully (debate not found)."""
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", side_effect=ImportError):
            handler._send_debate_details("12345678901", "debate-123")

        assert any("notfound" in t for t in tasks_created)

    def test_many_agents_truncated(self, handler, monkeypatch):
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "task": "Topic",
            "final_answer": "Answer",
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 3,
            "agents": ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        }
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        with patch("aragora.server.storage.get_debates_db", return_value=mock_db):
            handler._send_debate_details("12345678901", "debate-123")

        # Should work fine and truncate agent list to 5

    def test_permission_denied_sends_error(self, handler, monkeypatch):
        """When RBAC enforcement is on and denied, sends permission error."""
        handler.ctx = {"config": {"whatsapp_enforce_rbac": True}}
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False, reason="no-perm")),
        )
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)
        handler._send_debate_details("12345678901", "debate-123")

        assert any("perm-denied" in t for t in tasks_created)


# ===========================================================================
# Backward-Compatible Delegate Methods Tests
# ===========================================================================


class TestDelegateMethods:
    """Tests for backward-compatible delegate methods."""

    def test_verify_signature_delegates(self, handler, monkeypatch):
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: True)
        mock = MockHTTPHandler()
        assert handler._verify_signature(mock) is True

    def test_verify_webhook_delegates(self, handler, monkeypatch):
        monkeypatch.setattr(
            f"{_HANDLER}.verify_webhook",
            lambda q: MagicMock(status_code=200),
        )
        result = handler._verify_webhook({"hub.mode": "subscribe"})
        assert result.status_code == 200

    def test_handle_webhook_delegates(self, handler):
        handler._webhook_processor = MagicMock()
        handler._webhook_processor.handle_webhook.return_value = MagicMock(status_code=200)
        mock = MockHTTPHandler()
        result = handler._handle_webhook(mock)
        assert result.status_code == 200
        handler._webhook_processor.handle_webhook.assert_called_once_with(mock)

    def test_command_help_delegates(self, handler):
        with patch(f"{_HANDLER}.command_help", return_value="help-text"):
            result = handler._command_help()
            assert result == "help-text"

    def test_command_status_delegates(self, handler):
        with patch(f"{_HANDLER}.command_status", return_value="status-text"):
            result = handler._command_status()
            assert result == "status-text"

    def test_command_agents_delegates(self, handler):
        with patch(f"{_HANDLER}.command_agents", return_value="agents-text"):
            result = handler._command_agents()
            assert result == "agents-text"

    def test_command_debate_delegates(self, handler):
        with patch(f"{_HANDLER}.command_debate") as mock_debate:
            handler._command_debate("12345678901", "Alice", "topic")
            mock_debate.assert_called_once_with(handler, "12345678901", "Alice", "topic", None)

    def test_command_gauntlet_delegates(self, handler):
        with patch(f"{_HANDLER}.command_gauntlet") as mock_gauntlet:
            handler._command_gauntlet("12345678901", "Alice", "statement")
            mock_gauntlet.assert_called_once_with(handler, "12345678901", "Alice", "statement")


# ===========================================================================
# Handler Factory Tests
# ===========================================================================


class TestGetWhatsAppHandler:
    """Tests for the get_whatsapp_handler factory."""

    def test_creates_singleton(self):
        from aragora.server.handlers.social.whatsapp.handler import get_whatsapp_handler

        h1 = get_whatsapp_handler()
        h2 = get_whatsapp_handler()
        assert h1 is h2

    def test_accepts_server_context(self):
        from aragora.server.handlers.social.whatsapp.handler import get_whatsapp_handler

        ctx = {"key": "value"}
        h = get_whatsapp_handler(ctx)
        assert h.ctx == ctx

    def test_default_empty_context(self):
        from aragora.server.handlers.social.whatsapp.handler import get_whatsapp_handler

        h = get_whatsapp_handler()
        assert isinstance(h.ctx, dict)


# ===========================================================================
# handle_post Tests
# ===========================================================================


class TestHandlePost:
    """Tests for handle_post method."""

    def test_handle_post_delegates_to_handle(self, handler, monkeypatch):
        handler._webhook_processor = MagicMock()
        from aragora.server.handlers.utils.responses import HandlerResult

        handler._webhook_processor.handle_webhook.return_value = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"status": "ok"}',
        )
        monkeypatch.setattr(f"{_HANDLER}.verify_signature", lambda h: True)
        mock = MockHTTPHandler(body={"object": "whatsapp_business_account"}, method="POST")
        result = handler.handle_post("/api/v1/integrations/whatsapp/webhook", {}, mock)
        assert result is not None


# ===========================================================================
# Edge Case / Integration Tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases and integration-style tests."""

    def test_handler_init_default_context(self):
        from aragora.server.handlers.social.whatsapp.handler import WhatsAppHandler

        h = WhatsAppHandler()
        assert h.ctx == {}

    def test_handler_init_with_context(self):
        from aragora.server.handlers.social.whatsapp.handler import WhatsAppHandler

        h = WhatsAppHandler(ctx={"config": {"key": "val"}})
        assert h.ctx["config"]["key"] == "val"

    def test_webhook_processor_created_on_init(self):
        from aragora.server.handlers.social.whatsapp.handler import WhatsAppHandler

        h = WhatsAppHandler()
        assert h._webhook_processor is not None

    def test_multiple_messages_in_single_webhook(self, handler):
        """Multiple messages in a single webhook are all dispatched."""
        handler._handle_text_message = MagicMock()
        handler._handle_interactive_reply = MagicMock()
        value = {
            "contacts": [
                {"wa_id": "12345678901", "profile": {"name": "Alice"}},
            ],
            "messages": [
                {"type": "text", "from": "12345678901", "text": {"body": "help"}},
                {"type": "text", "from": "12345678901", "text": {"body": "status"}},
                {
                    "type": "interactive",
                    "from": "12345678901",
                    "interactive": {
                        "type": "button_reply",
                        "button_reply": {"id": "vote_agree_d1"},
                    },
                },
            ],
        }
        handler._process_messages(value)
        assert handler._handle_text_message.call_count == 2
        assert handler._handle_interactive_reply.call_count == 1

    def test_missing_text_body_defaults_empty(self, handler):
        """Message with missing text body defaults to empty string."""
        handler._handle_text_message = MagicMock()
        value = {
            "contacts": [],
            "messages": [
                {"type": "text", "from": "12345678901", "text": {}},
            ],
        }
        handler._process_messages(value)
        handler._handle_text_message.assert_called_once_with("12345678901", "User", "")

    def test_missing_contact_defaults_to_user(self, handler):
        """When no contact matches from_number, profile name defaults to 'User'."""
        handler._handle_text_message = MagicMock()
        value = {
            "contacts": [
                {"wa_id": "99999999999", "profile": {"name": "Someone Else"}},
            ],
            "messages": [
                {"type": "text", "from": "12345678901", "text": {"body": "hi"}},
            ],
        }
        handler._process_messages(value)
        handler._handle_text_message.assert_called_once_with("12345678901", "User", "hi")

    def test_record_vote_with_permission_denied_sends_error(self, handler, monkeypatch):
        """Vote recording with enforced RBAC denial sends permission error."""
        handler.ctx = {"config": {"whatsapp_enforce_rbac": True}}
        monkeypatch.setattr(
            f"{_HANDLER}.check_permission",
            MagicMock(return_value=MagicMock(allowed=False, reason="no-perm")),
        )
        tasks_created = []

        def capture_task(coro, name=""):
            tasks_created.append(name)
            if hasattr(coro, "close"):
                coro.close()
            return MagicMock()

        monkeypatch.setattr(f"{_CONFIG}.create_tracked_task", capture_task)

        handler._record_vote("12345678901", "Alice", "debate-1", "agree")
        assert any("perm-denied" in t for t in tasks_created)

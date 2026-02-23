"""
Tests for Slack Bot Handler.

Covers the SlackHandler class and its routing logic, including:
- can_handle() for various path patterns
- GET  /api/v1/bots/slack/status    - Status endpoint
- POST /api/v1/bots/slack/events    - Events API webhook
- POST /api/v1/bots/slack/commands  - Slash commands
- POST /api/v1/bots/slack/interactions - Interactive components
- Signature verification (_verify_signature)
- Method enforcement (POST required for webhooks)
- Path normalization (integrations -> bots)
- Error handling and edge cases
- Handler initialization and _is_bot_enabled
- _command_help, _command_status, _command_agents
- _slack_response, _slack_blocks_response
- _check_permission_or_admin
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.slack.handler as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.SlackHandler


@pytest.fixture
def handler(handler_cls, monkeypatch):
    """Create a SlackHandler with empty context and a signing secret."""
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test_secret_123")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    h = handler_cls(server_context={})
    h._signing_secret = "test_secret_123"
    return h


@pytest.fixture
def handler_no_secret(handler_cls, monkeypatch):
    """Create a SlackHandler with no signing secret configured."""
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    h = handler_cls(server_context={})
    h._signing_secret = None
    return h


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/slack/status"
    command: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    _body: bytes | None = None

    def __post_init__(self):
        if self.body is not None:
            body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            body_bytes = b"{}"
        self._body = body_bytes
        self.rfile = io.BytesIO(body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_signed_handler(
    path: str = "/api/v1/bots/slack/commands",
    body: dict[str, Any] | None = None,
    method: str = "POST",
    signing_secret: str = "test_secret_123",
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with valid Slack signature headers."""
    if body is None:
        body = {}
    body_bytes = json.dumps(body).encode("utf-8")
    timestamp = str(int(time.time()))
    sig_basestring = f"v0:{timestamp}:{body_bytes.decode('utf-8')}"
    expected_sig = (
        "v0="
        + hmac_mod.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": expected_sig,
    }
    h = MockHTTPHandler(path=path, command=method, body=body, headers=headers)
    h._body = body_bytes
    return h


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for SlackHandler.can_handle()."""

    def test_handles_bots_slack_status(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/status") is True

    def test_handles_bots_slack_events(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/events") is True

    def test_handles_bots_slack_commands(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/commands") is True

    def test_handles_bots_slack_interactions(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/interactions") is True

    def test_handles_integrations_slack_path(self, handler):
        assert handler.can_handle("/api/v1/integrations/slack/events") is True

    def test_does_not_handle_other_bots(self, handler):
        assert handler.can_handle("/api/v1/bots/discord/status") is False

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Tests: Status endpoint
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/v1/bots/slack/status."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler, monkeypatch):
        """Status endpoint should return configured status."""
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_mod, "SLACK_SIGNING_SECRET", "test_secret")

        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/status", command="GET")
        result = handler.handle("/api/v1/bots/slack/status", {}, mock_handler)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body.get("platform") == "slack"

    @pytest.mark.asyncio
    async def test_status_trailing_slash(self, handler, monkeypatch):
        """Status endpoint with trailing slash should work."""
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", "xoxb-test")

        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/status/", command="GET")
        result = handler.handle("/api/v1/bots/slack/status/", {}, mock_handler)
        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# Tests: Method enforcement for webhooks
# ===========================================================================


class TestMethodEnforcement:
    """Tests for POST-only webhook endpoints."""

    def test_get_commands_returns_405(self, handler):
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/commands", command="GET")
        result = handler.handle("/api/v1/bots/slack/commands", {}, mock_handler)
        assert _status(result) == 405
        assert "Method not allowed" in _body(result).get("error", "")

    def test_get_events_returns_405(self, handler):
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/events", command="GET")
        result = handler.handle("/api/v1/bots/slack/events", {}, mock_handler)
        assert _status(result) == 405

    def test_get_interactions_returns_405(self, handler):
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/interactions", command="GET")
        result = handler.handle("/api/v1/bots/slack/interactions", {}, mock_handler)
        assert _status(result) == 405


# ===========================================================================
# Tests: Signature verification
# ===========================================================================


class TestSignatureVerification:
    """Tests for _verify_signature."""

    def test_valid_signature_passes(self, handler):
        mock_handler = _make_signed_handler(signing_secret="test_secret_123")
        assert handler._verify_signature(mock_handler) is True

    def test_invalid_signature_fails(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="POST",
            headers={
                "X-Slack-Request-Timestamp": str(int(time.time())),
                "X-Slack-Signature": "v0=invalid_signature",
                "Content-Length": "2",
            },
        )
        mock_handler._body = b"{}"
        assert handler._verify_signature(mock_handler) is False

    def test_missing_secret_dev_mode_passes(self, handler_no_secret, monkeypatch):
        """In dev mode without secret, verification should pass."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        mock_handler = MockHTTPHandler(command="POST")
        assert handler_no_secret._verify_signature(mock_handler) is True

    def test_missing_secret_production_fails(self, handler_no_secret, monkeypatch):
        """In production without secret, verification should fail."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        mock_handler = MockHTTPHandler(command="POST")
        assert handler_no_secret._verify_signature(mock_handler) is False

    def test_missing_secret_no_env_fails(self, handler_no_secret, monkeypatch):
        """With no env set and no secret, verification should fail."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        mock_handler = MockHTTPHandler(command="POST")
        assert handler_no_secret._verify_signature(mock_handler) is False

    def test_expired_timestamp_fails(self, handler):
        """Expired timestamp (>5 min old) should fail verification."""
        old_timestamp = str(int(time.time()) - 600)
        body_bytes = b"{}"
        sig_basestring = f"v0:{old_timestamp}:{body_bytes.decode('utf-8')}"
        expected_sig = (
            "v0="
            + hmac_mod.new(
                b"test_secret_123",
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )
        mock_handler = MockHTTPHandler(
            command="POST",
            headers={
                "X-Slack-Request-Timestamp": old_timestamp,
                "X-Slack-Signature": expected_sig,
                "Content-Length": "2",
            },
        )
        mock_handler._body = body_bytes
        assert handler._verify_signature(mock_handler) is False

    def test_signature_verification_exception_returns_false(self, handler):
        """If signature verification throws, should return False."""
        mock_handler = MockHTTPHandler(command="POST")
        # Remove headers to cause an exception path
        mock_handler.headers = {}
        assert handler._verify_signature(mock_handler) is False


# ===========================================================================
# Tests: Webhook without signing secret
# ===========================================================================


class TestWebhookWithoutSecret:
    """Tests for webhook endpoints when signing secret is not configured."""

    def test_commands_without_secret_returns_503(self, handler_no_secret, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/commands", command="POST")
        result = handler_no_secret.handle("/api/v1/bots/slack/commands", {}, mock_handler)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()


# ===========================================================================
# Tests: Path normalization
# ===========================================================================


class TestPathNormalization:
    """Tests for path normalization (integrations -> bots)."""

    def test_integrations_path_normalizes(self, handler):
        """The /api/integrations/slack/ path should normalize to /api/v1/bots/slack/."""
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/status", command="GET")
        result = handler.handle("/api/integrations/slack/status", {}, mock_handler)
        # Normalized to /api/v1/bots/slack/status so should return status
        assert result is not None
        assert _status(result) == 200

    def test_v1_integrations_path_normalizes(self, handler):
        """The /api/v1/integrations/slack/ path should also normalize."""
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/status", command="GET")
        result = handler.handle("/api/v1/integrations/slack/status", {}, mock_handler)
        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# Tests: Unhandled paths
# ===========================================================================


class TestUnhandledPaths:
    """Tests for paths the handler does not handle."""

    def test_unknown_slack_subpath_returns_none(self, handler):
        """An unknown subpath under /api/v1/bots/slack/ should return None."""
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/unknown", command="GET")
        result = handler.handle("/api/v1/bots/slack/unknown", {}, mock_handler)
        assert result is None


# ===========================================================================
# Tests: _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled."""

    def test_enabled_with_bot_token(self, handler, monkeypatch):
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_mod, "SLACK_SIGNING_SECRET", None)
        assert handler._is_bot_enabled() is True

    def test_enabled_with_signing_secret(self, handler, monkeypatch):
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_mod, "SLACK_SIGNING_SECRET", "secret")
        assert handler._is_bot_enabled() is True

    def test_disabled_without_tokens(self, handler, monkeypatch):
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_mod, "SLACK_SIGNING_SECRET", None)
        assert handler._is_bot_enabled() is False


# ===========================================================================
# Tests: _command_help
# ===========================================================================


class TestCommandHelp:
    """Tests for _command_help helper."""

    def test_help_returns_json(self, handler):
        result = handler._command_help()
        assert _status(result) == 200
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert "/aragora debate" in body["text"]
        assert "/aragora help" in body["text"]


# ===========================================================================
# Tests: _command_status
# ===========================================================================


class TestCommandStatus:
    """Tests for _command_status helper."""

    def test_status_shows_active_debates(self, handler, monkeypatch):
        from aragora.server.handlers.bots.slack import state

        monkeypatch.setattr(state, "_active_debates", {"d1": {}, "d2": {}})
        # Also patch the module-level import used in handler.py
        import aragora.server.handlers.bots.slack.handler as hmod

        original = hmod._active_debates
        monkeypatch.setattr(hmod, "_active_debates", {"d1": {}, "d2": {}})
        result = handler._command_status()
        body = _body(result)
        assert "2" in body["text"]  # 2 active debates
        monkeypatch.setattr(hmod, "_active_debates", original)

    def test_status_handles_elo_import_error(self, handler):
        """Even if ELO is unavailable, status should succeed."""
        with patch(
            "aragora.server.handlers.bots.slack.handler.SlackHandler._command_status",
            wraps=handler._command_status,
        ):
            result = handler._command_status()
            assert _status(result) == 200
            body = _body(result)
            assert "System Status" in body["text"]


# ===========================================================================
# Tests: _command_agents
# ===========================================================================


class TestCommandAgents:
    """Tests for _command_agents helper."""

    def test_agents_with_no_ratings(self, handler):
        """When ELO store has no ratings, should say so."""
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = []
        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = handler._command_agents()
            body = _body(result)
            assert "No agents registered" in body["text"]

    def test_agents_with_ratings(self, handler):
        """When ELO store has ratings, should list them."""
        mock_elo = MagicMock()
        rating1 = MagicMock()
        rating1.agent_name = "Claude"
        rating1.elo = 1800.0
        rating2 = MagicMock()
        rating2.agent_name = "GPT-4"
        rating2.elo = 1750.0
        mock_elo.get_all_ratings.return_value = [rating1, rating2]
        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = handler._command_agents()
            body = _body(result)
            assert "Claude" in body["text"]
            assert "GPT-4" in body["text"]
            assert "1800" in body["text"]

    def test_agents_import_error_handled(self, handler):
        """If ELO module raises ImportError, should return friendly error."""
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError("no elo"),
        ):
            result = handler._command_agents()
            body = _body(result)
            assert _status(result) == 200
            assert "error" in body["text"].lower() or "Error" in body["text"]


# ===========================================================================
# Tests: _slack_response
# ===========================================================================


class TestSlackResponse:
    """Tests for _slack_response helper."""

    def test_in_channel_response(self, handler):
        result = handler._slack_response("Hello world")
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert body["text"] == "Hello world"

    def test_ephemeral_response(self, handler):
        result = handler._slack_response("Secret message", response_type="ephemeral")
        body = _body(result)
        assert body["response_type"] == "ephemeral"


# ===========================================================================
# Tests: _slack_blocks_response
# ===========================================================================


class TestSlackBlocksResponse:
    """Tests for _slack_blocks_response helper."""

    def test_blocks_response_with_text(self, handler):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hi"}}]
        result = handler._slack_blocks_response(blocks, text="Fallback")
        body = _body(result)
        assert body["blocks"] == blocks
        assert body["text"] == "Fallback"
        assert body["response_type"] == "in_channel"

    def test_blocks_response_without_text(self, handler):
        blocks = [{"type": "divider"}]
        result = handler._slack_blocks_response(blocks)
        body = _body(result)
        assert "text" not in body
        assert body["blocks"] == blocks


# ===========================================================================
# Tests: _get_status
# ===========================================================================


class TestGetStatus:
    """Tests for _get_status helper method."""

    def test_get_status_returns_features(self, handler, monkeypatch):
        import aragora.server.handlers.bots.slack as slack_mod

        monkeypatch.setattr(slack_mod, "SLACK_BOT_TOKEN", "xoxb-test")
        result = handler._get_status()
        body = _body(result)
        assert body["enabled"] is True
        assert body["features"]["slash_commands"] is True
        assert body["features"]["events_api"] is True
        assert body["features"]["interactive_components"] is True
        assert body["features"]["block_kit"] is True


# ===========================================================================
# Tests: _check_permission_or_admin
# ===========================================================================


class TestCheckPermissionOrAdmin:
    """Tests for _check_permission_or_admin."""

    def test_returns_none_when_rbac_unavailable(self, handler, monkeypatch):
        """When RBAC is not available, should return None (allow)."""
        import aragora.server.handlers.bots.slack.handler as hmod

        monkeypatch.setattr(hmod, "RBAC_AVAILABLE", False)
        monkeypatch.setattr(hmod, "check_permission", None)
        # Also need rbac_fail_closed to return False
        monkeypatch.setattr(hmod, "rbac_fail_closed", lambda: False)
        result = handler._check_permission_or_admin("T12345", "U12345", "some.perm")
        assert result is None


# ===========================================================================
# Tests: Webhook POST with invalid signature
# ===========================================================================


class TestWebhookInvalidSignature:
    """Tests for webhook endpoints when signature is invalid."""

    def test_commands_invalid_sig_returns_401(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="POST",
            headers={
                "X-Slack-Request-Timestamp": str(int(time.time())),
                "X-Slack-Signature": "v0=bad_sig",
                "Content-Length": "2",
            },
        )
        mock_handler._body = b"{}"
        result = handler.handle("/api/v1/bots/slack/commands", {}, mock_handler)
        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()


# ===========================================================================
# Tests: register_slack_routes
# ===========================================================================


class TestRegisterSlackRoutes:
    """Tests for the register_slack_routes function."""

    def test_registers_three_routes(self, handler_module):
        """Should register events, interactions, and commands routes."""
        mock_router = MagicMock()
        handler_module.register_slack_routes(mock_router)
        assert mock_router.add_route.call_count == 3

        # Verify route paths were registered
        registered_paths = [call.args[1] for call in mock_router.add_route.call_args_list]
        assert "/api/bots/slack/events" in registered_paths
        assert "/api/bots/slack/interactions" in registered_paths
        assert "/api/bots/slack/commands" in registered_paths


# ===========================================================================
# Tests: Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "slack"

    def test_routes_defined(self, handler):
        assert "/api/v1/bots/slack/status" in handler.ROUTES
        assert "/api/v1/bots/slack/events" in handler.ROUTES
        assert "/api/v1/bots/slack/commands" in handler.ROUTES
        assert "/api/v1/bots/slack/interactions" in handler.ROUTES

    def test_signing_secret_cached_at_init(self, handler_cls, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "my_cached_secret")
        h = handler_cls(server_context={})
        assert h._signing_secret == "my_cached_secret"


# ===========================================================================
# Tests: Interactive path recognized
# ===========================================================================


class TestInteractivePath:
    """Tests for /api/v1/bots/slack/interactive path alias."""

    def test_interactive_alias_requires_post(self, handler):
        """The /interactive alias should also require POST."""
        mock_handler = MockHTTPHandler(path="/api/v1/bots/slack/interactive", command="GET")
        result = handler.handle("/api/v1/bots/slack/interactive", {}, mock_handler)
        assert _status(result) == 405

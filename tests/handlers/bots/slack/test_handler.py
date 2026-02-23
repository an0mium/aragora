"""
Tests for Slack Bot main handler.

Covers all routes and behavior of the SlackHandler class:
- can_handle() routing for all defined routes and path aliases
- GET  /api/v1/bots/slack/status       - Bot status (RBAC-protected)
- POST /api/v1/bots/slack/events       - Events API webhook
- POST /api/v1/bots/slack/interactions  - Interactive component webhook
- POST /api/v1/bots/slack/commands      - Slash command webhook
- Path normalization: /api/integrations/slack/* and /api/v1/integrations/slack/*
- Slack signature verification (_verify_signature)
  - Missing signing secret (dev vs production)
  - Valid/invalid signatures
  - Body reading from _body attribute vs rfile
  - Signature verification errors
- _is_bot_enabled() with various token/secret combinations
- _get_status() direct method
- _command_help() returns command help text
- _command_status() returns system status
- _command_agents() returns agent list
  - No agents registered
  - Agents sorted by ELO
  - Import error fallback
  - Unexpected error fallback
- _slack_response() and _slack_blocks_response() formatting
- _check_permission_or_admin() RBAC helper
- Method enforcement: POST required for webhook paths
- register_slack_routes() deprecated registration function
- Error handling: timeout, missing secret, invalid signature
"""

from __future__ import annotations

import hashlib
import hmac
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
def slack_pkg():
    """Import the top-level slack package module."""
    import aragora.server.handlers.bots.slack as mod
    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.SlackHandler


@pytest.fixture
def handler(handler_cls, monkeypatch):
    """Create a SlackHandler with patched signing secret."""
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
    h = handler_cls(server_context={})
    # Override the cached signing secret
    h._signing_secret = "test-signing-secret"
    return h


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


def _compute_slack_signature(
    body: bytes, timestamp: str, signing_secret: str
) -> str:
    """Compute a valid Slack signature for testing."""
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/slack/commands"
    command: str = "POST"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    _body_bytes: bytes = field(default=b"{}", repr=False)

    def __post_init__(self):
        if self.body is not None:
            self._body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            self._body_bytes = b"{}"
        self.rfile = io.BytesIO(self._body_bytes)
        self._body = self._body_bytes
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(self._body_bytes))
        self.client_address = ("127.0.0.1", 12345)

    async def json(self) -> dict[str, Any]:
        return self.body or {}

    async def body_bytes(self) -> bytes:
        return self._body_bytes


def _make_slack_handler(
    body: dict[str, Any] | None = None,
    method: str = "POST",
    path: str = "/api/v1/bots/slack/commands",
    signing_secret: str = "test-signing-secret",
    timestamp: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with valid Slack signature headers."""
    if timestamp is None:
        timestamp = str(int(time.time()))

    body_bytes = json.dumps(body or {}).encode("utf-8")
    signature = _compute_slack_signature(body_bytes, timestamp, signing_secret)

    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": signature,
        "Content-Length": str(len(body_bytes)),
    }
    return MockHTTPHandler(
        path=path,
        command=method,
        body=body,
        headers=headers,
    )


def _make_get_handler(path: str = "/api/v1/bots/slack/status") -> MockHTTPHandler:
    """Create a MockHTTPHandler for GET requests."""
    return MockHTTPHandler(
        path=path,
        command="GET",
        headers={"Content-Length": "0"},
    )


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/status") is True

    def test_events_route(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/events") is True

    def test_interactions_route(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/interactions") is True

    def test_commands_route(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/commands") is True

    def test_integrations_alias_route(self, handler):
        assert handler.can_handle("/api/v1/integrations/slack/status") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/bots/discord/webhook") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/bots/slack") is False

    def test_routes_list_complete(self, handler):
        """ROUTES list contains exactly the expected paths."""
        assert set(handler.ROUTES) == {
            "/api/v1/bots/slack/status",
            "/api/v1/bots/slack/events",
            "/api/v1/bots/slack/interactions",
            "/api/v1/bots/slack/commands",
        }


# ===========================================================================
# GET /api/v1/bots/slack/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    def test_status_returns_200(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        assert _status(result) == 200

    def test_status_body_has_platform(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        body = _body(result)
        assert body["platform"] == "slack"

    def test_status_body_has_configured(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        body = _body(result)
        assert body["configured"] is True

    def test_status_body_has_features(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        body = _body(result)
        assert "features" in body
        assert body["features"]["slash_commands"] is True
        assert body["features"]["events_api"] is True
        assert body["features"]["interactive_components"] is True
        assert body["features"]["block_kit"] is True

    def test_status_body_has_active_debates(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        body = _body(result)
        assert "active_debates" in body
        assert isinstance(body["active_debates"], int)

    def test_status_not_configured(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", None)
        http_handler = _make_get_handler()
        result = handler.handle("/api/v1/bots/slack/status", {}, http_handler)
        body = _body(result)
        assert body["configured"] is False

    def test_status_with_trailing_slash(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler("/api/v1/bots/slack/status/")
        result = handler.handle("/api/v1/bots/slack/status/", {}, http_handler)
        assert _status(result) == 200

    def test_status_via_integrations_alias(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler("/api/v1/integrations/slack/status")
        result = handler.handle("/api/v1/integrations/slack/status", {}, http_handler)
        assert _status(result) == 200

    def test_status_via_integrations_alias_no_version(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        http_handler = _make_get_handler("/api/integrations/slack/status")
        result = handler.handle("/api/integrations/slack/status", {}, http_handler)
        assert _status(result) == 200


# ===========================================================================
# POST webhook method enforcement
# ===========================================================================


class TestMethodEnforcement:
    """Webhook endpoints require POST, reject GET/PUT/DELETE."""

    def test_get_on_commands_returns_405(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="GET",
        )
        result = handler.handle("/api/v1/bots/slack/commands", {}, http_handler)
        assert _status(result) == 405
        assert "POST" in _body(result).get("error", "")

    def test_get_on_events_returns_405(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/events",
            command="GET",
        )
        result = handler.handle("/api/v1/bots/slack/events", {}, http_handler)
        assert _status(result) == 405

    def test_get_on_interactions_returns_405(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/interactions",
            command="GET",
        )
        result = handler.handle("/api/v1/bots/slack/interactions", {}, http_handler)
        assert _status(result) == 405

    def test_put_on_commands_returns_405(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="PUT",
        )
        result = handler.handle("/api/v1/bots/slack/commands", {}, http_handler)
        assert _status(result) == 405

    def test_delete_on_commands_returns_405(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="DELETE",
        )
        result = handler.handle("/api/v1/bots/slack/commands", {}, http_handler)
        assert _status(result) == 405


# ===========================================================================
# Signing secret configuration
# ===========================================================================


class TestSigningSecretConfiguration:
    """Tests for signing secret requirement on webhook endpoints."""

    def test_missing_signing_secret_returns_503(self, handler_cls, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        h = handler_cls(server_context={})
        h._signing_secret = None
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/slack/commands",
            command="POST",
        )
        result = h.handle("/api/v1/bots/slack/commands", {}, http_handler)
        assert _status(result) == 503
        assert "signing secret" in _body(result).get("error", "").lower()


# ===========================================================================
# _verify_signature
# ===========================================================================


class TestVerifySignature:
    """Tests for the _verify_signature method."""

    def test_valid_signature_returns_true(self, handler):
        http_handler = _make_slack_handler(
            body={"text": "hello"},
            signing_secret="test-signing-secret",
        )
        assert handler._verify_signature(http_handler) is True

    def test_invalid_signature_returns_false(self, handler):
        http_handler = _make_slack_handler(
            body={"text": "hello"},
            signing_secret="test-signing-secret",
        )
        http_handler.headers["X-Slack-Signature"] = "v0=badbadbadbad"
        assert handler._verify_signature(http_handler) is False

    def test_no_signing_secret_dev_mode_allows(self, handler_cls, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")
        h = handler_cls(server_context={})
        h._signing_secret = None
        http_handler = _make_slack_handler(body={})
        assert h._verify_signature(http_handler) is True

    def test_no_signing_secret_test_mode_allows(self, handler_cls, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")
        h = handler_cls(server_context={})
        h._signing_secret = None
        http_handler = _make_slack_handler(body={})
        assert h._verify_signature(http_handler) is True

    def test_no_signing_secret_production_rejects(self, handler_cls, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        h = handler_cls(server_context={})
        h._signing_secret = None
        http_handler = _make_slack_handler(body={})
        assert h._verify_signature(http_handler) is False

    def test_reads_body_from_body_attr(self, handler):
        """Signature verification reads from handler._body when available."""
        ts = str(int(time.time()))
        body_bytes = b'{"test":"data"}'
        sig = _compute_slack_signature(body_bytes, ts, "test-signing-secret")
        http_handler = MagicMock()
        http_handler._body = body_bytes
        http_handler.headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
            "Content-Length": str(len(body_bytes)),
        }
        assert handler._verify_signature(http_handler) is True

    def test_reads_body_from_rfile(self, handler):
        """Signature verification reads from rfile when _body not available."""
        ts = str(int(time.time()))
        body_bytes = b'{"test":"rfile"}'
        sig = _compute_slack_signature(body_bytes, ts, "test-signing-secret")
        http_handler = MagicMock(spec=[])
        http_handler.headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
            "Content-Length": str(len(body_bytes)),
        }
        http_handler.rfile = io.BytesIO(body_bytes)
        # Remove _body to force rfile path
        assert not hasattr(http_handler, "_body")
        assert handler._verify_signature(http_handler) is True

    def test_signature_verification_error_returns_false(self, handler):
        """Errors during signature verification return False."""
        http_handler = MagicMock()
        http_handler.headers = MagicMock()
        http_handler.headers.get = MagicMock(side_effect=OSError("read error"))
        assert handler._verify_signature(http_handler) is False

    def test_empty_body_handler(self, handler):
        """Handler with no _body and no rfile uses empty bytes."""
        ts = str(int(time.time()))
        body_bytes = b""
        sig = _compute_slack_signature(body_bytes, ts, "test-signing-secret")
        http_handler = MagicMock(spec=[])
        http_handler.headers = {
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
            "Content-Length": "0",
        }
        # No _body, no rfile - should use empty bytes
        assert handler._verify_signature(http_handler) is True


# ===========================================================================
# POST /api/v1/bots/slack/commands (with signature verification)
# ===========================================================================


class TestCommandsEndpoint:
    """Tests for the commands webhook endpoint."""

    def test_valid_post_dispatches_to_handle_post(self, handler, monkeypatch):
        """A POST with valid signature reaches handle_post which calls handle_slack_commands."""
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = json.dumps({"response_type": "ephemeral", "text": "ok"}).encode()

        with patch(
            "aragora.server.handlers.bots.slack.handler.handle_slack_commands",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            http_handler = _make_slack_handler(
                body={"command": "/aragora", "text": "help"},
                signing_secret="test-signing-secret",
            )
            result = handler.handle("/api/v1/bots/slack/commands", {}, http_handler)
            # Result should come through (either from handle_post or via the inner route)
            assert result is not None

    def test_invalid_signature_returns_401(self, handler):
        """POST with invalid signature returns 401."""
        http_handler = _make_slack_handler(
            body={"command": "/aragora"},
            signing_secret="test-signing-secret",
        )
        # Corrupt the signature
        http_handler.headers["X-Slack-Signature"] = "v0=0000000000000000"
        result = handler.handle("/api/v1/bots/slack/commands", {}, http_handler)
        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()


# ===========================================================================
# POST /api/v1/bots/slack/events (with signature verification)
# ===========================================================================


class TestEventsEndpoint:
    """Tests for the events webhook endpoint."""

    def test_valid_post_dispatches_to_events(self, handler, monkeypatch):
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = json.dumps({"ok": True}).encode()

        with patch(
            "aragora.server.handlers.bots.slack.handler.handle_slack_events",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            http_handler = _make_slack_handler(
                body={"type": "event_callback", "event": {"type": "message"}},
                signing_secret="test-signing-secret",
            )
            result = handler.handle("/api/v1/bots/slack/events", {}, http_handler)
            assert result is not None

    def test_invalid_signature_returns_401(self, handler):
        http_handler = _make_slack_handler(
            body={"type": "event_callback"},
            signing_secret="test-signing-secret",
        )
        http_handler.headers["X-Slack-Signature"] = "v0=invalid"
        result = handler.handle("/api/v1/bots/slack/events", {}, http_handler)
        assert _status(result) == 401


# ===========================================================================
# POST /api/v1/bots/slack/interactions (with signature verification)
# ===========================================================================


class TestInteractionsEndpoint:
    """Tests for the interactions webhook endpoint."""

    def test_valid_post_dispatches_to_interactions(self, handler, monkeypatch):
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = json.dumps({"ok": True}).encode()

        with patch(
            "aragora.server.handlers.bots.slack.handler.handle_slack_interactions",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            http_handler = _make_slack_handler(
                body={"type": "block_actions"},
                signing_secret="test-signing-secret",
            )
            result = handler.handle("/api/v1/bots/slack/interactions", {}, http_handler)
            assert result is not None

    def test_invalid_signature_returns_401(self, handler):
        http_handler = _make_slack_handler(
            body={"type": "block_actions"},
            signing_secret="test-signing-secret",
        )
        http_handler.headers["X-Slack-Signature"] = "v0=bad"
        result = handler.handle("/api/v1/bots/slack/interactions", {}, http_handler)
        assert _status(result) == 401


# ===========================================================================
# Path normalization
# ===========================================================================


class TestPathNormalization:
    """Tests for path normalization: /api/integrations/slack/* -> /api/v1/bots/slack/*."""

    def test_integrations_commands_normalizes(self, handler, monkeypatch):
        """Commands path via /api/integrations/slack/ should normalize and work."""
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = json.dumps({"ok": True}).encode()

        with patch(
            "aragora.server.handlers.bots.slack.handler.handle_slack_commands",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            http_handler = _make_slack_handler(
                body={"command": "/aragora"},
                path="/api/integrations/slack/commands",
                signing_secret="test-signing-secret",
            )
            result = handler.handle("/api/integrations/slack/commands", {}, http_handler)
            assert result is not None

    def test_v1_integrations_commands_normalizes(self, handler, monkeypatch):
        """Commands path via /api/v1/integrations/slack/ should normalize and work."""
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = json.dumps({"ok": True}).encode()

        with patch(
            "aragora.server.handlers.bots.slack.handler.handle_slack_commands",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            http_handler = _make_slack_handler(
                body={"command": "/aragora"},
                path="/api/v1/integrations/slack/commands",
                signing_secret="test-signing-secret",
            )
            result = handler.handle(
                "/api/v1/integrations/slack/commands", {}, http_handler
            )
            assert result is not None


# ===========================================================================
# handle() returns None for unrecognized paths
# ===========================================================================


class TestHandleReturnsNone:
    """The handle method returns None for paths it doesn't recognize."""

    def test_unknown_subpath_returns_none(self, handler):
        http_handler = _make_get_handler("/api/v1/bots/slack/unknown")
        result = handler.handle("/api/v1/bots/slack/unknown", {}, http_handler)
        assert result is None

    def test_unrelated_path_returns_none(self, handler):
        http_handler = _make_get_handler("/api/v1/debates")
        result = handler.handle("/api/v1/debates", {}, http_handler)
        assert result is None


# ===========================================================================
# _is_bot_enabled()
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled() checking Slack configuration."""

    def test_enabled_with_bot_token(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", None)
        assert handler._is_bot_enabled() is True

    def test_enabled_with_signing_secret(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        assert handler._is_bot_enabled() is True

    def test_enabled_with_both(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        assert handler._is_bot_enabled() is True

    def test_disabled_without_either(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", None)
        assert handler._is_bot_enabled() is False

    def test_disabled_with_empty_strings(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "")
        assert handler._is_bot_enabled() is False


# ===========================================================================
# _get_status() direct method
# ===========================================================================


class TestGetStatus:
    """Tests for the _get_status() method (bypasses RBAC/async wrapper)."""

    def test_get_status_returns_enabled(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        result = handler._get_status()
        body = _body(result)
        assert body["enabled"] is True

    def test_get_status_returns_features(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        result = handler._get_status()
        body = _body(result)
        assert body["features"]["slash_commands"] is True
        assert body["features"]["events_api"] is True

    def test_get_status_active_debates_count(self, handler, slack_pkg, monkeypatch):
        from aragora.server.handlers.bots.slack.state import _active_debates

        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        _active_debates["d1"] = {"topic": "test"}
        try:
            result = handler._get_status()
            body = _body(result)
            assert body["active_debates"] >= 1
        finally:
            _active_debates.pop("d1", None)


# ===========================================================================
# _command_help()
# ===========================================================================


class TestCommandHelp:
    """Tests for the _command_help() method."""

    def test_help_returns_ephemeral(self, handler):
        result = handler._command_help()
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_help_contains_commands(self, handler):
        result = handler._command_help()
        body = _body(result)
        text = body["text"]
        assert "/aragora debate" in text
        assert "/aragora status" in text
        assert "/aragora help" in text
        assert "/aragora agents" in text
        assert "/aragora vote" in text

    def test_help_contains_examples(self, handler):
        result = handler._command_help()
        body = _body(result)
        assert "Examples" in body["text"]

    def test_help_returns_200(self, handler):
        result = handler._command_help()
        assert _status(result) == 200


# ===========================================================================
# _command_status()
# ===========================================================================


class TestCommandStatus:
    """Tests for the _command_status() method."""

    def test_status_returns_ephemeral(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
            result = handler._command_status()
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_status_shows_active_debates(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
            result = handler._command_status()
        body = _body(result)
        assert "Active debates" in body["text"]

    def test_status_with_elo_store(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        mock_store = MagicMock()
        mock_rating = MagicMock()
        mock_rating.agent_name = "claude"
        mock_rating.elo = 1500.0
        mock_store.get_all_ratings.return_value = [mock_rating]

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_store,
        ):
            result = handler._command_status()
        body = _body(result)
        assert "Registered agents: 1" in body["text"]

    def test_status_import_error(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError("no module"),
        ):
            result = handler._command_status()
        body = _body(result)
        assert "Registered agents: 0" in body["text"]
        assert "ImportError" in body["text"]

    def test_status_runtime_error(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=RuntimeError("db unavailable"),
        ):
            result = handler._command_status()
        body = _body(result)
        assert "Registered agents: 0" in body["text"]

    def test_status_unexpected_error(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=TypeError("bad type"),
        ):
            result = handler._command_status()
        body = _body(result)
        assert "Registered agents: 0" in body["text"]
        assert "unexpected error" in body["text"]

    def test_status_configured_text(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", "secret")
        with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
            result = handler._command_status()
        body = _body(result)
        assert "Configured" in body["text"]

    def test_status_not_configured_text(self, handler, slack_pkg, monkeypatch):
        monkeypatch.setattr(slack_pkg, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(slack_pkg, "SLACK_SIGNING_SECRET", None)
        with patch("aragora.ranking.elo.get_elo_store", side_effect=ImportError):
            result = handler._command_status()
        body = _body(result)
        assert "Not configured" in body["text"]


# ===========================================================================
# _command_agents()
# ===========================================================================


class TestCommandAgents:
    """Tests for the _command_agents() method."""

    def test_agents_no_ratings(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = []
        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_store,
        ):
            result = handler._command_agents()
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert "No agents registered" in body["text"]

    def test_agents_sorted_by_elo(self, handler):
        mock_store = MagicMock()
        r1 = MagicMock()
        r1.agent_name = "claude"
        r1.elo = 1600.0
        r2 = MagicMock()
        r2.agent_name = "gpt4"
        r2.elo = 1500.0
        r3 = MagicMock()
        r3.agent_name = "gemini"
        r3.elo = 1550.0
        mock_store.get_all_ratings.return_value = [r1, r2, r3]

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_store,
        ):
            result = handler._command_agents()
        body = _body(result)
        text = body["text"]
        # Claude (1600) should appear before Gemini (1550) before GPT-4 (1500)
        claude_pos = text.index("claude")
        gemini_pos = text.index("gemini")
        gpt4_pos = text.index("gpt4")
        assert claude_pos < gemini_pos < gpt4_pos

    def test_agents_import_error(self, handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError("no module"),
        ):
            result = handler._command_agents()
        body = _body(result)
        assert "Error fetching agents" in body["text"]

    def test_agents_runtime_error(self, handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=RuntimeError("db fail"),
        ):
            result = handler._command_agents()
        body = _body(result)
        assert "Error fetching agents" in body["text"]

    def test_agents_unexpected_error(self, handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=TypeError("bad type"),
        ):
            result = handler._command_agents()
        body = _body(result)
        assert "unexpected error" in body["text"].lower()

    def test_agents_value_error(self, handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ValueError("bad value"),
        ):
            result = handler._command_agents()
        body = _body(result)
        assert "unexpected error" in body["text"].lower()

    def test_agents_limits_to_top_10(self, handler):
        mock_store = MagicMock()
        ratings = []
        for i in range(15):
            r = MagicMock()
            r.agent_name = f"agent{i}"
            r.elo = 1500.0 + i * 10
            ratings.append(r)
        mock_store.get_all_ratings.return_value = ratings

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_store,
        ):
            result = handler._command_agents()
        body = _body(result)
        # Should show top 10, not all 15
        text = body["text"]
        assert "agent14" in text  # Highest ELO (index 14 = 1640)
        assert "agent5" in text   # 10th highest (index 5 = 1550)
        # agent0-4 should not be in top 10
        assert "agent0:" not in text


# ===========================================================================
# _slack_response() and _slack_blocks_response()
# ===========================================================================


class TestSlackResponseFormatting:
    """Tests for the Slack response helper methods."""

    def test_slack_response_in_channel(self, handler):
        result = handler._slack_response("Hello world")
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert body["text"] == "Hello world"

    def test_slack_response_ephemeral(self, handler):
        result = handler._slack_response("Secret message", response_type="ephemeral")
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert body["text"] == "Secret message"

    def test_slack_blocks_response_basic(self, handler):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Block 1"}},
        ]
        result = handler._slack_blocks_response(blocks)
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert body["blocks"] == blocks
        assert "text" not in body

    def test_slack_blocks_response_with_text(self, handler):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]
        result = handler._slack_blocks_response(blocks, text="Fallback text")
        body = _body(result)
        assert body["text"] == "Fallback text"
        assert body["blocks"] == blocks

    def test_slack_blocks_response_ephemeral(self, handler):
        blocks = [{"type": "divider"}]
        result = handler._slack_blocks_response(
            blocks, response_type="ephemeral"
        )
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_slack_blocks_empty_text_not_included(self, handler):
        blocks = [{"type": "divider"}]
        result = handler._slack_blocks_response(blocks, text="")
        body = _body(result)
        assert "text" not in body


# ===========================================================================
# _check_permission_or_admin()
# ===========================================================================


class TestCheckPermissionOrAdmin:
    """Tests for the _check_permission_or_admin RBAC helper."""

    def test_returns_none_when_rbac_unavailable(self, handler, monkeypatch):
        """When RBAC is not available and not fail-closed, returns None (allow)."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.rbac_fail_closed",
            lambda: False,
        )
        result = handler._check_permission_or_admin("T123", "U123", "slack.test")
        assert result is None

    def test_returns_503_when_rbac_unavailable_and_fail_closed(self, handler, monkeypatch):
        """When RBAC is not available and fail-closed, returns 503."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.rbac_fail_closed",
            lambda: True,
        )
        result = handler._check_permission_or_admin("T123", "U123", "slack.test")
        assert _status(result) == 503

    def test_admin_permission_bypasses_specific_check(self, handler, monkeypatch):
        """Admin permission allows access without checking specific permission."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.RBAC_AVAILABLE", True
        )
        mock_check = MagicMock(return_value=mock_decision)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.check_permission",
            mock_check,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.build_auth_context_from_slack",
            MagicMock(return_value=MagicMock()),
        )
        result = handler._check_permission_or_admin("T123", "U123", "slack.test")
        assert result is None

    def test_null_context_returns_none(self, handler, monkeypatch):
        """When context cannot be built, returns None."""
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.RBAC_AVAILABLE", True
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.check_permission",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.build_auth_context_from_slack",
            MagicMock(return_value=None),
        )
        result = handler._check_permission_or_admin("T123", "U123", "slack.test")
        assert result is None


# ===========================================================================
# RBAC helper delegation methods
# ===========================================================================


class TestRBACHelpers:
    """Tests for RBAC helper delegation methods."""

    def test_build_auth_context_delegates(self, handler, monkeypatch):
        mock_fn = MagicMock(return_value="ctx")
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.build_auth_context_from_slack",
            mock_fn,
        )
        result = handler._build_auth_context_from_slack("T1", "U1", "C1")
        mock_fn.assert_called_once_with("T1", "U1", "C1")
        assert result == "ctx"

    def test_get_org_from_team_delegates(self, handler, monkeypatch):
        mock_fn = MagicMock(return_value="org-123")
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.get_org_from_team",
            mock_fn,
        )
        result = handler._get_org_from_team("T1")
        mock_fn.assert_called_once_with("T1")
        assert result == "org-123"

    def test_get_user_roles_delegates(self, handler, monkeypatch):
        mock_fn = MagicMock(return_value={"admin", "user"})
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.get_user_roles_from_slack",
            mock_fn,
        )
        result = handler._get_user_roles_from_slack("T1", "U1")
        mock_fn.assert_called_once_with("T1", "U1")
        assert result == {"admin", "user"}

    def test_check_workspace_authorized_delegates(self, handler, monkeypatch):
        mock_fn = MagicMock(return_value=(True, None))
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.check_workspace_authorized",
            mock_fn,
        )
        result = handler._check_workspace_authorized("T1")
        mock_fn.assert_called_once_with("T1")
        assert result == (True, None)

    def test_check_permission_delegates(self, handler, monkeypatch):
        mock_fn = MagicMock(return_value=None)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.check_user_permission",
            mock_fn,
        )
        result = handler._check_permission("T1", "U1", "slack.read", "C1")
        mock_fn.assert_called_once_with("T1", "U1", "slack.read", "C1")
        assert result is None


# ===========================================================================
# handle_post() routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() async method's path dispatching."""

    @pytest.mark.asyncio
    async def test_handle_post_events(self, handler, monkeypatch):
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = b'{"ok":true}'

        mock_events = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.handle_slack_events",
            mock_events,
        )
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = _make_slack_handler(
            body={"type": "event_callback"},
            signing_secret="test-signing-secret",
        )
        result = await handler.handle_post(
            "/api/v1/bots/slack/events", {}, http_handler
        )
        mock_events.assert_called_once_with(http_handler)

    @pytest.mark.asyncio
    async def test_handle_post_interactions(self, handler, monkeypatch):
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = b'{"ok":true}'

        mock_interactions = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.handle_slack_interactions",
            mock_interactions,
        )
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = _make_slack_handler(
            body={"type": "block_actions"},
            signing_secret="test-signing-secret",
        )
        result = await handler.handle_post(
            "/api/v1/bots/slack/interactions", {}, http_handler
        )
        mock_interactions.assert_called_once_with(http_handler)

    @pytest.mark.asyncio
    async def test_handle_post_commands(self, handler, monkeypatch):
        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.body = b'{"ok":true}'

        mock_commands = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "aragora.server.handlers.bots.slack.handler.handle_slack_commands",
            mock_commands,
        )
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = _make_slack_handler(
            body={"command": "/aragora", "text": "help"},
            signing_secret="test-signing-secret",
        )
        result = await handler.handle_post(
            "/api/v1/bots/slack/commands", {}, http_handler
        )
        mock_commands.assert_called_once_with(http_handler)

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path_returns_none(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = _make_slack_handler(
            body={},
            signing_secret="test-signing-secret",
        )
        result = await handler.handle_post(
            "/api/v1/bots/slack/unknown", {}, http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_missing_signing_secret_returns_503(
        self, handler, monkeypatch
    ):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        http_handler = _make_slack_handler(body={})
        result = await handler.handle_post(
            "/api/v1/bots/slack/events", {}, http_handler
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_handle_post_signature_verification_error(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
        http_handler = _make_slack_handler(
            body={"type": "event_callback"},
            signing_secret="test-signing-secret",
        )
        # Make headers.get raise an error to simulate verification failure
        http_handler.headers = MagicMock()
        http_handler.headers.get = MagicMock(side_effect=AttributeError("boom"))
        result = await handler.handle_post(
            "/api/v1/bots/slack/events", {}, http_handler
        )
        assert _status(result) == 401


# ===========================================================================
# register_slack_routes()
# ===========================================================================


class TestRegisterSlackRoutes:
    """Tests for the deprecated register_slack_routes() function."""

    def test_registers_three_routes(self, handler_module):
        mock_router = MagicMock()
        handler_module.register_slack_routes(mock_router)
        assert mock_router.add_route.call_count == 3

    def test_registers_events_route(self, handler_module):
        mock_router = MagicMock()
        handler_module.register_slack_routes(mock_router)
        calls = mock_router.add_route.call_args_list
        paths = [call[0][1] for call in calls]
        methods = [call[0][0] for call in calls]
        assert "/api/bots/slack/events" in paths
        idx = paths.index("/api/bots/slack/events")
        assert methods[idx] == "POST"

    def test_registers_interactions_route(self, handler_module):
        mock_router = MagicMock()
        handler_module.register_slack_routes(mock_router)
        calls = mock_router.add_route.call_args_list
        paths = [call[0][1] for call in calls]
        assert "/api/bots/slack/interactions" in paths

    def test_registers_commands_route(self, handler_module):
        mock_router = MagicMock()
        handler_module.register_slack_routes(mock_router)
        calls = mock_router.add_route.call_args_list
        paths = [call[0][1] for call in calls]
        assert "/api/bots/slack/commands" in paths


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for SlackHandler initialization."""

    def test_init_caches_signing_secret(self, handler_cls, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "my-secret")
        h = handler_cls(server_context={})
        assert h._signing_secret == "my-secret"

    def test_init_no_signing_secret(self, handler_cls, monkeypatch):
        monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
        h = handler_cls(server_context={})
        assert h._signing_secret is None

    def test_bot_platform_is_slack(self, handler):
        assert handler.bot_platform == "slack"

    def test_init_with_server_context(self, handler_cls, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")
        ctx = {"storage": MagicMock(), "elo": MagicMock()}
        h = handler_cls(server_context=ctx)
        assert h.ctx is ctx


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_slack_handler(self, handler_module):
        assert "SlackHandler" in handler_module.__all__

    def test_exports_register_slack_routes(self, handler_module):
        assert "register_slack_routes" in handler_module.__all__

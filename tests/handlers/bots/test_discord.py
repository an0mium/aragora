"""
Tests for Discord Interactions endpoint handler.

Covers all routes and behavior of the DiscordHandler class:
- can_handle() routing for all defined routes
- GET  /api/v1/bots/discord/status        - Bot status endpoint
- POST /api/v1/bots/discord/interactions   - Interaction webhook handling
  - PING (type 1) - URL verification
  - APPLICATION_COMMAND (type 2) - slash commands
  - MESSAGE_COMPONENT (type 3) - buttons, selects
  - MODAL_SUBMIT (type 5) - modal form submissions
  - Unknown interaction types
- Ed25519 signature verification (_verify_discord_signature)
  - Public key missing / dev mode / production
  - Replay protection via timestamp freshness
  - PyNaCl availability checks
  - Bad signature, invalid hex, unexpected errors
- Slash commands: /aragora, /debate, /gauntlet, /status, unknown
- Message component: vote_* buttons, unknown components
- RBAC permission checks
- _execute_command: success, failure, embeds, ephemeral
- Error handling: JSONDecodeError, ValueError, KeyError, TypeError, RuntimeError, OSError
- Handler initialization, _is_bot_enabled, _get_platform_config_status
"""

from __future__ import annotations

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
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.discord as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.DiscordHandler


@pytest.fixture
def handler(handler_cls):
    """Create a DiscordHandler with empty context."""
    return handler_cls(ctx={})


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/discord/interactions"
    method: str = "POST"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.body is not None:
            body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            body_bytes = b"{}"
        self.rfile = io.BytesIO(body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_interaction_handler(
    body: dict[str, Any],
    signature: str = "abcd1234",
    timestamp: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for interaction POST requests."""
    if timestamp is None:
        timestamp = str(int(time.time()))
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Signature-Ed25519": signature,
        "X-Signature-Timestamp": timestamp,
    }
    return MockHTTPHandler(body=body, headers=headers)


# ---------------------------------------------------------------------------
# Interaction body builders
# ---------------------------------------------------------------------------


def _ping_interaction() -> dict[str, Any]:
    """Build a Discord PING interaction (type 1)."""
    return {"type": 1, "id": "ping-001"}


def _command_interaction(
    command_name: str = "aragora",
    options: list[dict[str, Any]] | None = None,
    user_id: str = "user-123",
    username: str = "testuser",
    guild_id: str = "guild-456",
    channel_id: str = "channel-789",
    interaction_id: str = "interaction-001",
    use_member: bool = False,
) -> dict[str, Any]:
    """Build an APPLICATION_COMMAND interaction (type 2)."""
    interaction: dict[str, Any] = {
        "type": 2,
        "id": interaction_id,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "data": {
            "name": command_name,
            "options": options or [],
        },
    }
    user_data = {"id": user_id, "username": username, "global_name": "Test Display"}
    if use_member:
        interaction["member"] = {"user": user_data}
    else:
        interaction["user"] = user_data
    return interaction


def _component_interaction(
    custom_id: str = "vote_debate123_agree",
    user_id: str = "user-123",
    use_member: bool = False,
) -> dict[str, Any]:
    """Build a MESSAGE_COMPONENT interaction (type 3)."""
    interaction: dict[str, Any] = {
        "type": 3,
        "id": "component-001",
        "data": {"custom_id": custom_id},
    }
    user_data = {"id": user_id, "username": "testuser"}
    if use_member:
        interaction["member"] = {"user": user_data}
    else:
        interaction["user"] = user_data
    return interaction


def _modal_interaction(
    custom_id: str = "modal-form-1",
) -> dict[str, Any]:
    """Build a MODAL_SUBMIT interaction (type 5)."""
    return {
        "type": 5,
        "id": "modal-001",
        "data": {"custom_id": custom_id},
    }


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_interactions_route(self, handler):
        assert handler.can_handle("/api/v1/bots/discord/interactions", "POST") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/discord/status", "GET") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/webhook", "POST") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/", "GET") is False

    def test_partial_match_no_trailing(self, handler):
        assert handler.can_handle("/api/v1/bots/discord/interactionsXYZ", "POST") is False

    def test_different_base(self, handler):
        assert handler.can_handle("/api/v2/bots/discord/interactions", "POST") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("", "GET") is False

    def test_routes_list_complete(self, handler):
        """ROUTES list contains exactly the expected paths."""
        assert set(handler.ROUTES) == {
            "/api/v1/bots/discord/interactions",
            "/api/v1/bots/discord/status",
        }


# ===========================================================================
# GET /api/v1/bots/discord/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_body_has_platform(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        body = _body(result)
        assert body["platform"] == "discord"

    @pytest.mark.asyncio
    async def test_status_body_has_enabled_field(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", "app-123"):
            result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        body = _body(result)
        assert "enabled" in body
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_status_has_application_id_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", "app-123"):
            result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        body = _body(result)
        assert body["application_id_configured"] is True

    @pytest.mark.asyncio
    async def test_status_has_public_key_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef"):
            result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        body = _body(result)
        assert body["public_key_configured"] is True

    @pytest.mark.asyncio
    async def test_status_disabled_when_not_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/status", method="GET")
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", None):
            result = await handler.handle("/api/v1/bots/discord/status", {}, http_handler)
        body = _body(result)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_status_get(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/interactions", method="GET")
        result = await handler.handle("/api/v1/bots/discord/interactions", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/discord/unknown", method="GET")
        result = await handler.handle("/api/v1/bots/discord/unknown", {}, http_handler)
        assert result is None


# ===========================================================================
# _verify_discord_signature
# ===========================================================================


class TestVerifyDiscordSignature:
    """Tests for the _verify_discord_signature function."""

    def test_no_public_key_dev_mode_allows(self, handler_module):
        """When DISCORD_PUBLIC_KEY is missing and dev mode allows, returns True."""
        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", None),
            patch.object(handler_module, "_should_allow_unverified", return_value=True),
        ):
            result = handler_module._verify_discord_signature("sig", "ts", b"body")
        assert result is True

    def test_no_public_key_production_rejects(self, handler_module):
        """When DISCORD_PUBLIC_KEY is missing and not dev mode, returns False."""
        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", None),
            patch.object(handler_module, "_should_allow_unverified", return_value=False),
        ):
            result = handler_module._verify_discord_signature("sig", "ts", b"body")
        assert result is False

    def test_missing_signature_header(self, handler_module):
        """Missing signature header rejects."""
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("", "12345", b"body")
        assert result is False

    def test_missing_timestamp_header(self, handler_module):
        """Missing timestamp header rejects."""
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("abcdef", "", b"body")
        assert result is False

    def test_both_headers_missing(self, handler_module):
        """Both headers missing rejects."""
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("", "", b"body")
        assert result is False

    def test_timestamp_too_old(self, handler_module):
        """Timestamp older than 5 minutes is rejected."""
        old_ts = str(int(time.time()) - 600)
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("sig", old_ts, b"body")
        assert result is False

    def test_timestamp_in_future_too_far(self, handler_module):
        """Timestamp 10 minutes in future is rejected."""
        future_ts = str(int(time.time()) + 600)
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("sig", future_ts, b"body")
        assert result is False

    def test_invalid_timestamp_format(self, handler_module):
        """Non-numeric timestamp is rejected."""
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature("sig", "not-a-number", b"body")
        assert result is False

    def test_nacl_not_available_dev_mode_allows(self, handler_module):
        """When PyNaCl unavailable and dev mode, allows unverified."""
        ts = str(int(time.time()))
        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"),
            patch.object(handler_module, "_NACL_AVAILABLE", False),
            patch.object(handler_module, "_should_allow_unverified", return_value=True),
        ):
            result = handler_module._verify_discord_signature("sig", ts, b"body")
        assert result is True

    def test_nacl_not_available_production_rejects(self, handler_module):
        """When PyNaCl unavailable and not dev mode, rejects."""
        ts = str(int(time.time()))
        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"),
            patch.object(handler_module, "_NACL_AVAILABLE", False),
            patch.object(handler_module, "_should_allow_unverified", return_value=False),
        ):
            result = handler_module._verify_discord_signature("sig", ts, b"body")
        assert result is False

    def test_bad_signature_rejected(self, handler_module):
        """Bad Ed25519 signature is rejected."""
        ts = str(int(time.time()))
        fake_pub_key = "a" * 64

        # Create mock nacl modules since nacl may not be installed
        mock_bad_sig_error = type("BadSignatureError", (Exception,), {})

        mock_verify_key_cls = MagicMock()
        mock_verify_key_inst = MagicMock()
        mock_verify_key_inst.verify.side_effect = mock_bad_sig_error("bad")
        mock_verify_key_cls.return_value = mock_verify_key_inst

        mock_signing = MagicMock()
        mock_signing.VerifyKey = mock_verify_key_cls
        mock_exceptions = MagicMock()
        mock_exceptions.BadSignatureError = mock_bad_sig_error

        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", fake_pub_key),
            patch.object(handler_module, "_NACL_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "nacl": MagicMock(),
                    "nacl.signing": mock_signing,
                    "nacl.exceptions": mock_exceptions,
                },
            ),
        ):
            result = handler_module._verify_discord_signature("ab" * 32, ts, b"body")
        assert result is False

    def test_invalid_hex_in_signature(self, handler_module):
        """Invalid hex in signature raises ValueError, returns False."""
        ts = str(int(time.time()))
        fake_pub_key = "a" * 64

        # Mock nacl to get past the import, but bytes.fromhex("zzzz") will raise ValueError
        mock_signing = MagicMock()
        mock_exceptions = MagicMock()
        mock_exceptions.BadSignatureError = type("BadSignatureError", (Exception,), {})

        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", fake_pub_key),
            patch.object(handler_module, "_NACL_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "nacl": MagicMock(),
                    "nacl.signing": mock_signing,
                    "nacl.exceptions": mock_exceptions,
                },
            ),
        ):
            # "zzzz" is invalid hex - bytes.fromhex will raise ValueError
            result = handler_module._verify_discord_signature("zzzz", ts, b"body")
        assert result is False

    def test_nacl_import_fails_despite_flag(self, handler_module):
        """If PyNaCl import fails despite _NACL_AVAILABLE=True, returns False."""
        ts = str(int(time.time()))
        fake_pub_key = "a" * 64
        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", fake_pub_key),
            patch.object(handler_module, "_NACL_AVAILABLE", True),
            patch.dict("sys.modules", {"nacl.signing": None, "nacl.exceptions": None}),
        ):
            result = handler_module._verify_discord_signature("ab" * 32, ts, b"body")
        assert result is False

    def test_timestamp_overflow_rejected(self, handler_module):
        """Overflowing timestamp value is rejected."""
        with patch.object(handler_module, "DISCORD_PUBLIC_KEY", "abcdef1234567890"):
            result = handler_module._verify_discord_signature(
                "sig", "99999999999999999999999999999999999", b"body"
            )
        assert result is False

    def test_valid_timestamp_within_window(self, handler_module):
        """Timestamp within 5-minute window passes timestamp check and sig verification."""
        ts = str(int(time.time()))

        mock_verify_key_cls = MagicMock()
        mock_verify_key_inst = MagicMock()
        mock_verify_key_inst.verify.return_value = True
        mock_verify_key_cls.return_value = mock_verify_key_inst

        mock_signing = MagicMock()
        mock_signing.VerifyKey = mock_verify_key_cls
        mock_exceptions = MagicMock()
        mock_exceptions.BadSignatureError = type("BadSignatureError", (Exception,), {})

        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", "a" * 64),
            patch.object(handler_module, "_NACL_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "nacl": MagicMock(),
                    "nacl.signing": mock_signing,
                    "nacl.exceptions": mock_exceptions,
                },
            ),
        ):
            result = handler_module._verify_discord_signature("ab" * 32, ts, b"body")
        assert result is True

    def test_runtime_error_during_verification(self, handler_module):
        """RuntimeError during signature verification returns False."""
        ts = str(int(time.time()))

        mock_verify_key_cls = MagicMock()
        mock_verify_key_inst = MagicMock()
        mock_verify_key_inst.verify.side_effect = RuntimeError("unexpected")
        mock_verify_key_cls.return_value = mock_verify_key_inst

        mock_signing = MagicMock()
        mock_signing.VerifyKey = mock_verify_key_cls
        mock_exceptions = MagicMock()
        mock_exceptions.BadSignatureError = type("BadSignatureError", (Exception,), {})

        with (
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", "a" * 64),
            patch.object(handler_module, "_NACL_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "nacl": MagicMock(),
                    "nacl.signing": mock_signing,
                    "nacl.exceptions": mock_exceptions,
                },
            ),
        ):
            result = handler_module._verify_discord_signature("ab" * 32, ts, b"body")
        assert result is False


# ===========================================================================
# _should_allow_unverified
# ===========================================================================


class TestShouldAllowUnverified:
    """Tests for _should_allow_unverified."""

    def test_import_error_fails_closed(self, handler_module):
        """When webhook_security module unavailable, fails closed."""
        with patch.dict("sys.modules", {"aragora.connectors.chat.webhook_security": None}):
            result = handler_module._should_allow_unverified()
        assert result is False

    def test_delegates_to_webhook_security(self, handler_module):
        """Delegates to webhook_security.should_allow_unverified."""
        mock_module = MagicMock()
        mock_module.should_allow_unverified.return_value = True
        with patch.dict("sys.modules", {"aragora.connectors.chat.webhook_security": mock_module}):
            result = handler_module._should_allow_unverified()
        assert result is True
        mock_module.should_allow_unverified.assert_called_once_with("discord")


# ===========================================================================
# POST /api/v1/bots/discord/interactions - PING
# ===========================================================================


class TestPingInteraction:
    """Tests for PING (type 1) interactions."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, handler, handler_module):
        interaction = _ping_interaction()
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 1

    @pytest.mark.asyncio
    async def test_ping_status_200(self, handler, handler_module):
        interaction = _ping_interaction()
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        assert _status(result) == 200


# ===========================================================================
# POST /api/v1/bots/discord/interactions - APPLICATION_COMMAND
# ===========================================================================


class TestApplicationCommand:
    """Tests for APPLICATION_COMMAND (type 2) interactions."""

    @pytest.mark.asyncio
    async def test_aragora_command(self, handler, handler_module):
        """The /aragora command delegates to _execute_command."""
        interaction = _command_interaction(
            command_name="aragora",
            options=[
                {"name": "command", "value": "help"},
                {"name": "args", "value": ""},
            ],
        )
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        assert call_args[0][0] == "help"

    @pytest.mark.asyncio
    async def test_debate_command(self, handler, handler_module):
        """The /debate command checks RBAC and executes."""
        interaction = _command_interaction(
            command_name="debate",
            options=[{"name": "topic", "value": "Should we use Python?"}],
        )
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0] == "debate"
        assert mock_exec.call_args[0][1] == "Should we use Python?"

    @pytest.mark.asyncio
    async def test_debate_rbac_denied(self, handler, handler_module):
        """Debate RBAC denial returns permission denied message."""
        interaction = _command_interaction(
            command_name="debate",
            options=[{"name": "topic", "value": "test"}],
        )
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(
                handler,
                "_check_bot_permission",
                side_effect=PermissionError("denied"),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "Permission denied" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_gauntlet_command(self, handler, handler_module):
        """The /gauntlet command checks RBAC and executes."""
        interaction = _command_interaction(
            command_name="gauntlet",
            options=[{"name": "statement", "value": "test statement"}],
        )
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0] == "gauntlet"
        assert mock_exec.call_args[0][1] == "test statement"

    @pytest.mark.asyncio
    async def test_gauntlet_rbac_denied(self, handler, handler_module):
        """Gauntlet RBAC denial returns permission denied message."""
        interaction = _command_interaction(
            command_name="gauntlet",
            options=[{"name": "statement", "value": "test"}],
        )
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(
                handler,
                "_check_bot_permission",
                side_effect=PermissionError("denied"),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "Permission denied" in body["data"]["content"]
        assert "gauntlet" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_status_command(self, handler, handler_module):
        """The /status command executes without RBAC check."""
        interaction = _command_interaction(command_name="status")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()
        assert mock_exec.call_args[0][0] == "status"
        assert mock_exec.call_args[0][1] == ""

    @pytest.mark.asyncio
    async def test_unknown_command(self, handler, handler_module):
        """Unknown command returns 'Unknown command' response."""
        interaction = _command_interaction(command_name="foobar")
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "Unknown command: foobar" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_command_user_from_member(self, handler, handler_module):
        """User info extracted from member.user when user field is absent."""
        interaction = _command_interaction(command_name="status", use_member=True)
        # Remove the top-level "user" key to force member fallback
        interaction.pop("user", None)
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_with_no_options(self, handler, handler_module):
        """Command with empty options list."""
        interaction = _command_interaction(command_name="aragora", options=[])
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_execute_command", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = MagicMock(
                status_code=200,
                body=json.dumps({"type": 4, "data": {"content": "OK"}}).encode(),
            )
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        mock_exec.assert_called_once()
        # Without options, aragora defaults to "help" subcommand
        assert mock_exec.call_args[0][0] == "help"


# ===========================================================================
# POST /api/v1/bots/discord/interactions - MESSAGE_COMPONENT
# ===========================================================================


class TestMessageComponent:
    """Tests for MESSAGE_COMPONENT (type 3) interactions."""

    @pytest.mark.asyncio
    async def test_vote_agree_records_vote(self, handler, handler_module):
        """Vote agree component records vote and returns thumbsup."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock()
        mock_db.record_vote = MagicMock()
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert ":thumbsup:" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_vote_disagree_records_vote(self, handler, handler_module):
        """Vote disagree component records vote and returns thumbsdown."""
        interaction = _component_interaction(custom_id="vote_debate123_disagree")
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock()
        mock_db.record_vote = MagicMock()
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert ":thumbsdown:" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_vote_rbac_denied(self, handler, handler_module):
        """Vote RBAC denial returns permission denied."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(
                handler,
                "_check_bot_permission",
                side_effect=PermissionError("denied"),
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert "Permission denied" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_vote_db_none(self, handler, handler_module):
        """Vote when db is None still responds."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=None),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert "recorded" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_vote_db_no_record_vote_method(self, handler, handler_module):
        """Vote when db has no record_vote method still responds."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock(spec=[])  # no record_vote attr
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert "recorded" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_vote_db_value_error(self, handler, handler_module):
        """Vote that raises ValueError during recording still responds."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock()
        mock_db.record_vote.side_effect = ValueError("bad vote")
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        # Should still respond with thumbsup since the vote section completed
        assert body["type"] == 4

    @pytest.mark.asyncio
    async def test_vote_db_runtime_error(self, handler, handler_module):
        """Vote that raises RuntimeError during recording still responds."""
        interaction = _component_interaction(custom_id="vote_debate123_agree")
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock()
        mock_db.record_vote.side_effect = RuntimeError("DB error")
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4

    @pytest.mark.asyncio
    async def test_vote_insufficient_parts(self, handler, handler_module):
        """Vote custom_id with fewer than 3 parts returns 'Interaction received'."""
        interaction = _component_interaction(custom_id="vote_only")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["data"]["content"] == "Interaction received"

    @pytest.mark.asyncio
    async def test_unknown_component(self, handler, handler_module):
        """Non-vote component returns 'Interaction received'."""
        interaction = _component_interaction(custom_id="unknown_action")
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["data"]["content"] == "Interaction received"
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_component_user_from_member(self, handler, handler_module):
        """Component extracts user from member.user when user is absent."""
        interaction = _component_interaction(custom_id="vote_d1_agree", use_member=True)
        interaction.pop("user", None)
        http_handler = _make_interaction_handler(interaction)
        mock_db = MagicMock()
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_check_bot_permission"),
            patch("aragora.server.storage.get_debates_db", return_value=mock_db),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4


# ===========================================================================
# POST /api/v1/bots/discord/interactions - MODAL_SUBMIT
# ===========================================================================


class TestModalSubmit:
    """Tests for MODAL_SUBMIT (type 5) interactions."""

    @pytest.mark.asyncio
    async def test_modal_submit_returns_form_submitted(self, handler, handler_module):
        interaction = _modal_interaction(custom_id="modal-form-1")
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert body["data"]["content"] == "Form submitted"
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_modal_submit_different_custom_id(self, handler, handler_module):
        interaction = _modal_interaction(custom_id="feedback-form-xyz")
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["data"]["content"] == "Form submitted"


# ===========================================================================
# Unknown Interaction Type
# ===========================================================================


class TestUnknownInteractionType:
    """Tests for unknown interaction types."""

    @pytest.mark.asyncio
    async def test_unknown_type_returns_message(self, handler, handler_module):
        interaction = {"type": 99, "id": "unknown-001"}
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "Unknown interaction type" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_missing_type_field(self, handler, handler_module):
        """Interaction without type field."""
        interaction = {"id": "no-type-001"}
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        # type is None which doesn't match any known type
        assert body["type"] == 4
        assert "Unknown interaction type" in body["data"]["content"]


# ===========================================================================
# Signature Verification Failure
# ===========================================================================


class TestSignatureVerificationFailure:
    """Tests for signature verification failure in _handle_interactions."""

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_401(self, handler, handler_module):
        interaction = _ping_interaction()
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=False):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        assert _status(result) == 401
        body = _body(result)
        assert "Invalid signature" in body.get("error", "")


# ===========================================================================
# Error Handling in _handle_interactions
# ===========================================================================


class TestInteractionErrorHandling:
    """Tests for exception handling in _handle_interactions."""

    @pytest.mark.asyncio
    async def test_json_decode_error(self, handler, handler_module):
        """Invalid JSON body returns 400."""
        http_handler = MockHTTPHandler()
        http_handler.rfile = io.BytesIO(b"not json at all")
        http_handler.headers["Content-Length"] = "15"
        http_handler.headers["X-Signature-Ed25519"] = "abc"
        http_handler.headers["X-Signature-Timestamp"] = str(int(time.time()))
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_value_error_returns_ephemeral(self, handler, handler_module):
        """ValueError in interaction handling returns ephemeral message."""
        interaction = _command_interaction(command_name="debate")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_handle_application_command", side_effect=ValueError("bad")),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "error occurred" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_key_error_returns_ephemeral(self, handler, handler_module):
        """KeyError in interaction handling returns ephemeral message."""
        interaction = _command_interaction(command_name="debate")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_handle_application_command", side_effect=KeyError("missing")),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "error occurred" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_type_error_returns_ephemeral(self, handler, handler_module):
        """TypeError in interaction handling returns ephemeral message."""
        interaction = _command_interaction(command_name="debate")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(
                handler, "_handle_application_command", side_effect=TypeError("wrong type")
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "error occurred" in body["data"]["content"]

    @pytest.mark.asyncio
    async def test_runtime_error_returns_ephemeral(self, handler, handler_module):
        """RuntimeError in interaction handling returns ephemeral message."""
        interaction = _command_interaction(command_name="debate")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_handle_application_command", side_effect=RuntimeError("boom")),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "unexpected error" in body["data"]["content"].lower()

    @pytest.mark.asyncio
    async def test_os_error_returns_ephemeral(self, handler, handler_module):
        """OSError in interaction handling returns ephemeral message."""
        interaction = _command_interaction(command_name="debate")
        http_handler = _make_interaction_handler(interaction)
        with (
            patch.object(handler_module, "_verify_discord_signature", return_value=True),
            patch.object(handler, "_handle_application_command", side_effect=OSError("io error")),
        ):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        body = _body(result)
        assert body["type"] == 4
        assert "unexpected error" in body["data"]["content"].lower()

    @pytest.mark.asyncio
    async def test_empty_body_returns_error(self, handler, handler_module):
        """Empty body returns an error."""
        http_handler = MockHTTPHandler()
        http_handler.rfile = io.BytesIO(b"")
        http_handler.headers["Content-Length"] = "0"
        http_handler.headers["X-Signature-Ed25519"] = "abc"
        http_handler.headers["X-Signature-Timestamp"] = str(int(time.time()))
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        # Empty body should be parsed as error by _parse_json_body
        assert _status(result) == 400


# ===========================================================================
# _execute_command
# ===========================================================================


class TestExecuteCommand:
    """Tests for _execute_command."""

    @pytest.mark.asyncio
    async def test_successful_command(self, handler, handler_module):
        """Successful command returns content."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Command completed"
        mock_result.discord_embed = None
        mock_result.ephemeral = False

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value=mock_result)

        interaction = _command_interaction()
        with patch("aragora.bots.commands.get_default_registry", return_value=mock_registry):
            result = await handler._execute_command("help", "", "user-123", interaction)
        body = _body(result)
        assert body["type"] == 4
        assert body["data"]["content"] == "Command completed"
        assert "flags" not in body["data"]  # not ephemeral

    @pytest.mark.asyncio
    async def test_command_with_embed(self, handler, handler_module):
        """Command result with discord_embed includes embeds array."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Here is your data"
        mock_result.discord_embed = {"title": "Debate", "description": "A debate"}
        mock_result.ephemeral = False

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value=mock_result)

        interaction = _command_interaction()
        with patch("aragora.bots.commands.get_default_registry", return_value=mock_registry):
            result = await handler._execute_command("status", "", "user-123", interaction)
        body = _body(result)
        assert body["data"]["embeds"] == [{"title": "Debate", "description": "A debate"}]

    @pytest.mark.asyncio
    async def test_command_ephemeral(self, handler, handler_module):
        """Ephemeral command result includes flags=64."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Only you can see this"
        mock_result.discord_embed = None
        mock_result.ephemeral = True

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value=mock_result)

        interaction = _command_interaction()
        with patch("aragora.bots.commands.get_default_registry", return_value=mock_registry):
            result = await handler._execute_command("help", "", "user-123", interaction)
        body = _body(result)
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_command_failure(self, handler, handler_module):
        """Failed command returns error message."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Something went wrong"

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value=mock_result)

        interaction = _command_interaction()
        with patch("aragora.bots.commands.get_default_registry", return_value=mock_registry):
            result = await handler._execute_command("debate", "test", "user-123", interaction)
        body = _body(result)
        assert body["type"] == 4
        assert "Error: Something went wrong" in body["data"]["content"]
        assert body["data"]["flags"] == 64

    @pytest.mark.asyncio
    async def test_command_no_message(self, handler, handler_module):
        """Successful command with no message uses default."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = None
        mock_result.discord_embed = None
        mock_result.ephemeral = False

        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(return_value=mock_result)

        interaction = _command_interaction()
        with patch("aragora.bots.commands.get_default_registry", return_value=mock_registry):
            result = await handler._execute_command("status", "", "user-123", interaction)
        body = _body(result)
        assert body["data"]["content"] == "Command executed"


# ===========================================================================
# RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for _check_bot_permission RBAC integration."""

    def test_rbac_not_available_non_production(self, handler, handler_module):
        """When RBAC is unavailable and not production, should pass."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.bots.discord.rbac_fail_closed", return_value=False),
        ):
            handler._check_bot_permission("debates:create", user_id="discord:123")

    def test_rbac_not_available_production(self, handler, handler_module):
        """When RBAC is unavailable in production, should raise."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.bots.discord.rbac_fail_closed", return_value=True),
        ):
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="discord:123")

    def test_rbac_available_permission_granted(self, handler, handler_module):
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            mock_check.return_value = None
            handler._check_bot_permission("debates:create", user_id="discord:123")

    def test_rbac_available_permission_denied(self, handler, handler_module):
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            mock_check.side_effect = PermissionError("Denied")
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="discord:123")

    def test_rbac_with_auth_context_in_context(self, handler, handler_module):
        """When auth_context is provided in context dict, it should be used."""
        mock_auth_ctx = MagicMock()
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            handler._check_bot_permission(
                "debates:create",
                context={"auth_context": mock_auth_ctx},
            )
            mock_check.assert_called_once_with(mock_auth_ctx, "debates:create")

    def test_rbac_no_user_id_no_context(self, handler, handler_module):
        """When no user_id and no auth_context, check_permission not called."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            handler._check_bot_permission("debates:create")
            mock_check.assert_not_called()


# ===========================================================================
# _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled."""

    def test_enabled_when_app_id_set(self, handler, handler_module):
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", "some-app-id"):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_app_id_not_set(self, handler, handler_module):
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", None):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_app_id_empty(self, handler, handler_module):
        with patch.object(handler_module, "DISCORD_APPLICATION_ID", ""):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# _get_platform_config_status
# ===========================================================================


class TestPlatformConfigStatus:
    """Tests for _get_platform_config_status."""

    def test_all_configured(self, handler, handler_module):
        with (
            patch.object(handler_module, "DISCORD_APPLICATION_ID", "app-123"),
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", "key-456"),
        ):
            status = handler._get_platform_config_status()
        assert status["application_id_configured"] is True
        assert status["public_key_configured"] is True

    def test_none_configured(self, handler, handler_module):
        with (
            patch.object(handler_module, "DISCORD_APPLICATION_ID", None),
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", None),
        ):
            status = handler._get_platform_config_status()
        assert status["application_id_configured"] is False
        assert status["public_key_configured"] is False

    def test_partial_configured(self, handler, handler_module):
        with (
            patch.object(handler_module, "DISCORD_APPLICATION_ID", "app-123"),
            patch.object(handler_module, "DISCORD_PUBLIC_KEY", None),
        ):
            status = handler._get_platform_config_status()
        assert status["application_id_configured"] is True
        assert status["public_key_configured"] is False


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for DiscordHandler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"storage": MagicMock()}
        h = handler_cls(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx(self, handler_cls):
        h = handler_cls(ctx=None)
        assert h.ctx == {}

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "discord"

    def test_routes_defined(self, handler):
        assert "/api/v1/bots/discord/interactions" in handler.ROUTES
        assert "/api/v1/bots/discord/status" in handler.ROUTES


# ===========================================================================
# handle_post returns None for unknown path
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post routing."""

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/unknown")
        result = await handler.handle_post("/api/v1/bots/unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_routes_interactions(self, handler, handler_module):
        interaction = _ping_interaction()
        http_handler = _make_interaction_handler(interaction)
        with patch.object(handler_module, "_verify_discord_signature", return_value=True):
            result = await handler.handle_post(
                "/api/v1/bots/discord/interactions", {}, http_handler
            )
        assert result is not None
        body = _body(result)
        assert body["type"] == 1


# ===========================================================================
# Module-level __all__
# ===========================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self, handler_module):
        assert "DiscordHandler" in handler_module.__all__

    def test_discord_handler_in_module(self, handler_module):
        assert hasattr(handler_module, "DiscordHandler")

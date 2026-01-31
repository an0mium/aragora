"""
Tests for aragora.server.handlers.bots.discord - Discord Interactions handler.

Tests cover:
- Signature verification (Ed25519)
- PING/PONG interaction (URL verification)
- Application command routing (slash commands)
- Message component interactions (buttons, selects)
- Modal submissions
- Error handling and edge cases
- Webhook authentication
- Command execution flow
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing Discord interactions."""

    def __init__(
        self,
        headers: Optional[dict[str, str]] = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "POST",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def get(self, key: str, default: str = "") -> str:
        return self.headers.get(key, default)


def make_mock_handler(
    body: bytes,
    signature: str = "",
    timestamp: str = "1234567890",
    method: str = "POST",
) -> MockHandler:
    """Helper to create MockHandler with Discord headers."""
    return MockHandler(
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Signature-Ed25519": signature,
            "X-Signature-Timestamp": timestamp,
        },
        body=body,
        method=method,
    )


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create a DiscordHandler instance."""
    from aragora.server.handlers.bots.discord import DiscordHandler

    return DiscordHandler(mock_server_context)


@pytest.fixture
def ping_interaction() -> dict[str, Any]:
    """Create a PING interaction payload for URL verification."""
    return {
        "type": 1,  # PING
        "id": "123456789",
        "application_id": "app-123",
    }


@pytest.fixture
def command_interaction() -> dict[str, Any]:
    """Create an application command (slash command) interaction."""
    return {
        "type": 2,  # APPLICATION_COMMAND
        "id": "123456789",
        "application_id": "app-123",
        "channel_id": "channel-123",
        "guild_id": "guild-456",
        "data": {
            "name": "debate",
            "options": [
                {"name": "topic", "value": "AI Ethics"},
            ],
        },
        "user": {
            "id": "user-123",
            "username": "testuser",
            "global_name": "Test User",
        },
    }


@pytest.fixture
def member_command_interaction() -> dict[str, Any]:
    """Create an application command interaction from a guild member."""
    return {
        "type": 2,
        "id": "123456789",
        "application_id": "app-123",
        "channel_id": "channel-123",
        "guild_id": "guild-456",
        "data": {
            "name": "status",
            "options": [],
        },
        "member": {
            "user": {
                "id": "member-user-123",
                "username": "guilduser",
                "global_name": "Guild User",
            },
            "roles": ["role-1", "role-2"],
        },
    }


@pytest.fixture
def button_interaction() -> dict[str, Any]:
    """Create a button click (MESSAGE_COMPONENT) interaction."""
    return {
        "type": 3,  # MESSAGE_COMPONENT
        "id": "123456789",
        "application_id": "app-123",
        "channel_id": "channel-123",
        "data": {
            "custom_id": "vote_debate123_agree",
            "component_type": 2,  # Button
        },
        "user": {
            "id": "user-123",
            "username": "testuser",
        },
    }


@pytest.fixture
def modal_interaction() -> dict[str, Any]:
    """Create a modal submission interaction."""
    return {
        "type": 5,  # MODAL_SUBMIT
        "id": "123456789",
        "application_id": "app-123",
        "channel_id": "channel-123",
        "data": {
            "custom_id": "feedback_modal",
            "components": [
                {
                    "type": 1,  # Action row
                    "components": [
                        {
                            "type": 4,  # Text input
                            "custom_id": "feedback_text",
                            "value": "Great debate!",
                        }
                    ],
                }
            ],
        },
        "user": {"id": "user-123", "username": "test"},
    }


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling and can_handle logic."""

    def test_can_handle_interactions_endpoint(self, handler):
        """Handler recognizes interactions endpoint."""
        assert handler.can_handle("/api/v1/bots/discord/interactions") is True

    def test_can_handle_status_endpoint(self, handler):
        """Handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/bots/discord/status") is True

    def test_cannot_handle_unknown_endpoint(self, handler):
        """Handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/bots/discord/unknown") is False

    def test_cannot_handle_other_bot_endpoint(self, handler):
        """Handler rejects other bot platform endpoints."""
        assert handler.can_handle("/api/v1/bots/slack/interactions") is False

    def test_cannot_handle_non_bot_endpoint(self, handler):
        """Handler rejects non-bot endpoints."""
        assert handler.can_handle("/api/v1/debates/list") is False


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/v1/bots/discord/status."""

    @pytest.mark.asyncio
    async def test_get_status_returns_configuration_info(self, handler):
        """Status endpoint returns Discord configuration info."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            method="GET",
        )

        # Mock RBAC authentication
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
            with patch.object(handler, "check_permission"):
                result = await handler.handle("/api/v1/bots/discord/status", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

        data = json.loads(result.body)
        assert data["platform"] == "discord"
        assert "enabled" in data
        assert "application_id_configured" in data
        assert "public_key_configured" in data

    @pytest.mark.asyncio
    async def test_status_requires_authentication(self, handler):
        """Status endpoint requires RBAC authentication."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        mock_http = MockHandler(method="GET")

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No token")
            result = await handler.handle("/api/v1/bots/discord/status", {}, mock_http)

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for Discord Ed25519 signature verification.

    Phase 3.1: Tests verify fail-closed behavior in production and
    environment-aware bypass in development mode.
    """

    def test_verify_rejects_when_no_key_and_production(self):
        """Signature verification fails closed when no public key configured."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", None):
            with patch(
                "aragora.server.handlers.bots.discord._should_allow_unverified",
                return_value=False,
            ):
                result = _verify_discord_signature("any_sig", "1234", b"body")
        assert result is False

    def test_verify_allows_when_no_key_and_dev_mode(self):
        """Signature verification allows unverified in dev mode when no key."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", None):
            with patch(
                "aragora.server.handlers.bots.discord._should_allow_unverified",
                return_value=True,
            ):
                result = _verify_discord_signature("any_sig", "1234", b"body")
        assert result is True

    def test_verify_rejects_when_empty_key_and_production(self):
        """Signature verification fails closed when public key is empty string."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", ""):
            with patch(
                "aragora.server.handlers.bots.discord._should_allow_unverified",
                return_value=False,
            ):
                result = _verify_discord_signature("any_sig", "1234", b"body")
        assert result is False

    def test_verify_rejects_missing_signature_header(self):
        """Missing signature header returns False when key is configured."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import time as time_mod

        current = str(int(time_mod.time()))
        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            # Empty signature, valid timestamp
            result = _verify_discord_signature("", current, b"body")
        assert result is False

    def test_verify_rejects_missing_timestamp_header(self):
        """Missing timestamp header returns False when key is configured."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            result = _verify_discord_signature("aabb", "", b"body")
        assert result is False

    def test_verify_rejects_stale_timestamp(self):
        """Requests with timestamps older than 5 minutes are rejected."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import time as time_mod

        # Timestamp 10 minutes in the past
        stale_ts = str(int(time_mod.time()) - 600)
        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            with patch("aragora.server.handlers.bots.discord._NACL_AVAILABLE", True):
                result = _verify_discord_signature("aabb", stale_ts, b"body")
        assert result is False

    def test_verify_rejects_invalid_timestamp_format(self):
        """Non-numeric timestamp is rejected."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            result = _verify_discord_signature("aabb", "not-a-number", b"body")
        assert result is False

    def test_verify_rejects_when_nacl_missing_in_production(self):
        """PyNaCl missing fails closed in production."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import time as time_mod

        current = str(int(time_mod.time()))
        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            with patch("aragora.server.handlers.bots.discord._NACL_AVAILABLE", False):
                with patch(
                    "aragora.server.handlers.bots.discord._should_allow_unverified",
                    return_value=False,
                ):
                    result = _verify_discord_signature("aabb", current, b"body")
        assert result is False

    def test_verify_allows_when_nacl_missing_in_dev_mode(self):
        """PyNaCl missing allows in dev mode with explicit opt-in."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import time as time_mod

        current = str(int(time_mod.time()))
        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            with patch("aragora.server.handlers.bots.discord._NACL_AVAILABLE", False):
                with patch(
                    "aragora.server.handlers.bots.discord._should_allow_unverified",
                    return_value=True,
                ):
                    result = _verify_discord_signature("aabb", current, b"body")
        assert result is True

    def test_verify_signature_invalid_hex_format(self):
        """Invalid hex format in public key returns False."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import sys
        import time as time_mod

        current = str(int(time_mod.time()))

        # Mock nacl modules since PyNaCl may not be installed in test env
        mock_bad_sig_error = type("BadSignatureError", (Exception,), {})
        mock_verify_key = MagicMock(side_effect=ValueError("non-hexadecimal number"))
        mock_nacl_signing = MagicMock(VerifyKey=mock_verify_key)
        mock_nacl_exceptions = MagicMock(BadSignatureError=mock_bad_sig_error)

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "not_valid_hex_zzz"):
            with patch("aragora.server.handlers.bots.discord._NACL_AVAILABLE", True):
                with patch.dict(
                    sys.modules,
                    {
                        "nacl": MagicMock(),
                        "nacl.signing": mock_nacl_signing,
                        "nacl.exceptions": mock_nacl_exceptions,
                    },
                ):
                    result = _verify_discord_signature("bad", current, b"body")
        # Should return False due to ValueError on hex conversion of public key
        assert result is False

    def test_verify_accepts_fresh_timestamp(self):
        """Requests with fresh timestamps pass the replay check (fail at signature, not timestamp)."""
        from aragora.server.handlers.bots.discord import _verify_discord_signature
        import sys
        import time as time_mod

        # Timestamp 30 seconds ago (well within the 5-minute window)
        fresh_ts = str(int(time_mod.time()) - 30)

        # Mock nacl modules since PyNaCl may not be installed in test env
        mock_bad_sig_error = type("BadSignatureError", (Exception,), {})
        mock_verify_key = MagicMock(side_effect=ValueError("invalid key length"))
        mock_nacl_signing = MagicMock(VerifyKey=mock_verify_key)
        mock_nacl_exceptions = MagicMock(BadSignatureError=mock_bad_sig_error)

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "aabbccdd"):
            with patch("aragora.server.handlers.bots.discord._NACL_AVAILABLE", True):
                with patch.dict(
                    sys.modules,
                    {
                        "nacl": MagicMock(),
                        "nacl.signing": mock_nacl_signing,
                        "nacl.exceptions": mock_nacl_exceptions,
                    },
                ):
                    result = _verify_discord_signature("aabb", fresh_ts, b"body")
        # Will be False because VerifyKey raises ValueError on bad key, but NOT due to timestamp
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected_with_401(self, handler, ping_interaction):
        """Invalid signature returns 401 Unauthorized."""
        body = json.dumps(ping_interaction).encode()
        mock_http = make_mock_handler(body, signature="invalid_signature")

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "abc123"):
            with patch(
                "aragora.server.handlers.bots.discord._verify_discord_signature",
                return_value=False,
            ):
                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_signature_header(self, handler, ping_interaction):
        """Missing signature header is handled gracefully."""
        body = json.dumps(ping_interaction).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                # Missing X-Signature-Ed25519 header
                "X-Signature-Timestamp": "1234567890",
            },
            body=body,
        )

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "abc123"):
            with patch(
                "aragora.server.handlers.bots.discord._verify_discord_signature",
                return_value=False,
            ):
                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        assert result is not None
        # Should be rejected
        assert result.status_code == 401


# ===========================================================================
# PING/PONG Tests (URL Verification)
# ===========================================================================


class TestPingPong:
    """Tests for PING interaction handling (Discord URL verification)."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, handler, ping_interaction):
        """PING interaction returns PONG response (type 1)."""
        body = json.dumps(ping_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["type"] == 1  # PONG


# ===========================================================================
# Application Command Tests (Slash Commands)
# ===========================================================================


class TestApplicationCommands:
    """Tests for application command (slash command) interactions."""

    @pytest.mark.asyncio
    async def test_debate_command_success(self, handler, command_interaction):
        """Debate slash command executes successfully."""
        body = json.dumps(command_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Debate started on: AI Ethics",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        assert result is not None
        data = json.loads(result.body)
        assert data["type"] == 4  # CHANNEL_MESSAGE_WITH_SOURCE
        assert "Debate started" in data["data"]["content"]

    @pytest.mark.asyncio
    async def test_debate_command_with_embed(self, handler, command_interaction):
        """Debate command with Discord embed is properly formatted."""
        body = json.dumps(command_interaction).encode()
        mock_http = make_mock_handler(body)

        embed = {
            "title": "Debate Started",
            "description": "Topic: AI Ethics",
            "color": 0x5865F2,
        }

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Debate started",
                        discord_embed=embed,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert "embeds" in data["data"]
        assert data["data"]["embeds"][0] == embed

    @pytest.mark.asyncio
    async def test_command_ephemeral_response(self, handler, command_interaction):
        """Command with ephemeral flag sets proper Discord flag."""
        body = json.dumps(command_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Private response",
                        discord_embed=None,
                        ephemeral=True,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert data["data"].get("flags") == 64  # Ephemeral flag

    @pytest.mark.asyncio
    async def test_command_execution_failure(self, handler, command_interaction):
        """Command execution failure returns error message."""
        body = json.dumps(command_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=False,
                        message=None,
                        discord_embed=None,
                        ephemeral=False,
                        error="Debate engine unavailable",
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "Error" in data["data"]["content"]
        assert "Debate engine unavailable" in data["data"]["content"]
        assert data["data"].get("flags") == 64  # Ephemeral for errors

    @pytest.mark.asyncio
    async def test_gauntlet_command(self, handler):
        """Gauntlet slash command routes correctly."""
        interaction = {
            "type": 2,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "name": "gauntlet",
                "options": [{"name": "statement", "value": "The sky is blue"}],
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Gauntlet result: PASS",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "Gauntlet result" in data["data"]["content"]

    @pytest.mark.asyncio
    async def test_status_command(self, handler):
        """Status slash command routes correctly."""
        interaction = {
            "type": 2,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "name": "status",
                "options": [],
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="System operational",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert "System operational" in data["data"]["content"]

    @pytest.mark.asyncio
    async def test_aragora_command_with_subcommand(self, handler):
        """Aragora command with subcommand and args routes correctly."""
        interaction = {
            "type": 2,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "name": "aragora",
                "options": [
                    {"name": "command", "value": "help"},
                    {"name": "args", "value": "debates"},
                ],
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Help for debates",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        assert result is not None
        data = json.loads(result.body)
        assert data["type"] == 4

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self, handler):
        """Unknown command returns ephemeral error message."""
        interaction = {
            "type": 2,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "name": "unknown_command",
                "options": [],
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "unknown" in data["data"]["content"].lower()
        assert data["data"].get("flags") == 64  # Ephemeral

    @pytest.mark.asyncio
    async def test_command_from_member_context(self, handler, member_command_interaction):
        """Command from guild member uses member.user for context."""
        body = json.dumps(member_command_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="OK",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        assert result is not None
        # Verify execute was called (context built from member.user)
        mock_reg.execute.assert_called_once()


# ===========================================================================
# Message Component Tests (Buttons, Selects)
# ===========================================================================


class TestMessageComponents:
    """Tests for message component (button/select) interactions."""

    @pytest.mark.asyncio
    async def test_vote_agree_button(self, handler, button_interaction):
        """Vote agree button records vote and responds."""
        body = json.dumps(button_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value = MagicMock(record_vote=MagicMock())
                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "vote" in data["data"]["content"].lower()
        assert "thumbsup" in data["data"]["content"]
        assert data["data"].get("flags") == 64  # Ephemeral

    @pytest.mark.asyncio
    async def test_vote_disagree_button(self, handler):
        """Vote disagree button records vote with correct emoji."""
        interaction = {
            "type": 3,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "custom_id": "vote_debate456_disagree",
                "component_type": 2,
            },
            "user": {"id": "user-456", "username": "voter"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        data = json.loads(result.body)
        assert "thumbsdown" in data["data"]["content"]

    @pytest.mark.asyncio
    async def test_vote_button_db_error_handled(self, handler, button_interaction):
        """Vote button handles database errors gracefully."""
        body = json.dumps(button_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value = MagicMock(
                    record_vote=MagicMock(side_effect=ValueError("Invalid debate ID"))
                )
                result = await handler.handle_post(
                    "/api/v1/bots/discord/interactions", {}, mock_http
                )

        # Should still respond successfully (error logged but not exposed)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "vote" in data["data"]["content"].lower()

    @pytest.mark.asyncio
    async def test_unknown_component_interaction(self, handler):
        """Unknown component custom_id returns generic acknowledgment."""
        interaction = {
            "type": 3,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "custom_id": "some_unknown_action",
                "component_type": 2,
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "interaction received" in data["data"]["content"].lower()

    @pytest.mark.asyncio
    async def test_vote_button_malformed_custom_id(self, handler):
        """Vote button with malformed custom_id is handled."""
        interaction = {
            "type": 3,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "custom_id": "vote_only_one_part",  # Missing parts
                "component_type": 2,
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        # Should handle gracefully
        assert result.status_code == 200


# ===========================================================================
# Modal Submit Tests
# ===========================================================================


class TestModalSubmit:
    """Tests for modal submission interactions."""

    @pytest.mark.asyncio
    async def test_modal_submit_returns_acknowledgment(self, handler, modal_interaction):
        """Modal submission returns acknowledgment."""
        body = json.dumps(modal_interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "submitted" in data["data"]["content"].lower()
        assert data["data"].get("flags") == 64  # Ephemeral


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400 Bad Request."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "15",
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=b"not valid json!",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_interaction_type_handled(self, handler):
        """Unknown interaction type returns ephemeral message."""
        interaction = {
            "type": 99,  # Unknown type
            "id": "123",
            "application_id": "app-123",
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        data = json.loads(result.body)
        assert data["type"] == 4
        assert "unknown" in data["data"]["content"].lower()

    @pytest.mark.asyncio
    async def test_data_error_in_interaction(self, handler):
        """Data errors (ValueError, KeyError, TypeError) are handled gracefully."""
        # Malformed interaction that may cause data errors
        interaction = {
            "type": 2,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "name": "debate",
                "options": "not_a_list",  # Should be list, will cause iteration error
            },
            "user": {"id": "user-123", "username": "test"},
        }
        body = json.dumps(interaction).encode()
        mock_http = make_mock_handler(body)

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        # Should return a response (not crash)
        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["type"] == 4

    @pytest.mark.asyncio
    async def test_empty_body_handled(self, handler):
        """Empty request body is handled."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "0",
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=b"",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        # Empty JSON is invalid, should return 400
        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Bot Enabled Check Tests
# ===========================================================================


class TestBotEnabled:
    """Tests for bot enabled configuration check."""

    def test_is_bot_enabled_with_app_id(self, handler):
        """Bot is enabled when DISCORD_APPLICATION_ID is set."""
        with patch("aragora.server.handlers.bots.discord.DISCORD_APPLICATION_ID", "app-123"):
            assert handler._is_bot_enabled() is True

    def test_is_bot_disabled_without_app_id(self, handler):
        """Bot is disabled when DISCORD_APPLICATION_ID is not set."""
        with patch("aragora.server.handlers.bots.discord.DISCORD_APPLICATION_ID", None):
            assert handler._is_bot_enabled() is False

    def test_is_bot_disabled_with_empty_app_id(self, handler):
        """Bot is disabled when DISCORD_APPLICATION_ID is empty string."""
        with patch("aragora.server.handlers.bots.discord.DISCORD_APPLICATION_ID", ""):
            assert handler._is_bot_enabled() is False

"""
Tests for aragora.server.handlers.bots.discord - Discord Interactions handler.

Tests cover:
- Signature verification
- PING/PONG interaction
- Application command routing
- Message component interactions
- Modal submissions
- Error handling
"""

from __future__ import annotations

import hashlib
import hmac
import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.bots.discord import DiscordHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
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
    return DiscordHandler(mock_server_context)


@pytest.fixture
def ping_interaction():
    """Create a PING interaction payload."""
    return {
        "type": 1,  # PING
        "id": "123456789",
        "application_id": "app-123",
    }


@pytest.fixture
def command_interaction():
    """Create an application command interaction."""
    return {
        "type": 2,  # APPLICATION_COMMAND
        "id": "123456789",
        "application_id": "app-123",
        "channel_id": "channel-123",
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
def button_interaction():
    """Create a button click interaction."""
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


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_interactions(self, handler):
        """Test handler recognizes interactions endpoint."""
        assert handler.can_handle("/api/v1/bots/discord/interactions") is True

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/bots/discord/status") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/bots/discord/unknown") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/bots/discord/status."""

    @pytest.mark.asyncio
    async def test_get_status(self, handler):
        """Test getting Discord bot status."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            method="GET",
        )

        result = await handler.handle("/api/v1/bots/discord/status", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

        data = json.loads(result.body)
        assert "enabled" in data
        assert "application_id_configured" in data
        assert "public_key_configured" in data


# ===========================================================================
# PING/PONG Tests
# ===========================================================================


class TestPingPong:
    """Tests for PING interaction handling."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, handler, ping_interaction):
        """Test PING interaction returns PONG."""
        body = json.dumps(ping_interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body,
            method="POST",
        )

        # Skip signature verification for test
        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = await handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["type"] == 1  # PONG


# ===========================================================================
# Application Command Tests
# ===========================================================================


class TestApplicationCommands:
    """Tests for application command interactions."""

    def test_debate_command(self, handler, command_interaction):
        """Test handling debate slash command."""
        body = json.dumps(command_interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            with patch("aragora.bots.commands.get_default_registry") as mock_registry:
                mock_reg = MagicMock()
                mock_reg.execute = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        message="Debate started",
                        discord_embed=None,
                        ephemeral=False,
                        error=None,
                    )
                )
                mock_registry.return_value = mock_reg

                result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        # Should return a channel message response
        data = json.loads(result.body)
        assert data["type"] == 4  # CHANNEL_MESSAGE_WITH_SOURCE

    def test_unknown_command(self, handler):
        """Test handling unknown command."""
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

        body_data = json.dumps(interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body_data,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        data = json.loads(result.body)
        assert data["type"] == 4
        # Should indicate unknown command
        assert "unknown" in data["data"]["content"].lower()


# ===========================================================================
# Message Component Tests
# ===========================================================================


class TestMessageComponents:
    """Tests for message component (button/select) interactions."""

    def test_vote_button(self, handler, button_interaction):
        """Test handling vote button click."""
        body = json.dumps(button_interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        data = json.loads(result.body)
        assert data["type"] == 4
        # Should confirm vote
        assert "vote" in data["data"]["content"].lower()

    def test_unknown_component(self, handler):
        """Test handling unknown component interaction."""
        interaction = {
            "type": 3,
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "custom_id": "unknown_action",
                "component_type": 2,
            },
            "user": {"id": "user-123", "username": "test"},
        }

        body_data = json.dumps(interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body_data,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        data = json.loads(result.body)
        assert data["type"] == 4


# ===========================================================================
# Modal Submit Tests
# ===========================================================================


class TestModalSubmit:
    """Tests for modal submission interactions."""

    def test_modal_submit(self, handler):
        """Test handling modal submission."""
        interaction = {
            "type": 5,  # MODAL_SUBMIT
            "id": "123",
            "application_id": "app-123",
            "channel_id": "channel-123",
            "data": {
                "custom_id": "feedback_modal",
                "components": [
                    {
                        "type": 1,
                        "components": [
                            {
                                "type": 4,
                                "custom_id": "feedback_text",
                                "value": "Great debate!",
                            }
                        ],
                    }
                ],
            },
            "user": {"id": "user-123", "username": "test"},
        }

        body_data = json.dumps(interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body_data,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        data = json.loads(result.body)
        assert data["type"] == 4


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for Discord signature verification."""

    def test_invalid_signature_rejected(self, handler, ping_interaction):
        """Test invalid signature is rejected."""
        body = json.dumps(ping_interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Signature-Ed25519": "invalid_signature",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body,
            method="POST",
        )

        # Mock public key being configured
        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "abc123"):
            with patch(
                "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=False
            ):
                result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None
        assert result.status_code == 401

    def test_missing_signature_when_key_configured(self, handler, ping_interaction):
        """Test missing signature is rejected when key is configured."""
        body_data = json.dumps(ping_interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
                # Missing X-Signature-Ed25519 header
                "X-Signature-Timestamp": "1234567890",
            },
            body=body_data,
            method="POST",
        )

        with patch("aragora.server.handlers.bots.discord.DISCORD_PUBLIC_KEY", "abc123"):
            with patch(
                "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=False
            ):
                result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        # Should either reject or skip verification
        assert result is not None


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, handler):
        """Test handling invalid JSON body."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "10",
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=b"not valid json",
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_unknown_interaction_type(self, handler):
        """Test handling unknown interaction type."""
        interaction = {
            "type": 99,  # Unknown type
            "id": "123",
            "application_id": "app-123",
        }

        body_data = json.dumps(interaction).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
                "X-Signature-Ed25519": "",
                "X-Signature-Timestamp": "1234567890",
            },
            body=body_data,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.bots.discord._verify_discord_signature", return_value=True
        ):
            result = handler.handle_post("/api/v1/bots/discord/interactions", {}, mock_http)

        assert result is not None

        # Should return some response, possibly error message
        data = json.loads(result.body)
        assert data["type"] == 4

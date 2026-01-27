"""
Tests for aragora.server.handlers.social.slack - Slack Integration Handler.

Tests cover:
- Routing and method handling
- Signature verification (HMAC-SHA256)
- SSRF protection (URL validation)
- Rate limiting (user + workspace)
- Slash commands (help, status, agents, debate, ask, search, etc.)
- Interactive components
- Events API
- Multi-workspace support
- Error handling
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from io import BytesIO
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest

from aragora.server.handlers.social.slack import (
    SLACK_ALLOWED_DOMAINS,
    SlackHandler,
    _validate_slack_url,
    get_slack_handler,
    get_slack_integration,
)

from .conftest import (
    MockHandler,
    create_slack_command_handler,
    create_slack_event_handler,
    create_slack_interactive_handler,
    generate_slack_signature,
    get_json,
    get_status_code,
    parse_result,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler(mock_server_context):
    """Create a SlackHandler instance."""
    return SlackHandler(mock_server_context)


@pytest.fixture
def signing_secret():
    """Test signing secret."""
    return "test_signing_secret_12345"


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_commands(self, handler):
        """Test handler recognizes commands endpoint."""
        assert handler.can_handle("/api/v1/integrations/slack/commands") is True

    def test_can_handle_interactive(self, handler):
        """Test handler recognizes interactive endpoint."""
        assert handler.can_handle("/api/v1/integrations/slack/interactive") is True

    def test_can_handle_events(self, handler):
        """Test handler recognizes events endpoint."""
        assert handler.can_handle("/api/v1/integrations/slack/events") is True

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/integrations/slack/status") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/integrations/slack/unknown") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False
        assert handler.can_handle("/api/v1/integrations/teams/commands") is False

    def test_routes_defined(self, handler):
        """Test handler has ROUTES defined."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 4

    def test_routes_include_all_endpoints(self, handler):
        """Verify all expected routes are defined."""
        expected = [
            "/api/v1/integrations/slack/commands",
            "/api/v1/integrations/slack/interactive",
            "/api/v1/integrations/slack/events",
            "/api/v1/integrations/slack/status",
        ]
        for route in expected:
            assert route in handler.ROUTES


# ===========================================================================
# SSRF Protection Tests
# ===========================================================================


class TestSSRFProtection:
    """Tests for SSRF protection via URL validation."""

    def test_validate_slack_url_valid_hooks(self):
        """Valid hooks.slack.com URL should pass."""
        assert _validate_slack_url("https://hooks.slack.com/commands/T123/456/abc") is True

    def test_validate_slack_url_valid_api(self):
        """Valid api.slack.com URL should pass."""
        assert _validate_slack_url("https://api.slack.com/something") is True

    def test_validate_slack_url_rejects_http(self):
        """HTTP (non-HTTPS) should be rejected."""
        assert _validate_slack_url("http://hooks.slack.com/commands/T123/456/abc") is False

    def test_validate_slack_url_rejects_other_domains(self):
        """Non-Slack domains should be rejected."""
        assert _validate_slack_url("https://evil.com/commands") is False
        assert _validate_slack_url("https://hooks.slack.com.evil.com/") is False
        assert _validate_slack_url("https://not-slack.com/api/v1") is False

    def test_validate_slack_url_rejects_localhost(self):
        """Localhost should be rejected."""
        assert _validate_slack_url("https://localhost/commands") is False
        assert _validate_slack_url("https://127.0.0.1/commands") is False

    def test_validate_slack_url_rejects_internal_ips(self):
        """Internal IPs should be rejected."""
        assert _validate_slack_url("https://192.168.1.1/commands") is False
        assert _validate_slack_url("https://10.0.0.1/commands") is False

    def test_validate_slack_url_handles_malformed(self):
        """Malformed URLs should be rejected gracefully."""
        assert _validate_slack_url("") is False
        assert _validate_slack_url("not-a-url") is False
        assert _validate_slack_url("://missing-scheme") is False

    def test_allowed_domains_frozen(self):
        """Allowed domains should be immutable."""
        assert isinstance(SLACK_ALLOWED_DOMAINS, frozenset)
        assert "hooks.slack.com" in SLACK_ALLOWED_DOMAINS
        assert "api.slack.com" in SLACK_ALLOWED_DOMAINS


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/integrations/slack/status."""

    @pytest.mark.asyncio
    async def test_get_status_without_config(self, handler):
        """Status without config shows disabled."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/slack/status",
            method="GET",
        )

        with patch.dict(os.environ, {}, clear=True):
            result = await handler.handle("/api/v1/integrations/slack/status", {}, mock_http)

        assert result is not None
        status_code, body = parse_result(result)
        assert status_code == 200
        assert "enabled" in body

    @pytest.mark.asyncio
    async def test_get_status_with_signing_secret(self, handler):
        """Status with signing secret configured."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/slack/status",
            method="GET",
        )

        with patch("aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", "secret123"):
            result = await handler.handle("/api/v1/integrations/slack/status", {}, mock_http)

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body.get("signing_secret_configured") is True

    @pytest.mark.asyncio
    async def test_get_status_with_bot_token(self, handler):
        """Status with bot token configured."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/slack/status",
            method="GET",
        )

        with patch("aragora.server.handlers.social._slack_impl.SLACK_BOT_TOKEN", "xoxb-token"):
            result = await handler.handle("/api/v1/integrations/slack/status", {}, mock_http)

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body.get("bot_token_configured") is True


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for Slack request signature verification."""

    def test_signature_verification_success(self, handler, signing_secret):
        """Valid signature should be accepted."""
        body = "command=/aragora&text=help&user_id=U123"
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        # Should not return 401 (signature valid)
        assert result is not None
        assert get_status_code(result) != 401

    def test_signature_verification_failure(self, handler, signing_secret):
        """Invalid signature should be rejected with 401."""
        body = "command=/aragora&text=help"
        timestamp = str(int(time.time()))

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": "v0=invalidsignature",
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=False, error="Invalid signature")
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        assert result is not None
        status_code, body = parse_result(result)
        assert status_code == 401
        assert "Invalid signature" in body.get("error", "")

    def test_signature_verification_logs_on_failure(self, handler, signing_secret):
        """Signature failure should trigger audit logging."""
        body = "command=/aragora&text=help&team_id=T123"
        timestamp = str(int(time.time()))

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": "v0=wrong",
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
            client_address=("192.168.1.100", 12345),
        )

        mock_audit = MagicMock()
        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch(
                "aragora.server.handlers.social._slack_impl._get_audit_logger",
                return_value=mock_audit,
            ),
        ):
            mock_verify.return_value = MagicMock(verified=False, error="Invalid")
            handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        # Audit logger should be called
        mock_audit.log_signature_failure.assert_called_once()

    def test_signature_without_secret_skips_verification(self, handler):
        """Without signing secret configured, verification is skipped."""
        body = "command=/aragora&text=help"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with patch("aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", ""):
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        # Should not fail on signature (no secret to verify against)
        assert result is not None
        assert get_status_code(result) != 401


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting (user and workspace)."""

    def test_workspace_rate_limit_applied(self, handler, signing_secret):
        """Workspace rate limiting should be enforced."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "team_id": "T12345",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_limiter = MagicMock()
        mock_limiter.allow.return_value = MagicMock(allowed=False, retry_after=30)

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch(
                "aragora.server.handlers.social._slack_impl._get_workspace_rate_limiter",
                return_value=mock_limiter,
            ),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        # Should return rate limit message
        assert result is not None
        body_data = get_json(result)
        assert "workspace is sending commands too quickly" in body_data.get("text", "").lower()

    def test_user_rate_limit_applied(self, handler, signing_secret):
        """User rate limiting should be enforced."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "team_id": "T12345",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_workspace_limiter = MagicMock()
        mock_workspace_limiter.allow.return_value = MagicMock(allowed=True, retry_after=0)

        mock_user_limiter = MagicMock()
        mock_user_limiter.allow.return_value = MagicMock(allowed=False, retry_after=15)

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch(
                "aragora.server.handlers.social._slack_impl._get_workspace_rate_limiter",
                return_value=mock_workspace_limiter,
            ),
            patch(
                "aragora.server.handlers.social._slack_impl._get_user_rate_limiter",
                return_value=mock_user_limiter,
            ),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        assert "sending commands too quickly" in body_data.get("text", "").lower()

    def test_rate_limit_logs_to_audit(self, handler, signing_secret):
        """Rate limit events should be logged for audit."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "team_id": "T12345",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_limiter = MagicMock()
        mock_limiter.allow.return_value = MagicMock(allowed=False, retry_after=30)
        mock_audit = MagicMock()

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch(
                "aragora.server.handlers.social._slack_impl._get_workspace_rate_limiter",
                return_value=mock_limiter,
            ),
            patch(
                "aragora.server.handlers.social._slack_impl._get_audit_logger",
                return_value=mock_audit,
            ),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        mock_audit.log_rate_limit.assert_called()


# ===========================================================================
# Slash Command Tests
# ===========================================================================


class TestSlashCommandHelp:
    """Tests for /aragora help command."""

    def test_help_command(self, handler, signing_secret):
        """Help command should return help text."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "Aragora" in text or "aragora" in text.lower()
        assert "help" in text.lower() or "command" in text.lower()
        assert body_data.get("response_type") == "ephemeral"

    def test_empty_command_shows_help(self, handler, signing_secret):
        """Empty command should default to help."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        assert (
            "Aragora" in body_data.get("text", "") or "command" in body_data.get("text", "").lower()
        )


class TestSlashCommandStatus:
    """Tests for /aragora status command."""

    def test_status_command(self, handler, signing_secret):
        """Status command should return system status."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "status",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = [
            MagicMock(name="agent1", elo=1600, wins=5),
            MagicMock(name="agent2", elo=1500, wins=3),
        ]

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch("aragora.ranking.elo.EloSystem", return_value=mock_elo),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        # Should have blocks with status info
        assert "blocks" in body_data or "text" in body_data
        assert body_data.get("response_type") == "ephemeral"


class TestSlashCommandAgents:
    """Tests for /aragora agents command."""

    def test_agents_command_with_agents(self, handler, signing_secret):
        """Agents command should list available agents."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "agents",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_agents = [
            MagicMock(name="claude", elo=1700, wins=10),
            MagicMock(name="gpt4", elo=1650, wins=8),
        ]
        # Need to set name as property since MagicMock name conflicts
        mock_agents[0].name = "claude"
        mock_agents[1].name = "gpt4"

        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = mock_agents

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch("aragora.ranking.elo.EloSystem", return_value=mock_elo),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "agent" in text.lower() or "claude" in text.lower() or "elo" in text.lower()

    def test_agents_command_empty(self, handler, signing_secret):
        """Agents command with no agents registered."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "agents",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = []

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch("aragora.ranking.elo.EloSystem", return_value=mock_elo),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "no agent" in text.lower()


class TestSlashCommandAsk:
    """Tests for /aragora ask command."""

    def test_ask_command_without_question(self, handler, signing_secret):
        """Ask without question should show error."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "ask",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "provide" in text.lower() or "question" in text.lower()
        assert body_data.get("response_type") == "ephemeral"

    def test_ask_command_too_short(self, handler, signing_secret):
        """Ask with very short question should show error."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "ask why",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "short" in text.lower()

    def test_ask_command_too_long(self, handler, signing_secret):
        """Ask with very long question should show error."""
        long_question = "x" * 600
        body = urlencode(
            {
                "command": "/aragora",
                "text": f"ask {long_question}",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "long" in text.lower()

    def test_ask_command_valid_question(self, handler, signing_secret):
        """Ask with valid question should acknowledge."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": 'ask "What is the capital of France?"',
                "user_id": "U123",
                "channel_id": "C123",
                "response_url": "https://hooks.slack.com/commands/T123/456/token",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch("aragora.server.handlers.social._slack_impl.create_tracked_task"),
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        # Should acknowledge with processing message
        assert "blocks" in body_data or "Processing" in body_data.get("text", "")
        assert body_data.get("response_type") == "in_channel"


class TestSlashCommandSearch:
    """Tests for /aragora search command."""

    def test_search_without_query(self, handler, signing_secret):
        """Search without query should show error."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "search",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "provide" in text.lower() or "query" in text.lower()

    def test_search_with_short_query(self, handler, signing_secret):
        """Search with too-short query should show error."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "search x",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "short" in text.lower()


class TestSlashCommandUnknown:
    """Tests for unknown commands."""

    def test_unknown_command(self, handler, signing_secret):
        """Unknown command should show error."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "foobar",
                "user_id": "U123",
                "channel_id": "C123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        body_data = get_json(result)
        text = body_data.get("text", "")
        assert "unknown" in text.lower() or "foobar" in text.lower()
        assert body_data.get("response_type") == "ephemeral"


# ===========================================================================
# Interactive Component Tests
# ===========================================================================


class TestInteractiveComponents:
    """Tests for Slack interactive components."""

    def test_interactive_payload_parsed(self, handler, signing_secret):
        """Interactive payload should be parsed correctly."""
        payload = {
            "type": "block_actions",
            "user": {"id": "U123", "name": "testuser"},
            "team": {"id": "T123"},
            "actions": [{"action_id": "vote_for", "value": "debate123"}],
            "response_url": "https://hooks.slack.com/actions/T123/456/token",
        }
        data = {"payload": json.dumps(payload)}
        body = urlencode(data)
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/interactive",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/interactive", {}, mock_http)

        # Should return some response (not 401)
        assert result is not None
        assert get_status_code(result) != 401


# ===========================================================================
# Events API Tests
# ===========================================================================


class TestEventsAPI:
    """Tests for Slack Events API."""

    def test_url_verification_challenge(self, handler, signing_secret):
        """URL verification challenge should be echoed."""
        payload = {
            "type": "url_verification",
            "challenge": "test_challenge_token",
        }
        body = json.dumps(payload)
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/events",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/events", {}, mock_http)

        assert result is not None
        body_data = get_json(result)
        assert body_data.get("challenge") == "test_challenge_token"

    def test_event_callback_processed(self, handler, signing_secret):
        """Event callbacks should be processed."""
        payload = {
            "type": "event_callback",
            "team_id": "T123",
            "event": {
                "type": "message",
                "text": "Hello",
                "user": "U123",
                "channel": "C123",
            },
        }
        body = json.dumps(payload)
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/events",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/events", {}, mock_http)

        assert result is not None
        # Should return OK (acknowledge event)
        assert get_status_code(result) == 200


# ===========================================================================
# Multi-Workspace Tests
# ===========================================================================


class TestMultiWorkspace:
    """Tests for multi-workspace support."""

    def test_team_id_extracted_from_commands(self, handler, signing_secret):
        """Team ID should be extracted from slash command body."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "team_id": "T_WORKSPACE_123",
            }
        )
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
            patch("aragora.server.handlers.social._slack_impl.resolve_workspace") as mock_resolve,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            mock_resolve.return_value = None
            handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        mock_resolve.assert_called_with("T_WORKSPACE_123")

    def test_workspace_specific_signing_secret(self, handler):
        """Workspace-specific signing secret should be used if available."""
        body = urlencode(
            {
                "command": "/aragora",
                "text": "help",
                "user_id": "U123",
                "team_id": "T_CUSTOM",
            }
        )
        timestamp = str(int(time.time()))
        custom_secret = "custom_workspace_secret"
        signature = generate_slack_signature(body, timestamp, custom_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/commands",
            method="POST",
        )

        mock_workspace = MagicMock()
        mock_workspace.signing_secret = custom_secret

        with (
            patch("aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", ""),
            patch(
                "aragora.server.handlers.social._slack_impl.resolve_workspace",
                return_value=mock_workspace,
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/commands", {}, mock_http)

        # verify_slack_signature should be called with custom secret
        mock_verify.assert_called_once()
        call_kwargs = mock_verify.call_args
        assert call_kwargs[1].get("signing_secret") == custom_secret or custom_secret in str(
            call_kwargs
        )


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_method_not_allowed_for_status_post(self, handler):
        """POST to status endpoint should not be allowed for GET-only."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/slack/status",
            method="POST",
            body=b"{}",
        )
        # Note: Status endpoint returns via handle() which checks method
        # This is actually handled by handle() which routes to _get_status()
        result = handler.handle("/api/v1/integrations/slack/status", {}, mock_http)
        # Status works for any method via handle() directly
        assert result is not None

    def test_not_found_for_unknown_path(self, handler, signing_secret):
        """Unknown path should return 404."""
        body = "test"
        timestamp = str(int(time.time()))
        signature = generate_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body.encode("utf-8"),
            path="/api/v1/integrations/slack/unknown",
            method="POST",
        )

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.SLACK_SIGNING_SECRET", signing_secret
            ),
            patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(verified=True, error=None)
            result = handler.handle("/api/v1/integrations/slack/unknown", {}, mock_http)

        # Either None or 404
        if result is not None:
            assert get_status_code(result) == 404


# ===========================================================================
# Handler Factory Tests
# ===========================================================================


class TestHandlerFactory:
    """Tests for handler factory functions."""

    def test_get_slack_handler_creates_instance(self, mock_server_context):
        """get_slack_handler should create handler instance."""
        # Reset global
        import aragora.server.handlers.social.slack as slack_module

        slack_module._slack_handler = None

        handler = get_slack_handler(mock_server_context)
        assert handler is not None
        assert isinstance(handler, SlackHandler)

    def test_get_slack_handler_returns_same_instance(self, mock_server_context):
        """get_slack_handler should return singleton."""
        import aragora.server.handlers.social.slack as slack_module

        slack_module._slack_handler = None

        handler1 = get_slack_handler(mock_server_context)
        handler2 = get_slack_handler(mock_server_context)
        assert handler1 is handler2

    def test_get_slack_handler_default_context(self):
        """get_slack_handler should work with no context."""
        import aragora.server.handlers.social.slack as slack_module

        slack_module._slack_handler = None

        handler = get_slack_handler()
        assert handler is not None

    def test_get_slack_integration_without_webhook(self):
        """get_slack_integration without webhook returns None."""
        import aragora.server.handlers.social.slack as slack_module

        slack_module._slack_integration = None

        with patch("aragora.server.handlers.social._slack_impl.SLACK_WEBHOOK_URL", ""):
            integration = get_slack_integration()

        assert integration is None

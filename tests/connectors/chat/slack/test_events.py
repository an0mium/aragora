"""
Tests for Slack events module - webhook verification and event parsing.

Tests cover:
- Webhook signature verification
- URL verification challenge parsing
- Slash command parsing
- Button click interaction parsing
- Select menu interaction parsing
- Modal submission parsing
- Message event callback parsing
- Event type extraction
- User and channel object creation
- Error handling for malformed payloads
- Content-Type handling (JSON vs form data)
"""

from __future__ import annotations

import json
import time
import hmac
import hashlib
from urllib.parse import urlencode
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_slack_connector():
    """Create a mock connector with SlackEventsMixin methods."""
    from aragora.connectors.chat.slack.events import SlackEventsMixin

    # Create a class that implements the protocol
    class MockConnector(SlackEventsMixin):
        def __init__(self, signing_secret: str | None = None):
            self.signing_secret = signing_secret

        @property
        def platform_name(self) -> str:
            return "slack"

    return MockConnector


def compute_slack_signature(signing_secret: str, timestamp: str, body: bytes) -> str:
    """Compute valid Slack signature for testing."""
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


# ---------------------------------------------------------------------------
# Webhook Verification Tests
# ---------------------------------------------------------------------------


class TestWebhookVerification:
    """Tests for verify_webhook method."""

    def test_verify_webhook_valid_signature(self, mock_slack_connector):
        """Should verify valid webhook signature."""
        connector = mock_slack_connector(signing_secret="test-secret")

        timestamp = str(int(time.time()))
        body = b'{"type":"url_verification","challenge":"abc123"}'
        signature = compute_slack_signature("test-secret", timestamp, body)

        headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        }

        result = connector.verify_webhook(headers, body)
        assert result is True

    def test_verify_webhook_invalid_signature(self, mock_slack_connector):
        """Should reject invalid webhook signature."""
        connector = mock_slack_connector(signing_secret="test-secret")

        timestamp = str(int(time.time()))
        body = b'{"type":"event"}'

        headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": "v0=invalid_signature_here",
        }

        result = connector.verify_webhook(headers, body)
        assert result is False

    def test_verify_webhook_expired_timestamp(self, mock_slack_connector):
        """Should reject expired timestamps (replay attack protection)."""
        connector = mock_slack_connector(signing_secret="test-secret")

        # Timestamp from 10 minutes ago (Slack allows 5 minutes)
        old_timestamp = str(int(time.time()) - 600)
        body = b'{"type":"event"}'
        signature = compute_slack_signature("test-secret", old_timestamp, body)

        headers = {
            "X-Slack-Request-Timestamp": old_timestamp,
            "X-Slack-Signature": signature,
        }

        result = connector.verify_webhook(headers, body)
        assert result is False

    def test_verify_webhook_missing_timestamp(self, mock_slack_connector):
        """Should reject request with missing timestamp."""
        connector = mock_slack_connector(signing_secret="test-secret")

        headers = {
            "X-Slack-Signature": "v0=some_signature",
        }

        result = connector.verify_webhook(headers, b"test")
        assert result is False

    def test_verify_webhook_missing_signature(self, mock_slack_connector):
        """Should reject request with missing signature."""
        connector = mock_slack_connector(signing_secret="test-secret")

        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
        }

        result = connector.verify_webhook(headers, b"test")
        assert result is False

    def test_verify_webhook_no_signing_secret(self, mock_slack_connector):
        """Should handle missing signing secret gracefully."""
        connector = mock_slack_connector(signing_secret=None)

        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=any",
        }

        # Should fail closed when signing secret not configured
        result = connector.verify_webhook(headers, b"test")
        assert result is False


# ---------------------------------------------------------------------------
# URL Verification Tests
# ---------------------------------------------------------------------------


class TestURLVerification:
    """Tests for URL verification challenge handling."""

    def test_parse_url_verification_challenge(self, mock_slack_connector):
        """Should parse URL verification challenge."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "url_verification",
                "challenge": "abc123xyz",
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "url_verification"
        assert event.challenge == "abc123xyz"
        assert event.platform == "slack"

    def test_url_verification_is_verification_property(self, mock_slack_connector):
        """Should set is_verification property correctly."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "url_verification",
                "challenge": "test_challenge",
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.is_verification is True


# ---------------------------------------------------------------------------
# Slash Command Tests
# ---------------------------------------------------------------------------


class TestSlashCommandParsing:
    """Tests for slash command webhook parsing."""

    def test_parse_slash_command_basic(self, mock_slack_connector):
        """Should parse basic slash command."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/debate",
                "text": "topic argument",
                "user_id": "U123",
                "user_name": "testuser",
                "channel_id": "C456",
                "channel_name": "general",
                "team_id": "T789",
                "response_url": "https://hooks.slack.com/response/...",
                "trigger_id": "trigger123",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "slash_command"
        assert event.command is not None
        assert event.command.name == "debate"
        assert event.command.args == ["topic", "argument"]

    def test_parse_slash_command_user_info(self, mock_slack_connector):
        """Should extract user info from slash command."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/test",
                "text": "",
                "user_id": "U_USER123",
                "user_name": "johndoe",
                "channel_id": "C123",
                "channel_name": "test",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.user.id == "U_USER123"
        assert event.command.user.username == "johndoe"
        assert event.command.user.platform == "slack"

    def test_parse_slash_command_channel_info(self, mock_slack_connector):
        """Should extract channel info from slash command."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/test",
                "text": "",
                "user_id": "U123",
                "channel_id": "C_CHANNEL456",
                "channel_name": "dev-team",
                "team_id": "T_TEAM789",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.channel.id == "C_CHANNEL456"
        assert event.command.channel.name == "dev-team"
        assert event.command.channel.team_id == "T_TEAM789"

    def test_parse_slash_command_strips_leading_slash(self, mock_slack_connector):
        """Should strip leading slash from command name."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/mycommand",
                "text": "arg1 arg2",
                "user_id": "U1",
                "channel_id": "C1",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.name == "mycommand"

    def test_parse_slash_command_empty_text(self, mock_slack_connector):
        """Should handle empty command text."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/cmd",
                "text": "",
                "user_id": "U1",
                "channel_id": "C1",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.args == []

    def test_parse_slash_command_response_url(self, mock_slack_connector):
        """Should include response URL for async responses."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/async",
                "text": "",
                "user_id": "U1",
                "channel_id": "C1",
                "response_url": "https://hooks.slack.com/commands/T123/456",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.response_url == "https://hooks.slack.com/commands/T123/456"

    def test_parse_slash_command_trigger_id_in_metadata(self, mock_slack_connector):
        """Should include trigger_id in metadata."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/modal",
                "text": "",
                "user_id": "U1",
                "channel_id": "C1",
                "trigger_id": "modal_trigger_123",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.command.metadata["trigger_id"] == "modal_trigger_123"


# ---------------------------------------------------------------------------
# Button Click Interaction Tests
# ---------------------------------------------------------------------------


class TestButtonClickParsing:
    """Tests for button click interaction parsing."""

    def test_parse_button_click_basic(self, mock_slack_connector):
        """Should parse basic button click interaction."""
        from aragora.connectors.chat.models import InteractionType

        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "trigger_id": "trigger123",
            "user": {"id": "U123", "username": "testuser"},
            "channel": {"id": "C456", "name": "general"},
            "team": {"id": "T789"},
            "actions": [
                {
                    "type": "button",
                    "action_id": "approve_btn",
                    "value": "approved",
                }
            ],
            "message": {"ts": "1234567890.123456"},
            "response_url": "https://hooks.slack.com/response/...",
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "block_actions"
        assert event.interaction is not None
        assert event.interaction.interaction_type == InteractionType.BUTTON_CLICK
        assert event.interaction.action_id == "approve_btn"
        assert event.interaction.value == "approved"

    def test_parse_button_click_user_info(self, mock_slack_connector):
        """Should extract user info from button click."""
        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "user": {"id": "U_CLICKER", "username": "clicker", "name": "Clicker User"},
            "channel": {"id": "C1"},
            "actions": [{"type": "button", "action_id": "btn", "value": "val"}],
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.user.id == "U_CLICKER"
        assert event.interaction.user.username == "clicker"

    def test_parse_button_click_message_id(self, mock_slack_connector):
        """Should extract message ID from button click."""
        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "actions": [{"type": "button", "action_id": "btn"}],
            "message": {"ts": "1704067200.123456"},
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.message_id == "1704067200.123456"


# ---------------------------------------------------------------------------
# Select Menu Interaction Tests
# ---------------------------------------------------------------------------


class TestSelectMenuParsing:
    """Tests for select menu interaction parsing."""

    def test_parse_select_menu_interaction(self, mock_slack_connector):
        """Should parse select menu interaction."""
        from aragora.connectors.chat.models import InteractionType

        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "user": {"id": "U123"},
            "channel": {"id": "C456"},
            "actions": [
                {
                    "type": "static_select",
                    "action_id": "priority_select",
                    "selected_option": {"value": "high"},
                }
            ],
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.interaction_type == InteractionType.SELECT_MENU

    def test_parse_select_menu_selected_options(self, mock_slack_connector):
        """Should include selected options in values."""
        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "user": {"id": "U123"},
            "channel": {"id": "C456"},
            "actions": [
                {
                    "type": "multi_static_select",
                    "action_id": "tags_select",
                    "selected_options": [
                        {"value": "tag1"},
                        {"value": "tag2"},
                    ],
                }
            ],
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.values == [{"value": "tag1"}, {"value": "tag2"}]


# ---------------------------------------------------------------------------
# Modal Submission Tests
# ---------------------------------------------------------------------------


class TestModalSubmissionParsing:
    """Tests for modal/view submission parsing."""

    def test_parse_modal_submission(self, mock_slack_connector):
        """Should parse modal submission."""
        from aragora.connectors.chat.models import InteractionType

        connector = mock_slack_connector()

        payload = {
            "type": "view_submission",
            "trigger_id": "trigger_modal",
            "user": {"id": "U123", "username": "submitter"},
            "view": {
                "id": "V123",
                "callback_id": "feedback_modal",
                "state": {
                    "values": {"input_block": {"feedback_input": {"value": "Great product!"}}}
                },
            },
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "view_submission"
        assert event.interaction is not None
        assert event.interaction.interaction_type == InteractionType.MODAL_SUBMIT

    def test_parse_modal_submission_callback_id(self, mock_slack_connector):
        """Should extract callback_id as action_id."""
        connector = mock_slack_connector()

        payload = {
            "type": "view_submission",
            "user": {"id": "U123"},
            "view": {
                "callback_id": "my_modal_callback",
            },
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.action_id == "my_modal_callback"

    def test_parse_modal_submission_view_in_metadata(self, mock_slack_connector):
        """Should include full view in metadata."""
        connector = mock_slack_connector()

        view_data = {
            "id": "V_VIEW123",
            "callback_id": "modal",
            "state": {"values": {}},
        }

        payload = {
            "type": "view_submission",
            "user": {"id": "U123"},
            "view": view_data,
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.interaction.metadata["view"] == view_data


# ---------------------------------------------------------------------------
# Message Event Callback Tests
# ---------------------------------------------------------------------------


class TestMessageEventParsing:
    """Tests for Events API message callback parsing."""

    def test_parse_message_event(self, mock_slack_connector):
        """Should parse message event callback."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "team_id": "T789",
                "event": {
                    "type": "message",
                    "user": "U123",
                    "channel": "C456",
                    "text": "Hello world",
                    "ts": "1234567890.123456",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "message"
        assert event.message is not None
        assert event.message.content == "Hello world"

    def test_parse_message_event_user_info(self, mock_slack_connector):
        """Should extract user info from message event."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U_AUTHOR",
                    "channel": "C1",
                    "text": "Test",
                    "ts": "1.0",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.message.author.id == "U_AUTHOR"
        assert event.message.author.platform == "slack"

    def test_parse_message_event_channel_info(self, mock_slack_connector):
        """Should extract channel info from message event."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "team_id": "T_TEAM",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C_CHANNEL",
                    "text": "Test",
                    "ts": "1.0",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.message.channel.id == "C_CHANNEL"
        assert event.message.channel.team_id == "T_TEAM"

    def test_parse_message_event_thread_ts(self, mock_slack_connector):
        """Should extract thread_ts as thread_id."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C1",
                    "text": "Reply",
                    "ts": "1704067300.000001",
                    "thread_ts": "1704067200.000001",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.message.thread_id == "1704067200.000001"

    def test_parse_message_event_ignores_bot_messages(self, mock_slack_connector):
        """Should not create message for bot messages."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "bot_id": "B_BOT123",
                    "channel": "C1",
                    "text": "Bot message",
                    "ts": "1.0",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "message"
        assert event.message is None  # Bot messages filtered out

    def test_parse_message_event_message_id(self, mock_slack_connector):
        """Should use ts as message ID."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C1",
                    "text": "Test",
                    "ts": "1704067200.123456",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.message.id == "1704067200.123456"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in event parsing."""

    def test_parse_invalid_json(self, mock_slack_connector):
        """Should handle invalid JSON gracefully."""
        connector = mock_slack_connector()

        body = b"not valid json"
        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "error"
        assert event.platform == "slack"

    def test_parse_unknown_event_type(self, mock_slack_connector):
        """Should handle unknown event types."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "type": "unknown_event_type",
                "data": "something",
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "unknown_event_type"

    def test_parse_missing_event_type(self, mock_slack_connector):
        """Should handle missing event type."""
        connector = mock_slack_connector()

        body = json.dumps(
            {
                "data": "no type field",
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "unknown"

    def test_parse_interaction_no_actions(self, mock_slack_connector):
        """Should handle block_actions with empty actions array."""
        connector = mock_slack_connector()

        payload = {
            "type": "block_actions",
            "user": {"id": "U123"},
            "channel": {"id": "C456"},
            "actions": [],
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "block_actions"
        # No interaction created when actions is empty
        assert event.interaction is None


# ---------------------------------------------------------------------------
# Raw Payload Preservation Tests
# ---------------------------------------------------------------------------


class TestRawPayloadPreservation:
    """Tests for raw payload preservation."""

    def test_preserves_raw_payload_json(self, mock_slack_connector):
        """Should preserve raw JSON payload."""
        connector = mock_slack_connector()

        original_payload = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U1",
                "channel": "C1",
                "text": "Test",
                "ts": "1.0",
            },
            "custom_field": "preserved",
        }

        body = json.dumps(original_payload).encode()
        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.raw_payload["custom_field"] == "preserved"

    def test_preserves_raw_payload_form(self, mock_slack_connector):
        """Should preserve raw form data payload."""
        connector = mock_slack_connector()

        body = urlencode(
            {
                "command": "/test",
                "text": "args",
                "user_id": "U1",
                "channel_id": "C1",
                "custom_field": "value",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert "custom_field" in event.raw_payload

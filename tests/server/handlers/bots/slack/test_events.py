"""
Tests for Slack events handler.

Tests cover:
- URL verification challenge
- App mention events
- Message events
- App uninstall events
- Attachment extraction
- Event validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_request():
    """Create a mock HTTP request for Slack events."""

    class MockRequest:
        def __init__(self, body_data: dict):
            self._body = json.dumps(body_data).encode()
            self.headers = {
                "X-Slack-Signature": "v0=test",
                "X-Slack-Request-Timestamp": "1234567890",
            }

        async def body(self) -> bytes:
            return self._body

    return MockRequest


class TestURLVerification:
    """Tests for URL verification challenge."""

    @pytest.mark.asyncio
    async def test_url_verification_returns_challenge(self, mock_request):
        """Test URL verification returns challenge token."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request({"type": "url_verification", "challenge": "test-challenge-abc123"})

        result = await handle_slack_events(request)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body.get("challenge") == "test-challenge-abc123"

    @pytest.mark.asyncio
    async def test_url_verification_missing_challenge(self, mock_request):
        """Test URL verification without challenge."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request({"type": "url_verification"})

        result = await handle_slack_events(request)

        # Should handle missing challenge gracefully
        assert result is not None


class TestEventCallback:
    """Tests for event callback handling."""

    @pytest.mark.asyncio
    async def test_event_callback_app_mention(self, mock_request):
        """Test app mention event handling."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "user": "U12345678",
                    "text": "<@BOTID> What is AI?",
                    "channel": "C12345678",
                    "ts": "1234567890.123456",
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        assert result is not None
        # Should acknowledge the event
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_event_callback_message(self, mock_request):
        """Test message event handling."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U12345678",
                    "text": "Hello world",
                    "channel": "C12345678",
                    "ts": "1234567890.123456",
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_event_callback_app_uninstalled(self, mock_request):
        """Test app uninstalled event handling."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "app_uninstalled",
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_event_callback_tokens_revoked(self, mock_request):
        """Test tokens revoked event handling."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "tokens_revoked",
                    "tokens": {"oauth": ["token1"], "bot": ["token2"]},
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        assert result is not None
        assert result.status_code == 200


class TestAttachmentExtraction:
    """Tests for attachment extraction."""

    def test_extract_slack_attachments_files(self):
        """Test extracting file attachments."""
        from aragora.server.handlers.bots.slack.events import _extract_slack_attachments

        event = {
            "files": [
                {
                    "id": "F12345",
                    "name": "document.pdf",
                    "mimetype": "application/pdf",
                    "size": 1024,
                    "url_private": "https://files.slack.com/xxx",
                }
            ]
        }

        attachments = _extract_slack_attachments(event)

        assert len(attachments) == 1
        assert attachments[0]["type"] == "slack_file"
        assert attachments[0]["file_id"] == "F12345"
        assert attachments[0]["filename"] == "document.pdf"

    def test_extract_slack_attachments_message_attachments(self):
        """Test extracting message attachments."""
        from aragora.server.handlers.bots.slack.events import _extract_slack_attachments

        event = {
            "attachments": [
                {
                    "title": "Link Preview",
                    "text": "Preview text here",
                    "title_link": "https://example.com",
                }
            ]
        }

        attachments = _extract_slack_attachments(event)

        assert len(attachments) == 1
        assert attachments[0]["type"] == "slack_attachment"
        assert attachments[0]["title"] == "Link Preview"

    def test_extract_slack_attachments_truncates_long_preview(self):
        """Test attachment preview truncation."""
        from aragora.server.handlers.bots.slack.events import _extract_slack_attachments

        long_preview = "x" * 5000
        event = {"files": [{"id": "F123", "preview_plain_text": long_preview}]}

        attachments = _extract_slack_attachments(event, max_preview=100)

        assert len(attachments) == 1
        assert len(attachments[0].get("text", "")) <= 103  # 100 + "..."

    def test_extract_slack_attachments_empty_event(self):
        """Test extraction with empty event."""
        from aragora.server.handlers.bots.slack.events import _extract_slack_attachments

        attachments = _extract_slack_attachments({})

        assert attachments == []

    def test_extract_slack_attachments_invalid_files(self):
        """Test extraction handles invalid file entries."""
        from aragora.server.handlers.bots.slack.events import _extract_slack_attachments

        event = {
            "files": [
                None,
                "invalid",
                {"id": "F123", "name": "valid.txt"},
            ]
        }

        attachments = _extract_slack_attachments(event)

        # Should only extract the valid file
        assert len(attachments) == 1


class TestInputValidation:
    """Tests for event input validation."""

    @pytest.mark.asyncio
    async def test_invalid_user_id_in_event(self, mock_request):
        """Test invalid user ID in event is handled."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "user": "<script>alert(1)</script>",
                    "text": "test",
                    "channel": "C12345678",
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_channel_id_in_event(self, mock_request):
        """Test invalid channel ID in event is handled."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request(
            {
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "user": "U12345678",
                    "text": "test",
                    "channel": "../../../etc/passwd",
                },
                "team_id": "T12345678",
            }
        )

        result = await handle_slack_events(request)

        # Should handle gracefully
        assert result is not None


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_malformed_json_body(self):
        """Test handling of malformed JSON body."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        class BadRequest:
            async def body(self):
                return b"not valid json"

        result = await handle_slack_events(BadRequest())

        assert result is not None
        # Should return error response
        assert result.status_code in (400, 500)

    @pytest.mark.asyncio
    async def test_missing_event_type(self, mock_request):
        """Test handling of missing event type."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        request = mock_request({"some": "data"})

        result = await handle_slack_events(request)

        assert result is not None


class TestHydrateAttachments:
    """Tests for attachment hydration."""

    @pytest.mark.asyncio
    async def test_hydrate_empty_attachments(self):
        """Test hydrating empty attachments list."""
        from aragora.server.handlers.bots.slack.events import _hydrate_slack_attachments

        result = await _hydrate_slack_attachments([])

        assert result == []

    @pytest.mark.asyncio
    async def test_hydrate_skips_large_files(self):
        """Test hydration skips files exceeding max bytes."""
        from aragora.server.handlers.bots.slack.events import _hydrate_slack_attachments

        attachments = [{"file_id": "F123", "size": 10_000_000}]

        result = await _hydrate_slack_attachments(attachments, max_bytes=1_000_000)

        # Should skip but keep the attachment
        assert len(result) == 1
        assert "data" not in result[0]

    @pytest.mark.asyncio
    async def test_hydrate_skips_already_hydrated(self):
        """Test hydration skips already hydrated attachments."""
        from aragora.server.handlers.bots.slack.events import _hydrate_slack_attachments

        attachments = [{"file_id": "F123", "data": b"existing data"}]

        result = await _hydrate_slack_attachments(attachments)

        assert len(result) == 1
        assert result[0]["data"] == b"existing data"

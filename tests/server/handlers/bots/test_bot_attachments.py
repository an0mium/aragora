"""Tests for bot handler attachment extraction and forwarding."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.google_chat import GoogleChatHandler


# =============================================================================
# Google Chat Attachment Extraction
# =============================================================================


class TestGoogleChatAttachments:
    """Tests for GoogleChatHandler._extract_attachments."""

    def setup_method(self):
        self.handler = GoogleChatHandler({})

    def test_extract_no_message(self):
        """Returns empty list when event has no message."""
        assert self.handler._extract_attachments({}) == []

    def test_extract_message_not_dict(self):
        """Returns empty list when message is not a dict."""
        assert self.handler._extract_attachments({"message": "text"}) == []

    def test_extract_no_attachments(self):
        """Returns empty list when message has no attachments."""
        assert self.handler._extract_attachments({"message": {}}) == []

    def test_extract_attachments_not_list(self):
        """Returns empty list when attachments is not a list."""
        assert self.handler._extract_attachments({"message": {"attachments": "bad"}}) == []

    def test_extract_single_attachment_with_name(self):
        """Extracts attachment with name field."""
        event = {
            "message": {"attachments": [{"name": "report.pdf", "contentType": "application/pdf"}]}
        }
        result = self.handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["name"] == "report.pdf"
        assert result[0]["filename"] == "report.pdf"

    def test_extract_attachment_with_filename(self):
        """Uses filename field directly when present."""
        event = {"message": {"attachments": [{"filename": "data.csv"}]}}
        result = self.handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "data.csv"

    def test_extract_attachment_with_content_name(self):
        """Falls back to contentName field."""
        event = {"message": {"attachments": [{"contentName": "image.png"}]}}
        result = self.handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "image.png"

    def test_extract_attachment_default_filename(self):
        """Uses default filename when no name fields present."""
        event = {"message": {"attachments": [{"contentType": "application/octet-stream"}]}}
        result = self.handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "attachment_1"

    def test_extract_multiple_attachments(self):
        """Extracts multiple attachments with correct indexing."""
        event = {
            "message": {
                "attachments": [
                    {"name": "a.pdf"},
                    {"contentType": "image/jpeg"},
                    {"filename": "c.txt"},
                ]
            }
        }
        result = self.handler._extract_attachments(event)
        assert len(result) == 3
        assert result[0]["filename"] == "a.pdf"
        assert result[1]["filename"] == "attachment_2"
        assert result[2]["filename"] == "c.txt"

    def test_extract_skips_non_dict_entries(self):
        """Skips non-dict entries in attachments list."""
        event = {
            "message": {
                "attachments": [
                    {"name": "valid.pdf"},
                    "not_a_dict",
                    42,
                    {"name": "also_valid.txt"},
                ]
            }
        }
        result = self.handler._extract_attachments(event)
        assert len(result) == 2
        assert result[0]["filename"] == "valid.pdf"
        assert result[1]["filename"] == "also_valid.txt"

    def test_extract_preserves_original_fields(self):
        """Preserves all original attachment fields."""
        event = {
            "message": {
                "attachments": [
                    {
                        "name": "doc.pdf",
                        "contentType": "application/pdf",
                        "downloadUri": "https://example.com/doc.pdf",
                        "driveDataRef": {"driveFileId": "abc123"},
                    }
                ]
            }
        }
        result = self.handler._extract_attachments(event)
        assert result[0]["contentType"] == "application/pdf"
        assert result[0]["downloadUri"] == "https://example.com/doc.pdf"
        assert result[0]["driveDataRef"]["driveFileId"] == "abc123"


# =============================================================================
# Slack Debate Attachments
# =============================================================================


class TestSlackDebateAttachments:
    """Tests for start_slack_debate attachment parameter."""

    @pytest.mark.asyncio
    async def test_start_debate_passes_attachments(self):
        """start_slack_debate passes attachments to DecisionInput."""
        from aragora.server.handlers.bots.slack.debates import start_slack_debate

        captured_input = {}

        async def mock_route(decision_input):
            captured_input["attachments"] = decision_input.attachments
            return MagicMock(request_id="test-123")

        mock_router = MagicMock()
        mock_router.route = AsyncMock(side_effect=mock_route)

        with (
            patch(
                "aragora.core.get_decision_router",
                return_value=mock_router,
            ),
            patch(
                "aragora.server.handlers.bots.slack.debates._active_debates",
                {},
            ),
        ):
            result = await start_slack_debate(
                topic="test question",
                channel_id="C123",
                user_id="U456",
                attachments=[{"filename": "doc.pdf", "url": "https://example.com/doc.pdf"}],
            )

        assert result == "test-123"
        assert captured_input["attachments"] == [
            {"filename": "doc.pdf", "url": "https://example.com/doc.pdf"}
        ]

    @pytest.mark.asyncio
    async def test_start_debate_defaults_empty_attachments(self):
        """start_slack_debate defaults to empty attachments."""
        from aragora.server.handlers.bots.slack.debates import start_slack_debate

        captured_input = {}

        async def mock_route(decision_input):
            captured_input["attachments"] = decision_input.attachments
            return MagicMock(request_id="test-456")

        mock_router = MagicMock()
        mock_router.route = AsyncMock(side_effect=mock_route)

        with (
            patch(
                "aragora.core.get_decision_router",
                return_value=mock_router,
            ),
            patch(
                "aragora.server.handlers.bots.slack.debates._active_debates",
                {},
            ),
        ):
            result = await start_slack_debate(
                topic="test question",
                channel_id="C123",
                user_id="U456",
            )

        assert result == "test-456"
        assert captured_input["attachments"] == []


# =============================================================================
# Teams Event Attachment Extraction
# =============================================================================


class TestTeamsEventAttachments:
    """Tests for Teams event attachment extraction."""

    def test_extract_attachments_from_activity(self):
        """Teams events extract attachments list from activity."""
        activity = {
            "text": "<at>Aragora</at> debate this topic",
            "attachments": [
                {"contentType": "application/pdf", "name": "report.pdf"},
                {"contentType": "image/png", "name": "chart.png"},
            ],
            "from": {"id": "user1", "name": "Test User"},
            "conversation": {"id": "conv1"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
        }
        attachments = activity.get("attachments")
        assert isinstance(attachments, list)
        assert len(attachments) == 2

    def test_missing_attachments_defaults_to_empty(self):
        """Missing attachments field defaults to empty list."""
        activity = {
            "text": "<at>Aragora</at> debate this topic",
            "from": {"id": "user1"},
        }
        attachments = activity.get("attachments")
        if not isinstance(attachments, list):
            attachments = []
        assert attachments == []

    def test_non_list_attachments_defaults_to_empty(self):
        """Non-list attachments field defaults to empty list."""
        activity = {
            "text": "debate topic",
            "attachments": "not-a-list",
        }
        attachments = activity.get("attachments")
        if not isinstance(attachments, list):
            attachments = []
        assert attachments == []

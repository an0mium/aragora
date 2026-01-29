"""Tests for Gmail Query Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.gmail_query import (
    GmailQueryHandler,
    QueryResponse,
    GMAIL_READ_PERMISSION,
    GMAIL_WRITE_PERMISSION,
)


@pytest.fixture
def handler():
    """Create handler instance."""
    return GmailQueryHandler(ctx={})


class TestQueryResponse:
    """Tests for QueryResponse dataclass."""

    def test_query_response_creation(self):
        """Test creating QueryResponse instance."""
        response = QueryResponse(
            answer="Test answer",
            sources=[{"id": "123", "subject": "Test"}],
            confidence=0.8,
            query="What is the test?",
        )
        assert response.answer == "Test answer"
        assert response.confidence == 0.8
        assert len(response.sources) == 1

    def test_query_response_to_dict(self):
        """Test QueryResponse serialization."""
        response = QueryResponse(
            answer="Test answer",
            sources=[{"id": "123"}],
            confidence=0.9,
            query="Test query",
        )
        result = response.to_dict()
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.9
        assert result["query"] == "Test query"
        assert len(result["sources"]) == 1

    def test_query_response_defaults(self):
        """Test QueryResponse default values."""
        response = QueryResponse(answer="Answer")
        assert response.sources == []
        assert response.confidence == 0.0
        assert response.query == ""


class TestGmailQueryHandler:
    """Tests for GmailQueryHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(GmailQueryHandler, "ROUTES")
        routes = GmailQueryHandler.ROUTES
        assert "/api/v1/gmail/query" in routes
        assert "/api/v1/gmail/query/voice" in routes
        assert "/api/v1/gmail/query/stream" in routes
        assert "/api/v1/gmail/inbox/priority" in routes
        assert "/api/v1/gmail/inbox/feedback" in routes

    def test_can_handle_query_routes(self, handler):
        """Test can_handle for query routes."""
        assert handler.can_handle("/api/v1/gmail/query") is True
        assert handler.can_handle("/api/v1/gmail/query/voice") is True
        assert handler.can_handle("/api/v1/gmail/query/stream") is True

    def test_can_handle_inbox_routes(self, handler):
        """Test can_handle for inbox routes."""
        assert handler.can_handle("/api/v1/gmail/inbox/priority") is True
        assert handler.can_handle("/api/v1/gmail/inbox/feedback") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/outlook/query") is False
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_permissions_defined(self):
        """Test that permission constants are defined."""
        assert GMAIL_READ_PERMISSION == "gmail:read"
        assert GMAIL_WRITE_PERMISSION == "gmail:write"


class TestGmailQueryAuthentication:
    """Tests for Gmail query authentication."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Test handle method requires authentication."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_handler)
            assert result is not None
            assert result.status == 401

    @pytest.mark.asyncio
    async def test_handle_post_requires_authentication(self):
        """Test handle_post method requires authentication."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle_post("/api/v1/gmail/query", {}, mock_handler)
            assert result is not None
            assert result.status == 401

    @pytest.mark.asyncio
    async def test_handle_checks_permission(self):
        """Test handle checks gmail:read permission."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = MagicMock()
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_handler)
            assert result is not None
            assert result.status == 403


class TestGmailQuery:
    """Tests for Gmail query functionality."""

    @pytest.mark.asyncio
    async def test_query_not_connected(self):
        """Test query fails when user is not connected."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch("aragora.server.handlers.features.gmail_query.get_user_state") as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle_post(
                "/api/v1/gmail/query", {"question": "Test?"}, mock_handler
            )
            assert result.status == 401

    @pytest.mark.asyncio
    async def test_query_requires_question(self):
        """Test query requires question parameter."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        mock_state = MagicMock()
        mock_state.refresh_token = "test_token"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch("aragora.server.handlers.features.gmail_query.get_user_state") as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = mock_state

            result = await handler.handle_post("/api/v1/gmail/query", {}, mock_handler)
            assert result.status == 400


class TestGmailVoiceQuery:
    """Tests for Gmail voice query functionality."""

    @pytest.mark.asyncio
    async def test_voice_query_requires_audio(self):
        """Test voice query requires audio data or URL."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        mock_state = MagicMock()
        mock_state.refresh_token = "test_token"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch("aragora.server.handlers.features.gmail_query.get_user_state") as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = mock_state

            result = await handler.handle_post("/api/v1/gmail/query/voice", {}, mock_handler)
            assert result.status == 400

    @pytest.mark.asyncio
    async def test_voice_query_not_connected(self):
        """Test voice query fails when not connected."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch("aragora.server.handlers.features.gmail_query.get_user_state") as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle_post(
                "/api/v1/gmail/query/voice", {"audio": "base64data"}, mock_handler
            )
            assert result.status == 401


class TestGmailPriorityInbox:
    """Tests for Gmail priority inbox functionality."""

    @pytest.mark.asyncio
    async def test_priority_inbox_not_connected(self):
        """Test priority inbox fails when not connected."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch("aragora.server.handlers.features.gmail_query.get_user_state") as mock_get_state,
        ):
            mock_auth.return_value = MagicMock()
            mock_get_state.return_value = None

            result = await handler.handle(
                "/api/v1/gmail/inbox/priority", {"user_id": "test"}, mock_handler
            )
            assert result.status == 401


class TestGmailFeedback:
    """Tests for Gmail feedback functionality."""

    @pytest.mark.asyncio
    async def test_feedback_requires_email_id(self):
        """Test feedback requires email_id."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await handler.handle_post(
                "/api/v1/gmail/inbox/feedback", {"action": "opened"}, mock_handler
            )
            assert result.status == 400

    @pytest.mark.asyncio
    async def test_feedback_requires_action(self):
        """Test feedback requires action."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await handler.handle_post(
                "/api/v1/gmail/inbox/feedback", {"email_id": "123"}, mock_handler
            )
            assert result.status == 400

    @pytest.mark.asyncio
    async def test_feedback_validates_action(self):
        """Test feedback validates action value."""
        handler = GmailQueryHandler(ctx={})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await handler.handle_post(
                "/api/v1/gmail/inbox/feedback",
                {"email_id": "123", "action": "invalid_action"},
                mock_handler,
            )
            assert result.status == 400


class TestSimpleAnswer:
    """Tests for simple answer generation fallback."""

    def test_simple_answer_how_many(self):
        """Test simple answer for count questions."""
        handler = GmailQueryHandler(ctx={})
        emails = ["email1", "email2", "email3"]

        answer = handler._simple_answer("how many emails?", emails)
        assert "3" in answer

    def test_simple_answer_from_question(self):
        """Test simple answer extracts senders."""
        handler = GmailQueryHandler(ctx={})
        emails = ["From: test@example.com\nSubject: Test"]

        answer = handler._simple_answer("who sent me emails?", emails)
        assert "test@example.com" in answer

    def test_simple_answer_about_question(self):
        """Test simple answer extracts subjects."""
        handler = GmailQueryHandler(ctx={})
        emails = ["From: test@example.com\nSubject: Important Meeting"]

        answer = handler._simple_answer("what are the emails about?", emails)
        assert "Important Meeting" in answer

    def test_simple_answer_fallback(self):
        """Test simple answer fallback response."""
        handler = GmailQueryHandler(ctx={})
        emails = ["email1", "email2"]

        answer = handler._simple_answer("random question", emails)
        assert "2" in answer or "relevant" in answer.lower()

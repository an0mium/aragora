"""
Tests for Email Prioritization Handler.

Tests the email prioritization API endpoints including:
- Email scoring and prioritization
- Inbox ranking
- User feedback
- Configuration management
- Gmail OAuth flows
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.email import (
    EmailHandler,
    handle_prioritize_email,
    handle_rank_inbox,
    handle_email_feedback,
    handle_get_config,
    handle_update_config,
    get_gmail_connector,
    get_prioritizer,
    get_context_service,
    _user_configs,
    _user_configs_lock,
)


class TestEmailHandler:
    """Test EmailHandler class."""

    def test_routes_defined(self):
        """Handler should define expected routes."""
        assert "/api/v1/email/prioritize" in EmailHandler.ROUTES
        assert "/api/v1/email/rank-inbox" in EmailHandler.ROUTES
        assert "/api/v1/email/feedback" in EmailHandler.ROUTES
        assert "/api/v1/email/inbox" in EmailHandler.ROUTES
        assert "/api/v1/email/config" in EmailHandler.ROUTES

    def test_can_handle_static_routes(self):
        """Should handle static routes."""
        handler = EmailHandler({})
        assert handler.can_handle("/api/v1/email/prioritize") is True
        assert handler.can_handle("/api/v1/email/rank-inbox") is True
        assert handler.can_handle("/api/v1/email/feedback") is True

    def test_can_handle_prefix_routes(self):
        """Should handle prefix routes like /api/email/context/:email."""
        handler = EmailHandler({})
        assert handler.can_handle("/api/v1/email/context/user@example.com") is True
        # Note: Empty prefix path is handled by the handler (returns True)

    def test_cannot_handle_unknown_routes(self):
        """Should not handle unknown routes."""
        handler = EmailHandler({})
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/debates") is False


class TestHandlePrioritizeEmail:
    """Test handle_prioritize_email function."""

    @pytest.mark.asyncio
    async def test_prioritize_email_success(self):
        """Should successfully prioritize email."""
        email_data = {
            "id": "msg_123",
            "subject": "Test Subject",
            "from_address": "sender@example.com",
            "body_text": "Test body",
            "labels": ["INBOX"],
            "is_read": False,
            "is_starred": False,
            "is_important": False,
        }

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "score": 75,
            "confidence": 0.85,
            "tier": "high",
        }

        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(return_value=mock_result)
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_prioritize_email(email_data)

            assert result["success"] is True
            assert "result" in result
            assert result["result"]["score"] == 75

    @pytest.mark.asyncio
    async def test_prioritize_email_missing_fields(self):
        """Should handle emails with missing optional fields."""
        email_data = {
            "id": "msg_123",
            "subject": "Test",
            "from_address": "sender@example.com",
        }

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"score": 50}

        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(return_value=mock_result)
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_prioritize_email(email_data)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_prioritize_email_error(self):
        """Should handle errors gracefully."""
        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.score_email = AsyncMock(side_effect=Exception("Prioritization failed"))
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_prioritize_email({"id": "test"})

            assert result["success"] is False
            assert "error" in result


class TestHandleRankInbox:
    """Test handle_rank_inbox function."""

    @pytest.mark.asyncio
    async def test_rank_inbox_success(self):
        """Should successfully rank inbox emails."""
        emails = [
            {"id": "1", "subject": "Urgent", "from_address": "boss@example.com"},
            {"id": "2", "subject": "Newsletter", "from_address": "news@example.com"},
        ]

        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(
                return_value=[
                    MagicMock(
                        email=MagicMock(id="1"),
                        score=90,
                        to_dict=lambda: {"email_id": "1", "score": 90},
                    ),
                    MagicMock(
                        email=MagicMock(id="2"),
                        score=30,
                        to_dict=lambda: {"email_id": "2", "score": 30},
                    ),
                ]
            )
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_rank_inbox(emails)

            assert result["success"] is True
            assert "results" in result
            assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_rank_inbox_with_limit(self):
        """Should respect limit parameter."""
        emails = [{"id": str(i), "subject": f"Email {i}"} for i in range(10)]

        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.rank_inbox = AsyncMock(return_value=[])
            mock_get_prioritizer.return_value = mock_prioritizer

            await handle_rank_inbox(emails, limit=5)

            # Check that rank_inbox was called (limit handling is in the function)
            mock_prioritizer.rank_inbox.assert_called_once()


class TestHandleEmailFeedback:
    """Test handle_email_feedback function."""

    @pytest.mark.asyncio
    async def test_feedback_read_action(self):
        """Should handle read action feedback."""
        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_email_feedback("msg_123", "read")

            assert result["success"] is True
            assert result["email_id"] == "msg_123"
            assert result["action"] == "read"
            mock_prioritizer.record_user_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_archive_action(self):
        """Should handle archive action feedback."""
        with patch("aragora.server.handlers.email.get_prioritizer") as mock_get_prioritizer:
            mock_prioritizer = MagicMock()
            mock_prioritizer.record_user_action = AsyncMock()
            mock_get_prioritizer.return_value = mock_prioritizer

            result = await handle_email_feedback("msg_123", "archived")

            assert result["success"] is True
            assert result["action"] == "archived"


class TestConfigOperations:
    """Test configuration get/update operations."""

    @pytest.mark.asyncio
    async def test_get_config_default(self):
        """Should return default config for new users."""
        import uuid

        # Use a unique user_id to ensure clean state (not affected by persistent store)
        unique_user = f"test_user_default_{uuid.uuid4().hex[:8]}"

        # Clear any existing config
        with _user_configs_lock:
            _user_configs.pop(unique_user, None)

        result = await handle_get_config(unique_user)

        assert "config" in result
        # Default config should have empty lists for new users
        assert result["config"].get("vip_domains", []) == []

    @pytest.mark.asyncio
    async def test_update_config(self):
        """Should update user config."""
        new_config = {
            "vip_domains": ["important.com"],
            "vip_addresses": ["ceo@company.com"],
        }

        result = await handle_update_config("test_user", new_config)

        assert result["success"] is True

        # Verify config was saved
        with _user_configs_lock:
            saved = _user_configs.get("test_user", {})
            assert "important.com" in saved.get("vip_domains", [])


class TestThreadSafety:
    """Test thread-safety of global state."""

    def test_user_configs_lock_exists(self):
        """User configs should have a lock for thread safety."""
        import threading

        assert isinstance(_user_configs_lock, type(threading.Lock()))

    def test_config_access_uses_lock(self):
        """Config access should use the lock."""
        # This tests that the lock pattern is correctly implemented
        # by verifying the lock can be acquired and released
        with _user_configs_lock:
            # Should be able to read safely
            _ = _user_configs.get("nonexistent", {})

        # Lock should be released
        assert _user_configs_lock.acquire(blocking=False)
        _user_configs_lock.release()


class TestLazyInitialization:
    """Test lazy initialization of services."""

    def test_gmail_connector_lazy_init(self):
        """Gmail connector should be lazily initialized."""
        import aragora.server.handlers.email as email_module

        # Reset global state
        email_module._gmail_connector = None

        with patch(
            "aragora.connectors.enterprise.communication.gmail.GmailConnector"
        ) as mock_connector_class:
            mock_connector_class.return_value = MagicMock()

            # First call initializes
            connector1 = get_gmail_connector("user1")

            # Should have created connector
            assert connector1 is not None

    def test_context_service_lazy_init(self):
        """Context service should be lazily initialized."""
        import aragora.server.handlers.email as email_module

        # Reset global state for clean test
        email_module._context_service = None

        with patch(
            "aragora.services.cross_channel_context.CrossChannelContextService"
        ) as mock_service_class:
            mock_service_class.return_value = MagicMock()

            service = get_context_service()

            # Should have created service
            assert service is not None


class TestEmailHandlerMethods:
    """Test EmailHandler async methods."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        ctx = {"user_id": "test_user"}
        handler = EmailHandler(ctx)
        handler._get_user_id = MagicMock(return_value="test_user")
        return handler

    @pytest.mark.asyncio
    async def test_handle_post_prioritize(self, handler):
        """Should handle POST /api/email/prioritize."""
        data = {
            "email": {
                "id": "msg_123",
                "subject": "Test",
                "from_address": "test@example.com",
            }
        }

        with patch(
            "aragora.server.handlers.email.handle_prioritize_email",
            new_callable=AsyncMock,
        ) as mock_prioritize:
            mock_prioritize.return_value = {"success": True, "result": {"score": 75}}

            result = await handler.handle_post_prioritize(data)

            # HandlerResult is a dataclass with status_code attribute
            assert result.status_code == 200
            mock_prioritize.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_feedback_missing_fields(self, handler):
        """Should reject feedback with missing required fields."""
        # Missing email_id
        result = await handler.handle_post_feedback({"action": "read"})
        assert result.status_code == 400

        # Missing action
        result = await handler.handle_post_feedback({"email_id": "123"})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_get_config(self, handler):
        """Should handle GET /api/email/config."""
        with patch(
            "aragora.server.handlers.email.handle_get_config",
            new_callable=AsyncMock,
        ) as mock_get_config:
            mock_get_config.return_value = {"config": {"vip_domains": []}}

            result = await handler.handle_get_config({})

            # HandlerResult is a dataclass with status_code attribute
            assert result.status_code == 200
            mock_get_config.assert_called_once_with("test_user")

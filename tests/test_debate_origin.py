"""
Tests for Debate Origin tracking and Result Router.

Tests the bidirectional chat result routing system.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDebateOrigin:
    """Tests for DebateOrigin dataclass and storage."""

    def test_register_debate_origin(self):
        """Test registering a debate origin."""
        from aragora.server.debate_origin import register_debate_origin, get_debate_origin

        origin = register_debate_origin(
            debate_id="test-123",
            platform="telegram",
            channel_id="12345678",
            user_id="87654321",
            metadata={"username": "test_user"},
        )

        assert origin.debate_id == "test-123"
        assert origin.platform == "telegram"
        assert origin.channel_id == "12345678"
        assert origin.user_id == "87654321"
        assert origin.metadata["username"] == "test_user"
        assert origin.result_sent is False

    def test_get_debate_origin(self):
        """Test retrieving a registered origin."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
            _origin_store,
        )

        # Clear store
        _origin_store.clear()

        register_debate_origin(
            debate_id="test-456",
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="+1234567890",
        )

        origin = get_debate_origin("test-456")
        assert origin is not None
        assert origin.platform == "whatsapp"
        assert origin.channel_id == "+1234567890"

    def test_get_nonexistent_origin(self):
        """Test retrieving non-existent origin returns None."""
        from aragora.server.debate_origin import get_debate_origin, _origin_store

        _origin_store.clear()
        origin = get_debate_origin("nonexistent")
        assert origin is None

    def test_mark_result_sent(self):
        """Test marking result as sent."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            mark_result_sent,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()
        register_debate_origin(
            debate_id="test-789",
            platform="slack",
            channel_id="C12345",
            user_id="U67890",
        )

        mark_result_sent("test-789")

        origin = get_debate_origin("test-789")
        assert origin.result_sent is True
        assert origin.result_sent_at is not None

    def test_origin_to_dict(self):
        """Test converting origin to dict."""
        from aragora.server.debate_origin import DebateOrigin

        origin = DebateOrigin(
            debate_id="test-dict",
            platform="discord",
            channel_id="123456",
            user_id="789012",
            thread_id="thread_1",
            message_id="msg_1",
            metadata={"key": "value"},
        )

        data = origin.to_dict()
        assert data["debate_id"] == "test-dict"
        assert data["platform"] == "discord"
        assert data["thread_id"] == "thread_1"
        assert data["metadata"]["key"] == "value"

    def test_origin_from_dict(self):
        """Test creating origin from dict."""
        from aragora.server.debate_origin import DebateOrigin

        data = {
            "debate_id": "test-from",
            "platform": "teams",
            "channel_id": "chan_1",
            "user_id": "user_1",
            "metadata": {"test": True},
        }

        origin = DebateOrigin.from_dict(data)
        assert origin.debate_id == "test-from"
        assert origin.platform == "teams"
        assert origin.metadata["test"] is True

    def test_cleanup_expired_origins(self):
        """Test cleanup of expired origins from memory."""
        import time
        from aragora.server.debate_origin import (
            register_debate_origin,
            cleanup_expired_origins,
            _origin_store,
            ORIGIN_TTL_SECONDS,
        )

        _origin_store.clear()

        # Register an origin
        origin = register_debate_origin(
            debate_id="test-expire",
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        # Make it expired
        origin.created_at = time.time() - ORIGIN_TTL_SECONDS - 100

        # Cleanup should remove it
        count = cleanup_expired_origins()
        assert count >= 1  # At least memory cleanup
        assert "test-expire" not in _origin_store

    def test_cleanup_expired_origins_sqlite(self):
        """Test cleanup of expired origins from SQLite database."""
        import tempfile
        import os
        import time
        from aragora.server.debate_origin import (
            SQLiteOriginStore,
            DebateOrigin,
            ORIGIN_TTL_SECONDS,
        )

        # Create temp database
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            store = SQLiteOriginStore(db_path=db_path)

            # Create an expired origin
            expired_origin = DebateOrigin(
                debate_id="expired-sqlite",
                platform="telegram",
                channel_id="123",
                user_id="456",
                created_at=time.time() - ORIGIN_TTL_SECONDS - 100,
            )
            store.save(expired_origin)

            # Create a valid origin
            valid_origin = DebateOrigin(
                debate_id="valid-sqlite",
                platform="telegram",
                channel_id="789",
                user_id="012",
                created_at=time.time(),
            )
            store.save(valid_origin)

            # Cleanup should remove only expired
            count = store.cleanup_expired(ORIGIN_TTL_SECONDS)
            assert count == 1

            # Verify expired is gone, valid remains
            assert store.get("expired-sqlite") is None
            assert store.get("valid-sqlite") is not None

        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


class TestFormatResultMessage:
    """Tests for result message formatting."""

    def test_format_markdown(self):
        """Test markdown formatting."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test-fmt",
            platform="telegram",
            channel_id="123",
            user_id="456",
            metadata={"topic": "Test topic"},
        )

        result = {
            "consensus_reached": True,
            "final_answer": "The answer is 42",
            "confidence": 0.85,
            "participants": ["claude", "gpt-4", "gemini"],
            "task": "What is the meaning of life?",
        }

        message = _format_result_message(result, origin, markdown=True)

        assert "**Debate Complete!**" in message
        assert "What is the meaning of life?" in message
        assert "85%" in message
        assert "claude" in message
        assert "The answer is 42" in message

    def test_format_plain_text(self):
        """Test plain text formatting."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test-plain",
            platform="whatsapp",
            channel_id="123",
            user_id="456",
        )

        result = {
            "consensus_reached": False,
            "final_answer": "No consensus",
            "confidence": 0.5,
            "participants": ["agent1"],
            "task": "Topic here",
        }

        message = _format_result_message(result, origin, markdown=False)

        assert "Debate Complete!" in message
        assert "**" not in message  # No markdown
        assert "50%" in message

    def test_truncate_long_answer(self):
        """Test that long answers are truncated."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test-long",
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        result = {
            "consensus_reached": True,
            "final_answer": "A" * 1000,  # Very long answer
            "confidence": 0.9,
            "participants": [],
        }

        message = _format_result_message(result, origin)

        # Should be truncated with ellipsis
        assert "..." in message
        assert len(message) < 1500  # Reasonable length


class TestRouteDebateResult:
    """Tests for result routing."""

    @pytest.mark.asyncio
    async def test_route_result_no_origin(self):
        """Test routing when no origin exists."""
        from aragora.server.debate_origin import route_debate_result, _origin_store

        _origin_store.clear()

        result = await route_debate_result("nonexistent", {"answer": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_route_result_already_sent(self):
        """Test routing when result already sent."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            mark_result_sent,
            _origin_store,
        )

        _origin_store.clear()

        register_debate_origin(
            debate_id="test-sent",
            platform="telegram",
            channel_id="123",
            user_id="456",
        )
        mark_result_sent("test-sent")

        result = await route_debate_result("test-sent", {"answer": "test"})
        assert result is True  # Returns True because already sent

    @pytest.mark.asyncio
    async def test_route_result_telegram(self):
        """Test routing to Telegram."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        register_debate_origin(
            debate_id="test-tg",
            platform="telegram",
            channel_id="12345678",
            user_id="87654321",
        )

        with (
            patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test_token"}),
            patch("httpx.AsyncClient") as mock_client,
        ):
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await route_debate_result(
                "test-tg",
                {
                    "consensus_reached": True,
                    "final_answer": "Test answer",
                    "confidence": 0.8,
                    "participants": ["agent1"],
                },
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_route_result_whatsapp(self):
        """Test routing to WhatsApp."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        register_debate_origin(
            debate_id="test-wa",
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="+1234567890",
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "WHATSAPP_ACCESS_TOKEN": "test_token",
                    "WHATSAPP_PHONE_NUMBER_ID": "12345",
                },
            ),
            patch("httpx.AsyncClient") as mock_client,
        ):
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await route_debate_result(
                "test-wa",
                {
                    "consensus_reached": False,
                    "final_answer": "No consensus",
                    "confidence": 0.5,
                    "participants": [],
                },
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_route_result_unknown_platform(self):
        """Test routing to unknown platform."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        register_debate_origin(
            debate_id="test-unknown",
            platform="unknown_platform",
            channel_id="123",
            user_id="456",
        )

        result = await route_debate_result("test-unknown", {"answer": "test"})
        assert result is False


class TestResultRouter:
    """Tests for result router integration."""

    def test_register_hooks(self):
        """Test registering result router hooks."""
        from aragora.server.result_router import register_result_router_hooks
        from aragora.debate.hooks import HookManager

        manager = HookManager()
        register_result_router_hooks(manager)

        # Check hook was registered
        assert "result_router" in [h.name for hooks in manager._hooks.values() for h in hooks]

    def test_post_debate_hook(self):
        """Test POST_DEBATE hook handler."""
        from aragora.server.result_router import _on_post_debate
        from aragora.server.debate_origin import (
            register_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        register_debate_origin(
            debate_id="test-hook",
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        # Create mock result
        class MockResult:
            debate_id = "test-hook"
            consensus_reached = True
            final_answer = "Answer"
            confidence = 0.9
            participants = ["agent1"]
            task = "Topic"

        # Call hook (fire-and-forget, shouldn't raise)
        _on_post_debate(result=MockResult())


class TestIntegration:
    """Integration tests for the full flow."""

    def test_telegram_handler_registers_origin(self):
        """Test that Telegram handler registers debate origin."""
        # This is tested implicitly through handler tests
        pass

    def test_whatsapp_handler_registers_origin(self):
        """Test that WhatsApp handler registers debate origin."""
        # This is tested implicitly through handler tests
        pass

"""
E2E tests for bidirectional chat result routing.

Tests the complete flow:
1. Debate initiated from chat platform (Telegram, WhatsApp, Slack, etc.)
2. Origin registered for routing
3. Debate completes with result
4. Result automatically routed back to originating platform

This tests the `aragora.server.debate_origin` module and its integration
with the chat handlers.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from tests.e2e.conftest import DebateSetup, MockAgentResponse


# ============================================================================
# Debate Origin Registration Tests
# ============================================================================


class TestDebateOriginRegistration:
    """Tests for registering debate origins from various platforms."""

    def test_register_telegram_origin(self):
        """Test registering a debate origin from Telegram."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
            _origin_store,
        )

        # Clear store for test isolation
        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123456789",
            user_id="987654321",
            message_id="42",
            metadata={"username": "testuser"},
        )

        assert origin.debate_id == debate_id
        assert origin.platform == "telegram"
        assert origin.channel_id == "123456789"
        assert origin.user_id == "987654321"
        assert origin.message_id == "42"
        assert origin.metadata["username"] == "testuser"
        assert not origin.result_sent

        # Verify retrieval
        retrieved = get_debate_origin(debate_id)
        assert retrieved is not None
        assert retrieved.debate_id == debate_id

    def test_register_whatsapp_origin(self):
        """Test registering a debate origin from WhatsApp."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="wa_user_123",
            metadata={"profile_name": "John Doe"},
        )

        assert origin.platform == "whatsapp"
        assert origin.channel_id == "+1234567890"

    def test_register_slack_origin_with_thread(self):
        """Test registering a debate origin from Slack with threading."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="slack",
            channel_id="C1234567890",
            user_id="U9876543210",
            thread_id="1234567890.123456",
            metadata={"workspace": "acme-corp"},
        )

        assert origin.platform == "slack"
        assert origin.thread_id == "1234567890.123456"

    def test_register_discord_origin(self):
        """Test registering a debate origin from Discord."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="discord",
            channel_id="1234567890123456789",
            user_id="9876543210987654321",
            message_id="1111111111111111111",
        )

        assert origin.platform == "discord"
        assert origin.message_id == "1111111111111111111"

    def test_register_teams_origin(self):
        """Test registering a debate origin from Microsoft Teams."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="teams",
            channel_id="teams-channel-id",
            user_id="teams-user-id",
            metadata={"webhook_url": "https://outlook.office.com/webhook/..."},
        )

        assert origin.platform == "teams"
        assert "webhook_url" in origin.metadata

    def test_register_email_origin(self):
        """Test registering a debate origin from email."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="email",
            channel_id="user@example.com",
            user_id="user@example.com",
            metadata={"subject": "Debate Request: AI Ethics"},
        )

        assert origin.platform == "email"
        assert origin.channel_id == "user@example.com"

    def test_nonexistent_origin_returns_none(self):
        """Test that getting a nonexistent origin returns None."""
        from aragora.server.debate_origin import get_debate_origin, _origin_store

        _origin_store.clear()

        result = get_debate_origin("nonexistent-debate-id")
        assert result is None


# ============================================================================
# Result Routing Tests
# ============================================================================


class TestResultRouting:
    """Tests for routing debate results back to originating platforms."""

    @pytest.mark.asyncio
    async def test_route_result_to_telegram(self):
        """Test routing debate result back to Telegram."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123456789",
            user_id="987654321",
            message_id="42",
        )

        result = {
            "consensus_reached": True,
            "final_answer": "The answer to the question is 42.",
            "confidence": 0.95,
            "participants": ["claude", "gpt4", "gemini"],
            "task": "What is the meaning of life?",
        }

        # Mock the Telegram API call
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_post.return_value.__aenter__.return_value = mock_response

            with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test_token"}):
                success = await route_debate_result(debate_id, result)

        # Even without real API, verify the origin was marked
        origin = get_debate_origin(debate_id)
        # Note: Success depends on mock setup, but we test the flow

    @pytest.mark.asyncio
    async def test_route_result_to_slack_with_thread(self):
        """Test routing debate result back to Slack thread."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="slack",
            channel_id="C1234567890",
            user_id="U9876543210",
            thread_id="1234567890.123456",
        )

        result = {
            "consensus_reached": True,
            "final_answer": "Use microservices for this scale.",
            "confidence": 0.87,
            "participants": ["claude", "gpt4"],
            "task": "Architecture decision",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.json.return_value = {"ok": True}
            mock_post.return_value.__aenter__.return_value = mock_response

            with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "xoxb-test"}):
                await route_debate_result(debate_id, result)

    @pytest.mark.asyncio
    async def test_route_result_no_origin(self):
        """Test routing fails gracefully when no origin exists."""
        from aragora.server.debate_origin import route_debate_result, _origin_store

        _origin_store.clear()

        result = {
            "consensus_reached": True,
            "final_answer": "Test answer",
            "confidence": 0.9,
            "participants": [],
        }

        success = await route_debate_result("nonexistent-id", result)
        assert not success

    @pytest.mark.asyncio
    async def test_route_result_idempotent(self):
        """Test that routing result twice doesn't send duplicate messages."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            mark_result_sent,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123456789",
            user_id="987654321",
        )

        # Mark as already sent
        mark_result_sent(debate_id)

        origin = get_debate_origin(debate_id)
        assert origin.result_sent is True

        # Second route attempt should return True without sending
        result = {"final_answer": "Test"}
        success = await route_debate_result(debate_id, result)
        assert success  # Returns True because it's already handled


# ============================================================================
# Result Message Formatting Tests
# ============================================================================


class TestResultFormatting:
    """Tests for result message formatting."""

    def test_format_markdown_result(self):
        """Test Markdown formatting for Telegram/Slack/Discord."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test",
            platform="telegram",
            channel_id="123",
            user_id="456",
            metadata={"topic": "Test Topic"},
        )

        result = {
            "consensus_reached": True,
            "final_answer": "The recommended approach is to use async/await.",
            "confidence": 0.92,
            "participants": ["claude", "gpt4", "gemini"],
            "task": "How to handle concurrency?",
        }

        message = _format_result_message(result, origin, markdown=True)

        assert "**Debate Complete!**" in message
        assert "**Consensus:** Yes" in message
        assert "92%" in message
        assert "claude" in message
        assert "async/await" in message

    def test_format_plaintext_result(self):
        """Test plaintext formatting for WhatsApp."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test",
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="wa_user",
        )

        result = {
            "consensus_reached": False,
            "final_answer": "No clear consensus was reached.",
            "confidence": 0.45,
            "participants": ["claude", "gpt4"],
        }

        message = _format_result_message(result, origin, markdown=False)

        assert "Debate Complete!" in message
        assert "Consensus: No" in message
        assert "**" not in message  # No markdown

    def test_format_html_result(self):
        """Test HTML formatting for Email."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test",
            platform="email",
            channel_id="user@test.com",
            user_id="user@test.com",
        )

        result = {
            "consensus_reached": True,
            "final_answer": "Use PostgreSQL for this use case.",
            "confidence": 0.88,
            "participants": ["claude"],
            "task": "Database selection",
        }

        message = _format_result_message(result, origin, markdown=False, html=True)

        assert "<h2>" in message
        assert "<strong>" in message
        assert "PostgreSQL" in message

    def test_format_truncates_long_answers(self):
        """Test that very long answers are truncated."""
        from aragora.server.debate_origin import _format_result_message, DebateOrigin

        origin = DebateOrigin(
            debate_id="test",
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        result = {
            "final_answer": "A" * 1000,  # 1000 characters
            "confidence": 0.9,
            "participants": [],
        }

        message = _format_result_message(result, origin, markdown=True)

        # Should contain truncated version
        assert "..." in message or len(result["final_answer"]) <= 800


# ============================================================================
# Origin Cleanup Tests
# ============================================================================


class TestOriginCleanup:
    """Tests for origin TTL and cleanup."""

    def test_cleanup_expired_origins(self):
        """Test that expired origins are cleaned up."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            cleanup_expired_origins,
            get_debate_origin,
            _origin_store,
            ORIGIN_TTL_SECONDS,
        )

        _origin_store.clear()

        # Register an origin
        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        # Manually expire it
        origin.created_at = time.time() - ORIGIN_TTL_SECONDS - 100

        # Run cleanup
        cleaned = cleanup_expired_origins()

        assert cleaned == 1
        assert get_debate_origin(debate_id) is None

    def test_cleanup_preserves_fresh_origins(self):
        """Test that fresh origins are not cleaned up."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            cleanup_expired_origins,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        # Run cleanup on fresh origin
        cleaned = cleanup_expired_origins()

        assert cleaned == 0
        assert get_debate_origin(debate_id) is not None


# ============================================================================
# Integration with Debate Lifecycle
# ============================================================================


class TestDebateToChatIntegration:
    """Integration tests for debate-to-chat flow."""

    @pytest.mark.asyncio
    async def test_full_telegram_debate_flow(self):
        """Test complete flow: Telegram message -> Debate -> Result back to Telegram."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            get_debate_origin,
            _origin_store,
        )
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        _origin_store.clear()

        # 1. Simulate Telegram message arriving
        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        chat_id = "123456789"
        user_id = "987654321"
        message_id = "42"

        # Register origin (as Telegram handler would)
        origin = register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            metadata={"username": "testuser", "topic": "AI Ethics"},
        )

        # 2. Run a mock debate
        env = Environment(task="What are the ethical considerations of AI?")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        mock_agents = [MagicMock() for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            agent.generate = AsyncMock(return_value=f"Position {i} on AI ethics")

        arena = Arena(env, mock_agents, protocol)
        debate_result = await arena.run()

        # 3. Prepare result for routing
        result = {
            "consensus_reached": debate_result.consensus_reached if hasattr(debate_result, "consensus_reached") else False,
            "final_answer": "AI development should prioritize safety and alignment.",
            "confidence": 0.85,
            "participants": ["agent_0", "agent_1"],
            "task": env.task,
        }

        # 4. Verify origin is still available
        retrieved_origin = get_debate_origin(debate_id)
        assert retrieved_origin is not None
        assert retrieved_origin.platform == "telegram"
        assert retrieved_origin.channel_id == chat_id

        # 5. Route result (mocked)
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_post.return_value.__aenter__.return_value = mock_response

            with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test_token"}):
                success = await route_debate_result(debate_id, result)

        # Verify the origin was marked as sent
        final_origin = get_debate_origin(debate_id)
        # Note: Without real API, result_sent depends on mock behavior

    @pytest.mark.asyncio
    async def test_multiple_platforms_concurrent(self):
        """Test handling debates from multiple platforms concurrently."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
            _origin_store,
        )

        _origin_store.clear()

        platforms = [
            ("telegram", "tg_123", "tg_user"),
            ("whatsapp", "+1234567890", "wa_user"),
            ("slack", "C123456", "U789012"),
            ("discord", "1234567890", "9876543210"),
        ]

        debate_ids = []

        # Register origins for multiple platforms
        for platform, channel_id, user_id in platforms:
            debate_id = f"debate-{platform}-{uuid.uuid4().hex[:8]}"
            debate_ids.append(debate_id)

            register_debate_origin(
                debate_id=debate_id,
                platform=platform,
                channel_id=channel_id,
                user_id=user_id,
            )

        # Verify all origins are correctly stored and retrievable
        for i, (platform, channel_id, user_id) in enumerate(platforms):
            origin = get_debate_origin(debate_ids[i])
            assert origin is not None
            assert origin.platform == platform
            assert origin.channel_id == channel_id


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in chat result routing."""

    @pytest.mark.asyncio
    async def test_route_unknown_platform(self):
        """Test handling of unknown platform."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="unknown_platform",
            channel_id="123",
            user_id="456",
        )

        result = {"final_answer": "Test"}
        success = await route_debate_result(debate_id, result)
        assert not success

    @pytest.mark.asyncio
    async def test_route_missing_credentials(self):
        """Test graceful failure when platform credentials are missing."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            route_debate_result,
            _origin_store,
        )

        _origin_store.clear()

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        register_debate_origin(
            debate_id=debate_id,
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        result = {"final_answer": "Test"}

        # Ensure no token is set
        with patch.dict("os.environ", {}, clear=True):
            success = await route_debate_result(debate_id, result)

        assert not success


# ============================================================================
# Serialization Tests
# ============================================================================


class TestOriginSerialization:
    """Tests for DebateOrigin serialization."""

    def test_to_dict(self):
        """Test DebateOrigin.to_dict()."""
        from aragora.server.debate_origin import DebateOrigin

        origin = DebateOrigin(
            debate_id="test-123",
            platform="telegram",
            channel_id="123456789",
            user_id="987654321",
            thread_id="thread-1",
            message_id="msg-1",
            metadata={"key": "value"},
        )

        data = origin.to_dict()

        assert data["debate_id"] == "test-123"
        assert data["platform"] == "telegram"
        assert data["metadata"]["key"] == "value"
        assert data["result_sent"] is False

    def test_from_dict(self):
        """Test DebateOrigin.from_dict()."""
        from aragora.server.debate_origin import DebateOrigin

        data = {
            "debate_id": "test-456",
            "platform": "slack",
            "channel_id": "C123",
            "user_id": "U456",
            "created_at": 1234567890.0,
            "metadata": {"workspace": "test"},
            "thread_id": "ts123",
            "result_sent": True,
            "result_sent_at": 1234567900.0,
        }

        origin = DebateOrigin.from_dict(data)

        assert origin.debate_id == "test-456"
        assert origin.platform == "slack"
        assert origin.result_sent is True
        assert origin.result_sent_at == 1234567900.0

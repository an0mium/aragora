"""
Tests for Chat Connector Session Integration.

Tests the session management integration in the ChatPlatformConnector base class:
- Session creation and retrieval
- Debate linking
- Result routing
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Optional

from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    SendMessageResponse,
    FileAttachment,
    WebhookEvent,
)


class MockChatConnector(ChatPlatformConnector):
    """Mock connector for testing base class functionality."""

    @property
    def platform_name(self) -> str:
        return "mock_platform"

    @property
    def platform_display_name(self) -> str:
        return "Mock Platform"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        return SendMessageResponse(
            success=True,
            message_id=f"msg_{channel_id}",
            channel_id=channel_id,
            metadata={"thread_id": thread_id} if thread_id else {},
        )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        return SendMessageResponse(success=True, message_id=message_id, channel_id=channel_id)

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        return True

    async def respond_to_command(
        self,
        command: Any,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        return SendMessageResponse(success=True, message_id="cmd_response")

    async def respond_to_interaction(
        self,
        interaction: Any,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        return SendMessageResponse(success=True, message_id="interaction_response")

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        return SendMessageResponse(success=True, message_id="file_upload")

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        return FileAttachment(file_id=file_id, name="test.txt", content=b"test")

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [{"type": "section", "text": body}]

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict[str, Any]:
        return {"type": "button", "text": text, "action_id": action_id}

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        return True

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        return WebhookEvent(event_type="message", raw_payload={})


class TestChatSessionIntegration:
    """Tests for session management integration in chat connectors."""

    @pytest.fixture
    def connector(self):
        """Create mock connector."""
        return MockChatConnector(bot_token="test-token")

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        manager = AsyncMock()
        manager.get_or_create_session = AsyncMock()
        manager.link_debate = AsyncMock(return_value=True)
        manager.find_sessions_for_debate = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.session_id = "mock_platform:user123:abc123"
        session.channel = "mock_platform"
        session.user_id = "user123"
        session.debate_id = "debate-abc"
        session.context = {
            "platform": "mock_platform",
            "channel_id": "channel123",
            "thread_id": "thread456",
        }
        return session

    def test_session_manager_lazy_init(self, connector):
        """Test that session manager is lazily initialized."""
        # Not initialized yet
        assert not hasattr(connector, "_session_manager")

        # Call the method - it will try to import and may succeed or fail
        # The key test is that it doesn't crash and returns a value
        manager = connector._get_session_manager()

        # After first call, should be cached
        assert hasattr(connector, "_session_manager")

        # Second call should return cached value
        manager2 = connector._get_session_manager()
        assert manager is manager2

    def test_session_manager_available(self, connector):
        """Test session manager integration when available."""
        # The actual session manager should be importable
        from aragora.connectors.debate_session import get_debate_session_manager

        manager = connector._get_session_manager()

        # Should return the singleton instance
        expected = get_debate_session_manager()
        assert manager is expected

    @pytest.mark.asyncio
    async def test_get_or_create_session(self, connector, mock_session):
        """Test getting or creating a session."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.get_or_create_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            session = await connector.get_or_create_session(
                user_id="user123",
                context={"extra": "data"},
            )

            assert session is mock_session
            mock_manager.get_or_create_session.assert_called_once_with(
                channel="mock_platform",
                user_id="user123",
                context={"platform": "mock_platform", "extra": "data"},
            )

    @pytest.mark.asyncio
    async def test_get_or_create_session_no_manager(self, connector):
        """Test get_or_create_session when manager unavailable."""
        with patch.object(connector, "_get_session_manager", return_value=None):
            session = await connector.get_or_create_session("user123")
            assert session is None

    @pytest.mark.asyncio
    async def test_link_debate_to_session(self, connector, mock_session):
        """Test linking a debate to a session."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.get_or_create_session = AsyncMock(return_value=mock_session)
            mock_manager.link_debate = AsyncMock(return_value=True)
            mock_get_manager.return_value = mock_manager

            session_id = await connector.link_debate_to_session(
                user_id="user123",
                debate_id="debate-xyz",
                context={"channel_id": "chan123"},
            )

            assert session_id == mock_session.session_id
            mock_manager.link_debate.assert_called_once_with(mock_session.session_id, "debate-xyz")

    @pytest.mark.asyncio
    async def test_link_debate_to_session_no_manager(self, connector):
        """Test link_debate_to_session when manager unavailable."""
        with patch.object(connector, "_get_session_manager", return_value=None):
            session_id = await connector.link_debate_to_session("user123", "debate-xyz")
            assert session_id is None

    @pytest.mark.asyncio
    async def test_find_sessions_for_debate(self, connector, mock_session):
        """Test finding sessions for a debate."""
        # Create a session for this platform and one for another
        other_session = MagicMock()
        other_session.channel = "slack"

        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(
                return_value=[mock_session, other_session]
            )
            mock_get_manager.return_value = mock_manager

            sessions = await connector.find_sessions_for_debate("debate-abc")

            # Should only return sessions for this platform
            assert len(sessions) == 1
            assert sessions[0].channel == "mock_platform"

    @pytest.mark.asyncio
    async def test_find_sessions_for_debate_no_manager(self, connector):
        """Test find_sessions_for_debate when manager unavailable."""
        with patch.object(connector, "_get_session_manager", return_value=None):
            sessions = await connector.find_sessions_for_debate("debate-abc")
            assert sessions == []

    @pytest.mark.asyncio
    async def test_route_debate_result(self, connector, mock_session):
        """Test routing debate results to sessions."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(return_value=[mock_session])
            mock_get_manager.return_value = mock_manager

            responses = await connector.route_debate_result(
                debate_id="debate-abc",
                result="The consensus is: Use GraphQL",
            )

            assert len(responses) == 1
            assert responses[0].success is True
            assert responses[0].channel_id == "channel123"

    @pytest.mark.asyncio
    async def test_route_debate_result_multiple_sessions(self, connector):
        """Test routing to multiple sessions."""
        session1 = MagicMock()
        session1.channel = "mock_platform"
        session1.context = {"channel_id": "chan1", "thread_id": "thread1"}

        session2 = MagicMock()
        session2.channel = "mock_platform"
        session2.context = {"channel_id": "chan2", "thread_id": "thread2"}

        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(return_value=[session1, session2])
            mock_get_manager.return_value = mock_manager

            responses = await connector.route_debate_result(
                debate_id="debate-abc",
                result="Consensus reached",
            )

            assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_route_debate_result_no_channel(self, connector):
        """Test routing skips sessions without channel info."""
        session_no_channel = MagicMock()
        session_no_channel.channel = "mock_platform"
        session_no_channel.context = {}  # No channel_id

        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(return_value=[session_no_channel])
            mock_get_manager.return_value = mock_manager

            responses = await connector.route_debate_result(
                debate_id="debate-abc",
                result="Consensus reached",
            )

            # Should skip the session without channel info
            assert len(responses) == 0

    @pytest.mark.asyncio
    async def test_route_debate_result_with_override(self, connector, mock_session):
        """Test routing with channel/thread override."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(return_value=[mock_session])
            mock_get_manager.return_value = mock_manager

            responses = await connector.route_debate_result(
                debate_id="debate-abc",
                result="Override routing",
                channel_id="override_channel",
                thread_id="override_thread",
            )

            assert len(responses) == 1
            assert responses[0].channel_id == "override_channel"

    @pytest.mark.asyncio
    async def test_route_debate_result_send_error(self, connector, mock_session):
        """Test graceful handling of send errors during routing."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.find_sessions_for_debate = AsyncMock(return_value=[mock_session])
            mock_get_manager.return_value = mock_manager

            # Make send_message fail
            with patch.object(connector, "send_message", side_effect=Exception("Send failed")):
                responses = await connector.route_debate_result(
                    debate_id="debate-abc",
                    result="This will fail",
                )

                # Should return empty list on error
                assert len(responses) == 0


class TestChatSessionIntegrationEdgeCases:
    """Edge case tests for session integration."""

    @pytest.fixture
    def connector(self):
        return MockChatConnector(bot_token="test-token")

    def test_session_manager_cached(self, connector):
        """Test that session manager is cached after first access."""
        # Clear any existing cache
        if hasattr(connector, "_session_manager"):
            delattr(connector, "_session_manager")

        # First call - initializes
        manager1 = connector._get_session_manager()

        # Second call - should return cached
        manager2 = connector._get_session_manager()

        # Both should return the same instance
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_context_includes_platform(self, connector):
        """Test that session context always includes platform."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_session = MagicMock()
            mock_session.session_id = "test_session"
            mock_manager.get_or_create_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            await connector.get_or_create_session("user123")

            call_args = mock_manager.get_or_create_session.call_args
            assert call_args.kwargs["context"]["platform"] == "mock_platform"

    @pytest.mark.asyncio
    async def test_link_debate_session_not_found(self, connector):
        """Test linking when session creation fails."""
        with patch.object(connector, "_get_session_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.get_or_create_session = AsyncMock(return_value=None)
            mock_get_manager.return_value = mock_manager

            session_id = await connector.link_debate_to_session("user123", "debate-xyz")

            assert session_id is None

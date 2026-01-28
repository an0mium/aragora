"""
Tests for Knowledge Chat Handler.

Tests cover:
- Knowledge search from chat context
- Knowledge injection into conversations
- Storing chat messages as knowledge
- Channel knowledge summaries
- Permission checks
- Error handling
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_chat import (
    KnowledgeChatHandler,
    handle_knowledge_search,
    handle_knowledge_inject,
    handle_store_chat_knowledge,
    handle_channel_knowledge_summary,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create handler instance."""
    return KnowledgeChatHandler({})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json"}
    return mock


@pytest.fixture
def mock_bridge():
    """Create mock knowledge chat bridge."""
    bridge = AsyncMock()
    return bridge


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"author": "user1", "content": "What's our vacation policy?"},
        {"author": "user2", "content": "I think it's in the handbook"},
    ]


@pytest.fixture
def sample_knowledge_context():
    """Sample knowledge context result."""
    context = MagicMock()
    context.to_dict.return_value = {
        "items": [
            {
                "content": "Vacation policy: 15 days per year",
                "confidence": 0.85,
                "source": "handbook",
            }
        ],
        "query": "vacation policy",
        "item_count": 1,
    }
    return context


# =============================================================================
# Test can_handle
# =============================================================================


class TestCanHandle:
    """Tests for path matching."""

    def test_handles_search_route(self, handler):
        """Test handler matches search route."""
        assert handler.can_handle("/api/v1/chat/knowledge/search") is True

    def test_handles_inject_route(self, handler):
        """Test handler matches inject route."""
        assert handler.can_handle("/api/v1/chat/knowledge/inject") is True

    def test_handles_store_route(self, handler):
        """Test handler matches store route."""
        assert handler.can_handle("/api/v1/chat/knowledge/store") is True

    def test_handles_channel_summary_route(self, handler):
        """Test handler matches channel summary routes."""
        assert handler.can_handle("/api/v1/chat/knowledge/channel/C123/summary") is True

    def test_rejects_other_paths(self, handler):
        """Test handler rejects unrelated paths."""
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/agents") is False
        assert handler.can_handle("/api/health") is False


# =============================================================================
# Test handle_knowledge_search
# =============================================================================


class TestKnowledgeSearch:
    """Tests for knowledge search function."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_bridge, sample_knowledge_context):
        """Test successful knowledge search."""
        mock_bridge.search_knowledge.return_value = sample_knowledge_context

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="vacation policy",
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert "items" in result
        mock_bridge.search_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_all_params(self, mock_bridge, sample_knowledge_context):
        """Test search with all parameters."""
        mock_bridge.search_knowledge.return_value = sample_knowledge_context

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="policy",
                workspace_id="ws_123",
                channel_id="C123",
                user_id="U456",
                scope="channel",
                strategy="semantic",
                node_types=["policy", "document"],
                min_confidence=0.5,
                max_results=20,
            )

        assert result["success"] is True
        mock_bridge.search_knowledge.assert_called_once()
        call_kwargs = mock_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["max_results"] == 20
        assert call_kwargs["min_confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_search_invalid_scope_falls_back(self, mock_bridge, sample_knowledge_context):
        """Test search with invalid scope falls back to workspace."""
        mock_bridge.search_knowledge.return_value = sample_knowledge_context

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="test",
                scope="invalid_scope",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_bridge):
        """Test search error handling returns sanitized error."""
        mock_bridge.search_knowledge.side_effect = Exception("Search failed")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(query="test")

        assert result["success"] is False
        assert "error" in result
        # Error message is sanitized by safe_error_message
        assert isinstance(result["error"], str)


# =============================================================================
# Test handle_knowledge_inject
# =============================================================================


class TestKnowledgeInject:
    """Tests for knowledge injection function."""

    @pytest.mark.asyncio
    async def test_inject_success(self, mock_bridge, sample_messages):
        """Test successful knowledge injection."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"content": "Relevant info", "score": 0.9}
        mock_bridge.inject_knowledge_for_conversation.return_value = [mock_result]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=sample_messages,
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["item_count"] == 1
        assert "context" in result
        mock_bridge.inject_knowledge_for_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_empty_results(self, mock_bridge, sample_messages):
        """Test injection with no relevant knowledge."""
        mock_bridge.inject_knowledge_for_conversation.return_value = []

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=sample_messages,
            )

        assert result["success"] is True
        assert result["item_count"] == 0
        assert result["context"] == []

    @pytest.mark.asyncio
    async def test_inject_error_handling(self, mock_bridge, sample_messages):
        """Test injection error handling."""
        mock_bridge.inject_knowledge_for_conversation.side_effect = Exception("Injection failed")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(messages=sample_messages)

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Test handle_store_chat_knowledge
# =============================================================================


class TestStoreKnowledge:
    """Tests for storing chat knowledge."""

    @pytest.mark.asyncio
    async def test_store_success(self, mock_bridge, sample_messages):
        """Test successful knowledge storage."""
        mock_bridge.store_chat_as_knowledge.return_value = "node_123"

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=sample_messages,
                workspace_id="ws_123",
                channel_id="C123",
                channel_name="#general",
                platform="slack",
            )

        assert result["success"] is True
        assert result["node_id"] == "node_123"
        assert result["message_count"] == 2
        mock_bridge.store_chat_as_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_minimum_messages_required(self, mock_bridge):
        """Test that at least 2 messages are required."""
        result = await handle_store_chat_knowledge(
            messages=[{"author": "user1", "content": "Single message"}],
        )

        assert result["success"] is False
        assert "At least 2 messages" in result["error"]

    @pytest.mark.asyncio
    async def test_store_empty_messages(self, mock_bridge):
        """Test storage with empty messages."""
        result = await handle_store_chat_knowledge(messages=[])

        assert result["success"] is False
        assert "At least 2 messages" in result["error"]

    @pytest.mark.asyncio
    async def test_store_failed(self, mock_bridge, sample_messages):
        """Test storage failure."""
        mock_bridge.store_chat_as_knowledge.return_value = None

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(messages=sample_messages)

        assert result["success"] is False
        assert "Failed to store" in result["error"]

    @pytest.mark.asyncio
    async def test_store_error_handling(self, mock_bridge, sample_messages):
        """Test storage error handling."""
        mock_bridge.store_chat_as_knowledge.side_effect = Exception("Storage error")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(messages=sample_messages)

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Test handle_channel_knowledge_summary
# =============================================================================


class TestChannelSummary:
    """Tests for channel knowledge summary."""

    @pytest.mark.asyncio
    async def test_summary_success(self, mock_bridge):
        """Test successful channel summary."""
        mock_bridge.get_channel_knowledge_summary.return_value = {
            "channel_id": "C123",
            "item_count": 5,
            "topics": ["policy", "procedures"],
        }

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="C123",
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["channel_id"] == "C123"
        assert result["item_count"] == 5

    @pytest.mark.asyncio
    async def test_summary_with_max_items(self, mock_bridge):
        """Test summary with custom max_items."""
        mock_bridge.get_channel_knowledge_summary.return_value = {
            "channel_id": "C123",
            "item_count": 20,
        }

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="C123",
                max_items=20,
            )

        assert result["success"] is True
        call_kwargs = mock_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["max_items"] == 20

    @pytest.mark.asyncio
    async def test_summary_error_handling(self, mock_bridge):
        """Test summary error handling."""
        mock_bridge.get_channel_knowledge_summary.side_effect = Exception("Summary failed")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(channel_id="C123")

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Test KnowledgeChatHandler class
# =============================================================================


class TestKnowledgeChatHandler:
    """Tests for KnowledgeChatHandler class."""

    def test_routes_defined(self, handler):
        """Test that routes are properly defined."""
        assert "/api/v1/chat/knowledge/search" in handler.ROUTES
        assert "/api/v1/chat/knowledge/inject" in handler.ROUTES
        assert "/api/v1/chat/knowledge/store" in handler.ROUTES

    def test_route_prefixes_defined(self, handler):
        """Test that route prefixes are defined."""
        assert any("/api/v1/chat/knowledge/channel/" in p for p in handler.ROUTE_PREFIXES)

    def test_permission_keys_defined(self, handler):
        """Test that permission keys are defined."""
        assert handler.KNOWLEDGE_READ_PERMISSION == "knowledge.read"
        assert handler.KNOWLEDGE_WRITE_PERMISSION == "knowledge.write"

    def test_can_handle_all_routes(self, handler):
        """Test can_handle returns True for all defined routes."""
        for route in handler.ROUTES:
            assert handler.can_handle(route) is True

    def test_can_handle_channel_prefix_routes(self, handler):
        """Test can_handle works with channel prefix routes."""
        assert handler.can_handle("/api/v1/chat/knowledge/channel/ABC123/summary") is True
        assert handler.can_handle("/api/v1/chat/knowledge/channel/xyz/summary") is True


# =============================================================================
# Test Permission Integration
# =============================================================================


class TestPermissions:
    """Tests for RBAC permission integration."""

    @pytest.mark.asyncio
    async def test_handler_requires_permission(self, handler, mock_http_handler):
        """Test that handler checks permissions."""
        # Mock require_permission_or_error to return an error
        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, MagicMock(status_code=403)),
        ):
            result = await handler.handle(
                "/api/v1/chat/knowledge/channel/C123/summary",
                {},
                mock_http_handler,
            )

            assert result is not None
            assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_handler_read_permission_checked(self, handler, mock_http_handler):
        """Test that GET requests check read permission."""
        with patch.object(
            handler,
            "require_permission_or_error",
        ) as mock_perm:
            mock_perm.return_value = (None, MagicMock(status_code=403))

            await handler.handle(
                "/api/v1/chat/knowledge/channel/C123/summary",
                {},
                mock_http_handler,
            )

            # Should have been called with read permission
            mock_perm.assert_called_with(mock_http_handler, "knowledge.read")

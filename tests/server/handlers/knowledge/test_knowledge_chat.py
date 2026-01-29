"""
Tests for KnowledgeChatHandler.

Tests cover:
- Knowledge search endpoint
- Knowledge injection endpoint
- Knowledge storage endpoint
- Channel knowledge summary endpoint
- RBAC permission checks
- Input validation and bounds checking
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.knowledge_chat import (
    KnowledgeChatHandler,
    handle_knowledge_search,
    handle_knowledge_inject,
    handle_store_chat_knowledge,
    handle_channel_knowledge_summary,
    MAX_RESULTS_LIMIT,
    MAX_CONTEXT_ITEMS_LIMIT,
    MAX_ITEMS_LIMIT,
)


@pytest.fixture
def handler():
    """Create a KnowledgeChatHandler instance."""
    return KnowledgeChatHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with auth context."""
    mock = MagicMock()
    mock.headers = {"Authorization": "Bearer test-token"}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


@pytest.fixture
def mock_bridge():
    """Create a mock knowledge chat bridge."""
    mock = MagicMock()
    mock.search_knowledge = AsyncMock()
    mock.inject_knowledge_for_conversation = AsyncMock()
    mock.store_chat_as_knowledge = AsyncMock()
    mock.get_channel_knowledge_summary = AsyncMock()
    return mock


class TestKnowledgeChatHandlerRouting:
    """Test handler routing logic."""

    def test_can_handle_search_route(self, handler):
        """Test handler can handle search route."""
        assert handler.can_handle("/api/v1/chat/knowledge/search")

    def test_can_handle_inject_route(self, handler):
        """Test handler can handle inject route."""
        assert handler.can_handle("/api/v1/chat/knowledge/inject")

    def test_can_handle_store_route(self, handler):
        """Test handler can handle store route."""
        assert handler.can_handle("/api/v1/chat/knowledge/store")

    def test_can_handle_channel_summary_route(self, handler):
        """Test handler can handle channel summary route."""
        assert handler.can_handle("/api/v1/chat/knowledge/channel/test-channel/summary")

    def test_cannot_handle_unknown_route(self, handler):
        """Test handler rejects unknown routes."""
        assert not handler.can_handle("/api/v1/unknown/route")
        assert not handler.can_handle("/api/v1/chat/knowledge")
        assert not handler.can_handle("/api/v1/chat/knowledge/channel")


class TestKnowledgeSearchFunction:
    """Test the knowledge search function."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_bridge):
        """Test successful knowledge search."""
        mock_context = MagicMock()
        mock_context.to_dict.return_value = {
            "results": [{"id": "node-1", "content": "Test content"}],
            "total": 1,
        }
        mock_bridge.search_knowledge.return_value = mock_context

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="test query",
                workspace_id="ws-123",
                max_results=10,
            )

        assert result["success"] is True
        assert "results" in result
        mock_bridge.search_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_all_parameters(self, mock_bridge):
        """Test search with all optional parameters."""
        mock_context = MagicMock()
        mock_context.to_dict.return_value = {"results": [], "total": 0}
        mock_bridge.search_knowledge.return_value = mock_context

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="test",
                workspace_id="ws-123",
                channel_id="ch-456",
                user_id="user-789",
                scope="channel",
                strategy="semantic",
                node_types=["policy", "document"],
                min_confidence=0.5,
                max_results=20,
            )

        assert result["success"] is True
        call_kwargs = mock_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["channel_id"] == "ch-456"
        assert call_kwargs["user_id"] == "user-789"
        assert call_kwargs["node_types"] == ["policy", "document"]
        assert call_kwargs["min_confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_search_invalid_scope_uses_default(self, mock_bridge):
        """Test invalid scope falls back to default."""
        mock_context = MagicMock()
        mock_context.to_dict.return_value = {"results": []}
        mock_bridge.search_knowledge.return_value = mock_context

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
        """Test search handles errors gracefully."""
        mock_bridge.search_knowledge.side_effect = Exception("Bridge error")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(query="test")

        assert result["success"] is False
        assert "error" in result


class TestKnowledgeInjectFunction:
    """Test the knowledge injection function."""

    @pytest.mark.asyncio
    async def test_inject_success(self, mock_bridge):
        """Test successful knowledge injection."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"id": "ctx-1", "content": "Relevant info"}
        mock_bridge.inject_knowledge_for_conversation.return_value = [mock_result]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=[
                    {"author": "user1", "content": "Question?"},
                    {"author": "user2", "content": "Answer!"},
                ],
                workspace_id="ws-123",
            )

        assert result["success"] is True
        assert result["item_count"] == 1
        assert len(result["context"]) == 1

    @pytest.mark.asyncio
    async def test_inject_with_channel(self, mock_bridge):
        """Test injection with channel context."""
        mock_bridge.inject_knowledge_for_conversation.return_value = []

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=[{"author": "user", "content": "test"}],
                workspace_id="ws-123",
                channel_id="ch-456",
                max_context_items=3,
            )

        assert result["success"] is True
        call_kwargs = mock_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["channel_id"] == "ch-456"
        assert call_kwargs["max_context_items"] == 3

    @pytest.mark.asyncio
    async def test_inject_error_handling(self, mock_bridge):
        """Test injection handles errors gracefully."""
        mock_bridge.inject_knowledge_for_conversation.side_effect = Exception("Injection failed")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=[{"content": "test"}],
            )

        assert result["success"] is False
        assert "error" in result


class TestStoreChatKnowledgeFunction:
    """Test the store chat knowledge function."""

    @pytest.mark.asyncio
    async def test_store_success(self, mock_bridge):
        """Test successful knowledge storage."""
        mock_bridge.store_chat_as_knowledge.return_value = "node-12345"

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=[
                    {"author": "user1", "content": "We decided X"},
                    {"author": "user2", "content": "Agreed"},
                ],
                workspace_id="ws-123",
                channel_id="ch-456",
                channel_name="#decisions",
                platform="slack",
            )

        assert result["success"] is True
        assert result["node_id"] == "node-12345"
        assert result["message_count"] == 2

    @pytest.mark.asyncio
    async def test_store_requires_minimum_messages(self, mock_bridge):
        """Test storage requires at least 2 messages."""
        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=[{"author": "user1", "content": "Only one message"}],
            )

        assert result["success"] is False
        assert "2 messages required" in result["error"]
        mock_bridge.store_chat_as_knowledge.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_failure_returns_error(self, mock_bridge):
        """Test storage failure returns error."""
        mock_bridge.store_chat_as_knowledge.return_value = None  # Indicates failure

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=[
                    {"content": "msg1"},
                    {"content": "msg2"},
                ],
            )

        assert result["success"] is False
        assert "Failed to store" in result["error"]

    @pytest.mark.asyncio
    async def test_store_with_custom_node_type(self, mock_bridge):
        """Test storage with custom node type."""
        mock_bridge.store_chat_as_knowledge.return_value = "node-custom"

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=[{"content": "m1"}, {"content": "m2"}],
                node_type="decision",
            )

        assert result["success"] is True
        call_kwargs = mock_bridge.store_chat_as_knowledge.call_args.kwargs
        assert call_kwargs["node_type"] == "decision"


class TestChannelKnowledgeSummaryFunction:
    """Test the channel knowledge summary function."""

    @pytest.mark.asyncio
    async def test_summary_success(self, mock_bridge):
        """Test successful channel summary."""
        mock_bridge.get_channel_knowledge_summary.return_value = {
            "channel_id": "ch-123",
            "knowledge_items": 5,
            "topics": ["python", "testing"],
        }

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="ch-123",
                workspace_id="ws-456",
                max_items=10,
            )

        assert result["success"] is True
        assert result["channel_id"] == "ch-123"
        assert result["knowledge_items"] == 5

    @pytest.mark.asyncio
    async def test_summary_error_handling(self, mock_bridge):
        """Test summary handles errors gracefully."""
        mock_bridge.get_channel_knowledge_summary.side_effect = Exception("Summary failed")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="ch-123",
            )

        assert result["success"] is False
        assert "error" in result


class TestKnowledgeChatHandlerRBAC:
    """Test RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_search_requires_read_permission(self, handler, mock_http_handler):
        """Test search endpoint requires read permission."""
        from aragora.server.handlers.utils.responses import HandlerResult

        mock_http_handler.rfile.read.return_value = b'{"query": "test"}'
        forbidden_result = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error": "Forbidden"}'
        )

        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, forbidden_result),
        ):
            result = await handler.handle_post(
                "/api/v1/chat/knowledge/search",
                {},
                mock_http_handler,
            )

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_store_requires_write_permission(self, handler, mock_http_handler):
        """Test store endpoint requires write permission."""
        from aragora.server.handlers.utils.responses import HandlerResult

        mock_http_handler.rfile.read.return_value = b'{"messages": [{"c": "1"}, {"c": "2"}]}'
        forbidden_result = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error": "Forbidden"}'
        )

        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, forbidden_result),
        ):
            result = await handler.handle_post(
                "/api/v1/chat/knowledge/store",
                {},
                mock_http_handler,
            )

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_get_requires_read_permission(self, handler, mock_http_handler):
        """Test GET endpoint requires read permission."""
        from aragora.server.handlers.utils.responses import HandlerResult

        forbidden_result = HandlerResult(
            status_code=403, content_type="application/json", body=b'{"error": "Forbidden"}'
        )
        with patch.object(
            handler,
            "require_permission_or_error",
            return_value=(None, forbidden_result),
        ):
            result = await handler.handle(
                "/api/v1/chat/knowledge/channel/test/summary",
                {},
                mock_http_handler,
            )

        assert result.status_code == 403


class TestKnowledgeChatHandlerInputValidation:
    """Test input validation and bounds checking."""

    @pytest.mark.asyncio
    async def test_search_validates_query_required(self, handler, mock_http_handler):
        """Test search requires query parameter."""
        import json

        mock_http_handler.rfile.read.return_value = b"{}"

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            result = await handler.handle_post(
                "/api/v1/chat/knowledge/search",
                {},
                mock_http_handler,
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_inject_validates_messages_required(self, handler, mock_http_handler):
        """Test inject requires messages parameter."""
        import json

        mock_http_handler.rfile.read.return_value = b"{}"

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            result = await handler.handle_post(
                "/api/v1/chat/knowledge/inject",
                {},
                mock_http_handler,
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "messages is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_store_validates_minimum_messages(self, handler, mock_http_handler):
        """Test store validates minimum message count."""
        import json

        mock_http_handler.rfile.read.return_value = b'{"messages": [{"content": "one"}]}'

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            result = await handler.handle_post(
                "/api/v1/chat/knowledge/store",
                {},
                mock_http_handler,
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "2 messages required" in body.get("error", "")

    def test_max_results_limit_constant(self):
        """Test MAX_RESULTS_LIMIT is reasonable."""
        assert MAX_RESULTS_LIMIT == 100

    def test_max_context_items_limit_constant(self):
        """Test MAX_CONTEXT_ITEMS_LIMIT is reasonable."""
        assert MAX_CONTEXT_ITEMS_LIMIT == 50

    def test_max_items_limit_constant(self):
        """Test MAX_ITEMS_LIMIT is reasonable."""
        assert MAX_ITEMS_LIMIT == 100


class TestKnowledgeChatHandlerChannelSummary:
    """Test channel summary endpoint."""

    @pytest.mark.asyncio
    async def test_channel_summary_extracts_channel_id(
        self, handler, mock_http_handler, mock_bridge
    ):
        """Test channel ID is extracted from path."""
        mock_bridge.get_channel_knowledge_summary.return_value = {
            "channel_id": "test-channel-123",
            "items": [],
        }

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            with patch(
                "aragora.server.handlers.knowledge_chat._get_bridge",
                return_value=mock_bridge,
            ):
                result = await handler.handle(
                    "/api/v1/chat/knowledge/channel/test-channel-123/summary",
                    {"workspace_id": "ws-1"},
                    mock_http_handler,
                )

        # Verify the channel_id was extracted correctly
        call_kwargs = mock_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["channel_id"] == "test-channel-123"

    @pytest.mark.asyncio
    async def test_channel_summary_clamps_max_items(self, handler, mock_http_handler, mock_bridge):
        """Test max_items is clamped to bounds."""
        mock_bridge.get_channel_knowledge_summary.return_value = {"items": []}

        with patch.object(handler, "require_permission_or_error", return_value=(True, None)):
            with patch(
                "aragora.server.handlers.knowledge_chat._get_bridge",
                return_value=mock_bridge,
            ):
                # Request more than MAX_ITEMS_LIMIT
                await handler.handle(
                    "/api/v1/chat/knowledge/channel/ch/summary",
                    {"max_items": "999"},
                    mock_http_handler,
                )

        call_kwargs = mock_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["max_items"] <= MAX_ITEMS_LIMIT


__all__ = [
    "TestKnowledgeChatHandlerRouting",
    "TestKnowledgeSearchFunction",
    "TestKnowledgeInjectFunction",
    "TestStoreChatKnowledgeFunction",
    "TestChannelKnowledgeSummaryFunction",
    "TestKnowledgeChatHandlerRBAC",
    "TestKnowledgeChatHandlerInputValidation",
    "TestKnowledgeChatHandlerChannelSummary",
]

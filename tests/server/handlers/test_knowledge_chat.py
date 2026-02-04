"""Tests for the KnowledgeChatHandler."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.knowledge_chat import (
    KnowledgeChatHandler,
    _get_bridge,
    MAX_RESULTS_LIMIT,
    MAX_CONTEXT_ITEMS_LIMIT,
)


class TestKnowledgeChatHandler:
    """Tests for KnowledgeChatHandler."""

    def _make_handler(self, ctx: dict | None = None) -> KnowledgeChatHandler:
        return KnowledgeChatHandler(ctx=ctx)

    # -------------------------------------------------------------------------
    # ROUTES tests
    # -------------------------------------------------------------------------

    def test_routes_include_search(self):
        handler = self._make_handler()
        assert "/api/v1/chat/knowledge/search" in handler.ROUTES

    def test_routes_include_inject(self):
        handler = self._make_handler()
        assert "/api/v1/chat/knowledge/inject" in handler.ROUTES

    def test_routes_include_store(self):
        handler = self._make_handler()
        assert "/api/v1/chat/knowledge/store" in handler.ROUTES

    def test_route_prefixes_include_channel(self):
        handler = self._make_handler()
        assert "/api/v1/chat/knowledge/channel/" in handler.ROUTE_PREFIXES

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_init_with_context(self):
        ctx = {"key": "value"}
        handler = KnowledgeChatHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_init_without_context(self):
        handler = KnowledgeChatHandler()
        assert handler.ctx == {}

    # -------------------------------------------------------------------------
    # Permission constants tests
    # -------------------------------------------------------------------------

    def test_knowledge_read_permission(self):
        handler = self._make_handler()
        assert handler.KNOWLEDGE_READ_PERMISSION == "knowledge.read"

    def test_knowledge_write_permission(self):
        handler = self._make_handler()
        assert handler.KNOWLEDGE_WRITE_PERMISSION == "knowledge.write"

    # -------------------------------------------------------------------------
    # Bridge tests
    # -------------------------------------------------------------------------

    def test_get_bridge_creates_bridge(self):
        with patch("aragora.server.handlers.knowledge_chat._bridge", None):
            with patch(
                "aragora.services.knowledge_chat_bridge.get_knowledge_chat_bridge"
            ) as mock_get:
                mock_bridge = MagicMock()
                mock_get.return_value = mock_bridge

                result = _get_bridge()

                mock_get.assert_called_once()
                assert result is mock_bridge


class TestKnowledgeChatConstants:
    """Tests for module constants."""

    def test_max_results_limit(self):
        assert MAX_RESULTS_LIMIT == 100

    def test_max_context_items_limit(self):
        assert MAX_CONTEXT_ITEMS_LIMIT == 50


class TestHandleKnowledgeSearch:
    """Tests for the handle_knowledge_search function."""

    @pytest.mark.asyncio
    async def test_search_success(self):
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        mock_context = MagicMock()
        mock_context.to_dict.return_value = {
            "results": [],
            "result_count": 0,
        }

        mock_bridge = MagicMock()
        mock_bridge.search_knowledge = AsyncMock(return_value=mock_context)

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            # Access the unwrapped function to bypass decorators
            result = await handle_knowledge_search.__wrapped__(
                query="test query",
                workspace_id="ws-1",
            )

        assert result["success"] is True
        mock_bridge.search_knowledge.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_error(self):
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        mock_bridge = MagicMock()
        mock_bridge.search_knowledge = AsyncMock(side_effect=RuntimeError("Search failed"))

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search.__wrapped__(
                query="test query",
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_with_all_params(self):
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        mock_context = MagicMock()
        mock_context.to_dict.return_value = {
            "results": [{"id": "1"}],
            "result_count": 1,
        }

        mock_bridge = MagicMock()
        mock_bridge.search_knowledge = AsyncMock(return_value=mock_context)

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search.__wrapped__(
                query="test query",
                workspace_id="ws-1",
                channel_id="ch-1",
                user_id="user-1",
                scope="channel",
                strategy="semantic",
                node_types=["document", "policy"],
                min_confidence=0.5,
                max_results=20,
            )

        assert result["success"] is True
        call_kwargs = mock_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["workspace_id"] == "ws-1"
        assert call_kwargs["channel_id"] == "ch-1"
        assert call_kwargs["max_results"] == 20


class TestHandleKnowledgeInject:
    """Tests for the handle_knowledge_inject function."""

    @pytest.mark.asyncio
    async def test_inject_success(self):
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        mock_items = [MagicMock()]
        mock_items[0].to_dict.return_value = {"id": "1", "content": "test"}

        mock_bridge = MagicMock()
        mock_bridge.inject_knowledge_for_conversation = AsyncMock(return_value=mock_items)

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject.__wrapped__(
                messages=[{"author": "user1", "content": "test message"}],
                workspace_id="ws-1",
            )

        assert result["success"] is True
        assert result["item_count"] == 1

    @pytest.mark.asyncio
    async def test_inject_error(self):
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        mock_bridge = MagicMock()
        mock_bridge.inject_knowledge_for_conversation = AsyncMock(
            side_effect=RuntimeError("Inject failed")
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject.__wrapped__(
                messages=[{"author": "user1", "content": "test message"}],
            )

        assert result["success"] is False
        assert "error" in result


class TestHandleStoreChatKnowledge:
    """Tests for the handle_store_chat_knowledge function."""

    @pytest.mark.asyncio
    async def test_store_success(self):
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        mock_bridge = MagicMock()
        mock_bridge.store_chat_as_knowledge = AsyncMock(return_value="node-123")

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            # Need at least 2 messages
            result = await handle_store_chat_knowledge.__wrapped__(
                messages=[{"content": "test 1"}, {"content": "test 2"}],
                workspace_id="ws-1",
            )

        assert result["success"] is True
        assert result["node_id"] == "node-123"
        assert result["message_count"] == 2

    @pytest.mark.asyncio
    async def test_store_error(self):
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        mock_bridge = MagicMock()
        mock_bridge.store_chat_as_knowledge = AsyncMock(side_effect=RuntimeError("Store failed"))

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            # Need at least 2 messages
            result = await handle_store_chat_knowledge.__wrapped__(
                messages=[{"content": "test 1"}, {"content": "test 2"}],
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_store_requires_min_messages(self):
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        # Single message should fail
        result = await handle_store_chat_knowledge.__wrapped__(
            messages=[{"content": "single message"}],
        )

        assert result["success"] is False
        assert "at least 2 messages" in result["error"].lower()

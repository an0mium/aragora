"""Tests for KnowledgeChatHandler and standalone knowledge-chat endpoint functions.

Covers:
- KnowledgeChatHandler: can_handle routing, handle() for GET channel summary,
  handle_post() for search/inject/store, permission checks, input validation,
  parameter clamping, error handling
- handle_knowledge_search: success paths, enum fallbacks, exception handling
- handle_knowledge_inject: success paths, exception handling
- handle_store_chat_knowledge: success paths, min messages validation, null node_id
- handle_channel_knowledge_summary: success paths, exception handling
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_chat import (
    KnowledgeChatHandler,
    MAX_CONTEXT_ITEMS_LIMIT,
    MAX_ITEMS_LIMIT,
    MAX_RESULTS_LIMIT,
    handle_channel_knowledge_summary,
    handle_knowledge_inject,
    handle_knowledge_search,
    handle_store_chat_knowledge,
)
from aragora.server.handlers.utils.responses import HandlerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEARCH_PATH = "/api/v1/chat/knowledge/search"
INJECT_PATH = "/api/v1/chat/knowledge/inject"
STORE_PATH = "/api/v1/chat/knowledge/store"


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract JSON body dict from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _data(result: HandlerResult) -> dict[str, Any]:
    """Extract the 'data' payload from a success_response HandlerResult."""
    body = _body(result)
    return body.get("data", body)


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code."""
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for KnowledgeChatHandler tests."""

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        method: str = "GET",
        content_type: str = "application/json",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {
            "User-Agent": "test-agent",
            "Content-Type": content_type,
        }

        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile = io.BytesIO(body_bytes)
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile = io.BytesIO(b"{}")
            self.headers["Content-Length"] = "2"


def _make_search_result() -> dict[str, Any]:
    """Create a mock search result dict."""
    return {
        "channel_id": "C123",
        "workspace_id": "ws_1",
        "query": "test",
        "results": [
            {
                "node_id": "n1",
                "content": "Test content",
                "node_type": "fact",
                "confidence": 0.9,
                "relevance_score": 0.85,
                "source": "chat",
                "created_at": "2026-01-01T00:00:00Z",
                "metadata": {},
                "provenance": "direct",
            }
        ],
        "result_count": 1,
        "search_scope": "workspace",
        "search_time_ms": 12.5,
        "suggestions": [],
    }


@dataclass
class MockSearchContext:
    """Mock return from bridge.search_knowledge."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data


@dataclass
class MockKnowledgeItem:
    """Mock knowledge item returned from inject."""

    node_id: str = "n1"
    content: str = "Test knowledge"
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    """Reset the module-level _bridge singleton between tests."""
    import aragora.server.handlers.knowledge_chat as mod

    monkeypatch.setattr(mod, "_bridge", None)


@pytest.fixture
def mock_bridge():
    """Create a mock bridge with all methods."""
    bridge = AsyncMock()
    bridge.search_knowledge = AsyncMock(return_value=MockSearchContext(_make_search_result()))
    bridge.inject_knowledge_for_conversation = AsyncMock(
        return_value=[MockKnowledgeItem(), MockKnowledgeItem(node_id="n2")]
    )
    bridge.store_chat_as_knowledge = AsyncMock(return_value="node_abc123")
    bridge.get_channel_knowledge_summary = AsyncMock(
        return_value={
            "channel_id": "C123",
            "total_items": 5,
            "topics": ["engineering", "policy"],
        }
    )
    return bridge


@pytest.fixture
def patch_bridge(mock_bridge):
    """Patch _get_bridge to return our mock."""
    with patch(
        "aragora.server.handlers.knowledge_chat._get_bridge",
        return_value=mock_bridge,
    ):
        yield mock_bridge


@pytest.fixture
def handler():
    """Create a KnowledgeChatHandler instance."""
    return KnowledgeChatHandler(ctx={})


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Test KnowledgeChatHandler.can_handle routing."""

    def test_search_path(self, handler):
        assert handler.can_handle(SEARCH_PATH) is True

    def test_inject_path(self, handler):
        assert handler.can_handle(INJECT_PATH) is True

    def test_store_path(self, handler):
        assert handler.can_handle(STORE_PATH) is True

    def test_channel_wildcard(self, handler):
        assert handler.can_handle("/api/v1/chat/knowledge/channel/*") is True

    def test_channel_summary_wildcard(self, handler):
        assert handler.can_handle("/api/v1/chat/knowledge/channel/*/summary") is True

    def test_channel_prefix_match(self, handler):
        assert handler.can_handle("/api/v1/chat/knowledge/channel/C123/summary") is True

    def test_channel_prefix_match_any_id(self, handler):
        assert handler.can_handle("/api/v1/chat/knowledge/channel/my-channel-99/info") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_similar_but_different_path(self, handler):
        assert handler.can_handle("/api/v1/chat/knowledge") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False


# ===========================================================================
# Tests: handle_knowledge_search (standalone function)
# ===========================================================================


class TestHandleKnowledgeSearch:
    """Test the standalone search function."""

    @pytest.mark.asyncio
    async def test_search_success(self, patch_bridge):
        result = await handle_knowledge_search(query="test query")
        assert result["success"] is True
        assert "results" in result

    @pytest.mark.asyncio
    async def test_search_passes_all_params(self, patch_bridge):
        await handle_knowledge_search(
            query="policy",
            workspace_id="ws_42",
            channel_id="C999",
            user_id="U001",
            scope="channel",
            strategy="semantic",
            node_types=["fact", "policy"],
            min_confidence=0.5,
            max_results=20,
        )
        patch_bridge.search_knowledge.assert_awaited_once()
        call_kwargs = patch_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["query"] == "policy"
        assert call_kwargs["workspace_id"] == "ws_42"
        assert call_kwargs["channel_id"] == "C999"
        assert call_kwargs["user_id"] == "U001"
        assert call_kwargs["max_results"] == 20

    @pytest.mark.asyncio
    async def test_search_invalid_scope_falls_back(self, patch_bridge):
        """Invalid scope enum value falls back to WORKSPACE."""
        result = await handle_knowledge_search(query="test", scope="NOT_A_SCOPE")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_invalid_strategy_falls_back(self, patch_bridge):
        """Invalid strategy enum value falls back to HYBRID."""
        result = await handle_knowledge_search(query="test", strategy="NOT_A_STRATEGY")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_key_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = KeyError("missing key")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_value_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = ValueError("bad value")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_type_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = TypeError("bad type")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_attribute_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = AttributeError("no attr")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_runtime_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = RuntimeError("broken")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_os_error(self, patch_bridge):
        patch_bridge.search_knowledge.side_effect = OSError("disk fail")
        result = await handle_knowledge_search(query="test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_default_params(self, patch_bridge):
        """Defaults: workspace_id='default', scope='workspace', strategy='hybrid'."""
        await handle_knowledge_search(query="q")
        call_kwargs = patch_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"
        assert call_kwargs["min_confidence"] == 0.3
        assert call_kwargs["max_results"] == 10


# ===========================================================================
# Tests: handle_knowledge_inject (standalone function)
# ===========================================================================


class TestHandleKnowledgeInject:
    """Test the standalone inject function."""

    @pytest.mark.asyncio
    async def test_inject_success(self, patch_bridge):
        msgs = [{"author": "u1", "content": "Hello"}]
        result = await handle_knowledge_inject(messages=msgs)
        assert result["success"] is True
        assert result["item_count"] == 2
        assert len(result["context"]) == 2

    @pytest.mark.asyncio
    async def test_inject_passes_params(self, patch_bridge):
        msgs = [{"author": "u1", "content": "test"}]
        await handle_knowledge_inject(
            messages=msgs,
            workspace_id="ws_99",
            channel_id="C555",
            max_context_items=15,
        )
        call_kwargs = patch_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws_99"
        assert call_kwargs["channel_id"] == "C555"
        assert call_kwargs["max_context_items"] == 15

    @pytest.mark.asyncio
    async def test_inject_empty_results(self, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.return_value = []
        result = await handle_knowledge_inject(messages=[{"content": "hi"}])
        assert result["success"] is True
        assert result["item_count"] == 0
        assert result["context"] == []

    @pytest.mark.asyncio
    async def test_inject_key_error(self, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.side_effect = KeyError("fail")
        result = await handle_knowledge_inject(messages=[{"content": "hi"}])
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_inject_value_error(self, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.side_effect = ValueError("bad")
        result = await handle_knowledge_inject(messages=[{"content": "hi"}])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_inject_runtime_error(self, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.side_effect = RuntimeError("oops")
        result = await handle_knowledge_inject(messages=[{"content": "hi"}])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_inject_os_error(self, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.side_effect = OSError("io")
        result = await handle_knowledge_inject(messages=[{"content": "hi"}])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_inject_defaults(self, patch_bridge):
        await handle_knowledge_inject(messages=[{"content": "x"}])
        call_kwargs = patch_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"
        assert call_kwargs["channel_id"] is None
        assert call_kwargs["max_context_items"] == 5


# ===========================================================================
# Tests: handle_store_chat_knowledge (standalone function)
# ===========================================================================


class TestHandleStoreChatKnowledge:
    """Test the standalone store function."""

    @pytest.mark.asyncio
    async def test_store_success(self, patch_bridge):
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is True
        assert result["node_id"] == "node_abc123"
        assert result["message_count"] == 2

    @pytest.mark.asyncio
    async def test_store_too_few_messages_zero(self, patch_bridge):
        result = await handle_store_chat_knowledge(messages=[])
        assert result["success"] is False
        assert "2 messages" in result["error"]

    @pytest.mark.asyncio
    async def test_store_too_few_messages_one(self, patch_bridge):
        result = await handle_store_chat_knowledge(messages=[{"content": "only one"}])
        assert result["success"] is False
        assert "2 messages" in result["error"]

    @pytest.mark.asyncio
    async def test_store_exactly_two_messages(self, patch_bridge):
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_many_messages(self, patch_bridge):
        msgs = [{"content": f"msg_{i}"} for i in range(20)]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is True
        assert result["message_count"] == 20

    @pytest.mark.asyncio
    async def test_store_passes_all_params(self, patch_bridge):
        msgs = [{"content": "a"}, {"content": "b"}]
        await handle_store_chat_knowledge(
            messages=msgs,
            workspace_id="ws_77",
            channel_id="C100",
            channel_name="#general",
            platform="slack",
            node_type="decision",
        )
        call_kwargs = patch_bridge.store_chat_as_knowledge.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws_77"
        assert call_kwargs["channel_id"] == "C100"
        assert call_kwargs["channel_name"] == "#general"
        assert call_kwargs["platform"] == "slack"
        assert call_kwargs["node_type"] == "decision"

    @pytest.mark.asyncio
    async def test_store_null_node_id(self, patch_bridge):
        """When bridge returns None/empty, report failure."""
        patch_bridge.store_chat_as_knowledge.return_value = None
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is False
        assert "Failed to store" in result["error"]

    @pytest.mark.asyncio
    async def test_store_empty_string_node_id(self, patch_bridge):
        """Empty string node_id is falsy, should report failure."""
        patch_bridge.store_chat_as_knowledge.return_value = ""
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_store_key_error(self, patch_bridge):
        patch_bridge.store_chat_as_knowledge.side_effect = KeyError("k")
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_store_runtime_error(self, patch_bridge):
        patch_bridge.store_chat_as_knowledge.side_effect = RuntimeError("boom")
        msgs = [{"content": "a"}, {"content": "b"}]
        result = await handle_store_chat_knowledge(messages=msgs)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_store_defaults(self, patch_bridge):
        msgs = [{"content": "a"}, {"content": "b"}]
        await handle_store_chat_knowledge(messages=msgs)
        call_kwargs = patch_bridge.store_chat_as_knowledge.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"
        assert call_kwargs["channel_id"] == ""
        assert call_kwargs["channel_name"] == ""
        assert call_kwargs["platform"] == "unknown"
        assert call_kwargs["node_type"] == "chat_context"


# ===========================================================================
# Tests: handle_channel_knowledge_summary (standalone function)
# ===========================================================================


class TestHandleChannelKnowledgeSummary:
    """Test the standalone channel summary function."""

    @pytest.mark.asyncio
    async def test_summary_success(self, patch_bridge):
        result = await handle_channel_knowledge_summary(channel_id="C123")
        assert result["success"] is True
        assert result["channel_id"] == "C123"
        assert result["total_items"] == 5

    @pytest.mark.asyncio
    async def test_summary_passes_params(self, patch_bridge):
        await handle_channel_knowledge_summary(
            channel_id="C999",
            workspace_id="ws_55",
            max_items=25,
        )
        call_kwargs = patch_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["channel_id"] == "C999"
        assert call_kwargs["workspace_id"] == "ws_55"
        assert call_kwargs["max_items"] == 25

    @pytest.mark.asyncio
    async def test_summary_defaults(self, patch_bridge):
        await handle_channel_knowledge_summary(channel_id="C1")
        call_kwargs = patch_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["workspace_id"] == "default"
        assert call_kwargs["max_items"] == 10

    @pytest.mark.asyncio
    async def test_summary_key_error(self, patch_bridge):
        patch_bridge.get_channel_knowledge_summary.side_effect = KeyError("bad")
        result = await handle_channel_knowledge_summary(channel_id="C1")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_summary_value_error(self, patch_bridge):
        patch_bridge.get_channel_knowledge_summary.side_effect = ValueError("oops")
        result = await handle_channel_knowledge_summary(channel_id="C1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_summary_runtime_error(self, patch_bridge):
        patch_bridge.get_channel_knowledge_summary.side_effect = RuntimeError("fail")
        result = await handle_channel_knowledge_summary(channel_id="C1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_summary_os_error(self, patch_bridge):
        patch_bridge.get_channel_knowledge_summary.side_effect = OSError("disk")
        result = await handle_channel_knowledge_summary(channel_id="C1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_summary_merges_dict_into_response(self, patch_bridge):
        """The summary dict is merged into the top-level response via **."""
        patch_bridge.get_channel_knowledge_summary.return_value = {
            "channel_id": "C1",
            "topics": ["a", "b"],
            "total_items": 2,
            "extra_field": "custom",
        }
        result = await handle_channel_knowledge_summary(channel_id="C1")
        assert result["success"] is True
        assert result["extra_field"] == "custom"


# ===========================================================================
# Tests: KnowledgeChatHandler.handle (GET)
# ===========================================================================


class TestHandlerHandleGet:
    """Test the handler's handle() method (GET requests)."""

    @pytest.mark.asyncio
    async def test_channel_summary_success(self, handler, patch_bridge):
        path = "/api/v1/chat/knowledge/channel/C123/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {}, http)
        assert result is not None
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_channel_summary_with_workspace(self, handler, patch_bridge):
        path = "/api/v1/chat/knowledge/channel/C456/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {"workspace_id": "ws_42"}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_channel_summary_with_max_items(self, handler, patch_bridge):
        path = "/api/v1/chat/knowledge/channel/C789/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {"max_items": "25"}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_channel_summary_max_items_clamped_high(self, handler, patch_bridge):
        """max_items above MAX_ITEMS_LIMIT should be clamped."""
        path = "/api/v1/chat/knowledge/channel/C789/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {"max_items": "9999"}, http)
        assert _status(result) == 200
        call_kwargs = patch_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["max_items"] <= MAX_ITEMS_LIMIT

    @pytest.mark.asyncio
    async def test_channel_summary_max_items_clamped_low(self, handler, patch_bridge):
        """max_items below 1 should be clamped to 1."""
        path = "/api/v1/chat/knowledge/channel/C789/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {"max_items": "0"}, http)
        assert _status(result) == 200
        call_kwargs = patch_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["max_items"] >= 1

    @pytest.mark.asyncio
    async def test_channel_summary_error_returns_400(self, handler, patch_bridge):
        patch_bridge.get_channel_knowledge_summary.side_effect = RuntimeError("fail")
        path = "/api/v1/chat/knowledge/channel/C123/summary"
        http = MockHTTPHandler()
        result = await handler.handle(path, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_channel_summary_path_too_short(self, handler, patch_bridge):
        """Path with fewer than 8 segments should not match."""
        path = "/api/v1/chat/knowledge/channel/summary"
        http = MockHTTPHandler()
        # This path has only 7 segments when split by "/", so it won't match
        result = await handler.handle(path, {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_unmatched_get_path(self, handler, patch_bridge):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/debates", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_channel_path_not_ending_summary(self, handler, patch_bridge):
        """A channel path not ending in /summary should return None."""
        path = "/api/v1/chat/knowledge/channel/C123/info"
        http = MockHTTPHandler()
        result = await handler.handle(path, {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_channel_summary_extracts_channel_id(self, handler, patch_bridge):
        """Verify the channel_id is correctly extracted from path segment 6."""
        path = "/api/v1/chat/knowledge/channel/MY_CHAN_42/summary"
        http = MockHTTPHandler()
        await handler.handle(path, {}, http)
        call_kwargs = patch_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["channel_id"] == "MY_CHAN_42"


# ===========================================================================
# Tests: KnowledgeChatHandler.handle_post (POST)
# ===========================================================================


class TestHandlerHandlePostSearch:
    """Test POST /search via handler."""

    @pytest.mark.asyncio
    async def test_search_success(self, handler, patch_bridge):
        body = {"query": "What is the policy?"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(SEARCH_PATH, {}, http)
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_search_missing_query(self, handler, patch_bridge):
        body = {"workspace_id": "ws_1"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(SEARCH_PATH, {}, http)
        assert _status(result) == 400
        assert "query" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, patch_bridge):
        body = {"query": ""}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(SEARCH_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_max_results_clamped_high(self, handler, patch_bridge):
        body = {"query": "test", "max_results": 9999}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(SEARCH_PATH, {}, http)
        call_kwargs = patch_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["max_results"] <= MAX_RESULTS_LIMIT

    @pytest.mark.asyncio
    async def test_search_max_results_clamped_low(self, handler, patch_bridge):
        body = {"query": "test", "max_results": -5}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(SEARCH_PATH, {}, http)
        call_kwargs = patch_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["max_results"] >= 1

    @pytest.mark.asyncio
    async def test_search_with_all_optional_params(self, handler, patch_bridge):
        body = {
            "query": "remote work",
            "workspace_id": "ws_99",
            "channel_id": "C555",
            "user_id": "U001",
            "scope": "channel",
            "strategy": "semantic",
            "node_types": ["fact", "decision"],
            "min_confidence": 0.7,
            "max_results": 25,
        }
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(SEARCH_PATH, {}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_bridge_error_returns_400(self, handler, patch_bridge):
        patch_bridge.search_knowledge.side_effect = RuntimeError("fail")
        body = {"query": "test"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(SEARCH_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_default_max_results(self, handler, patch_bridge):
        """When max_results is omitted, defaults to 10."""
        body = {"query": "test"}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(SEARCH_PATH, {}, http)
        call_kwargs = patch_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["max_results"] == 10


class TestHandlerHandlePostInject:
    """Test POST /inject via handler."""

    @pytest.mark.asyncio
    async def test_inject_success(self, handler, patch_bridge):
        body = {"messages": [{"content": "hello"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(INJECT_PATH, {}, http)
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True
        assert data["item_count"] == 2

    @pytest.mark.asyncio
    async def test_inject_missing_messages(self, handler, patch_bridge):
        body = {"workspace_id": "ws_1"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(INJECT_PATH, {}, http)
        assert _status(result) == 400
        assert "messages" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_inject_empty_messages(self, handler, patch_bridge):
        body = {"messages": []}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(INJECT_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_inject_max_context_items_clamped_high(self, handler, patch_bridge):
        body = {"messages": [{"content": "hi"}], "max_context_items": 9999}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(INJECT_PATH, {}, http)
        call_kwargs = patch_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["max_context_items"] <= MAX_CONTEXT_ITEMS_LIMIT

    @pytest.mark.asyncio
    async def test_inject_max_context_items_clamped_low(self, handler, patch_bridge):
        body = {"messages": [{"content": "hi"}], "max_context_items": -3}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(INJECT_PATH, {}, http)
        call_kwargs = patch_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["max_context_items"] >= 1

    @pytest.mark.asyncio
    async def test_inject_with_workspace_and_channel(self, handler, patch_bridge):
        body = {
            "messages": [{"content": "msg"}],
            "workspace_id": "ws_88",
            "channel_id": "C777",
        }
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(INJECT_PATH, {}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_inject_bridge_error_returns_400(self, handler, patch_bridge):
        patch_bridge.inject_knowledge_for_conversation.side_effect = ValueError("bad")
        body = {"messages": [{"content": "hi"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(INJECT_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_inject_default_max_context_items(self, handler, patch_bridge):
        """When max_context_items is omitted, defaults to 5."""
        body = {"messages": [{"content": "hi"}]}
        http = MockHTTPHandler(body=body)
        await handler.handle_post(INJECT_PATH, {}, http)
        call_kwargs = patch_bridge.inject_knowledge_for_conversation.call_args.kwargs
        assert call_kwargs["max_context_items"] == 5


class TestHandlerHandlePostStore:
    """Test POST /store via handler."""

    @pytest.mark.asyncio
    async def test_store_success(self, handler, patch_bridge):
        body = {"messages": [{"content": "a"}, {"content": "b"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 200
        data = _data(result)
        assert data["success"] is True
        assert data["node_id"] == "node_abc123"

    @pytest.mark.asyncio
    async def test_store_missing_messages(self, handler, patch_bridge):
        body = {"workspace_id": "ws_1"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 400
        assert "2 messages" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_store_one_message(self, handler, patch_bridge):
        body = {"messages": [{"content": "only one"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_store_empty_messages(self, handler, patch_bridge):
        body = {"messages": []}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_store_with_all_params(self, handler, patch_bridge):
        body = {
            "messages": [{"content": "a"}, {"content": "b"}],
            "workspace_id": "ws_11",
            "channel_id": "C222",
            "channel_name": "#eng",
            "platform": "slack",
            "node_type": "decision",
        }
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_store_bridge_failure_returns_400(self, handler, patch_bridge):
        """When bridge returns None node_id, handler returns 400."""
        patch_bridge.store_chat_as_knowledge.return_value = None
        body = {"messages": [{"content": "a"}, {"content": "b"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_store_bridge_exception_returns_400(self, handler, patch_bridge):
        patch_bridge.store_chat_as_knowledge.side_effect = OSError("disk")
        body = {"messages": [{"content": "a"}, {"content": "b"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_store_message_count_in_response(self, handler, patch_bridge):
        """Verify message_count reflects input length."""
        body = {"messages": [{"content": "a"}, {"content": "b"}, {"content": "c"}]}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post(STORE_PATH, {}, http)
        data = _data(result)
        assert data["message_count"] == 3


class TestHandlerHandlePostUnknown:
    """Test POST to unknown paths."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler, patch_bridge):
        body = {"query": "test"}
        http = MockHTTPHandler(body=body)
        result = await handler.handle_post("/api/v1/unknown", {}, http)
        assert result is None


# ===========================================================================
# Tests: Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_default_ctx(self):
        h = KnowledgeChatHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = KnowledgeChatHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_none_ctx_defaults_to_empty(self):
        h = KnowledgeChatHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Tests: ROUTES and ROUTE_PREFIXES constants
# ===========================================================================


class TestRouteConstants:
    """Test that route constants are correctly defined."""

    def test_routes_contains_search(self):
        assert SEARCH_PATH in KnowledgeChatHandler.ROUTES

    def test_routes_contains_inject(self):
        assert INJECT_PATH in KnowledgeChatHandler.ROUTES

    def test_routes_contains_store(self):
        assert STORE_PATH in KnowledgeChatHandler.ROUTES

    def test_route_prefixes_contains_channel(self):
        assert "/api/v1/chat/knowledge/channel/" in KnowledgeChatHandler.ROUTE_PREFIXES

    def test_permission_constants(self):
        assert KnowledgeChatHandler.KNOWLEDGE_READ_PERMISSION == "knowledge.read"
        assert KnowledgeChatHandler.KNOWLEDGE_WRITE_PERMISSION == "knowledge.write"


# ===========================================================================
# Tests: _get_bridge lazy loading
# ===========================================================================


class TestGetBridge:
    """Test lazy bridge initialization."""

    def test_bridge_is_cached(self):
        """After first call, bridge should be cached."""
        mock_b = MagicMock()
        with patch(
            "aragora.services.knowledge_chat_bridge.get_knowledge_chat_bridge",
            return_value=mock_b,
        ):
            from aragora.server.handlers.knowledge_chat import _get_bridge

            b1 = _get_bridge()
            b2 = _get_bridge()
            assert b1 is b2

    def test_bridge_calls_factory(self):
        """First call should invoke get_knowledge_chat_bridge."""
        mock_b = MagicMock()
        with patch(
            "aragora.services.knowledge_chat_bridge.get_knowledge_chat_bridge",
            return_value=mock_b,
        ) as factory:
            from aragora.server.handlers.knowledge_chat import _get_bridge

            _get_bridge()
            factory.assert_called_once()


# ===========================================================================
# Tests: input bounds constants
# ===========================================================================


class TestInputBounds:
    """Test the module-level input bound constants."""

    def test_max_results_limit(self):
        assert MAX_RESULTS_LIMIT == 100

    def test_max_context_items_limit(self):
        assert MAX_CONTEXT_ITEMS_LIMIT == 50

    def test_max_items_limit(self):
        assert MAX_ITEMS_LIMIT == 100


# ===========================================================================
# Tests: module __all__ exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports the expected symbols."""

    def test_all_exports(self):
        from aragora.server.handlers.knowledge_chat import __all__

        expected = {
            "KnowledgeChatHandler",
            "handle_knowledge_search",
            "handle_knowledge_inject",
            "handle_store_chat_knowledge",
            "handle_channel_knowledge_summary",
        }
        assert set(__all__) == expected

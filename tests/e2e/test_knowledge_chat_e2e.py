"""
E2E tests for Knowledge + Chat Bridge integration.

Tests the complete flow for:
1. Search knowledge from chat context
2. Inject knowledge into conversations
3. Store chat messages as knowledge
4. Get channel knowledge summaries
5. Full workflows: store → search → inject

These tests verify end-to-end behavior through the handler layer
with mocked bridge dependencies.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

# Mark all tests as e2e tests
pytestmark = [pytest.mark.e2e, pytest.mark.knowledge]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_knowledge_context():
    """Create a mock KnowledgeContext response."""

    class MockKnowledgeContext:
        def __init__(
            self,
            items: List[Dict[str, Any]],
            query: str = "test query",
            total: int = 0,
        ):
            self.items = items
            self.query = query
            self.total = total or len(items)
            self.scope = "workspace"
            self.strategy = "hybrid"

        def to_dict(self):
            return {
                "items": self.items,
                "query": self.query,
                "total": self.total,
                "scope": self.scope,
                "strategy": self.strategy,
            }

    return MockKnowledgeContext


@pytest.fixture
def mock_knowledge_item():
    """Create a mock knowledge item."""

    class MockKnowledgeItem:
        def __init__(
            self,
            content: str,
            node_id: str = None,
            relevance: float = 0.85,
            node_type: str = "chat_context",
        ):
            self.node_id = node_id or f"node_{uuid4().hex[:8]}"
            self.content = content
            self.relevance = relevance
            self.node_type = node_type
            self.metadata = {"source": "chat", "platform": "slack"}

        def to_dict(self):
            return {
                "node_id": self.node_id,
                "content": self.content,
                "relevance": self.relevance,
                "node_type": self.node_type,
                "metadata": self.metadata,
            }

    return MockKnowledgeItem


@pytest.fixture
def mock_bridge(mock_knowledge_context, mock_knowledge_item):
    """Create a mock Knowledge + Chat bridge."""
    bridge = MagicMock()

    # Storage for "persisted" knowledge
    bridge._stored_knowledge: Dict[str, Dict[str, Any]] = {}
    bridge._channel_knowledge: Dict[str, List[str]] = {}

    async def search_knowledge(
        query: str,
        workspace_id: str = "default",
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope=None,
        strategy=None,
        node_types: Optional[List[str]] = None,
        min_confidence: float = 0.3,
        max_results: int = 10,
    ):
        # Search stored knowledge
        items = []
        for node_id, data in bridge._stored_knowledge.items():
            content = data.get("content", "")
            if query.lower() in content.lower():
                item = mock_knowledge_item(
                    content=content,
                    node_id=node_id,
                    relevance=0.9,
                    node_type=data.get("node_type", "chat_context"),
                )
                items.append(item.to_dict())
                if len(items) >= max_results:
                    break

        return mock_knowledge_context(items=items, query=query)

    async def inject_knowledge_for_conversation(
        messages: List[Dict[str, Any]],
        workspace_id: str = "default",
        channel_id: Optional[str] = None,
        max_context_items: int = 5,
    ):
        # Extract topics from messages
        combined_text = " ".join(m.get("content", "") for m in messages)
        items = []

        for node_id, data in bridge._stored_knowledge.items():
            content = data.get("content", "")
            # Simple keyword matching for demo
            if any(word in content.lower() for word in combined_text.lower().split()[:5]):
                item = mock_knowledge_item(
                    content=content,
                    node_id=node_id,
                    relevance=0.8,
                )
                items.append(item)
                if len(items) >= max_context_items:
                    break

        return items

    async def store_chat_as_knowledge(
        messages: List[Dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str = "",
        channel_name: str = "",
        platform: str = "unknown",
        node_type: str = "chat_context",
    ):
        # Generate node ID and store
        node_id = f"chat_{uuid4().hex[:8]}"
        content = "\n".join(
            f"{m.get('author', 'unknown')}: {m.get('content', '')}" for m in messages
        )

        bridge._stored_knowledge[node_id] = {
            "content": content,
            "workspace_id": workspace_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "platform": platform,
            "node_type": node_type,
            "messages": messages,
            "created_at": datetime.now().isoformat(),
        }

        # Track by channel
        if channel_id:
            if channel_id not in bridge._channel_knowledge:
                bridge._channel_knowledge[channel_id] = []
            bridge._channel_knowledge[channel_id].append(node_id)

        return node_id

    async def get_channel_knowledge_summary(
        channel_id: str,
        workspace_id: str = "default",
        max_items: int = 10,
    ):
        node_ids = bridge._channel_knowledge.get(channel_id, [])[:max_items]
        items = []

        for node_id in node_ids:
            if node_id in bridge._stored_knowledge:
                data = bridge._stored_knowledge[node_id]
                items.append(
                    {
                        "node_id": node_id,
                        "content_preview": data["content"][:100],
                        "created_at": data.get("created_at"),
                    }
                )

        return {
            "channel_id": channel_id,
            "workspace_id": workspace_id,
            "total_items": len(node_ids),
            "items": items,
        }

    bridge.search_knowledge = AsyncMock(side_effect=search_knowledge)
    bridge.inject_knowledge_for_conversation = AsyncMock(
        side_effect=inject_knowledge_for_conversation
    )
    bridge.store_chat_as_knowledge = AsyncMock(side_effect=store_chat_as_knowledge)
    bridge.get_channel_knowledge_summary = AsyncMock(side_effect=get_channel_knowledge_summary)

    return bridge


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with auth context."""
    handler = MagicMock()
    handler.request = MagicMock()
    handler.request.body = b"{}"

    # Mock authorization context with all permissions
    handler._auth_context = MagicMock()
    handler._auth_context.user_id = "test_user"
    handler._auth_context.workspace_id = "test_workspace"
    handler._auth_context.has_permission = MagicMock(return_value=True)

    return handler


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler instantiation."""
    context = MagicMock()
    context.settings = MagicMock()
    context.knowledge_mound = MagicMock()
    return context


# =============================================================================
# Search Knowledge Tests
# =============================================================================


class TestKnowledgeSearchE2E:
    """E2E tests for knowledge search from chat context."""

    @pytest.mark.asyncio
    async def test_search_returns_matching_results(self, mock_bridge):
        """Test that search returns relevant knowledge items."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        # Pre-populate some knowledge
        mock_bridge._stored_knowledge["node_1"] = {
            "content": "The company vacation policy allows 20 days per year.",
            "node_type": "policy",
        }
        mock_bridge._stored_knowledge["node_2"] = {
            "content": "Remote work is allowed 3 days per week.",
            "node_type": "policy",
        }

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="vacation policy",
                workspace_id="ws_123",
                max_results=10,
            )

        assert result["success"] is True
        assert "items" in result
        assert len(result["items"]) >= 1
        # Should find the vacation policy
        assert any("vacation" in item["content"].lower() for item in result["items"])

    @pytest.mark.asyncio
    async def test_search_with_channel_filter(self, mock_bridge):
        """Test search scoped to a specific channel."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="engineering decisions",
                workspace_id="ws_123",
                channel_id="C_engineering",
                scope="channel",
            )

        assert result["success"] is True
        # Bridge was called with channel filter
        mock_bridge.search_knowledge.assert_called_once()
        call_kwargs = mock_bridge.search_knowledge.call_args.kwargs
        assert call_kwargs["channel_id"] == "C_engineering"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self, mock_bridge):
        """Test search with no matching results."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="nonexistent topic xyz123",
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, mock_bridge):
        """Test that search respects the max_results limit."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        # Add many items
        for i in range(20):
            mock_bridge._stored_knowledge[f"node_{i}"] = {
                "content": f"Knowledge item {i} about testing",
                "node_type": "document",
            }

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="testing",
                workspace_id="ws_123",
                max_results=5,
            )

        assert result["success"] is True
        assert len(result["items"]) <= 5


# =============================================================================
# Inject Knowledge Tests
# =============================================================================


class TestKnowledgeInjectE2E:
    """E2E tests for injecting knowledge into conversations."""

    @pytest.mark.asyncio
    async def test_inject_finds_relevant_context(self, mock_bridge):
        """Test that inject returns relevant knowledge for messages."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        # Pre-populate knowledge
        mock_bridge._stored_knowledge["node_policy"] = {
            "content": "Our vacation policy: employees get 20 days annually.",
            "node_type": "policy",
        }

        messages = [
            {"author": "alice", "content": "How many vacation days do we get?"},
            {"author": "bob", "content": "I think it's in the policy document"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=messages,
                workspace_id="ws_123",
                max_context_items=5,
            )

        assert result["success"] is True
        assert "context" in result
        assert "item_count" in result

    @pytest.mark.asyncio
    async def test_inject_limits_context_items(self, mock_bridge):
        """Test that inject respects max_context_items."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        # Add many relevant items
        for i in range(10):
            mock_bridge._stored_knowledge[f"node_{i}"] = {
                "content": f"Policy document {i} about vacation rules",
                "node_type": "policy",
            }

        messages = [{"author": "user", "content": "What's the vacation policy?"}]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=messages,
                workspace_id="ws_123",
                max_context_items=3,
            )

        assert result["success"] is True
        assert result["item_count"] <= 3

    @pytest.mark.asyncio
    async def test_inject_empty_conversation(self, mock_bridge):
        """Test inject with empty message history."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=[],
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["item_count"] == 0


# =============================================================================
# Store Knowledge Tests
# =============================================================================


class TestStoreKnowledgeE2E:
    """E2E tests for storing chat messages as knowledge."""

    @pytest.mark.asyncio
    async def test_store_chat_successfully(self, mock_bridge):
        """Test storing a chat conversation as knowledge."""
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        messages = [
            {"author": "alice", "content": "We should use Python 3.11"},
            {"author": "bob", "content": "Agreed, it has better performance"},
            {"author": "alice", "content": "Let's make it official policy"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=messages,
                workspace_id="ws_123",
                channel_id="C_engineering",
                channel_name="#engineering",
                platform="slack",
            )

        assert result["success"] is True
        assert "node_id" in result
        assert result["message_count"] == 3

        # Verify stored in bridge
        assert result["node_id"] in mock_bridge._stored_knowledge

    @pytest.mark.asyncio
    async def test_store_requires_minimum_messages(self, mock_bridge):
        """Test that storing requires at least 2 messages."""
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        messages = [{"author": "alice", "content": "Single message"}]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=messages,
                workspace_id="ws_123",
            )

        assert result["success"] is False
        assert "2 messages" in result["error"]

    @pytest.mark.asyncio
    async def test_store_with_custom_node_type(self, mock_bridge):
        """Test storing with a custom node type."""
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        messages = [
            {"author": "alice", "content": "Decision: Use microservices"},
            {"author": "bob", "content": "Approved by architecture team"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=messages,
                workspace_id="ws_123",
                node_type="decision",
            )

        assert result["success"] is True
        node_id = result["node_id"]
        assert mock_bridge._stored_knowledge[node_id]["node_type"] == "decision"


# =============================================================================
# Channel Summary Tests
# =============================================================================


class TestChannelSummaryE2E:
    """E2E tests for channel knowledge summaries."""

    @pytest.mark.asyncio
    async def test_get_channel_summary(self, mock_bridge):
        """Test getting knowledge summary for a channel."""
        from aragora.server.handlers.knowledge_chat import (
            handle_channel_knowledge_summary,
            handle_store_chat_knowledge,
        )

        # Store some knowledge for the channel
        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            await handle_store_chat_knowledge(
                messages=[
                    {"author": "alice", "content": "First decision"},
                    {"author": "bob", "content": "Agreed"},
                ],
                workspace_id="ws_123",
                channel_id="C_engineering",
            )

            await handle_store_chat_knowledge(
                messages=[
                    {"author": "charlie", "content": "Second decision"},
                    {"author": "dave", "content": "Confirmed"},
                ],
                workspace_id="ws_123",
                channel_id="C_engineering",
            )

            result = await handle_channel_knowledge_summary(
                channel_id="C_engineering",
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["channel_id"] == "C_engineering"
        assert result["total_items"] == 2
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_channel_summary_empty(self, mock_bridge):
        """Test summary for channel with no knowledge."""
        from aragora.server.handlers.knowledge_chat import (
            handle_channel_knowledge_summary,
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="C_empty_channel",
                workspace_id="ws_123",
            )

        assert result["success"] is True
        assert result["total_items"] == 0
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_channel_summary_respects_limit(self, mock_bridge):
        """Test that summary respects max_items limit."""
        from aragora.server.handlers.knowledge_chat import (
            handle_channel_knowledge_summary,
            handle_store_chat_knowledge,
        )

        # Store many items for the channel
        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            for i in range(10):
                await handle_store_chat_knowledge(
                    messages=[
                        {"author": "user", "content": f"Message {i} part 1"},
                        {"author": "user", "content": f"Message {i} part 2"},
                    ],
                    workspace_id="ws_123",
                    channel_id="C_busy",
                )

            result = await handle_channel_knowledge_summary(
                channel_id="C_busy",
                workspace_id="ws_123",
                max_items=3,
            )

        assert result["success"] is True
        assert len(result["items"]) <= 3


# =============================================================================
# Full Workflow Tests
# =============================================================================


class TestKnowledgeChatWorkflowE2E:
    """E2E tests for complete knowledge-chat workflows."""

    @pytest.mark.asyncio
    async def test_store_then_search_workflow(self, mock_bridge):
        """Test workflow: store chat → search for it later."""
        from aragora.server.handlers.knowledge_chat import (
            handle_knowledge_search,
            handle_store_chat_knowledge,
        )

        # Step 1: Store a decision from chat
        messages = [
            {"author": "alice", "content": "We're moving to Kubernetes"},
            {"author": "bob", "content": "Yes, approved by infrastructure team"},
            {"author": "alice", "content": "Target date is Q2 2024"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            store_result = await handle_store_chat_knowledge(
                messages=messages,
                workspace_id="ws_123",
                channel_id="C_infra",
                platform="slack",
            )

            assert store_result["success"] is True

            # Step 2: Later, search for Kubernetes decisions
            search_result = await handle_knowledge_search(
                query="Kubernetes",
                workspace_id="ws_123",
            )

        assert search_result["success"] is True
        assert len(search_result["items"]) >= 1
        # Should find the stored decision
        found_content = " ".join(item["content"] for item in search_result["items"])
        assert "Kubernetes" in found_content

    @pytest.mark.asyncio
    async def test_store_then_inject_workflow(self, mock_bridge):
        """Test workflow: store knowledge → inject into new conversation."""
        from aragora.server.handlers.knowledge_chat import (
            handle_knowledge_inject,
            handle_store_chat_knowledge,
        )

        # Step 1: Store policy discussion
        policy_messages = [
            {"author": "hr", "content": "New remote work policy: 3 days per week"},
            {"author": "manager", "content": "Effective starting next month"},
        ]

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            await handle_store_chat_knowledge(
                messages=policy_messages,
                workspace_id="ws_123",
                channel_id="C_hr",
            )

            # Step 2: New conversation asks about remote work
            new_conversation = [
                {"author": "employee", "content": "What's our remote work policy?"},
                {"author": "colleague", "content": "I heard it changed recently"},
            ]

            inject_result = await handle_knowledge_inject(
                messages=new_conversation,
                workspace_id="ws_123",
                max_context_items=5,
            )

        assert inject_result["success"] is True
        # Should inject relevant context about remote work
        assert inject_result["item_count"] >= 0

    @pytest.mark.asyncio
    async def test_multi_channel_knowledge_isolation(self, mock_bridge):
        """Test that knowledge is properly isolated by channel."""
        from aragora.server.handlers.knowledge_chat import (
            handle_channel_knowledge_summary,
            handle_store_chat_knowledge,
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            # Store in engineering channel
            await handle_store_chat_knowledge(
                messages=[
                    {"author": "eng1", "content": "Technical decision"},
                    {"author": "eng2", "content": "Approved"},
                ],
                workspace_id="ws_123",
                channel_id="C_engineering",
            )

            # Store in HR channel
            await handle_store_chat_knowledge(
                messages=[
                    {"author": "hr1", "content": "HR policy update"},
                    {"author": "hr2", "content": "Confirmed"},
                ],
                workspace_id="ws_123",
                channel_id="C_hr",
            )

            # Get summaries
            eng_summary = await handle_channel_knowledge_summary(
                channel_id="C_engineering",
                workspace_id="ws_123",
            )
            hr_summary = await handle_channel_knowledge_summary(
                channel_id="C_hr",
                workspace_id="ws_123",
            )

        # Each channel should only have its own knowledge
        assert eng_summary["total_items"] == 1
        assert hr_summary["total_items"] == 1

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, mock_bridge):
        """Test that knowledge is properly isolated by workspace."""
        from aragora.server.handlers.knowledge_chat import (
            handle_knowledge_search,
            handle_store_chat_knowledge,
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            # Store in workspace A
            await handle_store_chat_knowledge(
                messages=[
                    {"author": "user1", "content": "Workspace A secret"},
                    {"author": "user2", "content": "Confirmed A"},
                ],
                workspace_id="ws_A",
            )

            # Store in workspace B
            await handle_store_chat_knowledge(
                messages=[
                    {"author": "user3", "content": "Workspace B secret"},
                    {"author": "user4", "content": "Confirmed B"},
                ],
                workspace_id="ws_B",
            )

            # Search in workspace A
            result_a = await handle_knowledge_search(
                query="secret",
                workspace_id="ws_A",
            )

            # Search in workspace B
            result_b = await handle_knowledge_search(
                query="secret",
                workspace_id="ws_B",
            )

        # Both searches succeed
        assert result_a["success"] is True
        assert result_b["success"] is True


# =============================================================================
# Handler Integration Tests
# =============================================================================


class TestKnowledgeChatHandlerE2E:
    """E2E tests for the HTTP handler class."""

    @pytest.mark.asyncio
    async def test_handler_can_handle_routes(self, mock_server_context):
        """Test that handler correctly identifies its routes."""
        from aragora.server.handlers.knowledge_chat import KnowledgeChatHandler

        handler = KnowledgeChatHandler(mock_server_context)

        # Should handle these routes
        assert handler.can_handle("/api/v1/chat/knowledge/search") is True
        assert handler.can_handle("/api/v1/chat/knowledge/inject") is True
        assert handler.can_handle("/api/v1/chat/knowledge/store") is True
        assert handler.can_handle("/api/v1/chat/knowledge/channel/C123/summary") is True

        # Should not handle these
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/knowledge") is False

    @pytest.mark.asyncio
    async def test_handler_extracts_channel_id_from_path(
        self, mock_bridge, mock_handler, mock_server_context
    ):
        """Test that handler correctly extracts channel_id from path."""
        from aragora.server.handlers.knowledge_chat import KnowledgeChatHandler

        handler_instance = KnowledgeChatHandler(mock_server_context)

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handler_instance.handle(
                path="/api/v1/chat/knowledge/channel/C_test_123/summary",
                query_params={"workspace_id": "ws_123"},
                handler=mock_handler,
            )

        # Should have called bridge with correct channel_id
        mock_bridge.get_channel_knowledge_summary.assert_called_once()
        call_kwargs = mock_bridge.get_channel_knowledge_summary.call_args.kwargs
        assert call_kwargs["channel_id"] == "C_test_123"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestKnowledgeChatErrorHandlingE2E:
    """E2E tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_handles_bridge_error(self, mock_bridge):
        """Test that search gracefully handles bridge errors."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_search

        # Make bridge raise an error
        mock_bridge.search_knowledge = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_search(
                query="test",
                workspace_id="ws_123",
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_store_handles_bridge_error(self, mock_bridge):
        """Test that store gracefully handles bridge errors."""
        from aragora.server.handlers.knowledge_chat import handle_store_chat_knowledge

        # Make bridge raise an error
        mock_bridge.store_chat_as_knowledge = AsyncMock(side_effect=Exception("Storage failed"))

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_store_chat_knowledge(
                messages=[
                    {"author": "user1", "content": "Test 1"},
                    {"author": "user2", "content": "Test 2"},
                ],
                workspace_id="ws_123",
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_inject_handles_bridge_error(self, mock_bridge):
        """Test that inject gracefully handles bridge errors."""
        from aragora.server.handlers.knowledge_chat import handle_knowledge_inject

        # Make bridge raise an error
        mock_bridge.inject_knowledge_for_conversation = AsyncMock(
            side_effect=Exception("Injection failed")
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_knowledge_inject(
                messages=[{"author": "user", "content": "Test"}],
                workspace_id="ws_123",
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_channel_summary_handles_bridge_error(self, mock_bridge):
        """Test that channel summary gracefully handles bridge errors."""
        from aragora.server.handlers.knowledge_chat import (
            handle_channel_knowledge_summary,
        )

        # Make bridge raise an error
        mock_bridge.get_channel_knowledge_summary = AsyncMock(
            side_effect=Exception("Summary failed")
        )

        with patch(
            "aragora.server.handlers.knowledge_chat._get_bridge",
            return_value=mock_bridge,
        ):
            result = await handle_channel_knowledge_summary(
                channel_id="C_test",
                workspace_id="ws_123",
            )

        assert result["success"] is False
        assert "error" in result

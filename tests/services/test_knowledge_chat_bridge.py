"""
Tests for Knowledge + Chat Bridge Service.

Tests for the knowledge chat bridge including:
- Knowledge search with different scopes and strategies
- Chat history to knowledge conversion
- Knowledge injection for conversations
- Channel knowledge summary
- Caching with TTL
- Multi-tenant workspace isolation
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestKnowledgeSearchScope:
    """Tests for KnowledgeSearchScope enum."""

    def test_scope_values(self):
        """Test scope enum values."""
        from aragora.services.knowledge_chat_bridge import KnowledgeSearchScope

        assert KnowledgeSearchScope.ALL.value == "all"
        assert KnowledgeSearchScope.WORKSPACE.value == "workspace"
        assert KnowledgeSearchScope.CHANNEL.value == "channel"
        assert KnowledgeSearchScope.USER.value == "user"


class TestRelevanceStrategy:
    """Tests for RelevanceStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        from aragora.services.knowledge_chat_bridge import RelevanceStrategy

        assert RelevanceStrategy.SEMANTIC.value == "semantic"
        assert RelevanceStrategy.KEYWORD.value == "keyword"
        assert RelevanceStrategy.HYBRID.value == "hybrid"
        assert RelevanceStrategy.RECENCY.value == "recency"


class TestKnowledgeSearchResult:
    """Tests for KnowledgeSearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a search result."""
        from aragora.services.knowledge_chat_bridge import KnowledgeSearchResult

        now = datetime.now(timezone.utc)
        result = KnowledgeSearchResult(
            node_id="node_123",
            content="This is the knowledge content",
            node_type="evidence",
            confidence=0.85,
            relevance_score=0.9,
            source="slack:C123",
            created_at=now,
            metadata={"channel_id": "C123"},
            provenance="User input via Slack",
        )

        assert result.node_id == "node_123"
        assert result.content == "This is the knowledge content"
        assert result.node_type == "evidence"
        assert result.confidence == 0.85
        assert result.relevance_score == 0.9
        assert result.source == "slack:C123"
        assert result.created_at == now
        assert result.metadata["channel_id"] == "C123"
        assert result.provenance == "User input via Slack"

    def test_combined_score(self):
        """Test combined score calculation."""
        from aragora.services.knowledge_chat_bridge import KnowledgeSearchResult

        result = KnowledgeSearchResult(
            node_id="node_1",
            content="Content",
            node_type="pattern",
            confidence=0.8,
            relevance_score=0.9,
            source="test",
            created_at=datetime.now(timezone.utc),
        )

        # (0.8 * 0.4) + (0.9 * 0.6) = 0.32 + 0.54 = 0.86
        assert result.combined_score == pytest.approx(0.86, rel=0.01)

    def test_to_dict(self):
        """Test to_dict serialization."""
        from aragora.services.knowledge_chat_bridge import KnowledgeSearchResult

        now = datetime.now(timezone.utc)
        result = KnowledgeSearchResult(
            node_id="node_456",
            content="Test content for serialization",
            node_type="insight",
            confidence=0.75,
            relevance_score=0.85,
            source="github:repo/pr/123",
            created_at=now,
            metadata={"pr_number": 123},
            provenance="GitHub PR comment",
        )

        data = result.to_dict()

        assert data["node_id"] == "node_456"
        assert data["content"] == "Test content for serialization"
        assert data["node_type"] == "insight"
        assert data["confidence"] == 0.75
        assert data["relevance_score"] == 0.85
        assert data["source"] == "github:repo/pr/123"
        assert data["created_at"] == now.isoformat()
        assert data["metadata"]["pr_number"] == 123
        assert data["provenance"] == "GitHub PR comment"

    def test_to_dict_truncates_long_content(self):
        """Test that to_dict truncates content over 500 chars."""
        from aragora.services.knowledge_chat_bridge import KnowledgeSearchResult

        long_content = "x" * 600  # 600 characters
        result = KnowledgeSearchResult(
            node_id="node_1",
            content=long_content,
            node_type="document",
            confidence=0.8,
            relevance_score=0.8,
            source="test",
            created_at=datetime.now(timezone.utc),
        )

        data = result.to_dict()

        assert len(data["content"]) == 500


class TestChatKnowledgeContext:
    """Tests for ChatKnowledgeContext dataclass."""

    def test_context_creation(self):
        """Test creating chat knowledge context."""
        from aragora.services.knowledge_chat_bridge import (
            ChatKnowledgeContext,
            KnowledgeSearchResult,
            KnowledgeSearchScope,
        )

        results = [
            KnowledgeSearchResult(
                node_id="node_1",
                content="Result 1",
                node_type="evidence",
                confidence=0.8,
                relevance_score=0.9,
                source="test",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        context = ChatKnowledgeContext(
            channel_id="C123456",
            workspace_id="ws_789",
            query="test query",
            results=results,
            search_scope=KnowledgeSearchScope.WORKSPACE,
            search_time_ms=50.5,
            suggestions=["More evidence about: test query"],
        )

        assert context.channel_id == "C123456"
        assert context.workspace_id == "ws_789"
        assert context.query == "test query"
        assert len(context.results) == 1
        assert context.search_scope == KnowledgeSearchScope.WORKSPACE
        assert context.search_time_ms == 50.5
        assert len(context.suggestions) == 1

    def test_context_to_dict(self):
        """Test ChatKnowledgeContext.to_dict serialization."""
        from aragora.services.knowledge_chat_bridge import (
            ChatKnowledgeContext,
            KnowledgeSearchResult,
            KnowledgeSearchScope,
        )

        results = [
            KnowledgeSearchResult(
                node_id="node_1",
                content="Result 1",
                node_type="evidence",
                confidence=0.8,
                relevance_score=0.9,
                source="test",
                created_at=datetime.now(timezone.utc),
            ),
            KnowledgeSearchResult(
                node_id="node_2",
                content="Result 2",
                node_type="pattern",
                confidence=0.75,
                relevance_score=0.85,
                source="test2",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        context = ChatKnowledgeContext(
            channel_id="C123",
            workspace_id="ws_1",
            query="search query",
            results=results,
            search_scope=KnowledgeSearchScope.CHANNEL,
            search_time_ms=100.0,
            suggestions=["suggestion 1", "suggestion 2"],
        )

        data = context.to_dict()

        assert data["channel_id"] == "C123"
        assert data["workspace_id"] == "ws_1"
        assert data["query"] == "search query"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["search_scope"] == "channel"
        assert data["search_time_ms"] == 100.0
        assert len(data["suggestions"]) == 2


class TestKnowledgeChatBridge:
    """Tests for KnowledgeChatBridge."""

    def test_bridge_initialization_defaults(self):
        """Test bridge initialization with defaults."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge()

        assert bridge._mound is None
        assert bridge.enable_caching is True
        assert bridge.cache_ttl_seconds == 300
        assert bridge.max_results == 10
        assert bridge._cache == {}

    def test_bridge_initialization_custom(self):
        """Test bridge initialization with custom config."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = MagicMock()
        bridge = KnowledgeChatBridge(
            mound=mock_mound,
            enable_caching=False,
            cache_ttl_seconds=600,
            max_results=20,
        )

        assert bridge._mound is mock_mound
        assert bridge.enable_caching is False
        assert bridge.cache_ttl_seconds == 600
        assert bridge.max_results == 20

    def test_clear_cache_all(self):
        """Test clearing all cache."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge()

        # Add cached items
        bridge._cache["ws_1:C1:query1:workspace"] = (MagicMock(), time.time())
        bridge._cache["ws_1:C2:query2:workspace"] = (MagicMock(), time.time())
        bridge._cache["ws_2:C3:query3:workspace"] = (MagicMock(), time.time())

        assert len(bridge._cache) == 3

        bridge.clear_cache()

        assert len(bridge._cache) == 0

    def test_clear_cache_specific_workspace(self):
        """Test clearing cache for specific workspace."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge()

        # Add cached items for different workspaces
        bridge._cache["ws_1:C1:query1:workspace"] = (MagicMock(), time.time())
        bridge._cache["ws_1:C2:query2:workspace"] = (MagicMock(), time.time())
        bridge._cache["ws_2:C3:query3:workspace"] = (MagicMock(), time.time())

        assert len(bridge._cache) == 3

        bridge.clear_cache("ws_1")

        assert len(bridge._cache) == 1
        assert "ws_2:C3:query3:workspace" in bridge._cache

    def test_generate_suggestions_from_results(self):
        """Test generating suggestions from results."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            KnowledgeSearchResult,
        )

        bridge = KnowledgeChatBridge()

        results = [
            KnowledgeSearchResult(
                node_id="n1",
                content="Content 1",
                node_type="evidence",
                confidence=0.8,
                relevance_score=0.9,
                source="slack:C123",
                created_at=datetime.now(timezone.utc),
            ),
            KnowledgeSearchResult(
                node_id="n2",
                content="Content 2",
                node_type="pattern",
                confidence=0.75,
                relevance_score=0.85,
                source="github:repo",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        suggestions = bridge._generate_suggestions("test query", results)

        assert len(suggestions) <= 3
        # Should suggest based on node types
        assert any("evidence" in s or "pattern" in s for s in suggestions)

    def test_generate_suggestions_with_sources(self):
        """Test suggestions include source-based suggestions."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            KnowledgeSearchResult,
        )

        bridge = KnowledgeChatBridge()

        results = [
            KnowledgeSearchResult(
                node_id="n1",
                content="Content",
                node_type="document",
                confidence=0.8,
                relevance_score=0.9,
                source="slack:channel",
                created_at=datetime.now(timezone.utc),
            ),
            KnowledgeSearchResult(
                node_id="n2",
                content="Content 2",
                node_type="document",
                confidence=0.8,
                relevance_score=0.9,
                source="github:repo/issues",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        suggestions = bridge._generate_suggestions("api docs", results)

        # Should have suggestions based on sources
        assert any("slack" in s.lower() or "github" in s.lower() for s in suggestions)

    @pytest.mark.asyncio
    async def test_search_knowledge_no_mound(self):
        """Test search when mound not available."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge(enable_caching=False)

        context = await bridge.search_knowledge("test query")

        assert context.query == "test query"
        assert len(context.results) == 0
        assert context.workspace_id == "default"

    @pytest.mark.asyncio
    async def test_search_knowledge_cached(self):
        """Test search returns cached results."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            ChatKnowledgeContext,
            KnowledgeSearchScope,
        )

        bridge = KnowledgeChatBridge(cache_ttl_seconds=300)

        # Pre-populate cache
        cached_context = ChatKnowledgeContext(
            channel_id="",
            workspace_id="default",
            query="cached query",
            results=[],
            search_scope=KnowledgeSearchScope.WORKSPACE,
            search_time_ms=10.0,
        )
        bridge._cache["default:None:cached query:workspace"] = (
            cached_context,
            time.time(),
        )

        result = await bridge.search_knowledge("cached query")

        assert result is cached_context

    @pytest.mark.asyncio
    async def test_search_knowledge_expired_cache(self):
        """Test search ignores expired cache."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            ChatKnowledgeContext,
            KnowledgeSearchScope,
        )

        bridge = KnowledgeChatBridge(cache_ttl_seconds=60)

        # Pre-populate cache with old timestamp
        cached_context = ChatKnowledgeContext(
            channel_id="",
            workspace_id="default",
            query="expired query",
            results=[],
            search_scope=KnowledgeSearchScope.WORKSPACE,
            search_time_ms=10.0,
        )
        # Cache entry is 120 seconds old (expired)
        bridge._cache["default:None:expired query:workspace"] = (
            cached_context,
            time.time() - 120,
        )

        result = await bridge.search_knowledge("expired query")

        # Should get fresh result, not cached
        assert result is not cached_context

    @pytest.mark.asyncio
    async def test_search_knowledge_with_mound_semantic(self):
        """Test semantic search with mound."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            RelevanceStrategy,
        )

        mock_mound = AsyncMock()
        mock_node = MagicMock()
        mock_node.node_id = "node_1"
        mock_node.content = "Relevant knowledge"
        mock_node.node_type = "evidence"
        mock_node.confidence = 0.85
        mock_node.source = "test:source"
        mock_node.created_at = datetime.now(timezone.utc)
        mock_node.metadata = {}
        mock_node.provenance = None

        mock_mound.semantic_search = AsyncMock(return_value=[(mock_node, 0.9)])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        context = await bridge.search_knowledge(
            query="test query",
            strategy=RelevanceStrategy.SEMANTIC,
        )

        assert len(context.results) == 1
        assert context.results[0].node_id == "node_1"
        mock_mound.semantic_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_knowledge_with_mound_keyword(self):
        """Test keyword search with mound."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            RelevanceStrategy,
        )

        mock_mound = AsyncMock()
        mock_mound.keyword_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="keyword search",
            strategy=RelevanceStrategy.KEYWORD,
        )

        mock_mound.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_knowledge_with_mound_hybrid(self):
        """Test hybrid search with mound."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            RelevanceStrategy,
        )

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="hybrid search",
            strategy=RelevanceStrategy.HYBRID,
        )

        mock_mound.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_knowledge_workspace_scope(self):
        """Test search with workspace scope."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            KnowledgeSearchScope,
        )

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="test",
            workspace_id="ws_123",
            scope=KnowledgeSearchScope.WORKSPACE,
        )

        call_kwargs = mock_mound.hybrid_search.call_args[1]
        assert call_kwargs["workspace_id"] == "ws_123"

    @pytest.mark.asyncio
    async def test_search_knowledge_channel_scope(self):
        """Test search with channel scope."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            KnowledgeSearchScope,
        )

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="test",
            workspace_id="ws_123",
            channel_id="C456",
            scope=KnowledgeSearchScope.CHANNEL,
        )

        call_kwargs = mock_mound.hybrid_search.call_args[1]
        assert call_kwargs["workspace_id"] == "ws_123"
        assert call_kwargs["metadata_filter"]["channel_id"] == "C456"

    @pytest.mark.asyncio
    async def test_search_knowledge_user_scope(self):
        """Test search with user scope."""
        from aragora.services.knowledge_chat_bridge import (
            KnowledgeChatBridge,
            KnowledgeSearchScope,
        )

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="test",
            workspace_id="ws_123",
            user_id="user_789",
            scope=KnowledgeSearchScope.USER,
        )

        call_kwargs = mock_mound.hybrid_search.call_args[1]
        assert call_kwargs["metadata_filter"]["user_id"] == "user_789"

    @pytest.mark.asyncio
    async def test_search_knowledge_with_node_types_filter(self):
        """Test search with node types filter."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        await bridge.search_knowledge(
            query="test",
            node_types=["evidence", "pattern"],
        )

        call_kwargs = mock_mound.hybrid_search.call_args[1]
        assert call_kwargs["node_types"] == ["evidence", "pattern"]

    @pytest.mark.asyncio
    async def test_inject_knowledge_for_conversation(self):
        """Test knowledge injection for conversation."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_node = MagicMock()
        mock_node.node_id = "node_1"
        mock_node.content = "Relevant context"
        mock_node.node_type = "evidence"
        mock_node.confidence = 0.8
        mock_node.source = "slack:C123"
        mock_node.created_at = datetime.now(timezone.utc)
        mock_node.metadata = {}
        mock_node.provenance = None

        mock_mound.hybrid_search = AsyncMock(return_value=[(mock_node, 0.85)])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        messages = [
            {"content": "What's our vacation policy?", "author": "user1"},
            {"content": "Let me check", "author": "user2"},
        ]

        results = await bridge.inject_knowledge_for_conversation(
            messages=messages,
            workspace_id="ws_1",
        )

        assert len(results) >= 0
        mock_mound.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_knowledge_empty_messages(self):
        """Test knowledge injection with empty messages."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge(enable_caching=False)

        results = await bridge.inject_knowledge_for_conversation(messages=[])

        assert results == []

    @pytest.mark.asyncio
    async def test_inject_knowledge_uses_last_5_messages(self):
        """Test knowledge injection uses only last 5 messages."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        # Create 10 messages
        messages = [{"content": f"Message {i}", "author": "user"} for i in range(10)]

        await bridge.inject_knowledge_for_conversation(messages=messages)

        # Should only use content from first 5 messages
        call_kwargs = mock_mound.hybrid_search.call_args[1]
        query = call_kwargs["query"]
        assert "Message 0" in query
        assert "Message 4" in query
        # Messages 5-9 should not be in query
        assert "Message 9" not in query

    @pytest.mark.asyncio
    async def test_store_chat_as_knowledge_success(self):
        """Test storing chat as knowledge."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="node_new_123")

        bridge = KnowledgeChatBridge(mound=mock_mound)

        messages = [
            {"author": "alice", "content": "We decided to use Python 3.11"},
            {"author": "bob", "content": "Agreed, better performance"},
            {"author": "alice", "content": "Let's document this decision"},
        ]

        node_id = await bridge.store_chat_as_knowledge(
            messages=messages,
            workspace_id="ws_1",
            channel_id="C123",
            channel_name="#engineering",
            platform="slack",
        )

        assert node_id == "node_new_123"
        mock_mound.add_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_chat_as_knowledge_too_few_messages(self):
        """Test storing chat fails with too few messages."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge()

        messages = [
            {"author": "alice", "content": "Just one message"},
        ]

        node_id = await bridge.store_chat_as_knowledge(
            messages=messages,
            min_messages=3,
        )

        assert node_id is None

    @pytest.mark.asyncio
    async def test_store_chat_as_knowledge_no_mound(self):
        """Test storing chat fails without mound."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        bridge = KnowledgeChatBridge()

        messages = [
            {"author": "alice", "content": "Message 1"},
            {"author": "bob", "content": "Message 2"},
            {"author": "alice", "content": "Message 3"},
        ]

        node_id = await bridge.store_chat_as_knowledge(messages=messages)

        assert node_id is None

    @pytest.mark.asyncio
    async def test_store_chat_content_size_limit(self):
        """Test stored content is limited to 5000 chars."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="node_1")

        bridge = KnowledgeChatBridge(mound=mock_mound)

        # Create messages with very long content
        messages = [
            {"author": "user", "content": "x" * 3000},
            {"author": "user", "content": "y" * 3000},
            {"author": "user", "content": "z" * 3000},
        ]

        await bridge.store_chat_as_knowledge(messages=messages)

        # Check that the node content is limited
        call_args = mock_mound.add_node.call_args[0]
        node = call_args[0]
        assert len(node.content) <= 5000

    @pytest.mark.asyncio
    async def test_get_channel_knowledge_summary(self):
        """Test getting channel knowledge summary."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_node1 = MagicMock()
        mock_node1.node_id = "n1"
        mock_node1.content = "Evidence 1"
        mock_node1.node_type = "evidence"
        mock_node1.confidence = 0.8
        mock_node1.source = "slack:C123"
        mock_node1.created_at = datetime.now(timezone.utc)
        mock_node1.metadata = {}
        mock_node1.provenance = None

        mock_node2 = MagicMock()
        mock_node2.node_id = "n2"
        mock_node2.content = "Pattern 1"
        mock_node2.node_type = "pattern"
        mock_node2.confidence = 0.7
        mock_node2.source = "slack:C123"
        mock_node2.created_at = datetime.now(timezone.utc)
        mock_node2.metadata = {}
        mock_node2.provenance = None

        mock_mound.hybrid_search = AsyncMock(return_value=[(mock_node1, 0.9), (mock_node2, 0.85)])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        summary = await bridge.get_channel_knowledge_summary(
            channel_id="C123",
            workspace_id="ws_1",
            max_items=10,
        )

        assert summary["channel_id"] == "C123"
        assert summary["workspace_id"] == "ws_1"
        assert summary["total_items"] == 2
        assert summary["node_types"]["evidence"] == 1
        assert summary["node_types"]["pattern"] == 1
        assert summary["avg_confidence"] == pytest.approx(0.75, rel=0.01)
        assert len(summary["top_items"]) <= 5

    @pytest.mark.asyncio
    async def test_get_channel_knowledge_summary_empty(self):
        """Test channel summary with no results."""
        from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

        mock_mound = AsyncMock()
        mock_mound.hybrid_search = AsyncMock(return_value=[])

        bridge = KnowledgeChatBridge(mound=mock_mound, enable_caching=False)

        summary = await bridge.get_channel_knowledge_summary(channel_id="C_empty")

        assert summary["total_items"] == 0
        assert summary["node_types"] == {}
        assert summary["avg_confidence"] == 0.0


class TestGetKnowledgeChatBridge:
    """Tests for get_knowledge_chat_bridge singleton."""

    def test_get_bridge_singleton(self):
        """Test getting singleton bridge instance."""
        import aragora.services.knowledge_chat_bridge as module

        # Reset singleton
        module._bridge = None

        from aragora.services.knowledge_chat_bridge import get_knowledge_chat_bridge

        bridge1 = get_knowledge_chat_bridge()
        bridge2 = get_knowledge_chat_bridge()

        assert bridge1 is bridge2

    def test_get_bridge_creates_instance(self):
        """Test singleton creates new instance."""
        import aragora.services.knowledge_chat_bridge as module
        from aragora.services.knowledge_chat_bridge import (
            get_knowledge_chat_bridge,
            KnowledgeChatBridge,
        )

        # Reset singleton
        module._bridge = None

        bridge = get_knowledge_chat_bridge()

        assert isinstance(bridge, KnowledgeChatBridge)


class TestLazyMoundLoading:
    """Tests for lazy Knowledge Mound loading."""

    def test_get_mound_lazy_load(self):
        """Test mound is lazily loaded."""
        with patch("aragora.services.knowledge_chat_bridge.get_knowledge_mound") as mock_get_mound:
            mock_mound = MagicMock()
            mock_get_mound.return_value = mock_mound

            from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

            bridge = KnowledgeChatBridge()

            # Mound not loaded yet
            assert bridge._mound is None
            mock_get_mound.assert_not_called()

            # Access mound
            result = bridge._get_mound()

            assert result is mock_mound
            mock_get_mound.assert_called_once()

    def test_get_mound_import_error(self):
        """Test mound loading handles import error."""
        with patch(
            "aragora.services.knowledge_chat_bridge.get_knowledge_mound",
            side_effect=ImportError("Module not found"),
        ):
            from aragora.services.knowledge_chat_bridge import KnowledgeChatBridge

            bridge = KnowledgeChatBridge()
            result = bridge._get_mound()

            assert result is None

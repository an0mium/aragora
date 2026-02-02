"""Tests for Knowledge Chat namespace API.

Tests the KnowledgeChatAPI (sync) and AsyncKnowledgeChatAPI classes which
provide methods for Knowledge + Chat bridge integration:
- Search knowledge from chat context
- Inject knowledge into conversations
- Store chat messages as knowledge
- Channel knowledge summaries
"""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# ---------------------------------------------------------------------------
# Sync: Search
# ---------------------------------------------------------------------------


class TestKnowledgeChatSearch:
    """Tests for search() method."""

    def test_search_minimal(self, client: AragoraClient, mock_request) -> None:
        """Search with only required query parameter."""
        mock_request.return_value = {
            "results": [{"id": "node_1", "content": "Vacation policy info"}],
            "total": 1,
            "scope": "workspace",
        }

        result = client.knowledge_chat.search("What's our vacation policy?")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "What's our vacation policy?",
                "workspace_id": "default",
            },
        )
        assert result["total"] == 1
        assert result["results"][0]["id"] == "node_1"

    def test_search_with_channel_context(self, client: AragoraClient, mock_request) -> None:
        """Search with channel context."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "deployment process",
            workspace_id="ws_123",
            channel_id="C123456",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "deployment process",
                "workspace_id": "ws_123",
                "channel_id": "C123456",
            },
        )

    def test_search_with_user_context(self, client: AragoraClient, mock_request) -> None:
        """Search with user personalization."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "my projects",
            user_id="U789",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "my projects",
                "workspace_id": "default",
                "user_id": "U789",
            },
        )

    def test_search_with_scope_global(self, client: AragoraClient, mock_request) -> None:
        """Search with global scope."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "best practices",
            scope="global",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "best practices",
                "workspace_id": "default",
                "scope": "global",
            },
        )

    def test_search_with_channel_scope(self, client: AragoraClient, mock_request) -> None:
        """Search with channel scope."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "channel history",
            scope="channel",
            channel_id="C123",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["scope"] == "channel"
        assert call_kwargs["json"]["channel_id"] == "C123"

    def test_search_with_user_scope(self, client: AragoraClient, mock_request) -> None:
        """Search with user scope."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "my notes",
            scope="user",
            user_id="U456",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["scope"] == "user"
        assert call_kwargs["json"]["user_id"] == "U456"

    def test_search_with_semantic_strategy(self, client: AragoraClient, mock_request) -> None:
        """Search with semantic strategy."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "similar concepts",
            strategy="semantic",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "similar concepts",
                "workspace_id": "default",
                "strategy": "semantic",
            },
        )

    def test_search_with_keyword_strategy(self, client: AragoraClient, mock_request) -> None:
        """Search with keyword strategy."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "exact term",
            strategy="keyword",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["strategy"] == "keyword"

    def test_search_with_exact_strategy(self, client: AragoraClient, mock_request) -> None:
        """Search with exact match strategy."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "ERROR_CODE_123",
            strategy="exact",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["strategy"] == "exact"

    def test_search_with_node_types(self, client: AragoraClient, mock_request) -> None:
        """Search filtered by node types."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "api documentation",
            node_types=["fact", "insight", "decision"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "api documentation",
                "workspace_id": "default",
                "node_types": ["fact", "insight", "decision"],
            },
        )

    def test_search_with_min_confidence(self, client: AragoraClient, mock_request) -> None:
        """Search with custom confidence threshold."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "verified facts",
            min_confidence=0.8,
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "verified facts",
                "workspace_id": "default",
                "min_confidence": 0.8,
            },
        )

    def test_search_with_max_results(self, client: AragoraClient, mock_request) -> None:
        """Search with custom result limit."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge_chat.search(
            "recent updates",
            max_results=25,
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "recent updates",
                "workspace_id": "default",
                "max_results": 25,
            },
        )

    def test_search_with_all_options(self, client: AragoraClient, mock_request) -> None:
        """Search with all options specified."""
        mock_request.return_value = {
            "results": [{"id": "node_1"}, {"id": "node_2"}],
            "total": 2,
            "scope": "channel",
            "strategy": "semantic",
        }

        result = client.knowledge_chat.search(
            "comprehensive search",
            workspace_id="ws_enterprise",
            channel_id="C_engineering",
            user_id="U_admin",
            scope="channel",
            strategy="semantic",
            node_types=["fact", "decision"],
            min_confidence=0.7,
            max_results=50,
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/search",
            json={
                "query": "comprehensive search",
                "workspace_id": "ws_enterprise",
                "channel_id": "C_engineering",
                "user_id": "U_admin",
                "scope": "channel",
                "strategy": "semantic",
                "node_types": ["fact", "decision"],
                "min_confidence": 0.7,
                "max_results": 50,
            },
        )
        assert result["total"] == 2

    def test_search_default_values_not_sent(self, client: AragoraClient, mock_request) -> None:
        """Verify default values are not included in request."""
        mock_request.return_value = {"results": [], "total": 0}

        # Call with only query (all defaults)
        client.knowledge_chat.search("test query")

        call_kwargs = mock_request.call_args[1]
        request_json = call_kwargs["json"]

        # Should NOT include defaults
        assert "scope" not in request_json  # default is "workspace"
        assert "strategy" not in request_json  # default is "hybrid"
        assert "min_confidence" not in request_json  # default is 0.3
        assert "max_results" not in request_json  # default is 10
        assert "node_types" not in request_json
        assert "channel_id" not in request_json
        assert "user_id" not in request_json


# ---------------------------------------------------------------------------
# Sync: Inject
# ---------------------------------------------------------------------------


class TestKnowledgeChatInject:
    """Tests for inject() method."""

    def test_inject_minimal(self, client: AragoraClient, mock_request) -> None:
        """Inject with minimal parameters."""
        mock_request.return_value = {
            "context_items": [{"id": "ctx_1", "content": "Relevant knowledge", "score": 0.92}],
            "count": 1,
        }

        messages = [
            {"author": "user1", "content": "How do we handle auth?"},
            {"author": "user2", "content": "I think we use OAuth 2.0"},
        ]

        result = client.knowledge_chat.inject(messages)

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/inject",
            json={
                "messages": messages,
                "workspace_id": "default",
            },
        )
        assert result["count"] == 1
        assert result["context_items"][0]["id"] == "ctx_1"

    def test_inject_with_workspace(self, client: AragoraClient, mock_request) -> None:
        """Inject with workspace context."""
        mock_request.return_value = {"context_items": [], "count": 0}

        messages = [{"author": "bot", "content": "Hello!"}]

        client.knowledge_chat.inject(
            messages,
            workspace_id="ws_acme",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/inject",
            json={
                "messages": messages,
                "workspace_id": "ws_acme",
            },
        )

    def test_inject_with_channel(self, client: AragoraClient, mock_request) -> None:
        """Inject with channel context."""
        mock_request.return_value = {"context_items": [], "count": 0}

        messages = [
            {"author": "user", "content": "Question about CI/CD"},
        ]

        client.knowledge_chat.inject(
            messages,
            channel_id="C_devops",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/inject",
            json={
                "messages": messages,
                "workspace_id": "default",
                "channel_id": "C_devops",
            },
        )

    def test_inject_with_custom_max_items(self, client: AragoraClient, mock_request) -> None:
        """Inject with custom max context items."""
        mock_request.return_value = {"context_items": [], "count": 0}

        messages = [{"author": "assistant", "content": "Processing..."}]

        client.knowledge_chat.inject(
            messages,
            max_context_items=15,
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/inject",
            json={
                "messages": messages,
                "workspace_id": "default",
                "max_context_items": 15,
            },
        )

    def test_inject_with_all_options(self, client: AragoraClient, mock_request) -> None:
        """Inject with all options specified."""
        mock_request.return_value = {
            "context_items": [
                {"id": "ctx_1", "content": "OAuth 2.0 docs", "score": 0.95},
                {"id": "ctx_2", "content": "Auth best practices", "score": 0.88},
            ],
            "count": 2,
        }

        messages = [
            {"author": "user1", "content": "We need to secure the API"},
            {"author": "user2", "content": "What auth method should we use?"},
            {"author": "user3", "content": "OAuth seems standard"},
        ]

        result = client.knowledge_chat.inject(
            messages,
            workspace_id="ws_security",
            channel_id="C_api_team",
            max_context_items=10,
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/inject",
            json={
                "messages": messages,
                "workspace_id": "ws_security",
                "channel_id": "C_api_team",
                "max_context_items": 10,
            },
        )
        assert result["count"] == 2

    def test_inject_default_max_items_not_sent(self, client: AragoraClient, mock_request) -> None:
        """Verify default max_context_items is not included in request."""
        mock_request.return_value = {"context_items": [], "count": 0}

        client.knowledge_chat.inject([{"author": "test", "content": "test"}])

        call_kwargs = mock_request.call_args[1]
        request_json = call_kwargs["json"]

        # Default (5) should not be sent
        assert "max_context_items" not in request_json


# ---------------------------------------------------------------------------
# Sync: Store
# ---------------------------------------------------------------------------


class TestKnowledgeChatStore:
    """Tests for store() method."""

    def test_store_minimal(self, client: AragoraClient, mock_request) -> None:
        """Store with minimum required messages."""
        mock_request.return_value = {
            "node_id": "kn_12345",
            "message_count": 2,
            "stored": True,
        }

        messages = [
            {"author": "user1", "content": "We decided to use Python 3.11"},
            {"author": "user2", "content": "Agreed, better performance"},
        ]

        result = client.knowledge_chat.store(messages)

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "default",
            },
        )
        assert result["node_id"] == "kn_12345"
        assert result["message_count"] == 2

    def test_store_validation_error_empty_messages(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Store raises ValueError for empty messages list."""
        with pytest.raises(ValueError, match="At least 2 messages required"):
            client.knowledge_chat.store([])

        # Verify no HTTP request was made
        mock_request.assert_not_called()

    def test_store_validation_error_single_message(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Store raises ValueError for single message."""
        with pytest.raises(ValueError, match="At least 2 messages required"):
            client.knowledge_chat.store([{"author": "user", "content": "Just one message"}])

        mock_request.assert_not_called()

    def test_store_with_workspace(self, client: AragoraClient, mock_request) -> None:
        """Store with workspace specified."""
        mock_request.return_value = {"node_id": "kn_ws", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(
            messages,
            workspace_id="ws_acme",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "ws_acme",
            },
        )

    def test_store_with_channel_id(self, client: AragoraClient, mock_request) -> None:
        """Store with channel ID."""
        mock_request.return_value = {"node_id": "kn_ch", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(
            messages,
            channel_id="C123456",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "default",
                "channel_id": "C123456",
            },
        )

    def test_store_with_channel_name(self, client: AragoraClient, mock_request) -> None:
        """Store with human-readable channel name."""
        mock_request.return_value = {"node_id": "kn_name", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(
            messages,
            channel_name="#engineering",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "default",
                "channel_name": "#engineering",
            },
        )

    def test_store_with_platform_slack(self, client: AragoraClient, mock_request) -> None:
        """Store with Slack platform."""
        mock_request.return_value = {"node_id": "kn_slack", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(
            messages,
            platform="slack",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "default",
                "platform": "slack",
            },
        )

    def test_store_with_platform_teams(self, client: AragoraClient, mock_request) -> None:
        """Store with Microsoft Teams platform."""
        mock_request.return_value = {"node_id": "kn_teams", "message_count": 3}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
            {"author": "c", "content": "msg3"},
        ]

        client.knowledge_chat.store(
            messages,
            platform="teams",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["platform"] == "teams"

    def test_store_with_platform_discord(self, client: AragoraClient, mock_request) -> None:
        """Store with Discord platform."""
        mock_request.return_value = {"node_id": "kn_discord", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(
            messages,
            platform="discord",
        )

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["platform"] == "discord"

    def test_store_with_custom_node_type(self, client: AragoraClient, mock_request) -> None:
        """Store with custom node type."""
        mock_request.return_value = {"node_id": "kn_decision", "message_count": 2}

        messages = [
            {"author": "pm", "content": "We should use Redis for caching"},
            {"author": "eng", "content": "Agreed, it fits our scale"},
        ]

        client.knowledge_chat.store(
            messages,
            node_type="decision",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "default",
                "node_type": "decision",
            },
        )

    def test_store_with_all_options(self, client: AragoraClient, mock_request) -> None:
        """Store with all options specified."""
        mock_request.return_value = {
            "node_id": "kn_full_123",
            "message_count": 3,
            "stored": True,
            "platform": "slack",
        }

        messages = [
            {"author": "cto", "content": "Architecture review complete"},
            {"author": "lead", "content": "Microservices approach approved"},
            {"author": "dev", "content": "Starting implementation tomorrow"},
        ]

        result = client.knowledge_chat.store(
            messages,
            workspace_id="ws_eng",
            channel_id="C_architecture",
            channel_name="#architecture-review",
            platform="slack",
            node_type="architectural_decision",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/chat/knowledge/store",
            json={
                "messages": messages,
                "workspace_id": "ws_eng",
                "channel_id": "C_architecture",
                "channel_name": "#architecture-review",
                "platform": "slack",
                "node_type": "architectural_decision",
            },
        )
        assert result["node_id"] == "kn_full_123"
        assert result["message_count"] == 3

    def test_store_default_values_not_sent(self, client: AragoraClient, mock_request) -> None:
        """Verify default values are not included in request."""
        mock_request.return_value = {"node_id": "kn_test", "message_count": 2}

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        client.knowledge_chat.store(messages)

        call_kwargs = mock_request.call_args[1]
        request_json = call_kwargs["json"]

        # Defaults should not be sent
        assert "channel_id" not in request_json  # default is ""
        assert "channel_name" not in request_json  # default is ""
        assert "platform" not in request_json  # default is "unknown"
        assert "node_type" not in request_json  # default is "chat_context"


# ---------------------------------------------------------------------------
# Sync: Get Channel Summary
# ---------------------------------------------------------------------------


class TestKnowledgeChatGetChannelSummary:
    """Tests for get_channel_summary() method."""

    def test_get_channel_summary_minimal(self, client: AragoraClient, mock_request) -> None:
        """Get channel summary with only channel ID."""
        mock_request.return_value = {
            "channel_id": "C123",
            "total_items": 42,
            "recent_topics": ["API design", "Testing"],
            "knowledge_items": [],
        }

        result = client.knowledge_chat.get_channel_summary("C123")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/chat/knowledge/channel/C123/summary",
            params={"workspace_id": "default"},
        )
        assert result["channel_id"] == "C123"
        assert result["total_items"] == 42

    def test_get_channel_summary_with_workspace(self, client: AragoraClient, mock_request) -> None:
        """Get channel summary with workspace specified."""
        mock_request.return_value = {
            "channel_id": "C456",
            "total_items": 10,
            "knowledge_items": [],
        }

        client.knowledge_chat.get_channel_summary(
            "C456",
            workspace_id="ws_acme",
        )

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/chat/knowledge/channel/C456/summary",
            params={"workspace_id": "ws_acme"},
        )

    def test_get_channel_summary_with_max_items(self, client: AragoraClient, mock_request) -> None:
        """Get channel summary with custom max items."""
        mock_request.return_value = {
            "channel_id": "C789",
            "total_items": 100,
            "knowledge_items": [{"id": "k1"}, {"id": "k2"}],
        }

        client.knowledge_chat.get_channel_summary(
            "C789",
            max_items=25,
        )

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/chat/knowledge/channel/C789/summary",
            params={"workspace_id": "default", "max_items": 25},
        )

    def test_get_channel_summary_with_all_options(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Get channel summary with all options specified."""
        mock_request.return_value = {
            "channel_id": "C_eng",
            "workspace_id": "ws_prod",
            "total_items": 500,
            "recent_topics": ["Infrastructure", "Scaling", "Monitoring"],
            "knowledge_items": [
                {"id": "k1", "type": "decision"},
                {"id": "k2", "type": "fact"},
            ],
        }

        result = client.knowledge_chat.get_channel_summary(
            "C_eng",
            workspace_id="ws_prod",
            max_items=50,
        )

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/chat/knowledge/channel/C_eng/summary",
            params={"workspace_id": "ws_prod", "max_items": 50},
        )
        assert result["total_items"] == 500
        assert len(result["knowledge_items"]) == 2

    def test_get_channel_summary_default_max_items_not_sent(
        self, client: AragoraClient, mock_request
    ) -> None:
        """Verify default max_items is not included in params."""
        mock_request.return_value = {"channel_id": "C1", "total_items": 0}

        client.knowledge_chat.get_channel_summary("C1")

        call_kwargs = mock_request.call_args[1]
        assert "max_items" not in call_kwargs["params"]


# ---------------------------------------------------------------------------
# Async Tests: Search
# ---------------------------------------------------------------------------


class TestAsyncKnowledgeChatSearch:
    """Tests for async search() method."""

    @pytest.mark.asyncio
    async def test_search_minimal(self, mock_async_request) -> None:
        """Async search with minimal parameters."""
        mock_async_request.return_value = {"results": [], "total": 0}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.search("test query")

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/chat/knowledge/search",
                json={
                    "query": "test query",
                    "workspace_id": "default",
                },
            )
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_with_all_options(self, mock_async_request) -> None:
        """Async search with all options."""
        mock_async_request.return_value = {
            "results": [{"id": "r1"}],
            "total": 1,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.search(
                "comprehensive async search",
                workspace_id="ws_async",
                channel_id="C_async",
                user_id="U_async",
                scope="channel",
                strategy="semantic",
                node_types=["fact"],
                min_confidence=0.9,
                max_results=5,
            )

            call_kwargs = mock_async_request.call_args[1]
            request_json = call_kwargs["json"]
            assert request_json["query"] == "comprehensive async search"
            assert request_json["workspace_id"] == "ws_async"
            assert request_json["channel_id"] == "C_async"
            assert request_json["user_id"] == "U_async"
            assert request_json["scope"] == "channel"
            assert request_json["strategy"] == "semantic"
            assert request_json["node_types"] == ["fact"]
            assert request_json["min_confidence"] == 0.9
            assert request_json["max_results"] == 5
            assert result["total"] == 1


# ---------------------------------------------------------------------------
# Async Tests: Inject
# ---------------------------------------------------------------------------


class TestAsyncKnowledgeChatInject:
    """Tests for async inject() method."""

    @pytest.mark.asyncio
    async def test_inject_minimal(self, mock_async_request) -> None:
        """Async inject with minimal parameters."""
        mock_async_request.return_value = {"context_items": [], "count": 0}

        messages = [{"author": "user", "content": "Hello"}]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.inject(messages)

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/chat/knowledge/inject",
                json={
                    "messages": messages,
                    "workspace_id": "default",
                },
            )
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_inject_with_all_options(self, mock_async_request) -> None:
        """Async inject with all options."""
        mock_async_request.return_value = {
            "context_items": [{"id": "ctx_1"}],
            "count": 1,
        }

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.inject(
                messages,
                workspace_id="ws_inject",
                channel_id="C_inject",
                max_context_items=20,
            )

            call_kwargs = mock_async_request.call_args[1]
            request_json = call_kwargs["json"]
            assert request_json["workspace_id"] == "ws_inject"
            assert request_json["channel_id"] == "C_inject"
            assert request_json["max_context_items"] == 20
            assert result["count"] == 1


# ---------------------------------------------------------------------------
# Async Tests: Store
# ---------------------------------------------------------------------------


class TestAsyncKnowledgeChatStore:
    """Tests for async store() method."""

    @pytest.mark.asyncio
    async def test_store_minimal(self, mock_async_request) -> None:
        """Async store with minimal parameters."""
        mock_async_request.return_value = {
            "node_id": "kn_async_1",
            "message_count": 2,
        }

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
        ]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.store(messages)

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/chat/knowledge/store",
                json={
                    "messages": messages,
                    "workspace_id": "default",
                },
            )
            assert result["node_id"] == "kn_async_1"

    @pytest.mark.asyncio
    async def test_store_validation_error_empty(self, mock_async_request) -> None:
        """Async store raises ValueError for empty messages."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            with pytest.raises(ValueError, match="At least 2 messages required"):
                await client.knowledge_chat.store([])

            mock_async_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_validation_error_single(self, mock_async_request) -> None:
        """Async store raises ValueError for single message."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            with pytest.raises(ValueError, match="At least 2 messages required"):
                await client.knowledge_chat.store([{"author": "user", "content": "one"}])

            mock_async_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_all_options(self, mock_async_request) -> None:
        """Async store with all options."""
        mock_async_request.return_value = {
            "node_id": "kn_async_full",
            "message_count": 3,
        }

        messages = [
            {"author": "a", "content": "msg1"},
            {"author": "b", "content": "msg2"},
            {"author": "c", "content": "msg3"},
        ]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.store(
                messages,
                workspace_id="ws_store",
                channel_id="C_store",
                channel_name="#async-channel",
                platform="slack",
                node_type="insight",
            )

            call_kwargs = mock_async_request.call_args[1]
            request_json = call_kwargs["json"]
            assert request_json["workspace_id"] == "ws_store"
            assert request_json["channel_id"] == "C_store"
            assert request_json["channel_name"] == "#async-channel"
            assert request_json["platform"] == "slack"
            assert request_json["node_type"] == "insight"
            assert result["message_count"] == 3


# ---------------------------------------------------------------------------
# Async Tests: Get Channel Summary
# ---------------------------------------------------------------------------


class TestAsyncKnowledgeChatGetChannelSummary:
    """Tests for async get_channel_summary() method."""

    @pytest.mark.asyncio
    async def test_get_channel_summary_minimal(self, mock_async_request) -> None:
        """Async get channel summary with minimal parameters."""
        mock_async_request.return_value = {
            "channel_id": "C_async",
            "total_items": 5,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.get_channel_summary("C_async")

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/chat/knowledge/channel/C_async/summary",
                params={"workspace_id": "default"},
            )
            assert result["channel_id"] == "C_async"

    @pytest.mark.asyncio
    async def test_get_channel_summary_with_all_options(self, mock_async_request) -> None:
        """Async get channel summary with all options."""
        mock_async_request.return_value = {
            "channel_id": "C_full",
            "workspace_id": "ws_full",
            "total_items": 100,
            "knowledge_items": [{"id": "k1"}, {"id": "k2"}],
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge_chat.get_channel_summary(
                "C_full",
                workspace_id="ws_full",
                max_items=30,
            )

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/chat/knowledge/channel/C_full/summary",
                params={"workspace_id": "ws_full", "max_items": 30},
            )
            assert result["total_items"] == 100
            assert len(result["knowledge_items"]) == 2

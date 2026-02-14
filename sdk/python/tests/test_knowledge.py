"""Tests for Knowledge namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# ---------------------------------------------------------------------------
# Sync: Fact CRUD
# ---------------------------------------------------------------------------

class TestKnowledgeFactCRUD:
    """Tests for basic fact create / read / update / delete."""

    def test_create_fact_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"id": "fact_1", "statement": "Sky is blue"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.knowledge.create_fact("Sky is blue")
            mock.assert_called_once_with(
                "POST",
                "/api/v1/knowledge/facts",
                json={
                    "statement": "Sky is blue",
                    "workspace_id": "default",
                    "confidence": 0.5,
                },
            )
            assert result["id"] == "fact_1"
            client.close()

    def test_create_fact_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"id": "fact_2"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.knowledge.create_fact(
                "Water boils at 100C",
                workspace_id="ws_1",
                confidence=0.95,
                topics=["physics", "chemistry"],
                evidence_ids=["ev_1"],
                source_documents=["doc_1"],
                metadata={"verified": True},
            )
            mock.assert_called_once_with(
                "POST",
                "/api/v1/knowledge/facts",
                json={
                    "statement": "Water boils at 100C",
                    "workspace_id": "ws_1",
                    "confidence": 0.95,
                    "topics": ["physics", "chemistry"],
                    "evidence_ids": ["ev_1"],
                    "source_documents": ["doc_1"],
                    "metadata": {"verified": True},
                },
            )
            client.close()

# ---------------------------------------------------------------------------
# Sync: Search & Query
# ---------------------------------------------------------------------------

class TestKnowledgeSearchQuery:
    """Tests for search and query operations."""

    def test_search_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"results": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.knowledge.search("rate limiting")
            mock.assert_called_once_with(
                "GET",
                "/api/v1/knowledge/search",
                params={"q": "rate limiting", "limit": 10},
            )
            assert result["total"] == 0
            client.close()

    def test_search_with_workspace_and_limit(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"results": [{"id": "r1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.knowledge.search("rate limiting", workspace_id="ws_1", limit=5)
            mock.assert_called_once_with(
                "GET",
                "/api/v1/knowledge/search",
                params={"q": "rate limiting", "limit": 5, "workspace_id": "ws_1"},
            )
            client.close()

    def test_query(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"answer": "42", "sources": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.knowledge.query("What is the meaning of life?")
            mock.assert_called_once_with(
                "POST",
                "/api/v1/knowledge/query",
                json={"prompt": "What is the meaning of life?"},
            )
            assert result["answer"] == "42"
            client.close()

    def test_list_facts_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"facts": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.knowledge.list_facts(
                workspace_id="ws_1",
                topic="physics",
                min_confidence=0.8,
                status="verified",
                include_superseded=True,
                limit=20,
                offset=10,
            )
            mock.assert_called_once_with(
                "GET",
                "/api/v1/knowledge/facts",
                params={
                    "min_confidence": 0.8,
                    "include_superseded": True,
                    "limit": 20,
                    "offset": 10,
                    "workspace_id": "ws_1",
                    "topic": "physics",
                    "status": "verified",
                },
            )
            client.close()

# ---------------------------------------------------------------------------
# Sync: Relations & Contradictions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sync: Stats & Mound
# ---------------------------------------------------------------------------

class TestKnowledgeStatsMound:
    """Tests for stats and mound operations."""

    def test_get_stats_with_workspace(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"total_facts": 42}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.knowledge.get_stats(workspace_id="ws_1")
            mock.assert_called_once_with(
                "GET", "/api/v1/knowledge/stats", params={"workspace_id": "ws_1"}
            )
            client.close()

    def test_add_node(self) -> None:
        with patch.object(AragoraClient, "request") as mock:
            mock.return_value = {"id": "node_1"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.knowledge.add_node(
                "Important finding",
                node_type="insight",
                metadata={"source": "debate_1"},
                workspace_id="ws_1",
            )
            mock.assert_called_once_with(
                "POST",
                "/api/knowledge/mound/nodes",
                json={
                    "content": "Important finding",
                    "node_type": "insight",
                    "metadata": {"source": "debate_1"},
                    "workspace_id": "ws_1",
                },
            )
            assert result["id"] == "node_1"
            client.close()

# ---------------------------------------------------------------------------
# Async Tests
# ---------------------------------------------------------------------------

class TestAsyncKnowledge:
    """Tests for async knowledge methods."""

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock:
            mock.return_value = {"results": [{"id": "r1"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.knowledge.search("embeddings", limit=20)
            mock.assert_called_once_with(
                "GET",
                "/api/v1/knowledge/search",
                params={"q": "embeddings", "limit": 20},
            )
            assert len(result["results"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_create_fact(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock:
            mock.return_value = {"id": "fact_async_1"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.knowledge.create_fact(
                "Async fact", confidence=0.8, topics=["testing"]
            )
            mock.assert_called_once_with(
                "POST",
                "/api/v1/knowledge/facts",
                json={
                    "statement": "Async fact",
                    "workspace_id": "default",
                    "confidence": 0.8,
                    "topics": ["testing"],
                },
            )
            assert result["id"] == "fact_async_1"
            await client.close()

    @pytest.mark.asyncio
    async def test_resolve_contradiction(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock:
            mock.return_value = {"resolved": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.knowledge.resolve_contradiction(
                "contra_1", resolution="keep_newer", keep_node_id="node_5"
            )
            mock.assert_called_once_with(
                "POST",
                "/api/knowledge/mound/contradictions/contra_1/resolve",
                json={"resolution": "keep_newer", "keep_node_id": "node_5"},
            )
            assert result["resolved"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_from_debate(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock:
            mock.return_value = {"claims": [], "count": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.knowledge.extract_from_debate(
                "debate_42", confidence_threshold=0.9, auto_promote=True
            )
            mock.assert_called_once_with(
                "POST",
                "/api/v1/knowledge/mound/extraction/debate",
                json={
                    "debate_id": "debate_42",
                    "confidence_threshold": 0.9,
                    "auto_promote": True,
                },
            )
            assert result["count"] == 0
            await client.close()

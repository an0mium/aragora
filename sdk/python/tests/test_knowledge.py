"""Tests for Knowledge namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestKnowledgeSearch:
    """Tests for knowledge search and query operations."""

    def test_search_with_query(self, client: AragoraClient, mock_request) -> None:
        """Search knowledge base with a query string."""
        mock_request.return_value = {
            "results": [{"fact_id": "f_1", "statement": "Python is interpreted"}],
            "total": 1,
        }

        result = client.knowledge.search("Python")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/knowledge/search",
            params={"q": "Python", "limit": 10},
        )
        assert result["total"] == 1

    def test_search_with_workspace_filter(self, client: AragoraClient, mock_request) -> None:
        """Search knowledge base filtered by workspace."""
        mock_request.return_value = {"results": [], "total": 0}

        client.knowledge.search("query", workspace_id="ws_123", limit=5)

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/knowledge/search",
            params={"q": "query", "limit": 5, "workspace_id": "ws_123"},
        )

    def test_query_natural_language(self, client: AragoraClient, mock_request) -> None:
        """Run a natural language query against the knowledge base."""
        mock_request.return_value = {
            "answer": "Python was created by Guido van Rossum",
            "sources": [],
        }

        result = client.knowledge.query("Who created Python?")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/knowledge/query",
            json={"prompt": "Who created Python?"},
        )
        assert "answer" in result


class TestKnowledgeFactsCRUD:
    """Tests for fact CRUD operations."""

    def test_list_facts_default(self, client: AragoraClient, mock_request) -> None:
        """List facts with default parameters."""
        mock_request.return_value = {"facts": [], "total": 0}

        client.knowledge.list_facts()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/knowledge/facts",
            params={
                "min_confidence": 0.0,
                "include_superseded": False,
                "limit": 50,
                "offset": 0,
            },
        )

    def test_list_facts_with_filters(self, client: AragoraClient, mock_request) -> None:
        """List facts with workspace and topic filters."""
        mock_request.return_value = {"facts": []}

        client.knowledge.list_facts(
            workspace_id="ws_1",
            topic="security",
            min_confidence=0.8,
            status="verified",
            include_superseded=True,
            limit=10,
            offset=5,
        )

        call_args = mock_request.call_args
        params = call_args[1]["params"]
        assert params["workspace_id"] == "ws_1"
        assert params["topic"] == "security"
        assert params["min_confidence"] == 0.8
        assert params["status"] == "verified"
        assert params["include_superseded"] is True

    def test_get_fact(self, client: AragoraClient, mock_request) -> None:
        """Get a fact by ID."""
        mock_request.return_value = {
            "fact_id": "f_123",
            "statement": "Earth orbits the Sun",
            "confidence": 0.99,
        }

        result = client.knowledge.get_fact("f_123")

        mock_request.assert_called_once_with("GET", "/api/v1/knowledge/facts/f_123")
        assert result["confidence"] == 0.99

    def test_create_fact_minimal(self, client: AragoraClient, mock_request) -> None:
        """Create a fact with only required fields."""
        mock_request.return_value = {"fact_id": "f_new", "statement": "Water is wet"}

        result = client.knowledge.create_fact("Water is wet")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/knowledge/facts",
            json={
                "statement": "Water is wet",
                "workspace_id": "default",
                "confidence": 0.5,
            },
        )
        assert result["fact_id"] == "f_new"

    def test_create_fact_full(self, client: AragoraClient, mock_request) -> None:
        """Create a fact with all optional fields."""
        mock_request.return_value = {"fact_id": "f_new"}

        client.knowledge.create_fact(
            statement="Python 3.12 supports f-strings",
            workspace_id="ws_dev",
            confidence=0.9,
            topics=["python", "programming"],
            evidence_ids=["ev_1", "ev_2"],
            source_documents=["doc_1"],
            metadata={"author": "test"},
        )

        call_json = mock_request.call_args[1]["json"]
        assert call_json["statement"] == "Python 3.12 supports f-strings"
        assert call_json["workspace_id"] == "ws_dev"
        assert call_json["confidence"] == 0.9
        assert call_json["topics"] == ["python", "programming"]
        assert call_json["evidence_ids"] == ["ev_1", "ev_2"]
        assert call_json["metadata"] == {"author": "test"}

    def test_update_fact(self, client: AragoraClient, mock_request) -> None:
        """Update an existing fact."""
        mock_request.return_value = {"fact_id": "f_123", "confidence": 0.95}

        client.knowledge.update_fact("f_123", confidence=0.95, validation_status="verified")

        mock_request.assert_called_once_with(
            "PUT",
            "/api/v1/knowledge/facts/f_123",
            json={"confidence": 0.95, "validation_status": "verified"},
        )

    def test_update_fact_superseded(self, client: AragoraClient, mock_request) -> None:
        """Mark a fact as superseded by another."""
        mock_request.return_value = {"fact_id": "f_old"}

        client.knowledge.update_fact("f_old", superseded_by="f_new")

        call_json = mock_request.call_args[1]["json"]
        assert call_json["superseded_by"] == "f_new"

    def test_delete_fact(self, client: AragoraClient, mock_request) -> None:
        """Delete a fact."""
        mock_request.return_value = {"deleted": True}

        result = client.knowledge.delete_fact("f_123")

        mock_request.assert_called_once_with("DELETE", "/api/v1/knowledge/facts/f_123")
        assert result["deleted"] is True

    def test_verify_fact(self, client: AragoraClient, mock_request) -> None:
        """Verify a fact using agents."""
        mock_request.return_value = {"verified": True, "confidence": 0.92}

        result = client.knowledge.verify_fact("f_123")

        mock_request.assert_called_once_with("POST", "/api/v1/knowledge/facts/f_123/verify")
        assert result["verified"] is True


class TestKnowledgeRelations:
    """Tests for fact relations and contradictions."""

    def test_list_contradictions(self, client: AragoraClient, mock_request) -> None:
        """List contradictions for a fact."""
        mock_request.return_value = {"contradictions": []}

        client.knowledge.list_contradictions("f_123")

        mock_request.assert_called_once_with("GET", "/api/v1/knowledge/facts/f_123/contradictions")

    def test_list_relations_default(self, client: AragoraClient, mock_request) -> None:
        """List relations for a fact with defaults."""
        mock_request.return_value = {"relations": []}

        client.knowledge.list_relations("f_123")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/knowledge/facts/f_123/relations",
            params={"as_source": True, "as_target": True},
        )

    def test_list_relations_filtered(self, client: AragoraClient, mock_request) -> None:
        """List relations filtered by type and direction."""
        mock_request.return_value = {"relations": []}

        client.knowledge.list_relations(
            "f_123", relation_type="supports", as_source=True, as_target=False
        )

        call_params = mock_request.call_args[1]["params"]
        assert call_params["type"] == "supports"
        assert call_params["as_source"] is True
        assert call_params["as_target"] is False

    def test_add_relation(self, client: AragoraClient, mock_request) -> None:
        """Add a relation from one fact to another."""
        mock_request.return_value = {"relation_id": "rel_1"}

        client.knowledge.add_relation("f_1", "f_2", "supports")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/knowledge/facts/f_1/relations",
            json={"target_fact_id": "f_2", "relation_type": "supports"},
        )

    def test_add_relation_between_facts(self, client: AragoraClient, mock_request) -> None:
        """Add a relation between two facts with confidence."""
        mock_request.return_value = {"relation_id": "rel_2"}

        client.knowledge.add_relation_between_facts("f_1", "f_2", "contradicts", confidence=0.85)

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/knowledge/facts/relations",
            json={
                "source_fact_id": "f_1",
                "target_fact_id": "f_2",
                "relation_type": "contradicts",
                "confidence": 0.85,
            },
        )


class TestKnowledgeStats:
    """Tests for knowledge base statistics."""

    def test_get_stats(self, client: AragoraClient, mock_request) -> None:
        """Get knowledge base statistics."""
        mock_request.return_value = {"total_facts": 1500, "verified": 1200}

        result = client.knowledge.get_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/knowledge/stats", params={})
        assert result["total_facts"] == 1500

    def test_get_stats_with_workspace(self, client: AragoraClient, mock_request) -> None:
        """Get knowledge base statistics for a specific workspace."""
        mock_request.return_value = {"total_facts": 300}

        client.knowledge.get_stats(workspace_id="ws_1")

        call_params = mock_request.call_args[1]["params"]
        assert call_params["workspace_id"] == "ws_1"

    def test_get_mound_stats(self, client: AragoraClient, mock_request) -> None:
        """Get Knowledge Mound statistics."""
        mock_request.return_value = {"nodes": 500, "edges": 1200}

        result = client.knowledge.get_mound_stats()

        mock_request.assert_called_once_with("GET", "/api/v1/knowledge/mound/stats", params={})
        assert result["nodes"] == 500


class TestAsyncKnowledge:
    """Tests for async knowledge API."""

    @pytest.mark.asyncio
    async def test_async_search(self, mock_async_request) -> None:
        """Search knowledge asynchronously."""
        mock_async_request.return_value = {"results": [], "total": 0}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge.search("test query")

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/knowledge/search",
                params={"q": "test query", "limit": 10},
            )
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_async_create_fact(self, mock_async_request) -> None:
        """Create a fact asynchronously."""
        mock_async_request.return_value = {"fact_id": "f_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge.create_fact("Async facts work")

            assert result["fact_id"] == "f_async"

    @pytest.mark.asyncio
    async def test_async_get_fact(self, mock_async_request) -> None:
        """Get a fact asynchronously."""
        mock_async_request.return_value = {"fact_id": "f_123", "confidence": 0.95}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge.get_fact("f_123")

            mock_async_request.assert_called_once_with("GET", "/api/v1/knowledge/facts/f_123")
            assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_async_delete_fact(self, mock_async_request) -> None:
        """Delete a fact asynchronously."""
        mock_async_request.return_value = {"deleted": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.knowledge.delete_fact("f_123")

            mock_async_request.assert_called_once_with("DELETE", "/api/v1/knowledge/facts/f_123")
            assert result["deleted"] is True

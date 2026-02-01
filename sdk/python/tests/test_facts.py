"""Tests for Facts SDK namespace."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestFactsAPI:
    """Test synchronous FactsAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.facts import FactsAPI

        api = FactsAPI(mock_client)
        assert api._client is mock_client

    # ===========================================================================
    # Fact CRUD Operations
    # ===========================================================================

    def test_create_fact(self, mock_client: MagicMock) -> None:
        """Test create_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "id": "fact-123",
            "content": "Python is a programming language",
            "source": "docs",
            "confidence": 0.95,
            "created_at": "2024-01-01T00:00:00Z",
        }

        api = FactsAPI(mock_client)
        result = api.create_fact(
            content="Python is a programming language",
            source="docs",
            confidence=0.95,
        )

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts",
            json={
                "content": "Python is a programming language",
                "source": "docs",
                "confidence": 0.95,
            },
        )
        assert result["id"] == "fact-123"
        assert result["content"] == "Python is a programming language"

    def test_create_fact_with_tags_and_metadata(self, mock_client: MagicMock) -> None:
        """Test create_fact with optional tags and metadata."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"id": "fact-123"}

        api = FactsAPI(mock_client)
        api.create_fact(
            content="Test fact",
            source="test-source",
            confidence=0.8,
            tags=["python", "programming"],
            metadata={"author": "test", "version": 1},
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["content"] == "Test fact"
        assert json_body["source"] == "test-source"
        assert json_body["confidence"] == 0.8
        assert json_body["tags"] == ["python", "programming"]
        assert json_body["metadata"] == {"author": "test", "version": 1}

    def test_create_fact_default_confidence(self, mock_client: MagicMock) -> None:
        """Test create_fact uses default confidence of 1.0."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"id": "fact-123"}

        api = FactsAPI(mock_client)
        api.create_fact(content="Test", source="test")

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["confidence"] == 1.0

    def test_get_fact(self, mock_client: MagicMock) -> None:
        """Test get_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "id": "fact-123",
            "content": "Test fact",
            "source": "docs",
            "confidence": 0.95,
        }

        api = FactsAPI(mock_client)
        result = api.get_fact("fact-123")

        mock_client.request.assert_called_once_with("GET", "/api/v1/facts/fact-123")
        assert result["id"] == "fact-123"
        assert result["confidence"] == 0.95

    def test_update_fact(self, mock_client: MagicMock) -> None:
        """Test update_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "id": "fact-123",
            "content": "Updated content",
            "confidence": 0.99,
        }

        api = FactsAPI(mock_client)
        result = api.update_fact(
            fact_id="fact-123",
            updates={"content": "Updated content", "confidence": 0.99},
        )

        mock_client.request.assert_called_once_with(
            "PATCH",
            "/api/v1/facts/fact-123",
            json={"content": "Updated content", "confidence": 0.99},
        )
        assert result["content"] == "Updated content"
        assert result["confidence"] == 0.99

    def test_delete_fact(self, mock_client: MagicMock) -> None:
        """Test delete_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"deleted": True, "id": "fact-123"}

        api = FactsAPI(mock_client)
        result = api.delete_fact("fact-123")

        mock_client.request.assert_called_once_with("DELETE", "/api/v1/facts/fact-123")
        assert result["deleted"] is True

    # ===========================================================================
    # Search and List Operations
    # ===========================================================================

    def test_search_facts(self, mock_client: MagicMock) -> None:
        """Test search_facts calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "results": [
                {"id": "fact-1", "content": "Test", "relevance": 0.95},
                {"id": "fact-2", "content": "Test 2", "relevance": 0.85},
            ],
            "total": 2,
        }

        api = FactsAPI(mock_client)
        result = api.search_facts(query="programming language")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/search",
            json={"query": "programming language", "limit": 20, "offset": 0},
        )
        assert len(result["results"]) == 2
        assert result["total"] == 2

    def test_search_facts_with_filters(self, mock_client: MagicMock) -> None:
        """Test search_facts with filters and pagination."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"results": [], "total": 0}

        api = FactsAPI(mock_client)
        api.search_facts(
            query="test",
            filters={"semantic": True, "source": "docs", "min_relevance": 0.5},
            limit=50,
            offset=10,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["query"] == "test"
        assert json_body["filters"] == {
            "semantic": True,
            "source": "docs",
            "min_relevance": 0.5,
        }
        assert json_body["limit"] == 50
        assert json_body["offset"] == 10

    def test_list_facts(self, mock_client: MagicMock) -> None:
        """Test list_facts calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "facts": [{"id": "fact-1"}, {"id": "fact-2"}],
            "total": 100,
            "limit": 20,
            "offset": 0,
        }

        api = FactsAPI(mock_client)
        result = api.list_facts()

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/facts", params={"limit": 20, "offset": 0}
        )
        assert len(result["facts"]) == 2
        assert result["total"] == 100

    def test_list_facts_with_all_filters(self, mock_client: MagicMock) -> None:
        """Test list_facts with all optional filters."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"facts": [], "total": 0}

        api = FactsAPI(mock_client)
        api.list_facts(
            limit=50,
            offset=25,
            tags=["python", "api"],
            source="official-docs",
            min_confidence=0.7,
            max_confidence=1.0,
            sort_by="created_at",
            sort_order="desc",
        )

        call_args = mock_client.request.call_args
        params = call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 25
        assert params["tags"] == ["python", "api"]
        assert params["source"] == "official-docs"
        assert params["min_confidence"] == 0.7
        assert params["max_confidence"] == 1.0
        assert params["sort_by"] == "created_at"
        assert params["sort_order"] == "desc"

    # ===========================================================================
    # Relationship Operations
    # ===========================================================================

    def test_create_relationship(self, mock_client: MagicMock) -> None:
        """Test create_relationship calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "id": "rel-123",
            "source_fact_id": "fact-1",
            "target_fact_id": "fact-2",
            "relationship_type": "supports",
            "weight": 0.9,
        }

        api = FactsAPI(mock_client)
        result = api.create_relationship(
            source_id="fact-1",
            target_id="fact-2",
            relationship_type="supports",
            weight=0.9,
        )

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/relationships",
            json={
                "source_fact_id": "fact-1",
                "target_fact_id": "fact-2",
                "relationship_type": "supports",
                "weight": 0.9,
            },
        )
        assert result["id"] == "rel-123"
        assert result["relationship_type"] == "supports"

    def test_create_relationship_with_metadata(self, mock_client: MagicMock) -> None:
        """Test create_relationship with metadata."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"id": "rel-123"}

        api = FactsAPI(mock_client)
        api.create_relationship(
            source_id="fact-1",
            target_id="fact-2",
            relationship_type="contradicts",
            weight=0.8,
            metadata={"reason": "conflicting sources", "verified": True},
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["metadata"] == {"reason": "conflicting sources", "verified": True}

    def test_create_relationship_default_weight(self, mock_client: MagicMock) -> None:
        """Test create_relationship uses default weight of 1.0."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"id": "rel-123"}

        api = FactsAPI(mock_client)
        api.create_relationship(
            source_id="fact-1", target_id="fact-2", relationship_type="related_to"
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["weight"] == 1.0

    def test_get_relationships(self, mock_client: MagicMock) -> None:
        """Test get_relationships calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "relationships": [
                {"id": "rel-1", "relationship_type": "supports"},
                {"id": "rel-2", "relationship_type": "elaborates"},
            ]
        }

        api = FactsAPI(mock_client)
        result = api.get_relationships("fact-123")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/facts/fact-123/relationships", params={}
        )
        assert len(result["relationships"]) == 2

    def test_get_relationships_with_filters(self, mock_client: MagicMock) -> None:
        """Test get_relationships with all filters."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {"relationships": []}

        api = FactsAPI(mock_client)
        api.get_relationships(
            fact_id="fact-123",
            relationship_type="supports",
            direction="outgoing",
            min_weight=0.5,
        )

        call_args = mock_client.request.call_args
        params = call_args[1]["params"]
        assert params["relationship_type"] == "supports"
        assert params["direction"] == "outgoing"
        assert params["min_weight"] == 0.5

    # ===========================================================================
    # Batch Operations
    # ===========================================================================

    def test_batch_create_facts(self, mock_client: MagicMock) -> None:
        """Test batch_create_facts calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "created": [{"id": "fact-1"}, {"id": "fact-2"}],
            "failed": [],
            "total_created": 2,
            "total_failed": 0,
        }

        facts = [
            {"content": "Fact 1", "source": "source1", "confidence": 0.9},
            {"content": "Fact 2", "source": "source2", "confidence": 0.8},
        ]

        api = FactsAPI(mock_client)
        result = api.batch_create_facts(facts)

        mock_client.request.assert_called_once_with(
            "POST", "/api/v1/facts/batch", json={"facts": facts}
        )
        assert result["total_created"] == 2
        assert result["total_failed"] == 0

    def test_batch_create_facts_partial_failure(self, mock_client: MagicMock) -> None:
        """Test batch_create_facts handles partial failures."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "created": [{"id": "fact-1"}],
            "failed": [{"content": "Invalid fact", "error": "Missing source"}],
            "total_created": 1,
            "total_failed": 1,
        }

        api = FactsAPI(mock_client)
        result = api.batch_create_facts(
            [
                {"content": "Valid", "source": "src"},
                {"content": "Invalid fact"},  # Missing source
            ]
        )

        assert result["total_created"] == 1
        assert result["total_failed"] == 1
        assert len(result["failed"]) == 1

    # ===========================================================================
    # Verification Operations
    # ===========================================================================

    def test_verify_fact(self, mock_client: MagicMock) -> None:
        """Test verify_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "verified": True,
            "fact": {"id": "fact-123", "status": "verified"},
        }

        api = FactsAPI(mock_client)
        result = api.verify_fact("fact-123")

        mock_client.request.assert_called_once_with("POST", "/api/v1/facts/fact-123/verify")
        assert result["verified"] is True
        assert result["fact"]["status"] == "verified"

    def test_invalidate_fact(self, mock_client: MagicMock) -> None:
        """Test invalidate_fact calls correct endpoint."""
        from aragora.namespaces.facts import FactsAPI

        mock_client.request.return_value = {
            "invalidated": True,
            "fact": {"id": "fact-123", "status": "invalidated"},
            "reason": "Outdated information",
        }

        api = FactsAPI(mock_client)
        result = api.invalidate_fact("fact-123", reason="Outdated information")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/fact-123/invalidate",
            json={"reason": "Outdated information"},
        )
        assert result["invalidated"] is True
        assert result["reason"] == "Outdated information"


class TestAsyncFactsAPI:
    """Test asynchronous AsyncFactsAPI."""

    def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.facts import AsyncFactsAPI

        api = AsyncFactsAPI(mock_async_client)
        assert api._client is mock_async_client

    # ===========================================================================
    # Fact CRUD Operations
    # ===========================================================================

    @pytest.mark.asyncio
    async def test_create_fact(self, mock_async_client: MagicMock) -> None:
        """Test create_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "id": "fact-123",
            "content": "Python is a programming language",
            "source": "docs",
            "confidence": 0.95,
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.create_fact(
            content="Python is a programming language",
            source="docs",
            confidence=0.95,
        )

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts",
            json={
                "content": "Python is a programming language",
                "source": "docs",
                "confidence": 0.95,
            },
        )
        assert result["id"] == "fact-123"

    @pytest.mark.asyncio
    async def test_create_fact_with_tags_and_metadata(self, mock_async_client: MagicMock) -> None:
        """Test create_fact with optional tags and metadata."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"id": "fact-123"}

        api = AsyncFactsAPI(mock_async_client)
        await api.create_fact(
            content="Test fact",
            source="test-source",
            tags=["python"],
            metadata={"key": "value"},
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["tags"] == ["python"]
        assert json_body["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_fact(self, mock_async_client: MagicMock) -> None:
        """Test get_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "id": "fact-123",
            "content": "Test fact",
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.get_fact("fact-123")

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/facts/fact-123")
        assert result["id"] == "fact-123"

    @pytest.mark.asyncio
    async def test_update_fact(self, mock_async_client: MagicMock) -> None:
        """Test update_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "id": "fact-123",
            "content": "Updated",
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.update_fact(fact_id="fact-123", updates={"content": "Updated"})

        mock_async_client.request.assert_called_once_with(
            "PATCH", "/api/v1/facts/fact-123", json={"content": "Updated"}
        )
        assert result["content"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_fact(self, mock_async_client: MagicMock) -> None:
        """Test delete_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"deleted": True}

        api = AsyncFactsAPI(mock_async_client)
        result = await api.delete_fact("fact-123")

        mock_async_client.request.assert_called_once_with("DELETE", "/api/v1/facts/fact-123")
        assert result["deleted"] is True

    # ===========================================================================
    # Search and List Operations
    # ===========================================================================

    @pytest.mark.asyncio
    async def test_search_facts(self, mock_async_client: MagicMock) -> None:
        """Test search_facts calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "results": [{"id": "fact-1", "relevance": 0.9}],
            "total": 1,
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.search_facts(query="test query")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/search",
            json={"query": "test query", "limit": 20, "offset": 0},
        )
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_search_facts_with_filters(self, mock_async_client: MagicMock) -> None:
        """Test search_facts with filters."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"results": [], "total": 0}

        api = AsyncFactsAPI(mock_async_client)
        await api.search_facts(
            query="test",
            filters={"semantic": True},
            limit=10,
            offset=5,
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["filters"] == {"semantic": True}
        assert json_body["limit"] == 10
        assert json_body["offset"] == 5

    @pytest.mark.asyncio
    async def test_list_facts(self, mock_async_client: MagicMock) -> None:
        """Test list_facts calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "facts": [{"id": "fact-1"}],
            "total": 50,
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.list_facts()

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/facts", params={"limit": 20, "offset": 0}
        )
        assert result["total"] == 50

    @pytest.mark.asyncio
    async def test_list_facts_with_filters(self, mock_async_client: MagicMock) -> None:
        """Test list_facts with all filters."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"facts": [], "total": 0}

        api = AsyncFactsAPI(mock_async_client)
        await api.list_facts(
            limit=100,
            offset=50,
            tags=["tag1"],
            source="source1",
            min_confidence=0.5,
            max_confidence=0.9,
            sort_by="confidence",
            sort_order="asc",
        )

        call_args = mock_async_client.request.call_args
        params = call_args[1]["params"]
        assert params["limit"] == 100
        assert params["offset"] == 50
        assert params["tags"] == ["tag1"]
        assert params["source"] == "source1"
        assert params["min_confidence"] == 0.5
        assert params["max_confidence"] == 0.9
        assert params["sort_by"] == "confidence"
        assert params["sort_order"] == "asc"

    # ===========================================================================
    # Relationship Operations
    # ===========================================================================

    @pytest.mark.asyncio
    async def test_create_relationship(self, mock_async_client: MagicMock) -> None:
        """Test create_relationship calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "id": "rel-123",
            "relationship_type": "supports",
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.create_relationship(
            source_id="fact-1",
            target_id="fact-2",
            relationship_type="supports",
            weight=0.8,
        )

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/relationships",
            json={
                "source_fact_id": "fact-1",
                "target_fact_id": "fact-2",
                "relationship_type": "supports",
                "weight": 0.8,
            },
        )
        assert result["id"] == "rel-123"

    @pytest.mark.asyncio
    async def test_create_relationship_with_metadata(self, mock_async_client: MagicMock) -> None:
        """Test create_relationship with metadata."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"id": "rel-123"}

        api = AsyncFactsAPI(mock_async_client)
        await api.create_relationship(
            source_id="fact-1",
            target_id="fact-2",
            relationship_type="elaborates",
            metadata={"detail": "provides context"},
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["metadata"] == {"detail": "provides context"}

    @pytest.mark.asyncio
    async def test_get_relationships(self, mock_async_client: MagicMock) -> None:
        """Test get_relationships calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "relationships": [{"id": "rel-1"}, {"id": "rel-2"}]
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.get_relationships("fact-123")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/facts/fact-123/relationships", params={}
        )
        assert len(result["relationships"]) == 2

    @pytest.mark.asyncio
    async def test_get_relationships_with_filters(self, mock_async_client: MagicMock) -> None:
        """Test get_relationships with filters."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {"relationships": []}

        api = AsyncFactsAPI(mock_async_client)
        await api.get_relationships(
            fact_id="fact-123",
            relationship_type="contradicts",
            direction="incoming",
            min_weight=0.7,
        )

        call_args = mock_async_client.request.call_args
        params = call_args[1]["params"]
        assert params["relationship_type"] == "contradicts"
        assert params["direction"] == "incoming"
        assert params["min_weight"] == 0.7

    # ===========================================================================
    # Batch Operations
    # ===========================================================================

    @pytest.mark.asyncio
    async def test_batch_create_facts(self, mock_async_client: MagicMock) -> None:
        """Test batch_create_facts calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "created": [{"id": "fact-1"}, {"id": "fact-2"}, {"id": "fact-3"}],
            "failed": [],
            "total_created": 3,
            "total_failed": 0,
        }

        facts = [
            {"content": "Fact 1", "source": "s1"},
            {"content": "Fact 2", "source": "s2"},
            {"content": "Fact 3", "source": "s3"},
        ]

        api = AsyncFactsAPI(mock_async_client)
        result = await api.batch_create_facts(facts)

        mock_async_client.request.assert_called_once_with(
            "POST", "/api/v1/facts/batch", json={"facts": facts}
        )
        assert result["total_created"] == 3

    # ===========================================================================
    # Verification Operations
    # ===========================================================================

    @pytest.mark.asyncio
    async def test_verify_fact(self, mock_async_client: MagicMock) -> None:
        """Test verify_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "verified": True,
            "fact": {"id": "fact-123", "status": "verified"},
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.verify_fact("fact-123")

        mock_async_client.request.assert_called_once_with("POST", "/api/v1/facts/fact-123/verify")
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_invalidate_fact(self, mock_async_client: MagicMock) -> None:
        """Test invalidate_fact calls correct endpoint."""
        from aragora.namespaces.facts import AsyncFactsAPI

        mock_async_client.request.return_value = {
            "invalidated": True,
            "reason": "No longer accurate",
        }

        api = AsyncFactsAPI(mock_async_client)
        result = await api.invalidate_fact("fact-123", reason="No longer accurate")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v1/facts/fact-123/invalidate",
            json={"reason": "No longer accurate"},
        )
        assert result["invalidated"] is True
        assert result["reason"] == "No longer accurate"

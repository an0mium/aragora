"""Tests for Aragora SDK Knowledge API."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_client.knowledge import (
    Fact,
    KnowledgeAPI,
    KnowledgeEntry,
    KnowledgeSearchResult,
    KnowledgeStats,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock()
    client._get = AsyncMock()
    client._post = AsyncMock()
    client._put = AsyncMock()
    client._delete = AsyncMock()
    return client


@pytest.fixture
def knowledge_api(mock_client: MagicMock) -> KnowledgeAPI:
    """Create KnowledgeAPI with mock client."""
    return KnowledgeAPI(mock_client)


@pytest.fixture
def entry_response() -> dict[str, Any]:
    """Standard knowledge entry response."""
    return {
        "id": "entry-123",
        "content": "The capital of France is Paris.",
        "source": "wikipedia",
        "source_type": "encyclopedia",
        "metadata": {"language": "en"},
        "confidence": 0.95,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "tags": ["geography", "cities"],
    }


@pytest.fixture
def search_result_response() -> dict[str, Any]:
    """Standard search result response."""
    return {
        "id": "entry-123",
        "content": "The capital of France is Paris.",
        "score": 0.92,
        "source": "wikipedia",
        "metadata": {"language": "en"},
    }


@pytest.fixture
def fact_response() -> dict[str, Any]:
    """Standard fact response."""
    return {
        "id": "fact-123",
        "content": "Water boils at 100 degrees Celsius at sea level.",
        "source": "physics_textbook",
        "confidence": 0.99,
        "verified": True,
        "metadata": {"subject": "physics"},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def stats_response() -> dict[str, Any]:
    """Standard knowledge stats response."""
    return {
        "total_entries": 10000,
        "total_facts": 5000,
        "sources": {"wikipedia": 3000, "textbooks": 2000},
        "categories": {"science": 4000, "history": 3000},
        "avg_confidence": 0.85,
        "last_updated": "2026-01-01T12:00:00Z",
    }


# =============================================================================
# Model Tests
# =============================================================================


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry model."""

    def test_minimal_creation(self) -> None:
        """Test creating entry with minimal fields."""
        entry = KnowledgeEntry(content="Test content")
        assert entry.content == "Test content"
        assert entry.id is None
        assert entry.source is None

    def test_full_creation(self, entry_response: dict[str, Any]) -> None:
        """Test creating entry with all fields."""
        entry = KnowledgeEntry.model_validate(entry_response)
        assert entry.id == "entry-123"
        assert entry.content == "The capital of France is Paris."
        assert entry.confidence == 0.95
        assert "geography" in entry.tags


class TestKnowledgeSearchResult:
    """Tests for KnowledgeSearchResult model."""

    def test_creation(self, search_result_response: dict[str, Any]) -> None:
        """Test creating search result."""
        result = KnowledgeSearchResult.model_validate(search_result_response)
        assert result.id == "entry-123"
        assert result.score == 0.92
        assert result.source == "wikipedia"


class TestKnowledgeStats:
    """Tests for KnowledgeStats model."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = KnowledgeStats()
        assert stats.total_entries == 0
        assert stats.total_facts == 0

    def test_full_creation(self, stats_response: dict[str, Any]) -> None:
        """Test creating stats with all fields."""
        stats = KnowledgeStats.model_validate(stats_response)
        assert stats.total_entries == 10000
        assert stats.sources["wikipedia"] == 3000


class TestFact:
    """Tests for Fact model."""

    def test_minimal_creation(self) -> None:
        """Test creating fact with minimal fields."""
        fact = Fact(id="f-1", content="A fact")
        assert fact.id == "f-1"
        assert fact.verified is False

    def test_full_creation(self, fact_response: dict[str, Any]) -> None:
        """Test creating fact with all fields."""
        fact = Fact.model_validate(fact_response)
        assert fact.id == "fact-123"
        assert fact.verified is True
        assert fact.confidence == 0.99


# =============================================================================
# KnowledgeAPI Tests - Search and Query
# =============================================================================


class TestKnowledgeAPISearch:
    """Tests for KnowledgeAPI.search()."""

    @pytest.mark.asyncio
    async def test_search_basic(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        search_result_response: dict[str, Any],
    ) -> None:
        """Test basic search."""
        mock_client._get.return_value = {"results": [search_result_response]}

        results = await knowledge_api.search("capital of France")

        mock_client._get.assert_called_once_with(
            "/api/v1/knowledge/search",
            params={"query": "capital of France", "limit": 10},
        )
        assert len(results) == 1
        assert isinstance(results[0], KnowledgeSearchResult)

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test search with filters."""
        mock_client._get.return_value = {"results": []}

        await knowledge_api.search(
            "test query",
            limit=20,
            min_score=0.8,
            source_filter="wikipedia",
            tags=["science", "physics"],
        )

        mock_client._get.assert_called_once_with(
            "/api/v1/knowledge/search",
            params={
                "query": "test query",
                "limit": 20,
                "min_score": 0.8,
                "source": "wikipedia",
                "tags": "science,physics",
            },
        )

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test search with no results."""
        mock_client._get.return_value = {}

        results = await knowledge_api.search("nonexistent query")

        assert results == []


class TestKnowledgeAPIQuery:
    """Tests for KnowledgeAPI.query()."""

    @pytest.mark.asyncio
    async def test_query_basic(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test basic natural language query."""
        mock_client._post.return_value = {
            "answer": "Paris",
            "sources": ["wikipedia"],
        }

        result = await knowledge_api.query("What is the capital of France?")

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/query",
            {"question": "What is the capital of France?", "include_sources": True},
        )
        assert result["answer"] == "Paris"

    @pytest.mark.asyncio
    async def test_query_with_context(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test query with additional context."""
        mock_client._post.return_value = {"answer": "Paris"}

        await knowledge_api.query(
            "What is the capital?",
            context="We are discussing European countries",
            include_sources=False,
        )

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/query",
            {
                "question": "What is the capital?",
                "include_sources": False,
                "context": "We are discussing European countries",
            },
        )


# =============================================================================
# KnowledgeAPI Tests - CRUD
# =============================================================================


class TestKnowledgeAPIAdd:
    """Tests for KnowledgeAPI.add()."""

    @pytest.mark.asyncio
    async def test_add_minimal(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test adding entry with minimal fields."""
        mock_client._post.return_value = {"id": "entry-new", "created_at": "2026-01-01"}

        result = await knowledge_api.add("New knowledge content")

        call_body = mock_client._post.call_args[0][1]
        assert call_body["content"] == "New knowledge content"
        assert result["id"] == "entry-new"

    @pytest.mark.asyncio
    async def test_add_full(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test adding entry with all fields."""
        mock_client._post.return_value = {"id": "entry-full"}

        await knowledge_api.add(
            "Full content",
            source="manual_entry",
            source_type="manual",
            metadata={"author": "user123"},
            tags=["test", "example"],
            confidence=0.9,
        )

        call_body = mock_client._post.call_args[0][1]
        assert call_body["content"] == "Full content"
        assert call_body["source"] == "manual_entry"
        assert call_body["source_type"] == "manual"
        assert call_body["tags"] == ["test", "example"]


class TestKnowledgeAPIGet:
    """Tests for KnowledgeAPI.get()."""

    @pytest.mark.asyncio
    async def test_get_entry(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        entry_response: dict[str, Any],
    ) -> None:
        """Test getting an entry by ID."""
        mock_client._get.return_value = entry_response

        result = await knowledge_api.get("entry-123")

        mock_client._get.assert_called_once_with("/api/v1/knowledge/entry-123")
        assert isinstance(result, KnowledgeEntry)
        assert result.id == "entry-123"


class TestKnowledgeAPIUpdate:
    """Tests for KnowledgeAPI.update()."""

    @pytest.mark.asyncio
    async def test_update_content(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        entry_response: dict[str, Any],
    ) -> None:
        """Test updating entry content."""
        mock_client._put.return_value = entry_response

        result = await knowledge_api.update("entry-123", content="Updated content")

        mock_client._put.assert_called_once_with(
            "/api/v1/knowledge/entry-123",
            {"content": "Updated content"},
        )
        assert isinstance(result, KnowledgeEntry)

    @pytest.mark.asyncio
    async def test_update_multiple_fields(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        entry_response: dict[str, Any],
    ) -> None:
        """Test updating multiple fields."""
        mock_client._put.return_value = entry_response

        await knowledge_api.update(
            "entry-123",
            content="New content",
            metadata={"updated": True},
            tags=["new-tag"],
            confidence=0.99,
        )

        call_body = mock_client._put.call_args[0][1]
        assert call_body["content"] == "New content"
        assert call_body["metadata"] == {"updated": True}
        assert call_body["tags"] == ["new-tag"]
        assert call_body["confidence"] == 0.99


class TestKnowledgeAPIDelete:
    """Tests for KnowledgeAPI.delete()."""

    @pytest.mark.asyncio
    async def test_delete_entry(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test deleting an entry."""
        mock_client._delete.return_value = None

        result = await knowledge_api.delete("entry-123")

        mock_client._delete.assert_called_once_with("/api/v1/knowledge/entry-123")
        assert result == {"deleted": True}


# =============================================================================
# KnowledgeAPI Tests - Facts
# =============================================================================


class TestKnowledgeAPIListFacts:
    """Tests for KnowledgeAPI.list_facts()."""

    @pytest.mark.asyncio
    async def test_list_facts_default(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        fact_response: dict[str, Any],
    ) -> None:
        """Test listing facts with defaults."""
        mock_client._get.return_value = {"facts": [fact_response]}

        results = await knowledge_api.list_facts()

        mock_client._get.assert_called_once_with(
            "/api/v1/knowledge/facts",
            params={"limit": 50, "offset": 0},
        )
        assert len(results) == 1
        assert isinstance(results[0], Fact)

    @pytest.mark.asyncio
    async def test_list_facts_filtered(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing facts with filters."""
        mock_client._get.return_value = {"facts": []}

        await knowledge_api.list_facts(
            limit=10,
            offset=5,
            verified=True,
            source="textbook",
        )

        mock_client._get.assert_called_once_with(
            "/api/v1/knowledge/facts",
            params={"limit": 10, "offset": 5, "verified": True, "source": "textbook"},
        )


class TestKnowledgeAPIGetFact:
    """Tests for KnowledgeAPI.get_fact()."""

    @pytest.mark.asyncio
    async def test_get_fact(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        fact_response: dict[str, Any],
    ) -> None:
        """Test getting a fact by ID."""
        mock_client._get.return_value = fact_response

        result = await knowledge_api.get_fact("fact-123")

        mock_client._get.assert_called_once_with("/api/v1/knowledge/facts/fact-123")
        assert isinstance(result, Fact)
        assert result.verified is True


class TestKnowledgeAPIAddFact:
    """Tests for KnowledgeAPI.add_fact()."""

    @pytest.mark.asyncio
    async def test_add_fact_minimal(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        fact_response: dict[str, Any],
    ) -> None:
        """Test adding fact with minimal fields."""
        mock_client._post.return_value = fact_response

        result = await knowledge_api.add_fact("A new fact")

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/facts",
            {"content": "A new fact"},
        )
        assert isinstance(result, Fact)

    @pytest.mark.asyncio
    async def test_add_fact_full(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        fact_response: dict[str, Any],
    ) -> None:
        """Test adding fact with all fields."""
        mock_client._post.return_value = fact_response

        await knowledge_api.add_fact(
            "Comprehensive fact",
            source="research_paper",
            confidence=0.95,
            metadata={"doi": "10.1234/test"},
        )

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/facts",
            {
                "content": "Comprehensive fact",
                "source": "research_paper",
                "confidence": 0.95,
                "metadata": {"doi": "10.1234/test"},
            },
        )


class TestKnowledgeAPIVerifyFact:
    """Tests for KnowledgeAPI.verify_fact()."""

    @pytest.mark.asyncio
    async def test_verify_fact_default(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test verifying fact with default agents."""
        mock_client._post.return_value = {"verified": True, "confidence": 0.95}

        result = await knowledge_api.verify_fact("fact-123")

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/facts/fact-123/verify",
            {},
        )
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_verify_fact_with_agents(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test verifying fact with specific agents."""
        mock_client._post.return_value = {"verified": True}

        await knowledge_api.verify_fact("fact-123", agents=["claude", "gpt-4"])

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/facts/fact-123/verify",
            {"agents": ["claude", "gpt-4"]},
        )


class TestKnowledgeAPIGetContradictions:
    """Tests for KnowledgeAPI.get_contradictions()."""

    @pytest.mark.asyncio
    async def test_get_contradictions(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        fact_response: dict[str, Any],
    ) -> None:
        """Test getting contradicting facts."""
        contradiction = {
            **fact_response,
            "id": "fact-456",
            "content": "Contradicting fact",
        }
        mock_client._get.return_value = {"contradictions": [contradiction]}

        results = await knowledge_api.get_contradictions("fact-123")

        mock_client._get.assert_called_once_with(
            "/api/v1/knowledge/facts/fact-123/contradictions"
        )
        assert len(results) == 1
        assert results[0].id == "fact-456"


# =============================================================================
# KnowledgeAPI Tests - Statistics
# =============================================================================


class TestKnowledgeAPIGetStats:
    """Tests for KnowledgeAPI.get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
        stats_response: dict[str, Any],
    ) -> None:
        """Test getting knowledge base statistics."""
        mock_client._get.return_value = stats_response

        result = await knowledge_api.get_stats()

        mock_client._get.assert_called_once_with("/api/v1/knowledge/stats")
        assert isinstance(result, KnowledgeStats)
        assert result.total_entries == 10000
        assert result.avg_confidence == 0.85


# =============================================================================
# KnowledgeAPI Tests - Bulk Operations
# =============================================================================


class TestKnowledgeAPIBulkImport:
    """Tests for KnowledgeAPI.bulk_import()."""

    @pytest.mark.asyncio
    async def test_bulk_import_default(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test bulk import with defaults."""
        mock_client._post.return_value = {"imported": 10, "skipped": 2}
        entries = [
            {"content": "Fact 1"},
            {"content": "Fact 2"},
        ]

        result = await knowledge_api.bulk_import(entries)

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/bulk-import",
            {"entries": entries, "skip_duplicates": True},
        )
        assert result["imported"] == 10

    @pytest.mark.asyncio
    async def test_bulk_import_no_skip(
        self,
        knowledge_api: KnowledgeAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test bulk import without skipping duplicates."""
        mock_client._post.return_value = {"imported": 5, "errors": 0}

        await knowledge_api.bulk_import(
            [{"content": "Entry"}],
            skip_duplicates=False,
        )

        mock_client._post.assert_called_once_with(
            "/api/v1/knowledge/bulk-import",
            {"entries": [{"content": "Entry"}], "skip_duplicates": False},
        )

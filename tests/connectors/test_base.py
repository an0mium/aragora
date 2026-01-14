"""Tests for base connector functionality."""

import pytest
import time
from typing import Optional

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType


class MockConnector(BaseConnector):
    """Mock connector for testing base functionality."""

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def name(self) -> str:
        return "MockConnector"

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[Evidence]:
        return [
            Evidence(
                id=f"mock:{i}",
                source_type=self.source_type,
                source_id=f"result-{i}",
                content=f"Mock content for {query}",
                title=f"Mock Result {i}",
            )
            for i in range(min(limit, 5))
        ]

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        if evidence_id.startswith("mock:"):
            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=evidence_id,
                content="Mock fetched content",
                title="Fetched Result",
            )
        return None


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self):
        """Test basic evidence creation."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.DOCUMENT,
            source_id="doc-123",
            content="Test content",
            title="Test Title",
        )

        assert evidence.id == "test-1"
        assert evidence.source_type == SourceType.DOCUMENT
        assert evidence.content == "Test content"
        assert evidence.title == "Test Title"

    def test_evidence_defaults(self):
        """Test evidence default values."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="web-123",
            content="Content",
        )

        assert evidence.confidence == 0.5
        assert evidence.freshness == 1.0
        assert evidence.authority == 0.5
        assert evidence.title == ""
        assert evidence.metadata == {}

    def test_content_hash(self):
        """Test content hash generation."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="web-123",
            content="Test content for hashing",
        )

        hash_val = evidence.content_hash
        assert len(hash_val) == 16
        assert hash_val.isalnum()

        # Same content should produce same hash
        evidence2 = Evidence(
            id="test-2",
            source_type=SourceType.WEB_SEARCH,
            source_id="web-456",
            content="Test content for hashing",
        )
        assert evidence.content_hash == evidence2.content_hash

    def test_reliability_score(self):
        """Test reliability score calculation."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.DOCUMENT,
            source_id="doc-123",
            content="Content",
            confidence=0.8,
            freshness=0.9,
            authority=0.7,
        )

        # Expected: 0.4 * 0.8 + 0.3 * 0.9 + 0.3 * 0.7 = 0.32 + 0.27 + 0.21 = 0.80
        assert abs(evidence.reliability_score - 0.80) < 0.001

    def test_to_dict(self):
        """Test serialization to dictionary."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="api-123",
            content="API content",
            title="API Result",
            author="API Author",
            url="https://example.com",
            confidence=0.9,
            freshness=0.8,
            authority=0.85,
            metadata={"key": "value"},
        )

        data = evidence.to_dict()

        assert data["id"] == "test-1"
        assert data["source_type"] == "external_api"
        assert data["content"] == "API content"
        assert data["title"] == "API Result"
        assert data["author"] == "API Author"
        assert data["url"] == "https://example.com"
        assert data["confidence"] == 0.9
        assert data["freshness"] == 0.8
        assert data["authority"] == 0.85
        assert "reliability_score" in data
        assert "content_hash" in data
        assert data["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-1",
            "source_type": "document",
            "source_id": "doc-123",
            "content": "Reconstructed content",
            "title": "Reconstructed Title",
            "confidence": 0.75,
            "freshness": 0.85,
            "authority": 0.65,
        }

        evidence = Evidence.from_dict(data)

        assert evidence.id == "test-1"
        assert evidence.source_type == SourceType.DOCUMENT
        assert evidence.content == "Reconstructed content"
        assert evidence.confidence == 0.75

    def test_from_dict_unknown_source_type(self):
        """Test deserialization with unknown source type."""
        data = {
            "id": "test-1",
            "source_type": "unknown_type",
            "source_id": "unknown-123",
            "content": "Content",
        }

        evidence = Evidence.from_dict(data)
        assert evidence.source_type == SourceType.WEB_SEARCH  # Falls back to default


class TestBaseConnector:
    """Tests for BaseConnector abstract class."""

    @pytest.fixture
    def connector(self):
        """Create a mock connector for testing."""
        return MockConnector()

    def test_connector_properties(self, connector):
        """Test connector property methods."""
        assert connector.source_type == SourceType.WEB_SEARCH
        assert connector.name == "MockConnector"

    @pytest.mark.asyncio
    async def test_search(self, connector):
        """Test search functionality."""
        results = await connector.search("test query", limit=3)

        assert len(results) == 3
        assert all(isinstance(r, Evidence) for r in results)
        assert all(r.source_type == SourceType.WEB_SEARCH for r in results)

    @pytest.mark.asyncio
    async def test_fetch(self, connector):
        """Test fetch functionality."""
        result = await connector.fetch("mock:123")

        assert result is not None
        assert result.id == "mock:123"
        assert result.content == "Mock fetched content"

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, connector):
        """Test fetch returns None for non-existent ID."""
        result = await connector.fetch("invalid:123")
        assert result is None

    def test_cache_operations(self, connector):
        """Test cache put/get operations."""
        evidence = Evidence(
            id="cache-test",
            source_type=SourceType.WEB_SEARCH,
            source_id="test-123",
            content="Cached content",
        )

        # Initially not in cache
        assert connector._cache_get("cache-test") is None

        # Add to cache
        connector._cache_put("cache-test", evidence)

        # Should be retrievable
        cached = connector._cache_get("cache-test")
        assert cached is not None
        assert cached.id == "cache-test"
        assert cached.content == "Cached content"

    def test_cache_ttl_expiry(self):
        """Test cache TTL expiry."""
        connector = MockConnector(cache_ttl_seconds=0.1)  # 100ms TTL

        evidence = Evidence(
            id="ttl-test",
            source_type=SourceType.WEB_SEARCH,
            source_id="test-123",
            content="Short-lived content",
        )

        connector._cache_put("ttl-test", evidence)
        assert connector._cache_get("ttl-test") is not None

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        assert connector._cache_get("ttl-test") is None

    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        connector = MockConnector(max_cache_entries=3)

        # Add 4 items, first should be evicted
        for i in range(4):
            evidence = Evidence(
                id=f"lru-{i}",
                source_type=SourceType.WEB_SEARCH,
                source_id=f"test-{i}",
                content=f"Content {i}",
            )
            connector._cache_put(f"lru-{i}", evidence)

        # First item should be evicted
        assert connector._cache_get("lru-0") is None

        # Later items should still exist
        assert connector._cache_get("lru-1") is not None
        assert connector._cache_get("lru-2") is not None
        assert connector._cache_get("lru-3") is not None

    def test_cache_stats(self, connector):
        """Test cache statistics."""
        evidence = Evidence(
            id="stats-test",
            source_type=SourceType.WEB_SEARCH,
            source_id="test-123",
            content="Stats content",
        )
        connector._cache_put("stats-test", evidence)

        stats = connector._cache_stats()

        assert stats["total_entries"] == 1
        assert stats["active_entries"] == 1
        assert stats["expired_entries"] == 0
        assert stats["max_entries"] == 500
        assert stats["ttl_seconds"] == 3600.0

    def test_calculate_freshness_recent(self, connector):
        """Test freshness calculation for recent content."""
        from datetime import datetime, timezone

        # Content from today
        now = datetime.now(timezone.utc).isoformat()
        freshness = connector.calculate_freshness(now)
        assert freshness == 1.0

    def test_calculate_freshness_old(self, connector):
        """Test freshness calculation for old content."""
        # Content from 2 years ago
        freshness = connector.calculate_freshness("2024-01-01T00:00:00Z")
        assert freshness == 0.3  # > 1 year old

    def test_calculate_freshness_invalid(self, connector):
        """Test freshness calculation with invalid date."""
        freshness = connector.calculate_freshness("invalid-date")
        assert freshness == 0.5  # Unknown age default

    def test_repr(self, connector):
        """Test string representation."""
        repr_str = repr(connector)
        assert "MockConnector" in repr_str
        assert "web_search" in repr_str

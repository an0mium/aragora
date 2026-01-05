"""
Tests for Base Connector and Evidence dataclass.

Tests cover:
- Evidence dataclass (creation, properties, serialization)
- BaseConnector LRU cache behavior
- Freshness calculation based on content age
- Provenance recording
"""

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from aragora.connectors.base import Evidence, BaseConnector
from aragora.reasoning.provenance import SourceType, ProvenanceManager


class TestEvidenceDataclass:
    """Tests for Evidence dataclass."""

    def test_evidence_creation_with_defaults(self):
        """Evidence should be created with default values."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com",
            content="Test content",
        )
        assert evidence.id == "test-1"
        assert evidence.source_type == SourceType.WEB_SEARCH
        assert evidence.title == ""
        assert evidence.confidence == 0.5
        assert evidence.freshness == 1.0
        assert evidence.authority == 0.5
        assert evidence.metadata == {}

    def test_evidence_creation_with_all_fields(self):
        """Evidence should store all provided fields."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="github/issues/1",
            content="Issue content",
            title="Bug Report",
            created_at="2026-01-01T00:00:00Z",
            author="user123",
            url="https://github.com/org/repo/issues/1",
            confidence=0.8,
            freshness=0.9,
            authority=0.7,
            metadata={"labels": ["bug"]},
        )
        assert evidence.title == "Bug Report"
        assert evidence.author == "user123"
        assert evidence.confidence == 0.8
        assert evidence.metadata["labels"] == ["bug"]

    def test_content_hash_computation(self):
        """content_hash should be deterministic for same content."""
        evidence1 = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Same content",
        )
        evidence2 = Evidence(
            id="e2",
            source_type=SourceType.DOCUMENT,
            source_id="other",
            content="Same content",
        )
        # Same content = same hash
        assert evidence1.content_hash == evidence2.content_hash

    def test_content_hash_differs_for_different_content(self):
        """content_hash should differ for different content."""
        evidence1 = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Content A",
        )
        evidence2 = Evidence(
            id="e2",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Content B",
        )
        assert evidence1.content_hash != evidence2.content_hash

    def test_reliability_score_calculation(self):
        """reliability_score should be weighted average of factors."""
        evidence = Evidence(
            id="test",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test",
            confidence=1.0,
            freshness=1.0,
            authority=1.0,
        )
        # All 1.0 = weighted average = 1.0
        assert evidence.reliability_score == pytest.approx(1.0, abs=0.01)

    def test_reliability_score_weighted_average(self):
        """reliability_score should weight confidence (0.4), freshness (0.3), authority (0.3)."""
        evidence = Evidence(
            id="test",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test",
            confidence=0.5,
            freshness=0.5,
            authority=0.5,
        )
        # All 0.5 = 0.5
        assert evidence.reliability_score == pytest.approx(0.5, abs=0.01)

    def test_to_dict_serialization(self):
        """to_dict should include all fields and computed properties."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.WEB_SEARCH,
            source_id="https://example.com",
            content="Test content",
            title="Test Title",
            confidence=0.8,
        )
        d = evidence.to_dict()

        assert d["id"] == "test-1"
        assert d["source_type"] == "web_search"
        assert d["source_id"] == "https://example.com"
        assert d["content"] == "Test content"
        assert d["title"] == "Test Title"
        assert d["confidence"] == 0.8
        assert "reliability_score" in d
        assert "content_hash" in d

    def test_from_dict_deserialization(self):
        """from_dict should restore Evidence from dictionary."""
        original = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="github/issues/1",
            content="Issue content",
            title="Bug Report",
            confidence=0.9,
            metadata={"type": "issue"},
        )
        d = original.to_dict()
        restored = Evidence.from_dict(d)

        assert restored.id == original.id
        assert restored.source_type == original.source_type
        assert restored.source_id == original.source_id
        assert restored.content == original.content
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata

    def test_from_dict_handles_unknown_source_type(self):
        """from_dict should handle unknown source_type gracefully."""
        d = {
            "id": "test",
            "source_type": "unknown_type",
            "source_id": "test",
            "content": "Test",
        }
        evidence = Evidence.from_dict(d)
        # Should fallback to WEB_SEARCH
        assert evidence.source_type == SourceType.WEB_SEARCH


class ConcreteConnector(BaseConnector):
    """Concrete implementation for testing abstract BaseConnector."""

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def name(self) -> str:
        return "Test Connector"

    async def search(self, query: str, limit: int = 10, **kwargs):
        return []

    async def fetch(self, evidence_id: str):
        return self._cache.get(evidence_id)


class TestBaseConnectorCache:
    """Tests for BaseConnector LRU cache."""

    @pytest.fixture
    def connector(self):
        """Create connector with small cache for testing."""
        return ConcreteConnector(max_cache_entries=3)

    def test_cache_insertion(self, connector):
        """_cache_put should add evidence to cache."""
        evidence = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test",
        )
        connector._cache_put("e1", evidence)

        assert "e1" in connector._cache
        assert connector._cache["e1"] == evidence

    def test_cache_eviction_at_limit(self, connector):
        """Cache should evict oldest when at max_cache_entries."""
        for i in range(5):  # Add 5 to cache with limit of 3
            evidence = Evidence(
                id=f"e{i}",
                source_type=SourceType.WEB_SEARCH,
                source_id=f"test{i}",
                content=f"Content {i}",
            )
            connector._cache_put(f"e{i}", evidence)

        # Only 3 should remain (most recent)
        assert len(connector._cache) == 3
        assert "e0" not in connector._cache
        assert "e1" not in connector._cache
        assert "e4" in connector._cache

    def test_cache_move_to_end_on_access(self, connector):
        """Accessing cached item should move it to end (LRU)."""
        for i in range(3):
            evidence = Evidence(
                id=f"e{i}",
                source_type=SourceType.WEB_SEARCH,
                source_id=f"test{i}",
                content=f"Content {i}",
            )
            connector._cache_put(f"e{i}", evidence)

        # Access e0 (oldest) - should move to end
        connector._cache_put("e0", connector._cache["e0"])

        # Add new item - e1 should be evicted (now oldest)
        new_evidence = Evidence(
            id="e3",
            source_type=SourceType.WEB_SEARCH,
            source_id="test3",
            content="Content 3",
        )
        connector._cache_put("e3", new_evidence)

        assert "e0" in connector._cache  # Was accessed, moved to end
        assert "e1" not in connector._cache  # Evicted
        assert "e2" in connector._cache
        assert "e3" in connector._cache

    def test_empty_cache_behavior(self, connector):
        """Empty cache should work correctly."""
        assert len(connector._cache) == 0

        # Can still put
        evidence = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test",
        )
        connector._cache_put("e1", evidence)
        assert len(connector._cache) == 1


class TestFreshnessCalculation:
    """Tests for BaseConnector.calculate_freshness."""

    @pytest.fixture
    def connector(self):
        return ConcreteConnector()

    def test_freshness_recent_content(self, connector):
        """Content < 7 days old should have freshness 1.0."""
        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        freshness = connector.calculate_freshness(recent)
        assert freshness == 1.0

    def test_freshness_one_month(self, connector):
        """Content 7-30 days old should have freshness 0.9."""
        one_month = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        freshness = connector.calculate_freshness(one_month)
        assert freshness == 0.9

    def test_freshness_three_months(self, connector):
        """Content 30-90 days old should have freshness 0.7."""
        three_months = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        freshness = connector.calculate_freshness(three_months)
        assert freshness == 0.7

    def test_freshness_one_year(self, connector):
        """Content 90-365 days old should have freshness 0.5."""
        one_year = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        freshness = connector.calculate_freshness(one_year)
        assert freshness == 0.5

    def test_freshness_old_content(self, connector):
        """Content > 1 year old should have freshness 0.3."""
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
        freshness = connector.calculate_freshness(old)
        assert freshness == 0.3

    def test_freshness_invalid_date(self, connector):
        """Invalid date should return 0.5 (unknown)."""
        freshness = connector.calculate_freshness("not-a-date")
        assert freshness == 0.5

    def test_freshness_none_date(self, connector):
        """None date should return 0.5 (unknown)."""
        freshness = connector.calculate_freshness(None)
        assert freshness == 0.5


class TestProvenanceRecording:
    """Tests for evidence recording with provenance."""

    @pytest.fixture
    def mock_provenance(self):
        """Mock ProvenanceManager."""
        manager = Mock(spec=ProvenanceManager)
        mock_record = Mock()
        mock_record.id = "record-1"
        manager.record_evidence.return_value = mock_record
        return manager

    @pytest.fixture
    def connector_with_provenance(self, mock_provenance):
        """Connector with provenance manager."""
        return ConcreteConnector(provenance=mock_provenance)

    @pytest.fixture
    def connector_without_provenance(self):
        """Connector without provenance manager."""
        return ConcreteConnector()

    def test_record_evidence_with_provenance(self, connector_with_provenance, mock_provenance):
        """record_evidence should call provenance manager."""
        evidence = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test content",
            title="Test",
        )

        record = connector_with_provenance.record_evidence(evidence)

        assert record is not None
        mock_provenance.record_evidence.assert_called_once()

    def test_record_evidence_without_provenance(self, connector_without_provenance):
        """record_evidence without provenance manager should return None."""
        evidence = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test",
        )

        record = connector_without_provenance.record_evidence(evidence)

        assert record is None

    def test_record_evidence_with_claim_id(self, connector_with_provenance, mock_provenance):
        """record_evidence with claim_id should create citation."""
        evidence = Evidence(
            id="e1",
            source_type=SourceType.WEB_SEARCH,
            source_id="test",
            content="Test content",
        )

        connector_with_provenance.record_evidence(evidence, claim_id="claim-1")

        mock_provenance.cite_evidence.assert_called_once()


class TestConnectorRepr:
    """Tests for connector string representation."""

    def test_repr(self):
        """__repr__ should show class and source type."""
        connector = ConcreteConnector()
        repr_str = repr(connector)

        assert "ConcreteConnector" in repr_str
        assert "web_search" in repr_str

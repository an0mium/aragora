"""Tests for MemoryAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.memory import MemoryAPI


class TestMemoryAPI:
    """Tests for MemoryAPI resource."""

    def test_memory_api_exists(self):
        """Test that MemoryAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.memory, MemoryAPI)

    def test_memory_api_has_analytics_methods(self):
        """Test that MemoryAPI has analytics methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "analytics")
        assert hasattr(client.memory, "analytics_async")
        assert callable(client.memory.analytics)

    def test_memory_api_has_tier_stats_methods(self):
        """Test that MemoryAPI has tier_stats methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "tier_stats")
        assert hasattr(client.memory, "tier_stats_async")

    def test_memory_api_has_snapshot_methods(self):
        """Test that MemoryAPI has snapshot methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "snapshot")
        assert hasattr(client.memory, "snapshot_async")

    def test_memory_api_has_stats_methods(self):
        """Test that MemoryAPI has stats methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "stats")
        assert hasattr(client.memory, "stats_async")

    def test_memory_api_has_search_methods(self):
        """Test that MemoryAPI has search methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "search")
        assert hasattr(client.memory, "search_async")

    def test_memory_api_has_get_tiers_methods(self):
        """Test that MemoryAPI has get_tiers methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "get_tiers")
        assert hasattr(client.memory, "get_tiers_async")

    def test_memory_api_has_get_critiques_methods(self):
        """Test that MemoryAPI has get_critiques methods."""
        client = AragoraClient()
        assert hasattr(client.memory, "get_critiques")
        assert hasattr(client.memory, "get_critiques_async")


class TestMemoryModels:
    """Tests for Memory model classes."""

    def test_memory_entry_import(self):
        """Test MemoryEntry model can be imported."""
        from aragora.client.models import MemoryEntry

        entry = MemoryEntry(
            id="mem_001",
            content="Test memory content",
            tier="fast",
        )
        assert entry.id == "mem_001"
        assert entry.content == "Test memory content"

    def test_memory_tier_stats_import(self):
        """Test MemoryTierStats model can be imported."""
        from aragora.client.models import MemoryTierStats

        tier_stats = MemoryTierStats(
            tier_name="fast",
            count=100,
            hit_rate=0.95,
        )
        assert tier_stats.tier_name == "fast"
        assert tier_stats.hit_rate == 0.95

    def test_memory_analytics_response_import(self):
        """Test MemoryAnalyticsResponse model can be imported."""
        from aragora.client.models import MemoryAnalyticsResponse

        # Check that the model can be imported
        assert MemoryAnalyticsResponse is not None

    def test_memory_snapshot_response_import(self):
        """Test MemorySnapshotResponse model can be imported."""
        from aragora.client.models import MemorySnapshotResponse

        # Check that the model can be imported
        assert MemorySnapshotResponse is not None

    def test_critique_entry_import(self):
        """Test CritiqueEntry model can be imported."""
        from aragora.client.models import CritiqueEntry

        # Check that the model can be imported
        assert CritiqueEntry is not None


class TestMemoryTierValues:
    """Tests for memory tier constants and validation."""

    def test_memory_tier_type_enum(self):
        """Test MemoryTierType enum exists."""
        from aragora.client.models import MemoryTierType

        assert MemoryTierType.FAST == "fast"
        assert MemoryTierType.MEDIUM == "medium"
        assert MemoryTierType.SLOW == "slow"
        assert MemoryTierType.GLACIAL == "glacial"

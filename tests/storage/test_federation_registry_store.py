"""
Tests for FederationRegistryStore backends.

Tests all three backends: InMemory, SQLite, and Redis (with fallback).
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aragora.storage.federation_registry_store import (
    FederatedRegionConfig,
    InMemoryFederationRegistryStore,
    SQLiteFederationRegistryStore,
    RedisFederationRegistryStore,
    get_federation_registry_store,
    reset_federation_registry_store,
    set_federation_registry_store,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_federation_registry.db"


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryFederationRegistryStore()


@pytest.fixture
def sqlite_store(temp_db_path):
    """Create a SQLite store for testing."""
    return SQLiteFederationRegistryStore(temp_db_path)


@pytest.fixture
def sample_region():
    """Create a sample region config."""
    return FederatedRegionConfig(
        region_id="us-west-2",
        endpoint_url="https://west.example.com/api",
        api_key="test-api-key-123",
        mode="bidirectional",
        sync_scope="summary",
        enabled=True,
        workspace_id="workspace-1",
    )


@pytest.fixture
def sample_region_disabled():
    """Create a disabled region config."""
    return FederatedRegionConfig(
        region_id="eu-central-1",
        endpoint_url="https://eu.example.com/api",
        api_key="test-api-key-456",
        mode="push",
        sync_scope="metadata",
        enabled=False,
        workspace_id="workspace-1",
    )


@pytest.fixture
def sample_region_another_workspace():
    """Create a region in another workspace."""
    return FederatedRegionConfig(
        region_id="ap-southeast-1",
        endpoint_url="https://asia.example.com/api",
        api_key="test-api-key-789",
        mode="pull",
        sync_scope="full",
        enabled=True,
        workspace_id="workspace-2",
    )


class TestFederatedRegionConfig:
    """Tests for FederatedRegionConfig dataclass."""

    def test_to_dict(self, sample_region):
        """Should serialize to dict."""
        result = sample_region.to_dict()
        assert result["region_id"] == "us-west-2"
        assert result["endpoint_url"] == "https://west.example.com/api"
        assert result["mode"] == "bidirectional"
        assert result["sync_scope"] == "summary"
        assert result["enabled"] is True

    def test_from_dict(self):
        """Should create from dict."""
        data = {
            "region_id": "test-region",
            "endpoint_url": "https://test.example.com/api",
            "api_key": "key-123",
            "mode": "push",
            "sync_scope": "full",
            "enabled": False,
        }
        config = FederatedRegionConfig.from_dict(data)
        assert config.region_id == "test-region"
        assert config.mode == "push"
        assert config.enabled is False

    def test_to_json_roundtrip(self, sample_region):
        """JSON serialization should preserve data."""
        json_str = sample_region.to_json()
        restored = FederatedRegionConfig.from_json(json_str)
        assert restored.region_id == sample_region.region_id
        assert restored.endpoint_url == sample_region.endpoint_url
        assert restored.api_key == sample_region.api_key
        assert restored.mode == sample_region.mode

    def test_default_timestamps(self):
        """Should set default timestamps if not provided."""
        config = FederatedRegionConfig(
            region_id="test",
            endpoint_url="https://test.com",
            api_key="key",
        )
        assert config.created_at != ""
        assert config.updated_at != ""

    def test_default_values(self):
        """Should use sensible defaults."""
        config = FederatedRegionConfig(
            region_id="test",
            endpoint_url="https://test.com",
            api_key="key",
        )
        assert config.mode == "bidirectional"
        assert config.sync_scope == "summary"
        assert config.enabled is True
        assert config.total_pushes == 0
        assert config.total_pulls == 0


class TestInMemoryFederationRegistryStore:
    """Tests for InMemoryFederationRegistryStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, memory_store, sample_region):
        """Should save and retrieve region."""
        await memory_store.save(sample_region)
        retrieved = await memory_store.get("us-west-2", "workspace-1")
        assert retrieved is not None
        assert retrieved.region_id == "us-west-2"
        assert retrieved.endpoint_url == "https://west.example.com/api"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_store):
        """Should return None for nonexistent region."""
        result = await memory_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, memory_store, sample_region):
        """Should delete region."""
        await memory_store.save(sample_region)
        deleted = await memory_store.delete("us-west-2", "workspace-1")
        assert deleted is True
        result = await memory_store.get("us-west-2", "workspace-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_store):
        """Should return False when deleting nonexistent."""
        deleted = await memory_store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_all(
        self, memory_store, sample_region, sample_region_disabled
    ):
        """Should list all regions."""
        await memory_store.save(sample_region)
        await memory_store.save(sample_region_disabled)

        all_regions = await memory_store.list_all()
        assert len(all_regions) == 2

    @pytest.mark.asyncio
    async def test_list_all_by_workspace(
        self, memory_store, sample_region, sample_region_another_workspace
    ):
        """Should list regions filtered by workspace."""
        await memory_store.save(sample_region)
        await memory_store.save(sample_region_another_workspace)

        ws1_regions = await memory_store.list_all("workspace-1")
        assert len(ws1_regions) == 1
        assert ws1_regions[0].region_id == "us-west-2"

        ws2_regions = await memory_store.list_all("workspace-2")
        assert len(ws2_regions) == 1
        assert ws2_regions[0].region_id == "ap-southeast-1"

    @pytest.mark.asyncio
    async def test_list_enabled(
        self, memory_store, sample_region, sample_region_disabled
    ):
        """Should list only enabled regions."""
        await memory_store.save(sample_region)
        await memory_store.save(sample_region_disabled)

        enabled = await memory_store.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].region_id == "us-west-2"

    @pytest.mark.asyncio
    async def test_update_sync_status_push(self, memory_store, sample_region):
        """Should update sync status for push."""
        await memory_store.save(sample_region)

        await memory_store.update_sync_status(
            "us-west-2", "push", nodes_synced=10, workspace_id="workspace-1"
        )

        updated = await memory_store.get("us-west-2", "workspace-1")
        assert updated is not None
        assert updated.total_pushes == 1
        assert updated.total_nodes_synced == 10
        assert updated.last_push_at is not None
        assert updated.last_sync_at is not None

    @pytest.mark.asyncio
    async def test_update_sync_status_pull(self, memory_store, sample_region):
        """Should update sync status for pull."""
        await memory_store.save(sample_region)

        await memory_store.update_sync_status(
            "us-west-2", "pull", nodes_synced=5, workspace_id="workspace-1"
        )

        updated = await memory_store.get("us-west-2", "workspace-1")
        assert updated is not None
        assert updated.total_pulls == 1
        assert updated.total_nodes_synced == 5
        assert updated.last_pull_at is not None

    @pytest.mark.asyncio
    async def test_update_sync_status_with_error(self, memory_store, sample_region):
        """Should track sync errors."""
        await memory_store.save(sample_region)

        await memory_store.update_sync_status(
            "us-west-2", "push", error="Connection timeout", workspace_id="workspace-1"
        )

        updated = await memory_store.get("us-west-2", "workspace-1")
        assert updated is not None
        assert updated.total_sync_errors == 1
        assert updated.last_sync_error == "Connection timeout"


class TestSQLiteFederationRegistryStore:
    """Tests for SQLiteFederationRegistryStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, sqlite_store, sample_region):
        """Should save and retrieve region."""
        await sqlite_store.save(sample_region)
        retrieved = await sqlite_store.get("us-west-2", "workspace-1")
        assert retrieved is not None
        assert retrieved.region_id == "us-west-2"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_store):
        """Should return None for nonexistent region."""
        result = await sqlite_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_region):
        """Should delete region."""
        await sqlite_store.save(sample_region)
        deleted = await sqlite_store.delete("us-west-2", "workspace-1")
        assert deleted is True
        result = await sqlite_store.get("us-west-2", "workspace-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db_path, sample_region):
        """Data should persist across store instances."""
        store1 = SQLiteFederationRegistryStore(temp_db_path)
        await store1.save(sample_region)
        await store1.close()

        store2 = SQLiteFederationRegistryStore(temp_db_path)
        retrieved = await store2.get("us-west-2", "workspace-1")
        assert retrieved is not None
        assert retrieved.region_id == "us-west-2"
        await store2.close()

    @pytest.mark.asyncio
    async def test_update_existing(self, sqlite_store, sample_region):
        """Should update existing region."""
        await sqlite_store.save(sample_region)

        sample_region.mode = "push"
        sample_region.enabled = False
        await sqlite_store.save(sample_region)

        retrieved = await sqlite_store.get("us-west-2", "workspace-1")
        assert retrieved is not None
        assert retrieved.mode == "push"
        assert retrieved.enabled is False

    @pytest.mark.asyncio
    async def test_list_enabled(
        self, sqlite_store, sample_region, sample_region_disabled
    ):
        """Should list only enabled regions."""
        await sqlite_store.save(sample_region)
        await sqlite_store.save(sample_region_disabled)

        enabled = await sqlite_store.list_enabled("workspace-1")
        assert len(enabled) == 1
        assert enabled[0].region_id == "us-west-2"

    @pytest.mark.asyncio
    async def test_update_sync_status(self, sqlite_store, sample_region):
        """Should update sync status."""
        await sqlite_store.save(sample_region)

        await sqlite_store.update_sync_status(
            "us-west-2", "push", nodes_synced=15, workspace_id="workspace-1"
        )
        await sqlite_store.update_sync_status(
            "us-west-2", "pull", nodes_synced=8, workspace_id="workspace-1"
        )

        updated = await sqlite_store.get("us-west-2", "workspace-1")
        assert updated is not None
        assert updated.total_pushes == 1
        assert updated.total_pulls == 1
        assert updated.total_nodes_synced == 23  # 15 + 8


class TestRedisFederationRegistryStore:
    """Tests for RedisFederationRegistryStore (with SQLite fallback)."""

    @pytest.fixture
    def redis_store(self, temp_db_path):
        """Create a Redis store (will use SQLite fallback if Redis unavailable)."""
        return RedisFederationRegistryStore(temp_db_path, redis_url="redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_save_and_get(self, redis_store, sample_region):
        """Should save and retrieve region."""
        await redis_store.save(sample_region)
        retrieved = await redis_store.get("us-west-2", "workspace-1")
        assert retrieved is not None
        assert retrieved.region_id == "us-west-2"

    @pytest.mark.asyncio
    async def test_delete(self, redis_store, sample_region):
        """Should delete region."""
        await redis_store.save(sample_region)
        deleted = await redis_store.delete("us-west-2", "workspace-1")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_fallback_persistence(self, temp_db_path, sample_region):
        """SQLite fallback should persist data."""
        store1 = RedisFederationRegistryStore(temp_db_path)
        await store1.save(sample_region)
        await store1.close()

        store2 = RedisFederationRegistryStore(temp_db_path)
        retrieved = await store2.get("us-west-2", "workspace-1")
        assert retrieved is not None
        await store2.close()

    @pytest.mark.asyncio
    async def test_list_enabled(
        self, redis_store, sample_region, sample_region_disabled
    ):
        """Should list only enabled regions."""
        await redis_store.save(sample_region)
        await redis_store.save(sample_region_disabled)

        enabled = await redis_store.list_enabled("workspace-1")
        assert len(enabled) == 1
        assert enabled[0].region_id == "us-west-2"


class TestGlobalStore:
    """Tests for global store factory functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_federation_registry_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_federation_registry_store()

    def test_get_default_store(self, monkeypatch, temp_db_path):
        """Should create default SQLite store."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(temp_db_path.parent))
        store = get_federation_registry_store()
        assert isinstance(store, SQLiteFederationRegistryStore)

    def test_get_memory_store(self, monkeypatch):
        """Should create in-memory store when configured."""
        monkeypatch.setenv("ARAGORA_FEDERATION_STORE_BACKEND", "memory")
        store = get_federation_registry_store()
        assert isinstance(store, InMemoryFederationRegistryStore)

    def test_set_custom_store(self):
        """Should allow setting custom store."""
        custom_store = InMemoryFederationRegistryStore()
        set_federation_registry_store(custom_store)
        store = get_federation_registry_store()
        assert store is custom_store

    def test_singleton_pattern(self, monkeypatch):
        """Should return same instance on multiple calls."""
        monkeypatch.setenv("ARAGORA_FEDERATION_STORE_BACKEND", "memory")
        store1 = get_federation_registry_store()
        store2 = get_federation_registry_store()
        assert store1 is store2


class TestWorkspaceIsolation:
    """Tests for workspace isolation in region storage."""

    @pytest.mark.asyncio
    async def test_regions_isolated_by_workspace(self, memory_store):
        """Regions should be isolated by workspace."""
        region1 = FederatedRegionConfig(
            region_id="shared-region",
            endpoint_url="https://ws1.example.com",
            api_key="key1",
            workspace_id="workspace-1",
        )
        region2 = FederatedRegionConfig(
            region_id="shared-region",
            endpoint_url="https://ws2.example.com",
            api_key="key2",
            workspace_id="workspace-2",
        )

        await memory_store.save(region1)
        await memory_store.save(region2)

        retrieved1 = await memory_store.get("shared-region", "workspace-1")
        retrieved2 = await memory_store.get("shared-region", "workspace-2")

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.endpoint_url == "https://ws1.example.com"
        assert retrieved2.endpoint_url == "https://ws2.example.com"

    @pytest.mark.asyncio
    async def test_delete_only_affects_workspace(self, memory_store):
        """Delete should only affect the specified workspace."""
        region1 = FederatedRegionConfig(
            region_id="shared-region",
            endpoint_url="https://ws1.example.com",
            api_key="key1",
            workspace_id="workspace-1",
        )
        region2 = FederatedRegionConfig(
            region_id="shared-region",
            endpoint_url="https://ws2.example.com",
            api_key="key2",
            workspace_id="workspace-2",
        )

        await memory_store.save(region1)
        await memory_store.save(region2)

        await memory_store.delete("shared-region", "workspace-1")

        retrieved1 = await memory_store.get("shared-region", "workspace-1")
        retrieved2 = await memory_store.get("shared-region", "workspace-2")

        assert retrieved1 is None
        assert retrieved2 is not None


class TestMetricAccumulation:
    """Tests for sync metrics accumulation."""

    @pytest.mark.asyncio
    async def test_metrics_accumulate(self, memory_store, sample_region):
        """Metrics should accumulate over multiple sync operations."""
        await memory_store.save(sample_region)

        # Multiple push operations
        await memory_store.update_sync_status(
            "us-west-2", "push", nodes_synced=10, workspace_id="workspace-1"
        )
        await memory_store.update_sync_status(
            "us-west-2", "push", nodes_synced=5, workspace_id="workspace-1"
        )
        await memory_store.update_sync_status(
            "us-west-2", "push", error="Timeout", workspace_id="workspace-1"
        )

        # Pull operations
        await memory_store.update_sync_status(
            "us-west-2", "pull", nodes_synced=8, workspace_id="workspace-1"
        )

        updated = await memory_store.get("us-west-2", "workspace-1")
        assert updated is not None
        assert updated.total_pushes == 3
        assert updated.total_pulls == 1
        assert updated.total_nodes_synced == 23  # 10 + 5 + 8
        assert updated.total_sync_errors == 1

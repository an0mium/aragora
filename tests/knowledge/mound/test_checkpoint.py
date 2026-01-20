"""
Tests for Knowledge Mound Checkpoint Store.

Tests checkpoint creation, restoration, and management for KM state persistence.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.checkpoint import (
    KMCheckpointStore,
    KMCheckpointMetadata,
    KMCheckpointContent,
    RestoreResult,
    get_km_checkpoint_store,
    reset_km_checkpoint_store,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound instance."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"
    mound.config = MagicMock()
    mound.config.backend = MagicMock(value="sqlite")
    mound.config.enable_staleness_detection = True
    mound.config.enable_culture_accumulator = True

    # Mock meta store with nodes
    mock_node = MagicMock()
    mock_node.id = "node_001"
    mock_node.node_type = "consensus"
    mock_node.content = "Test content"
    mock_node.confidence = 0.9
    mock_node.workspace_id = "test_workspace"
    mock_node.metadata = {"key": "value"}
    mock_node.topics = ["topic1", "topic2"]
    mock_node.created_at = datetime.now()
    mock_node.provenance = None

    mound._meta_store = MagicMock()
    mound._meta_store.query_nodes = MagicMock(return_value=[mock_node])
    mound._meta_store.get_all_relationships = MagicMock(return_value=[])
    mound._meta_store.clear_workspace = MagicMock()

    # Mock save methods
    mound._save_node = AsyncMock()
    mound._save_relationship = AsyncMock()

    # Mock culture accumulator
    mound._culture_accumulator = MagicMock()
    mound._culture_accumulator.export_state = MagicMock(return_value={"patterns": []})
    mound._culture_accumulator.import_state = MagicMock()

    # Mock staleness detector
    mound._staleness_detector = MagicMock()
    mound._staleness_detector.export_state = MagicMock(return_value={"stale_nodes": []})
    mound._staleness_detector.import_state = MagicMock()

    # Mock semantic store
    mound._semantic_store = MagicMock()
    mound._semantic_store.export_embeddings = MagicMock(return_value={})
    mound._semantic_store.import_embeddings = MagicMock()

    return mound


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_store(mock_mound, temp_checkpoint_dir):
    """Create a checkpoint store with mock mound."""
    return KMCheckpointStore(
        mound=mock_mound,
        checkpoint_dir=temp_checkpoint_dir,
        compress=True,
        max_checkpoints=5,
    )


# ============================================================================
# KMCheckpointMetadata Tests
# ============================================================================


class TestKMCheckpointMetadata:
    """Tests for KMCheckpointMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        metadata = KMCheckpointMetadata(
            id="ckpt_001",
            created_at="2024-01-15T10:30:00",
            description="Test checkpoint",
            workspace_id="test_ws",
        )

        assert metadata.id == "ckpt_001"
        assert metadata.description == "Test checkpoint"
        assert metadata.workspace_id == "test_ws"
        assert metadata.mound_version == "1.0"
        assert metadata.compressed is True
        assert metadata.incremental is False

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = KMCheckpointMetadata(
            id="ckpt_001",
            created_at="2024-01-15T10:30:00",
            description="Test checkpoint",
            workspace_id="test_ws",
            node_count=100,
            relationship_count=50,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "ckpt_001"
        assert result["node_count"] == 100
        assert result["relationship_count"] == 50

    def test_metadata_with_incremental(self):
        """Test metadata for incremental checkpoint."""
        metadata = KMCheckpointMetadata(
            id="ckpt_002",
            created_at="2024-01-15T11:00:00",
            description="Incremental",
            workspace_id="test_ws",
            incremental=True,
            parent_checkpoint_id="ckpt_001",
        )

        assert metadata.incremental is True
        assert metadata.parent_checkpoint_id == "ckpt_001"


# ============================================================================
# KMCheckpointContent Tests
# ============================================================================


class TestKMCheckpointContent:
    """Tests for KMCheckpointContent dataclass."""

    def test_content_creation(self):
        """Test creating checkpoint content."""
        content = KMCheckpointContent()

        assert content.nodes == []
        assert content.relationships == []
        assert content.culture_patterns == {}
        assert content.staleness_state == {}
        assert content.vector_embeddings == {}

    def test_content_with_data(self):
        """Test content with actual data."""
        content = KMCheckpointContent(
            nodes=[{"id": "n1", "content": "test"}],
            relationships=[{"from": "n1", "to": "n2"}],
            culture_patterns={"pattern1": {"count": 5}},
        )

        assert len(content.nodes) == 1
        assert len(content.relationships) == 1
        assert "pattern1" in content.culture_patterns

    def test_content_to_dict(self):
        """Test converting content to dictionary."""
        content = KMCheckpointContent(
            nodes=[{"id": "n1"}],
            workspace_metadata={"workspace_id": "test"},
        )

        result = content.to_dict()

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "workspace_metadata" in result


# ============================================================================
# KMCheckpointStore Tests
# ============================================================================


class TestKMCheckpointStore:
    """Tests for KMCheckpointStore."""

    def test_store_initialization(self, mock_mound, temp_checkpoint_dir):
        """Test checkpoint store initialization."""
        store = KMCheckpointStore(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
            compress=True,
            max_checkpoints=10,
        )

        assert store.mound == mock_mound
        assert store.compress is True
        assert store.max_checkpoints == 10
        assert store.checkpoint_dir.exists()

    def test_store_default_directory(self, mock_mound):
        """Test default checkpoint directory."""
        store = KMCheckpointStore(mound=mock_mound)

        assert "km_checkpoints" in str(store.checkpoint_dir)

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_store):
        """Test creating a checkpoint."""
        checkpoint_id = await checkpoint_store.create_checkpoint(
            description="Test checkpoint",
            include_vectors=False,
            include_culture=True,
            include_staleness=True,
        )

        assert checkpoint_id.startswith("km_ckpt_")

        # Verify checkpoint files exist
        checkpoint_path = checkpoint_store.checkpoint_dir / checkpoint_id
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()
        assert (checkpoint_path / "content.json.gz").exists()

    @pytest.mark.asyncio
    async def test_create_checkpoint_uncompressed(self, mock_mound, temp_checkpoint_dir):
        """Test creating an uncompressed checkpoint."""
        store = KMCheckpointStore(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
            compress=False,
        )

        checkpoint_id = await store.create_checkpoint(description="Uncompressed")

        checkpoint_path = store.checkpoint_dir / checkpoint_id
        assert (checkpoint_path / "content.json").exists()

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, checkpoint_store):
        """Test listing checkpoints."""
        # Create multiple checkpoints
        await checkpoint_store.create_checkpoint(description="First")
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await checkpoint_store.create_checkpoint(description="Second")

        checkpoints = await checkpoint_store.list_checkpoints()

        assert len(checkpoints) == 2
        # Should be sorted by creation time (newest first)
        assert checkpoints[0].description == "Second"
        assert checkpoints[1].description == "First"

    @pytest.mark.asyncio
    async def test_get_checkpoint_metadata(self, checkpoint_store):
        """Test getting checkpoint metadata."""
        checkpoint_id = await checkpoint_store.create_checkpoint(
            description="Test metadata"
        )

        metadata = await checkpoint_store.get_checkpoint_metadata(checkpoint_id)

        assert metadata is not None
        assert metadata.id == checkpoint_id
        assert metadata.description == "Test metadata"
        assert metadata.workspace_id == "test_workspace"

    @pytest.mark.asyncio
    async def test_get_nonexistent_metadata(self, checkpoint_store):
        """Test getting metadata for nonexistent checkpoint."""
        metadata = await checkpoint_store.get_checkpoint_metadata("nonexistent")

        assert metadata is None

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpoint_store):
        """Test deleting a checkpoint."""
        checkpoint_id = await checkpoint_store.create_checkpoint(description="To delete")

        # Verify it exists
        assert (checkpoint_store.checkpoint_dir / checkpoint_id).exists()

        # Delete it
        result = await checkpoint_store.delete_checkpoint(checkpoint_id)

        assert result is True
        assert not (checkpoint_store.checkpoint_dir / checkpoint_id).exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, checkpoint_store):
        """Test deleting a nonexistent checkpoint."""
        result = await checkpoint_store.delete_checkpoint("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpoint_store):
        """Test restoring from a checkpoint."""
        # Create checkpoint
        checkpoint_id = await checkpoint_store.create_checkpoint(description="To restore")

        # Restore it
        result = await checkpoint_store.restore_checkpoint(
            checkpoint_id,
            clear_existing=True,
            restore_vectors=False,
        )

        assert result.success is True
        assert result.checkpoint_id == checkpoint_id
        assert result.nodes_restored >= 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_restore_nonexistent_checkpoint(self, checkpoint_store):
        """Test restoring from nonexistent checkpoint."""
        result = await checkpoint_store.restore_checkpoint("nonexistent")

        assert result.success is False
        assert "not found" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_compare_checkpoints(self, checkpoint_store):
        """Test comparing two checkpoints."""
        ckpt1 = await checkpoint_store.create_checkpoint(description="First")
        await asyncio.sleep(0.01)
        ckpt2 = await checkpoint_store.create_checkpoint(description="Second")

        comparison = await checkpoint_store.compare_checkpoints(ckpt1, ckpt2)

        assert "checkpoint_1" in comparison
        assert "checkpoint_2" in comparison
        assert "node_count_diff" in comparison
        assert "time_diff_seconds" in comparison

    @pytest.mark.asyncio
    async def test_compare_with_missing_checkpoint(self, checkpoint_store):
        """Test comparing when one checkpoint is missing."""
        ckpt1 = await checkpoint_store.create_checkpoint(description="Exists")

        comparison = await checkpoint_store.compare_checkpoints(ckpt1, "nonexistent")

        assert "error" in comparison

    @pytest.mark.asyncio
    async def test_checkpoint_pruning(self, mock_mound, temp_checkpoint_dir):
        """Test automatic pruning of old checkpoints."""
        store = KMCheckpointStore(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=3,
        )

        # Create more checkpoints than the limit
        for i in range(5):
            await store.create_checkpoint(description=f"Checkpoint {i}")
            await asyncio.sleep(0.01)

        checkpoints = await store.list_checkpoints()

        # Should only have max_checkpoints remaining
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_checkpoint_with_vectors(self, checkpoint_store):
        """Test checkpoint with vector embeddings included."""
        checkpoint_id = await checkpoint_store.create_checkpoint(
            description="With vectors",
            include_vectors=True,
        )

        metadata = await checkpoint_store.get_checkpoint_metadata(checkpoint_id)

        assert metadata.includes_vectors is True

    @pytest.mark.asyncio
    async def test_incremental_checkpoint(self, checkpoint_store):
        """Test creating an incremental checkpoint."""
        parent_id = await checkpoint_store.create_checkpoint(description="Parent")

        child_id = await checkpoint_store.create_checkpoint(
            description="Incremental child",
            incremental=True,
            parent_checkpoint_id=parent_id,
        )

        metadata = await checkpoint_store.get_checkpoint_metadata(child_id)

        assert metadata.incremental is True
        assert metadata.parent_checkpoint_id == parent_id


# ============================================================================
# RestoreResult Tests
# ============================================================================


class TestRestoreResult:
    """Tests for RestoreResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful restore result."""
        result = RestoreResult(
            success=True,
            checkpoint_id="ckpt_001",
            nodes_restored=100,
            relationships_restored=50,
            culture_restored=True,
            duration_ms=500,
        )

        assert result.success is True
        assert result.nodes_restored == 100
        assert result.errors == []

    def test_failed_result(self):
        """Test creating a failed restore result."""
        result = RestoreResult(
            success=False,
            checkpoint_id="ckpt_001",
            errors=["Checksum mismatch", "Node restore failed"],
        )

        assert result.success is False
        assert len(result.errors) == 2


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_checkpoint_store_without_mound(self):
        """Test getting store without mound returns None."""
        reset_km_checkpoint_store()

        store = get_km_checkpoint_store()

        assert store is None

    def test_get_checkpoint_store_with_mound(self, mock_mound, temp_checkpoint_dir):
        """Test getting store with mound creates instance."""
        reset_km_checkpoint_store()

        store = get_km_checkpoint_store(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
        )

        assert store is not None
        assert isinstance(store, KMCheckpointStore)

    def test_singleton_returns_same_instance(self, mock_mound, temp_checkpoint_dir):
        """Test singleton returns the same instance."""
        reset_km_checkpoint_store()

        store1 = get_km_checkpoint_store(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
        )
        store2 = get_km_checkpoint_store()

        assert store1 is store2

    def test_reset_singleton(self, mock_mound, temp_checkpoint_dir):
        """Test resetting the singleton."""
        reset_km_checkpoint_store()

        store1 = get_km_checkpoint_store(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
        )

        reset_km_checkpoint_store()

        store2 = get_km_checkpoint_store()

        assert store2 is None


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_checkpoint_with_empty_mound(self, temp_checkpoint_dir):
        """Test checkpoint when mound has no data."""
        empty_mound = MagicMock()
        empty_mound.workspace_id = "empty"
        empty_mound.config = MagicMock()
        empty_mound.config.backend = MagicMock(value="sqlite")
        empty_mound.config.enable_staleness_detection = False
        empty_mound.config.enable_culture_accumulator = False
        empty_mound._meta_store = MagicMock()
        empty_mound._meta_store.query_nodes = MagicMock(return_value=[])
        empty_mound._meta_store.get_all_relationships = MagicMock(return_value=[])
        empty_mound._culture_accumulator = None
        empty_mound._staleness_detector = None
        empty_mound._semantic_store = None

        store = KMCheckpointStore(
            mound=empty_mound,
            checkpoint_dir=temp_checkpoint_dir,
        )

        checkpoint_id = await store.create_checkpoint(description="Empty mound")

        metadata = await store.get_checkpoint_metadata(checkpoint_id)
        assert metadata.node_count == 0
        assert metadata.relationship_count == 0

    @pytest.mark.asyncio
    async def test_restore_with_checksum_mismatch(self, checkpoint_store):
        """Test restore detects checksum mismatch."""
        checkpoint_id = await checkpoint_store.create_checkpoint(description="Test")

        # Corrupt the content file
        checkpoint_path = checkpoint_store.checkpoint_dir / checkpoint_id
        content_file = checkpoint_path / "content.json.gz"
        content_file.write_bytes(b"corrupted data")

        result = await checkpoint_store.restore_checkpoint(checkpoint_id)

        # Should have checksum error but may still partially succeed
        assert any("checksum" in e.lower() for e in result.errors) or not result.success

    @pytest.mark.asyncio
    async def test_checkpoint_directory_creation(self, mock_mound):
        """Test checkpoint directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nested" / "checkpoint" / "dir"

            store = KMCheckpointStore(
                mound=mock_mound,
                checkpoint_dir=str(new_dir),
            )

            assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_large_checkpoint(self, mock_mound, temp_checkpoint_dir):
        """Test checkpoint with many nodes triggers compression benefit."""
        # Create mock with many nodes
        nodes = [
            MagicMock(
                id=f"node_{i}",
                node_type="test",
                content=f"Content {i}" * 100,  # Larger content
                confidence=0.9,
                workspace_id="test",
                metadata={},
                topics=[],
                created_at=datetime.now(),
                provenance=None,
            )
            for i in range(100)
        ]

        mock_mound._meta_store.query_nodes = MagicMock(return_value=nodes)

        store = KMCheckpointStore(
            mound=mock_mound,
            checkpoint_dir=temp_checkpoint_dir,
            compress=True,
        )

        checkpoint_id = await store.create_checkpoint(description="Large")

        metadata = await store.get_checkpoint_metadata(checkpoint_id)
        assert metadata.node_count == 100
        assert metadata.compressed is True
